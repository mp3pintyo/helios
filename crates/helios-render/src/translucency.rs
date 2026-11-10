//! Forward translucency: glass, water and alpha-blended (glTF `BLEND`) surfaces.
//!
//! As in UE5, translucent surfaces are rendered in a separate forward pass after the deferred
//! lighting and before TSR, over the lit HDR image (`shaders/translucency.wgsl`).
//!
//! * **Which surfaces.** Instances whose material is translucent ([`MaterialKind::is_translucent`])
//!   carry the `TRANSLUCENT` instance flag; the main view's culling skips them, so they never
//!   enter the visibility buffer or the HZB and never occlude the opaque scene. Shadow casting
//!   (the instance's `cast_shadow`) and Lumen (whether the mesh has a distance field) remain
//!   per-instance / per-mesh choices: stained glass keeps its distance field so the surface
//!   cache still lights the interior with coloured daylight; water has none.
//! * **Geometry path.** Each translucent instance is one conventional indexed draw of its mesh's
//!   finest level held in memory, decoded from the virtual mesh once. Translucent surfaces are
//!   few and simple (panes, water planes, blended glTF meshes) and need back-to-front order per
//!   instance, which the GPU-driven cluster lists (unordered, one indirect draw, no
//!   `first_instance` for indirect draws in WebGPU) cannot provide; LOD and streaming buy
//!   nothing for them. Instances are frustum culled and sorted by distance on the CPU, and the
//!   draw's `first_instance` is the instance index (allowed for direct draws).
//! * **Opaque inputs.** The lit HDR image is copied into a scene-colour texture with a few mips
//!   (refraction, blurred transmission and screen-space reflections read it); the merged depth
//!   is tested in the fragment shader (it is not a depth attachment) and gives the thickness of
//!   the medium behind a surface.
//! * **Layer order (depth peeling).** Sorting instances orders neither intersecting surfaces nor
//!   a concave mesh's own triangles, and a refracting surface (water, glass) replaces the pixel
//!   with the scene-colour copy, losing any translucent layer drawn behind it. Where translucent
//!   surfaces may overlap on screen, the nearest `P` layers per pixel are peeled first: depth-only
//!   draws of all surfaces, each keeping the nearest fragment behind the previous layer, into one
//!   layer of a depth texture array. Shading then composites back to front: everything behind
//!   the peeled layers in per-instance order (as without peeling, so pixels with more layers
//!   degrade gracefully), then each peeled layer, after a fresh scene-colour copy when a surface
//!   may refract it — water refracts the glass standing in it, glass the glass behind it. Every
//!   shading pass keeps the fragments of its [`LayerRange`]. Pixels with at most `P + 1` layers
//!   are composited in exact depth order, each surface with its own shading model. `P + 1` ≤
//!   [`MAX_LAYERS`] is an upper bound from the CPU: the most screen rectangles of visible
//!   instances overlapping at one point, a mesh that may cover a pixel twice (neither flat nor
//!   convex) counting as all. Views without overlapping translucent surfaces (the common case)
//!   peel nothing and cost what one pass costs.
//! * **Motion vectors.** TSR would otherwise reproject translucent pixels with the motion of the
//!   opaque surface behind them; the pass writes the surface's own motion where it is mostly
//!   opaque (refracting surfaces, or coverage ≥ 0.5).
//! * **Underwater.** When the camera is below a water body's surface ([`crate::water`]), a
//!   full-screen draw first fogs the opaque scene with the water's absorption and scattering and
//!   adds caustics to the bed; water surfaces are drawn two-sided, and seen from below they show
//!   the world above through Snell's window and total internal reflection around it.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use helios_geometry::{lod_cut_resident, Aabb, LodView, VirtualMesh};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{BindingType, ShaderStages};

use crate::camera::frustum_planes;
use crate::gpu_scene::GpuScene;
use crate::motion::SceneChanges;
use crate::passes::shading::LightingInputs;
use crate::profiler::Recorder;
use crate::scene::{MaterialKind, Scene};
use crate::shader::create_module;
use crate::targets::{FrameTargets, DEPTH_FORMAT, HDR_FORMAT};
use crate::textures::TextureLibrary;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, mip_count, mip_view, uniform_buffer, workgroups, Lazy};
use crate::water::{water_bodies, WaterBody, WaterUniform};

/// Mips of the scene-colour copy (blur for rough refraction and reflections).
const SCENE_COLOR_MIPS: u32 = 5;

/// Translucent layers per pixel composited in exact depth order: `MAX_LAYERS - 1` peeled ones
/// plus everything behind them in per-instance order.
const MAX_LAYERS: usize = 4;

/// Meshes with more triangles are not tested for convexity (the test is quadratic) and count
/// as concave.
const CONVEXITY_TEST_TRIANGLES: usize = 4096;

/// Vertex of the forward-drawn translucent meshes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct TranslucentVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Second UV set (the first where the mesh has none).
    pub uv1: [f32; 2],
    /// Linear RGBA vertex colour (white where the mesh has none).
    pub color: [f32; 4],
}

/// Decodes the finest crack-free LOD cut of `mesh` whose page payloads are in memory (the full
/// detail of a baked mesh; for a streamed one, the finest level loaded) into an indexed mesh.
/// Uses the virtual mesh's own decoders, so it follows its payload format.
pub(crate) fn decode_finest(mesh: &VirtualMesh) -> (Vec<TranslucentVertex>, Vec<u32>) {
    let view = LodView { position: Vec3::ZERO, lod_scale: 1.0, threshold: 0.0, orthographic: true };
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for cluster in lod_cut_resident(mesh, &view, &|page| mesh.has_payload(page as usize)) {
        let c = cluster as usize;
        let base = vertices.len() as u32;
        vertices.extend((0..mesh.clusters[c].vertex_count()).map(|i| {
            let (position, normal, uv) = mesh.cluster_vertex(c, i);
            let (uv1, color) = mesh.cluster_vertex_extra(c, i);
            TranslucentVertex { position: position.into(), normal: normal.into(), uv: uv.into(), uv1: uv1.into(), color: color.into() }
        }));
        indices.extend(mesh.cluster_triangles(c).flatten().map(|i| base + i));
    }
    (vertices, indices)
}

/// Shading model of a translucent material: one pipeline each (`SHADING` in translucency.wgsl).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shading {
    Water = 0,
    Glass = 1,
    /// Alpha-blended PBR, premultiplied "over".
    Blend = 2,
    /// The full-screen underwater view (not a material: drawn when the camera is under water).
    Underwater = 3,
}

impl Shading {
    const ALL: [Shading; 4] = [Shading::Water, Shading::Glass, Shading::Blend, Shading::Underwater];
    /// The materials' models (drawn as meshes), indexed by their value.
    const SURFACES: [Shading; 3] = [Shading::Water, Shading::Glass, Shading::Blend];

    fn of(kind: MaterialKind) -> Option<Self> {
        match kind {
            MaterialKind::Water => Some(Self::Water),
            MaterialKind::Glass => Some(Self::Glass),
            MaterialKind::Translucent => Some(Self::Blend),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MeshRange {
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
}

/// How often a decoded mesh may cover a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// In one plane: once, drawn one- or two-sided.
    Flat,
    /// Once with back faces culled (a closed convex mesh, or an open patch of one).
    Convex,
    /// Any number of times.
    Concave,
}

impl Shape {
    /// Classifies a mesh: every vertex lies on or behind the plane of every triangle (within
    /// the positions' quantisation) for a convex one, on the plane of the first for a flat one.
    fn of(vertices: &[TranslucentVertex], indices: &[u32]) -> Self {
        let positions: Vec<Vec3> = vertices.iter().map(|v| Vec3::from(v.position)).collect();
        let (lo, hi) = positions.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        let tolerance = (hi - lo).max_element().max(0.0) * 1e-3;
        let mut planes = indices.chunks_exact(3).filter_map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
            let n = (b - a).cross(c - a).try_normalize()?;
            Some((n, n.dot(a)))
        });
        let Some((n, d)) = planes.next() else { return Shape::Flat };
        if positions.iter().all(|p| (n.dot(*p) - d).abs() <= tolerance) {
            return Shape::Flat;
        }
        let convex = indices.len() / 3 <= CONVEXITY_TEST_TRIANGLES
            && std::iter::once((n, d)).chain(planes).all(|(n, d)| positions.iter().all(|p| n.dot(*p) - d <= tolerance));
        if convex {
            Shape::Convex
        } else {
            Shape::Concave
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DecodedMesh {
    range: MeshRange,
    shape: Shape,
}

/// A translucent instance with its world bounding sphere and box.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    instance: u32,
    shading: Shading,
    range: MeshRange,
    center: Vec3,
    radius: f32,
    aabb: Aabb,
    /// The instance covers any pixel at most once.
    single_layer: bool,
}

impl Candidate {
    /// Scene instance `index` with its decoded mesh and shading model, placed where it is now.
    fn of(scene: &Scene, index: usize, shading: Shading, decoded: DecodedMesh) -> Self {
        let instance = &scene.instances[index];
        let mesh = &scene.meshes[instance.mesh.0 as usize];
        let world = instance.transform;
        let linear = glam::Mat3::from_mat4(world);
        let scale = linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length());
        Self {
            instance: index as u32,
            shading,
            range: decoded.range,
            center: world.transform_point3(mesh.bounds.center),
            radius: mesh.bounds.radius * scale,
            aabb: Aabb::from_points(corners(&mesh.aabb).map(|p| world.transform_point3(p))),
            // Water is drawn two-sided: a closed convex body would show twice.
            single_layer: match decoded.shape {
                Shape::Flat => true,
                Shape::Convex => shading != Shading::Water,
                Shape::Concave => false,
            },
        }
    }
}

/// Indices into `candidates` of those intersecting the frustum, farthest first (painter's
/// order between instances; within an instance, back-face culling is all there is).
fn sort_back_to_front(candidates: &[Candidate], eye: Vec3, planes: &[Vec4; 6]) -> Vec<usize> {
    let mut visible: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| planes.iter().all(|p| p.truncate().dot(c.center) + p.w >= -c.radius))
        .map(|(i, c)| (i, c.center.distance_squared(eye)))
        .collect();
    visible.sort_by(|a, b| b.1.total_cmp(&a.1));
    visible.into_iter().map(|(i, _)| i).collect()
}

/// The corners of `aabb`.
fn corners(aabb: &Aabb) -> [Vec3; 8] {
    std::array::from_fn(|i| Vec3::select(glam::BVec3::new(i & 1 != 0, i & 2 != 0, i & 4 != 0), aabb.max, aabb.min))
}

/// Where a candidate may cover the screen (normalised device coordinates) and how many layers
/// deep.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ScreenRect {
    min: Vec2,
    max: Vec2,
    layers: usize,
}

impl ScreenRect {
    /// The candidate's world box projected, clipped to the screen; the whole screen when the box
    /// reaches behind the camera, `None` when it misses the screen.
    fn of(c: &Candidate, view_proj: &Mat4) -> Option<Self> {
        let clip = corners(&c.aabb).map(|corner| *view_proj * corner.extend(1.0));
        let behind = clip.iter().filter(|c| c.w <= 1e-6).count();
        let (min, max) = match behind {
            0 => clip.iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), c| {
                let ndc = c.truncate().truncate() / c.w;
                (lo.min(ndc), hi.max(ndc))
            }),
            8 => return None,
            _ => (Vec2::NEG_ONE, Vec2::ONE),
        };
        let (min, max) = (min.max(Vec2::NEG_ONE), max.min(Vec2::ONE));
        let layers = if c.single_layer { 1 } else { MAX_LAYERS };
        min.cmplt(max).all().then_some(Self { min, max, layers })
    }
}

/// The most layers stacked at any point: the deepest point of a set of rectangles lies on the
/// left edge of one of them, so each left edge is swept bottom to top. O(n² log n) for n
/// rectangles; stops at [`MAX_LAYERS`].
fn max_overlap(rects: &[ScreenRect]) -> usize {
    let mut deepest = 0;
    let mut events = Vec::with_capacity(rects.len() * 2);
    for probe in rects {
        let x = probe.min.x;
        events.clear();
        for r in rects.iter().filter(|r| r.min.x <= x && x <= r.max.x) {
            events.push((r.min.y, r.layers as isize));
            events.push((r.max.y, -(r.layers as isize)));
        }
        // Opening before closing at the same height: touching rectangles count as overlapping.
        events.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut depth = 0;
        for &(_, delta) in &events {
            depth += delta;
            deepest = deepest.max(depth as usize);
        }
        if deepest >= MAX_LAYERS {
            break;
        }
    }
    deepest
}

/// The lit HDR image of the opaque scene with a short mip chain.
struct SceneColor {
    size: (u32, u32),
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    mips: Vec<wgpu::TextureView>,
}

impl SceneColor {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let mips = SCENE_COLOR_MIPS.min(mip_count(size.0, size.1));
        let texture = crate::util::texture(
            device,
            crate::util::TextureSpec {
                label: "translucency scene colour",
                width: size.0,
                height: size.1,
                layers: 1,
                mips,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            },
        );
        Self { size, view: texture.create_view(&Default::default()), mips: (0..mips).map(|m| mip_view(&texture, m)).collect(), texture }
    }
}

/// Depth of the peeled translucent layers, nearest first (reverse-Z; 0 where a pixel has fewer
/// layers): one array texture whose layers are rendered one by one and read together by the
/// shading passes (one binding: fragment shaders may sample only 16 textures).
struct PeeledLayers {
    size: (u32, u32),
    layers: Vec<wgpu::TextureView>,
    array: wgpu::TextureView,
}

impl PeeledLayers {
    fn new(device: &wgpu::Device, size: (u32, u32), count: usize) -> Self {
        let texture = crate::util::texture(
            device,
            crate::util::TextureSpec {
                label: "translucency layer depth",
                width: size.0,
                height: size.1,
                layers: count as u32,
                mips: 1,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            },
        );
        let layer = |k| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: k as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let array = texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        Self { size, layers: (0..count).map(layer).collect(), array }
    }
}

/// A 1x1 depth beyond any reverse-Z depth (2.0): the near bound of the nearest peeled layer
/// (`fs_peel` clamps its reads to the texture); as an array, the layers bound when none are
/// peeled (then never read).
struct Unbounded {
    view: wgpu::TextureView,
    array: wgpu::TextureView,
}

impl Unbounded {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        use wgpu::util::DeviceExt;
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("translucency unbounded depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::bytes_of(&2.0f32),
        );
        Self {
            view: texture.create_view(&Default::default()),
            array: texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() }),
        }
    }
}

/// Which fragments a shading pass composites, as indices into the peeled layers
/// (`LayerRange` in translucency.wgsl): those with `far <= depth < near`, over a background
/// whose nearest composited surface is `behind`. [`NO_LAYER`] stands for the opaque depth (far,
/// behind) or no bound (near).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct LayerRange {
    far: i32,
    near: i32,
    behind: i32,
    pad: i32,
}

const NO_LAYER: i32 = -1;

/// Uniform offset alignment (WebGPU's default `minUniformBufferOffsetAlignment`).
const LAYER_RANGE_STRIDE: u64 = 256;

/// The shading passes' ranges with `peel_count` peeled layers, in drawing order: everything
/// behind the peeled layers, then each layer back to front.
fn layer_ranges(peel_count: usize) -> Vec<LayerRange> {
    let index = |k: usize| if k < peel_count { k as i32 } else { NO_LAYER };
    let behind_peeled = LayerRange { far: NO_LAYER, near: peel_count as i32 - 1, behind: NO_LAYER, pad: 0 };
    let layer = |k: usize| LayerRange { far: k as i32, near: k as i32 - 1, behind: index(k + 1), pad: 0 };
    std::iter::once(behind_peeled).chain((0..peel_count).rev().map(layer)).collect()
}

pub struct Translucency {
    layout: wgpu::BindGroupLayout,
    /// By [`Shading`], each compiled when a frame first draws that model ([`Lazy`]; in the
    /// browser in the background once the scene has it): a scene without water never pays the
    /// water pipeline's compile, 1–1.5 s in Firefox.
    pipelines: Vec<Lazy<wgpu::RenderPipeline>>,
    peel_layout: wgpu::BindGroupLayout,
    /// Depth-only peeling draws by [`Shading::SURFACES`].
    peel_pipelines: Vec<Lazy<wgpu::RenderPipeline>>,
    downsample_layout: wgpu::BindGroupLayout,
    downsample: Lazy<wgpu::ComputePipeline>,
    /// [`WaterUniform`]: the medium and the water bodies (the G-buffer resolve reads it too).
    water: wgpu::Buffer,
    water_bodies: Vec<WaterBody>,
    /// The water body the camera is in this frame.
    camera_body: Option<usize>,
    vertices: Option<wgpu::Buffer>,
    indices: Option<wgpu::Buffer>,
    /// Decoded meshes by scene mesh index (valid for `geometry_version`).
    decoded: Vec<Option<DecodedMesh>>,
    cpu_vertices: Vec<TranslucentVertex>,
    cpu_indices: Vec<u32>,
    candidates: Vec<Candidate>,
    /// Per scene instance: its candidate (`u32::MAX`: none).
    candidate_of: Vec<u32>,
    /// This frame's draws (indices into `candidates`), back to front.
    draws: Vec<usize>,
    /// Layers peeled this frame (0: one pass in per-instance order).
    peel_count: usize,
    /// Whether a refracting surface is drawn this frame (peeled layers need a fresh scene colour).
    refracting: bool,
    versions: (u64, u64, u64),
    scene_color: Option<SceneColor>,
    peeled: Option<PeeledLayers>,
    unbounded: Option<Unbounded>,
    /// [`LayerRange`]s of the shading passes, `LAYER_RANGE_STRIDE` apart.
    layer_ranges: wgpu::Buffer,
    /// The peel count `layer_ranges` was written for.
    ranges_written: Option<usize>,
}

impl Translucency {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "translucency.wgsl");
        let layout = layout(device);
        type Make = fn(&wgpu::Device, &wgpu::ShaderModule, &wgpu::BindGroupLayout, Shading) -> wgpu::RenderPipeline;
        let lazy = |make: Make, layout: &wgpu::BindGroupLayout, shading: Shading| {
            let (device, module, layout) = (device.clone(), module.clone(), layout.clone());
            Lazy::new(move || make(&device, &module, &layout, shading))
        };
        let pipelines = Shading::ALL.iter().map(|&s| lazy(pipeline, &layout, s)).collect();
        let peel_layout = bind_group_layout(
            device,
            "translucency peel",
            ShaderStages::VERTEX_FRAGMENT,
            &[(0, bind::uniform()), (1, bind::storage(true)), (4, bind::texture(false)), (108, bind::texture(false))],
        );
        let peel_pipelines = Shading::SURFACES.iter().map(|&s| lazy(peel_pipeline, &peel_layout, s)).collect();
        // The bloom chain's 13-tap downsample filter doubles as the blur of the mip chain.
        let bloom = create_module(device, "bloom.wgsl");
        let downsample_layout = bind_group_layout(
            device,
            "translucency downsample",
            ShaderStages::COMPUTE,
            &[(0, bind::texture(true)), (2, bind::sampler()), (3, bind::storage_texture(HDR_FORMAT))],
        );
        Self {
            downsample: compute_pipeline(device, "translucency downsample", &bloom, "downsample", &downsample_layout, &[]),
            downsample_layout,
            water: uniform_buffer(device, "water", std::mem::size_of::<WaterUniform>() as u64),
            water_bodies: Vec::new(),
            camera_body: None,
            layout,
            pipelines,
            peel_layout,
            peel_pipelines,
            vertices: None,
            indices: None,
            decoded: Vec::new(),
            cpu_vertices: Vec::new(),
            cpu_indices: Vec::new(),
            candidates: Vec::new(),
            candidate_of: Vec::new(),
            draws: Vec::new(),
            peel_count: 0,
            refracting: false,
            versions: (u64::MAX, u64::MAX, u64::MAX),
            scene_color: None,
            peeled: None,
            unbounded: None,
            layer_ranges: uniform_buffer(device, "translucency layer ranges", LAYER_RANGE_STRIDE * MAX_LAYERS as u64),
            ranges_written: None,
        }
    }

    /// Whether the camera was inside a water body at the last [`Translucency::update`].
    pub fn camera_in_water(&self) -> bool {
        self.camera_body.is_some()
    }

    /// The water uniform ([`WaterUniform`], written by [`Translucency::update`]).
    pub fn water_uniform(&self) -> &wgpu::Buffer {
        &self.water
    }

    /// Mirrors the scene's translucent instances and water bodies (moved ones in place), culls
    /// and sorts them for this frame's view, decides how many layers to peel (none with `layers`
    /// false) and sizes the scene-colour copy and the layers. With `enabled` false nothing is
    /// drawn (the water uniform is still written: lake beds need it).
    #[allow(clippy::too_many_arguments)]
    /// Queues the pipelines of the translucent models the scene's visible instances use for the
    /// browser's background compile.
    pub fn prefetch(&self, scene: &Scene) {
        for instance in scene.instances.iter().filter(|i| i.visible) {
            let Some(shading) = scene.materials.get(instance.material.0 as usize).and_then(|m| Shading::of(m.kind)) else { continue };
            self.pipelines[shading as usize].prefetch();
            self.peel_pipelines[shading as usize].prefetch();
            if shading == Shading::Water {
                self.pipelines[Shading::Underwater as usize].prefetch();
            }
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        changes: &SceneChanges,
        eye: Vec3,
        view_proj: &Mat4,
        size: (u32, u32),
        enabled: bool,
        layers: bool,
    ) {
        let versions = (scene.geometry_version, scene.material_version, scene.instance_version);
        if versions != self.versions || changes.structural {
            self.sync(device, scene, versions.0 != self.versions.0);
            self.versions = versions;
        } else {
            self.move_instances(scene, changes);
        }
        self.draws = if enabled { sort_back_to_front(&self.candidates, eye, &frustum_planes(view_proj)) } else { Vec::new() };
        self.camera_body = if enabled { self.water_bodies.iter().position(|b| b.contains(eye)) } else { None };
        let uniform = WaterUniform::new(&scene.water, &self.water_bodies, self.camera_body);
        queue.write_buffer(&self.water, 0, bytemuck::bytes_of(&uniform));
        let peel_count = if layers {
            let rects: Vec<ScreenRect> = self.draws.iter().filter_map(|&i| ScreenRect::of(&self.candidates[i], view_proj)).collect();
            max_overlap(&rects).min(MAX_LAYERS).saturating_sub(1)
        } else {
            0
        };
        if peel_count != self.peel_count {
            log::debug!("translucency: peeling {peel_count} layers");
            self.peel_count = peel_count;
        }
        self.refracting = self.draws.iter().any(|&i| self.candidates[i].shading != Shading::Blend);
        if self.draws.is_empty() && self.camera_body.is_none() {
            return;
        }
        if self.scene_color.as_ref().is_none_or(|c| c.size != size) {
            self.scene_color = Some(SceneColor::new(device, size));
        }
        // Kept once allocated (grown, never shrunk), so views moving in and out of overlap don't
        // reallocate.
        if self.peel_count > 0 && self.peeled.as_ref().is_none_or(|p| p.size != size || p.layers.len() < self.peel_count) {
            let count = self.peel_count.max(self.peeled.as_ref().map_or(0, |p| p.layers.len()));
            self.peeled = Some(PeeledLayers::new(device, size, count));
        }
        if self.unbounded.is_none() {
            self.unbounded = Some(Unbounded::new(device, queue));
        }
        if self.ranges_written != Some(self.peel_count) {
            for (i, range) in layer_ranges(self.peel_count).iter().enumerate() {
                queue.write_buffer(&self.layer_ranges, i as u64 * LAYER_RANGE_STRIDE, bytemuck::bytes_of(range));
            }
            self.ranges_written = Some(self.peel_count);
        }
    }

    /// Re-places the candidates of moved instances (and the water bodies if one moved).
    fn move_instances(&mut self, scene: &Scene, changes: &SceneChanges) {
        let mut water_moved = false;
        for m in changes.motions(scene) {
            let Some(&c) = self.candidate_of.get(m.index as usize).filter(|&&c| c != u32::MAX) else { continue };
            let c = &mut self.candidates[c as usize];
            let mesh = scene.instances[m.index as usize].mesh.0 as usize;
            let decoded = self.decoded[mesh].expect("candidates have decoded meshes");
            *c = Candidate::of(scene, m.index as usize, c.shading, decoded);
            water_moved |= c.shading == Shading::Water;
        }
        if water_moved {
            self.water_bodies = water_bodies(scene);
        }
    }

    /// Rebuilds the candidate list; decodes meshes that translucent instances newly use.
    fn sync(&mut self, device: &wgpu::Device, scene: &Scene, geometry_changed: bool) {
        if geometry_changed {
            self.decoded.clear();
            self.cpu_vertices.clear();
            self.cpu_indices.clear();
        }
        self.decoded.resize(scene.meshes.len(), None);
        self.water_bodies = water_bodies(scene);
        let mut appended = false;
        self.candidates.clear();
        self.candidate_of = vec![u32::MAX; scene.instances.len()];
        for (index, instance) in scene.instances.iter().enumerate().filter(|(_, i)| i.visible) {
            let Some(shading) = scene.materials.get(instance.material.0 as usize).and_then(|m| Shading::of(m.kind)) else { continue };
            let mesh_index = instance.mesh.0 as usize;
            let mesh = &scene.meshes[mesh_index];
            let decoded = match self.decoded[mesh_index] {
                Some(decoded) => decoded,
                None => {
                    let (vertices, indices) = decode_finest(mesh);
                    let range = MeshRange {
                        first_index: self.cpu_indices.len() as u32,
                        index_count: indices.len() as u32,
                        base_vertex: self.cpu_vertices.len() as i32,
                    };
                    let decoded = DecodedMesh { range, shape: Shape::of(&vertices, &indices) };
                    self.cpu_vertices.extend(vertices);
                    self.cpu_indices.extend(indices);
                    self.decoded[mesh_index] = Some(decoded);
                    appended = true;
                    decoded
                }
            };
            if decoded.range.index_count == 0 {
                continue;
            }
            self.candidate_of[index] = self.candidates.len() as u32;
            self.candidates.push(Candidate::of(scene, index, shading, decoded));
        }
        if appended {
            use wgpu::util::DeviceExt;
            let buffer = |label, contents: &[u8], usage| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
            };
            self.vertices = Some(buffer("translucent vertices", bytemuck::cast_slice(&self.cpu_vertices), wgpu::BufferUsages::VERTEX));
            self.indices = Some(buffer("translucent indices", bytemuck::cast_slice(&self.cpu_indices), wgpu::BufferUsages::INDEX));
            log::info!(
                "translucency: {} instances, {} triangles decoded",
                self.candidates.len(),
                self.cpu_indices.len() / 3
            );
        }
    }

    /// Records the layer peeling, the scene-colour copies and the translucent draws over
    /// `targets.hdr` (after the deferred lighting, before TSR).
    pub fn record(&self, rec: &mut Recorder, inputs: &LightingInputs, scene: &GpuScene, textures: &TextureLibrary) {
        let Some(color) = &self.scene_color else { return };
        if self.draws.is_empty() && self.camera_body.is_none() {
            return;
        }
        let Some(unbounded) = &self.unbounded else { return };
        let t = inputs.targets;
        let peeled = match &self.peeled {
            Some(p) if self.peel_count > 0 => p,
            _ => {
                // One pass in per-instance order.
                self.copy_scene_color(rec, inputs, color);
                self.shade(rec, inputs, scene, textures, color, &unbounded.array, 0, self.camera_body.is_some());
                return;
            }
        };
        let layers = &peeled.layers[..self.peel_count];
        // The nearest layers' depths, front to back: each keeps the nearest fragment behind the
        // one before.
        for (k, layer) in layers.iter().enumerate() {
            let near = if k == 0 { &unbounded.view } else { &layers[k - 1] };
            let group = bind_group(
                rec.device,
                "translucency peel",
                &self.peel_layout,
                &[(0, inputs.frame.as_entire_binding()), (1, scene.instances.as_entire_binding()), (4, TextureView(&t.depth)), (108, TextureView(near))],
            );
            let mut pass = rec.render_pass(
                "translucency peel",
                &[],
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
            );
            let size = internal_size(t, color.size);
            pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
            pass.set_bind_group(0, &group, &[]);
            self.draw_surfaces(&mut pass, &self.peel_pipelines);
        }
        // Everything behind the peeled layers in per-instance order, the underwater view first;
        // then the layers back to front, each over a fresh copy of what lies behind it when it
        // may refract that (see `layer_ranges`).
        for pass in 0..=self.peel_count {
            if pass == 0 || self.refracting {
                self.copy_scene_color(rec, inputs, color);
            }
            self.shade(rec, inputs, scene, textures, color, &peeled.array, pass, pass == 0 && self.camera_body.is_some());
        }
    }

    /// Draws the depth of the nearest translucent surface in front of the opaque scene into
    /// `target` (0 elsewhere): where particles and volumes are split into what the translucent
    /// surfaces cover and what lies in front of them (`oit.rs`). The first peeling layer,
    /// recorded on its own. Returns false (and leaves `target` alone) when no translucent
    /// surface is drawn this frame.
    pub fn record_nearest_depth(&self, rec: &mut Recorder, frame: &wgpu::Buffer, scene: &GpuScene, targets: &FrameTargets, target: &wgpu::TextureView) -> bool {
        let (Some(color), Some(unbounded)) = (&self.scene_color, &self.unbounded) else { return false };
        if self.draws.is_empty() {
            return false;
        }
        let group = bind_group(
            rec.device,
            "translucency nearest depth",
            &self.peel_layout,
            &[(0, frame.as_entire_binding()), (1, scene.instances.as_entire_binding()), (4, TextureView(&targets.depth)), (108, TextureView(&unbounded.view))],
        );
        let mut pass = rec.render_pass(
            "translucency nearest depth",
            &[],
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: target,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
        );
        let size = internal_size(targets, color.size);
        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
        pass.set_bind_group(0, &group, &[]);
        self.draw_surfaces(&mut pass, &self.peel_pipelines);
        true
    }

    /// Copies the HDR image into the scene colour and blurs its mip chain: the corner the
    /// internal resolution covers (the copy is allocated for the largest one).
    fn copy_scene_color(&self, rec: &mut Recorder, inputs: &LightingInputs, color: &SceneColor) {
        let t = inputs.targets;
        let hdr = t.hdr.texture();
        let size = internal_size(t, color.size);
        rec.encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: hdr, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo { texture: &color.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
        let device = rec.device;
        let mut pass = rec.compute_pass("translucency scene colour");
        pass.set_pipeline(&self.downsample);
        for level in 1..color.mips.len() {
            let group = bind_group(
                device,
                "translucency downsample",
                &self.downsample_layout,
                &[(0, TextureView(&color.mips[level - 1])), (2, Sampler(&inputs.samplers.linear)), (3, TextureView(&color.mips[level]))],
            );
            pass.set_bind_group(0, &group, &[]);
            let (w, h) = ((size.0 >> level).max(1), (size.1 >> level).max(1));
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }
    }

    /// Shading pass `pass` over `targets.hdr`: the fragments in its [`LayerRange`] (of the
    /// peeled depths `layers`), over the current scene colour.
    #[allow(clippy::too_many_arguments)]
    fn shade(
        &self,
        rec: &mut Recorder,
        inputs: &LightingInputs,
        scene: &GpuScene,
        textures: &TextureLibrary,
        color: &SceneColor,
        layers: &wgpu::TextureView,
        pass: usize,
        underwater: bool,
    ) {
        let range = wgpu::BufferBinding {
            buffer: &self.layer_ranges,
            offset: pass as u64 * LAYER_RANGE_STRIDE,
            size: wgpu::BufferSize::new(std::mem::size_of::<LayerRange>() as u64),
        };
        let t = inputs.targets;
        let group = bind_group(
            rec.device,
            "translucency",
            &self.layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, scene.instances.as_entire_binding()),
                (2, scene.materials.as_entire_binding()),
                (3, self.water.as_entire_binding()),
                (4, TextureView(&t.depth)),
                (5, TextureView(&color.view)),
                (6, TextureView(&inputs.sky.transmittance)),
                (7, TextureView(&inputs.sky.sky_view)),
                (8, Sampler(&inputs.samplers.linear)),
                (9, inputs.sky.sh.as_entire_binding()),
                (10, inputs.vsm.uniform.as_entire_binding()),
                (11, inputs.vsm.cache.entries.as_entire_binding()),
                (12, TextureView(&inputs.vsm.atlas)),
                (13, Sampler(&inputs.samplers.shadow)),
                (14, inputs.lights.lights.as_entire_binding()),
                (15, inputs.lights.grid.as_entire_binding()),
                (16, TextureView(inputs.cloud_texture)),
                (17, TextureView(&inputs.clouds.shadow)),
                (18, inputs.clouds.uniform.as_entire_binding()),
                (19, TextureView(inputs.fog.integrated())),
                (20, inputs.fog.uniform.as_entire_binding()),
                (21, inputs.lumen.uniform().as_entire_binding()),
                (22, TextureView(inputs.lumen.debug_views().0)),
                (23, TextureView(inputs.lumen.debug_views().1)),
                (24, inputs.lumen.radiance_probes().as_entire_binding()),
                (25, Sampler(&inputs.samplers.linear)),
                (26, inputs.lumen.near_field().0.as_entire_binding()),
                (27, TextureView(inputs.lumen.near_field().1)),
                (28, TextureView(&textures.albedo)),
                (29, TextureView(&textures.data)),
                (30, Sampler(&inputs.samplers.material)),
                (31, TextureView(&t.gbuffer_normal)),
                (60, inputs.vsm.local.uniform.as_entire_binding()),
                (105, TextureView(layers)),
                (106, wgpu::BindingResource::Buffer(range)),
            ],
        );
        let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
        let size = internal_size(t, color.size);
        let mut pass = rec.render_pass(
            "translucency",
            &[
                Some(wgpu::RenderPassColorAttachment { view: &t.hdr, depth_slice: None, resolve_target: None, ops: load }),
                Some(wgpu::RenderPassColorAttachment { view: &t.gbuffer_motion, depth_slice: None, resolve_target: None, ops: load }),
            ],
            None,
        );
        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
        pass.set_bind_group(0, &group, &[]);
        // Under water: the water between the camera and the opaque scene, before any surface.
        if underwater {
            pass.set_pipeline(&self.pipelines[Shading::Underwater as usize]);
            pass.draw(0..3, 0..1);
        }
        self.draw_surfaces(&mut pass, &self.pipelines);
    }

    /// Draws this frame's surfaces in back-to-front instance order with `pipelines` (by
    /// [`Shading`]).
    fn draw_surfaces(&self, pass: &mut wgpu::RenderPass, pipelines: &[Lazy<wgpu::RenderPipeline>]) {
        let (Some(vertices), Some(indices)) = (&self.vertices, &self.indices) else { return };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        let mut current = None;
        for &i in &self.draws {
            let c = &self.candidates[i];
            if current != Some(c.shading) {
                pass.set_pipeline(&pipelines[c.shading as usize]);
                current = Some(c.shading);
            }
            let r = c.range;
            pass.draw_indexed(r.first_index..r.first_index + r.index_count, r.base_vertex, c.instance..c.instance + 1);
        }
    }
}

/// This frame's internal resolution: the top-left corner of the targets (allocated at
/// `capacity`) that the frame renders to.
fn internal_size(targets: &FrameTargets, capacity: (u32, u32)) -> (u32, u32) {
    (targets.width.min(capacity.0), targets.height.min(capacity.1))
}

fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let fragment = ShaderStages::FRAGMENT;
    let both = ShaderStages::VERTEX_FRAGMENT;
    let entries: [(u32, ShaderStages, BindingType); 35] = [
        (0, both, bind::uniform()),
        (1, both, bind::storage(true)),
        (2, fragment, bind::uniform()),
        (3, fragment, bind::uniform()),
        (4, fragment, bind::texture(false)),
        (5, fragment, bind::texture(true)),
        (6, fragment, bind::texture(true)),
        (7, fragment, bind::texture(true)),
        (8, fragment, bind::sampler()),
        (9, fragment, bind::storage(true)),
        (10, fragment, bind::uniform()),
        (11, fragment, bind::storage(true)),
        (12, fragment, bind::texture_depth()),
        (13, fragment, bind::sampler_comparison()),
        (14, fragment, bind::storage(true)),
        (15, fragment, bind::storage(true)),
        (16, fragment, bind::texture(true)),
        (17, fragment, bind::texture(true)),
        (18, fragment, bind::uniform()),
        (19, fragment, bind::texture_3d(true)),
        (20, fragment, bind::uniform()),
        (21, fragment, bind::uniform()),
        (22, fragment, bind::texture_3d(true)),
        (23, fragment, bind::texture_3d(true)),
        (24, fragment, bind::storage(true)),
        (25, fragment, bind::sampler()),
        (26, fragment, bind::storage(true)),
        (27, fragment, bind::texture_3d(true)),
        (28, fragment, bind::texture_array(true)),
        (29, fragment, bind::texture_array(true)),
        (30, fragment, bind::sampler()),
        (31, fragment, bind::texture(false)),
        (60, fragment, bind::uniform()),
        (105, fragment, bind::texture_array(false)),
        (106, fragment, bind::uniform()),
    ];
    let entries: Vec<wgpu::BindGroupLayoutEntry> =
        entries.iter().map(|&(binding, visibility, ty)| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None }).collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("translucency"), entries: &entries })
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2, 4 => Float32x4];

fn mesh_buffer() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<TranslucentVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    }
}

fn primitive(shading: Shading) -> wgpu::PrimitiveState {
    // Water is two-sided: from below, its surface shows Snell's window.
    let cull_mode = match shading {
        Shading::Water | Shading::Underwater => None,
        Shading::Glass | Shading::Blend => Some(wgpu::Face::Back),
    };
    wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, front_face: wgpu::FrontFace::Ccw, cull_mode, ..Default::default() }
}

fn pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, shading: Shading) -> wgpu::RenderPipeline {
    let pipeline_layout = crate::util::pipeline_layout(device, "translucency", &[layout]);
    let constants = [("SHADING", shading as u32 as f64)];
    let mesh_buffers = [Some(mesh_buffer())];
    let (vertex_entry, fragment_entry, buffers): (_, _, &[_]) =
        if shading == Shading::Underwater { ("vs_fullscreen", "fs_underwater", &[]) } else { ("vs_main", "fs_main", &mesh_buffers) };
    // Refracting models output the final pixel (they read the background themselves); blended
    // surfaces are premultiplied "over". Motion vectors blend by the w channel (0 or 1).
    let over = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let color_blend = (shading == Shading::Blend).then_some(wgpu::BlendState { color: over, alpha: over });
    let motion_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent::REPLACE,
    };
    let targets = [
        Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: color_blend, write_mask: wgpu::ColorWrites::COLOR }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(motion_blend),
            write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
        }),
    ];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("translucency"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
            buffers,
        },
        primitive: primitive(shading),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Depth-only draw of `shading`'s surfaces that peels a layer: `fs_peel` keeps the fragments in
/// front of the opaque scene and behind the previous layer, the depth test the nearest of them.
fn peel_pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, shading: Shading) -> wgpu::RenderPipeline {
    let pipeline_layout = crate::util::pipeline_layout(device, "translucency peel", &[layout]);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("translucency peel"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(mesh_buffer())] },
        primitive: primitive(shading),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some("fs_peel"), compilation_options: Default::default(), targets: &[] }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use helios_geometry::{build_virtual_mesh, primitives, BuildConfig};

    #[test]
    fn decoding_recovers_the_full_detail_mesh() {
        let source = primitives::rock(3, 4);
        let mesh = build_virtual_mesh(&source, &BuildConfig::default());
        let (vertices, indices) = decode_finest(&mesh);
        assert_eq!(indices.len(), source.triangle_count() * 3);
        assert!(indices.iter().all(|&i| (i as usize) < vertices.len()));
        let (lo, hi) = vertices.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| {
            (lo.min(v.position.into()), hi.max(v.position.into()))
        });
        // Positions are quantised in the page payload: equal up to a small fraction of the size.
        let tolerance = mesh.aabb.extent().max_element() * 1e-3;
        assert!(lo.abs_diff_eq(mesh.aabb.min, tolerance) && hi.abs_diff_eq(mesh.aabb.max, tolerance));
    }

    #[test]
    fn streamed_meshes_decode_their_finest_loaded_level() {
        let source = primitives::rock(5, 4);
        let mut mesh = build_virtual_mesh(&source, &BuildConfig::default());
        // Keep only the root pages in memory, as a mesh loaded from its header would.
        let roots = mesh.root_tail_start();
        let tail = mesh.page_offsets[roots] as usize;
        mesh.page_data.drain(..tail);
        mesh.payload_first_page = roots as u32;
        let (_, indices) = decode_finest(&mesh);
        assert!(!indices.is_empty() && indices.len() < source.triangle_count() * 3);
    }

    #[test]
    fn draws_are_culled_and_sorted_back_to_front() {
        let candidates = [at(-5.0, 1.0), at(10.0, 1.0), at(-50.0, 1.0), at(-20.0, 1.0)];
        assert_eq!(sort_back_to_front(&candidates, Vec3::ZERO, &frustum_planes(&view_proj())), vec![2, 3, 0]);
    }

    /// A single-layer candidate: a box of half size `r` at (0, 0, z).
    fn at(z: f32, r: f32) -> Candidate {
        let center = Vec3::new(0.0, 0.0, z);
        Candidate {
            instance: 0,
            shading: Shading::Glass,
            range: MeshRange { first_index: 0, index_count: 3, base_vertex: 0 },
            center,
            radius: r * 3f32.sqrt(),
            aabb: Aabb { min: center - r, max: center + r },
            single_layer: true,
        }
    }

    /// Camera at the origin looking down -Z, square aspect.
    fn view_proj() -> Mat4 {
        let camera = crate::Camera { position: Vec3::ZERO, ..Default::default() };
        camera.projection(1.0) * camera.view_matrix()
    }

    fn shape(mesh: &helios_geometry::Mesh) -> Shape {
        let (vertices, indices) = decode_finest(&build_virtual_mesh(mesh, &BuildConfig::default()));
        Shape::of(&vertices, &indices)
    }

    #[test]
    fn meshes_are_classified_by_how_often_they_cover_a_pixel() {
        assert_eq!(shape(&primitives::terrain(10.0, 8, 0.0, 1)), Shape::Flat);
        let pane = helios_geometry::modeling::cuboid(Vec3::new(-1.2, -3.0, -0.02), Vec3::new(1.2, 1.0, 0.02));
        assert_eq!(shape(&pane), Shape::Convex);
        assert_eq!(shape(&primitives::torus_knot(2, 3, 64, 8, 0.3)), Shape::Concave);
        assert_eq!(shape(&primitives::terrain(10.0, 8, 2.0, 1)), Shape::Concave);
    }

    fn rect(min: (f32, f32), max: (f32, f32), layers: usize) -> ScreenRect {
        ScreenRect { min: Vec2::new(min.0, min.1), max: Vec2::new(max.0, max.1), layers }
    }

    #[test]
    fn overlap_counts_the_deepest_point() {
        assert_eq!(max_overlap(&[]), 0);
        let a = rect((-1.0, -1.0), (0.0, 0.0), 1);
        let b = rect((0.5, 0.5), (1.0, 1.0), 1);
        assert_eq!(max_overlap(&[a, b]), 1);
        let c = rect((-0.5, -0.5), (0.75, 0.75), 1);
        assert_eq!(max_overlap(&[a, b, c]), 2);
        // Pairwise overlapping without a common point: still two deep.
        let d = rect((-1.0, 0.25), (1.0, 0.3), 1);
        assert_eq!(max_overlap(&[rect((-1.0, -1.0), (-0.5, 1.0), 1), rect((0.5, -1.0), (1.0, 1.0), 1), d]), 2);
        // Three stacked; touching edges count (conservative).
        assert_eq!(max_overlap(&[a, c, rect((0.0, 0.0), (1.0, 1.0), 1)]), 3);
        // A concave instance alone needs every layer.
        assert_eq!(max_overlap(&[rect((-0.1, -0.1), (0.1, 0.1), MAX_LAYERS)]), MAX_LAYERS);
    }

    #[test]
    fn screen_rects_cover_what_the_camera_sees() {
        let vp = view_proj();
        let near = ScreenRect::of(&at(-10.0, 1.0), &vp).unwrap();
        let far = ScreenRect::of(&at(-40.0, 1.0), &vp).unwrap();
        assert!(near.min.abs_diff_eq(-near.max, 1e-5) && far.min.abs_diff_eq(-far.max, 1e-5));
        assert!(far.max.x < near.max.x && near.max.x < 1.0);
        assert_eq!(max_overlap(&[near, far]), 2);
        // Off to the side, behind the camera, around the camera.
        let mut side = at(-10.0, 1.0);
        side.aabb.min.x += 100.0;
        side.aabb.max.x += 100.0;
        assert_eq!(ScreenRect::of(&side, &vp), None);
        assert_eq!(ScreenRect::of(&at(10.0, 1.0), &vp), None);
        let around = ScreenRect::of(&at(0.0, 1.0), &vp).unwrap();
        assert_eq!((around.min, around.max), (Vec2::NEG_ONE, Vec2::ONE));
    }

    #[test]
    fn layer_ranges_composite_back_to_front() {
        let r = |far, near, behind| LayerRange { far, near, behind, pad: 0 };
        // No peeling: everything in front of the opaque scene.
        assert_eq!(layer_ranges(0), vec![r(NO_LAYER, NO_LAYER, NO_LAYER)]);
        // Behind layer 1, then layer 1 over it, then layer 0 over layer 1.
        assert_eq!(layer_ranges(2), vec![r(NO_LAYER, 1, NO_LAYER), r(1, 0, NO_LAYER), r(0, NO_LAYER, 1)]);
        assert_eq!(layer_ranges(MAX_LAYERS - 1).len(), MAX_LAYERS);
    }
}
