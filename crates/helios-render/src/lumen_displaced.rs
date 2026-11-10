//! Lumen follows Nanite tessellation's displacement: every instance whose material displaces its
//! mesh is represented in Lumen — mesh distance field, mesh cards and through them the global
//! distance field, the near field and the voxel and card surface caches — by a *displaced
//! variant* of its mesh, baked from the displaced surface, instead of the mesh's own.
//!
//! Displacement is a property of the material at render time, and the same mesh is displaced
//! differently by different materials (the gallery's bed is grown into mushrooms by one and
//! ripples as cloth with another), so variants are per (mesh, displacement): per (mesh, material)
//! pair, and per instance where the displacement depends on where the instance stands (graphs that
//! read the world position or normal, world-space vectors, world triplanar heights).
//!
//! **Baking.** The mesh's triangles (a LOD cut finer than the variant's voxels) are refined into a
//! grid of points spaced at most ¾ of a voxel apart, shared points welded. The GPU displaces
//! the points with the tessellator's functions (`lumen_displace.wgsl`: `displacement_position` of
//! `displacement_height.wgsl` and, for material graphs, the dicing passes' generated module from
//! [`graphs::scene_dispatch_module`]), so the variant shows exactly the surface the tessellator
//! renders — texture heights, graph heights and vectors alike — filtered for the point spacing.
//! The positions are read back and the distance field and the cards' triangles are baked on the
//! CPU with the same code as every other mesh's ([`MeshSdf::bake_volume`] /
//! [`MeshSdf::bake_heightfield`] at the mesh's own voxel size, [`helios_geometry::cards`]).
//! The CPU work runs off the render thread (`lumen_offload.rs`: worker threads natively, Web
//! Workers in the browser). Until a variant is ready (a few frames after the scene changes) its
//! instances keep the previous variant or the undisplaced mesh.
//!
//! **Animated displacement** (graphs that read the time) is represented by its *mean surface*:
//! the displacement averaged over [`ANIMATED_SAMPLES`] times spread over [`ANIMATED_WINDOW`]
//! seconds. It is the static surface closest to the moving one on average (a snapshot is off by
//! √2 times as much, and a periodic refresh would have to rebake several times per period of the
//! motion to do better, at a full bake and Lumen upload each); the bake logs how far the surface
//! moves from it (RMS at the point that moves most: 3.6 cm for the gallery's cloth, about one of
//! its 3.3 cm voxels).
//!
//! Scenes and meshes without displacement cost nothing: no variant is requested, and Lumen uploads
//! the meshes' own fields as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use helios_geometry::cards::CardVertex;
use helios_geometry::{lod_cut_resident, Aabb, LodView, MeshSdf, SdfKind, VirtualMesh};
use wgpu::BindingResource::{Sampler, TextureView};

use crate::cards::{CornerSource, SurfaceCapture};
use crate::gpu_scene::GpuScene;
use crate::lumen_offload::{self, BakeMesh, Output, Work};
use crate::profiler::Recorder;
use crate::scene::{Scene, TextureMapping};
use crate::shader::{compose, compose_variant, create_module_from};
use crate::tessellation::{displacement_source, graphs, DisplacementSource, TessView, BINDING_TESSELLATION, UNIFORM_SIZE};
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, storage_buffer_init, uniform_buffer};

/// An animated displacement is averaged over this many seconds ...
pub const ANIMATED_WINDOW: f32 = 8.0;
/// ... at this many times.
pub const ANIMATED_SAMPLES: u32 = 32;
/// Refined triangles per variant at most (the point spacing grows to stay within it).
const MAX_TRIANGLES: usize = 1 << 19;
/// Point spacing in voxels; refinement rounds the segments of an edge up to a power of two, so
/// points end up ⅜–¾ of a voxel apart.
const SPACING: f32 = 0.75;
/// Points displaced per job at most (variants beyond wait for the next job).
const MAX_JOB_POINTS: usize = 1 << 21;
/// One variant per job in the browser: the page uploads a job's points and hands its positions to
/// the bakes in one frame, so a job of one variant bounds that frame's copying (the GPU work of a
/// job is small; a variant waits a few frames longer).
const ONE_VARIANT_PER_JOB: bool = cfg!(target_arch = "wasm32");
/// Segments per base triangle edge at most.
const MAX_SEGMENTS: u32 = 64;
/// Displacement filter of the job's `tess` header: a dice segment is `max(distance, FAR) ·
/// spacing / FAR`, i.e. the point spacing anywhere in the scene.
const FAR: f32 = 1.0e7;

/// A mesh as Lumen represents it: its own distance field (and cards), or a displaced variant's.
#[derive(Clone, Copy)]
pub(crate) enum Surface<'a> {
    Mesh(&'a VirtualMesh),
    Displaced(&'a DisplacedVariant),
}

impl<'a> Surface<'a> {
    pub fn sdf(&self) -> Option<&'a MeshSdf> {
        match self {
            Surface::Mesh(mesh) => mesh.sdf.as_ref(),
            Surface::Displaced(variant) => Some(&variant.sdf),
        }
    }

    /// Mesh-space bounds.
    pub fn aabb(&self) -> Aabb {
        match self {
            Surface::Mesh(mesh) => mesh.aabb,
            Surface::Displaced(variant) => variant.aabb,
        }
    }
}

/// The distance field and card geometry of a displaced (mesh, displacement) pair, in mesh space.
#[derive(Debug, PartialEq)]
pub(crate) struct DisplacedVariant {
    pub sdf: MeshSdf,
    pub aabb: Aabb,
    /// The displaced surface (positions and normals), which the mesh cards capture.
    pub triangles: Vec<[CardVertex; 3]>,
    /// What the cards' material capture evaluates at the triangles' corners: per refined point
    /// its base point (before displacement) and texture coordinates, and per triangle its
    /// points (`triangles` are those points displaced).
    pub base_points: Vec<CornerSource>,
    pub corners: Vec<[u32; 3]>,
    /// Its cards and their texels' material inputs, captured with the bake for the instances'
    /// largest scale at the default texel: (scale, texel, capture). Lumen's upload captures again
    /// only if those differ.
    pub cards: Option<(f32, f32, SurfaceCapture)>,
    /// Unique per bake: identifies its cached card capture.
    pub serial: u64,
}

impl DisplacedVariant {
    /// The material sources of `triangles`' corners.
    pub fn sources(&self) -> Vec<[CornerSource; 3]> {
        self.corners.iter().map(|t| t.map(|i| self.base_points[i as usize])).collect()
    }
}

/// What a displaced variant is of. `content` hashes everything the displacement reads (the
/// material's displacement settings, texture and mapping, its graph's displacement code and
/// parameter values); `placement` is the instance transform for displacements that depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Slot {
    mesh: u32,
    material: u32,
    placement: Option<[u32; 16]>,
}

/// A variant some instance wants: its slot, content, the instance displacing the points on the
/// GPU, the point spacing (mesh units), the largest scale the slot is instanced at (for its
/// cards) and whether the displacement is animated.
#[derive(Clone, Copy, Debug)]
struct Request {
    slot: Slot,
    content: u64,
    instance: u32,
    spacing: f32,
    card_scale: f32,
    animated: bool,
}

/// One point of a refined surface (`SurfacePoint` in lumen_displace.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct SurfacePoint {
    pub object_u: [f32; 4],
    pub normal_v: [f32; 4],
}

/// `DisplaceJob` in lumen_displace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JobUniform {
    instance: u32,
    first: u32,
    count: u32,
    samples: u32,
    time: Vec4,
}

/// A mesh's triangles refined for sampling a displacement: welded points and triangles over them.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Refined {
    pub points: Vec<SurfacePoint>,
    pub triangles: Vec<[u32; 3]>,
}

/// A triangle corner: position, normal, UV.
pub(crate) type Corner = (Vec3, Vec3, Vec2);

fn corner_bits(c: &Corner) -> [u32; 8] {
    [c.0.x, c.0.y, c.0.z, c.1.x, c.1.y, c.1.z, c.2.x, c.2.y].map(f32::to_bits)
}

fn lerp_corner(a: &Corner, b: &Corner, t: f32) -> Corner {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
}

/// Point `k` of `n` along the edge from `a` to `b`, computed from the edge's canonical end so
/// both triangles of a shared edge make bit-identical points (which then weld).
fn edge_point(a: &Corner, b: &Corner, k: u32, n: u32) -> Corner {
    if corner_bits(a) <= corner_bits(b) {
        lerp_corner(a, b, k as f32 / n as f32)
    } else {
        lerp_corner(b, a, (n - k) as f32 / n as f32)
    }
}

/// Segments per edge of triangle `t` so that its refined edges are at most `spacing` long: a
/// power of two, so the points of a shared edge refined more finely on one side include the other
/// side's (they weld; the finer side's extra points leave sub-spacing T-junctions, which the
/// distance-field bake tolerates like any hairline crack).
fn segments(t: &[Corner; 3], spacing: f32) -> u32 {
    let longest = (0..3).map(|k| t[k].0.distance(t[(k + 1) % 3].0)).fold(0.0f32, f32::max);
    ((longest / spacing.max(1e-9)).ceil() as u32).clamp(1, MAX_SEGMENTS).next_power_of_two()
}

/// Splits every triangle into n² by a uniform barycentric grid (n from [`segments`], the spacing
/// grown until the refined triangles fit [`MAX_TRIANGLES`]); points shared by triangles (vertices,
/// edges with equal attributes at both ends) are emitted once.
pub(crate) fn refine(triangles: &[[Corner; 3]], spacing: f32) -> Refined {
    let mut spacing = spacing;
    while triangles.iter().map(|t| segments(t, spacing).pow(2) as usize).sum::<usize>() > MAX_TRIANGLES.max(triangles.len()) {
        spacing *= 1.25;
    }
    let mut out = Refined::default();
    let mut index: HashMap<[u32; 8], u32> = HashMap::new();
    let mut grid = Vec::new();
    for t in triangles {
        let [a, b, c] = t;
        let n = segments(t, spacing);
        grid.clear();
        for i in 0..=n {
            for j in 0..=n - i {
                let p = if i == 0 && j == 0 {
                    *a
                } else if i == n {
                    *b
                } else if j == n {
                    *c
                } else if j == 0 {
                    edge_point(a, b, i, n)
                } else if i == 0 {
                    edge_point(a, c, j, n)
                } else if i + j == n {
                    edge_point(b, c, j, n)
                } else {
                    let (wb, wc) = (i as f32 / n as f32, j as f32 / n as f32);
                    let wa = 1.0 - wb - wc;
                    (a.0 * wa + b.0 * wb + c.0 * wc, a.1 * wa + b.1 * wb + c.1 * wc, a.2 * wa + b.2 * wb + c.2 * wc)
                };
                let next = out.points.len() as u32;
                let id = *index.entry(corner_bits(&p)).or_insert_with(|| {
                    out.points.push(SurfacePoint { object_u: p.0.extend(p.2.x).to_array(), normal_v: p.1.extend(p.2.y).to_array() });
                    next
                });
                grid.push(id);
            }
        }
        // Row i holds n + 1 - i points, starting at i(2n + 3 - i) / 2.
        let at = |i: u32, j: u32| grid[(i * (2 * n + 3 - i) / 2 + j) as usize];
        for i in 0..n {
            for j in 0..n - i {
                out.triangles.push([at(i, j), at(i + 1, j), at(i, j + 1)]);
                if i + j + 1 < n {
                    out.triangles.push([at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
                }
            }
        }
    }
    out
}

/// Triangles of `mesh` (mesh space) at about `error` geometric error, from the pages in memory.
fn mesh_corners(mesh: &VirtualMesh, error: f32) -> Vec<[Corner; 3]> {
    let view = LodView { position: Vec3::ZERO, lod_scale: 1.0, threshold: error, orthographic: true };
    let mut triangles = Vec::new();
    for cluster in lod_cut_resident(mesh, &view, &|page| mesh.has_payload(page as usize)) {
        let c = cluster as usize;
        let vertices: Vec<Corner> = (0..mesh.clusters[c].vertex_count()).map(|i| mesh.cluster_vertex(c, i)).collect();
        triangles.extend(mesh.cluster_triangles(c).map(|t| t.map(|i| vertices[i as usize])));
    }
    triangles
}

/// The baked variant of displaced `positions` over a refinement's triangles (`refined`), with
/// `base`'s kind of distance field at its voxel size.
pub(crate) fn bake(base: &MeshSdf, refined: &[[u32; 3]], positions: &[Vec3], card_scale: f32, serial: u64) -> Option<DisplacedVariant> {
    let soup: Vec<[Vec3; 3]> = refined.iter().map(|t| t.map(|i| positions[i as usize])).collect();
    let aabb = Aabb::from_points(positions.iter().copied());
    let extent = aabb.extent();
    let sdf = match base.kind {
        SdfKind::Volume => {
            // bake_volume spreads its resolution over the extent plus a 2-voxel margin.
            let resolution = (extent.max_element() / base.voxel_size + 5.0).ceil() as u32;
            MeshSdf::bake_volume(&soup, resolution.clamp(8, 256))?
        }
        SdfKind::Heightfield => {
            let resolution = (extent.x.max(extent.z) / base.voxel_size + 1.0).ceil() as u32;
            MeshSdf::bake_heightfield(&soup, resolution.clamp(16, 4096))?
        }
    };
    if sdf.kind == SdfKind::Heightfield {
        // Heightfields get no cards.
        return Some(DisplacedVariant { sdf, aabb, triangles: Vec::new(), base_points: Vec::new(), corners: Vec::new(), cards: None, serial });
    }
    // Area-weighted normals of the welded displaced surface for the cards.
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for (t, p) in refined.iter().zip(&soup) {
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        for &i in t {
            normals[i as usize] += n;
        }
    }
    let triangles: Vec<[CardVertex; 3]> = refined
        .iter()
        .zip(&soup)
        .map(|(t, p)| {
            let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            std::array::from_fn(|k| (p[k], normals[t[k] as usize].try_normalize().unwrap_or(face)))
        })
        .collect();
    // The material is evaluated at the base points (as the resolve does for displaced pixels).
    let base_points: Vec<CornerSource> =
        refined.points.iter().map(|p| (Vec3::from_slice(&p.object_u[..3]), Vec2::new(p.object_u[3], p.normal_v[3]))).collect();
    let corners = refined.triangles.clone();
    let sources: Vec<[CornerSource; 3]> = corners.iter().map(|t| t.map(|i| base_points[i as usize])).collect();
    let texel = crate::cards::CARD_TEXEL;
    let config = crate::cards::capture_config(sdf.two_sided, card_scale, texel);
    let cards = Some((card_scale, texel, crate::cards::capture_cards(&triangles, &sources, aabb, &config)));
    Some(DisplacedVariant { sdf, aabb, triangles, base_points, corners, cards, serial })
}

/// FNV-1a.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn write(mut self, bytes: &[u8]) -> Self {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
        self
    }
}

/// What the displacement of instance material `material` reads, hashed (see [`Slot`]), and
/// whether it depends on the instance's placement.
fn displacement_content(scene: &Scene, material: u32) -> Option<(u64, bool)> {
    let m = scene.materials.get(material as usize)?;
    let source = displacement_source(scene, m)?;
    let d = m.displacement?;
    let mut hash = Fnv::new().write(format!("{d:?} {:?} {:?} {:?} {}", m.texture, m.mapping, m.uv, m.uv_scale).as_bytes());
    let placed = match source {
        DisplacementSource::Texture(_) => {
            hash = hash.write(&scene.texture_version.to_le_bytes());
            m.mapping == TextureMapping::WorldTriplanar
        }
        DisplacementSource::Graph(g) => {
            let compiled = scene.compiled_material_graph(g)?;
            let body = compiled.displacement.as_deref().unwrap_or("");
            hash = hash.write(&compiled.displacement_hash().unwrap_or(0).to_le_bytes()).write(format!("{:?}", compiled.parameters).as_bytes());
            let mut parameters: Vec<String> = scene.material_parameters(crate::MaterialHandle(material)).map(|(k, v)| format!("{k}={v:?}")).collect();
            parameters.sort();
            hash = hash.write(parameters.join(",").as_bytes());
            // Graphs speak world space: a vector, or a height from the world position or normal,
            // differs between instances of the mesh.
            compiled.displacement_kind == helios_material::DisplacementKind::Vector
                || ["input.world_position", "input.dpdx", "input.dpdy", "input.normal"].iter().any(|field| body.contains(field))
        }
    };
    Some((hash.0, placed))
}

/// World units per mesh unit (the largest axis scale of `transform`).
fn max_scale(transform: Mat4) -> f32 {
    let linear = glam::Mat3::from_mat4(transform);
    linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length()).max(1e-6)
}

/// The variants `scene`'s instances want (with tessellation on): per scene instance its request.
fn requests(scene: &Scene) -> Vec<Option<Request>> {
    let mut contents: HashMap<u32, Option<(u64, bool, bool)>> = HashMap::new();
    let mut out: Vec<Option<Request>> = scene
        .instances
        .iter()
        .enumerate()
        .map(|(index, instance)| {
            let mesh = scene.meshes.get(instance.mesh.0 as usize)?;
            let sdf = mesh.sdf.as_ref().filter(|_| instance.visible)?;
            let (content, placed, animated): (u64, bool, bool) = (*contents.entry(instance.material.0).or_insert_with(|| {
                let (content, placed) = displacement_content(scene, instance.material.0)?;
                let m = &scene.materials[instance.material.0 as usize];
                Some((content, placed, crate::tessellation::displacement_animated(scene, m)))
            }))?;
            // Within a voxel: the field interpolates between voxels as the bake does between points.
            // Cards (~10 cm texels, finer on small meshes) rasterise the same triangles: where
            // their texels are finer than the points, they see the displaced surface's facets.
            let spacing = SPACING * sdf.voxel_size;
            let placement = placed.then(|| instance.transform.to_cols_array().map(f32::to_bits));
            let slot = Slot { mesh: instance.mesh.0, material: instance.material.0, placement };
            Some(Request { slot, content, instance: index as u32, spacing, card_scale: max_scale(instance.transform), animated })
        })
        .collect();
    // Instances sharing a slot share its finest spacing and its largest scale.
    let mut shared: HashMap<Slot, (f32, f32)> = HashMap::new();
    for r in out.iter().flatten() {
        let s = shared.entry(r.slot).or_insert((f32::MAX, 0.0));
        *s = (s.0.min(r.spacing), s.1.max(r.card_scale));
    }
    for r in out.iter_mut().flatten() {
        (r.spacing, r.card_scale) = shared[&r.slot];
    }
    out
}

/// A variant on its way: refined on the CPU, displaced on the GPU (its points from `first` on in
/// the job's buffers), then baked.
struct Pending {
    refinement: Refinement,
    first: usize,
}

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

/// A GPU job: the points of some pending variants and their displaced positions' read-back.
struct Job {
    pending: Vec<Pending>,
    readback: wgpu::Buffer,
    status: Arc<AtomicU32>,
    /// Recorded into a frame but not yet mapped.
    submitted: bool,
}

/// A finished bake: the request, its variant, how far an animated surface moves from the baked
/// mean (world units, RMS at the point moving most) and the CPU time of the bake (ms).
type Baked = (Request, Option<DisplacedVariant>, f32, f32);

/// CPU work off the render thread (`lumen_offload.rs`): a [`Work`] and what to make of its
/// [`Output`]. Natively it runs on a worker thread; in the browser on a Web Worker if the page
/// installed them, else (or if the worker fails) on the render thread, run by [`finished`] one
/// task per frame.
enum Task<T> {
    #[cfg(not(target_arch = "wasm32"))]
    Thread(std::sync::mpsc::Receiver<T>),
    /// Offloaded; the work is kept to run here should the worker fail.
    #[cfg(target_arch = "wasm32")]
    Remote(lumen_offload::Offloaded, Work, Box<dyn FnOnce(Output) -> T>),
    #[cfg(target_arch = "wasm32")]
    Deferred(Work, Box<dyn FnOnce(Output) -> T>),
}

impl<T: Send + 'static> Task<T> {
    fn spawn(work: Work, finish: impl FnOnce(Output) -> T + Send + 'static) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (send, receive) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = send.send(finish(work.run()));
            });
            Task::Thread(receive)
        }
        #[cfg(target_arch = "wasm32")]
        match lumen_offload::submit(&work) {
            Some(id) => Task::Remote(id, work, Box::new(finish)),
            None => Task::Deferred(work, Box::new(finish)),
        }
    }
}

/// Removes the finished tasks from `tasks` and returns their results. A deferred task runs only
/// while `run` is set, which it then clears.
#[cfg_attr(not(target_arch = "wasm32"), allow(unused_variables))]
fn finished<T>(tasks: &mut Vec<Task<T>>, run: &mut bool) -> Vec<T> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tasks.len() {
        match &tasks[i] {
            #[cfg(not(target_arch = "wasm32"))]
            Task::Thread(receiver) => match receiver.try_recv() {
                Ok(value) => {
                    out.push(value);
                    tasks.remove(i);
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    tasks.remove(i);
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            },
            // A worker's result is copied into the page's memory: one per frame too, like a
            // deferred task's run (a bed's bake is about 9 MB).
            #[cfg(target_arch = "wasm32")]
            Task::Remote(task, ..) if *run => {
                let result = match task.poll() {
                    lumen_offload::BakePoll::Pending => None,
                    lumen_offload::BakePoll::Done(bytes) => Some(Output::decode(&bytes)),
                    lumen_offload::BakePoll::Failed => Some(None),
                };
                if let Some(output) = result {
                    let Task::Remote(_, work, finish) = tasks.remove(i) else { unreachable!() };
                    match output {
                        Some(output) => {
                            *run = false;
                            out.push(finish(output));
                        }
                        // The worker failed: the task runs here (the next pass of the loop).
                        None => tasks.insert(i, Task::Deferred(work, finish)),
                    }
                    continue;
                }
            }
            #[cfg(target_arch = "wasm32")]
            Task::Remote(..) => {}
            #[cfg(target_arch = "wasm32")]
            Task::Deferred(..) => {
                if std::mem::take(run) {
                    let Task::Deferred(work, finish) = tasks.remove(i) else { unreachable!() };
                    out.push(finish(work.run()));
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// `sdf` without its voxels: what a variant's bake reads of its mesh's field (kind, voxel size).
fn sdf_header(sdf: &MeshSdf) -> MeshSdf {
    MeshSdf { words: Vec::new(), ..*sdf }
}

/// A refined mesh surface for a request.
struct Refinement {
    request: Request,
    /// The mesh's own field without its voxels (the variant takes its kind and voxel size).
    base: MeshSdf,
    /// World units per mesh unit of the instance displacing the points.
    scale: f32,
    refined: Refined,
    /// Offloaded (a Web Worker refined it): the LOD cut, which the bake's worker refines again
    /// for the triangles instead of them crossing to the page and back (`refined` then has the
    /// points only). Refining is deterministic, so the triangles are the same.
    corners: Option<Vec<[Corner; 3]>>,
}

/// What the displacement job binds besides its own buffers.
pub struct DisplaceInputs<'a> {
    pub scene: &'a GpuScene,
    pub albedo: &'a wgpu::TextureView,
    pub data: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub graph_parameters: &'a wgpu::Buffer,
}

/// Displaced variants of the scene's meshes (see the module docs).
pub(crate) struct DisplacedSurfaces {
    layout: Option<wgpu::BindGroupLayout>,
    /// Pipelines by the hash of the displacing-graph set (`graphs::set_hash`; 0: none).
    pipelines: HashMap<u64, wgpu::ComputePipeline>,
    /// Scene versions and the tessellation setting the requests were made for.
    key: Option<(u64, u64, u64, u64, u64, u64, bool)>,
    /// Per scene instance: the variant it wants.
    requests: Vec<Option<Request>>,
    /// Baked variants by slot, with the content they were baked from (an instance keeps a
    /// variant of an older content until the new one is ready).
    ready: HashMap<Slot, (u64, Arc<DisplacedVariant>)>,
    /// Slots being refined, displaced or baked, with the content.
    in_flight: HashMap<Slot, u64>,
    /// Refinements running, refined surfaces waiting for a GPU job, the job, bakes running.
    refining: Vec<Task<Refinement>>,
    refined: Vec<Refinement>,
    job: Option<Job>,
    bakes: Vec<Task<Baked>>,
    serial: u64,
    /// The variants the last upload used (Lumen surface `meshes + k`), and per scene instance
    /// its surface.
    active: Vec<Arc<DisplacedVariant>>,
    surface_of: Vec<usize>,
    /// The [`DisplacedSurfaces::assignment`] of the last upload.
    assigned: Vec<Option<usize>>,
    /// A variant became ready since the last upload.
    changed: bool,
    /// In the browser: this frame's [`DisplacedSurfaces::sync`] already did its one piece of
    /// render-thread work (handed a job's positions to a bake, took in a worker's result or ran a
    /// task), so [`DisplacedSurfaces::record`] leaves the next GPU job to the next frame.
    busy: bool,
}

impl DisplacedSurfaces {
    pub fn new() -> Self {
        Self {
            layout: None,
            pipelines: HashMap::new(),
            key: None,
            requests: Vec::new(),
            ready: HashMap::new(),
            in_flight: HashMap::new(),
            refining: Vec::new(),
            refined: Vec::new(),
            job: None,
            bakes: Vec::new(),
            serial: 0,
            active: Vec::new(),
            surface_of: Vec::new(),
            assigned: Vec::new(),
            changed: false,
            busy: false,
        }
    }

    /// Per scene instance: the ready variant it would be assigned (by address).
    fn assignment(&self, instances: usize) -> Vec<Option<usize>> {
        let mut out: Vec<Option<usize>> = self.requests.iter().map(|r| r.and_then(|r| self.ready.get(&r.slot)).map(|(_, v)| Arc::as_ptr(v) as usize)).collect();
        out.resize(instances, None);
        out
    }

    /// Whether `request`'s variant (its slot at its content) is still wanted.
    fn wanted(&self, request: &Request) -> bool {
        self.requests.iter().flatten().any(|r| r.slot == request.slot && r.content == request.content)
    }

    /// Updates the requests when the scene or the tessellation setting changed and takes
    /// finished work. Returns whether Lumen's surfaces changed (it then re-uploads its scene):
    /// at once when the requests changed, else once every variant in flight is ready (so a
    /// scene's variants arrive in one upload).
    pub fn sync(&mut self, scene: &Scene, tessellation: bool) -> bool {
        let graphs = &scene.graphs;
        let key = (scene.geometry_version, scene.instance_version, scene.material_version, scene.texture_version, graphs.version, graphs.parameter_version, tessellation);
        let mut changed = false;
        if self.key != Some(key) {
            self.key = Some(key);
            self.requests = if tessellation { requests(scene) } else { vec![None; scene.instances.len()] };
            // With tessellation off nothing is wanted, but the variants are kept for turning it
            // back on.
            if tessellation {
                let wanted: std::collections::HashSet<Slot> = self.requests.iter().flatten().map(|r| r.slot).collect();
                self.ready.retain(|slot, _| wanted.contains(slot));
            }
            // Only a different assignment re-uploads Lumen (most material or parameter edits leave
            // the displacement alone).
            changed = self.assignment(scene.instances.len()) != self.assigned;
        }
        // In the browser one piece of render-thread work per frame (see `busy`).
        let mut run = !(self.take_job() && cfg!(target_arch = "wasm32"));
        for baked in finished(&mut self.bakes, &mut run) {
            self.accept(baked);
        }
        for refinement in finished(&mut self.refining, &mut run) {
            let request = &refinement.request;
            if self.wanted(request) {
                self.refined.push(refinement);
            } else if self.in_flight.get(&request.slot) == Some(&request.content) {
                self.in_flight.remove(&request.slot);
            }
        }
        self.busy = !run;
        // (The upload too waits for a frame without other work.)
        if changed || (self.changed && self.in_flight.is_empty() && !self.busy) {
            self.changed = false;
            return true;
        }
        false
    }

    /// Stores a finished bake if its content is still wanted.
    fn accept(&mut self, (request, variant, motion, ms): Baked) {
        if self.in_flight.get(&request.slot) == Some(&request.content) {
            self.in_flight.remove(&request.slot);
        }
        let Some(variant) = variant.filter(|_| self.wanted(&request)) else { return };
        let sdf = &variant.sdf;
        log::info!(
            "lumen: displaced variant of mesh {} with material {}: {} triangles, {:?} field {}x{}x{} ({:.1} cm voxels), baked in {ms:.0} ms{}",
            request.slot.mesh,
            request.slot.material,
            variant.triangles.len(),
            sdf.kind,
            sdf.resolution[0],
            sdf.resolution[1],
            sdf.resolution[2],
            sdf.voxel_size * 100.0,
            if request.animated { format!(", the mean surface of an animated displacement (moving up to {:.1} cm RMS from it)", motion * 100.0) } else { String::new() }
        );
        self.ready.insert(request.slot, (request.content, Arc::new(variant)));
        self.changed = true;
    }

    /// Takes a finished read-back: bakes its variants off the render thread. Returns whether there
    /// was one.
    fn take_job(&mut self) -> bool {
        let Some(job) = self.job.as_ref().filter(|j| j.status.load(Ordering::Acquire) == READY) else { return false };
        // Per variant its positions (xyz) and the largest motion (w). Read without a cast: the
        // mapped bytes need not be aligned for Vec4 (in the browser).
        let read = job.readback.get_mapped_range(..).ok().map(|view| {
            let words = |p: &Pending| view[p.first * 16..(p.first + p.refinement.refined.points.len()) * 16].chunks_exact(16).map(bytemuck::pod_read_unaligned::<Vec4>);
            job.pending.iter().map(|p| (words(p).map(Vec4::truncate).collect::<Vec<Vec3>>(), words(p).map(|w| w.w).fold(0.0, f32::max))).collect::<Vec<_>>()
        });
        job.readback.unmap();
        let job = self.job.take().unwrap();
        let Some(read) = read else {
            // Lost: the variants are requested again.
            for p in &job.pending {
                self.in_flight.remove(&p.refinement.request.slot);
            }
            return true;
        };
        for (Pending { refinement, .. }, (positions, motion)) in job.pending.into_iter().zip(read) {
            let Refinement { request, base, scale, refined, corners } = refinement;
            let motion = motion * scale;
            self.serial += 1;
            let mesh = match corners {
                Some(corners) => BakeMesh::Refine(corners, request.spacing),
                None => BakeMesh::Triangles(refined.triangles),
            };
            let work = Work::Bake { base, mesh, positions, card_scale: request.card_scale, serial: self.serial };
            self.bakes.push(Task::spawn(work, move |output| match output {
                Output::Baked(variant, ms) => (request, variant, motion, ms),
                Output::Refined(_) => (request, None, motion, 0.0),
            }));
        }
        true
    }

    /// Lumen's surfaces for an upload: the scene's meshes, then the ready variants; and per scene
    /// instance its surface (its variant once baked, else its mesh).
    pub fn assign(&mut self, scene: &Scene) {
        self.assigned = self.assignment(scene.instances.len());
        let mut index: HashMap<Slot, usize> = HashMap::new();
        self.active.clear();
        self.surface_of = scene.instances.iter().map(|i| i.mesh.0 as usize).collect();
        for (i, request) in self.requests.iter().enumerate() {
            let Some((_, variant)) = request.and_then(|r| self.ready.get(&r.slot)) else { continue };
            let slot = request.unwrap().slot;
            let k = *index.entry(slot).or_insert_with(|| {
                self.active.push(variant.clone());
                self.active.len() - 1
            });
            if let Some(s) = self.surface_of.get_mut(i) {
                *s = scene.meshes.len() + k;
            }
        }
    }

    /// The surfaces of the last [`DisplacedSurfaces::assign`].
    pub fn surfaces<'a>(&'a self, scene: &'a Scene) -> Vec<Surface<'a>> {
        scene.meshes.iter().map(Surface::Mesh).chain(self.active.iter().map(|v| Surface::Displaced(v))).collect()
    }

    /// Per scene instance its surface (see [`DisplacedSurfaces::assign`]).
    pub fn surface_of(&self) -> &[usize] {
        &self.surface_of
    }

    /// Starts refining the variants that are neither ready nor in flight, and records the GPU
    /// displacement of refined ones if no job is running. Call [`DisplacedSurfaces::after_submit`]
    /// after submitting.
    pub fn record(&mut self, rec: &mut Recorder, queue: &wgpu::Queue, scene: &Scene, inputs: &DisplaceInputs) {
        let mut seen: std::collections::HashSet<Slot> = std::collections::HashSet::new();
        for request in self.requests.iter().flatten() {
            // One task per slot: while an edit (a parameter dragged in the editor) changes the
            // content every frame, the next refinement starts once the previous one is done.
            let done = self.ready.get(&request.slot).is_some_and(|(c, _)| *c == request.content) || self.in_flight.contains_key(&request.slot);
            if done || !seen.insert(request.slot) {
                continue;
            }
            let mesh = &scene.meshes[request.slot.mesh as usize];
            let Some(base) = mesh.sdf.as_ref().map(sdf_header) else { continue };
            // The LOD cut needs the mesh; the refinement runs off the render thread.
            let corners = mesh_corners(mesh, request.spacing);
            let scale = max_scale(scene.instances[request.instance as usize].transform);
            let request = *request;
            self.in_flight.insert(request.slot, request.content);
            // Offloaded, the page keeps the (small) LOD cut and gets only the points back.
            let keep = lumen_offload::offloading().then(|| corners.clone());
            let work = Work::Refine { corners, spacing: request.spacing, triangles: keep.is_none() };
            self.refining.push(Task::spawn(work, move |output| {
                let refined = match output {
                    Output::Refined(refined) => refined,
                    Output::Baked(..) => Refined::default(),
                };
                Refinement { request, base, scale, refined, corners: keep }
            }));
        }
        if self.job.is_some() || self.refined.is_empty() || self.busy {
            return;
        }
        let mut pending: Vec<Pending> = Vec::new();
        let mut total = 0;
        let mut waiting = Vec::new();
        for refinement in std::mem::take(&mut self.refined) {
            let request = &refinement.request;
            if !self.wanted(request) {
                if self.in_flight.get(&request.slot) == Some(&request.content) {
                    self.in_flight.remove(&request.slot);
                }
                continue;
            }
            let count = refinement.refined.points.len();
            if count == 0 {
                self.in_flight.remove(&request.slot);
                continue;
            }
            if !pending.is_empty() && (total + count > MAX_JOB_POINTS || ONE_VARIANT_PER_JOB) {
                waiting.push(refinement);
                continue;
            }
            pending.push(Pending { first: total, refinement });
            total += count;
        }
        self.refined = waiting;
        let joined: Vec<SurfacePoint>;
        let points: &[SurfacePoint] = match pending.as_slice() {
            [] => return,
            [one] => &one.refinement.refined.points,
            _ => {
                joined = pending.iter().flat_map(|p| p.refinement.refined.points.iter().copied()).collect();
                &joined
            }
        };
        log::debug!("lumen: displacing {} variants ({} points)", pending.len(), points.len());
        let device = rec.device;
        let layout = self.layout.get_or_insert_with(|| {
            let ro = bind::storage(true);
            bind_group_layout(
                device,
                "lumen displace",
                wgpu::ShaderStages::COMPUTE,
                &[
                    (0, bind::uniform()),
                    (1, ro),
                    (2, bind::storage(false)),
                    (3, ro),
                    (40, bind::uniform()),
                    (graphs::BINDING_ALBEDO, bind::texture_array(true)),
                    (43, bind::texture_array(true)),
                    (44, bind::sampler()),
                    (BINDING_TESSELLATION, bind::uniform()),
                    (graphs::BINDING_GRAPH_PARAMETERS, bind::uniform()),
                ],
            )
        });
        // The dicing passes' permutation of the scene's displacing graphs (none: the base module).
        let set = graphs::displacing_graphs(scene);
        let key = if set.is_empty() { 0 } else { graphs::set_hash(&set) };
        let pipeline = self.pipelines.entry(key).or_insert_with(|| {
            let wgsl = if set.is_empty() {
                compose("lumen_displace.wgsl")
            } else {
                compose_variant("lumen_displace.wgsl", &[("displacement_graphs_off.wgsl", &graphs::scene_dispatch_module(scene, &set))])
            };
            compute_pipeline(device, "lumen displace", &create_module_from(device, "lumen displace", wgsl), "displace", layout, &[])
        });

        let point_buffer = storage_buffer_init(device, queue, "lumen displace points", bytemuck::cast_slice(points));
        let size = (points.len() * 16) as u64;
        let output = storage_buffer(device, "lumen displaced points", size, wgpu::BufferUsages::COPY_SRC);
        let mut groups = Vec::new();
        for p in &pending {
            let r = &p.refinement.request;
            let world_spacing = r.spacing * p.refinement.scale;
            // The tessellation uniform: this job's filter header, then the scene's material table.
            let tess = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lumen displace tessellation"),
                size: UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let header = TessView { dice: Vec4::new(world_spacing, FAR, FAR, 0.0), eye: Vec4::ZERO, time: Vec4::ZERO };
            queue.write_buffer(&tess, 0, bytemuck::bytes_of(&header));
            let at = std::mem::size_of::<TessView>() as u64;
            rec.encoder.copy_buffer_to_buffer(&inputs.scene.tessellation, at, &tess, at, Some(UNIFORM_SIZE - at));
            let samples = if r.animated { ANIMATED_SAMPLES } else { 1 };
            let uniform = JobUniform {
                instance: r.instance,
                first: p.first as u32,
                count: p.refinement.refined.points.len() as u32,
                samples,
                time: Vec4::new(0.0, ANIMATED_WINDOW / samples as f32, 0.0, 0.0),
            };
            let job = uniform_buffer(device, "lumen displace job", std::mem::size_of::<JobUniform>() as u64);
            queue.write_buffer(&job, 0, bytemuck::bytes_of(&uniform));
            let group = bind_group(
                device,
                "lumen displace",
                layout,
                &[
                    (0, job.as_entire_binding()),
                    (1, point_buffer.as_entire_binding()),
                    (2, output.as_entire_binding()),
                    (3, inputs.scene.instances.as_entire_binding()),
                    (40, inputs.scene.materials.as_entire_binding()),
                    (graphs::BINDING_ALBEDO, TextureView(inputs.albedo)),
                    (43, TextureView(inputs.data)),
                    (44, Sampler(inputs.sampler)),
                    (BINDING_TESSELLATION, tess.as_entire_binding()),
                    (graphs::BINDING_GRAPH_PARAMETERS, inputs.graph_parameters.as_entire_binding()),
                ],
            );
            groups.push((group, uniform.count));
        }
        {
            let mut pass = rec.compute_pass("lumen displace");
            pass.set_pipeline(pipeline);
            for (group, count) in &groups {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            }
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lumen displaced points (read-back)"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        rec.encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, Some(size));
        self.job = Some(Job { pending, readback, status: Arc::new(AtomicU32::new(IDLE)), submitted: false });
    }

    /// Maps a job recorded into the frame just submitted.
    pub fn after_submit(&mut self) {
        let Some(job) = self.job.as_mut().filter(|j| !j.submitted) else { return };
        job.submitted = true;
        job.status.store(MAPPING, Ordering::Release);
        let status = job.status.clone();
        job.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner(x: f32, z: f32) -> Corner {
        (Vec3::new(x, 0.0, z), Vec3::Y, Vec2::new(x, z))
    }

    #[test]
    fn refinement_welds_shared_edges() {
        // Two triangles of a unit quad sharing the diagonal, listed in opposite directions.
        let quad = [[corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0)], [corner(0.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)]];
        for n in [1, 2, 4, 8] {
            // The diagonal (√2) is the longest edge.
            let r = refine(&quad, 1.001 * std::f32::consts::SQRT_2 / n as f32);
            assert_eq!(r.triangles.len(), 2 * (n * n) as usize);
            // A welded (n + 1)² grid: the diagonal's points are shared.
            assert_eq!(r.points.len(), ((n + 1) * (n + 1)) as usize, "n = {n}");
            // Every edge of the refined surface is used twice inside, once on the boundary.
            let mut uses: HashMap<(u32, u32), u32> = HashMap::new();
            for t in &r.triangles {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    *uses.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
            assert_eq!(uses.values().filter(|&&u| u == 1).count(), 4 * n as usize);
            assert!(uses.values().all(|&u| u <= 2));
            // Orientation is kept: every refined triangle faces the way the quad does.
            let facing = (quad[0][1].0 - quad[0][0].0).cross(quad[0][2].0 - quad[0][0].0).y.signum();
            for t in &r.triangles {
                let p = t.map(|i| Vec3::from_slice(&r.points[i as usize].object_u[..3]));
                assert_eq!((p[1] - p[0]).cross(p[2] - p[0]).y.signum(), facing);
            }
        }
    }

    #[test]
    fn spacing_sets_the_segments_within_the_cap() {
        let triangle = [corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0)];
        // The longest edge (the diagonal, √2) in pieces of at most the spacing: 15, rounded up to
        // a power of two.
        assert_eq!(segments(&triangle, 0.1), 16);
        assert_eq!(segments(&triangle, 10.0), 1);
        let many = vec![triangle; MAX_TRIANGLES / 16];
        let r = refine(&many, 0.01);
        assert!(r.triangles.len() <= MAX_TRIANGLES && r.triangles.len() >= MAX_TRIANGLES / 4, "{}", r.triangles.len());
        // A large triangle next to small ones: refined finely where it is large only, and the
        // shared edge's coarse points weld into the fine side's.
        let mixed = [[corner(0.0, 0.0), corner(1.0, 0.0), corner(0.0, 1.0)], [corner(1.0, 0.0), corner(0.0, 0.0), corner(0.5, -0.1)]];
        let r = refine(&mixed, 0.25);
        assert_eq!(r.triangles.len(), 64 + 16);
        // 45 grid points of the large triangle, plus the small one's 15 minus the 5 shared.
        assert_eq!(r.points.len(), 45 + 15 - 5);
    }

    #[test]
    fn a_displaced_sheet_bakes_its_relief() {
        // A 1 m sheet refined and "displaced" by a bump in the middle (as the GPU would): its
        // field has the bump, which the undisplaced sheet's has not.
        let quad = [[corner(-0.5, -0.5), corner(0.5, 0.5), corner(0.5, -0.5)], [corner(-0.5, -0.5), corner(-0.5, 0.5), corner(0.5, 0.5)]];
        let flat: Vec<[Vec3; 3]> = quad.iter().map(|t| t.map(|c| c.0)).collect();
        let base = MeshSdf::bake_volume(&flat, 32).unwrap();
        let refined = refine(&quad, 0.06);
        let positions: Vec<Vec3> = refined
            .points
            .iter()
            .map(|p| {
                let q = Vec3::from_slice(&p.object_u[..3]);
                q + Vec3::Y * (0.2 * (1.0 - (q.x * q.x + q.z * q.z) / 0.09).max(0.0))
            })
            .collect();
        let variant = bake(&base, &refined.triangles, &positions, 1.0, 1).unwrap();
        assert!((variant.aabb.max.y - 0.2).abs() < 1e-3);
        assert!(variant.sdf.voxel_size <= base.voxel_size * 1.01, "{} vs {}", variant.sdf.voxel_size, base.voxel_size);
        // The top of the bump: on the displaced surface (a sheet: a thin two-sided shell), far
        // above the undisplaced one.
        let probe = Vec3::new(0.0, 0.2, 0.0);
        assert!(variant.sdf.sample(probe).abs() < 0.5 * variant.sdf.voxel_size, "{}", variant.sdf.sample(probe));
        assert!(base.sample(probe) > variant.sdf.sample(probe) + base.voxel_size, "{}", base.sample(probe));
        assert_eq!(variant.triangles.len(), refined.triangles.len());
        assert!(variant.triangles.iter().flatten().all(|v| (v.1.length() - 1.0).abs() < 1e-3));
    }
}
