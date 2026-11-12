//! Sunlight through glass: stained glass throws coloured light.
//!
//! Glass is translucent ([`MaterialKind::Glass`](crate::MaterialKind::Glass)): panes that let
//! light through do not cast an opaque shadow (the virtual shadow map lets the sun pass), so
//! without more the sun through a stained-glass window lands white. As UE5's translucent
//! shadows, the panes are drawn from the sun into a small map of their own:
//!
//! * **The map.** An orthographic view along the sunlight fitted to the world boxes of every
//!   visible glass instance: square texels, 256² to 2048² of them as the panes' extent needs
//!   (2 cm texels at best; the cathedral's 84 m of windows get 4 cm). Each texel keeps the pane nearest the sun (depth test): its
//!   transmittance in rgb — the light `shade_glass` transmits, diffused part included, lead came
//!   none — and its depth along the sunlight in alpha (`fs_glass_shadow` in translucency.wgsl).
//!   Texels without glass hold white and depth 0.
//! * **The receivers** (`glass_shadow.wgsl`): a point behind the pane of its texel gets the sun
//!   filtered by that pane: the deferred lighting (blurred with the distance behind the pane:
//!   the sun's disc and the hammered glass), the volumetric fog (coloured shafts) and Lumen's
//!   card lighting (the coloured patches bounce coloured light).
//! * **When it is drawn.** Only when its projection (the sun, the panes' boxes) or the scene
//!   changes, or a pane moves; a scene without glass has none.
//!
//! Limits: one pane per texel (light through two panes in a row gets the nearer one's colour),
//! and the resolution follows the extent of all panes: glass spread over a large world gets
//! coarse texels.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use helios_geometry::Aabb;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::ShaderStages;

use crate::profiler::Recorder;
use crate::util::{bind, bind_group, bind_group_layout, uniform_buffer, Lazy};

/// Width and height of the map in texels at most and at least.
const MAP_SIZE: u32 = 2048;
const MIN_MAP_SIZE: u32 = 256;
/// The finest texels worth having (metres): a small group of panes gets a small map.
const FINEST_TEXEL: f32 = 0.02;
/// Transmittance (rgb) and depth (a) of the nearest pane.
const MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The depth test's buffer (16 bits: ~1 mm over 60 m of panes along the light).
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
/// Space kept around the panes, metres (across the map and along the sunlight).
const MARGIN: f32 = 0.25;
/// How far behind a pane (metres) its colour applies: stone in the window's plane stays white.
const DEPTH_BIAS: f32 = 0.05;

/// `GlassShadow` in glass_shadow.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct GlassShadowUniform {
    view_proj: [[f32; 4]; 4],
    /// x: 1 while the map holds panes, y: depth bias, z: depth units per metre, w: map width
    /// in metres.
    params: [f32; 4],
}

impl GlassShadowUniform {
    /// The orthographic projection along the sunlight (`sun`: towards the sun) that holds
    /// `boxes`: x, y in clip space across a square, z 0..1 along the light. `None` without boxes
    /// or with the sun below the horizon.
    pub(crate) fn fit(sun: Vec3, boxes: impl IntoIterator<Item = Aabb>) -> Option<Self> {
        let forward = -sun.try_normalize()?;
        if forward.y >= 0.0 {
            return None;
        }
        let up = if forward.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
        let right = forward.cross(up).normalize();
        let up = right.cross(forward);
        let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
        for aabb in boxes {
            for corner in corners(&aabb) {
                let p = Vec3::new(corner.dot(right), corner.dot(up), corner.dot(forward));
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if lo.x > hi.x {
            return None;
        }
        let (lo, hi) = (lo - MARGIN, hi + MARGIN);
        let extent = (hi.x - lo.x).max(hi.y - lo.y);
        let center = (lo + hi) * 0.5;
        let s = 2.0 / extent;
        let k = 1.0 / (hi.z - lo.z);
        let rows = [
            (right * s).extend(-center.x * s),
            (up * s).extend(-center.y * s),
            (forward * k).extend(-lo.z * k),
            Vec4::W,
        ];
        let view_proj = Mat4::from_cols(rows[0], rows[1], rows[2], rows[3]).transpose();
        Some(Self { view_proj: view_proj.to_cols_array_2d(), params: [1.0, DEPTH_BIAS * k, k, extent] })
    }

    #[cfg(test)]
    fn view_proj(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.view_proj)
    }
}

/// Texels across a map `extent` metres wide: a power of two (the size changes rarely as the sun
/// moves) giving texels down to [`FINEST_TEXEL`], within the bounds.
fn map_size(extent: f32) -> u32 {
    ((extent / FINEST_TEXEL).ceil() as u32).next_power_of_two().clamp(MIN_MAP_SIZE, MAP_SIZE)
}

fn corners(aabb: &Aabb) -> [Vec3; 8] {
    std::array::from_fn(|i| Vec3::select(glam::BVec3::new(i & 1 != 0, i & 2 != 0, i & 4 != 0), aabb.max, aabb.min))
}

/// A pane's draw: its decoded mesh range and scene instance (`first_instance`).
pub(crate) struct PaneDraw {
    pub first_index: u32,
    pub index_count: u32,
    pub base_vertex: i32,
    pub instance: u32,
}

/// The inputs `fs_glass_shadow` evaluates the panes' materials with.
pub(crate) struct PaneInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub instances: &'a wgpu::Buffer,
    pub materials: &'a wgpu::Buffer,
    pub textures: &'a crate::textures::TextureLibrary,
    pub samplers: &'a crate::passes::Samplers,
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
}

struct Map {
    size: u32,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
}

pub struct GlassShadow {
    /// [`GlassShadowUniform`] (params.x 0: no map).
    pub(crate) uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline: Lazy<wgpu::RenderPipeline>,
    /// 1×1 white, bound while there is no map.
    placeholder: wgpu::TextureView,
    map: Option<Map>,
    /// This frame's projection; `None`: no map this frame.
    current: Option<GlassShadowUniform>,
    /// What the map holds: its projection and the scene versions it was drawn for.
    drawn: Option<(GlassShadowUniform, (u64, u64, u64))>,
}

impl GlassShadow {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, module: &wgpu::ShaderModule) -> Self {
        let layout = bind_group_layout(
            device,
            "glass shadow",
            ShaderStages::VERTEX_FRAGMENT,
            &[
                (0, bind::uniform()),
                (1, bind::storage(true)),
                (2, bind::uniform()),
                (28, bind::texture_array(true)),
                (29, bind::texture_array(true)),
                (30, bind::sampler()),
                (90, bind::uniform()),
            ],
        );
        let pipeline = {
            let (device, module, layout) = (device.clone(), module.clone(), layout.clone());
            Lazy::new(move || pipeline(&device, &module, &layout))
        };
        let placeholder = {
            use wgpu::util::DeviceExt;
            // 1.0 as f16 in rgb, depth 0.
            let white: [u16; 4] = [0x3c00, 0x3c00, 0x3c00, 0];
            let texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("glass shadow placeholder"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: MAP_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&white),
            );
            texture.create_view(&Default::default())
        };
        Self {
            uniform: uniform_buffer(device, "glass shadow", std::mem::size_of::<GlassShadowUniform>() as u64),
            layout,
            pipeline,
            placeholder,
            map: None,
            current: None,
            drawn: None,
        }
    }

    /// Queues the pipeline for the browser's background compile (the scene has glass).
    pub(crate) fn prefetch(&self) {
        self.pipeline.prefetch();
    }

    /// Fits this frame's map to the panes' `boxes` (none, or `enabled` false: no map).
    /// `moved`: a pane moved since the last frame, so the map is redrawn.
    pub(crate) fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sun: Vec3, boxes: impl IntoIterator<Item = Aabb>, enabled: bool, moved: bool) {
        let current = if enabled { GlassShadowUniform::fit(sun, boxes) } else { None };
        if moved {
            self.drawn = None;
        }
        if current != self.current {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&current.unwrap_or_default()));
            self.current = current;
        }
        let Some(u) = current else { return };
        let size = map_size(u.params[3]);
        if self.map.as_ref().is_none_or(|m| m.size != size) {
            let mib = size as f32 * size as f32 * 10.0 / (1 << 20) as f32;
            log::info!("glass shadow: {size}² map ({mib:.1} MiB) over {:.0} m of glass, {:.1} cm texels", u.params[3], u.params[3] / size as f32 * 100.0);
            self.drawn = None;
            let target = |label, format| {
                let texture = crate::util::texture(
                    device,
                    crate::util::TextureSpec {
                        label,
                        width: size,
                        height: size,
                        layers: 1,
                        mips: 1,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    },
                );
                texture.create_view(&Default::default())
            };
            self.map = Some(Map { size, color: target("glass shadow map", MAP_FORMAT), depth: target("glass shadow depth", DEPTH_FORMAT) });
        }
    }

    /// Whether the map is used this frame.
    pub fn active(&self) -> bool {
        self.current.is_some()
    }

    /// The map the receivers read (binding 91; a white texel without one).
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        match (&self.map, self.current) {
            (Some(map), Some(_)) => &map.color,
            _ => &self.placeholder,
        }
    }

    /// Draws `panes` into the map unless it already holds them for this projection and
    /// `versions` (geometry, material, instance) of the scene.
    pub(crate) fn record(&mut self, rec: &mut Recorder, inputs: &PaneInputs, panes: &[PaneDraw], versions: (u64, u64, u64)) {
        let (Some(current), Some(map)) = (self.current, &self.map) else { return };
        if self.drawn == Some((current, versions)) {
            return;
        }
        let group = bind_group(
            rec.device,
            "glass shadow",
            &self.layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, inputs.instances.as_entire_binding()),
                (2, inputs.materials.as_entire_binding()),
                (28, TextureView(&inputs.textures.albedo)),
                (29, TextureView(&inputs.textures.data)),
                (30, Sampler(&inputs.samplers.material)),
                (90, self.uniform.as_entire_binding()),
            ],
        );
        let mut pass = rec.render_pass(
            "glass shadow",
            &[Some(wgpu::RenderPassColorAttachment {
                view: &map.color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 0.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &map.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_vertex_buffer(0, inputs.vertices.slice(..));
        pass.set_index_buffer(inputs.indices.slice(..), wgpu::IndexFormat::Uint32);
        for p in panes {
            pass.draw_indexed(p.first_index..p.first_index + p.index_count, p.base_vertex, p.instance..p.instance + 1);
        }
        drop(pass);
        self.drawn = Some((current, versions));
    }
}

fn pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let pipeline_layout = crate::util::pipeline_layout(device, "glass shadow", &[layout]);
    crate::util::render_pipeline(device, &wgpu::RenderPipelineDescriptor {
        label: Some("glass shadow"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_glass_shadow"),
            compilation_options: Default::default(),
            buffers: &[Some(crate::translucency::mesh_buffer())],
        },
        // Panes face either way relative to the sun.
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_glass_shadow"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format: MAP_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(u: &GlassShadowUniform, p: Vec3) -> Vec3 {
        let c = u.view_proj() * p.extend(1.0);
        Vec3::new(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5, c.z)
    }

    #[test]
    fn the_map_holds_every_pane_inside_its_depth_range() {
        let sun = Vec3::new(0.4, 0.7, 0.3).normalize();
        let panes = [Aabb { min: Vec3::new(-10.0, 2.0, -1.0), max: Vec3::new(-9.0, 6.0, 1.0) }, Aabb { min: Vec3::new(30.0, 8.0, 4.0), max: Vec3::new(31.0, 12.0, 5.0) }];
        let u = GlassShadowUniform::fit(sun, panes).unwrap();
        for p in panes.iter().flat_map(corners) {
            let m = project(&u, p);
            assert!(m.cmpgt(Vec3::ZERO).all() && m.cmplt(Vec3::ONE).all(), "{p} -> {m}");
        }
        // Square texels: the two axes span the same metres.
        assert!(u.params[3] > 0.0);
    }

    #[test]
    fn depth_grows_along_the_sunlight() {
        let sun = Vec3::new(-0.3, 0.8, 0.5).normalize();
        let pane = Aabb { min: Vec3::new(-1.0, 9.0, -1.0), max: Vec3::new(1.0, 11.0, 1.0) };
        let u = GlassShadowUniform::fit(sun, [pane]).unwrap();
        let at_pane = project(&u, Vec3::new(0.0, 10.0, 0.0));
        // The floor the light reaches through the pane: same map texel, deeper.
        let floor = Vec3::new(0.0, 10.0, 0.0) - sun * (10.0 / sun.y);
        let on_floor = project(&u, floor);
        assert!(at_pane.truncate().abs_diff_eq(on_floor.truncate(), 1e-4));
        assert!(on_floor.z > at_pane.z + u.params[1]);
        // A metre further along the light is params.z further in depth.
        let deeper = project(&u, Vec3::new(0.0, 10.0, 0.0) - sun);
        assert!((deeper.z - at_pane.z - u.params[2]).abs() < 1e-4);
        // Towards the sun: in front of the pane.
        assert!(project(&u, Vec3::new(0.0, 10.0, 0.0) + sun * 3.0).z < at_pane.z);
    }

    #[test]
    fn small_groups_of_panes_get_small_maps() {
        assert_eq!(map_size(84.0), MAP_SIZE);
        assert_eq!(map_size(5.0), MIN_MAP_SIZE);
        assert_eq!(map_size(15.0), 1024);
    }

    #[test]
    fn no_map_without_panes_or_sun() {
        let pane = Aabb { min: Vec3::ZERO, max: Vec3::ONE };
        assert!(GlassShadowUniform::fit(Vec3::Y, std::iter::empty()).is_none());
        assert!(GlassShadowUniform::fit(Vec3::new(0.0, -0.2, 1.0), [pane]).is_none());
        // Straight overhead: still a valid basis.
        assert!(GlassShadowUniform::fit(Vec3::Y, [pane]).is_some());
    }
}
