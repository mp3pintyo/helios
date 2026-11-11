//! Local lights through glass: a candle behind a stained-glass screen throws coloured light.
//!
//! The sun's glass map (`glass_shadow.rs`) colours sunlight only; local lights shone through
//! glass white. The same idea, per light: for the few local lights nearest the camera whose range
//! reaches a glass pane ([`MAX_GLASS_LIGHTS`]), the panes within that range are drawn from the
//! light into a cube map of its own (`fs_glass_light` in translucency.wgsl) — per texel the
//! transmittance of the pane nearest the light and its distance from it. A receiver farther from
//! the light than its texel's pane gets the light filtered by the pane (`glass_light.wgsl`, one
//! tap): the deferred lighting's clustered loop and MegaLights' shading call it per light.
//!
//! **Why cube maps of their own**, not a colour layer in the lights' virtual shadow maps: those
//! are paged, cached and filtered for depth only, and drawn by the GPU-driven cluster passes; glass
//! is a few hundred triangles drawn conventionally. A map per light costs memory (8 layers ×
//! 6 × 256² × 8 B = 25 MiB at most) but nothing per page, and a light's map is redrawn only when
//! it moves or the panes change (the swinging chandeliers: every frame, ~6 small passes each).
//! A world-space transmittance volume would be shared by all lights but can't tell which side of
//! a pane a light is on, nor resolve a 4 cm lead came. Lights beyond the eight, and the
//! volumetric fog, particles and Lumen's surface cache, see glass as clear for local lights.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use helios_geometry::Aabb;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{BindingType, ShaderStages};

use crate::glass_shadow::{PaneDraw, PaneInputs};
use crate::profiler::Recorder;
use crate::util::{bind, uniform_buffer, Lazy};

/// Lights with a glass map at most.
pub(crate) const MAX_GLASS_LIGHTS: usize = 8;
/// Texels per cube face side.
const FACE_SIZE: u32 = 256;
const MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Near plane of the faces (metres).
const NEAR: f32 = 0.02;
/// How far behind a pane (metres) its colour applies.
const BIAS: f32 = 0.05;
/// Lights the renderer uploads at most (`lights.rs`); the uniform maps each to a layer.
const MAX_LIGHTS: usize = crate::lights::MAX_LIGHTS;
/// Uniform offset alignment of the per-face uniforms.
const FACE_STRIDE: u64 = 256;

/// `GlassLights` in glass_light.wgsl.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
struct GlassLightsUniform {
    layers: [[u32; 4]; MAX_LIGHTS / 16],
    params: [f32; 4],
}

/// `GlassLightFace` in translucency.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FaceUniform {
    view_proj: [[f32; 4]; 4],
    light: [f32; 4],
}

/// The cube face directions and up vectors (WebGPU's cube map face order and orientation, with
/// left-handed views).
const FACES: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    (Vec3::Y, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::Z),
    (Vec3::Z, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y),
];

/// The view-projection of face `face` of a cube map around `position` reaching `range`.
fn face_view_proj(position: Vec3, range: f32, face: usize) -> Mat4 {
    let (f, up) = FACES[face];
    // Left-handed look-at: x = up × forward, y = forward × x, z = forward.
    let s = up.cross(f);
    let u = f.cross(s);
    let view = Mat4::from_cols(s.extend(-s.dot(position)), u.extend(-u.dot(position)), f.extend(-f.dot(position)), glam::Vec4::W).transpose();
    // Left-handed perspective, 90°, depth 0..1 from NEAR to `range`.
    let far = range.max(NEAR * 2.0);
    let r = far / (far - NEAR);
    let projection = Mat4::from_cols(glam::Vec4::X, glam::Vec4::Y, glam::Vec4::new(0.0, 0.0, r, -r * NEAR), glam::Vec4::Z).transpose();
    projection * view
}

/// What a layer holds: the light (index, position, range) and the scene versions.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LayerKey {
    light: u32,
    position: Vec3,
    range: f32,
    versions: (u64, u64, u64),
}

struct Maps {
    array: wgpu::TextureView,
    faces: Vec<wgpu::TextureView>,
    depth: wgpu::TextureView,
}

pub(crate) struct GlassLights {
    uniform: wgpu::Buffer,
    faces: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline: Lazy<wgpu::RenderPipeline>,
    placeholder: wgpu::TextureView,
    sampler: wgpu::Sampler,
    maps: Option<Maps>,
    /// Frames without a light through glass (the maps are released after a while).
    idle: crate::gpu_memory::Idle,
    /// Per layer: the light it is for and the panes (indices into the caller's list) to draw.
    assigned: Vec<Option<(LayerKey, Vec<usize>)>>,
    /// Per layer: what it holds.
    drawn: Vec<Option<LayerKey>>,
    written: Option<GlassLightsUniform>,
}

impl GlassLights {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, module: &wgpu::ShaderModule) -> Self {
        let fragment = ShaderStages::FRAGMENT;
        let both = ShaderStages::VERTEX_FRAGMENT;
        let face_uniform = BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<FaceUniform>() as u64),
        };
        let entries = [
            (0, both, bind::uniform()),
            (1, both, bind::storage(true)),
            (2, fragment, bind::uniform()),
            (28, fragment, bind::texture_array(true)),
            (29, fragment, bind::texture_array(true)),
            (30, fragment, bind::sampler()),
            (95, both, face_uniform),
        ];
        let entries: Vec<wgpu::BindGroupLayoutEntry> =
            entries.iter().map(|&(binding, visibility, ty)| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None }).collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("glass light"), entries: &entries });
        let pipeline = {
            let (device, module, layout) = (device.clone(), module.clone(), layout.clone());
            Lazy::new(move || pipeline(&device, &module, &layout))
        };
        let placeholder = {
            use wgpu::util::DeviceExt;
            // 1.0 as f16 in rgb, distance 0, six faces.
            let white: [u16; 24] = std::array::from_fn(|i| if i % 4 == 3 { 0 } else { 0x3c00 });
            let texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("glass light placeholder"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
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
            texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::CubeArray), ..Default::default() })
        };
        Self {
            uniform: uniform_buffer(device, "glass lights", std::mem::size_of::<GlassLightsUniform>() as u64),
            faces: uniform_buffer(device, "glass light faces", FACE_STRIDE * 6 * MAX_GLASS_LIGHTS as u64),
            layout,
            pipeline,
            placeholder,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("glass light"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            maps: None,
            idle: Default::default(),
            assigned: vec![None; MAX_GLASS_LIGHTS],
            drawn: vec![None; MAX_GLASS_LIGHTS],
            written: None,
        }
    }

    pub(crate) fn prefetch(&self) {
        self.pipeline.prefetch();
    }

    /// Picks the lights nearest `eye` whose range reaches one of `panes` (world boxes), keeping
    /// a light's layer while it stays picked, and uploads the light → layer table. `enabled`
    /// false (or no panes): no light is filtered. `moved`: a pane moved (every map is redrawn).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lights: &[crate::scene::Light], panes: &[Aabb], eye: Vec3, versions: (u64, u64, u64), enabled: bool, moved: bool) {
        if moved {
            self.drawn.iter_mut().for_each(|d| *d = None);
        }
        let mut picked: Vec<(usize, f32, Vec<usize>)> = Vec::new();
        if enabled && !panes.is_empty() {
            for (index, light) in lights.iter().enumerate().take(MAX_LIGHTS) {
                if light.range <= 0.0 || light.intensity <= 0.0 {
                    continue;
                }
                let reached: Vec<usize> = panes.iter().enumerate().filter(|(_, b)| sphere_meets_box(light.position, light.range, b)).map(|(i, _)| i).collect();
                if !reached.is_empty() {
                    picked.push((index, light.position.distance_squared(eye), reached));
                }
            }
            picked.sort_by(|a, b| a.1.total_cmp(&b.1));
            picked.truncate(MAX_GLASS_LIGHTS);
        }
        // Lights keep their layers; new ones take the free layers.
        let mut layers: Vec<Option<(LayerKey, Vec<usize>)>> = vec![None; MAX_GLASS_LIGHTS];
        let mut new = Vec::new();
        for (index, _, reached) in picked {
            let light = &lights[index];
            let key = LayerKey { light: index as u32, position: light.position, range: light.range, versions };
            match self.assigned.iter().position(|a| a.as_ref().is_some_and(|(k, _)| k.light == key.light)) {
                Some(layer) if layers[layer].is_none() => layers[layer] = Some((key, reached)),
                _ => new.push((key, reached)),
            }
        }
        for entry in new {
            if let Some(free) = layers.iter().position(Option::is_none) {
                layers[free] = Some(entry);
            }
        }
        if layers.iter().map(|l| l.as_ref().map(|(k, _)| k.light)).ne(self.assigned.iter().map(|l| l.as_ref().map(|(k, _)| k.light))) {
            let picked: Vec<String> = layers.iter().flatten().map(|(k, p)| format!("{} at {:.1} ({} panes)", k.light, k.position, p.len())).collect();
            log::debug!("glass lights: {}", picked.join(", "));
        }
        self.assigned = layers;
        let mut uniform = GlassLightsUniform { layers: [[0; 4]; MAX_LIGHTS / 16], params: [0.0, BIAS, 0.0, 0.0] };
        for (layer, (key, _)) in self.assigned.iter().enumerate().filter_map(|(l, a)| a.as_ref().map(|a| (l, a))) {
            let i = key.light as usize;
            uniform.layers[i / 16][(i / 4) % 4] |= (layer as u32 + 1) << (8 * (i % 4));
            uniform.params[0] = 1.0;
        }
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
        if self.idle.tick(uniform.params[0] != 0.0) {
            // No light has reached a pane for a while (e.g. the scene has no glass any more).
            if let Some(maps) = self.maps.take() {
                crate::gpu_memory::retire_view(&maps.array);
                crate::gpu_memory::retire_view(&maps.depth);
                self.drawn.iter_mut().for_each(|d| *d = None);
            }
        }
        if uniform.params[0] == 0.0 {
            return;
        }
        if self.maps.is_none() {
            self.drawn.iter_mut().for_each(|d| *d = None);
            self.maps = Some(Maps::new(device));
        }
        for (layer, entry) in self.assigned.iter().enumerate() {
            let Some((key, _)) = entry else { continue };
            if self.drawn[layer] == Some(*key) {
                continue;
            }
            for face in 0..6 {
                let f = FaceUniform { view_proj: face_view_proj(key.position, key.range, face).to_cols_array_2d(), light: key.position.extend(key.range).to_array() };
                queue.write_buffer(&self.faces, (layer * 6 + face) as u64 * FACE_STRIDE, bytemuck::bytes_of(&f));
            }
        }
    }

    /// Whether any light is filtered this frame.
    pub(crate) fn active(&self) -> bool {
        self.written.is_some_and(|u| u.params[0] != 0.0)
    }

    /// Draws the maps of the lights that moved or are new. `panes`: the list `update` saw.
    pub(crate) fn record(&mut self, rec: &mut Recorder, inputs: &PaneInputs, panes: &[PaneDraw]) {
        let Some(maps) = &self.maps else { return };
        if !self.active() {
            return;
        }
        let stale: Vec<usize> = (0..MAX_GLASS_LIGHTS).filter(|&l| self.assigned[l].as_ref().is_some_and(|(k, _)| self.drawn[l] != Some(*k))).collect();
        if stale.is_empty() {
            return;
        }
        let group = rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glass light"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: inputs.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inputs.instances.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: inputs.materials.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 28, resource: TextureView(&inputs.textures.albedo) },
                wgpu::BindGroupEntry { binding: 29, resource: TextureView(&inputs.textures.data) },
                wgpu::BindGroupEntry { binding: 30, resource: Sampler(&inputs.samplers.material) },
                wgpu::BindGroupEntry {
                    binding: 95,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.faces,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<FaceUniform>() as u64),
                    }),
                },
            ],
        });
        for layer in stale {
            let Some((key, reached)) = &self.assigned[layer] else { continue };
            for face in 0..6 {
                let mut pass = rec.render_pass(
                    "glass light",
                    &[Some(wgpu::RenderPassColorAttachment {
                        view: &maps.faces[layer * 6 + face],
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 0.0 }), store: wgpu::StoreOp::Store },
                    })],
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &maps.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                        stencil_ops: None,
                    }),
                );
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &group, &[((layer * 6 + face) as u64 * FACE_STRIDE) as u32]);
                pass.set_vertex_buffer(0, inputs.vertices.slice(..));
                pass.set_index_buffer(inputs.indices.slice(..), wgpu::IndexFormat::Uint32);
                for p in reached.iter().filter_map(|&i| panes.get(i)) {
                    pass.draw_indexed(p.first_index..p.first_index + p.index_count, p.base_vertex, p.instance..p.instance + 1);
                }
            }
            self.drawn[layer] = Some(*key);
        }
    }

    /// Layout entries of the receivers (`glass_light.wgsl`).
    pub(crate) fn entries() -> [(u32, BindingType); 3] {
        [(92, bind::uniform()), (93, BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::CubeArray, multisampled: false }), (94, bind::sampler())]
    }

    pub(crate) fn resources(&self) -> [(u32, wgpu::BindingResource<'_>); 3] {
        let maps = self.maps.as_ref().filter(|_| self.active()).map_or(&self.placeholder, |m| &m.array);
        [(92, self.uniform.as_entire_binding()), (93, TextureView(maps)), (94, Sampler(&self.sampler))]
    }
}

impl Maps {
    fn new(device: &wgpu::Device) -> Self {
        let layers = 6 * MAX_GLASS_LIGHTS as u32;
        let texture = crate::util::texture(
            device,
            crate::util::TextureSpec {
                label: "glass light maps",
                width: FACE_SIZE,
                height: FACE_SIZE,
                layers,
                mips: 1,
                format: MAP_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            },
        );
        let faces = (0..layers)
            .map(|l| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: l,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let depth = crate::util::texture(
            device,
            crate::util::TextureSpec { label: "glass light depth", width: FACE_SIZE, height: FACE_SIZE, layers: 1, mips: 1, format: DEPTH_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT },
        );
        log::info!("glass lights: {MAX_GLASS_LIGHTS} cube maps of {FACE_SIZE}² ({:.1} MiB)", (layers as u64 * (FACE_SIZE * FACE_SIZE) as u64 * 8) as f64 / (1 << 20) as f64);
        Self {
            array: texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::CubeArray), ..Default::default() }),
            faces,
            depth: depth.create_view(&Default::default()),
        }
    }
}

fn sphere_meets_box(center: Vec3, radius: f32, b: &Aabb) -> bool {
    center.clamp(b.min, b.max).distance_squared(center) <= radius * radius
}

fn pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let pipeline_layout = crate::util::pipeline_layout(device, "glass light", &[layout]);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("glass light"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_glass_light"),
            compilation_options: Default::default(),
            buffers: &[Some(crate::translucency::mesh_buffer())],
        },
        // Panes face either way relative to the light.
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
            entry_point: Some("fs_glass_light"),
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

    #[test]
    fn uniform_matches_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GlassLightsUniform>(), 64 * 16 + 16);
        assert!(std::mem::size_of::<FaceUniform>() as u64 <= FACE_STRIDE);
    }

    /// A point straight along each face's direction lands in the middle of that face, and the
    /// face's right and up match WebGPU's cube map lookup (u along the major axis' s, v along -t).
    #[test]
    fn faces_project_like_cube_map_lookups() {
        let light = Vec3::new(1.0, 2.0, 3.0);
        for (face, (forward, _)) in FACES.iter().enumerate() {
            let m = face_view_proj(light, 10.0, face);
            let centre = m.project_point3(light + *forward * 2.0);
            assert!(centre.truncate().length() < 1e-4, "face {face}: {centre}");
            assert!(centre.z > 0.0 && centre.z < 1.0);
        }
        // +X face: a direction towards -Z lands on the right (u grows with -z), +Y at the top.
        let m = face_view_proj(Vec3::ZERO, 10.0, 0);
        let p = m.project_point3(Vec3::new(1.0, 0.3, -0.4));
        assert!(p.x > 0.0 && p.y > 0.0, "{p}");
        // +Y face: +X right, -Z up.
        let m = face_view_proj(Vec3::ZERO, 10.0, 2);
        let p = m.project_point3(Vec3::new(0.3, 1.0, -0.4));
        assert!(p.x > 0.0 && p.y > 0.0, "{p}");
    }

    #[test]
    fn lights_reach_panes_within_their_range() {
        let pane = Aabb { min: Vec3::new(4.0, 0.0, -1.0), max: Vec3::new(4.1, 3.0, 1.0) };
        assert!(sphere_meets_box(Vec3::new(0.0, 1.0, 0.0), 4.5, &pane));
        assert!(!sphere_meets_box(Vec3::new(0.0, 1.0, 0.0), 3.5, &pane));
    }
}
