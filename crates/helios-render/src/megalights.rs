//! MegaLights: stochastic direct lighting for many shadowed local lights (UE 5.5's MegaLights).
//!
//! The classic deferred path loops over every light of a pixel's froxel and can afford shadows
//! for few of them: the 32 most influential lights get virtual shadow maps, four more per pixel
//! distance-field traces, every other light leaks through walls. MegaLights replaces that loop
//! with a fixed budget per pixel, whatever the number of lights:
//!
//! 1. **Sample** (`megalights_sample.wgsl`, `generate`): per 2×2 block, at a representative pixel
//!    that moves every frame, four lights are drawn from the froxel's light list in proportion to
//!    their unshadowed contribution through the shared BRDF (weighted resampling over at most 32
//!    candidates; longer lists are thinned to an unbiased stratified subset). Lights whose rays
//!    were all blocked in the same screen tile last frame get a tenth of their weight (visibility
//!    guiding), so the rays go where light arrives.
//!    A light picked more than once where last frame's rays to it in the tile all agreed (fully
//!    lit or fully shadowed, no penumbra) traces one ray for all its picks (UE 5.8); the rays
//!    left are compacted into a list that sizes the trace's indirect dispatch.
//! 2. **Trace** (`trace`): one shadow ray per listed sample, to a random point of the light's
//!    disk (soft shadows from the source radius): a short screen trace over the closest-depth
//!    HZB (`hzb_trace.wgsl`) for contact shadows, and over the whole ray the light's virtual shadow map where it has one (shadow-map
//!    ray marching), otherwise Lumen's distance fields (mesh SDFs within 2 m of either end, the
//!    global field between). Every light casts shadows; the 32-light shadow-map budget only
//!    decides which method.
//! 3. **Shade** (`megalights_shade.wgsl`): every pixel evaluates the BRDF of its block's four
//!    samples and writes its direct light *demodulated* — diffuse without the diffuse colour,
//!    specular divided by the environment BRDF — so denoising never blurs texture detail.
//! 4. **Denoise** (`megalights_denoise.wgsl`): temporal accumulation with reprojection,
//!    disocclusion and variance clipping, then four à-trous iterations guided by the variance
//!    (SVGF), on diffuse and specular separately.
//!
//! The last iteration modulates the result back (`diffuse colour × diffuse + environment BRDF ×
//! specular`), which the lighting pass adds instead of running its light loop.
//!
//! Cost: one ray and four BRDF evaluations per pixel for tracing and shading, at most 32 per 2×2
//! block for sampling, plus the denoiser. Only the share of rays that need the distance fields
//! (lights without a shadow map) grows with the light count. The textures (56 bytes per internal
//! pixel) and buffers exist only while the mode is on.

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat as F, TextureUsages as U};

use crate::lights::LightCulling;
use crate::lumen::Lumen;
use crate::passes::hzb::{ClosestHzb, Hzb};
use crate::passes::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, lazy_compute_pipeline_with, Lazy, storage_buffer, texture, workgroups, TextureSpec};
use crate::vsm::VirtualShadowMap;

const C: ShaderStages = ShaderStages::COMPUTE;
/// Words per 2×2 block in the sample buffer (`ML_BLOCK_WORDS` in megalights_common.wgsl).
const BLOCK_WORDS: u64 = 9;
/// Pixels per side of a guiding tile and words per tile (`ML_GUIDE_TILE`, `ML_GUIDE_WORDS`).
const GUIDE_TILE: u32 = 8;
const GUIDE_WORDS: u64 = 24;
/// Strides of the à-trous iterations (3×3 taps each: a 31-pixel footprint in all).
const SPATIAL_STRIDES: [u32; 4] = [1, 2, 4, 8];

/// What MegaLights reads of the frame.
pub struct MegaLightsInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub targets: &'a FrameTargets,
    pub lights: &'a LightCulling,
    pub vsm: &'a VirtualShadowMap,
    pub lumen: &'a Lumen,
    pub samplers: &'a Samplers,
    /// Builds the closest-depth HZB of the screen traces.
    pub hzb: &'a Hzb,
    /// Shared rays outside penumbrae and HZB screen traces ([`crate::RenderSettings`]).
    pub shared_rays: bool,
    pub hzb_traces: bool,
    /// Bind group 1 of the shading's slab variant while the scene has slab materials.
    pub substrate: Option<&'a wgpu::BindGroup>,
    /// Local lights through glass (`glass_light.rs`).
    pub glass_lights: &'a crate::glass_light::GlassLights,
}

/// Textures and buffers at the internal resolution's capacity.
struct Targets {
    capacity: (u32, u32),
    /// This frame's shaded samples (diffuse, specular); reused by the spatial iterations.
    raw: [wgpu::TextureView; 2],
    /// Per frame parity: accumulated diffuse, specular and info (frames, depth). The previous
    /// frame's set is free once the temporal pass has read it and holds spatial iterations.
    history: [[wgpu::TextureView; 3]; 2],
    samples: wgpu::Buffer,
    /// Visibility guiding masks, two halves of `guide_half` bytes: this frame's (frame parity,
    /// written by the trace) and last frame's.
    guide: wgpu::Buffer,
    guide_half: u64,
    /// The rays to trace (count, then block · 4 + sample) and the trace's indirect arguments.
    rays: wgpu::Buffer,
    ray_args: wgpu::Buffer,
    /// Closest-depth pyramid of the final depth buffer, for the screen traces.
    hzb: ClosestHzb,
}

impl Targets {
    fn new(device: &wgpu::Device, capacity: (u32, u32)) -> Self {
        let (w, h) = capacity;
        let usage = U::STORAGE_BINDING | U::TEXTURE_BINDING;
        let make = |label: &str, format| texture(device, TextureSpec { label, width: w, height: h, layers: 1, mips: 1, format, usage }).create_view(&Default::default());
        let blocks = w.div_ceil(2) as u64 * h.div_ceil(2) as u64;
        let tiles = w.div_ceil(GUIDE_TILE) as u64 * h.div_ceil(GUIDE_TILE) as u64;
        Self {
            capacity,
            raw: [make("megalights diffuse", F::Rgba16Float), make("megalights specular", F::Rgba16Float)],
            history: [0, 1].map(|i| {
                [
                    make(&format!("megalights history diffuse {i}"), F::Rgba16Float),
                    make(&format!("megalights history specular {i}"), F::Rgba16Float),
                    make(&format!("megalights history info {i}"), F::R32Uint),
                ]
            }),
            samples: storage_buffer(device, "megalights samples", blocks * BLOCK_WORDS * 4, wgpu::BufferUsages::empty()),
            guide: storage_buffer(device, "megalights guide", 2 * tiles * GUIDE_WORDS * 4, wgpu::BufferUsages::empty()),
            guide_half: tiles * GUIDE_WORDS * 4,
            rays: storage_buffer(device, "megalights rays", (blocks * 4 + 1) * 4, wgpu::BufferUsages::empty()),
            ray_args: storage_buffer(device, "megalights ray args", 16, wgpu::BufferUsages::INDIRECT),
            hzb: ClosestHzb::new(device, capacity),
        }
    }
}

/// The pipelines are compiled when MegaLights first runs ([`Lazy`]): it is off by default, and a
/// browser compiles every pipeline the renderer creates before its first frame.
pub struct MegaLights {
    sample_layout: wgpu::BindGroupLayout,
    /// Group 1 of generate and ray_args: the trace's indirect arguments.
    args_layout: wgpu::BindGroupLayout,
    /// Per permutation: [one ray per sample, shared rays outside penumbrae].
    generate: [Lazy<wgpu::ComputePipeline>; 2],
    ray_args: Lazy<wgpu::ComputePipeline>,
    /// Per permutation: [linear screen traces, HZB screen traces].
    trace: [Lazy<wgpu::ComputePipeline>; 2],
    shade_layout: wgpu::BindGroupLayout,
    shade: Lazy<wgpu::ComputePipeline>,
    /// Shading with Substrate slabs (`substrate_gpu.rs`), compiled when first needed.
    shade_substrate: Option<Lazy<wgpu::ComputePipeline>>,
    debug_layout: wgpu::BindGroupLayout,
    debug: Lazy<wgpu::ComputePipeline>,
    temporal_layout: wgpu::BindGroupLayout,
    temporal: Lazy<wgpu::ComputePipeline>,
    spatial_layout: wgpu::BindGroupLayout,
    spatial: Lazy<wgpu::ComputePipeline>,
    /// Per spatial iteration: (stride, first iteration).
    spatial_params: Vec<wgpu::Buffer>,
    targets: Option<Targets>,
}

impl MegaLights {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let tex = bind::texture(false);
        let sample_module = create_module(device, "megalights_sample.wgsl");
        let sample_layout = bind_group_layout(
            device,
            "megalights sample",
            C,
            &[
                (0, bind::uniform()),
                (1, tex),
                (2, tex),
                (3, tex),
                (4, tex),
                (5, bind::storage(true)),
                (6, bind::storage(true)),
                (7, bind::storage(false)),
                (9, bind::storage(false)),
                (10, bind::uniform()),
                (11, bind::texture_3d(true)),
                (12, bind::sampler()),
                (13, bind::storage(true)),
                (14, bind::texture_3d(true)),
                (15, bind::uniform()),
                (16, bind::storage(true)),
                (17, bind::texture_depth()),
                (18, bind::uniform()),
                (22, tex),
                (23, bind::storage(false)),
            ],
        );
        let args_layout = bind_group_layout(device, "megalights ray args", C, &[(0, bind::storage(false))]);
        let shade_module = create_module(device, "megalights_shade.wgsl");
        let [glass_uniform, glass_maps, glass_sampler] = crate::glass_light::GlassLights::entries();
        // The G-buffer, the lights, the samples and the local lights' glass maps.
        let surface = [(0, bind::uniform()), (1, tex), (2, tex), (3, tex), (4, tex), (5, bind::storage(true)), (7, bind::storage(true)), glass_uniform, glass_maps, glass_sampler];
        let out = bind::storage_texture(F::Rgba16Float);
        let shade_layout = bind_group_layout(device, "megalights shade", C, &[&surface[..], &[(8, out), (9, out)]].concat());
        let debug_layout = bind_group_layout(device, "megalights debug", C, &[&surface[..], &[(10, tex), (12, out)]].concat());
        let denoise_module = create_module(device, "megalights_denoise.wgsl");
        // Frame, depth, normal and motion.
        let geometry = [(0, bind::uniform()), (1, tex), (2, tex), (3, tex)];
        let info = bind::texture_uint();
        let temporal = [(4, tex), (5, tex), (6, tex), (7, tex), (8, info), (9, out), (10, out), (11, bind::storage_texture(F::R32Uint))];
        let temporal_layout = bind_group_layout(device, "megalights temporal", C, &[&geometry[..], &temporal].concat());
        let spatial = [(9, out), (10, out), (12, tex), (13, tex), (14, info), (15, bind::uniform()), (16, tex)];
        let spatial_layout = bind_group_layout(device, "megalights spatial", C, &[&geometry[..], &spatial].concat());
        let spatial_params = SPATIAL_STRIDES
            .iter()
            .enumerate()
            .map(|(i, &stride)| {
                let buffer = crate::util::uniform_buffer(device, "megalights spatial pass", 16);
                let last = i + 1 == SPATIAL_STRIDES.len();
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[stride, (i == 0) as u32, last as u32, 0]));
                buffer
            })
            .collect();
        Self {
            generate: [0.0, 1.0].map(|shared| {
                let layouts = [&sample_layout, &args_layout];
                lazy_compute_pipeline_with(device, "megalights generate", &sample_module, "generate", &layouts, &[("ML_SHARED_RAYS", shared)])
            }),
            ray_args: lazy_compute_pipeline_with(device, "megalights ray args", &sample_module, "ray_args_main", &[&sample_layout, &args_layout], &[]),
            trace: [0.0, 1.0].map(|hzb| lazy_compute_pipeline(device, "megalights trace", &sample_module, "trace", &sample_layout, &[("ML_HZB_TRACE", hzb)])),
            sample_layout,
            args_layout,
            shade: lazy_compute_pipeline(device, "megalights shade", &shade_module, "shade", &shade_layout, &[]),
            shade_substrate: None,
            shade_layout,
            debug: lazy_compute_pipeline(device, "megalights debug", &shade_module, "debug", &debug_layout, &[]),
            debug_layout,
            temporal: lazy_compute_pipeline(device, "megalights temporal", &denoise_module, "temporal", &temporal_layout, &[]),
            temporal_layout,
            spatial: lazy_compute_pipeline(device, "megalights spatial", &denoise_module, "spatial", &spatial_layout, &[]),
            spatial_layout,
            spatial_params,
            targets: None,
        }
    }

    /// Compiles the slab variant of the shading once a scene needs it (`substrate`: its group 1
    /// layout).
    pub fn sync_substrate(&mut self, device: &wgpu::Device, substrate: Option<&wgpu::BindGroupLayout>) {
        if let (Some(group), None) = (substrate, &self.shade_substrate) {
            let source = crate::shader::compose_variant("megalights_shade.wgsl", &crate::shader::substrate_lighting());
            let module = crate::shader::create_module_from(device, "megalights shade (substrate)", source);
            let layouts = [&self.shade_layout, group];
            self.shade_substrate = Some(crate::util::compute_pipeline_with(device, "megalights shade (substrate)", &module, "shade", &layouts, &[]));
        }
    }

    /// Allocates the targets for `capacity` while MegaLights runs and frees them while it does
    /// not (new targets start without history).
    pub fn update(&mut self, device: &wgpu::Device, capacity: (u32, u32), enabled: bool) {
        if !enabled {
            self.targets = None;
        } else if self.targets.as_ref().is_none_or(|t| t.capacity != capacity) {
            self.targets = Some(Targets::new(device, capacity));
        }
    }

    /// The denoised local lighting of the frame of parity `current` (modulated: the radiance the
    /// lighting pass adds), once recorded.
    pub fn output(&self, current: usize) -> Option<&wgpu::TextureView> {
        let t = self.targets.as_ref()?;
        Some(spatial_target(t, current, SPATIAL_STRIDES.len() - 1)[0])
    }

    /// Records sampling, tracing, shading and denoising. `current` is the frame parity (history
    /// and guide ping-pong).
    pub fn record(&self, rec: &mut Recorder, inputs: &MegaLightsInputs, current: usize) {
        let Some(t) = &self.targets else { return };
        let device = rec.device;
        let ft = inputs.targets;
        let (w, h) = (ft.width, ft.height);
        let (blocks_x, blocks_y) = (w.div_ceil(2), h.div_ceil(2));
        let (global_sdf, _) = inputs.lumen.debug_views();
        let (mesh_sdf_data, mesh_sdf_atlas) = inputs.lumen.near_field();
        rec.encoder.clear_buffer(&t.guide, current as u64 * t.guide_half, Some(t.guide_half));
        rec.encoder.clear_buffer(&t.ray_args, 0, None);
        if inputs.hzb_traces {
            inputs.hzb.record_closest(rec, ft, &t.hzb);
        }

        let sample_group = bind_group(
            device,
            "megalights sample",
            &self.sample_layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, TextureView(&ft.depth)),
                (2, TextureView(&ft.gbuffer_albedo)),
                (3, TextureView(&ft.gbuffer_normal)),
                (4, TextureView(&ft.gbuffer_motion)),
                (5, inputs.lights.lights.as_entire_binding()),
                (6, inputs.lights.grid.as_entire_binding()),
                (7, t.samples.as_entire_binding()),
                (9, t.guide.as_entire_binding()),
                (10, inputs.lumen.uniform().as_entire_binding()),
                (11, TextureView(global_sdf)),
                (12, Sampler(&inputs.samplers.linear)),
                (13, mesh_sdf_data.as_entire_binding()),
                (14, TextureView(mesh_sdf_atlas)),
                (15, inputs.vsm.uniform.as_entire_binding()),
                (16, inputs.vsm.cache.entries.as_entire_binding()),
                (17, TextureView(&inputs.vsm.atlas)),
                (18, inputs.vsm.local.uniform.as_entire_binding()),
                (22, TextureView(&t.hzb.view)),
                (23, t.rays.as_entire_binding()),
            ],
        );
        let args_group = bind_group(device, "megalights ray args", &self.args_layout, &[(0, t.ray_args.as_entire_binding())]);
        {
            let mut pass = rec.compute_pass("megalights sample");
            pass.set_pipeline(&self.generate[inputs.shared_rays as usize]);
            pass.set_bind_group(0, &sample_group, &[]);
            pass.set_bind_group(1, &args_group, &[]);
            pass.dispatch_workgroups(workgroups(blocks_x, 8), workgroups(blocks_y, 8), 1);
            pass.set_pipeline(&self.ray_args);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = rec.compute_pass("megalights trace");
            pass.set_pipeline(&self.trace[inputs.hzb_traces as usize]);
            pass.set_bind_group(0, &sample_group, &[]);
            pass.dispatch_workgroups_indirect(&t.ray_args, 0);
        }

        let shade_group = bind_group(
            device,
            "megalights shade",
            &self.shade_layout,
            &[&self.surface_entries(inputs, t)[..], &[(8, TextureView(&t.raw[0])), (9, TextureView(&t.raw[1]))]].concat(),
        );
        {
            let mut pass = rec.compute_pass("megalights shade");
            pass.set_bind_group(0, &shade_group, &[]);
            match (inputs.substrate, &self.shade_substrate) {
                (Some(slabs), Some(pipeline)) => {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, slabs, &[]);
                }
                _ => pass.set_pipeline(&self.shade),
            }
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }

        let geometry = [
            (0, inputs.frame.as_entire_binding()),
            (1, TextureView(&ft.depth)),
            (2, TextureView(&ft.gbuffer_normal)),
            (3, TextureView(&ft.gbuffer_motion)),
        ];
        let (accumulated, previous) = (&t.history[current], &t.history[1 - current]);
        let temporal = [
            (4, TextureView(&t.raw[0])),
            (5, TextureView(&t.raw[1])),
            (6, TextureView(&previous[0])),
            (7, TextureView(&previous[1])),
            (8, TextureView(&previous[2])),
            (9, TextureView(&accumulated[0])),
            (10, TextureView(&accumulated[1])),
            (11, TextureView(&accumulated[2])),
        ];
        let temporal_group = bind_group(device, "megalights temporal", &self.temporal_layout, &[&geometry[..], &temporal].concat());
        let spatial_groups: Vec<wgpu::BindGroup> = (0..SPATIAL_STRIDES.len())
            .map(|i| {
                let [in_d, in_s] = if i == 0 { [&accumulated[0], &accumulated[1]] } else { spatial_target(t, current, i - 1) };
                let [out_d, out_s] = spatial_target(t, current, i);
                let spatial = [
                    (9, TextureView(out_d)),
                    (10, TextureView(out_s)),
                    (12, TextureView(in_d)),
                    (13, TextureView(in_s)),
                    (14, TextureView(&accumulated[2])),
                    (15, self.spatial_params[i].as_entire_binding()),
                    (16, TextureView(&ft.gbuffer_albedo)),
                ];
                bind_group(device, "megalights spatial", &self.spatial_layout, &[&geometry[..], &spatial].concat())
            })
            .collect();
        {
            let mut pass = rec.compute_pass("megalights temporal");
            pass.set_pipeline(&self.temporal);
            pass.set_bind_group(0, &temporal_group, &[]);
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }
        let mut pass = rec.compute_pass("megalights spatial");
        pass.set_pipeline(&self.spatial);
        for group in &spatial_groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }
    }

    /// Debug views 18 (samples) and 19 (noisy vs denoised): drawn over the lit HDR image.
    pub fn record_debug(&self, rec: &mut Recorder, inputs: &MegaLightsInputs, current: usize) {
        let (Some(t), Some(lighting)) = (&self.targets, self.output(current)) else { return };
        let ft = inputs.targets;
        let group = bind_group(
            rec.device,
            "megalights debug",
            &self.debug_layout,
            &[&self.surface_entries(inputs, t)[..], &[(10, TextureView(lighting)), (12, TextureView(&ft.hdr))]].concat(),
        );
        let mut pass = rec.compute_pass("megalights debug");
        pass.set_pipeline(&self.debug);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(ft.width, 8), workgroups(ft.height, 8), 1);
    }

    /// Bindings 0–7 of the shading and debug passes: the G-buffer, the lights and the samples;
    /// 92–94: the local lights' glass maps.
    fn surface_entries<'a>(&self, inputs: &'a MegaLightsInputs, t: &'a Targets) -> Vec<(u32, wgpu::BindingResource<'a>)> {
        let ft = inputs.targets;
        let mut entries = vec![
            (0, inputs.frame.as_entire_binding()),
            (1, TextureView(&ft.depth)),
            (2, TextureView(&ft.gbuffer_albedo)),
            (3, TextureView(&ft.gbuffer_normal)),
            (4, TextureView(&ft.gbuffer_motion)),
            (5, inputs.lights.lights.as_entire_binding()),
            (7, t.samples.as_entire_binding()),
        ];
        entries.extend(inputs.glass_lights.resources());
        entries
    }
}

/// Where spatial iteration `i` writes: alternately the previous frame's accumulation textures
/// (free once the temporal pass has read them) and the raw textures (free once it has too).
fn spatial_target(t: &Targets, current: usize, i: usize) -> [&wgpu::TextureView; 2] {
    if i.is_multiple_of(2) {
        [&t.history[1 - current][0], &t.history[1 - current][1]]
    } else {
        [&t.raw[0], &t.raw[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer layouts here and in megalights_common.wgsl must agree.
    #[test]
    fn layout_constants_match_the_shaders() {
        let source = crate::shader::compose("megalights_shade.wgsl");
        for (name, value) in [("ML_BLOCK_WORDS", BLOCK_WORDS), ("ML_GUIDE_TILE", GUIDE_TILE as u64), ("ML_GUIDE_WORDS", GUIDE_WORDS)] {
            assert!(source.contains(&format!("const {name}: u32 = {value}u;")), "{name} != {value}");
        }
    }
}
