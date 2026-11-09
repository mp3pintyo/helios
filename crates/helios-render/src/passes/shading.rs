//! Global illumination and deferred lighting.

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use super::sky::Sky;
use super::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, workgroups};

const C: ShaderStages = ShaderStages::COMPUTE;

fn dispatch_full(pass: &mut wgpu::ComputePass, width: u32, height: u32) {
    pass.dispatch_workgroups(workgroups(width, 8), workgroups(height, 8), 1);
}

// ---- Screen-space global illumination ----------------------------------------------------------

pub struct GiPass {
    temporal: wgpu::ComputePipeline,
    temporal_layout: wgpu::BindGroupLayout,
}

impl GiPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "gi.wgsl");
        let out = bind::storage_texture(TextureFormat::Rgba16Float);
        let temporal_layout = bind_group_layout(
            device,
            "gi temporal",
            C,
            &[
                (0, bind::uniform()),
                (1, bind::texture(false)),
                (2, bind::texture(false)),
                (3, bind::texture(false)),
                (6, bind::sampler()),
                (7, out),
                (8, bind::texture(false)),
                (9, bind::texture(true)),
            ],
        );
        Self { temporal: compute_pipeline(device, "gi temporal", &module, "temporal", &temporal_layout, &[]), temporal_layout }
    }

    /// Denoises Lumen's half-resolution diffuse gather (`gi_raw`) into `gi_history[current]`.
    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, samplers: &Samplers, current: usize) {
        let device = rec.device;
        let temporal_group = bind_group(
            device,
            "gi temporal",
            &self.temporal_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, TextureView(&targets.depth)),
                (2, TextureView(&targets.gbuffer_normal)),
                (3, TextureView(&targets.gbuffer_motion)),
                (6, Sampler(&samplers.linear)),
                (7, TextureView(&targets.gi_history[current])),
                (8, TextureView(&targets.gi_raw)),
                (9, TextureView(&targets.gi_history[1 - current])),
            ],
        );
        let (w, h) = (targets.width.div_ceil(2), targets.height.div_ceil(2));
        let mut pass = rec.compute_pass("gi temporal");
        pass.set_pipeline(&self.temporal);
        pass.set_bind_group(0, &temporal_group, &[]);
        dispatch_full(&mut pass, w, h);
    }
}

// ---- Deferred lighting -----------------------------------------------------------------------

pub struct LightingPass {
    pipeline: wgpu::ComputePipeline,
    /// Soft shadows of the strongest shadow-mapped local lights per pixel, before the shading.
    local_shadows: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct LightingInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub targets: &'a FrameTargets,
    pub sky: &'a Sky,
    pub samplers: &'a Samplers,
    pub vsm: &'a crate::vsm::VirtualShadowMap,
    pub gi: &'a wgpu::TextureView,
    pub reflection: &'a wgpu::TextureView,
    pub lumen: &'a crate::lumen::Lumen,
    pub lights: &'a crate::lights::LightCulling,
    pub clouds: &'a crate::clouds::Clouds,
    /// Resolved clouds of this frame (quarter resolution).
    pub cloud_texture: &'a wgpu::TextureView,
    pub fog: &'a crate::fog::VolumetricFog,
    /// MegaLights' denoised local lighting, which replaces the clustered light loop; `None`:
    /// the loop shades.
    pub megalights: Option<&'a wgpu::TextureView>,
    /// A 1×1 black texture for unused inputs.
    pub black: &'a wgpu::TextureView,
}

impl LightingPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "lighting.wgsl");
        let layout = bind_group_layout(
            device,
            "lighting",
            C,
            &[
                (0, bind::uniform()),
                (1, bind::texture(false)),
                (2, bind::texture(false)),
                (3, bind::texture(false)),
                (4, bind::texture(false)),
                (5, bind::texture_depth()),
                (6, bind::sampler_comparison()),
                (7, bind::texture(true)),
                (8, bind::texture(true)),
                (9, bind::sampler()),
                (10, bind::storage(true)),
                (11, bind::texture(false)),
                (12, bind::storage_texture(TextureFormat::Rgba16Float)),
                (13, bind::uniform()),
                (14, bind::storage(true)),
                (15, bind::texture(true)),
                (16, bind::uniform()),
                (17, bind::texture_3d(true)),
                (18, bind::texture_3d(true)),
                (19, bind::storage(true)),
                (20, bind::sampler()),
                (21, bind::storage(true)),
                (22, bind::storage(true)),
                (23, bind::texture(true)),
                (24, bind::texture(true)),
                (25, bind::uniform()),
                (26, bind::texture_3d(true)),
                (27, bind::uniform()),
                (28, bind::storage(true)),
                (29, bind::texture_3d(true)),
                (60, bind::uniform()),
                (61, bind::storage(false)),
                (62, bind::texture(false)),
            ],
        );
        Self {
            pipeline: compute_pipeline(device, "lighting", &module, "shade", &layout, &[]),
            local_shadows: compute_pipeline(device, "local light shadows", &module, "local_shadows", &layout, &[]),
            layout,
        }
    }

    pub fn record(&self, rec: &mut Recorder, inputs: &LightingInputs) {
        let device = rec.device;
        let t = inputs.targets;
        let group = bind_group(
            device,
            "lighting",
            &self.layout,
            &[
                (0, inputs.frame.as_entire_binding()),
                (1, TextureView(&t.depth)),
                (2, TextureView(&t.gbuffer_albedo)),
                (3, TextureView(&t.gbuffer_normal)),
                (4, TextureView(&t.gbuffer_motion)),
                (5, TextureView(&inputs.vsm.atlas)),
                (6, Sampler(&inputs.samplers.shadow)),
                (7, TextureView(&inputs.sky.transmittance)),
                (8, TextureView(&inputs.sky.sky_view)),
                (9, Sampler(&inputs.samplers.linear)),
                (10, inputs.sky.sh.as_entire_binding()),
                (11, TextureView(inputs.gi)),
                (12, TextureView(&t.hdr)),
                (13, inputs.vsm.uniform.as_entire_binding()),
                (14, inputs.vsm.cache.entries.as_entire_binding()),
                (15, TextureView(inputs.reflection)),
                (16, inputs.lumen.uniform().as_entire_binding()),
                (17, TextureView(inputs.lumen.debug_views().0)),
                (18, TextureView(inputs.lumen.debug_views().1)),
                (19, inputs.lumen.radiance_probes().as_entire_binding()),
                (20, Sampler(&inputs.samplers.linear)),
                (21, inputs.lights.lights.as_entire_binding()),
                (22, inputs.lights.grid.as_entire_binding()),
                (23, TextureView(inputs.cloud_texture)),
                (24, TextureView(&inputs.clouds.shadow)),
                (25, inputs.clouds.uniform.as_entire_binding()),
                (26, TextureView(inputs.fog.integrated())),
                (27, inputs.fog.uniform.as_entire_binding()),
                (28, inputs.lumen.near_field().0.as_entire_binding()),
                (29, TextureView(inputs.lumen.near_field().1)),
                (60, inputs.vsm.local.uniform.as_entire_binding()),
                (61, t.local_shadow_mask.as_entire_binding()),
                (62, TextureView(inputs.megalights.unwrap_or(inputs.black))),
            ],
        );
        // MegaLights traces the shadows of every light itself.
        if inputs.vsm.local.view().is_some() && inputs.megalights.is_none() {
            let mut pass = rec.compute_pass("local light shadows");
            pass.set_pipeline(&self.local_shadows);
            pass.set_bind_group(0, &group, &[]);
            dispatch_full(&mut pass, t.width, t.height);
        }
        let mut pass = rec.compute_pass("lighting");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        dispatch_full(&mut pass, t.width, t.height);
    }
}
