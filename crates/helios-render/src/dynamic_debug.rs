//! [`crate::DebugView::Invalidation`]: what moving instances invalidated this frame — movable
//! instances tinted orange, sun shadow pages rendered this frame in red, distance-field bricks
//! recomposed for moved instances in cyan (`shaders/dynamic_debug.wgsl`). Drawn over the
//! lighting pass's output, like the mesh-card debug view.

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use crate::lumen::Lumen;
use crate::passes::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, workgroups};
use crate::vsm::VirtualShadowMap;

pub(crate) struct InvalidationDebug {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl InvalidationDebug {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = bind_group_layout(
            device,
            "invalidation debug",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::texture(false)),
                (2, bind::texture(false)),
                (3, bind::texture(false)),
                (4, bind::uniform()),
                (5, bind::storage(true)),
                (6, bind::texture_depth()),
                (7, bind::sampler_comparison()),
                (8, bind::uniform()),
                (9, bind::storage(true)),
                (10, bind::storage_texture(TextureFormat::Rgba16Float)),
            ],
        );
        let module = create_module(device, "dynamic_debug.wgsl");
        Self { pipeline: compute_pipeline(device, "invalidation debug", &module, "invalidation", &layout, &[]), layout }
    }

    /// Overwrites the lit pixels of `targets.hdr` with the view.
    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, vsm: &VirtualShadowMap, lumen: &Lumen, samplers: &Samplers) {
        let group = bind_group(
            rec.device,
            "invalidation debug",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, TextureView(&targets.depth)),
                (2, TextureView(&targets.gbuffer_albedo)),
                (3, TextureView(&targets.gbuffer_normal)),
                (4, vsm.uniform.as_entire_binding()),
                (5, vsm.cache.entries.as_entire_binding()),
                (6, TextureView(&vsm.atlas)),
                (7, Sampler(&samplers.shadow)),
                (8, lumen.uniform().as_entire_binding()),
                (9, lumen.brick_stamps().as_entire_binding()),
                (10, TextureView(&targets.hdr)),
            ],
        );
        let mut pass = rec.compute_pass("invalidation debug");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
    }
}
