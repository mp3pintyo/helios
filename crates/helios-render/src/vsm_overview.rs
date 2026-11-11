//! [`crate::DebugView::ShadowOverview`]: several views of the sun's virtual shadow map side by side
//! (UE 5.9's "Virtual Shadow Map Overview") — clipmap levels, what each page did this frame,
//! why pages were invalidated, and the physical page pool with its pressure
//! (`shaders/vsm_overview.wgsl`). Drawn over the lighting pass's output, like the invalidation
//! view; its pipeline is compiled when the view is first selected, so it costs nothing otherwise.

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use crate::passes::Samplers;
use crate::pipelines::Lazy;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, workgroups};
use crate::vsm::{VirtualShadowMap, LOCAL_ENTRY_BASE};

pub(crate) struct ShadowOverview {
    layout: wgpu::BindGroupLayout,
    pipeline: Lazy<wgpu::ComputePipeline>,
}

impl ShadowOverview {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = bind_group_layout(
            device,
            "shadow overview",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::texture(false)),
                (2, bind::texture(false)),
                (3, bind::uniform()),
                (4, bind::storage(true)),
                (5, bind::texture_depth()),
                (6, bind::sampler_comparison()),
                (7, bind::storage(true)),
                (8, bind::storage(true)),
                (9, bind::storage(true)),
                (10, bind::storage_texture(TextureFormat::Rgba16Float)),
                (11, bind::texture(false)),
            ],
        );
        let module = create_module(device, "vsm_overview.wgsl");
        let constants = [("LOCAL_BASE", LOCAL_ENTRY_BASE as f64)];
        Self { pipeline: lazy_compute_pipeline(device, "shadow overview", &module, "overview", &layout, &constants), layout }
    }

    /// Overwrites `targets.hdr` with the four panels.
    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, vsm: &VirtualShadowMap, samplers: &Samplers) {
        let group = bind_group(
            rec.device,
            "shadow overview",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, TextureView(&targets.depth)),
                (2, TextureView(&targets.gbuffer_normal)),
                (3, vsm.uniform.as_entire_binding()),
                (4, vsm.cache.entries.as_entire_binding()),
                (5, TextureView(&vsm.atlas)),
                (6, Sampler(&samplers.shadow)),
                (7, vsm.info().as_entire_binding()),
                (8, vsm.cache.owner.as_entire_binding()),
                (9, vsm.cache.lists.as_entire_binding()),
                (10, TextureView(&targets.hdr)),
                (11, TextureView(&targets.gbuffer_albedo)),
            ],
        );
        let mut pass = rec.compute_pass("shadow overview");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
    }
}
