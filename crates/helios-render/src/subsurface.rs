//! Screen-space subsurface scattering (`subsurface.wgsl`; UE's subsurface profile, separable).
//!
//! The deferred lighting writes the diffuse light of slab pixels with
//! [`Slab::subsurface`](crate::Slab::subsurface) apart (demodulated, fog-attenuated); here a
//! horizontal compute pass and a vertical full-screen pass diffuse it with each material's
//! profile, and the vertical pass adds it, times the diffuse colour, to the lit image through
//! additive blending (storage textures cannot read-modify-write the HDR format). Runs only while
//! the scene has subsurface materials, in the lit view.

use wgpu::BindingResource::TextureView;
use wgpu::ShaderStages;

use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::substrate_gpu::{ExtendedMaterials, SUBSURFACE_FORMAT};
use crate::targets::{FrameTargets, HDR_FORMAT};
use crate::util::{bind, bind_group, bind_group_layout, lazy_compute_pipeline, pipeline_layout, workgroups};
use crate::pipelines::Lazy;

/// Its pipelines are compiled when a scene first has subsurface materials ([`Lazy::new`]).
pub struct SubsurfacePass {
    horizontal_layout: wgpu::BindGroupLayout,
    horizontal: Lazy<wgpu::ComputePipeline>,
    vertical_layout: wgpu::BindGroupLayout,
    vertical: Lazy<wgpu::RenderPipeline>,
}

impl SubsurfacePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "subsurface.wgsl");
        let tex = bind::texture(false);
        let common = [(0, bind::uniform()), (1, tex), (2, bind::texture_uint()), (3, bind::uniform()), (4, tex), (7, tex)];
        let horizontal_layout = bind_group_layout(device, "subsurface horizontal", ShaderStages::COMPUTE, &[&common[..], &[(5, bind::storage_texture(SUBSURFACE_FORMAT))]].concat());
        let vertical_layout = bind_group_layout(device, "subsurface vertical", ShaderStages::FRAGMENT, &[&common[..], &[(6, tex)]].concat());
        let vertical = {
            let (device, module, vertical_layout) = (device.clone(), module.clone(), vertical_layout.clone());
            Lazy::new(move || {
                let additive = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
                crate::util::render_pipeline(&device, &wgpu::RenderPipelineDescriptor {
                    label: Some("subsurface vertical"),
                    layout: Some(&pipeline_layout(&device, "subsurface vertical", &[&vertical_layout])),
                    vertex: wgpu::VertexState { module: &module, entry_point: Some("fullscreen"), compilation_options: Default::default(), buffers: &[] },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("vertical"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: HDR_FORMAT,
                            blend: Some(wgpu::BlendState { color: additive, alpha: additive }),
                            write_mask: wgpu::ColorWrites::COLOR,
                        })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                })
            })
        };
        Self {
            horizontal: lazy_compute_pipeline(device, "subsurface horizontal", &module, "horizontal", &horizontal_layout, &[]),
            horizontal_layout,
            vertical_layout,
            vertical,
        }
    }

    /// Queues the pipelines for the browser's background compile (the scene has subsurface
    /// materials).
    pub fn prefetch(&self) {
        self.horizontal.prefetch();
        self.vertical.prefetch();
    }

    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, extended: &ExtendedMaterials) {
        let Some((slabs, substrate, [light, diffused])) = extended.subsurface_inputs() else { return };
        let common = [
            (0, frame.as_entire_binding()),
            (1, TextureView(&targets.depth)),
            (2, TextureView(substrate)),
            (3, slabs.as_entire_binding()),
            (7, TextureView(&targets.gbuffer_motion)),
        ];
        let device = rec.device;
        let horizontal = [(4, TextureView(light)), (5, TextureView(diffused))];
        let horizontal = bind_group(device, "subsurface horizontal", &self.horizontal_layout, &[&common[..], &horizontal].concat());
        let vertical = [(4, TextureView(diffused)), (6, TextureView(&targets.gbuffer_albedo))];
        let vertical = bind_group(device, "subsurface vertical", &self.vertical_layout, &[&common[..], &vertical].concat());
        {
            let mut pass = rec.compute_pass("subsurface");
            pass.set_pipeline(&self.horizontal);
            pass.set_bind_group(0, &horizontal, &[]);
            pass.dispatch_workgroups(workgroups(targets.width, 8), workgroups(targets.height, 8), 1);
        }
        let attachment = wgpu::RenderPassColorAttachment {
            view: &targets.hdr,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        };
        let mut pass = rec.render_pass("subsurface blend", &[Some(attachment)], None);
        pass.set_viewport(0.0, 0.0, targets.width as f32, targets.height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.vertical);
        pass.set_bind_group(0, &vertical, &[]);
        pass.draw(0..3, 0..1);
    }
}
