//! Eye adaptation, bloom and tone mapping (at output resolution, after TSR).

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{ShaderStages, TextureFormat};

use super::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{self, bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, workgroups};

const C: ShaderStages = ShaderStages::COMPUTE;
const HDR: TextureFormat = TextureFormat::Rgba16Float;

// ---- Eye adaptation -----------------------------------------------------------------------------

pub struct ExposurePass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// [exposure, average luminance, -, -]
    pub buffer: wgpu::Buffer,
}

impl ExposurePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "exposure.wgsl");
        let layout = bind_group_layout(
            device,
            "exposure",
            C,
            &[(0, bind::uniform()), (1, bind::texture(true)), (2, bind::sampler()), (3, bind::storage(false))],
        );
        Self {
            pipeline: compute_pipeline(device, "exposure", &module, "adapt", &layout, &[]),
            layout,
            buffer: storage_buffer(device, "exposure", 16, wgpu::BufferUsages::empty()),
        }
    }

    pub fn record(
        &self,
        rec: &mut Recorder,
        frame: &wgpu::Buffer,
        hdr: &wgpu::TextureView,
        samplers: &Samplers,
    ) {
        let device = rec.device;
        let group = bind_group(
            device,
            "exposure",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, TextureView(hdr)),
                (2, Sampler(&samplers.linear)),
                (3, self.buffer.as_entire_binding()),
            ],
        );
        let mut pass = rec.compute_pass("exposure");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

// ---- Bloom -------------------------------------------------------------------------------------

pub struct BloomPass {
    downsample_first: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    upsample: wgpu::ComputePipeline,
    down_layout: wgpu::BindGroupLayout,
    up_layout: wgpu::BindGroupLayout,
}

impl BloomPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "bloom.wgsl");
        let down_layout = bind_group_layout(
            device,
            "bloom down",
            C,
            &[(0, bind::texture(true)), (2, bind::sampler()), (3, bind::storage_texture(HDR))],
        );
        let up_layout = bind_group_layout(
            device,
            "bloom up",
            C,
            &[(0, bind::texture(true)), (1, bind::texture(true)), (2, bind::sampler()), (3, bind::storage_texture(HDR))],
        );
        Self {
            downsample_first: compute_pipeline(device, "bloom down first", &module, "downsample", &down_layout, &[("FIRST_LEVEL", 1.0)]),
            downsample: compute_pipeline(device, "bloom down", &module, "downsample", &down_layout, &[]),
            upsample: compute_pipeline(device, "bloom up", &module, "upsample", &up_layout, &[]),
            down_layout,
            up_layout,
        }
    }

    /// Builds the bloom chain from `source`; the result ends up in `targets.bloom_up[0]`.
    pub fn record(
        &self,
        rec: &mut Recorder,
        source: &wgpu::TextureView,
        targets: &FrameTargets,
        samplers: &Samplers,
    ) {
        let device = rec.device;
        let levels = targets.bloom_down.len();
        let size = |level: usize| {
            ((targets.output.0.div_ceil(2) >> level).max(1), (targets.output.1.div_ceil(2) >> level).max(1))
        };
        let mut pass = rec.compute_pass("bloom");

        for level in 0..levels {
            let input = if level == 0 { source } else { &targets.bloom_down[level - 1] };
            let group = bind_group(
                device,
                "bloom down",
                &self.down_layout,
                &[(0, TextureView(input)), (2, Sampler(&samplers.linear)), (3, TextureView(&targets.bloom_down[level]))],
            );
            pass.set_pipeline(if level == 0 { &self.downsample_first } else { &self.downsample });
            pass.set_bind_group(0, &group, &[]);
            let (w, h) = size(level);
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }

        // up[i] = down[i] + tent(up[i + 1]); the coarsest level reads the down chain directly.
        for level in (0..levels.saturating_sub(1)).rev() {
            let lower = if level + 1 == levels - 1 { &targets.bloom_down[level + 1] } else { &targets.bloom_up[level + 1] };
            let group = bind_group(
                device,
                "bloom up",
                &self.up_layout,
                &[
                    (0, TextureView(&targets.bloom_down[level])),
                    (1, TextureView(lower)),
                    (2, Sampler(&samplers.linear)),
                    (3, TextureView(&targets.bloom_up[level])),
                ],
            );
            pass.set_pipeline(&self.upsample);
            pass.set_bind_group(0, &group, &[]);
            let (w, h) = size(level);
            pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
        }
    }
}

// ---- Tone mapping ------------------------------------------------------------------------------

pub struct TonemapPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl TonemapPass {
    pub fn new(device: &wgpu::Device, output_format: TextureFormat) -> Self {
        let module = create_module(device, "tonemap.wgsl");
        let layout = bind_group_layout(
            device,
            "tonemap",
            ShaderStages::FRAGMENT,
            &[
                (0, bind::uniform()),
                (1, bind::texture(false)),
                (2, bind::texture(true)),
                (3, bind::sampler()),
                (4, bind::storage(true)),
            ],
        );
        let pipeline_layout = util::pipeline_layout(device, "tonemap", &[&layout]);
        let constants = [("OUTPUT_SRGB", if output_format.is_srgb() { 1.0 } else { 0.0 })];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
                targets: &[Some(wgpu::ColorTargetState { format: output_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        rec: &mut Recorder,
        frame: &wgpu::Buffer,
        hdr: &wgpu::TextureView,
        bloom: &wgpu::TextureView,
        exposure: &wgpu::Buffer,
        samplers: &Samplers,
        output: &wgpu::TextureView,
    ) {
        let device = rec.device;
        let group = bind_group(
            device,
            "tonemap",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, TextureView(hdr)),
                (2, TextureView(bloom)),
                (3, Sampler(&samplers.linear)),
                (4, exposure.as_entire_binding()),
            ],
        );
        let mut pass = rec.render_pass(
            "tonemap",
            &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            None,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}
