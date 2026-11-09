//! Editor selection outline: a full-screen pass over the output that tints the selected
//! instances and outlines them in orange, read from the visibility buffer (`selection.wgsl`).
//!
//! The selection is one bit per instance in a storage buffer, uploaded when it changes. The pass
//! only runs while something is selected, so a frame without a selection is unchanged and
//! costs nothing.

use wgpu::{ShaderStages, TextureFormat};

use crate::profiler::Recorder;
use crate::scene::InstanceHandle;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{self, bind, bind_group, bind_group_layout, storage_buffer};

pub struct SelectionOutline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bits: wgpu::Buffer,
    /// The selection to upload (`None`: uploaded).
    changed: Option<Vec<u32>>,
    count: usize,
}

impl SelectionOutline {
    pub fn new(device: &wgpu::Device, output_format: TextureFormat) -> Self {
        let module = create_module(device, "selection.wgsl");
        let layout = bind_group_layout(
            device,
            "selection outline",
            ShaderStages::FRAGMENT,
            &[(0, bind::uniform()), (1, bind::texture_uint()), (2, bind::storage(true)), (3, bind::storage(true))],
        );
        let pipeline_layout = util::pipeline_layout(device, "selection outline", &[&layout]);
        let constants = [("OUTPUT_SRGB", if output_format.is_srgb() { 1.0 } else { 0.0 })];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection outline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout, bits: storage_buffer(device, "selection", 16, wgpu::BufferUsages::empty()), changed: None, count: 0 }
    }

    /// Replaces the selection.
    pub fn set(&mut self, selection: &[InstanceHandle]) {
        let words = selection.iter().map(|h| h.0 as usize / 32 + 1).max().unwrap_or(0);
        let mut bits = vec![0u32; words];
        for h in selection {
            bits[h.0 as usize / 32] |= 1 << (h.0 % 32);
        }
        self.count = selection.len();
        self.changed = Some(bits);
    }

    pub fn record(&mut self, rec: &mut Recorder, queue: &wgpu::Queue, frame: &wgpu::Buffer, targets: &FrameTargets, visible_clusters: &wgpu::Buffer, output: &wgpu::TextureView) {
        if let Some(bits) = self.changed.take() {
            let bytes: &[u8] = bytemuck::cast_slice(&bits);
            if (self.bits.size() as usize) < bytes.len() {
                self.bits = storage_buffer(rec.device, "selection", bytes.len().next_power_of_two() as u64, wgpu::BufferUsages::empty());
            }
            // Clear what a larger earlier selection left behind, then write the new bits.
            let mut padded = bytes.to_vec();
            padded.resize(self.bits.size() as usize, 0);
            queue.write_buffer(&self.bits, 0, &padded);
        }
        if self.count == 0 {
            return;
        }
        let group = bind_group(
            rec.device,
            "selection outline",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, wgpu::BindingResource::TextureView(&targets.visibility)),
                (2, visible_clusters.as_entire_binding()),
                (3, self.bits.as_entire_binding()),
            ],
        );
        let mut pass = rec.render_pass(
            "selection outline",
            &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            None,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}
