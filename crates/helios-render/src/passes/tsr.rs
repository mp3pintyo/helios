//! Temporal super resolution (`shaders/tsr.wgsl`): reconstructs the output-resolution image
//! from the jittered internal-resolution frame and the reprojected history. The screen
//! percentage, jitter sequence and dynamic resolution controller are in `resolution.rs`.

use glam::Vec4;
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::ShaderStages;

use super::Samplers;
use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::targets::{FrameTargets, HDR_FORMAT, HISTORY_MOTION_FORMAT};
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, uniform_buffer, workgroups};
use crate::pipelines::Lazy;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TsrUniform {
    /// Output width, height, 1 / width, 1 / height.
    output: Vec4,
}

pub struct TsrPass {
    pipeline: Lazy<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
}

impl TsrPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "tsr.wgsl");
        let layout = bind_group_layout(
            device,
            "tsr",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture(false)),
                (3, bind::texture(false)),
                (4, bind::texture(false)),
                (5, bind::texture(true)),
                (6, bind::texture(false)),
                (7, bind::sampler()),
                (8, bind::storage_texture(HDR_FORMAT)),
                (9, bind::storage_texture(HISTORY_MOTION_FORMAT)),
                (10, bind::storage(true)),
            ],
        );
        Self {
            pipeline: compute_pipeline(device, "tsr", &module, "resolve", &layout, &[]),
            layout,
            uniform: uniform_buffer(device, "tsr", std::mem::size_of::<TsrUniform>() as u64),
        }
    }

    /// Writes the output size for this frame.
    pub fn update(&self, queue: &wgpu::Queue, (w, h): (u32, u32)) {
        let uniform = TsrUniform { output: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32) };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Reconstructs `targets.hdr` (internal resolution) with `history[1 - current]` into
    /// `history[current]` (output resolution). `exposure` is the eye adaptation's buffer (its
    /// value from the previous frame).
    pub fn record(&self, rec: &mut Recorder, frame: &wgpu::Buffer, targets: &FrameTargets, samplers: &Samplers, exposure: &wgpu::Buffer, current: usize) {
        let (w, h) = targets.output;
        let group = bind_group(
            rec.device,
            "tsr",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, self.uniform.as_entire_binding()),
                (2, TextureView(&targets.hdr)),
                (3, TextureView(&targets.depth)),
                (4, TextureView(&targets.gbuffer_motion)),
                (5, TextureView(&targets.history[1 - current])),
                (6, TextureView(&targets.history_motion[1 - current])),
                (7, Sampler(&samplers.linear)),
                (8, TextureView(&targets.history[current])),
                (9, TextureView(&targets.history_motion[current])),
                (10, exposure.as_entire_binding()),
            ],
        );
        let mut pass = rec.compute_pass("tsr");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(workgroups(w, 8), workgroups(h, 8), 1);
    }
}
