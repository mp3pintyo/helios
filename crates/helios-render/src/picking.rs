//! Editor picking: which instance is under a pixel, read back from the visibility buffer.
//!
//! A pick is a request for one output pixel. The frame that sees it runs a single-thread
//! compute pass after the G-buffer resolve, which maps the pixel to the internal resolution,
//! decodes the visibility buffer entry into its instance and reconstructs the world position
//! from the depth and the G-buffer normal; the 48-byte result is copied into a mappable
//! buffer and read back asynchronously (WebGPU cannot block on the GPU), so it arrives a frame
//! or two later through [`Picker::take`]. Exact per pixel — alpha-tested holes, wind-displaced
//! leaves and software-rasterised clusters pick what is drawn — and free when nobody picks.
//! Translucent surfaces are not in the visibility buffer: callers test them against their
//! bounds (see the editor in `helios-demo`).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use glam::Vec3;
use wgpu::ShaderStages;

use crate::profiler::Recorder;
use crate::scene::InstanceHandle;
use crate::shader::create_module;
use crate::targets::FrameTargets;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, uniform_buffer};

const RESULT_BYTES: u64 = 48;
const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

/// What a pick found under its pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickResult {
    /// The request's id ([`Renderer::request_pick`](crate::Renderer::request_pick)).
    pub id: u32,
    /// The opaque instance drawn there (`None`: sky).
    pub instance: Option<InstanceHandle>,
    /// World position of the surface (meaningless for the sky).
    pub position: Vec3,
    /// Shading normal of the surface.
    pub normal: Vec3,
}

pub struct Picker {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    request_uniform: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    status: Arc<AtomicU32>,
    /// Waiting for a frame: id and output pixel.
    pending: Option<(u32, (u32, u32))>,
    /// Recorded, waiting for the read-back.
    in_flight: Option<u32>,
    copied: bool,
    finished: Option<PickResult>,
}

impl Picker {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = create_module(device, "pick.wgsl");
        let layout = bind_group_layout(
            device,
            "pick",
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture_uint()),
                (3, bind::texture(false)),
                (4, bind::texture(false)),
                (5, bind::storage(true)),
                (6, bind::storage(false)),
            ],
        );
        Self {
            pipeline: compute_pipeline(device, "pick", &module, "pick", &layout, &[]),
            layout,
            request_uniform: uniform_buffer(device, "pick request", 16),
            output: storage_buffer(device, "pick result", RESULT_BYTES, wgpu::BufferUsages::COPY_SRC),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pick readback"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            status: Arc::new(AtomicU32::new(IDLE)),
            pending: None,
            in_flight: None,
            copied: false,
            finished: None,
        }
    }

    /// Asks for the instance under output pixel `pixel`; a newer request replaces one that has
    /// not been recorded yet.
    pub fn request(&mut self, id: u32, pixel: (u32, u32)) {
        self.pending = Some((id, pixel));
    }

    /// The latest finished pick, once.
    pub fn take(&mut self) -> Option<PickResult> {
        self.poll();
        self.finished.take()
    }

    /// Records the pick pass if a request waits and no read-back is in flight.
    pub fn record(&mut self, rec: &mut Recorder, queue: &wgpu::Queue, frame: &wgpu::Buffer, targets: &FrameTargets, visible_clusters: &wgpu::Buffer) {
        self.poll();
        if self.in_flight.is_some() {
            return;
        }
        let Some((id, (x, y))) = self.pending.take() else { return };
        // Output pixel -> internal pixel (the frame renders into the internal viewport).
        let scale = |p: u32, internal: u32, output: u32| ((p as f32 + 0.5) * internal as f32 / output.max(1) as f32) as u32;
        let pixel = [scale(x, targets.width, targets.output.0), scale(y, targets.height, targets.output.1), 0, 0];
        queue.write_buffer(&self.request_uniform, 0, bytemuck::cast_slice(&pixel));
        let group = bind_group(
            rec.device,
            "pick",
            &self.layout,
            &[
                (0, frame.as_entire_binding()),
                (1, self.request_uniform.as_entire_binding()),
                (2, wgpu::BindingResource::TextureView(&targets.visibility)),
                (3, wgpu::BindingResource::TextureView(&targets.depth)),
                (4, wgpu::BindingResource::TextureView(&targets.gbuffer_normal)),
                (5, visible_clusters.as_entire_binding()),
                (6, self.output.as_entire_binding()),
            ],
        );
        {
            let mut pass = rec.compute_pass("pick");
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        rec.encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, RESULT_BYTES);
        self.in_flight = Some(id);
        self.copied = true;
    }

    /// Call after the frame's submission: starts mapping a freshly copied result.
    pub fn after_submit(&mut self) {
        if std::mem::take(&mut self.copied) {
            self.status.store(MAPPING, Ordering::Release);
            let status = self.status.clone();
            self.readback.map_async(wgpu::MapMode::Read, .., move |result| {
                status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
            });
        }
    }

    fn poll(&mut self) {
        if self.status.load(Ordering::Acquire) != READY {
            return;
        }
        if let (Some(id), Ok(view)) = (self.in_flight, self.readback.get_mapped_range(..)) {
            let words: &[u32] = bytemuck::cast_slice(&view);
            let vec = |i: usize| Vec3::new(f32::from_bits(words[i]), f32::from_bits(words[i + 1]), f32::from_bits(words[i + 2]));
            self.finished = Some(PickResult {
                id,
                instance: (words[0] != u32::MAX).then_some(InstanceHandle(words[0])),
                position: vec(4),
                normal: vec(8).normalize_or_zero(),
            });
        }
        self.readback.unmap();
        self.status.store(IDLE, Ordering::Release);
        self.in_flight = None;
    }
}
