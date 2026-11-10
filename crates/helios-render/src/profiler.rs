//! GPU pass timing with timestamp queries, plus the [`Recorder`] every pass records through.
//!
//! Each compute or render pass begun through the recorder gets beginning/end-of-pass
//! timestamps (when the adapter supports `TIMESTAMP_QUERY`). Results are resolved and read
//! back asynchronously, aggregated by label and smoothed, so the HUD shows a stable per-pass
//! breakdown without ever stalling the GPU.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

const CAPACITY: u32 = 512; // timestamps (two per pass)

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

pub struct GpuProfiler {
    query_set: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    readback: Option<wgpu::Buffer>,
    period_ns: f32,
    labels: Vec<String>,
    in_flight: Vec<String>,
    status: Arc<AtomicU32>,
    results: Vec<(String, f32)>,
    /// The latest read-back frame: (sequence number, unsmoothed total in milliseconds).
    latest: Option<(u64, f32)>,
    /// Per label, its minimum over the frames read back since [`GpuProfiler::reset_minima`];
    /// then the minimum frame total and the number of frames.
    minima: Vec<(String, f32)>,
    frame_minimum: Option<f32>,
    minima_frames: u32,
    pub enabled: bool,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (query_set, resolve, readback) = if supported {
            let bytes = CAPACITY as u64 * 8;
            (
                Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: CAPACITY,
                })),
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp resolve"),
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })),
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })),
            )
        } else {
            (None, None, None)
        };
        Self {
            query_set,
            resolve,
            readback,
            period_ns: queue.get_timestamp_period(),
            labels: Vec::new(),
            in_flight: Vec::new(),
            status: Arc::new(AtomicU32::new(IDLE)),
            results: Vec::new(),
            latest: None,
            minima: Vec::new(),
            frame_minimum: None,
            minima_frames: 0,
            enabled: supported,
        }
    }

    pub fn supported(&self) -> bool {
        self.query_set.is_some()
    }

    /// Smoothed GPU time per pass label, in milliseconds, in first-recorded order.
    pub fn results(&self) -> &[(String, f32)] {
        &self.results
    }

    pub fn total_ms(&self) -> f32 {
        self.results.iter().map(|(_, ms)| ms).sum()
    }

    /// The most recently read-back frame's GPU time (unsmoothed, the sum of its passes) with a
    /// sequence number that changes with every read-back, for controllers that average
    /// themselves (dynamic resolution).
    pub fn latest_frame(&self) -> Option<(u64, f32)> {
        self.latest
    }

    /// Per-pass minima (first-recorded order), the minimum frame total and the frame count since
    /// [`GpuProfiler::reset_minima`]. On a GPU shared with other work a pass is sometimes
    /// preempted or slowed; its minimum over many frames is its time on a quiet GPU (headless
    /// measurements; the scene's motion varies the work a little from frame to frame).
    pub fn minima(&self) -> (&[(String, f32)], Option<f32>, u32) {
        (&self.minima, self.frame_minimum, self.minima_frames)
    }

    pub fn reset_minima(&mut self) {
        self.minima.clear();
        self.frame_minimum = None;
        self.minima_frames = 0;
    }

    fn reserve(&mut self, label: &str) -> Option<u32> {
        let index = self.labels.len() as u32 * 2;
        if !self.enabled || self.query_set.is_none() || index + 2 > CAPACITY {
            return None;
        }
        self.labels.push(label.to_owned());
        Some(index)
    }

    fn begin_frame(&mut self) {
        self.labels.clear();
    }

    fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let (Some(set), Some(resolve), Some(readback)) = (&self.query_set, &self.resolve, &self.readback) else {
            return false;
        };
        let count = self.labels.len() as u32 * 2;
        if count == 0 || self.status.load(Ordering::Acquire) != IDLE {
            return false;
        }
        encoder.resolve_query_set(set, 0..count, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, Some(count as u64 * 8));
        self.in_flight = std::mem::take(&mut self.labels);
        true
    }

    fn map(&self) {
        let Some(readback) = &self.readback else { return };
        self.status.store(MAPPING, Ordering::Release);
        let status = self.status.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            status.store(if r.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    fn poll(&mut self) {
        if self.status.load(Ordering::Acquire) != READY {
            return;
        }
        let Some(readback) = &self.readback else { return };
        let mut frame: Vec<(String, f32)> = Vec::new();
        if let Ok(view) = readback.get_mapped_range(..) {
            let stamps: &[u64] = bytemuck::cast_slice(&view);
            for (i, label) in self.in_flight.iter().enumerate() {
                let (begin, end) = (stamps[i * 2], stamps[i * 2 + 1]);
                let ms = end.saturating_sub(begin) as f64 * self.period_ns as f64 / 1e6;
                match frame.iter_mut().find(|(l, _)| l == label) {
                    Some(entry) => entry.1 += ms as f32,
                    None => frame.push((label.clone(), ms as f32)),
                }
            }
        }
        readback.unmap();
        self.status.store(IDLE, Ordering::Release);
        let sequence = self.latest.map_or(0, |(n, _)| n + 1);
        let total: f32 = frame.iter().map(|(_, ms)| ms).sum();
        self.latest = Some((sequence, total));
        self.frame_minimum = Some(self.frame_minimum.map_or(total, |m| m.min(total)));
        self.minima_frames += 1;
        for (label, ms) in &frame {
            match self.minima.iter_mut().find(|(l, _)| l == label) {
                Some(entry) => entry.1 = entry.1.min(*ms),
                None => self.minima.push((label.clone(), *ms)),
            }
        }
        // Exponential smoothing; labels that disappear fade out.
        for (label, ms) in &frame {
            match self.results.iter_mut().find(|(l, _)| l == label) {
                Some(entry) => entry.1 = entry.1 * 0.8 + ms * 0.2,
                None => self.results.push((label.clone(), *ms)),
            }
        }
        self.results.retain_mut(|(label, ms)| {
            if !frame.iter().any(|(l, _)| l == label) {
                *ms *= 0.8;
            }
            *ms > 0.0005
        });
    }
}

/// Command recording context handed to every pass.
pub struct Recorder<'a> {
    pub device: &'a wgpu::Device,
    pub encoder: wgpu::CommandEncoder,
    profiler: &'a mut GpuProfiler,
}

impl<'a> Recorder<'a> {
    pub fn new(device: &'a wgpu::Device, profiler: &'a mut GpuProfiler, label: &str) -> Self {
        profiler.poll();
        profiler.begin_frame();
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        Self { device, encoder, profiler }
    }

    /// Begins a timed compute pass.
    pub fn compute_pass(&mut self, label: &str) -> wgpu::ComputePass<'_> {
        let index = self.profiler.reserve(label);
        let writes = index.and_then(|i| {
            self.profiler.query_set.as_ref().map(|set| wgpu::ComputePassTimestampWrites {
                query_set: set,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            })
        });
        self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: writes })
    }

    /// Begins a timed render pass.
    pub fn render_pass(
        &mut self,
        label: &str,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment>],
        depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment>,
    ) -> wgpu::RenderPass<'_> {
        let index = self.profiler.reserve(label);
        let writes = index.and_then(|i| {
            self.profiler.query_set.as_ref().map(|set| wgpu::RenderPassTimestampWrites {
                query_set: set,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            })
        });
        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes: writes,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Resolves timestamps and returns the finished command buffer plus whether a profiler
    /// read-back must be mapped after submission (see [`Recorder::after_submit`]).
    pub fn finish(mut self) -> (wgpu::CommandBuffer, bool) {
        let map = self.profiler.end_frame(&mut self.encoder);
        (self.encoder.finish(), map)
    }
}

impl GpuProfiler {
    /// Call after submitting a command buffer whose [`Recorder::finish`] returned true.
    pub fn after_submit(&self, map: bool) {
        if map {
            self.map();
        }
    }
}
