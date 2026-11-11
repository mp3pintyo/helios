//! Asynchronous read-back of GPU culling counters (never stalls the pipeline).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Per-frame statistics gathered from the GPU (a few frames old).
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    pub instances: u32,
    /// Pages that survived page culling for the main view with a cluster workgroup each (small
    /// pages list their clusters as candidates instead, cull.wgsl; beyond the candidates'
    /// capacity they come back here, so this is what can overflow).
    pub pages: u32,
    pub clusters: u32,
    pub triangles: u32,
    /// Clusters / triangles drawn by the software rasteriser.
    pub sw_clusters: u32,
    pub sw_triangles: u32,
    /// Hardware clusters drawn by the alpha-masked (alpha-testing) pipeline.
    pub masked_clusters: u32,
    /// Voxels of the voxel clusters drawn (main view and shadow views).
    pub voxels: u32,
    pub shadow_voxels: u32,
    /// Assembly part instances drawn instead of their assemblies' bases, and those requested
    /// (beyond the part slots, assemblies keep their bases).
    pub assembly_parts: u32,
    pub assembly_parts_requested: u32,
    pub shadow_clusters: u32,
    pub shadow_triangles: u32,
    /// Shadow triangles drawn by the software rasteriser.
    pub shadow_sw_triangles: u32,
    /// Clusters drawn in the early (previously visible) occlusion pass.
    pub early_clusters: u32,
    /// Work items dropped because a main-view list was full.
    pub overflow: u32,
    /// Physical shadow map pages in use (sun and local lights) and in the atlas.
    pub shadow_pages: u32,
    pub shadow_page_capacity: u32,
    /// Shadow pages rendered in the frame (newly needed or invalidated).
    pub shadow_dirty_pages: u32,
    /// Of those, pages that only re-rendered their dynamic (wind-animated) casters over their
    /// cached static layer, and pages rendered in full to start one.
    pub shadow_dynamic_pages: u32,
    pub shadow_split_pages: u32,
    /// Shadow pages requested in the frame (sun and local lights, resident or not), and those
    /// that got no physical page.
    pub shadow_pages_requested: u32,
    pub shadow_page_overflow: u32,
    /// (cluster, page) draws of the rendered shadow pages; beyond the capacity they are lost.
    pub shadow_draws: u32,
    pub shadow_draw_capacity: u32,
    /// The shadow refresh queue (vsm.wgsl): pages whose refresh the budget deferred, pages of
    /// distant levels waiting for their turn, and pages refreshed past the budget because they
    /// had waited too long.
    pub shadow_refresh_deferred: u32,
    pub shadow_refresh_held: u32,
    pub shadow_refresh_forced: u32,
}

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

/// Each view's cull-state buffer is 64 bytes (see `CullState` in cull.wgsl).
pub const STATE_SIZE: u64 = 64;

pub struct StatsReadback {
    buffer: wgpu::Buffer,
    views: usize,
    /// Views copied by the read-back in flight.
    recorded: usize,
    shadow_page_capacity: u32,
    shadow_draw_capacity: u32,
    status: Arc<AtomicU32>,
    pub latest: RenderStats,
}

impl StatsReadback {
    /// `views` cull-state buffers, then the headers (64 bytes each) of the shadow page cache
    /// (page_cache.wgsl) with `shadow_page_capacity` physical pages and of the shadow page draws
    /// (vsm_expand.wgsl) with room for `shadow_draw_capacity`, then the assembly part counter.
    pub fn new(device: &wgpu::Device, views: usize, shadow_page_capacity: u32, shadow_draw_capacity: u32) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stats readback"),
                size: STATE_SIZE * (views as u64 + 3),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            views,
            recorded: 0,
            shadow_page_capacity,
            shadow_draw_capacity,
            status: Arc::new(AtomicU32::new(IDLE)),
            latest: RenderStats::default(),
        }
    }

    /// Copies the state buffers (the main view's first), the shadow headers (page cache lists,
    /// page draws) and the assembly part counter (if the scene has assemblies) if the read-back
    /// buffer is free. Returns true if a copy was recorded.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        states: &[&wgpu::Buffer],
        shadow_headers: [&wgpu::Buffer; 2],
        assembly_parts: Option<&wgpu::Buffer>,
    ) -> bool {
        if self.status.load(Ordering::Acquire) != IDLE {
            return false;
        }
        self.recorded = states.len().min(self.views);
        for (i, state) in states[..self.recorded].iter().enumerate() {
            encoder.copy_buffer_to_buffer(state, 0, &self.buffer, i as u64 * STATE_SIZE, Some(STATE_SIZE));
        }
        for (i, header) in shadow_headers.into_iter().enumerate() {
            encoder.copy_buffer_to_buffer(header, 0, &self.buffer, (self.views + i) as u64 * STATE_SIZE, Some(STATE_SIZE));
        }
        let parts_offset = (self.views + 2) as u64 * STATE_SIZE;
        match assembly_parts {
            Some(counter) => encoder.copy_buffer_to_buffer(counter, 0, &self.buffer, parts_offset, Some(4)),
            None => encoder.clear_buffer(&self.buffer, parts_offset, Some(4)),
        }
        true
    }

    /// Call after submitting a frame in which [`StatsReadback::record`] returned true.
    pub fn map(&self) {
        self.status.store(MAPPING, Ordering::Release);
        let status = self.status.clone();
        self.buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    /// Consumes a completed read-back, if any. Returns true if `latest` was updated.
    pub fn poll(&mut self, instances: u32) -> bool {
        if self.status.load(Ordering::Acquire) != READY {
            return false;
        }
        if let Ok(view) = self.buffer.get_mapped_range(..) {
            let words: &[u32] = bytemuck::cast_slice(&view);
            let view_word = |v: usize, i: usize| words[v * (STATE_SIZE as usize / 4) + i];
            let mut stats = RenderStats {
                instances,
                pages: view_word(0, 1),
                clusters: view_word(0, 2) + view_word(0, 3),
                sw_clusters: view_word(0, 3),
                early_clusters: view_word(0, 4) + view_word(0, 5),
                triangles: view_word(0, 6),
                sw_triangles: view_word(0, 7),
                overflow: view_word(0, 8),
                masked_clusters: view_word(0, 9),
                voxels: view_word(0, 11),
                ..Default::default()
            };
            for v in 1..self.recorded {
                stats.shadow_clusters += view_word(v, 2) + view_word(v, 3);
                stats.shadow_triangles += view_word(v, 6);
                stats.shadow_sw_triangles += view_word(v, 7);
                stats.shadow_voxels += view_word(v, 11);
            }
            // Page cache header: [0] entries queued for allocation, [1] free pages before
            // allocating, [2] dirty pages, [4] requests without a page, [5] requests.
            let header = |i: usize| view_word(self.views, i);
            let free = header(1).min(self.shadow_page_capacity);
            stats.shadow_page_capacity = self.shadow_page_capacity;
            stats.shadow_pages = self.shadow_page_capacity - free + header(0).min(free);
            stats.shadow_dirty_pages = header(2);
            stats.shadow_page_overflow = header(4);
            stats.shadow_pages_requested = header(5);
            stats.shadow_draws = view_word(self.views + 1, 12);
            stats.shadow_dynamic_pages = view_word(self.views + 1, 13);
            stats.shadow_split_pages = view_word(self.views + 1, 14);
            stats.shadow_refresh_deferred = view_word(self.views + 1, 6);
            stats.shadow_refresh_held = view_word(self.views + 1, 7);
            stats.shadow_refresh_forced = view_word(self.views + 1, 11);
            stats.shadow_draw_capacity = self.shadow_draw_capacity;
            stats.assembly_parts_requested = view_word(self.views + 2, 0);
            stats.assembly_parts = stats.assembly_parts_requested.min(crate::assembly::PART_CAPACITY);
            self.latest = stats;
        }
        self.buffer.unmap();
        self.status.store(IDLE, Ordering::Release);
        true
    }
}
