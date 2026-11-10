//! Generic GPU page cache (see `shaders/page_cache.wgsl`): virtual pages requested by a
//! client are mapped to physical pages, cached across frames and evicted LRU-style. When the
//! pool cannot hold a frame's requests, they are served by priority (the client's buckets).

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use crate::profiler::Recorder;
use crate::shader::create_module;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline, storage_buffer, uniform_buffer, workgroups};

pub const INVALID: u32 = u32::MAX;
const HEADER_WORDS: u64 = 16;
/// Priority buckets (`BUCKETS` in page_cache.wgsl): per bucket four words after the lists.
const BUCKETS: u64 = 256;
/// Byte offset of the dirty-page dispatch arguments inside the lists buffer.
pub const DIRTY_ARGS_OFFSET: u64 = 8 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CacheParams {
    pub entry_count: u32,
    pub physical_count: u32,
    pub frame: u32,
    pub retain: u32,
    pub invalidate: u32,
    pub groups_per_page: u32,
    pub level_pages: u32,
    pub level_count: u32,
    pub origins: [[i32; 4]; 16],
    pub pinned: u32,
    /// Clipmap client: the first local light page entry (`INVALID`: none).
    pub local_base: u32,
    pub _pad: [u32; 2],
    /// Clipmap client: per local light slot its priority rank (0 = most influential on
    /// screen), which orders its pages when the pool is full (`bucket` in page_cache.wgsl).
    pub ranks: [u32; 32],
}

const _: () = assert!(std::mem::size_of::<CacheParams>() == 432);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheClient {
    Static = 0,
    Clipmap = 1,
}

pub struct PageCache {
    pub entries: wgpu::Buffer,
    pub requests: wgpu::Buffer,
    pub owner: wgpu::Buffer,
    /// Header, free list, allocation queue and dirty list.
    pub lists: wgpu::Buffer,
    pub params: wgpu::Buffer,
    /// Dispatch arguments of the reassignment (empty unless the pool is full), written by the
    /// passes that bind it; the others bind `no_args` in its place (a buffer cannot be both a
    /// dispatch's storage and its indirect arguments).
    reassign_args: wgpu::Buffer,
    no_args: wgpu::Buffer,
    pub entry_count: u32,
    pub physical_count: u32,
    layout: wgpu::BindGroupLayout,
    begin: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    free_list: wgpu::ComputePipeline,
    plan: wgpu::ComputePipeline,
    reassign: wgpu::ComputePipeline,
    allocate: wgpu::ComputePipeline,
    args: wgpu::ComputePipeline,
    label: &'static str,
}

impl PageCache {
    pub fn new(device: &wgpu::Device, label: &'static str, client: CacheClient, entry_count: u32, physical_count: u32) -> Self {
        let module = create_module(device, "page_cache.wgsl");
        let layout = bind_group_layout(
            device,
            label,
            ShaderStages::COMPUTE,
            &[
                (0, bind::uniform()),
                (1, bind::storage(false)),
                (2, bind::storage(false)),
                (3, bind::storage(false)),
                (4, bind::storage(false)),
                (5, bind::storage(false)),
            ],
        );
        let constants = [("CLIENT", client as u32 as f64)];
        let pipeline = |entry: &str| compute_pipeline(device, label, &module, entry, &layout, &constants);
        let entries = storage_buffer(device, label, entry_count as u64 * 16, wgpu::BufferUsages::empty());
        let owner = storage_buffer(device, label, physical_count as u64 * 4, wgpu::BufferUsages::empty());
        Self {
            requests: storage_buffer(device, label, entry_count as u64 * 4, wgpu::BufferUsages::empty()),
            // COPY_SRC: the header is read back for statistics.
            lists: storage_buffer(device, label, (HEADER_WORDS + 3 * physical_count as u64 + 4 * BUCKETS) * 4, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC),
            params: uniform_buffer(device, label, std::mem::size_of::<CacheParams>() as u64),
            reassign_args: storage_buffer(device, label, 16, wgpu::BufferUsages::INDIRECT),
            no_args: storage_buffer(device, label, 16, wgpu::BufferUsages::empty()),
            begin: pipeline("cache_begin"),
            update: pipeline("cache_update"),
            free_list: pipeline("cache_free_list"),
            plan: pipeline("cache_plan"),
            reassign: pipeline("cache_reassign"),
            allocate: pipeline("cache_allocate"),
            args: pipeline("cache_args"),
            layout,
            entries,
            owner,
            entry_count,
            physical_count,
            label,
        }
    }

    /// Marks every entry and physical page as empty (call once after creation, and to flush).
    pub fn clear(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.entries, 0, &vec![0xffu8; self.entry_count as usize * 16]);
        queue.write_buffer(&self.owner, 0, &vec![0xffu8; self.physical_count as usize * 4]);
    }

    pub fn write_params(&self, queue: &wgpu::Queue, params: &CacheParams) {
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(params));
    }

    pub fn default_params(&self, frame: u32, retain: u32, groups_per_page: u32) -> CacheParams {
        CacheParams {
            entry_count: self.entry_count,
            physical_count: self.physical_count,
            frame,
            retain,
            invalidate: 0,
            groups_per_page,
            level_pages: 0,
            level_count: 0,
            origins: [[0; 4]; 16],
            pinned: INVALID,
            local_base: INVALID,
            _pad: [0; 2],
            ranks: [0; 32],
        }
    }

    /// Records the allocation passes. The client's request marking must precede this and its
    /// page rendering (dispatched indirectly from [`DIRTY_ARGS_OFFSET`]) follow it.
    pub fn record(&self, rec: &mut Recorder) {
        let group = |args: &wgpu::Buffer| {
            bind_group(
                rec.device,
                self.label,
                &self.layout,
                &[
                    (0, self.params.as_entire_binding()),
                    (1, self.entries.as_entire_binding()),
                    (2, self.requests.as_entire_binding()),
                    (3, self.owner.as_entire_binding()),
                    (4, self.lists.as_entire_binding()),
                    (5, args.as_entire_binding()),
                ],
            )
        };
        let (with_args, without_args) = (group(&self.reassign_args), group(&self.no_args));
        let entry_groups = workgroups(self.entry_count, 64);
        let mut pass = rec.compute_pass(self.label);
        pass.set_bind_group(0, &with_args, &[]);
        pass.set_pipeline(&self.begin);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.update);
        pass.dispatch_workgroups(entry_groups.min(65535), entry_groups.div_ceil(65535), 1);
        pass.set_pipeline(&self.plan);
        pass.dispatch_workgroups(1, 1, 1);
        // Dispatched only while the pool is full.
        pass.set_bind_group(0, &without_args, &[]);
        pass.set_pipeline(&self.reassign);
        pass.dispatch_workgroups_indirect(&self.reassign_args, 0);
        pass.set_pipeline(&self.free_list);
        pass.dispatch_workgroups(workgroups(self.physical_count, 64), 1, 1);
        pass.set_pipeline(&self.allocate);
        pass.dispatch_workgroups(workgroups(self.physical_count, 64), 1, 1);
        pass.set_pipeline(&self.args);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// Word index where the dirty list starts, given the physical page count.
pub fn dirty_list_start(physical_count: u32) -> u32 {
    HEADER_WORDS as u32 + 2 * physical_count
}
