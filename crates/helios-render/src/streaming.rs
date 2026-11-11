//! Virtual geometry streaming: GPU feedback → CPU residency decisions → asynchronous page
//! reads → page uploads.
//!
//! Every frame the culling shaders stamp `page_feedback[page] = frame` for each page whose
//! clusters a resident parent cluster *wants* to refine into. Every few frames that buffer is
//! read back asynchronously (never stalling). The streamer then
//!
//! * requests needed, non-resident pages (coarse first, pulling in missing dependency pages),
//! * uploads arrived pages whose dependency pages are resident, within a per-frame budget, and
//! * evicts least-recently-needed pages without resident dependents when the pool is full.
//!
//! The GPU only sees the result through the page table (`page -> pool word offset`), so a
//! missing page simply means its parents keep being drawn — the cut stays watertight.
//!
//! ## Page life cycle
//!
//! ```text
//! Absent ──request──▶ Requested ──read completes──▶ Arrived ──dependencies resident,──▶ Resident
//!   ▲  (read failed: retry later) │                    │       pool space, budget        │
//!   └─────────────────────────────┴── not needed for ──┘                                 │
//!   └──────────────────────────────── a while ─────────────────────── evicted ◀──────────┘
//! ```
//!
//! Pages of meshes read from a `.hvm` with entropy-coded pages (version 7) pass through
//! **Decoding** between Requested and Arrived: a completed read goes to the decoders
//! (`page_decode.rs`: decode threads natively, the bake workers in the browser), and its pages
//! arrive once their payload words are back, usually a frame or two later.
//!
//! Pages of in-memory meshes go straight from Absent to Arrived. Pages are requested together
//! with their dependencies, so a whole chain of LOD levels is fetched in parallel; an arrived
//! page then waits on the CPU until its dependencies are resident. That is the **residency
//! invariant** the crack-free LOD cut relies on (see `helios_geometry::dag`): a page becomes
//! resident only when all its dependency pages are, and a page is evicted only when no resident
//! page depends on it. Requested and arrived pages do not pin their dependencies — if one is
//! evicted meanwhile, the dependent just waits (and its dependency is requested again).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use helios_geometry::VirtualMesh;

use crate::page_decode::{DecodeJob, Decoded, Decoders};
use crate::page_source::{coalesce, PageLocation, PageRead, PageSource, ReadCompletion};

pub const NOT_RESIDENT: u32 = u32::MAX;

/// Pages must stay unneeded for this many frames before they may be evicted (or, once
/// arrived, dropped from the CPU).
const RETAIN_FRAMES: u32 = 90;
/// Read the feedback buffer back every this many frames.
const FEEDBACK_INTERVAL: u32 = 4;
/// Upload budget per frame.
const UPLOAD_BUDGET_BYTES: usize = 24 << 20;
/// Largest coalesced read.
const MAX_READ_BYTES: u64 = 1 << 20;
/// Bound on payload bytes requested or arrived but not yet uploaded.
const MAX_STAGED_BYTES: usize = 32 << 20;
/// Frames before pages of a failed read are requested again.
const RETRY_FRAMES: u32 = 60;

/// First-fit allocator over a range of `u32` words with coalescing frees.
#[derive(Debug)]
pub struct RangeAllocator {
    free: Vec<(u32, u32)>, // (start, len), sorted by start
    pub capacity: u32,
}

impl RangeAllocator {
    pub fn new(capacity: u32) -> Self {
        Self { free: vec![(0, capacity)], capacity }
    }

    pub fn allocate(&mut self, len: u32) -> Option<u32> {
        let i = self.free.iter().position(|&(_, l)| l >= len)?;
        let (start, l) = self.free[i];
        if l == len {
            self.free.remove(i);
        } else {
            self.free[i] = (start + len, l - len);
        }
        Some(start)
    }

    pub fn release(&mut self, start: u32, len: u32) {
        let i = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(i, (start, len));
        // Merge with the following and preceding ranges.
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }

    pub fn free_words(&self) -> u32 {
        self.free.iter().map(|&(_, l)| l).sum()
    }

    pub fn used_words(&self) -> u32 {
        self.capacity - self.free_words()
    }
}

/// Where a page's payload comes from.
#[derive(Clone, Copy, Debug)]
enum Origin {
    /// Held by the mesh in memory.
    Memory,
    /// Byte `offset` of file `file` of source `source`.
    Stream { source: u32, file: u32, offset: u64 },
    /// Neither in memory nor locatable (a header-only mesh added without a source).
    Missing,
}

/// Where a global page lives in the scene.
#[derive(Clone, Copy, Debug)]
struct PageRef {
    mesh: u32,
    local: u32,
    words: u32,
    /// Bytes of the stored page in its file (its words, or its coded bytes when `coded`).
    stored: u32,
    coded: bool,
    root: bool,
    level: u32,
    origin: Origin,
}

impl PageRef {
    fn bytes(&self) -> usize {
        self.words as usize * 4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageState {
    Absent,
    /// Part of an outstanding read.
    Requested,
    /// Read; its coded payload is being decoded.
    Decoding,
    /// Payload on the CPU, waiting for its dependencies, pool space or upload budget.
    Arrived,
    Resident,
}

/// The payload of an arrived page.
enum Staged {
    /// Its bytes at `start` of a read's buffer (shared by all pages of a coalesced read, so
    /// arrival copies nothing).
    Read { buffer: Arc<Vec<u8>>, start: usize },
    /// Its decoded words.
    Decoded(Vec<u32>),
}

/// An outstanding read of consecutive pages of one file.
struct Read {
    source: u32,
    /// Byte range start; `pages` are at their own offsets inside it.
    start: u64,
    len: u64,
    pages: Vec<u32>,
}

struct SourceSlot {
    source: Arc<dyn PageSource>,
    in_flight: usize,
}

/// Streaming statistics for the HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamingStats {
    pub resident_pages: u32,
    pub total_pages: u32,
    pub resident_mb: f32,
    pub pool_mb: f32,
    /// Pages made resident / evicted in the last update.
    pub loads: u32,
    pub evictions: u32,
    /// Needed pages that are not resident yet (being read, waiting, or out of budget / pool).
    pub pending: u32,
    /// Pages being read from a [`PageSource`].
    pub in_flight: u32,
    /// Pages being decoded (read from a `.hvm` with entropy-coded pages).
    pub decoding: u32,
    /// Arrived pages waiting for their dependencies, pool space or upload budget.
    pub waiting: u32,
    /// CPU memory of requested and waiting page payloads.
    pub staged_mb: f32,
    /// Total payload received from page sources (stored bytes: coded pages are smaller).
    pub streamed_mb: f32,
    /// Total payload of the streamed pages once decoded (as resident in the pool).
    pub decoded_mb: f32,
    /// Total failed reads (their pages are requested again later).
    pub failed_reads: u32,
}

impl StreamingStats {
    /// Residency changed in the last update (cached shadow pages may be stale).
    pub fn changed(&self) -> bool {
        self.loads > 0 || self.evictions > 0
    }
}

/// The streaming state machine, independent of the GPU: page states, the pool allocator and
/// outstanding reads. Uploads go through a callback `(pool word offset, payload bytes)`.
pub(crate) struct Residency {
    pages: Vec<PageRef>,
    /// Global page ids of each page's dependencies.
    dependencies: Vec<Vec<u32>>,
    dependents_resident: Vec<u32>,
    state: Vec<PageState>,
    /// Global page -> pool word offset, or [`NOT_RESIDENT`] (what the GPU sees).
    page_table: Vec<u32>,
    last_needed: Vec<u32>,
    /// Earliest frame an absent page may be requested again (after a failed read).
    retry_at: Vec<u32>,
    allocator: RangeAllocator,
    sources: Vec<SourceSlot>,
    reads: HashMap<u64, Read>,
    next_ticket: u64,
    /// Arrived payloads of streamed pages.
    staged: HashMap<u32, Staged>,
    /// Payload bytes of requested pages plus `staged`.
    staged_bytes: usize,
    /// Pages the last feedback asked for (with their dependencies), coarse first.
    wanted: Vec<u32>,
    /// Pages in the `Arrived` state.
    arrived: Vec<u32>,
    /// Streamed pages to fetch while the pool has room, coarse first.
    prefetch: Vec<u32>,
    completions: Vec<ReadCompletion>,
    /// Where coded pages are decoded, and the pages of each outstanding decode by ticket.
    decoders: Decoders,
    decodes: HashMap<u64, Vec<u32>>,
    decoded: Vec<Decoded>,
    page_table_dirty: bool,
    stats: StreamingStats,
}

type Upload<'a> = dyn FnMut(u32, &[u8]) + 'a;

impl Residency {
    pub fn new(pool_words: u32) -> Self {
        Self {
            pages: Vec::new(),
            dependencies: Vec::new(),
            dependents_resident: Vec::new(),
            state: Vec::new(),
            page_table: Vec::new(),
            last_needed: Vec::new(),
            retry_at: Vec::new(),
            allocator: RangeAllocator::new(pool_words),
            sources: Vec::new(),
            reads: HashMap::new(),
            next_ticket: 0,
            staged: HashMap::new(),
            staged_bytes: 0,
            wanted: Vec::new(),
            arrived: Vec::new(),
            prefetch: Vec::new(),
            completions: Vec::new(),
            decoders: Decoders::default(),
            decodes: HashMap::new(),
            decoded: Vec::new(),
            page_table_dirty: true,
            stats: StreamingStats::default(),
        }
    }

    /// Registers all pages of `meshes` (global ids in mesh order; `locations[m]` says where a
    /// streamed mesh's payloads live), makes the root pages resident and prefetches
    /// coarse-to-fine while the pool has room. Outstanding reads of a previous scene are
    /// abandoned. Returns false if the roots alone do not fit in the pool.
    pub fn reset(&mut self, meshes: &[VirtualMesh], locations: &[Option<PageLocation>], upload: &mut Upload) -> bool {
        let capacity = self.allocator.capacity;
        // Tickets keep counting, so reads and decodes of the previous scene are recognised
        // (and dropped) when they complete; the decoders (and their threads) stay.
        let decoders = std::mem::take(&mut self.decoders);
        *self = Self { next_ticket: self.next_ticket, decoders, ..Self::new(capacity) };
        for (m, mesh) in meshes.iter().enumerate() {
            let base = self.pages.len() as u32;
            let location = locations.get(m).and_then(Option::as_ref);
            let stream = location.map(|l| (self.source_index(&l.source), l.file));
            for (p, page) in mesh.pages.iter().enumerate() {
                let range = mesh.page_byte_range(p);
                let origin = match stream {
                    _ if mesh.has_payload(p) => Origin::Memory,
                    Some((source, file)) => Origin::Stream { source, file, offset: range.start },
                    None => Origin::Missing,
                };
                let (stored, coded) = ((range.end - range.start) as u32, mesh.is_coded());
                self.pages.push(PageRef { mesh: m as u32, local: p as u32, words: page.data_words, stored, coded, root: page.root != 0, level: page.lod_level, origin });
                self.dependencies.push(mesh.page_dependencies(p).iter().map(|&d| d + base).collect());
            }
        }
        let n = self.pages.len();
        self.state = vec![PageState::Absent; n];
        self.page_table = vec![NOT_RESIDENT; n];
        self.dependents_resident = vec![0; n];
        self.last_needed = vec![0; n];
        self.retry_at = vec![0; n];
        self.stats = StreamingStats { total_pages: n as u32, pool_mb: capacity as f32 * 4.0 / 1048576.0, ..Default::default() };
        if self.pages.iter().any(|p| matches!(p.origin, Origin::Missing)) {
            log::error!("streamed meshes without a page source: some pages can never load");
        }

        // Roots must always be resident; streamed meshes normally keep theirs in memory (if
        // not, they are fetched first and the mesh appears once they arrive).
        let mut ok = true;
        let mut planned = vec![false; n];
        let roots: Vec<usize> = (0..n).filter(|&p| self.pages[p].root).collect();
        for p in roots {
            match self.pages[p].origin {
                Origin::Memory => ok &= self.load_now(meshes, p, upload),
                Origin::Stream { .. } => {
                    planned[p] = true;
                    self.prefetch.push(p as u32);
                }
                Origin::Missing => {}
            }
        }
        // Prefetch coarse-to-fine while the pool has room, so the first frames do not start
        // from root clusters only: in-memory pages right away, streamed pages asynchronously.
        // Streaming takes over once the pool is full.
        let budget = capacity / 4 * 3;
        let mut planned_words = 0;
        let mut order: Vec<usize> = (0..n).filter(|&p| !self.pages[p].root).collect();
        order.sort_by_key(|&p| std::cmp::Reverse(self.pages[p].level));
        for p in order {
            if self.allocator.used_words() + planned_words + self.pages[p].words > budget {
                break;
            }
            let deps_ok = self.dependencies[p].iter().all(|&d| self.state[d as usize] == PageState::Resident || planned[d as usize]);
            match self.pages[p].origin {
                Origin::Memory if deps_ok => {
                    self.load_now(meshes, p, upload);
                }
                Origin::Stream { .. } if deps_ok => {
                    planned[p] = true;
                    planned_words += self.pages[p].words;
                    self.prefetch.push(p as u32);
                }
                _ => {}
            }
        }
        self.page_table_dirty = true;
        ok
    }

    fn source_index(&mut self, source: &Arc<dyn PageSource>) -> u32 {
        let index = match self.sources.iter().position(|s| Arc::ptr_eq(&s.source, source)) {
            Some(i) => i,
            None => {
                self.sources.push(SourceSlot { source: source.clone(), in_flight: 0 });
                self.sources.len() - 1
            }
        };
        index as u32
    }

    pub fn page_table(&self) -> &[u32] {
        &self.page_table
    }

    pub fn stats(&self) -> StreamingStats {
        self.stats
    }

    /// One frame of streaming: collects completed reads, applies `feedback` (page feedback
    /// words read back from the GPU, stamped up to `feedback_frame`) if there is a new one,
    /// issues reads and uploads arrived pages.
    pub fn update(&mut self, meshes: &[VirtualMesh], frame: u32, feedback: Option<(&[u32], u32)>, upload: &mut Upload) {
        self.stats.loads = 0;
        self.stats.evictions = 0;
        self.receive(frame);
        if let Some((feedback, feedback_frame)) = feedback {
            self.apply_feedback(feedback, feedback_frame);
        }
        self.request(frame);
        self.upload_arrived(meshes, frame, upload);

        self.stats.pending = self.wanted.iter().filter(|&&p| self.state[p as usize] != PageState::Resident).count() as u32;
        self.stats.in_flight = self.reads.values().map(|r| r.pages.len() as u32).sum();
        self.stats.decoding = self.decodes.values().map(|pages| pages.len() as u32).sum();
        self.stats.waiting = self.arrived.len() as u32;
        self.stats.staged_mb = self.staged_bytes as f32 / 1048576.0;
    }

    fn receive(&mut self, frame: u32) {
        let mut completions = std::mem::take(&mut self.completions);
        for slot in &self.sources {
            slot.source.poll(&mut completions);
        }
        for completion in completions.drain(..) {
            // Unknown tickets belong to reads abandoned by a reset.
            let Some(read) = self.reads.remove(&completion.ticket) else { continue };
            self.sources[read.source as usize].in_flight -= 1;
            let bytes = match completion.result {
                Ok(bytes) if bytes.len() as u64 == read.len => bytes,
                result => {
                    let error = result.err().unwrap_or_else(|| "short read".into());
                    log::warn!("streaming: read of {} pages failed ({error}); retrying later", read.pages.len());
                    self.stats.failed_reads += 1;
                    for &p in &read.pages {
                        self.state[p as usize] = PageState::Absent;
                        self.retry_at[p as usize] = frame + RETRY_FRAMES;
                        self.staged_bytes -= self.pages[p as usize].bytes();
                    }
                    continue;
                }
            };
            self.stats.streamed_mb += bytes.len() as f32 / 1048576.0;
            let buffer = Arc::new(bytes);
            let mut coded = Vec::new();
            for &p in &read.pages {
                let page = self.pages[p as usize];
                let Origin::Stream { offset, .. } = page.origin else { unreachable!("only streamed pages are read") };
                let start = (offset - read.start) as usize;
                if page.coded {
                    coded.push((p, start..start + page.stored as usize));
                    self.state[p as usize] = PageState::Decoding;
                } else {
                    self.staged.insert(p, Staged::Read { buffer: buffer.clone(), start });
                    self.arrived(p);
                }
            }
            if !coded.is_empty() {
                let ticket = completion.ticket;
                self.decodes.insert(ticket, coded.iter().map(|c| c.0).collect());
                self.decoders.submit(DecodeJob { ticket, buffer, pages: coded });
            }
        }
        self.completions = completions;

        let mut decoded = std::mem::take(&mut self.decoded);
        self.decoders.poll(&mut decoded);
        for job in decoded.drain(..) {
            // Unknown tickets belong to decodes abandoned by a reset.
            if self.decodes.remove(&job.ticket).is_none() {
                continue;
            }
            for (p, result) in job.pages {
                let words = self.pages[p as usize].words as usize;
                match result {
                    Ok(payload) if payload.len() == words => {
                        self.stats.decoded_mb += words as f32 * 4.0 / 1048576.0;
                        self.staged.insert(p, Staged::Decoded(payload));
                        self.arrived(p);
                    }
                    result => {
                        let error = result.err().unwrap_or_else(|| "wrong size".into());
                        log::warn!("streaming: page {p} did not decode ({error}); retrying later");
                        self.stats.failed_reads += 1;
                        self.state[p as usize] = PageState::Absent;
                        self.retry_at[p as usize] = frame + RETRY_FRAMES;
                        self.staged_bytes -= self.pages[p as usize].bytes();
                    }
                }
            }
        }
        self.decoded = decoded;
    }

    fn arrived(&mut self, p: u32) {
        self.state[p as usize] = PageState::Arrived;
        self.arrived.push(p);
    }

    fn apply_feedback(&mut self, feedback: &[u32], feedback_frame: u32) {
        self.wanted.clear();
        let recent = feedback_frame.saturating_sub(2);
        for (p, &f) in feedback.iter().enumerate().take(self.pages.len()) {
            if f != 0 && f >= recent {
                self.last_needed[p] = self.last_needed[p].max(f);
                if self.state[p] != PageState::Resident {
                    self.wanted.push(p as u32);
                }
            }
        }
        // A wanted page may depend on sibling pages nobody asked for (its group's simplified
        // clusters were split across several next-level groups): pull those in too.
        let mut queued: std::collections::HashSet<u32> = self.wanted.iter().copied().collect();
        let mut i = 0;
        while i < self.wanted.len() {
            for &d in &self.dependencies[self.wanted[i] as usize] {
                // Dependencies of a needed page are needed too (also protects resident ones from eviction).
                self.last_needed[d as usize] = self.last_needed[d as usize].max(feedback_frame);
                if self.state[d as usize] != PageState::Resident && queued.insert(d) {
                    self.wanted.push(d);
                }
            }
            i += 1;
        }
        // Coarse pages first: finer pages depend on them.
        self.wanted.sort_by_key(|&p| std::cmp::Reverse(self.pages[p as usize].level));
    }

    /// Requests absent wanted pages (then prefetch pages while the pool has room) from their
    /// sources, coalescing adjacent pages and respecting each source's in-flight limit.
    fn request(&mut self, frame: u32) {
        let mut picked: Vec<Vec<u32>> = vec![Vec::new(); self.sources.len()];
        // Bytes each source can still take this frame, and the staging allowance.
        let mut room: Vec<u64> = self.sources.iter().map(|s| s.source.max_in_flight().saturating_sub(s.in_flight) as u64 * MAX_READ_BYTES).collect();
        let initial_staging = MAX_STAGED_BYTES.saturating_sub(self.staged_bytes);
        let mut staging = initial_staging;

        let wanted = std::mem::take(&mut self.wanted);
        for &p in &wanted {
            self.pick(p as usize, frame, &mut picked, &mut room, &mut staging);
        }
        self.wanted = wanted;

        if !self.prefetch.is_empty() {
            let budget = self.allocator.capacity as usize / 4 * 3 * 4;
            for i in 0..self.prefetch.len() {
                let p = self.prefetch[i] as usize;
                let used = self.allocator.used_words() as usize * 4 + self.staged_bytes + (initial_staging - staging);
                if (used > budget && !self.pages[p].root) || !self.pick(p, frame, &mut picked, &mut room, &mut staging) {
                    break;
                }
            }
        }

        for (source, pages) in picked.into_iter().enumerate() {
            self.issue(source as u32, pages);
        }
        // Prefetch pages stay queued until a read for them actually started.
        if !self.prefetch.is_empty() {
            let state = &self.state;
            self.prefetch.retain(|&p| state[p as usize] == PageState::Absent);
        }
    }

    /// Moves an absent page towards residency: in-memory pages arrive at once, streamed pages
    /// are added to their source's batch. Returns false if out of read budget.
    fn pick(&mut self, p: usize, frame: u32, picked: &mut [Vec<u32>], room: &mut [u64], staging: &mut usize) -> bool {
        if self.state[p] != PageState::Absent || self.retry_at[p] > frame {
            return true;
        }
        let page = self.pages[p];
        match page.origin {
            Origin::Memory => {
                self.state[p] = PageState::Arrived;
                self.arrived.push(p as u32);
            }
            Origin::Stream { source, .. } => {
                // Staging holds pages as they will be uploaded; reads move their stored bytes.
                let bytes = page.bytes();
                let source = source as usize;
                if bytes > *staging || room[source] == 0 {
                    return false;
                }
                *staging -= bytes;
                room[source] = room[source].saturating_sub(page.stored as u64);
                // Requested right away, so a page that is both wanted and prefetched is picked
                // once; `issue` returns it to Absent if its read cannot start this frame.
                self.state[p] = PageState::Requested;
                picked[source].push(p as u32);
            }
            Origin::Missing => return true,
        }
        // Freshly requested pages count as needed now, so they are not dropped while waiting.
        self.last_needed[p] = self.last_needed[p].max(frame);
        true
    }

    /// Coalesces `pages` (in priority order) into reads and starts the most urgent ones.
    fn issue(&mut self, source: u32, pages: Vec<u32>) {
        if pages.is_empty() {
            return;
        }
        let slot = &self.sources[source as usize];
        let free_slots = slot.source.max_in_flight().saturating_sub(slot.in_flight);
        let max_gap = slot.source.max_gap_bytes();
        let span = |p: u32| match self.pages[p as usize].origin {
            Origin::Stream { file, offset, .. } => (file, offset..offset + self.pages[p as usize].stored as u64),
            _ => unreachable!("only streamed pages are read"),
        };

        // Sort by file position, coalesce per file, then start runs in priority order (the
        // priority of a run is that of its most urgent page).
        let mut order: Vec<usize> = (0..pages.len()).collect();
        order.sort_by_key(|&i| {
            let (file, range) = span(pages[i]);
            (file, range.start)
        });
        let mut runs: Vec<(usize, u32, std::ops::Range<u64>, Vec<u32>)> = Vec::new();
        let mut i = 0;
        while i < order.len() {
            let file = span(pages[order[i]]).0;
            let end = i + order[i..].iter().take_while(|&&j| span(pages[j]).0 == file).count();
            let spans: Vec<_> = order[i..end].iter().map(|&j| span(pages[j]).1).collect();
            for run in coalesce(&spans, MAX_READ_BYTES, max_gap) {
                let members = &order[i + run.first..i + run.first + run.count];
                let priority = members.iter().copied().min().unwrap_or(0);
                runs.push((priority, file, run.range, members.iter().map(|&j| pages[j]).collect()));
            }
            i = end;
        }
        runs.sort_by_key(|r| r.0);

        for (i, (_, file, range, members)) in runs.into_iter().enumerate() {
            if i >= free_slots {
                for p in members {
                    self.state[p as usize] = PageState::Absent;
                }
                continue;
            }
            let ticket = self.next_ticket;
            self.next_ticket += 1;
            for &p in &members {
                self.staged_bytes += self.pages[p as usize].bytes();
            }
            let slot = &mut self.sources[source as usize];
            slot.in_flight += 1;
            slot.source.start(PageRead { ticket, file, range: range.clone() });
            self.reads.insert(ticket, Read { source, start: range.start, len: range.end - range.start, pages: members });
        }
    }

    /// Uploads arrived pages whose dependencies are resident, coarse first, within the frame's
    /// budget; drops arrived pages that have not been needed for a while.
    fn upload_arrived(&mut self, meshes: &[VirtualMesh], frame: u32, upload: &mut Upload) {
        if self.arrived.is_empty() {
            return;
        }
        let mut arrived = std::mem::take(&mut self.arrived);
        arrived.sort_by_key(|&p| std::cmp::Reverse(self.pages[p as usize].level));
        let mut budget = UPLOAD_BUDGET_BYTES;
        let mut pool_full = false;
        // Repeat while progress is made: an upload may unblock pages that depend on it.
        loop {
            let mut progress = false;
            for &p in &arrived {
                let p = p as usize;
                let bytes = self.pages[p].bytes();
                if self.state[p] != PageState::Arrived || bytes > budget || !self.dependencies_resident(p) {
                    continue;
                }
                let Some(offset) = self.allocate(p, frame, &mut pool_full) else { continue };
                self.make_resident(meshes, p, offset, frame, upload);
                budget -= bytes;
                self.stats.loads += 1;
                progress = true;
            }
            if !progress {
                break;
            }
        }
        for &p in &arrived {
            let p = p as usize;
            if self.state[p] == PageState::Arrived && frame.saturating_sub(self.last_needed[p]) > RETAIN_FRAMES {
                self.drop_staged(p);
                self.state[p] = PageState::Absent;
            }
        }
        arrived.retain(|&p| self.state[p as usize] == PageState::Arrived);
        self.arrived = arrived;
    }

    fn dependencies_resident(&self, p: usize) -> bool {
        self.dependencies[p].iter().all(|&d| self.state[d as usize] == PageState::Resident)
    }

    fn drop_staged(&mut self, p: usize) {
        if self.staged.remove(&(p as u32)).is_some() {
            self.staged_bytes -= self.pages[p].bytes();
        }
    }

    /// Pool space for page `p`, evicting if needed. After one failed eviction the pool counts
    /// as full for the rest of the frame (so a long queue does not rescan the pool per page).
    fn allocate(&mut self, p: usize, frame: u32, pool_full: &mut bool) -> Option<u32> {
        let words = self.pages[p].words.max(1);
        if let Some(offset) = self.allocator.allocate(words) {
            return Some(offset);
        }
        if !*pool_full && self.evict_for(words, frame, p) {
            if let Some(offset) = self.allocator.allocate(words) {
                return Some(offset);
            }
        }
        *pool_full = true;
        None
    }

    /// Synchronous load of an in-memory page whose dependencies are resident (reset only).
    fn load_now(&mut self, meshes: &[VirtualMesh], p: usize, upload: &mut Upload) -> bool {
        let mut pool_full = false;
        match self.allocate(p, 0, &mut pool_full) {
            Some(offset) => {
                self.make_resident(meshes, p, offset, 0, upload);
                true
            }
            None => false,
        }
    }

    fn make_resident(&mut self, meshes: &[VirtualMesh], p: usize, offset: u32, frame: u32, upload: &mut Upload) {
        debug_assert!(self.dependencies_resident(p), "residency invariant: dependencies first");
        let page = self.pages[p];
        match self.staged.remove(&(p as u32)) {
            Some(Staged::Read { buffer, start }) => {
                self.staged_bytes -= page.bytes();
                upload(offset, &buffer[start..start + page.bytes()]);
            }
            Some(Staged::Decoded(words)) => {
                self.staged_bytes -= page.bytes();
                upload(offset, bytemuck::cast_slice(&words));
            }
            None => {
                let payload = meshes[page.mesh as usize].page_payload(page.local as usize);
                if !payload.is_empty() {
                    upload(offset, bytemuck::cast_slice(payload));
                }
            }
        }
        self.state[p] = PageState::Resident;
        self.page_table[p] = offset;
        self.last_needed[p] = self.last_needed[p].max(frame);
        for &d in &self.dependencies[p] {
            self.dependents_resident[d as usize] += 1;
        }
        self.page_table_dirty = true;
        self.stats.resident_pages += 1;
        self.stats.resident_mb += page.words.max(1) as f32 * 4.0 / 1048576.0;
    }

    fn unload(&mut self, p: usize) {
        let words = self.pages[p].words.max(1);
        self.allocator.release(self.page_table[p], words);
        self.page_table[p] = NOT_RESIDENT;
        self.state[p] = PageState::Absent;
        for &d in &self.dependencies[p] {
            self.dependents_resident[d as usize] -= 1;
        }
        self.page_table_dirty = true;
        self.stats.resident_pages -= 1;
        self.stats.resident_mb -= words as f32 * 4.0 / 1048576.0;
        self.stats.evictions += 1;
    }

    /// Evicts least-recently-needed leaf pages until `words` could fit for page `for_page`.
    /// Its own dependencies are kept: they have no resident dependents yet, but evicting them
    /// would make `for_page` resident without them.
    fn evict_for(&mut self, words: u32, frame: u32, for_page: usize) -> bool {
        let mut candidates: Vec<(u32, usize)> = (0..self.pages.len())
            .filter(|&p| {
                self.state[p] == PageState::Resident
                    && !self.pages[p].root
                    && self.dependents_resident[p] == 0
                    && frame.saturating_sub(self.last_needed[p]) > RETAIN_FRAMES
                    && !self.dependencies[for_page].contains(&(p as u32))
            })
            .map(|p| (self.last_needed[p], p))
            .collect();
        candidates.sort_unstable();
        for (_, p) in candidates {
            if self.allocator.free_words() >= words * 2 {
                break;
            }
            // A page may have gained resident dependents since the list was built.
            if self.dependents_resident[p] == 0 {
                self.unload(p);
            }
        }
        self.allocator.free_words() >= words
    }
}

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

/// [`Residency`] driven by GPU feedback, uploading into the geometry pool.
pub struct Streamer {
    residency: Residency,
    readback: Option<wgpu::Buffer>,
    readback_frame: u32,
    status: Arc<AtomicU32>,
}

impl Streamer {
    pub fn new(pool_words: u32) -> Self {
        Self { residency: Residency::new(pool_words), readback: None, readback_frame: 0, status: Arc::new(AtomicU32::new(IDLE)) }
    }

    pub fn stats(&self) -> StreamingStats {
        self.residency.stats()
    }

    /// Writes the page table to `buffer` if residency changed.
    pub fn upload_page_table(&mut self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        if std::mem::take(&mut self.residency.page_table_dirty) && !self.residency.page_table().is_empty() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(self.residency.page_table()));
        }
    }

    /// Registers all pages of `meshes` and makes the root pages resident. Returns false if the
    /// roots alone do not fit in the pool.
    pub fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, geometry: &wgpu::Buffer, meshes: &[VirtualMesh], locations: &[Option<PageLocation>]) -> bool {
        let n: usize = meshes.iter().map(|m| m.pages.len()).sum();
        self.readback = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("page feedback readback"),
            size: (n.max(4) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.status.store(IDLE, Ordering::Release);
        self.residency.reset(meshes, locations, &mut |offset, bytes| queue.write_buffer(geometry, offset as u64 * 4, bytes))
    }

    /// Consumes a completed feedback read-back and streams pages. Call once per frame before
    /// recording.
    pub fn update(&mut self, queue: &wgpu::Queue, geometry: &wgpu::Buffer, meshes: &[VirtualMesh], frame: u32) {
        let mut upload = |offset: u32, bytes: &[u8]| queue.write_buffer(geometry, offset as u64 * 4, bytes);
        let ready = self.readback.as_ref().filter(|_| self.status.load(Ordering::Acquire) == READY);
        let Some(readback) = ready else {
            self.residency.update(meshes, frame, None, &mut upload);
            return;
        };
        match readback.get_mapped_range(..) {
            Ok(view) => self.residency.update(meshes, frame, Some((bytemuck::cast_slice(&view), self.readback_frame)), &mut upload),
            Err(_) => self.residency.update(meshes, frame, None, &mut upload),
        }
        readback.unmap();
        self.status.store(IDLE, Ordering::Release);
    }

    /// Records a copy of the GPU feedback buffer if the read-back buffer is free.
    pub fn record_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, feedback: &wgpu::Buffer, frame: u32) -> bool {
        if !frame.is_multiple_of(FEEDBACK_INTERVAL) || self.status.load(Ordering::Acquire) != IDLE {
            return false;
        }
        let Some(readback) = &self.readback else { return false };
        let bytes = (self.residency.pages.len() * 4) as u64;
        if bytes == 0 {
            return false;
        }
        encoder.copy_buffer_to_buffer(feedback, 0, readback, 0, Some(bytes));
        self.readback_frame = frame;
        true
    }

    /// Call after submitting a frame in which [`Streamer::record_feedback`] returned true.
    pub fn map_feedback(&self) {
        let Some(readback) = &self.readback else { return };
        self.status.store(MAPPING, Ordering::Release);
        let status = self.status.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use helios_geometry::{build_virtual_mesh, primitives, BuildConfig, SdfMode};

    use super::*;

    /// Streaming state (page sources included) must not make the renderer `!Send`.
    #[test]
    fn streaming_state_is_send() {
        fn is_send<T: Send>() {}
        is_send::<Streamer>();
        is_send::<crate::Scene>();
    }

    #[test]
    fn range_allocator_coalesces() {
        let mut a = RangeAllocator::new(100);
        let x = a.allocate(30).unwrap();
        let y = a.allocate(30).unwrap();
        let z = a.allocate(40).unwrap();
        assert!(a.allocate(1).is_none());
        a.release(y, 30);
        a.release(x, 30);
        assert_eq!(a.allocate(60), Some(0));
        a.release(z, 40);
        a.release(0, 60);
        assert_eq!(a.free_words(), 100);
        assert_eq!(a.allocate(100), Some(0));
    }

    /// A page source whose reads complete only when the test says so.
    struct ManualSource {
        file: Vec<u8>,
        max_in_flight: usize,
        started: Mutex<Vec<PageRead>>,
        ready: Mutex<Vec<ReadCompletion>>,
        bytes_read: Mutex<usize>,
    }

    impl ManualSource {
        fn new(file: Vec<u8>, max_in_flight: usize) -> Arc<Self> {
            Arc::new(Self { file, max_in_flight, started: Mutex::default(), ready: Mutex::default(), bytes_read: Mutex::default() })
        }

        /// Completes the oldest `count` outstanding reads; the ones listed in `fail` fail.
        fn complete(&self, count: usize, fail: &[u64]) -> usize {
            let mut started = self.started.lock().unwrap();
            let count = count.min(started.len());
            for read in started.drain(..count) {
                let result = if fail.contains(&read.ticket) { Err("injected failure".into()) } else { Ok(self.file[read.range.start as usize..read.range.end as usize].to_vec()) };
                self.ready.lock().unwrap().push(ReadCompletion { ticket: read.ticket, result });
            }
            count
        }

        fn outstanding(&self) -> Vec<PageRead> {
            self.started.lock().unwrap().clone()
        }

        fn bytes_read(&self) -> usize {
            *self.bytes_read.lock().unwrap()
        }
    }

    impl PageSource for ManualSource {
        fn start(&self, read: PageRead) {
            *self.bytes_read.lock().unwrap() += (read.range.end - read.range.start) as usize;
            self.started.lock().unwrap().push(read);
        }
        fn poll(&self, completed: &mut Vec<ReadCompletion>) {
            completed.append(&mut self.ready.lock().unwrap());
        }
        fn max_in_flight(&self) -> usize {
            self.max_in_flight
        }
    }

    /// A baked rock (in memory) and its `.hvm` bytes.
    fn rock() -> (VirtualMesh, Vec<u8>) {
        let vm = build_virtual_mesh(&primitives::rock(3, 5), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let bytes = vm.to_bytes();
        (vm, bytes)
    }

    /// The header-only mesh a streaming reader would load: header plus root tail.
    fn streamed(bytes: &[u8]) -> VirtualMesh {
        let len = VirtualMesh::header_len(bytes).unwrap();
        let mut mesh = VirtualMesh::from_header_bytes(&bytes[..len]).unwrap();
        let tail = mesh.root_tail_start();
        mesh.set_tail_payload(tail, &bytes[mesh.page_byte_range(tail).start as usize..]).unwrap();
        mesh
    }

    /// A fake GPU pool recording uploads.
    struct Pool(Vec<u32>);

    impl Pool {
        fn upload(&mut self) -> impl FnMut(u32, &[u8]) + '_ {
            |offset, bytes| {
                let words: Vec<u32> = bytemuck::pod_collect_to_vec(bytes);
                self.0[offset as usize..offset as usize + words.len()].copy_from_slice(&words);
            }
        }
    }

    /// The residency invariant, plus: resident pages hold exactly their payload.
    fn check(r: &Residency, reference: &[VirtualMesh], pool: &Pool) {
        for p in 0..r.pages.len() {
            let resident = r.state[p] == PageState::Resident;
            assert_eq!(resident, r.page_table[p] != NOT_RESIDENT);
            if resident {
                assert!(r.dependencies_resident(p), "page {p} resident before its dependencies");
                let page = r.pages[p];
                let payload = reference[page.mesh as usize].page_payload(page.local as usize);
                let offset = r.page_table[p] as usize;
                assert_eq!(&pool.0[offset..offset + payload.len()], payload, "page {p} payload");
            }
            let dependents = (0..r.pages.len()).filter(|&q| r.state[q] == PageState::Resident && r.dependencies[q].contains(&(p as u32))).count();
            assert_eq!(r.dependents_resident[p] as usize, dependents);
        }
        // No page is part of two outstanding reads, and only requested pages are.
        let mut seen = std::collections::HashSet::new();
        for read in r.reads.values() {
            for &p in &read.pages {
                assert!(seen.insert(p), "page {p} requested twice");
                assert_eq!(r.state[p as usize], PageState::Requested);
            }
        }
        assert_eq!(seen.len(), r.state.iter().filter(|&&s| s == PageState::Requested).count());
    }

    fn feedback_all(r: &Residency, frame: u32) -> Vec<u32> {
        vec![frame.max(1); r.pages.len()]
    }

    #[test]
    fn in_memory_pages_load_in_the_feedback_frame() {
        let (vm, _) = rock();
        let meshes = [vm];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        assert!(r.reset(&meshes, &[None], &mut pool.upload()));
        let n = r.pages.len();
        r.update(&meshes, 4, Some((&feedback_all(&r, 4), 4)), &mut pool.upload());
        check(&r, &meshes, &pool);
        assert_eq!(r.stats.resident_pages as usize, n, "a whole dependency chain resolves within one update");
        assert!(r.reads.is_empty());
    }

    #[test]
    fn streamed_pages_arrive_asynchronously_and_respect_dependencies() {
        let (vm, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 3);
        let meshes = [streamed(&bytes)];
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        assert!(r.reset(&meshes, &locations, &mut pool.upload()));
        let roots = meshes[0].pages.iter().filter(|p| p.root != 0).count();
        assert_eq!(r.stats.resident_pages as usize, roots, "only roots are resident before any read completes");
        let reference = [vm];
        check(&r, &reference, &pool);

        let n = r.pages.len();
        let mut frame = 1u32;
        let mut max_outstanding = 0;
        while r.stats.resident_pages < n as u32 {
            assert!(frame < 500, "streaming stalled at {} of {n} pages", r.stats.resident_pages);
            let feedback = frame.is_multiple_of(4).then(|| feedback_all(&r, frame));
            r.update(&meshes, frame, feedback.as_deref().map(|f| (f, frame)), &mut pool.upload());
            check(&r, &reference, &pool);
            max_outstanding = max_outstanding.max(source.outstanding().len());
            // Complete reads out of order of issue (newest first) to shake out ordering bugs.
            if frame.is_multiple_of(3) {
                source.started.lock().unwrap().reverse();
            }
            source.complete(2, &[]);
            frame += 1;
        }
        assert!(max_outstanding <= 3, "in-flight reads bounded by the source ({max_outstanding})");
        let streamed: usize = (0..n).filter(|&p| !r.pages[p].root).map(|p| r.pages[p].bytes()).sum();
        assert_eq!(source.bytes_read(), streamed, "every streamed page is read exactly once");
        let reads = r.next_ticket as usize;
        assert!(reads < n - roots, "adjacent pages coalesce: {reads} reads for {} pages", n - roots);
        assert_eq!(r.staged_bytes, 0);
    }

    #[test]
    fn root_pages_stream_too_when_not_in_memory() {
        let (vm, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 4);
        let header = VirtualMesh::from_header_bytes(&bytes).unwrap();
        let meshes = [header];
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        assert!(r.reset(&meshes, &locations, &mut pool.upload()));
        assert_eq!(r.stats.resident_pages, 0);
        let n = r.pages.len() as u32;
        let mut frame = 1u32;
        while r.stats.resident_pages < n {
            r.update(&meshes, frame, None, &mut pool.upload());
            check(&r, std::slice::from_ref(&vm), &pool);
            source.complete(usize::MAX, &[]);
            frame += 1;
            assert!(frame < 50, "prefetch alone loads a small mesh completely");
        }
    }

    #[test]
    fn a_page_both_wanted_and_prefetched_is_requested_once() {
        let (vm, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 64);
        let meshes = [streamed(&bytes)];
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        r.reset(&meshes, &locations, &mut pool.upload());
        assert!(!r.prefetch.is_empty());
        r.wanted = r.prefetch.clone();
        r.update(&meshes, 1, None, &mut pool.upload());
        check(&r, &[vm], &pool);
        let requested: usize = (0..r.pages.len()).filter(|&p| r.state[p] == PageState::Requested).map(|p| r.pages[p].bytes()).sum();
        assert!(requested > 0);
        assert_eq!(source.bytes_read(), requested);
    }

    #[test]
    fn evicting_a_dependency_while_a_dependent_is_in_flight_stays_consistent() {
        let (vm, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 64);
        let meshes = [streamed(&bytes)];
        let reference = [vm];
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        r.reset(&meshes, &locations, &mut pool.upload());
        // No prefetch: only explicitly wanted pages are read.
        r.prefetch.clear();
        let n = r.pages.len();
        let wants = |pages: &[usize], frame: u32| -> Vec<u32> { (0..n).map(|p| if pages.contains(&p) { frame } else { 0 }).collect() };

        // Pick a page X with a non-root dependency D, and make D resident.
        let x = (0..n).find(|&p| r.dependencies[p].iter().any(|&d| !r.pages[d as usize].root)).expect("a two-level chain");
        let d = r.dependencies[x].iter().copied().find(|&d| !r.pages[d as usize].root).unwrap() as usize;
        let mut frame = 1u32;
        while r.state[d] != PageState::Resident {
            r.update(&meshes, frame, Some((&wants(&[d], frame), frame)), &mut pool.upload());
            source.complete(usize::MAX, &[]);
            check(&r, &reference, &pool);
            frame += 1;
            assert!(frame < 100);
        }
        // Request X, then evict D while X's read is in flight.
        r.update(&meshes, frame, Some((&wants(&[x], frame), frame)), &mut pool.upload());
        assert_eq!(r.state[x], PageState::Requested);
        assert_eq!(r.dependents_resident[d], 0);
        r.unload(d);
        check(&r, &reference, &pool);

        // X arrives but must wait: D is not resident.
        source.complete(usize::MAX, &[]);
        frame += 1;
        r.update(&meshes, frame, None, &mut pool.upload());
        assert_eq!(r.state[x], PageState::Arrived);
        check(&r, &reference, &pool);

        // The next feedback still wants X, which pulls D in again; then X becomes resident.
        for _ in 0..4 {
            frame += 1;
            r.update(&meshes, frame, Some((&wants(&[x], frame), frame)), &mut pool.upload());
            source.complete(usize::MAX, &[]);
            check(&r, &reference, &pool);
        }
        assert_eq!(r.state[d], PageState::Resident);
        assert_eq!(r.state[x], PageState::Resident);
    }

    #[test]
    fn failed_reads_leave_pages_absent_and_are_retried() {
        let (vm, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 4);
        let meshes = [streamed(&bytes)];
        let reference = [vm];
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        r.reset(&meshes, &locations, &mut pool.upload());
        r.update(&meshes, 1, None, &mut pool.upload());
        let first = source.outstanding()[0].clone();
        source.complete(1, &[first.ticket]);
        r.update(&meshes, 2, None, &mut pool.upload());
        check(&r, &reference, &pool);
        assert_eq!(r.stats.failed_reads, 1);
        let failed: Vec<usize> = (0..r.pages.len()).filter(|&p| r.retry_at[p] > 2).collect();
        assert!(!failed.is_empty() && failed.iter().all(|&p| r.state[p] == PageState::Absent));

        let n = r.pages.len() as u32;
        let mut frame = 3u32;
        while r.stats.resident_pages < n {
            r.update(&meshes, frame, Some((&feedback_all(&r, frame), frame)), &mut pool.upload());
            source.complete(usize::MAX, &[]);
            check(&r, &reference, &pool);
            frame += 1;
            assert!(frame < 400, "failed pages were never retried");
        }
        assert!(frame > 2 + RETRY_FRAMES, "retried only after the back-off");
    }

    #[test]
    fn tiny_pool_with_moving_interest_keeps_the_invariant() {
        let (vm, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 4);
        let meshes = [streamed(&bytes), vm.clone()];
        let reference = [vm.clone(), vm];
        let location = PageLocation { source: source.clone(), file: 0 };
        let locations = [Some(location), None];
        let payload_words: u32 = reference.iter().map(|m| m.payload_bytes() as u32 / 4).sum();
        let mut pool = Pool(vec![0; payload_words as usize]);
        let mut r = Residency::new(payload_words / 3);
        assert!(r.reset(&meshes, &locations, &mut pool.upload()));
        let n = r.pages.len();
        let mut rng = 0x9e3779b9u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let mut evictions = 0;
        for frame in 1..1500u32 {
            // Interest moves: a window of pages that shifts every 100 frames.
            let window = (frame / 100) as usize * 7;
            let feedback: Vec<u32> = (0..n).map(|p| if (p + window) % n < n / 3 { frame } else { 0 }).collect();
            let feedback = frame.is_multiple_of(4).then_some(feedback);
            r.update(&meshes, frame, feedback.as_deref().map(|f| (f, frame)), &mut pool.upload());
            check(&r, &reference, &pool);
            evictions += r.stats.evictions;
            let fail = source.outstanding().first().filter(|_| next() % 17 == 0).map(|read| read.ticket);
            source.complete((next() % 3) as usize, fail.as_slice());
        }
        assert!(evictions > 0, "the pool is too small for the interest: pages must be evicted");
        assert!(r.stats.failed_reads > 0);
    }

    #[test]
    fn reset_abandons_outstanding_reads() {
        let (_, bytes) = rock();
        let source = ManualSource::new(bytes.clone(), 8);
        let meshes = [streamed(&bytes)];
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        r.reset(&meshes, &locations, &mut pool.upload());
        r.update(&meshes, 1, None, &mut pool.upload());
        assert!(!source.outstanding().is_empty());
        let resident = r.stats.resident_pages;
        r.reset(&meshes, &locations, &mut pool.upload());
        source.complete(usize::MAX, &[]);
        r.update(&meshes, 2, None, &mut pool.upload());
        assert_eq!(r.stats.resident_pages, resident, "stale completions are ignored");
        assert_eq!(r.stats.waiting, 0);
    }

    /// Pages of a `.hvm` with entropy-coded pages: read as their (smaller) stored bytes, decoded
    /// (inline, or on the decode threads a few updates later), uploaded as their payload words.
    #[test]
    fn coded_pages_are_read_small_and_decoded() {
        let (vm, _) = rock();
        let bytes = vm.to_coded_bytes();
        let meshes = [streamed(&bytes)];
        assert!(meshes[0].is_coded());
        let reference = [vm];
        for threads in [false, true] {
            let source = ManualSource::new(bytes.clone(), 3);
            let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
            let mut pool = Pool(vec![0; 1 << 22]);
            let mut r = Residency::new(1 << 22);
            if !threads {
                r.decoders = Decoders::Inline(Vec::new());
            }
            assert!(r.reset(&meshes, &locations, &mut pool.upload()));
            let n = r.pages.len();
            let mut frame = 1u32;
            let start = std::time::Instant::now();
            let mut decoding_seen = false;
            while r.stats.resident_pages < n as u32 {
                assert!(start.elapsed().as_secs() < 60, "streaming stalled at {} of {n} pages", r.stats.resident_pages);
                let feedback = frame.is_multiple_of(4).then(|| feedback_all(&r, frame));
                r.update(&meshes, frame, feedback.as_deref().map(|f| (f, frame)), &mut pool.upload());
                check(&r, &reference, &pool);
                decoding_seen |= r.stats.decoding > 0;
                source.complete(2, &[]);
                frame += 1;
                if threads {
                    std::thread::yield_now();
                }
            }
            assert!(!threads || decoding_seen || frame < 4, "threaded decodes take a frame or more");
            let stored: usize = (0..n).filter(|&p| !r.pages[p].root).map(|p| r.pages[p].stored as usize).sum();
            let words: usize = (0..n).filter(|&p| !r.pages[p].root).map(|p| r.pages[p].bytes()).sum();
            assert_eq!(source.bytes_read(), stored, "every streamed page is read exactly once, as stored");
            assert!(stored * 3 < words * 2, "coded pages are much smaller: {stored} of {words} bytes");
            assert_eq!(r.staged_bytes, 0);
            assert!(r.decodes.is_empty());
        }
    }

    #[test]
    fn corrupt_coded_pages_fail_and_are_retried() {
        let (vm, _) = rock();
        let mut bytes = vm.to_coded_bytes();
        let meshes = [streamed(&bytes)];
        // Damage the first non-root page's stored bytes (its tables and symbols).
        let range = meshes[0].page_byte_range(0);
        for b in &mut bytes[range.start as usize + 2..range.end as usize] {
            *b ^= 0x5a;
        }
        let source = ManualSource::new(bytes, 8);
        let locations = [Some(PageLocation { source: source.clone(), file: 0 })];
        let mut pool = Pool(vec![0; 1 << 22]);
        let mut r = Residency::new(1 << 22);
        r.decoders = Decoders::Inline(Vec::new());
        r.reset(&meshes, &locations, &mut pool.upload());
        for frame in 1..20 {
            r.update(&meshes, frame, Some((&feedback_all(&r, frame), frame)), &mut pool.upload());
            source.complete(usize::MAX, &[]);
        }
        let reference = [vm];
        check(&r, &reference, &pool);
        assert!(r.stats.failed_reads > 0, "the damaged page failed");
        assert_ne!(r.state[0], PageState::Resident);
    }
}
