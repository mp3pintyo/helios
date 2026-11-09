//! Tile residency of the streaming virtual texture: GPU feedback → CPU decisions → page table.
//!
//! Every virtual tile (layer, level, x, y) has one page-table entry. Feedback names the tiles
//! pixels want (at the finer level of their trilinear pair); a wanted tile also needs its
//! coarser ancestors, so requests are expanded to the whole chain and uploaded coarse first.
//! That keeps the **residency invariant** the shader relies on: every ancestor of a resident
//! tile is resident (the coarsest streamed level's parent is the always-resident array layer),
//! so the level above any resident tile can always be sampled for trilinear blending. Tiles
//! are evicted least-recently-needed first, finer before coarser, and never while the latest
//! feedback asks for them; evicting a tile evicts its resident descendants with it.
//!
//! Page-table entry of a tile: the atlas slot of its finest resident ancestor-or-self and the
//! minimum lod that may be sampled there (8.8 fixed point), or `INVALID` if no tile of that
//! chain is resident (the shader then samples the array). New tiles fade in over a few
//! frames by lowering that minimum lod gradually, so arriving detail does not pop.

use std::collections::VecDeque;

use super::pyramid::TILE;
use crate::textures::TEXTURE_SIZE;

pub const INVALID: u32 = u32::MAX;
/// Entries per row of the page-table texture.
pub const TABLE_WIDTH: u32 = 256;
/// Frames over which a newly resident tile blends in from its parent level.
const FADE_FRAMES: u32 = 8;

/// Where one layer's tiles live in the page table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerLayout {
    pub offset: u32,
    /// Streamed levels (0: the layer is not streamed).
    pub levels: u32,
}

impl LayerLayout {
    /// Tiles per side of `level`.
    pub fn tiles(self, level: u32) -> u32 {
        (TEXTURE_SIZE / TILE) << (self.levels - level)
    }

    /// First entry of `level`, relative to the layer's offset.
    pub fn level_offset(self, level: u32) -> u32 {
        (0..level).map(|m| self.tiles(m) * self.tiles(m)).sum()
    }

    pub fn entry_count(self) -> u32 {
        self.level_offset(self.levels)
    }
}

/// Supplies the tiles [`TileCache::update`] uploads.
pub trait TileSource {
    /// Gets a tile ready for upload (compresses it, say). False if that did not finish within
    /// the frame's budget: the tile stays first in line, and the frame's uploads end.
    fn prepare(&mut self, tile: TileId) -> bool;
    /// Writes a prepared tile into an atlas slot. False ends the frame's uploads.
    fn upload(&mut self, tile: TileId, slot: u32) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileId {
    pub layer: u32,
    pub level: u32,
    pub x: u32,
    pub y: u32,
}

/// Page-table layout of all layers.
#[derive(Debug, Default)]
pub struct Layout {
    layers: Vec<LayerLayout>,
    /// Streamed layers in entry order.
    streamed: Vec<u32>,
    pub entry_count: u32,
}

impl Layout {
    /// `levels[layer]`: streamed levels of each texture layer (0 for non-streamed layers).
    pub fn new(levels: &[u32]) -> Self {
        let mut offset = 0;
        let mut streamed = Vec::new();
        let layers = levels
            .iter()
            .enumerate()
            .map(|(i, &levels)| {
                let layout = LayerLayout { offset, levels };
                if levels > 0 {
                    streamed.push(i as u32);
                    offset += layout.entry_count();
                }
                layout
            })
            .collect();
        Self { layers, streamed, entry_count: offset }
    }

    pub fn layer(&self, layer: u32) -> LayerLayout {
        self.layers.get(layer as usize).copied().unwrap_or_default()
    }

    pub fn layers(&self) -> &[LayerLayout] {
        &self.layers
    }

    pub fn entry(&self, t: TileId) -> u32 {
        let l = self.layer(t.layer);
        l.offset + l.level_offset(t.level) + t.y * l.tiles(t.level) + t.x
    }

    pub fn tile(&self, entry: u32) -> TileId {
        let i = self.streamed.partition_point(|&layer| self.layers[layer as usize].offset <= entry) - 1;
        let layer = self.streamed[i];
        let l = self.layers[layer as usize];
        let mut local = entry - l.offset;
        let mut level = 0;
        while local >= l.tiles(level) * l.tiles(level) {
            local -= l.tiles(level) * l.tiles(level);
            level += 1;
        }
        TileId { layer, level, x: local % l.tiles(level), y: local / l.tiles(level) }
    }

    /// The tile one level coarser covering `entry`, if that level is streamed.
    pub fn parent(&self, entry: u32) -> Option<u32> {
        let t = self.tile(entry);
        (t.level + 1 < self.layer(t.layer).levels).then(|| self.entry(TileId { level: t.level + 1, x: t.x / 2, y: t.y / 2, ..t }))
    }

    /// The four tiles one level finer, if any.
    pub fn children(&self, entry: u32) -> Option<[u32; 4]> {
        let t = self.tile(entry);
        (t.level > 0).then(|| {
            let child = |dx, dy| self.entry(TileId { level: t.level - 1, x: t.x * 2 + dx, y: t.y * 2 + dy, ..t });
            [child(0, 0), child(1, 0), child(0, 1), child(1, 1)]
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    /// Atlas slot, or `INVALID`.
    slot: u32,
    /// Feedback frame that last asked for the tile (0: never).
    needed: u32,
    /// Frame the tile was uploaded.
    arrived: u32,
}

/// Residency counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub resident: u32,
    pub capacity: u32,
    /// Distinct tiles (with ancestors) the latest feedback asked for.
    pub requested: u32,
    /// Requested tiles still waiting for upload.
    pub pending: u32,
    pub uploads: u64,
    pub evictions: u64,
    /// Requested tiles that found no slot (the pool is full of tiles the view needs).
    pub overflow: u32,
}

pub struct TileCache {
    pub layout: Layout,
    entries: Vec<Entry>,
    /// Owning entry of each atlas slot, or `INVALID`.
    slots: Vec<u32>,
    free: Vec<u32>,
    /// Requested, non-resident tiles, coarsest first.
    queue: VecDeque<u32>,
    latest_feedback: u32,
    fading: Vec<u32>,
    dirty_layers: Vec<bool>,
    /// Page table, `TABLE_WIDTH`-padded.
    table: Vec<u32>,
    dirty_entries: Option<(u32, u32)>,
    stats: CacheStats,
}

impl TileCache {
    pub fn new(layout: Layout, slots: u32) -> Self {
        let n = layout.entry_count as usize;
        let mut cache = Self {
            entries: vec![Entry { slot: INVALID, needed: 0, arrived: 0 }; n],
            slots: vec![INVALID; slots as usize],
            free: (0..slots).rev().collect(),
            queue: VecDeque::new(),
            latest_feedback: 0,
            fading: Vec::new(),
            dirty_layers: vec![false; layout.layers().len()],
            table: vec![INVALID; n.max(1).next_multiple_of(TABLE_WIDTH as usize)],
            dirty_entries: None,
            stats: CacheStats { capacity: slots, ..Default::default() },
            layout,
        };
        cache.dirty_entries = (n > 0).then_some((0, n as u32 - 1));
        cache
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats { pending: self.queue.len() as u32, ..self.stats }
    }

    pub fn table(&self) -> &[u32] {
        &self.table
    }

    /// Range of page-table entries changed since the last call.
    pub fn take_dirty_entries(&mut self) -> Option<(u32, u32)> {
        self.dirty_entries.take()
    }

    fn resident(&self, entry: u32) -> bool {
        self.entries[entry as usize].slot != INVALID
    }

    /// One frame: applies new feedback (request words, entry + 1 or 0, from `feedback_frame`),
    /// uploads up to `budget` requested tiles from `source`, coarse first, and refreshes the
    /// page table.
    pub fn update(&mut self, frame: u32, feedback: Option<(&[u32], u32)>, budget: usize, source: &mut dyn TileSource) {
        if let Some((words, feedback_frame)) = feedback {
            self.apply_feedback(words, feedback_frame);
        }
        let mut uploaded = 0;
        while uploaded < budget {
            let Some(entry) = self.queue.pop_front() else { break };
            let parent_missing = self.layout.parent(entry).is_some_and(|p| !self.resident(p));
            if self.resident(entry) || parent_missing {
                continue;
            }
            let tile = self.layout.tile(entry);
            if !source.prepare(tile) {
                self.queue.push_front(entry);
                break;
            }
            let Some(slot) = self.allocate() else {
                self.stats.overflow = self.queue.len() as u32 + 1;
                self.queue.clear();
                break;
            };
            let more = source.upload(tile, slot);
            self.entries[entry as usize] = Entry { slot, arrived: frame, ..self.entries[entry as usize] };
            self.slots[slot as usize] = entry;
            self.fading.push(entry);
            self.mark_dirty(entry);
            self.stats.uploads += 1;
            self.stats.resident += 1;
            uploaded += 1;
            if !more {
                break;
            }
        }
        let fading = std::mem::take(&mut self.fading);
        for &entry in &fading {
            self.mark_dirty(entry);
        }
        self.fading = fading.into_iter().filter(|&e| self.resident(e) && frame.wrapping_sub(self.entries[e as usize].arrived) < FADE_FRAMES).collect();
        self.rebuild_table(frame);
    }

    fn apply_feedback(&mut self, words: &[u32], feedback_frame: u32) {
        self.latest_feedback = feedback_frame;
        let mut wanted = Vec::new();
        let mut previous = 0;
        for &word in words {
            // Neighbouring blocks mostly want the same tile.
            if std::mem::replace(&mut previous, word) == word {
                continue;
            }
            let mut next = word.checked_sub(1).filter(|&e| e < self.layout.entry_count);
            while let Some(entry) = next {
                // A marked tile's ancestors are marked already.
                if self.entries[entry as usize].needed == feedback_frame {
                    break;
                }
                self.entries[entry as usize].needed = feedback_frame;
                wanted.push(entry);
                next = self.layout.parent(entry);
            }
        }
        self.stats.requested = wanted.len() as u32;
        self.stats.overflow = 0;
        wanted.retain(|&e| !self.resident(e));
        wanted.sort_by_key(|&e| (std::cmp::Reverse(self.layout.tile(e).level), e));
        self.queue = wanted.into();
    }

    /// A free slot, or the slot of the least recently needed tile the latest feedback did not
    /// ask for (finer tiles first among equals).
    fn allocate(&mut self) -> Option<u32> {
        if self.free.is_empty() {
            let victim = self
                .slots
                .iter()
                .copied()
                .filter(|&e| e != INVALID && self.entries[e as usize].needed < self.latest_feedback)
                .min_by_key(|&e| (self.entries[e as usize].needed, self.layout.tile(e).level))?;
            self.evict(victim);
        }
        self.free.pop()
    }

    /// Evicts a tile and its resident descendants (which would otherwise lose their parent).
    fn evict(&mut self, entry: u32) {
        if let Some(children) = self.layout.children(entry) {
            for child in children {
                if self.resident(child) {
                    self.evict(child);
                }
            }
        }
        let slot = self.entries[entry as usize].slot;
        self.slots[slot as usize] = INVALID;
        self.entries[entry as usize].slot = INVALID;
        self.free.push(slot);
        self.mark_dirty(entry);
        self.stats.evictions += 1;
        self.stats.resident -= 1;
    }

    fn mark_dirty(&mut self, entry: u32) {
        self.dirty_layers[self.layout.tile(entry).layer as usize] = true;
    }

    /// Recomputes the page-table entries of changed layers, coarse levels first so every
    /// entry can inherit from its parent.
    fn rebuild_table(&mut self, frame: u32) {
        for layer in 0..self.dirty_layers.len() {
            if !std::mem::take(&mut self.dirty_layers[layer]) {
                continue;
            }
            let l = self.layout.layer(layer as u32);
            for level in (0..l.levels).rev() {
                for y in 0..l.tiles(level) {
                    for x in 0..l.tiles(level) {
                        let tile = TileId { layer: layer as u32, level, x, y };
                        let entry = self.layout.entry(tile);
                        let inherited = self.layout.parent(entry).map_or(INVALID, |p| self.table[p as usize]);
                        let e = self.entries[entry as usize];
                        self.table[entry as usize] = if e.slot == INVALID {
                            inherited
                        } else {
                            // Blend from the parent level while fading in; an ancestor that is
                            // itself still fading in (its minimum lod above its level) holds
                            // the whole chain back.
                            let fade = (frame.wrapping_sub(e.arrived) as f32 / FADE_FRAMES as f32).min(1.0);
                            let parent_level = (level + 1) as f32;
                            let held = if inherited == INVALID { 0.0 } else { min_lod(inherited) };
                            let held = if held > parent_level { held } else { 0.0 };
                            encode(e.slot, (level as f32 + 1.0 - fade).max(held))
                        };
                    }
                }
            }
            let (first, last) = (l.offset, l.offset + l.entry_count() - 1);
            self.dirty_entries = Some(self.dirty_entries.map_or((first, last), |(a, b)| (a.min(first), b.max(last))));
        }
    }
}

/// Page-table entry: atlas slot (16 bits) and minimum sampled lod (8.8 fixed point).
pub fn encode(slot: u32, min_lod: f32) -> u32 {
    slot | (((min_lod * 256.0).round() as u32) << 16)
}

pub fn min_lod(entry: u32) -> f32 {
    (entry >> 16) as f32 / 256.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(entry: u32) -> u32 {
        entry & 0xffff
    }

    /// Layer 0: procedural; layer 1: 1024² (one streamed level, 8×8 tiles); layer 2: 2048²
    /// (16×16 + 8×8 tiles).
    fn layout() -> Layout {
        Layout::new(&[0, 1, 2])
    }

    fn request(cache: &TileCache, tiles: &[TileId]) -> Vec<u32> {
        tiles.iter().map(|&t| cache.layout.entry(t) + 1).chain([0, 0]).collect()
    }

    /// Records uploads; `ready` and `more` are the answers of `prepare` and `upload`.
    struct Recorder {
        uploaded: Vec<TileId>,
        ready: bool,
        more: bool,
    }

    impl TileSource for Recorder {
        fn prepare(&mut self, _: TileId) -> bool {
            self.ready
        }

        fn upload(&mut self, tile: TileId, _: u32) -> bool {
            self.uploaded.push(tile);
            self.more
        }
    }

    fn uploads(cache: &mut TileCache, frame: u32, feedback: Option<(&[u32], u32)>, budget: usize) -> Vec<TileId> {
        let mut recorder = Recorder { uploaded: Vec::new(), ready: true, more: true };
        cache.update(frame, feedback, budget, &mut recorder);
        recorder.uploaded
    }

    #[test]
    fn layout_round_trips() {
        let l = layout();
        assert_eq!(l.entry_count, 64 + 256 + 64);
        assert_eq!(l.layer(2), LayerLayout { offset: 64, levels: 2 });
        for entry in 0..l.entry_count {
            assert_eq!(l.entry(l.tile(entry)), entry);
        }
        let fine = l.entry(TileId { layer: 2, level: 0, x: 13, y: 6 });
        let parent = l.parent(fine).unwrap();
        assert_eq!(l.tile(parent), TileId { layer: 2, level: 1, x: 6, y: 3 });
        assert!(l.children(parent).unwrap().contains(&fine));
        assert_eq!(l.parent(parent), None);
        assert_eq!(l.children(fine), None);
    }

    #[test]
    fn requests_load_coarse_first_and_fade_in() {
        let mut cache = TileCache::new(layout(), 16);
        let fine = TileId { layer: 2, level: 0, x: 3, y: 4 };
        let words = request(&cache, &[fine]);
        let loaded = uploads(&mut cache, 1, Some((&words, 1)), 1);
        let coarse = TileId { layer: 2, level: 1, x: 1, y: 2 };
        assert_eq!(loaded, [coarse]);
        let entry = cache.layout.entry(fine);
        assert_eq!(min_lod(cache.table()[entry as usize]), 2.0, "the coarse tile starts at the array's level");

        let start = 1 + FADE_FRAMES;
        assert_eq!(uploads(&mut cache, start, None, 8), [fine]);
        assert_eq!(cache.stats().resident, 2);
        assert_eq!(min_lod(cache.table()[entry as usize]), 1.0, "a new tile starts at its parent's level");
        uploads(&mut cache, start + FADE_FRAMES / 2, None, 8);
        assert_eq!(min_lod(cache.table()[entry as usize]), 0.5);
        uploads(&mut cache, start + FADE_FRAMES, None, 8);
        let value = cache.table()[entry as usize];
        assert_eq!(min_lod(value), 0.0);
        // Its neighbour inherits the parent's slot and lod.
        let neighbour = cache.layout.entry(TileId { x: 2, ..fine });
        let parent = cache.table()[cache.layout.parent(entry).unwrap() as usize];
        assert_eq!(cache.table()[neighbour as usize], parent);
        assert_eq!((min_lod(parent), slot(value) != slot(parent)), (1.0, true));
        // Untouched layers stay invalid.
        assert_eq!(cache.table()[0], INVALID);
    }

    #[test]
    fn eviction_is_lru_and_keeps_ancestors() {
        let mut cache = TileCache::new(layout(), 3);
        let a = TileId { layer: 2, level: 0, x: 0, y: 0 };
        let words = request(&cache, &[a]);
        assert_eq!(uploads(&mut cache, 1, Some((&words, 1)), 8).len(), 2);
        // A later feedback wants other tiles: the old leaf goes first (finer before coarser).
        let b = TileId { layer: 1, level: 0, x: 5, y: 5 };
        let c = TileId { layer: 1, level: 0, x: 6, y: 5 };
        let words = request(&cache, &[b, c]);
        let loaded = uploads(&mut cache, 2, Some((&words, 2)), 8);
        assert_eq!(loaded, [b, c]);
        let (a, a_parent) = (cache.layout.entry(a), cache.layout.parent(cache.layout.entry(a)).unwrap());
        assert!(!cache.resident(a) && cache.resident(a_parent));
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.table()[a as usize], cache.table()[a_parent as usize]);

        // Then the parent.
        let d = TileId { layer: 1, level: 0, x: 7, y: 7 };
        let words = request(&cache, &[b, c, d]);
        assert_eq!(uploads(&mut cache, 3, Some((&words, 3)), 8), [d]);
        assert_eq!(cache.table()[a as usize], INVALID);

        // Tiles the latest feedback asks for are never evicted: the rest overflows.
        let e = TileId { layer: 1, level: 0, x: 0, y: 7 };
        let words = request(&cache, &[b, c, d, e]);
        assert_eq!(uploads(&mut cache, 4, Some((&words, 4)), 8), []);
        assert_eq!(cache.stats().overflow, 1);
        assert_eq!(cache.stats().resident, 3);
    }

    #[test]
    fn an_exhausted_budget_ends_the_frame() {
        let mut cache = TileCache::new(layout(), 16);
        let tiles = [TileId { layer: 1, level: 0, x: 0, y: 0 }, TileId { layer: 1, level: 0, x: 1, y: 0 }];
        let words = request(&cache, &tiles);
        // A tile that is not ready keeps its place and takes no slot.
        let mut source = Recorder { uploaded: Vec::new(), ready: false, more: true };
        cache.update(1, Some((&words, 1)), 8, &mut source);
        assert_eq!((source.uploaded.len(), cache.stats().resident, cache.stats().pending, cache.free.len()), (0, 0, 2, 16));
        // The upload time running out ends the frame after that tile.
        source.ready = true;
        source.more = false;
        cache.update(2, None, 8, &mut source);
        assert_eq!((source.uploaded.as_slice(), cache.stats().resident, cache.stats().pending), (&tiles[..1], 1, 1));
        assert_eq!(uploads(&mut cache, 3, None, 8), [tiles[1]]);
    }

    #[test]
    fn evicting_a_parent_takes_its_children() {
        let mut cache = TileCache::new(layout(), 8);
        let fine = TileId { layer: 2, level: 0, x: 9, y: 9 };
        let words = request(&cache, &[fine]);
        uploads(&mut cache, 1, Some((&words, 1)), 8);
        let parent = cache.layout.parent(cache.layout.entry(fine)).unwrap();
        cache.evict(parent);
        assert_eq!(cache.stats().resident, 0);
        assert_eq!(cache.free.len(), 8);
    }

    #[test]
    fn invariant_holds_under_random_traffic() {
        let mut cache = TileCache::new(layout(), 20);
        let n = cache.layout.entry_count;
        let mut seed = 12345u32;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for frame in 1..400 {
            let feedback: Vec<u32> = (0..12).map(|_| random() % (n + 1)).collect();
            let feedback = (frame % 3 == 0).then_some(feedback);
            let mut source = Recorder { uploaded: Vec::new(), ready: frame % 7 != 0, more: frame % 5 != 0 };
            cache.update(frame, feedback.as_deref().map(|f| (f, frame)), 4, &mut source);
            for entry in 0..n {
                if cache.resident(entry) {
                    if let Some(p) = cache.layout.parent(entry) {
                        assert!(cache.resident(p), "resident tile without resident parent");
                    }
                    let value = cache.table()[entry as usize];
                    assert_eq!(slot(value), cache.entries[entry as usize].slot);
                    assert!(min_lod(value) >= cache.layout.tile(entry).level as f32);
                }
            }
            let used = cache.slots.iter().filter(|&&e| e != INVALID).count();
            assert_eq!(used + cache.free.len(), 20);
            assert_eq!(used as u32, cache.stats().resident);
        }
    }
}
