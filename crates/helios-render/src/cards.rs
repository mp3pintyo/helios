//! Lumen surface cache made of mesh cards (UE5 Lumen's card-based surface cache).
//!
//! The voxel surface cache (`lumen.rs`) stores the light leaving the surfaces inside each voxel
//! of a 0.5–4 m grid, so a thin wall's sunlit outside bleeds into its inside and small objects
//! blur into their surroundings. Cards store lighting on the surfaces themselves:
//!
//! * **Capture** (CPU, when the scene changes): every mesh with a volume distance field is
//!   captured by up to 6 directions × 3 depth layers of orthographic cards — depth and normal
//!   per texel, ~10 cm texels (coarser for very large meshes), rasterised from a LOD cut of the
//!   virtual mesh ([`helios_geometry::cards`]). Captures are per mesh, packed into one capture
//!   atlas; materials contribute their average albedo (base colour × the texture's 1×1 mip,
//!   as the global distance field does), since an instance has a single material and diffuse
//!   bounce light is low-frequency anyway.
//! * **Allocation**: each instance's cards are lit at a *mip* of their capture chosen from the
//!   instance's distance to the camera (lighting texels of ~1 % of that distance), in 8×8
//!   texel pages of a 2048² lighting atlas reached through per-card page tables. Instances
//!   beyond [`MAX_DISTANCE`] get no cards; a global LOD bias coarsens all mips when the atlas
//!   runs full. An instance switches mip only when all its new pages can be lit in the same
//!   frame, so its cards never show unlit pages.
//! * **Lighting** (`lumen_card_light.wgsl`): a budget of pages per frame — newly allocated
//!   ones first (closest instances first), then a round-robin refresh of the rest.
//! * **Sampling** (`lumen_cards.wgsl`): screen-probe and reflection hits are projected into the
//!   hit instance's cards; the near field reports the instance it hit, the global distance
//!   field stores the closest instance per voxel. Where no card matches, the voxel surface
//!   cache stays in place.

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use helios_geometry::cards::{CardConfig, CardVertex, MeshCards, CARD_AXES, MAX_CARD_LAYERS};
use helios_geometry::{lod_cut_resident, LodView, SdfKind, VirtualMesh};
use wgpu::{TextureFormat as F, TextureUsages as U};

use crate::scene::Scene;
use crate::util::{storage_buffer, storage_buffer_init};

/// Card slots per mesh / instance: directions × layers (`CARD_SLOTS` in lumen_cards.wgsl).
const CARD_SLOTS: usize = CARD_AXES * MAX_CARD_LAYERS;
/// Texels per side of a lighting page (`CARD_PAGE_TEXELS`).
const PAGE_TEXELS: u32 = 8;
/// The lighting atlas holds `ATLAS_PAGES`² pages (2048² rgba16f texels, 32 MiB).
const ATLAS_PAGES: u32 = 256;
const PAGE_COUNT: usize = (ATLAS_PAGES * ATLAS_PAGES) as usize;
/// Pages lit per frame: new allocations take up to three quarters, refreshes the rest.
const UPDATE_BUDGET: usize = 1024;
/// Target capture texel size (world metres) and the card side limit.
const CARD_TEXEL: f32 = 0.1;
const MAX_CARD_RESOLUTION: u32 = 256;
/// Captured texels of all meshes at most (4 bytes each); texels grow to stay within it.
const CAPTURE_BUDGET: f32 = 4.0e6;
const CAPTURE_ATLAS_WIDTH: u32 = 4096;
/// Lighting texel size per metre of distance from the camera.
const TEXEL_ANGLE: f32 = 0.01;
/// Instances farther away than this keep the voxel surface cache.
const MAX_DISTANCE: f32 = 100.0;
/// Coarsest global LOD bias (each step halves the lighting resolution).
const MAX_BIAS: u32 = 4;
/// u32 words per uploaded chunk of the page tables.
const CHUNK_WORDS: usize = 1024;
const NONE: u32 = u32::MAX;
/// vec4s per instance / card record (`CARD_INSTANCE_VEC4S`, `CARD_RECORD_VEC4S`).
const INSTANCE_VEC4S: usize = 7;
const CARD_VEC4S: usize = 3;

/// Mesh-card part of the Lumen uniform (`cards` in lumen_types.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CardUniform {
    /// Card instances (0: none), first card record (vec4s), atlas pages per row, -.
    counts: [u32; 4],
}

/// Texels of a card side at lighting mip `mip`.
fn mip_resolution(resolution: u32, mip: u32) -> u32 {
    resolution.div_ceil(1 << mip).max(1)
}

/// Pages (x, y) of a card at mip `mip`.
fn mip_pages(resolution: [u32; 2], mip: u32) -> [u32; 2] {
    resolution.map(|r| mip_resolution(r, mip).div_ceil(PAGE_TEXELS))
}

/// The lighting mip an instance wants: lighting texels of about `TEXEL_ANGLE` × `distance`
/// (times 2^bias), at most `max_mip`; `None` beyond [`MAX_DISTANCE`]. A current mip is kept
/// until the ideal one is a quarter mip finer, so instances don't flip between two mips as the
/// camera moves back and forth. (Coarsening is never delayed: instances then never hold more
/// pages than the LOD bias accounts for.)
fn desired_mip(distance: f32, texel: f32, max_mip: u32, bias: u32, current: Option<u32>) -> Option<u32> {
    let reach = if current.is_some() { MAX_DISTANCE * 1.1 } else { MAX_DISTANCE };
    if distance > reach {
        return None;
    }
    let ideal = (distance.max(1.0) * TEXEL_ANGLE * (1 << bias) as f32 / texel).log2();
    let mip = (ideal.floor().max(0.0) as u32).min(max_mip);
    match current {
        Some(c) if c == mip || (ideal >= c as f32 - 0.25 && ideal < c as f32 + 1.0) => Some(c),
        _ => Some(mip),
    }
}

/// Packs `sizes` into shelves of a `width`-texel wide atlas; returns the positions and height.
fn pack_shelves(sizes: &[[u32; 2]], width: u32) -> (Vec<[u32; 2]>, u32) {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(sizes[i][1]), i));
    let mut positions = vec![[0; 2]; sizes.len()];
    let (mut x, mut y, mut shelf) = (0, 0, 0);
    for i in order {
        let [w, h] = sizes[i];
        assert!(w <= width, "card {w} texels wide exceeds the capture atlas");
        if x + w > width {
            (x, y, shelf) = (0, y + shelf, 0);
        }
        positions[i] = [x, y];
        x += w;
        shelf = shelf.max(h);
    }
    (positions, y + shelf)
}

/// Triangles of `mesh` (mesh space) at about `error` geometric error, from the pages in memory.
fn mesh_triangles(mesh: &VirtualMesh, error: f32) -> Vec<[CardVertex; 3]> {
    let view = LodView { position: Vec3::ZERO, lod_scale: 1.0, threshold: error, orthographic: true };
    let mut triangles = Vec::new();
    for cluster in lod_cut_resident(mesh, &view, &|page| mesh.has_payload(page as usize)) {
        let c = cluster as usize;
        let vertices: Vec<CardVertex> = (0..mesh.clusters[c].vertex_count())
            .map(|i| {
                let (p, n, _) = mesh.cluster_vertex(c, i);
                (p, n)
            })
            .collect();
        triangles.extend(mesh.cluster_triangles(c).map(|t| t.map(|i| vertices[i as usize])));
    }
    triangles
}

/// A captured mesh as laid out on the GPU.
struct MeshEntry {
    /// Card record of slot 0 (slot = direction × layers + layer).
    first_record: u32,
    /// Layer count per direction, 2 bits each.
    layers: u32,
    /// Capture texel size (mesh units).
    texel: f32,
    /// Per slot: resolution ([0, 0] if empty) and offset of its page table in the instance's.
    slots: [([u32; 2], u32); CARD_SLOTS],
    /// Page-table entries per instance (every slot at mip 0).
    table_size: u32,
    /// Pages of all cards per mip; the last mip fits every card into one page.
    pages: Vec<u32>,
}

impl MeshEntry {
    fn new(cards: &MeshCards, first_record: u32) -> Self {
        let mut slots = [([0; 2], 0); CARD_SLOTS];
        let mut layers = 0;
        let mut table_size = 0;
        for card in &cards.cards {
            let slot = (card.axis * MAX_CARD_LAYERS as u32 + card.layer) as usize;
            let [px, py] = mip_pages(card.resolution, 0);
            slots[slot] = (card.resolution, table_size);
            table_size += px * py;
            layers += 1 << (card.axis * 2);
        }
        let mut pages = Vec::new();
        for mip in 0.. {
            pages.push(cards.cards.iter().map(|c| mip_pages(c.resolution, mip).iter().product::<u32>()).sum());
            if cards.cards.iter().all(|c| mip_pages(c.resolution, mip) == [1, 1]) {
                break;
            }
        }
        Self { first_record, layers, texel: cards.texel_size, slots, table_size, pages }
    }

    fn max_mip(&self) -> u32 {
        self.pages.len() as u32 - 1
    }
}

struct InstanceEntry {
    /// Index into the allocator's meshes (`None`: no cards).
    mesh: Option<usize>,
    /// World bounds.
    bounds: (Vec3, Vec3),
    /// World size of a mip-0 lighting texel.
    texel: f32,
    /// First page-table entry.
    table: u32,
    mip: Option<u32>,
    /// Physical pages held.
    pages: Vec<u32>,
}

impl InstanceEntry {
    fn new(mesh: Option<usize>, bounds: (Vec3, Vec3), texel: f32) -> Self {
        Self { mesh, bounds, texel, table: 0, mip: None, pages: Vec::new() }
    }
}

/// Page allocation and the lighting schedule (CPU side of the cache).
///
/// The GPU buffer `card_pages` mirrors `page_words`: one allocation word per instance card
/// (`page table start << 4 | mip`, or NONE), then each instance's page tables — a fixed region
/// per card, sized for mip 0 and used row by row (`pages x` wide) at coarser mips.
#[derive(Default)]
struct Allocator {
    meshes: Vec<MeshEntry>,
    instances: Vec<InstanceEntry>,
    /// The `card_pages` words and which chunks of them changed since the last upload.
    page_words: Vec<u32>,
    dirty: Vec<bool>,
    free: Vec<u32>,
    /// Per physical page: its update record (physical page, instance, slot | mip << 8,
    /// page x | y << 16), instance NONE if free.
    owners: Vec<[u32; 4]>,
    /// Frame each physical page was last scheduled in.
    scheduled: Vec<u32>,
    refresh_cursor: usize,
    bias: u32,
    frame: u32,
}

/// Mesh-card surface cache: captures, allocation and the per-frame lighting schedule.
pub struct CardCache {
    /// Instance records, then card records (`card_data`).
    pub data: wgpu::Buffer,
    /// Allocation words, then page tables (`card_pages`).
    pub pages: wgpu::Buffer,
    /// Captured texels (r32uint: depth16 | normal16).
    pub capture: wgpu::TextureView,
    /// Lit pages (rgba16f), read by the gather and written by `light_cards`.
    pub lighting: wgpu::TextureView,
    /// This frame's pages to light (`card_updates`).
    pub updates: wgpu::Buffer,
    update_capacity: usize,
    /// Pages scheduled this frame.
    update_count: usize,
    captures: Vec<Option<MeshCards>>,
    capture_version: u64,
    allocator: Allocator,
    uniform: CardUniform,
}

impl CardCache {
    pub fn new(device: &wgpu::Device) -> Self {
        let side = ATLAS_PAGES * PAGE_TEXELS;
        let lighting = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("card lighting"),
            size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: F::Rgba16Float,
            usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
            view_formats: &[],
        });
        Self {
            data: storage_buffer(device, "card data", 16, wgpu::BufferUsages::empty()),
            pages: storage_buffer(device, "card pages", 16, wgpu::BufferUsages::empty()),
            capture: capture_texture(device, 1, 1).create_view(&Default::default()),
            lighting: lighting.create_view(&Default::default()),
            updates: storage_buffer(device, "card updates", 16, wgpu::BufferUsages::empty()),
            update_capacity: 1,
            update_count: 0,
            captures: Vec::new(),
            capture_version: u64::MAX,
            allocator: Allocator::default(),
            uniform: CardUniform::default(),
        }
    }

    /// Pages to light this frame (`light_cards` workgroups).
    pub fn update_count(&self) -> u32 {
        self.update_count as u32
    }

    /// Captures the meshes (when the geometry changed) and lays out the cards of
    /// `instances` — the scene instances in the order of Lumen's distance-field instances,
    /// whose indices the shaders use — with their world `bounds`. Clears all allocations.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, instances: &[usize], bounds: &[(Vec3, Vec3)]) {
        // The meshes to capture, with the largest scale they are instanced at (0: unused).
        let mut mesh_scale = vec![0.0f32; scene.meshes.len()];
        for &i in instances {
            let instance = &scene.instances[i];
            let m = instance.mesh.0 as usize;
            if scene.meshes[m].sdf.as_ref().is_some_and(|s| s.kind == SdfKind::Volume) {
                mesh_scale[m] = mesh_scale[m].max(max_scale(instance.transform));
            }
        }
        let used: Vec<bool> = mesh_scale.iter().map(|&s| s > 0.0).collect();
        let missing = used.iter().enumerate().any(|(m, &u)| u && self.captures.get(m).is_none_or(Option::is_none));
        if self.capture_version != scene.geometry_version || missing {
            self.capture_version = scene.geometry_version;
            self.captures = capture_meshes(scene, &mesh_scale);
        }

        // Card records and the capture atlas.
        let mut mesh_index = vec![None; scene.meshes.len()];
        let mut sizes = Vec::new();
        let mut meshes = Vec::new();
        for (m, cards) in self.captures.iter().enumerate() {
            let Some(cards) = cards.as_ref().filter(|_| used[m]) else { continue };
            mesh_index[m] = Some(meshes.len());
            sizes.extend(cards.cards.iter().map(|c| c.resolution));
            meshes.push(MeshEntry::new(cards, (meshes.len() * CARD_SLOTS) as u32));
        }
        let (positions, height) = pack_shelves(&sizes, CAPTURE_ATLAS_WIDTH);
        let mut atlas = vec![u32::MAX; (CAPTURE_ATLAS_WIDTH * height.max(1)) as usize];
        let mut records = vec![[0u32; 4]; meshes.len() * CARD_SLOTS * CARD_VEC4S];
        let mut k = 0;
        for (m, cards) in self.captures.iter().enumerate() {
            let Some(entry) = mesh_index[m].map(|e| &meshes[e]) else { continue };
            let sdf_voxel = scene.meshes[m].sdf.as_ref().map_or(0.0, |s| s.voxel_size);
            for card in &cards.as_ref().unwrap().cards {
                let [x, y] = positions[k];
                k += 1;
                for row in 0..card.resolution[1] {
                    let src = (row * card.resolution[0]) as usize;
                    let dst = ((y + row) * CAPTURE_ATLAS_WIDTH + x) as usize;
                    atlas[dst..dst + card.resolution[0] as usize].copy_from_slice(&card.texels[src..src + card.resolution[0] as usize]);
                }
                let slot = (card.axis * MAX_CARD_LAYERS as u32 + card.layer) as usize;
                let base = (entry.first_record as usize + slot) * CARD_VEC4S;
                let [d0, d1] = card.depth_range;
                records[base] = Vec4::new(card.origin.x, card.origin.y, entry.texel, card.plane).to_array().map(f32::to_bits);
                records[base + 1] = Vec4::new(d0, (d1 - d0) / 65534.0, sdf_voxel, 0.0).to_array().map(f32::to_bits);
                records[base + 2] = [card.resolution[0] | (card.resolution[1] << 16), x | (y << 16), 0, 0];
            }
        }
        let capture = capture_texture(device, CAPTURE_ATLAS_WIDTH, height.max(1));
        queue.write_texture(
            capture.as_image_copy(),
            bytemuck::cast_slice(&atlas),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(CAPTURE_ATLAS_WIDTH * 4), rows_per_image: None },
            capture.size(),
        );
        self.capture = capture.create_view(&Default::default());

        // Instance records and the page-table layout.
        let mut entries = Vec::with_capacity(instances.len());
        let mut data: Vec<[u32; 4]> = Vec::with_capacity(instances.len() * INSTANCE_VEC4S + records.len());
        for (&i, &bounds) in instances.iter().zip(bounds) {
            let instance = &scene.instances[i];
            let m = instance.mesh.0 as usize;
            let world = instance.transform;
            let scale = max_scale(world);
            let entry = mesh_index[m].map(|e| &meshes[e]);
            let to_local = world.inverse();
            for row in 0..3 {
                data.push(to_local.row(row).to_array().map(f32::to_bits));
            }
            for row in 0..3 {
                data.push(world.row(row).to_array().map(f32::to_bits));
            }
            data.push([entry.map_or(0, |e| e.first_record), instance.material.0, entry.map_or(0, |e| e.layers), scale.to_bits()]);
            entries.push(InstanceEntry::new(mesh_index[m], bounds, entry.map_or(1.0, |e| e.texel * scale)));
        }
        self.uniform.counts = [if meshes.is_empty() { 0 } else { instances.len() as u32 }, data.len() as u32, ATLAS_PAGES, 0];
        data.extend_from_slice(&records);
        self.data = storage_buffer_init(device, queue, "card data", bytemuck::cast_slice(&data));

        let with_cards = entries.iter().filter(|e| e.mesh.is_some()).count();
        self.allocator = Allocator::new(meshes, entries);
        self.pages = storage_buffer_init(device, queue, "card pages", bytemuck::cast_slice(&self.allocator.page_words));
        log::info!(
            "lumen cards: {} meshes captured ({} cards, capture atlas {}x{}, {:.1} MiB), {} instances with cards",
            self.allocator.meshes.len(),
            sizes.len(),
            CAPTURE_ATLAS_WIDTH,
            height,
            atlas.len() as f64 * 4.0 / 1048576.0,
            with_cards
        );
    }

    /// Chooses every instance's lighting mip for the camera position, (re)allocates pages within
    /// the frame's budget, schedules refreshes and uploads the changes.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera: Vec3, frame: u32) -> CardUniform {
        let mut updates = Vec::new();
        if self.uniform.counts[0] > 0 {
            updates = self.allocator.update(camera, frame);
            let a = &mut self.allocator;
            for chunk in 0..a.dirty.len() {
                if std::mem::take(&mut a.dirty[chunk]) {
                    let range = chunk * CHUNK_WORDS..((chunk + 1) * CHUNK_WORDS).min(a.page_words.len());
                    queue.write_buffer(&self.pages, range.start as u64 * 4, bytemuck::cast_slice(&a.page_words[range]));
                }
            }
        }
        self.update_count = updates.len().min(65535);
        if self.update_count > self.update_capacity {
            self.update_capacity = self.update_count.next_power_of_two();
            self.updates = storage_buffer(device, "card updates", (self.update_capacity * 16) as u64, wgpu::BufferUsages::empty());
        }
        if self.update_count > 0 {
            queue.write_buffer(&self.updates, 0, bytemuck::cast_slice(&updates[..self.update_count]));
        }
        self.uniform
    }
}

impl Allocator {
    /// Lays out the page tables of `instances` (all cards unallocated).
    fn new(meshes: Vec<MeshEntry>, mut instances: Vec<InstanceEntry>) -> Self {
        let mut table = (instances.len() * CARD_SLOTS) as u32;
        for instance in &mut instances {
            instance.table = table;
            table += instance.mesh.map_or(0, |m| meshes[m].table_size);
        }
        let page_words = vec![NONE; table as usize];
        Self {
            dirty: vec![false; page_words.len().div_ceil(CHUNK_WORDS)],
            page_words,
            meshes,
            instances,
            free: (0..PAGE_COUNT as u32).rev().collect(),
            owners: vec![[0, NONE, 0, 0]; PAGE_COUNT],
            scheduled: vec![u32::MAX; PAGE_COUNT],
            refresh_cursor: 0,
            bias: 0,
            frame: 0,
        }
    }

    /// Chooses every instance's mip for `camera`, (re)allocates pages and returns the pages to
    /// light this frame.
    fn update(&mut self, camera: Vec3, frame: u32) -> Vec<[u32; 4]> {
        self.frame = frame;
        let mut updates = Vec::new();
        self.choose_bias(camera);
        self.allocate(camera, &mut updates);
        self.refresh(&mut updates);
        updates
    }

    fn distance(&self, i: usize, camera: Vec3) -> f32 {
        let (lo, hi) = self.instances[i].bounds;
        (lo - camera).max(camera - hi).max(Vec3::ZERO).length()
    }

    fn want(&self, i: usize, camera: Vec3, bias: u32, current: Option<u32>) -> Option<u32> {
        let instance = &self.instances[i];
        let mesh = &self.meshes[instance.mesh?];
        desired_mip(self.distance(i, camera), instance.texel, mesh.max_mip(), bias, current)
    }

    /// Pages all instances would need at `bias`.
    fn pages_wanted(&self, camera: Vec3, bias: u32) -> usize {
        (0..self.instances.len())
            .filter_map(|i| Some(self.meshes[self.instances[i].mesh?].pages[self.want(i, camera, bias, None)? as usize] as usize))
            .sum()
    }

    /// Coarsens all mips while the wanted pages would overfill the atlas, refines them again
    /// once they comfortably fit (the gap between the thresholds avoids flipping back and forth).
    fn choose_bias(&mut self, camera: Vec3) {
        let capacity = PAGE_COUNT as f32;
        let previous = self.bias;
        while self.bias < MAX_BIAS && self.pages_wanted(camera, self.bias) as f32 > 0.9 * capacity {
            self.bias += 1;
        }
        while self.bias > 0 && self.pages_wanted(camera, self.bias - 1) as f32 <= 0.6 * capacity {
            self.bias -= 1;
        }
        if self.bias != previous {
            log::debug!("lumen cards: LOD bias {previous} -> {} ({} pages in use)", self.bias, PAGE_COUNT - self.free.len());
        }
    }

    /// Moves instances to the mips they want, closest first, as far as the frame's budget and
    /// free pages allow.
    fn allocate(&mut self, camera: Vec3, updates: &mut Vec<[u32; 4]>) {
        let mut requests: Vec<(f32, usize, Option<u32>)> = (0..self.instances.len())
            .filter(|&i| self.instances[i].mesh.is_some())
            .filter_map(|i| {
                let want = self.want(i, camera, self.bias, self.instances[i].mip);
                (want != self.instances[i].mip).then(|| (self.distance(i, camera), i, want))
            })
            .collect();
        requests.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let budget = UPDATE_BUDGET * 3 / 4;
        for (_, i, want) in requests {
            let Some(mip) = want else {
                self.release(i);
                continue;
            };
            let need = self.meshes[self.instances[i].mesh.unwrap()].pages[mip as usize] as usize;
            // One request may exceed the budget on its own (a huge card set), but not join others.
            if updates.len() + need > budget && !updates.is_empty() {
                continue;
            }
            if need > self.free.len() + self.instances[i].pages.len() {
                continue;
            }
            self.release(i);
            self.assign(i, mip, updates);
        }
    }

    /// Frees an instance's pages.
    fn release(&mut self, i: usize) {
        let instance = &mut self.instances[i];
        for &page in &instance.pages {
            self.owners[page as usize][1] = NONE;
            self.free.push(page);
        }
        instance.pages.clear();
        instance.mip = None;
        for slot in 0..CARD_SLOTS {
            set_word(&mut self.page_words, &mut self.dirty, i * CARD_SLOTS + slot, NONE);
        }
    }

    /// Gives instance `i`'s cards pages at `mip` and schedules them all for lighting.
    fn assign(&mut self, i: usize, mip: u32, updates: &mut Vec<[u32; 4]>) {
        let mesh = &self.meshes[self.instances[i].mesh.unwrap()];
        let table = self.instances[i].table;
        for (slot, &(resolution, offset)) in mesh.slots.iter().enumerate() {
            if resolution[0] == 0 {
                continue;
            }
            let start = table + offset;
            set_word(&mut self.page_words, &mut self.dirty, i * CARD_SLOTS + slot, (start << 4) | mip);
            let [px, py] = mip_pages(resolution, mip);
            for y in 0..py {
                for x in 0..px {
                    let page = self.free.pop().expect("free pages were checked");
                    set_word(&mut self.page_words, &mut self.dirty, (start + y * px + x) as usize, page);
                    let record = [page, i as u32, slot as u32 | (mip << 8), x | (y << 16)];
                    self.owners[page as usize] = record;
                    self.scheduled[page as usize] = self.frame;
                    self.instances[i].pages.push(page);
                    updates.push(record);
                }
            }
        }
        self.instances[i].mip = Some(mip);
    }

    /// Fills the rest of the frame's budget with the allocated pages lit longest ago.
    fn refresh(&mut self, updates: &mut Vec<[u32; 4]>) {
        let mut scanned = 0;
        while updates.len() < UPDATE_BUDGET && scanned < PAGE_COUNT {
            let page = self.refresh_cursor;
            self.refresh_cursor = (self.refresh_cursor + 1) % PAGE_COUNT;
            scanned += 1;
            if self.owners[page][1] != NONE && self.scheduled[page] != self.frame {
                self.scheduled[page] = self.frame;
                updates.push(self.owners[page]);
            }
        }
    }
}

fn set_word(words: &mut [u32], dirty: &mut [bool], index: usize, value: u32) {
    words[index] = value;
    dirty[index / CHUNK_WORDS] = true;
}

fn capture_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("card capture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: F::R32Uint,
        usage: U::TEXTURE_BINDING | U::COPY_DST,
        view_formats: &[],
    })
}

/// Capture texel size (mesh units) of a mesh with `extent`, instanced at up to `scale`, for a
/// target world size of `texel`.
fn capture_texel(extent: Vec3, scale: f32, texel: f32) -> f32 {
    (texel / scale).max(extent.max_element() / MAX_CARD_RESOLUTION as f32)
}

/// World units per mesh unit (the largest axis scale of `transform`).
fn max_scale(transform: glam::Mat4) -> f32 {
    let linear = glam::Mat3::from_mat4(transform);
    linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length()).max(1e-6)
}

/// Captures the meshes with a scale (`mesh_scale`, 0: unused) with world texels grown uniformly
/// from [`CARD_TEXEL`] until all cards fit [`CAPTURE_BUDGET`] (estimated from the bounding boxes
/// first, then checked).
fn capture_meshes(scene: &Scene, mesh_scale: &[f32]) -> Vec<Option<MeshCards>> {
    let used_meshes = || scene.meshes.iter().zip(mesh_scale).filter(|(_, &s)| s > 0.0);
    // Texels of the bounding-box faces: about one layer per direction.
    let estimate = |texel: f32| -> f32 {
        used_meshes()
            .map(|(mesh, &scale)| {
                let e = mesh.aabb.extent().max(Vec3::splat(1e-3));
                let t = capture_texel(e, scale, texel);
                2.0 * (e.x * e.y + e.y * e.z + e.z * e.x) / (t * t)
            })
            .sum()
    };
    let mut texel = CARD_TEXEL;
    while estimate(texel) > CAPTURE_BUDGET {
        texel *= 1.25;
    }
    loop {
        let captures: Vec<Option<MeshCards>> = scene
            .meshes
            .iter()
            .zip(mesh_scale)
            .map(|(mesh, &scale)| {
                (scale > 0.0).then(|| {
                    let config = CardConfig {
                        texel_size: texel / scale,
                        max_resolution: MAX_CARD_RESOLUTION,
                        two_sided: mesh.sdf.as_ref().is_some_and(|s| s.two_sided),
                        ..Default::default()
                    };
                    let size = capture_texel(mesh.aabb.extent(), scale, texel);
                    MeshCards::capture(&mesh_triangles(mesh, size * 0.5), mesh.aabb, &config)
                })
            })
            .collect();
        let texels: usize = captures.iter().flatten().map(MeshCards::texel_count).sum();
        if texels as f32 <= CAPTURE_BUDGET || texel >= 1.0 {
            log::debug!("lumen cards: {texels} texels captured (target texel {texel:.2} m)");
            return captures;
        }
        texel *= (texels as f32 / CAPTURE_BUDGET).sqrt() * 1.05;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mips_follow_distance_with_hysteresis() {
        // 0.1 m texels: mip 0 up to 20 m, then one mip per doubling.
        assert_eq!(desired_mip(5.0, 0.1, 5, 0, None), Some(0));
        assert_eq!(desired_mip(25.0, 0.1, 5, 0, None), Some(1));
        assert_eq!(desired_mip(50.0, 0.1, 5, 0, None), Some(2));
        assert_eq!(desired_mip(50.0, 0.1, 1, 0, None), Some(1), "clamped to the coarsest useful mip");
        assert_eq!(desired_mip(25.0, 0.1, 5, 1, None), Some(2), "bias coarsens");
        assert_eq!(desired_mip(MAX_DISTANCE + 1.0, 0.1, 5, 0, None), None);
        // Hysteresis: mip 1 is kept a little closer than where it starts.
        assert_eq!(desired_mip(18.0, 0.1, 5, 0, Some(1)), Some(1));
        assert_eq!(desired_mip(15.0, 0.1, 5, 0, Some(1)), Some(0));
        assert_eq!(desired_mip(45.0, 0.1, 5, 0, Some(1)), Some(2));
        assert_eq!(desired_mip(MAX_DISTANCE + 1.0, 0.1, 5, 0, Some(3)), Some(3), "released a little farther out");
    }

    #[test]
    fn mip_pages_cover_the_card() {
        assert_eq!(mip_pages([70, 250], 0), [9, 32]);
        assert_eq!(mip_pages([70, 250], 2), [3, 8]);
        assert_eq!(mip_pages([3, 1], 4), [1, 1]);
        assert_eq!(mip_resolution(250, 3), 32);
        assert_eq!(mip_resolution(1, 5), 1);
    }

    #[test]
    fn shelves_do_not_overlap() {
        let sizes: Vec<[u32; 2]> = (0..300u32).map(|i| [1 + (i * 37) % 256, 1 + (i * 53) % 200]).collect();
        let (positions, height) = pack_shelves(&sizes, 1024);
        for (i, (a, sa)) in positions.iter().zip(&sizes).enumerate() {
            assert!(a[0] + sa[0] <= 1024 && a[1] + sa[1] <= height);
            for (b, sb) in positions.iter().zip(&sizes).skip(i + 1) {
                assert!(a[0] + sa[0] <= b[0] || b[0] + sb[0] <= a[0] || a[1] + sa[1] <= b[1] || b[1] + sb[1] <= a[1]);
            }
        }
    }

    /// Cards of a sphere of radius `radius` (6 cards, ~`2 * radius / 0.1` texels across).
    fn sphere_cards(radius: f32) -> MeshCards {
        let mesh = helios_geometry::primitives::icosphere(2).transformed(glam::Mat4::from_scale(Vec3::splat(radius)));
        let soup: Vec<[CardVertex; 3]> =
            mesh.triangles().map(|t| t.map(|i| (mesh.vertices[i as usize].position(), mesh.vertices[i as usize].normal()))).collect();
        MeshCards::capture(&soup, mesh.aabb(), &CardConfig::default())
    }

    /// A row of `count` spheres along +x, `spacing` apart, of two sizes.
    fn row(count: usize, spacing: f32) -> Allocator {
        let meshes = vec![MeshEntry::new(&sphere_cards(1.0), 0), MeshEntry::new(&sphere_cards(4.0), CARD_SLOTS as u32)];
        let instances = (0..count)
            .map(|i| {
                let c = Vec3::new(i as f32 * spacing, 0.0, 0.0);
                let (mesh, r) = if i % 3 == 0 { (1, 4.0) } else { (0, 1.0) };
                InstanceEntry::new(Some(mesh), (c - r, c + r), meshes[mesh].texel)
            })
            .collect();
        Allocator::new(meshes, instances)
    }

    /// Every allocated card's page table points at pages owned by exactly that card texel range.
    fn check_consistent(a: &Allocator) {
        let mut owned = vec![false; PAGE_COUNT];
        for (i, instance) in a.instances.iter().enumerate() {
            let mesh = &a.meshes[instance.mesh.unwrap()];
            let Some(mip) = instance.mip else {
                assert!(instance.pages.is_empty());
                assert!((0..CARD_SLOTS).all(|slot| a.page_words[i * CARD_SLOTS + slot] == NONE));
                continue;
            };
            assert_eq!(instance.pages.len(), mesh.pages[mip as usize] as usize);
            for (slot, &(resolution, offset)) in mesh.slots.iter().enumerate() {
                let word = a.page_words[i * CARD_SLOTS + slot];
                if resolution[0] == 0 {
                    assert_eq!(word, NONE);
                    continue;
                }
                assert_eq!(word, ((instance.table + offset) << 4) | mip);
                let [px, py] = mip_pages(resolution, mip);
                for y in 0..py {
                    for x in 0..px {
                        let page = a.page_words[(instance.table + offset + y * px + x) as usize];
                        assert_eq!(a.owners[page as usize], [page, i as u32, slot as u32 | (mip << 8), x | (y << 16)]);
                        assert!(!std::mem::replace(&mut owned[page as usize], true), "page {page} owned twice");
                    }
                }
            }
        }
        assert_eq!(owned.iter().filter(|&&o| o).count() + a.free.len(), PAGE_COUNT);
        assert!(a.free.iter().all(|&p| a.owners[p as usize][1] == NONE));
    }

    #[test]
    fn allocation_follows_the_camera() {
        let mut a = row(40, 10.0);
        let mut camera = Vec3::new(0.0, 2.0, 0.0);
        let mut lit = std::collections::HashSet::new();
        for frame in 0..60 {
            for update in a.update(camera, frame) {
                lit.insert(update[0]);
                assert!(update[0] < PAGE_COUNT as u32);
            }
            check_consistent(&a);
        }
        // Everything within reach got cards, finest close to the camera; every page was lit.
        for (i, instance) in a.instances.iter().enumerate() {
            let distance = a.distance(i, camera);
            assert_eq!(instance.mip.is_some(), distance <= MAX_DISTANCE, "instance {i} at {distance} m");
        }
        assert_eq!(a.instances[1].mip, Some(0));
        assert!(a.instances[9].mip.unwrap() > a.instances[3].mip.unwrap());
        assert!(a.instances.iter().flat_map(|i| &i.pages).all(|p| lit.contains(p)));

        // Fly to the other end: near instances refine, far ones are released.
        camera.x = 390.0;
        for frame in 60..140 {
            a.update(camera, frame);
            check_consistent(&a);
        }
        assert!(a.instances[0].mip.is_none() && a.instances[39].mip == Some(0));
    }

    #[test]
    fn new_cards_are_lit_in_the_frame_they_are_allocated() {
        let mut a = row(12, 3.0);
        for frame in 0..5 {
            let updates = a.update(Vec3::ZERO, frame);
            let scheduled: std::collections::HashSet<u32> = updates.iter().map(|u| u[0]).collect();
            assert_eq!(scheduled.len(), updates.len(), "a page is lit at most once per frame");
            for instance in &a.instances {
                if instance.mip.is_some() {
                    // Allocated before or now: its pages were lit now or in an earlier frame.
                    assert!(instance.pages.iter().all(|p| a.scheduled[*p as usize] <= frame));
                }
            }
            assert!(updates.len() <= UPDATE_BUDGET.max(a.meshes[1].pages[0] as usize));
        }
    }

    #[test]
    fn a_full_atlas_coarsens_the_bias() {
        // Thousands of large instances around the camera want far more pages than exist.
        let meshes = vec![MeshEntry::new(&sphere_cards(6.0), 0)];
        let instances = (0..3000)
            .map(|i| {
                let c = Vec3::new((i % 60) as f32 * 3.0 - 90.0, 0.0, (i / 60) as f32 * 3.0 - 75.0);
                InstanceEntry::new(Some(0), (c - 6.0, c + 6.0), meshes[0].texel)
            })
            .collect();
        let mut a = Allocator::new(meshes, instances);
        let missing = |a: &Allocator| (0..a.instances.len()).filter(|&i| a.instances[i].mip.is_none() && a.distance(i, Vec3::ZERO) <= MAX_DISTANCE).count();
        let mut frame = 0;
        while (frame == 0 || missing(&a) > 0) && frame < 5000 {
            a.update(Vec3::ZERO, frame);
            frame += 1;
        }
        check_consistent(&a);
        assert!(a.bias > 0);
        assert!(a.pages_wanted(Vec3::ZERO, a.bias) as f32 <= 0.9 * PAGE_COUNT as f32);
        assert_eq!(missing(&a), 0, "everything in reach fits at the coarser bias");
    }
}
