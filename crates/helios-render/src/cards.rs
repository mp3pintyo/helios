//! Lumen surface cache made of mesh cards (UE5 Lumen's card-based surface cache).
//!
//! The voxel surface cache (`lumen.rs`) stores the light leaving the surfaces inside each voxel
//! of a 0.5–4 m grid, so a thin wall's sunlit outside bleeds into its inside and small objects
//! blur into their surroundings. Cards store lighting on the surfaces themselves:
//!
//! * **Capture** (CPU, when the scene changes): every mesh with a volume distance field is
//!   captured by up to 6 directions × 3 depth layers of orthographic cards — depth and normal
//!   per texel, ~10 cm texels (coarser for very large meshes), rasterised from a LOD cut of the
//!   virtual mesh ([`helios_geometry::cards`]), or from the displaced surface of a displaced
//!   variant (`lumen_displaced.rs`). Captures are per mesh (or variant), packed into one capture
//!   atlas. The capture also records, per texel, the texture coordinates of the surface point
//!   and — for a displaced variant — the base point it was displaced from ([`SurfaceCapture`]).
//! * **Material capture** (GPU, `lumen_card_capture.wgsl`; UE5 renders each card's material into
//!   the surface cache's albedo and emissive atlases): per (surface, material) pair — and per
//!   instance where the material depends on where the instance stands (world-space textures and
//!   graphs, a rock's moss on upward faces) — the material is evaluated at every texel as the
//!   G-buffer resolve evaluates it (image and procedural textures, shading models, material
//!   graphs) into an albedo atlas (rgba8: albedo, emission). Captures run within a texel budget
//!   per frame when a scene loads, and again when a material, its graph or its parameters
//!   change; until a pair is captured its cards use the material's average albedo (base colour
//!   × the texture's 1×1 mip, as the global distance field does). Unchanged pairs keep their
//!   albedo through scene re-uploads (copied into the new atlas).
//! * **Allocation**: each instance's cards are lit at a *mip* of their capture chosen from the
//!   instance's distance to the camera (lighting texels of ~1 % of that distance), in 8×8
//!   texel pages of a 2048² lighting atlas reached through per-card page tables. Instances
//!   beyond [`MAX_DISTANCE`] get no cards; a global LOD bias coarsens all mips when the atlas
//!   runs full. An instance switches mip only when all its new pages can be lit in the same
//!   frame, so its cards never show unlit pages.
//! * **Lighting** (`lumen_card_light.wgsl`): a budget of pages per frame — newly allocated
//!   ones first (closest instances first), then a round-robin refresh of the rest. A texel's
//!   light is reflected with the captured albedo of the surface it covers (averaged over up to
//!   four captured texels at coarse mips) and adds its captured emission.
//! * **Sampling** (`lumen_cards.wgsl`): screen-probe and reflection hits are projected into the
//!   hit instance's cards; the near field reports the instance it hit, the global distance
//!   field stores the closest instance per voxel. Where no card matches, the voxel surface
//!   cache stays in place. The bilinear filter over lit texels is depth-aware: each lit texel
//!   stores the depth of the surface it was lit at, and only texels on the hit's tangent plane
//!   contribute (no light bleeding across depth discontinuities).
//! * **Voxel albedo**: the capture sums each pair's texels (`pair_sums`); the global distance
//!   field stores the closest instance's pair mean per voxel (`lumen_sdf.wgsl`), which the voxel
//!   surface cache reflects with. Lumen recomposes the field when captures finish.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};
use helios_geometry::cards::{Card, CardConfig, CardVertex, MeshCards, CARD_AXES, MAX_CARD_LAYERS};
use helios_geometry::{lod_cut_resident, Aabb, LodView, SdfKind, VirtualMesh};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{TextureFormat as F, TextureUsages as U};

use crate::lumen_displaced::Surface;
use crate::profiler::Recorder;
use crate::scene::{MaterialKind, Scene, TextureMapping};
use crate::shader::{compose_variant, create_module_from};
use crate::substrate_gpu::ExtendedMaterials;
use crate::textures::TextureLibrary;
use crate::util::{bind, bind_group, bind_group_layout, compute_pipeline_with, storage_buffer, Lazy, storage_buffer_init};
use crate::pipelines::{in_group, Group};

/// Card slots per mesh / instance: directions × layers (`CARD_SLOTS` in lumen_cards.wgsl).
const CARD_SLOTS: usize = CARD_AXES * MAX_CARD_LAYERS;
/// Texels per side of a lighting page (`CARD_PAGE_TEXELS`).
const PAGE_TEXELS: u32 = 8;
/// The lighting atlas holds `ATLAS_PAGES`² pages (2048² rgba16f texels, 32 MiB).
const ATLAS_PAGES: u32 = 256;
const PAGE_COUNT: usize = (ATLAS_PAGES * ATLAS_PAGES) as usize;
/// Pages lit per frame: new allocations take up to three quarters, refreshes the rest.
const UPDATE_BUDGET: usize = 1024;
/// Flag in an update record's third word (`FRESH_PAGE` in lumen_card_light.wgsl): the page was
/// just (re)allocated, so the local light accumulated for its texels (lumen_light_grid.rs)
/// belongs to another surface.
const FRESH_PAGE: u32 = 1 << 31;
/// Target capture texel size (world metres) and the card side limit.
pub(crate) const CARD_TEXEL: f32 = 0.1;
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
/// Width of the albedo atlas (rgba8, 4 bytes per texel).
const ALBEDO_ATLAS_WIDTH: u32 = 4096;
/// Albedo texels of all (surface, material) pairs at most (16 MiB); pairs beyond keep the
/// material's average albedo.
const ALBEDO_BUDGET: usize = 4 << 20;
/// Of that, texels for per-instance captures of materials that depend on the placement; further
/// instances share their pair's capture (made at its first instance's placement).
const PLACED_BUDGET: usize = 1 << 20;
/// Texels whose material is captured per frame (at least one pair): 16 k workgroups, at most ~0.3 ms.
const CAPTURE_TEXELS_PER_FRAME: usize = 1 << 20;
/// Tile records of a capture dispatch start on 256-byte boundaries (storage buffer offsets).
const TILE_ALIGN: usize = 16;
/// u32 sums per pair of its captured texels: albedo rgb and emission (bytes), count
/// (`PAIR_SUM_WORDS` in lumen_card_capture.wgsl and lumen_sdf.wgsl).
const PAIR_SUM_WORDS: usize = 5;

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

/// Triangles of `mesh` (mesh space) at about `error` geometric error, from the pages in memory,
/// and their corners' material sources.
fn mesh_triangles(mesh: &VirtualMesh, error: f32) -> (Vec<[CardVertex; 3]>, Vec<[CornerSource; 3]>) {
    let view = LodView { position: Vec3::ZERO, lod_scale: 1.0, threshold: error, orthographic: true };
    let mut triangles = Vec::new();
    let mut sources = Vec::new();
    for cluster in lod_cut_resident(mesh, &view, &|page| mesh.has_payload(page as usize)) {
        let c = cluster as usize;
        let vertices: Vec<(Vec3, Vec3, Vec2)> = (0..mesh.clusters[c].vertex_count()).map(|i| mesh.cluster_vertex(c, i)).collect();
        for t in mesh.cluster_triangles(c) {
            let corners = t.map(|i| vertices[i as usize]);
            triangles.push(corners.map(|(p, n, _)| (p, n)));
            sources.push(corners.map(|(p, _, uv)| (p, uv)));
        }
    }
    (triangles, sources)
}

/// What the material capture needs of a triangle corner besides its position: the point of the
/// base surface it stands for (mesh space; a displaced variant's point before displacement,
/// where the resolve evaluates the material, else the corner itself) and its texture
/// coordinates (UV set 0).
pub(crate) type CornerSource = (Vec3, Vec2);

/// A surface's mesh cards and the material inputs of their texels.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SurfaceCapture {
    pub cards: MeshCards,
    /// Per card and texel (`attributes` in lumen_card_capture.wgsl): uv (2 × f32), the base
    /// point's offset from the texel's surface point (3 × f16) and the texture coordinate change
    /// across the texel (f16); zero where the texel sees no surface.
    pub attributes: Vec<Vec<[u32; 4]>>,
}

impl SurfaceCapture {
    /// Captured texels (GPU atlas) plus their material inputs (kept on the CPU for recaptures).
    pub fn bytes(&self) -> usize {
        self.cards.texel_count() * 20
    }
}

/// Captures the cards of `triangles` (with their corners' `sources`) and the material inputs of
/// every texel.
pub(crate) fn capture_cards(triangles: &[[CardVertex; 3]], sources: &[[CornerSource; 3]], bounds: Aabb, config: &CardConfig) -> SurfaceCapture {
    debug_assert_eq!(triangles.len(), sources.len());
    let mut cards = MeshCards::capture(triangles, bounds, &CardConfig { sources: true, ..*config });
    let texel = cards.texel_size;
    let attributes = cards
        .cards
        .iter_mut()
        .map(|card| {
            let texel_sources = std::mem::take(&mut card.sources);
            let (_, u, v) = Card::coordinates(card.axis);
            let width = card.resolution[0];
            (0..card.texels.len())
                .map(|k| {
                    let Some((p, _)) = card.texel_surface(k as u32 % width, k as u32 / width, texel) else { return [0; 4] };
                    let s = texel_sources[k];
                    let (t, c) = (&triangles[s.triangle as usize], &sources[s.triangle as usize]);
                    let [b1, b2] = s.barycentrics;
                    let b0 = 1.0 - b1 - b2;
                    let base = c[0].0 * b0 + c[1].0 * b1 + c[2].0 * b2;
                    let uv = c[0].1 * b0 + c[1].1 * b1 + c[2].1 * b2;
                    let footprint = uv_footprint(t, c.map(|c| c.1), u, v) * texel;
                    let offset = base - p;
                    [uv.x.to_bits(), uv.y.to_bits(), f16_bits(offset.x) | (f16_bits(offset.y) << 16), f16_bits(offset.z) | (f16_bits(footprint.min(1.0e4)) << 16)]
                })
                .collect()
        })
        .collect();
    SurfaceCapture { cards, attributes }
}

/// The larger texture coordinate change of triangle `t` (with corner `uv`s) per mesh unit along
/// a card's plane axes `u` and `v` (0 for a triangle edge-on to the card).
fn uv_footprint(t: &[CardVertex; 3], uv: [Vec2; 3], u: usize, v: usize) -> f32 {
    let p = t.map(|c| Vec2::new(c.0[u], c.0[v]));
    let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
    let det = e1.perp_dot(e2);
    if det.abs() < 1e-12 {
        return 0.0;
    }
    let (f1, f2) = (uv[1] - uv[0], uv[2] - uv[0]);
    // A plane step d = α e1 + β e2 changes the coordinates by α f1 + β f2.
    let along_u = (f1 * e2.y - f2 * e1.y) / det;
    let along_v = (f2 * e1.x - f1 * e2.x) / det;
    along_u.length().max(along_v.length())
}

/// IEEE half-precision bits of `x` (rounded to nearest; overflow saturates, magnitudes below the
/// smallest normal half flush to zero), as WGSL's `unpack2x16float` reads them.
fn f16_bits(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    if x.is_nan() {
        return sign | 0x7e00;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7bff;
    }
    let mantissa = bits & 0x7f_ffff;
    let half = ((exponent as u32) << 10) | (mantissa >> 13);
    sign | (half + ((mantissa >> 12) & 1)).min(0x7bff)
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
    /// Moved instances whose pages are relit before the round-robin refresh.
    relight: Vec<usize>,
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
    /// Captures by [`capture_key`], the geometry version they were made at and the target texel
    /// of the last full capture.
    captures: HashMap<u64, Capture>,
    capture_version: u64,
    capture_texel: f32,
    /// Serial of the next capture (identifies a capture's texels for its albedo).
    next_serial: u64,
    allocator: Allocator,
    uniform: CardUniform,
    /// Captured materials (rgba8: sqrt(albedo), emission e / (1 + e)), read by `light_cards`.
    albedo_texture: wgpu::Texture,
    pub albedo: wgpu::TextureView,
    /// The (surface, material) pairs of the last upload, their mesh entries' order.
    pairs: Vec<AlbedoPair>,
    /// Pairs whose material is to be (re)captured, in order.
    pending: VecDeque<usize>,
    /// Graph and parameter versions the pairs' contents were computed at.
    material_versions: (u64, u64),
    /// Surface code of the scenes' graphs by hash (the capture pipelines' variants).
    graph_code: HashMap<u64, String>,
    capture_pipelines: Option<CapturePipelines>,
    /// Texels whose material was captured in the latest frame.
    pub captured_texels: usize,
    /// Per pair, the sums of its captured texels (the global distance field's voxel albedo).
    pub pair_sums: wgpu::Buffer,
    /// Captures finished since [`CardCache::take_albedo_changed`] was last called.
    albedo_changed: bool,
}

/// A surface's capture and its serial.
struct Capture {
    surface: SurfaceCapture,
    serial: u64,
}

/// What a material capture is of: a capture's texels, a material and, for materials that depend
/// on the instance's placement, the scene instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PairKey {
    capture: u64,
    material: u32,
    placement: Option<usize>,
}

/// A (surface, material) pair: one mesh entry (card records) of the upload and its material
/// capture.
struct AlbedoPair {
    key: PairKey,
    /// Key of the surface's capture in `captures`.
    surface: u64,
    /// Card instance whose placement and material are captured.
    representative: u32,
    /// The capture pipeline: shading model and graph hash (0: none); `None`: no albedo
    /// (terrain, water), the cards use the material's average.
    pipeline: Option<(u32, u64)>,
    /// Hash of everything the albedo depends on (see [`material_content`]).
    content: u64,
    /// Albedo atlas origin of each card (the capture's order); empty without albedo.
    origins: Vec<[u32; 2]>,
    first_record: u32,
    /// The content the albedo holds (captured, or copied from the previous atlas); `None`: not
    /// captured yet, the records point at no albedo.
    captured: Option<u64>,
}

/// Pipelines and layouts of the material capture.
struct CapturePipelines {
    layout: wgpu::BindGroupLayout,
    graph_layout: wgpu::BindGroupLayout,
    /// By (shading model, graph hash).
    pipelines: HashMap<(u32, u64), Lazy<wgpu::ComputePipeline>>,
    modules: HashMap<u64, wgpu::ShaderModule>,
}

/// What the material capture reads besides the cache's own resources.
pub struct CaptureInputs<'a> {
    pub frame: &'a wgpu::Buffer,
    pub materials: &'a wgpu::Buffer,
    pub textures: &'a TextureLibrary,
    pub sampler: &'a wgpu::Sampler,
    pub extended: &'a ExtendedMaterials,
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
        let albedo_texture = albedo_atlas(device, 1, 1);
        Self {
            data: storage_buffer(device, "card data", 16, wgpu::BufferUsages::empty()),
            pages: storage_buffer(device, "card pages", 16, wgpu::BufferUsages::empty()),
            capture: capture_texture(device, 1, 1).create_view(&Default::default()),
            lighting: lighting.create_view(&Default::default()),
            updates: storage_buffer(device, "card updates", 16, wgpu::BufferUsages::empty()),
            update_capacity: 1,
            update_count: 0,
            captures: HashMap::new(),
            capture_version: u64::MAX,
            capture_texel: CARD_TEXEL,
            next_serial: 0,
            allocator: Allocator::default(),
            uniform: CardUniform::default(),
            albedo: albedo_texture.create_view(&Default::default()),
            albedo_texture,
            pairs: Vec::new(),
            pending: VecDeque::new(),
            material_versions: (u64::MAX, u64::MAX),
            graph_code: HashMap::new(),
            capture_pipelines: None,
            captured_texels: 0,
            pair_sums: storage_buffer(device, "card pair sums", 16, wgpu::BufferUsages::COPY_SRC),
            albedo_changed: false,
        }
    }

    /// Per card instance: 1 + its pair if the pair's material is captured (its sums then hold its
    /// captured texels once done), else 0.
    pub fn instance_pairs(&self) -> Vec<u32> {
        self.allocator.instances.iter().map(|i| i.mesh.filter(|&p| !self.pairs[p].origins.is_empty()).map_or(0, |p| p as u32 + 1)).collect()
    }

    /// Whether captures finished (every pending pair captured) since the last call.
    pub fn take_albedo_changed(&mut self) -> bool {
        std::mem::take(&mut self.albedo_changed)
    }

    /// Pages to light this frame (`light_cards` workgroups).
    pub fn update_count(&self) -> u32 {
        self.update_count as u32
    }

    /// Moves card instance `k` (a distance-field instance) to `world` with world `bounds`: its
    /// cards move along (no recapture) and its pages are relit first in the next update, since
    /// their lighting depends on where they are.
    pub fn move_instance(&mut self, queue: &wgpu::Queue, k: usize, world: glam::Mat4, bounds: (Vec3, Vec3)) {
        let Some(instance) = self.allocator.instances.get_mut(k) else { return };
        instance.bounds = bounds;
        let to_local = world.inverse();
        let rows: [[u32; 4]; 6] = std::array::from_fn(|r| if r < 3 { to_local.row(r) } else { world.row(r - 3) }.to_array().map(f32::to_bits));
        queue.write_buffer(&self.data, (k * INSTANCE_VEC4S * 16) as u64, bytemuck::cast_slice(&rows));
        if instance.mesh.is_some() && !self.allocator.relight.contains(&k) {
            self.allocator.relight.push(k);
        }
    }

    /// Captures the meshes (when the geometry changed) and lays out the cards of
    /// `instances` — the scene instances in the order of Lumen's distance-field instances,
    /// whose indices the shaders use — with their world `bounds`. Clears all allocations.
    /// `surfaces` are what Lumen represents meshes with (scene meshes, then displaced variants),
    /// `surface_of` every scene instance's.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        surfaces: &[Surface],
        surface_of: &[usize],
        instances: &[usize],
        bounds: &[(Vec3, Vec3)],
    ) {
        let start = web_time::Instant::now();
        // The surfaces to capture, with the largest scale they are instanced at (0: unused).
        let mut mesh_scale = vec![0.0f32; surfaces.len()];
        for &i in instances {
            let instance = &scene.instances[i];
            let m = surface_of[i];
            if surfaces[m].sdf().is_some_and(|s| s.kind == SdfKind::Volume) {
                mesh_scale[m] = mesh_scale[m].max(max_scale(instance.transform));
            }
        }
        let used: Vec<bool> = mesh_scale.iter().map(|&s| s > 0.0).collect();
        let key = |m: usize| capture_key(scene, surfaces, m);
        // A newly used mesh recaptures them all (they share the texel budget); a new displaced
        // variant is captured on its own at the current texel.
        let missing = (0..scene.meshes.len()).any(|m| used[m] && !self.captures.contains_key(&key(m)));
        if self.capture_version != scene.geometry_version || missing {
            self.capture_version = scene.geometry_version;
            let (captures, texel) = capture_meshes(surfaces, &mesh_scale);
            let serial = &mut self.next_serial;
            self.captures = captures
                .into_iter()
                .enumerate()
                .filter_map(|(m, c)| {
                    *serial += 1;
                    Some((key(m), Capture { surface: c?, serial: *serial }))
                })
                .collect();
            self.capture_texel = texel;
        }
        for m in scene.meshes.len()..surfaces.len() {
            if used[m] && !self.captures.contains_key(&key(m)) {
                self.next_serial += 1;
                let surface = capture_surface(&surfaces[m], mesh_scale[m], self.capture_texel);
                self.captures.insert(key(m), Capture { surface, serial: self.next_serial });
            }
        }
        let keep: HashSet<u64> = (0..surfaces.len()).filter(|&m| m < scene.meshes.len() || used[m]).map(key).collect();
        self.captures.retain(|k, _| keep.contains(k));

        // The capture atlas: every used surface's cards.
        let mut first_card: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut sizes = Vec::new();
        for m in (0..surfaces.len()).filter(|&m| used[m]) {
            let Some(c) = self.captures.get(&key(m)) else { continue };
            first_card.insert(key(m), (m, sizes.len()));
            sizes.extend(c.surface.cards.cards.iter().map(|c| c.resolution));
        }
        let (positions, height) = pack_shelves(&sizes, CAPTURE_ATLAS_WIDTH);
        let mut atlas = vec![u32::MAX; (CAPTURE_ATLAS_WIDTH * height.max(1)) as usize];
        for (surface, &(_, first)) in &first_card {
            for (card, &[x, y]) in self.captures[surface].surface.cards.cards.iter().zip(&positions[first..]) {
                for row in 0..card.resolution[1] {
                    let src = (row * card.resolution[0]) as usize;
                    let dst = ((y + row) * CAPTURE_ATLAS_WIDTH + x) as usize;
                    atlas[dst..dst + card.resolution[0] as usize].copy_from_slice(&card.texels[src..src + card.resolution[0] as usize]);
                }
            }
        }
        let capture = capture_texture(device, CAPTURE_ATLAS_WIDTH, height.max(1));
        queue.write_texture(
            capture.as_image_copy(),
            bytemuck::cast_slice(&atlas),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(CAPTURE_ATLAS_WIDTH * 4), rows_per_image: None },
            capture.size(),
        );
        crate::gpu_memory::retire_view(&self.capture);
        self.capture = capture.create_view(&Default::default());

        // (Surface, material) pairs: one mesh entry (card records) each; per instance where the
        // material depends on the placement, within PLACED_BUDGET.
        let mut pair_index: HashMap<PairKey, usize> = HashMap::new();
        let mut pairs: Vec<AlbedoPair> = Vec::new();
        let mut instance_pair = vec![None; instances.len()];
        let mut placed_texels = 0;
        for (k, &i) in instances.iter().enumerate() {
            let m = surface_of[i];
            let Some(c) = self.captures.get(&key(m)).filter(|_| used[m]) else { continue };
            let material = scene.instances[i].material.0;
            let pipeline = capture_pipeline(scene, material, &mut self.graph_code).filter(|_| !c.surface.attributes.is_empty());
            let texels = c.surface.cards.texel_count();
            let placed = pipeline.is_some() && placement_dependent(scene, material) && placed_texels + texels <= PLACED_BUDGET;
            let pair = PairKey { capture: c.serial, material, placement: placed.then_some(i) };
            let p = *pair_index.entry(pair).or_insert_with(|| {
                placed_texels += if placed { texels } else { 0 };
                pairs.push(AlbedoPair {
                    key: pair,
                    surface: key(m),
                    representative: k as u32,
                    pipeline,
                    content: material_content(scene, material, placed.then_some(scene.instances[i].transform)),
                    origins: Vec::new(),
                    first_record: (pairs.len() * CARD_SLOTS) as u32,
                    captured: None,
                });
                pairs.len() - 1
            });
            instance_pair[k] = Some(p);
        }

        // The albedo atlas: the cards of every pair with a material capture, within ALBEDO_BUDGET.
        let mut albedo_sizes = Vec::new();
        let mut albedo_owner = Vec::new();
        let mut albedo_texels = 0;
        for (p, pair) in pairs.iter().enumerate() {
            let cards = &self.captures[&pair.surface].surface.cards;
            if pair.pipeline.is_none() || albedo_texels + cards.texel_count() > ALBEDO_BUDGET {
                continue;
            }
            albedo_texels += cards.texel_count();
            albedo_sizes.extend(cards.cards.iter().map(|c| c.resolution));
            albedo_owner.extend(std::iter::repeat_n(p, cards.cards.len()));
        }
        let (albedo_positions, albedo_height) = pack_shelves(&albedo_sizes, ALBEDO_ATLAS_WIDTH);
        for (&p, &origin) in albedo_owner.iter().zip(&albedo_positions) {
            pairs[p].origins.push(origin);
        }
        let albedo_texture = albedo_atlas(device, ALBEDO_ATLAS_WIDTH, albedo_height.max(1));
        // Pairs whose material is unchanged keep their albedo: copied from the previous atlas.
        let previous: HashMap<PairKey, &AlbedoPair> = self.pairs.iter().filter(|p| p.captured.is_some()).map(|p| (p.key, p)).collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("card albedo copies") });
        let pair_sums = storage_buffer(device, "card pair sums", (pairs.len() * PAIR_SUM_WORDS * 4) as u64, wgpu::BufferUsages::COPY_SRC);
        let sum_bytes = (PAIR_SUM_WORDS * 4) as u64;
        let old_index: HashMap<PairKey, usize> = self.pairs.iter().enumerate().map(|(i, p)| (p.key, i)).collect();
        let mut copied = 0;
        for (p, pair) in pairs.iter_mut().enumerate() {
            let Some(old) = previous.get(&pair.key).filter(|old| old.captured == Some(pair.content) && !pair.origins.is_empty()) else { continue };
            for ((card, from), to) in self.captures[&pair.surface].surface.cards.cards.iter().zip(&old.origins).zip(&pair.origins) {
                let at = |texture, [x, y]: [u32; 2]| wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All };
                let size = wgpu::Extent3d { width: card.resolution[0], height: card.resolution[1], depth_or_array_layers: 1 };
                encoder.copy_texture_to_texture(at(&self.albedo_texture, *from), at(&albedo_texture, *to), size);
            }
            encoder.copy_buffer_to_buffer(&self.pair_sums, old_index[&pair.key] as u64 * sum_bytes, &pair_sums, p as u64 * sum_bytes, Some(sum_bytes));
            pair.captured = Some(pair.content);
            copied += 1;
        }
        crate::gpu_memory::retire_buffer(&self.pair_sums);
        self.pair_sums = pair_sums;
        queue.submit([encoder.finish()]);
        crate::gpu_memory::retire_texture(&self.albedo_texture);
        self.albedo = albedo_texture.create_view(&Default::default());
        self.albedo_texture = albedo_texture;
        self.pending = (0..pairs.len()).filter(|&p| pairs[p].captured.is_none() && !pairs[p].origins.is_empty()).collect();
        self.material_versions = (scene.graphs.version, scene.graphs.parameter_version);

        // Card records: per pair, its surface's cards with its albedo.
        let mut meshes = Vec::with_capacity(pairs.len());
        let mut records = vec![[0u32; 4]; pairs.len() * CARD_SLOTS * CARD_VEC4S];
        for pair in &pairs {
            let cards = &self.captures[&pair.surface].surface.cards;
            let entry = MeshEntry::new(cards, pair.first_record);
            let (m, first) = first_card[&pair.surface];
            let sdf_voxel = surfaces[m].sdf().map_or(0.0, |s| s.voxel_size);
            for (k, card) in cards.cards.iter().enumerate() {
                let [x, y] = positions[first + k];
                let slot = (card.axis * MAX_CARD_LAYERS as u32 + card.layer) as usize;
                let base = (entry.first_record as usize + slot) * CARD_VEC4S;
                let [d0, d1] = card.depth_range;
                let albedo = if pair.captured.is_some() { pair.origins[k][0] | (pair.origins[k][1] << 16) } else { NONE };
                records[base] = Vec4::new(card.origin.x, card.origin.y, entry.texel, card.plane).to_array().map(f32::to_bits);
                records[base + 1] = Vec4::new(d0, (d1 - d0) / 65534.0, sdf_voxel, 0.0).to_array().map(f32::to_bits);
                records[base + 2] = [card.resolution[0] | (card.resolution[1] << 16), x | (y << 16), albedo, 0];
            }
            meshes.push(entry);
        }

        // Instance records and the page-table layout.
        let mut entries = Vec::with_capacity(instances.len());
        let mut data: Vec<[u32; 4]> = Vec::with_capacity(instances.len() * INSTANCE_VEC4S + records.len());
        for ((&i, &bounds), &pair) in instances.iter().zip(bounds).zip(&instance_pair) {
            let instance = &scene.instances[i];
            let world = instance.transform;
            let scale = max_scale(world);
            let entry = pair.map(|p| &meshes[p]);
            let to_local = world.inverse();
            for row in 0..3 {
                data.push(to_local.row(row).to_array().map(f32::to_bits));
            }
            for row in 0..3 {
                data.push(world.row(row).to_array().map(f32::to_bits));
            }
            data.push([entry.map_or(0, |e| e.first_record), instance.material.0, entry.map_or(0, |e| e.layers), scale.to_bits()]);
            entries.push(InstanceEntry::new(pair, bounds, entry.map_or(1.0, |e| e.texel * scale)));
        }
        self.uniform.counts = [if meshes.is_empty() { 0 } else { instances.len() as u32 }, data.len() as u32, ATLAS_PAGES, 0];
        data.extend_from_slice(&records);
        crate::gpu_memory::retire_buffer(&self.data);
        self.data = storage_buffer_init(device, queue, "card data", bytemuck::cast_slice(&data));

        let with_cards = entries.iter().filter(|e| e.mesh.is_some()).count();
        self.allocator = Allocator::new(meshes, entries);
        crate::gpu_memory::retire_buffer(&self.pages);
        self.pages = storage_buffer_init(device, queue, "card pages", bytemuck::cast_slice(&self.allocator.page_words));
        let attribute_bytes: usize = self.captures.values().map(|c| c.surface.bytes()).sum::<usize>();
        log::info!(
            "lumen cards: {} surfaces captured ({} cards, capture atlas {}x{}, {:.1} MiB; {:.1} MiB of capture data on the CPU), {} instances with cards, uploaded in {:.0} ms",
            first_card.len(),
            sizes.len(),
            CAPTURE_ATLAS_WIDTH,
            height,
            atlas.len() as f64 * 4.0 / 1048576.0,
            attribute_bytes as f64 / 1048576.0,
            with_cards,
            start.elapsed().as_secs_f64() * 1000.0
        );
        log::info!(
            "lumen cards: {} (surface, material) pairs ({} per instance), {} with material captures ({} texels, albedo atlas {}x{}, {:.1} MiB), {} kept, {} to capture",
            pairs.len(),
            pairs.iter().filter(|p| p.key.placement.is_some()).count(),
            pairs.iter().filter(|p| !p.origins.is_empty()).count(),
            albedo_texels,
            ALBEDO_ATLAS_WIDTH,
            albedo_height,
            (ALBEDO_ATLAS_WIDTH * albedo_height.max(1)) as f64 * 4.0 / 1048576.0,
            copied,
            self.pending.len()
        );
        self.pairs = pairs;
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

impl CardCache {
    /// Schedules recaptures of the pairs whose graph or parameter values changed (other
    /// material and texture changes re-upload the scene, which recaptures what changed).
    pub fn sync_materials(&mut self, scene: &Scene) {
        let versions = (scene.graphs.version, scene.graphs.parameter_version);
        if versions == self.material_versions {
            return;
        }
        self.material_versions = versions;
        for (p, pair) in self.pairs.iter_mut().enumerate() {
            let placement = pair.key.placement.and_then(|i| scene.instances.get(i)).map(|i| i.transform);
            pair.pipeline = capture_pipeline(scene, pair.key.material, &mut self.graph_code).filter(|_| !pair.origins.is_empty());
            pair.content = material_content(scene, pair.key.material, placement);
            if pair.pipeline.is_some() && !pair.origins.is_empty() && pair.captured != Some(pair.content) && !self.pending.contains(&p) {
                self.pending.push_back(p);
            }
        }
    }

    /// Queues the capture pipelines of a scene about to be switched to: those of its visible
    /// instances' materials (with the graphs among `active`, those the resolve runs), whose
    /// cards are not known yet (`Renderer::prefetch_next_scene`).
    pub fn prefetch_captures(&mut self, device: &wgpu::Device, scene: &Scene, active: &HashSet<u64>) {
        let capture = self.capture_pipelines.get_or_insert_with(|| CapturePipelines::new(device));
        let materials: std::collections::BTreeSet<u32> = scene.instances.iter().filter(|i| i.visible).map(|i| i.material.0).collect();
        for material in materials {
            if let Some((kind, graph)) = capture_pipeline(scene, material, &mut self.graph_code) {
                let key = capture.pipeline(device, kind, if active.contains(&graph) { graph } else { 0 }, &self.graph_code);
                capture.pipelines[&key].prefetch();
            }
        }
    }

    /// Records the material capture of the next pending pairs (within
    /// [`CAPTURE_TEXELS_PER_FRAME`]) and points their records at their albedo; their instances'
    /// cards are relit next.
    pub fn record_capture(&mut self, rec: &mut Recorder, queue: &wgpu::Queue, inputs: &CaptureInputs) {
        self.captured_texels = 0;
        if self.pending.is_empty() {
            return;
        }
        // Graphs the resolve runs (the parameter block holds their values): others are captured
        // with their material's own shading model, as the resolve renders them then.
        let active: HashSet<u64> = inputs.extended.slots.iter().filter_map(|s| s.graph.as_ref().map(|g| g.0)).collect();
        // In the browser the capture pipelines of the pending pairs are queued and compile in the
        // background (a new scene's five took 5 s compiled one by one inside a frame); the
        // captures wait for them, their cards lit without albedo meanwhile.
        if crate::pipelines::deferring() {
            let device = rec.device;
            let capture = self.capture_pipelines.get_or_insert_with(|| CapturePipelines::new(device));
            let mut compiling = false;
            for &p in &self.pending {
                if let Some((kind, graph)) = self.pairs[p].pipeline.filter(|_| !self.pairs[p].origins.is_empty()) {
                    let key = capture.pipeline(device, kind, if active.contains(&graph) { graph } else { 0 }, &self.graph_code);
                    compiling |= !capture.pipelines[&key].is_made();
                }
            }
            if compiling {
                return;
            }
        }
        let mut groups: BTreeMap<(u32, u64), Vec<[u32; 4]>> = BTreeMap::new();
        let mut attributes: Vec<[u32; 4]> = Vec::new();
        let record_base = self.uniform.counts[1] as usize;
        while let Some(&p) = self.pending.front() {
            let pair = &self.pairs[p];
            let capture = &self.captures[&pair.surface].surface;
            let texels = capture.cards.texel_count();
            if self.captured_texels > 0 && self.captured_texels + texels > CAPTURE_TEXELS_PER_FRAME {
                break;
            }
            self.pending.pop_front();
            let Some((kind, graph)) = pair.pipeline.filter(|_| !pair.origins.is_empty()) else { continue };
            self.captured_texels += texels;
            let tiles = groups.entry((kind, if active.contains(&graph) { graph } else { 0 })).or_default();
            for (k, (card, texel_attributes)) in capture.cards.cards.iter().zip(&capture.attributes).enumerate() {
                let slot = (card.axis * MAX_CARD_LAYERS as u32 + card.layer) as usize;
                let record = (record_base + (pair.first_record as usize + slot) * CARD_VEC4S) as u32;
                let first = attributes.len() as u32;
                attributes.extend_from_slice(texel_attributes);
                for y in 0..card.resolution[1].div_ceil(8) {
                    for x in 0..card.resolution[0].div_ceil(8) {
                        tiles.push([record, pair.representative, first, x | (y << 8) | (card.axis << 16)]);
                    }
                }
                if k == 0 {
                    // Its sums start again (the capture adds every texel).
                    queue.write_buffer(&self.pair_sums, (p * PAIR_SUM_WORDS * 4) as u64, &[0; PAIR_SUM_WORDS * 4]);
                }
                if pair.captured.is_none() {
                    // The record's albedo origin (read by the capture as its target, too).
                    let [ox, oy] = pair.origins[k];
                    queue.write_buffer(&self.data, (record as u64 + 2) * 16 + 8, bytemuck::bytes_of(&(ox | (oy << 16))));
                }
            }
            let pair = &mut self.pairs[p];
            pair.captured = Some(pair.content);
            for (i, instance) in self.allocator.instances.iter().enumerate() {
                if instance.mesh == Some(p) && instance.mip.is_some() && !self.allocator.relight.contains(&i) {
                    self.allocator.relight.push(i);
                }
            }
        }
        if groups.is_empty() {
            return;
        }
        self.albedo_changed |= self.pending.is_empty();

        let device = rec.device;
        let capture = self.capture_pipelines.get_or_insert_with(|| CapturePipelines::new(device));
        // One tile list, each group's part starting on a storage-offset boundary.
        let mut tiles: Vec<[u32; 4]> = Vec::new();
        let mut dispatches = Vec::new();
        for (&(kind, graph), group) in &groups {
            tiles.resize(tiles.len().next_multiple_of(TILE_ALIGN), [NONE; 4]);
            dispatches.push((capture.pipeline(device, kind, graph, &self.graph_code), tiles.len(), group.len()));
            tiles.extend_from_slice(group);
        }
        let attribute_buffer = storage_buffer_init(device, queue, "card capture attributes", bytemuck::cast_slice(&attributes));
        let tile_buffer = storage_buffer_init(device, queue, "card capture tiles", bytemuck::cast_slice(&tiles));
        let graph_group = bind_group(
            device,
            "card capture graphs",
            &capture.graph_layout,
            &[(0, inputs.extended.slab_buffer().as_entire_binding()), (1, inputs.extended.parameter_buffer().as_entire_binding())],
        );
        let groups: Vec<(wgpu::BindGroup, u32)> = dispatches
            .iter()
            .map(|&(_, first, count)| {
                let tile_range = wgpu::BufferBinding { buffer: &tile_buffer, offset: first as u64 * 16, size: wgpu::BufferSize::new(count as u64 * 16) };
                let group = bind_group(
                    device,
                    "card capture",
                    &capture.layout,
                    &[
                        (0, inputs.frame.as_entire_binding()),
                        (1, inputs.materials.as_entire_binding()),
                        (2, TextureView(&inputs.textures.albedo)),
                        (3, TextureView(&inputs.textures.data)),
                        (4, Sampler(inputs.sampler)),
                        (5, self.data.as_entire_binding()),
                        (6, TextureView(&self.capture)),
                        (7, attribute_buffer.as_entire_binding()),
                        (8, wgpu::BindingResource::Buffer(tile_range)),
                        (9, TextureView(&self.albedo)),
                        (10, self.pair_sums.as_entire_binding()),
                    ],
                );
                (group, count as u32)
            })
            .collect();
        let mut pass = rec.compute_pass("lumen card capture");
        pass.set_bind_group(1, &graph_group, &[]);
        for ((pipeline, _, _), (group, count)) in dispatches.iter().zip(&groups) {
            pass.set_pipeline(&capture.pipelines[pipeline]);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*count, 1, 1);
        }
        // This capture's inputs (up to ~11 MiB): destroyed once the frame is submitted.
        crate::gpu_memory::retire_buffer(&attribute_buffer);
        crate::gpu_memory::retire_buffer(&tile_buffer);
    }
}

impl CapturePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let layout = bind_group_layout(
            device,
            "card capture",
            c,
            &[
                (0, bind::uniform()),
                (1, bind::uniform()),
                (2, bind::texture_array(true)),
                (3, bind::texture_array(true)),
                (4, bind::sampler()),
                (5, bind::storage(true)),
                (6, bind::texture_uint()),
                (7, bind::storage(true)),
                (8, bind::storage(true)),
                (9, bind::storage_texture(F::Rgba8Unorm)),
                (10, bind::storage(false)),
            ],
        );
        let graph_layout = bind_group_layout(device, "card capture graphs", c, &[(0, bind::uniform()), (1, bind::uniform())]);
        Self { layout, graph_layout, pipelines: HashMap::new(), modules: HashMap::new() }
    }

    /// The pipeline of shading model `kind` and graph `graph` (0: none), compiled on first use;
    /// returns its key.
    fn pipeline(&mut self, device: &wgpu::Device, kind: u32, graph: u64, graph_code: &HashMap<u64, String>) -> (u32, u64) {
        let graph = if graph_code.contains_key(&graph) { graph } else { 0 };
        if !self.pipelines.contains_key(&(kind, graph)) {
            let module = self.modules.entry(graph).or_insert_with(|| create_module_from(device, "lumen card capture", capture_shader(graph_code.get(&graph).map(String::as_str))));
            let layouts = [&self.layout, &self.graph_layout];
            let constants = [("MATERIAL_KIND", kind as f64)];
            let pipeline = in_group(Group::GlobalIllumination, || compute_pipeline_with(device, "lumen card capture", module, "capture_albedo", &layouts, &constants));
            self.pipelines.insert((kind, graph), pipeline);
        }
        (kind, graph)
    }
}

/// lumen_card_capture.wgsl, with a graph's surface code in place of material_graph_off.wgsl.
pub(crate) fn capture_shader(graph: Option<&str>) -> String {
    match graph {
        Some(code) => compose_variant("lumen_card_capture.wgsl", &[("material_graph_off.wgsl", &format!("const HAS_GRAPH: bool = true;\n{code}"))]),
        None => compose_variant("lumen_card_capture.wgsl", &[]),
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
            relight: Vec::new(),
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
        self.relight(&mut updates);
        self.refresh(&mut updates);
        updates
    }

    /// Schedules the pages of moved instances within half the frame's budget, so the refresh of
    /// everything else keeps going while many instances move (the rest stay queued).
    fn relight(&mut self, updates: &mut Vec<[u32; 4]>) {
        let budget = (updates.len() + UPDATE_BUDGET / 2).min(UPDATE_BUDGET);
        while let Some(&i) = self.relight.first() {
            let pages = &self.instances[i].pages;
            if updates.len() + pages.len() > budget && !updates.is_empty() {
                break;
            }
            for &page in pages {
                if self.scheduled[page as usize] != self.frame {
                    self.scheduled[page as usize] = self.frame;
                    updates.push(self.owners[page as usize]);
                }
            }
            self.relight.remove(0);
        }
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
                    updates.push([page, i as u32, record[2] | FRESH_PAGE, record[3]]);
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

fn albedo_atlas(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("card albedo"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: F::Rgba8Unorm,
        usage: U::TEXTURE_BINDING | U::STORAGE_BINDING | U::COPY_SRC | U::COPY_DST,
        view_formats: &[],
    })
}

/// The capture pipeline of `material` — its shading model (`MATERIAL_KIND` of
/// lumen_card_capture.wgsl) and its graph's hash (0: none; the graph's surface code goes into
/// `graph_code`) — or `None` if its cards keep the average albedo (terrain, water).
fn capture_pipeline(scene: &Scene, material: u32, graph_code: &mut HashMap<u64, String>) -> Option<(u32, u64)> {
    let m = scene.materials.get(material as usize)?;
    let kind = match m.kind {
        MaterialKind::Standard | MaterialKind::Translucent => 0,
        MaterialKind::Rock => 1,
        MaterialKind::Metal => 3,
        MaterialKind::Foliage => 4,
        MaterialKind::Glass => 5,
        MaterialKind::Terrain | MaterialKind::Water => return None,
    };
    let graph = m.graph.filter(|_| !m.kind.is_translucent()).and_then(|g| scene.compiled_material_graph(g));
    let hash = graph.map_or(0, |g| {
        graph_code.entry(g.hash).or_insert_with(|| graph_surface_code(&g.wgsl).to_string());
        g.hash
    });
    Some((kind, hash))
}

/// A graph's surface function: its generated code without the displacement function (cards
/// capture the displaced surface's geometry; the material is evaluated on it).
pub(crate) fn graph_surface_code(wgsl: &str) -> &str {
    wgsl.find("const GRAPH_DISPLACEMENT_VECTOR").map_or(wgsl, |end| &wgsl[..end])
}

/// Whether `material`'s albedo depends on where an instance stands: textures projected in world
/// space, a rock's moss (on faces pointing up in the world), graphs that read the world position,
/// normal, tangent frame, camera or object axes.
fn placement_dependent(scene: &Scene, material: u32) -> bool {
    let Some(m) = scene.materials.get(material as usize) else { return false };
    if m.kind == MaterialKind::Rock || m.mapping == TextureMapping::WorldTriplanar {
        return true;
    }
    let Some(graph) = m.graph.and_then(|g| scene.compiled_material_graph(g)) else { return false };
    let code = graph_surface_code(&graph.wgsl);
    ["input.world_position", "input.normal", "input.tangent", "input.bitangent", "input.dpdx", "input.dpdy", "frame.camera", "graph_transform_vector"]
        .iter()
        .any(|field| code.contains(field))
}

/// Hash of what `material`'s captured albedo depends on: the material, the scene's textures, its
/// graph's code and parameter values, and the placement for placement-dependent pairs.
fn material_content(scene: &Scene, material: u32, placement: Option<glam::Mat4>) -> u64 {
    let Some(m) = scene.materials.get(material as usize) else { return 0 };
    let mut text = format!("{m:?} {}", scene.texture_version);
    if let Some(graph) = m.graph.and_then(|g| scene.compiled_material_graph(g)) {
        let mut parameters: Vec<String> = scene.material_parameters(crate::MaterialHandle(material)).map(|(k, v)| format!("{k}={v:?}")).collect();
        parameters.sort();
        text += &format!(" {} {:?} {}", graph.hash, graph.parameters, parameters.join(","));
    }
    if let Some(transform) = placement {
        text += &format!(" {:?}", transform.to_cols_array());
    }
    text.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
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
pub(crate) fn capture_texel(extent: Vec3, scale: f32, texel: f32) -> f32 {
    (texel / scale).max(extent.max_element() / MAX_CARD_RESOLUTION as f32)
}

/// World units per mesh unit (the largest axis scale of `transform`).
fn max_scale(transform: glam::Mat4) -> f32 {
    let linear = glam::Mat3::from_mat4(transform);
    linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length()).max(1e-6)
}

/// Key of surface `m`'s capture: a scene mesh by its index, a displaced variant by its serial.
fn capture_key(scene: &Scene, surfaces: &[Surface], m: usize) -> u64 {
    match surfaces[m] {
        Surface::Displaced(variant) if m >= scene.meshes.len() => variant.serial | (1 << 63),
        _ => m as u64,
    }
}

/// The capture settings of a mesh (`two_sided`: its distance field is a shell) instanced at up to
/// `scale`, for a target world `texel`.
pub(crate) fn capture_config(two_sided: bool, scale: f32, texel: f32) -> CardConfig {
    CardConfig { texel_size: texel / scale, max_resolution: MAX_CARD_RESOLUTION, two_sided, ..Default::default() }
}

/// Captures `surface`, instanced at up to `scale`, at a target world `texel`. A displaced variant
/// usually brings its capture along (made on the bake's worker thread for the default texel).
fn capture_surface(surface: &Surface, scale: f32, texel: f32) -> SurfaceCapture {
    let aabb = surface.aabb();
    let config = capture_config(surface.sdf().is_some_and(|s| s.two_sided), scale, texel);
    match surface {
        Surface::Mesh(mesh) => {
            let size = capture_texel(aabb.extent(), scale, texel);
            let (triangles, sources) = mesh_triangles(mesh, size * 0.5);
            capture_cards(&triangles, &sources, aabb, &config)
        }
        Surface::Displaced(variant) => match &variant.cards {
            Some((s, t, cards)) if *s == scale && *t == texel => cards.clone(),
            // (The base points are not kept: such a capture has no material inputs.)
            _ => SurfaceCapture { cards: MeshCards::capture(&variant.triangles, aabb, &config), attributes: Vec::new() },
        },
    }
}

/// Captures the surfaces with a scale (`mesh_scale`, 0: unused) with world texels grown
/// uniformly from [`CARD_TEXEL`] until all cards fit [`CAPTURE_BUDGET`] (estimated from the
/// bounding boxes first, then checked); returns the captures and the texel.
fn capture_meshes(surfaces: &[Surface], mesh_scale: &[f32]) -> (Vec<Option<SurfaceCapture>>, f32) {
    let used_meshes = || surfaces.iter().zip(mesh_scale).filter(|(_, &s)| s > 0.0);
    // Texels of the bounding-box faces: about one layer per direction.
    let estimate = |texel: f32| -> f32 {
        used_meshes()
            .map(|(surface, &scale)| {
                let e = surface.aabb().extent().max(Vec3::splat(1e-3));
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
        let captures: Vec<Option<SurfaceCapture>> =
            surfaces.iter().zip(mesh_scale).map(|(surface, &scale)| (scale > 0.0).then(|| capture_surface(surface, scale, texel))).collect();
        let texels: usize = captures.iter().flatten().map(|c| c.cards.texel_count()).sum();
        if texels as f32 <= CAPTURE_BUDGET || texel >= 1.0 {
            log::debug!("lumen cards: {texels} texels captured (target texel {texel:.2} m)");
            return (captures, texel);
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

    /// Decodes IEEE half bits (normal numbers and zero).
    fn f16_value(bits: u32) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1f) as i32;
        if exponent == 0 {
            return 0.0;
        }
        sign * (1.0 + (bits & 0x3ff) as f32 / 1024.0) * 2f32.powi(exponent - 15)
    }

    #[test]
    fn half_floats_round_and_saturate() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1.0e6), 0x7bff, "overflow saturates");
        assert_eq!(f16_bits(-1.0e6), 0xfbff);
        assert_eq!(f16_bits(1.0e-6), 0, "below the normal range");
        for x in [0.1f32, -0.0371, 3.3, 123.456, 0.0007] {
            assert!((f16_value(f16_bits(x)) - x).abs() <= x.abs() * 0.0005, "{x}");
        }
    }

    /// A 2 x 2 m quad facing +Y with UVs 3 repeats per metre, and its corners' sources with base
    /// points `lift` below the corners (a displaced surface's base).
    fn tiled_quad(lift: f32) -> (Vec<[CardVertex; 3]>, Vec<[CornerSource; 3]>) {
        let corner = |x: f32, z: f32| ((Vec3::new(x, 0.0, z), Vec3::Y), (Vec3::new(x, -lift, z), Vec2::new(x, z) * 3.0));
        let (a, b, c, d) = (corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0));
        let triangles = [[a, c, b], [a, d, c]];
        (triangles.iter().map(|t| t.map(|v| v.0)).collect(), triangles.iter().map(|t| t.map(|v| v.1)).collect())
    }

    #[test]
    fn captured_texels_carry_their_material_inputs() {
        let (triangles, sources) = tiled_quad(0.05);
        let bounds = Aabb { min: Vec3::new(-1.0, 0.0, -1.0), max: Vec3::new(1.0, 0.0, 1.0) };
        let config = CardConfig { texel_size: 0.1, ..Default::default() };
        let capture = capture_cards(&triangles, &sources, bounds, &config);
        assert_eq!(capture.cards.cards.len(), 1);
        let card = &capture.cards.cards[0];
        assert!(card.sources.is_empty(), "sources are dropped once converted");
        assert_eq!(capture.attributes[0].len(), card.texels.len());
        for y in 0..card.resolution[1] {
            for x in 0..card.resolution[0] {
                let (p, _) = card.texel_surface(x, y, capture.cards.texel_size).unwrap();
                let a = capture.attributes[0][(x + y * card.resolution[0]) as usize];
                let uv = Vec2::new(f32::from_bits(a[0]), f32::from_bits(a[1]));
                assert!(uv.distance(Vec2::new(p.x, p.z) * 3.0) < 1e-4, "{uv} at {p}");
                let offset = Vec3::new(f16_value(a[2] & 0xffff), f16_value(a[2] >> 16), f16_value(a[3] & 0xffff));
                assert!(offset.distance(Vec3::new(0.0, -0.05, 0.0)) < 1e-4, "{offset}");
                // 3 repeats per metre across 0.1 m texels.
                assert!((f16_value(a[3] >> 16) - 0.3).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn uv_footprint_follows_the_projection() {
        let (triangles, sources) = tiled_quad(0.0);
        let uv = sources[0].map(|c| c.1);
        // Seen from +Y (plane axes z, x) the quad's coordinates change by 3 per metre.
        assert!((uv_footprint(&triangles[0], uv, 2, 0) - 3.0).abs() < 1e-5);
        // Edge-on (seen along x: plane axes y, z) the projection is degenerate.
        assert_eq!(uv_footprint(&triangles[0], uv, 1, 2), 0.0);
    }

    #[test]
    fn graph_surface_code_leaves_the_displacement_out() {
        let compiled = helios_material::compile(&helios_material::presets::mushrooms()).unwrap();
        assert!(compiled.wgsl.contains("fn graph_displacement"));
        let code = graph_surface_code(&compiled.wgsl);
        assert!(code.contains("fn graph_eval") && !code.contains("graph_displacement") && !code.contains("GRAPH_DISPLACEMENT"));
        assert_eq!(graph_surface_code("fn graph_eval() {}"), "fn graph_eval() {}");
    }
}
