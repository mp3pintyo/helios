//! Streaming virtual texturing of image texture layers, after UE5's Streaming Virtual Textures
//! (sampling in `shaders/svt.wgsl`).
//!
//! The material texture arrays hold every layer at 512² (no bindless textures in WebGPU), so
//! imported 1k–4k images lose their detail up close. Image layers therefore also get a
//! virtual texture at their native resolution: the levels finer than the array layer are split
//! into 128² tiles (plus an 8-texel filter border) that live in three physical atlases —
//! albedo (sRGB, alpha), normal xy, and roughness-metallic — only while some pixel needs them:
//!
//! * **Feedback.** Where the material resolve samples a layer with mesh UVs (`sample_layer`
//!   in svt.wgsl), it computes the tile of the finer level of the pixel's trilinear pair. One
//!   pixel per 4×4 block (a different one each frame) writes it into a small `r32uint` texture
//!   that is read back asynchronously, as UE does.
//! * **Residency** (`cache.rs`). Requests pull in their coarser ancestors and load coarse
//!   first; tiles are evicted least recently needed; a page table texture maps every virtual
//!   tile to its finest resident ancestor-or-self, fading new tiles in.
//! * **Upload** (`pyramid.rs`). Tiles are cut from CPU mip pyramids of the decoded images
//!   (built once per scene) and written into the atlases within a per-frame budget.
//! * **Compression** (`bc.rs`). Where the device supports BC formats, the atlases are block
//!   compressed — BC7 albedo, BC5 normal xy (z is reconstructed), BC5 roughness-metallic:
//!   3 instead of 8 bytes per texel, so the same pool holds 2.7× the tiles and sampling reads
//!   less memory. As UE does for streaming virtual textures, tiles are compressed ahead of
//!   time rather than per upload: natively every level is encoded on worker threads when the
//!   scene loads, and an upload copies blocks; on the web (no threads) the blocks a tile needs
//!   are encoded when it is first requested and kept, and spare upload time encodes ahead
//!   (coarse levels first), all within the frame's upload time budget.
//!   Without BC support (or with [`TileFormat::Uncompressed`] forced) the atlases are RGBA8 /
//!   RG8 / RG8 with the same layout and shader.
//! * **Fallback.** The array layer is always resident and is the level above the coarsest
//!   streamed one, so nothing is ever missing: sampling falls back to the finest resident
//!   level, and with no tile at all to exactly the array path.
//!
//! Procedural layers stay on the array path: they are synthesised at 512² and repeat over the
//! surface (their density comes from the material's tiling, not a source resolution). So do
//! triplanar mappings, which tile such textures over large surfaces, and every pass that
//! samples layers without these bindings (landscape RVT, Lumen, raster shaders): the array
//! keeps every layer.

mod bc;
mod cache;
mod pyramid;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use web_time::{Duration, Instant};
use wgpu::{TextureFormat as F, TextureUsages as U};

use self::cache::{Layout, TileCache, TileId, TileSource, TABLE_WIDTH};
use self::pyramid::{Pyramid, SPAN, TILE};
use crate::scene::{Scene, TextureSource};
use crate::textures::TEXTURE_SIZE;
use crate::util::uniform_buffer;

/// Pixels per feedback texel side (`SVT_FEEDBACK_BLOCK` in svt.wgsl).
const FEEDBACK_BLOCK: u32 = 4;
/// CPU time per frame for preparing and writing tiles. Cutting an uncompressed level-0 tile
/// out of the source images costs ~0.25 ms (strided reads of three images) and copying a
/// pre-encoded compressed one ~0.02 ms; encoding one on demand (web) costs several ms, so
/// there the budget allows about one new tile per frame. Coarse tiles go first, so detail
/// refines over a few frames.
const UPLOAD_TIME: Duration = Duration::from_millis(2);
/// Anisotropy the shader resolves with hardware taps inside one tile (more selects a coarser
/// level; the 512² array path keeps the sampler's full 16×).
const MAX_ANISO: f32 = 8.0;
/// Layers addressable by the shader's layer table.
const MAX_LAYERS: usize = 256;
/// Order in which feedback visits the pixels of a block (a 4×4 Bayer matrix).
const FEEDBACK_ORDER: [u32; 16] = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12];
/// Encode every level when the scene loads (on worker threads), not on first request.
const ENCODE_AT_LOAD: bool = cfg!(not(target_arch = "wasm32"));

/// Texel formats of the tile atlases: albedo, normal xy, roughness-metallic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFormat {
    /// RGBA8 sRGB, RG8, RG8: 8 bytes per texel.
    Uncompressed,
    /// BC7 sRGB, BC5, BC5: 3 bytes per texel.
    Bc,
}

impl TileFormat {
    /// BC when the device has it and compression is wanted. (ETC2 and ASTC, the mobile
    /// formats, would need encoders of their own: those devices use the uncompressed path.)
    pub fn select(features: wgpu::Features, compress: bool) -> Self {
        if compress && features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) { Self::Bc } else { Self::Uncompressed }
    }

    fn formats(self) -> [F; 3] {
        match self {
            Self::Uncompressed => [F::Rgba8UnormSrgb, F::Rg8Unorm, F::Rg8Unorm],
            Self::Bc => [F::Bc7RgbaUnormSrgb, F::Bc5RgUnorm, F::Bc5RgUnorm],
        }
    }

    /// Bytes per row of a tile (texel or block rows) and rows per tile, per atlas.
    fn tile_rows(self) -> [(u32, u32); 3] {
        self.formats().map(|f| {
            let (w, h) = f.block_dimensions();
            (SPAN / w * f.block_copy_size(None).expect("colour format"), SPAN / h)
        })
    }

    /// GPU bytes of one atlas slot (a tile with its border in all three atlases).
    fn slot_bytes(self) -> u64 {
        self.tile_rows().iter().map(|&(row, rows)| (row * rows) as u64).sum()
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Uncompressed => "RGBA8/RG8/RG8",
            Self::Bc => "BC7/BC5/BC5",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SvtParams {
    tiles_per_row: u32,
    tile_texels: u32,
    border: u32,
    base_size: u32,
    atlas_size: f32,
    table_width: u32,
    frame: u32,
    enabled: u32,
    feedback_offset: [u32; 2],
    max_aniso: f32,
    _pad: u32,
    /// Per layer: page-table offset (bits 0–23) and streamed levels (bits 24–31).
    layers: [[u32; 4]; MAX_LAYERS / 4],
}

const _: () = assert!(std::mem::size_of::<SvtParams>() == 48 + MAX_LAYERS * 4);

/// Texture streaming statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextureStreamingStats {
    /// Image layers with streamed levels.
    pub layers: u32,
    pub virtual_tiles: u32,
    pub resident_tiles: u32,
    pub capacity: u32,
    /// Distinct tiles (with ancestors) the latest feedback asked for.
    pub requested: u32,
    pub pending: u32,
    pub uploads: u64,
    pub evictions: u64,
    /// Requested tiles that found no free slot.
    pub overflow: u32,
    /// GPU memory of the atlases and the page table.
    pub gpu_mb: f32,
    /// CPU memory of the reduced and compressed pyramid levels.
    pub cpu_mb: f32,
    /// Atlas formats (albedo / normal / roughness-metallic).
    pub format: &'static str,
    /// CPU time of encoding the pyramids when the scene loaded.
    pub encode_ms: f32,
    /// CPU time of preparing and writing tiles: the latest frame that uploaded any, and the
    /// most in any frame.
    pub upload_ms: f32,
    pub peak_upload_ms: f32,
}

const IDLE: u32 = 0;
const MAPPING: u32 = 1;
const READY: u32 = 2;

/// The feedback texture and its asynchronous read-back.
struct Feedback {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    readback: wgpu::Buffer,
    bytes_per_row: u32,
    status: Arc<AtomicU32>,
    frame: u32,
}

impl Feedback {
    fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        let size = (width.div_ceil(FEEDBACK_BLOCK).max(1), height.div_ceil(FEEDBACK_BLOCK).max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("svt feedback"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: F::R32Uint,
            usage: U::STORAGE_BINDING | U::COPY_SRC,
            view_formats: &[],
        });
        let bytes_per_row = (size.0 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        Self {
            view: texture.create_view(&Default::default()),
            texture,
            size,
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("svt feedback readback"),
                size: (bytes_per_row * size.1) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bytes_per_row,
            status: Arc::new(AtomicU32::new(IDLE)),
            frame: 0,
        }
    }

    /// The request words of a completed read-back (and the frame they were written in).
    fn take(&self) -> Option<(Vec<u32>, u32)> {
        if self.status.load(Ordering::Acquire) != READY {
            return None;
        }
        let words = self.readback.get_mapped_range(..).ok().map(|view| {
            let words: &[u32] = bytemuck::cast_slice(&view);
            let stride = (self.bytes_per_row / 4) as usize;
            words.chunks(stride).flat_map(|row| row[..self.size.0 as usize].iter().copied().filter(|&w| w != 0)).collect()
        });
        self.readback.unmap();
        self.status.store(IDLE, Ordering::Release);
        words.map(|w| (w, self.frame))
    }
}

pub struct StreamingVirtualTexture {
    pub params: wgpu::Buffer,
    pub page_table: wgpu::TextureView,
    /// sRGB albedo and alpha.
    pub albedo: wgpu::TextureView,
    /// Tangent-space normal xy.
    pub normal: wgpu::TextureView,
    /// Roughness, metallic.
    pub material: wgpu::TextureView,
    pub format: TileFormat,
    table_texture: wgpu::Texture,
    atlases: [wgpu::Texture; 3],
    feedback: Feedback,
    cache: TileCache,
    pyramids: Vec<Option<Pyramid>>,
    state: SvtParams,
    pool_bytes: u64,
    version: u64,
    /// Scratch for one tile's texels or blocks per atlas, and its uncompressed data texels.
    scratch: ([Vec<u8>; 3], Vec<u8>),
    encode_ms: f32,
    /// Every compressed block is encoded (or the atlases are uncompressed).
    encoded: bool,
    upload_ms: (f32, f32),
}

/// One frame's tile uploads, within [`UPLOAD_TIME`].
struct Uploader<'a> {
    queue: &'a wgpu::Queue,
    pyramids: &'a mut [Option<Pyramid>],
    atlases: &'a [wgpu::Texture; 3],
    scratch: &'a mut ([Vec<u8>; 3], Vec<u8>),
    format: TileFormat,
    tiles_per_row: u32,
    deadline: Instant,
    /// Some tile was prepared or uploaded.
    busy: bool,
}

impl TileSource for Uploader<'_> {
    /// Encodes the tile's blocks that are not encoded yet (as far as the deadline allows).
    fn prepare(&mut self, tile: TileId) -> bool {
        self.busy = true;
        match (&mut self.pyramids[tile.layer as usize], self.format) {
            (Some(pyramid), TileFormat::Bc) => pyramid.encode_tile(tile.level, tile.x, tile.y, Some(self.deadline)).is_some(),
            _ => true,
        }
    }

    fn upload(&mut self, tile: TileId, slot: u32) -> bool {
        if let Some(pyramid) = &self.pyramids[tile.layer as usize] {
            let (texels, data) = &mut *self.scratch;
            match self.format {
                TileFormat::Uncompressed => pyramid.uncompressed_tile(tile.level, tile.x, tile.y, texels, data),
                TileFormat::Bc => pyramid.compressed_tile(tile.level, tile.x, tile.y, texels),
            }
            for ((atlas, texels), (bytes_per_row, _)) in self.atlases.iter().zip(texels.iter()).zip(self.format.tile_rows()) {
                write_tile(self.queue, atlas, slot, self.tiles_per_row, bytes_per_row, texels);
            }
        }
        Instant::now() < self.deadline
    }
}

/// Tiles uploaded per frame at most, whatever the time budget allows.
const MAX_UPLOADS: usize = 64;

impl StreamingVirtualTexture {
    /// `pool_bytes`: upper bound on the three tile atlases.
    pub fn new(device: &wgpu::Device, pool_bytes: u64, format: TileFormat) -> Self {
        let atlases = atlases(device, format, 1);
        let table_texture = table(device, 1);
        let [albedo, normal, material] = views(&atlases);
        Self {
            params: uniform_buffer(device, "svt params", std::mem::size_of::<SvtParams>() as u64),
            page_table: table_texture.create_view(&Default::default()),
            albedo,
            normal,
            material,
            format,
            table_texture,
            atlases,
            feedback: Feedback::new(device, (1, 1)),
            cache: TileCache::new(Layout::default(), 0),
            pyramids: Vec::new(),
            state: SvtParams {
                tiles_per_row: 1,
                tile_texels: TILE,
                border: pyramid::BORDER,
                base_size: TEXTURE_SIZE,
                atlas_size: SPAN as f32,
                table_width: TABLE_WIDTH,
                frame: 0,
                enabled: 0,
                feedback_offset: [0; 2],
                max_aniso: MAX_ANISO,
                _pad: 0,
                layers: [[0; 4]; MAX_LAYERS / 4],
            },
            pool_bytes,
            version: u64::MAX,
            scratch: Default::default(),
            encode_ms: 0.0,
            encoded: true,
            upload_ms: (0.0, 0.0),
        }
    }

    /// The feedback texture the material pass writes requests into.
    pub fn feedback_view(&self) -> &wgpu::TextureView {
        &self.feedback.view
    }

    pub fn stats(&self) -> TextureStreamingStats {
        let c = self.cache.stats();
        let gpu: u64 = self.atlases.iter().chain([&self.table_texture]).map(texture_bytes).sum();
        TextureStreamingStats {
            layers: self.pyramids.iter().flatten().count() as u32,
            virtual_tiles: self.cache.layout.entry_count,
            resident_tiles: c.resident,
            capacity: c.capacity,
            requested: c.requested,
            pending: c.pending,
            uploads: c.uploads,
            evictions: c.evictions,
            overflow: c.overflow,
            gpu_mb: gpu as f32 / 1048576.0,
            cpu_mb: self.pyramids.iter().flatten().map(Pyramid::cpu_bytes).sum::<usize>() as f32 / 1048576.0,
            format: self.format.name(),
            encode_ms: self.encode_ms,
            upload_ms: self.upload_ms.0,
            peak_upload_ms: self.upload_ms.1,
        }
    }

    /// Rebuilds the virtual textures when the scene's textures changed, applies the latest
    /// feedback, uploads tiles and writes the frame's parameters. `size`: viewport.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, enabled: bool, frame: u32, size: (u32, u32)) {
        let rebuilt = self.version != scene.texture_version;
        if rebuilt {
            self.version = scene.texture_version;
            self.rebuild(device, scene);
        }
        // A new layout also discards feedback still in flight (it names the old tiles).
        let viewport = (size.0.div_ceil(FEEDBACK_BLOCK).max(1), size.1.div_ceil(FEEDBACK_BLOCK).max(1));
        if rebuilt || self.feedback.size != viewport {
            self.feedback = Feedback::new(device, size);
        }
        let active = enabled && self.cache.layout.entry_count > 0;
        if active {
            let feedback = self.feedback.take();
            let start = Instant::now();
            let deadline = start + UPLOAD_TIME;
            let mut uploader = Uploader {
                queue,
                pyramids: &mut self.pyramids,
                atlases: &self.atlases,
                scratch: &mut self.scratch,
                format: self.format,
                tiles_per_row: self.state.tiles_per_row,
                deadline,
                busy: false,
            };
            self.cache.update(frame, feedback.as_ref().map(|(w, f)| (w.as_slice(), *f)), MAX_UPLOADS, &mut uploader);
            let mut busy = uploader.busy;
            // Blocks still to encode (on the web): the rest of the budget encodes ahead, coarse
            // levels first, so that later requests find their tiles ready.
            if !self.encoded {
                busy = true;
                self.encoded = self.pyramids.iter_mut().flatten().all(|p| p.encode_ahead(deadline));
            }
            if busy {
                let ms = start.elapsed().as_secs_f32() * 1000.0;
                self.upload_ms = (ms, self.upload_ms.1.max(ms));
            }
            if let Some((first, last)) = self.cache.take_dirty_entries() {
                let rows = (first / TABLE_WIDTH, last / TABLE_WIDTH);
                let words = &self.cache.table()[(rows.0 * TABLE_WIDTH) as usize..((rows.1 + 1) * TABLE_WIDTH) as usize];
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.table_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: rows.0, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(words),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TABLE_WIDTH * 4), rows_per_image: None },
                    wgpu::Extent3d { width: TABLE_WIDTH, height: rows.1 - rows.0 + 1, depth_or_array_layers: 1 },
                );
            }
        }
        let k = FEEDBACK_ORDER[(frame % 16) as usize];
        self.state.feedback_offset = [k % FEEDBACK_BLOCK, k / FEEDBACK_BLOCK];
        self.state.frame = frame;
        self.state.enabled = active as u32;
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.state));
    }
    /// Records the feedback read-back if it is due. Returns true if one was recorded (then call
    /// [`Self::map_feedback`] after submitting).
    pub fn record_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u32) -> bool {
        if self.state.enabled == 0 || self.feedback.status.load(Ordering::Acquire) != IDLE {
            return false;
        }
        let f = &mut self.feedback;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &f.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &f.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(f.bytes_per_row), rows_per_image: None },
            },
            wgpu::Extent3d { width: f.size.0, height: f.size.1, depth_or_array_layers: 1 },
        );
        f.frame = frame;
        true
    }

    pub fn map_feedback(&self) {
        let f = &self.feedback;
        f.status.store(MAPPING, Ordering::Release);
        let status = f.status.clone();
        f.readback.map_async(wgpu::MapMode::Read, .., move |result| {
            status.store(if result.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
    }

    fn rebuild(&mut self, device: &wgpu::Device, scene: &Scene) {
        let max_layers = (device.limits().max_texture_array_layers as usize).min(MAX_LAYERS);
        self.pyramids = scene
            .textures
            .iter()
            .take(max_layers)
            .map(|t| match t {
                TextureSource::Images(images) => Pyramid::new(images),
                TextureSource::Procedural(_) => None,
            })
            .collect();
        let levels: Vec<u32> = self.pyramids.iter().map(|p| p.as_ref().map_or(0, |p| p.levels)).collect();
        let layout = Layout::new(&levels);
        self.state.layers = [[0; 4]; MAX_LAYERS / 4];
        for (i, l) in layout.layers().iter().enumerate().filter(|(_, l)| l.levels > 0) {
            self.state.layers[i / 4][i % 4] = l.offset | (l.levels << 24);
        }
        self.encode_ms = 0.0;
        self.encoded = self.format == TileFormat::Uncompressed;
        if self.format == TileFormat::Bc {
            self.pyramids.iter_mut().flatten().for_each(Pyramid::enable_compression);
            if ENCODE_AT_LOAD && !self.pyramids.is_empty() {
                let start = Instant::now();
                let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
                let blocks = pyramid::encode_all(&mut self.pyramids, threads);
                self.encode_ms = start.elapsed().as_secs_f32() * 1000.0;
                self.encoded = true;
                if blocks > 0 {
                    log::info!("texture tiles: encoded {blocks} BC7 + 2×{blocks} BC5 blocks in {:.0} ms on {threads} threads", self.encode_ms);
                }
            }
        }

        let max_side = device.limits().max_texture_dimension_2d / SPAN;
        let tiles_per_row = atlas_tiles_per_row(layout.entry_count, self.pool_bytes / self.format.slot_bytes(), max_side);
        let slots = (tiles_per_row * tiles_per_row).min(1 << 16);
        self.atlases = atlases(device, self.format, tiles_per_row);
        [self.albedo, self.normal, self.material] = views(&self.atlases);
        self.state.tiles_per_row = tiles_per_row;
        self.state.atlas_size = (tiles_per_row * SPAN) as f32;

        let entries = layout.entry_count;
        let streamed = self.pyramids.iter().flatten().count();
        self.cache = TileCache::new(layout, if entries > 0 { slots } else { 0 });
        self.table_texture = table(device, (self.cache.table().len() as u32 / TABLE_WIDTH).max(1));
        self.page_table = self.table_texture.create_view(&Default::default());
        self.upload_ms = (0.0, 0.0);
        if streamed > 0 {
            let s = self.stats();
            log::info!(
                "streaming virtual texture: {streamed} image layers above {TEXTURE_SIZE}² ({} virtual tiles of {TILE}²), {slots} atlas slots ({}, {:.1} MiB GPU), {:.1} MiB CPU pyramids",
                entries,
                self.format.name(),
                s.gpu_mb,
                s.cpu_mb
            );
        }
    }
}

/// Atlas slots per row: a square atlas with as many slots as there are virtual tiles, within
/// the pool's slots and the texture size limit.
fn atlas_tiles_per_row(virtual_tiles: u32, pool_slots: u64, max_tiles_per_row: u32) -> u32 {
    let side = |n: f64| n.sqrt();
    (side(virtual_tiles as f64).ceil() as u32).min(side(pool_slots as f64) as u32).clamp(1, max_tiles_per_row)
}

fn atlases(device: &wgpu::Device, format: TileFormat, tiles_per_row: u32) -> [wgpu::Texture; 3] {
    let side = tiles_per_row * SPAN;
    let labels = ["svt albedo atlas", "svt normal atlas", "svt material atlas"];
    std::array::from_fn(|k| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(labels[k]),
            size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.formats()[k],
            usage: U::TEXTURE_BINDING | U::COPY_DST,
            view_formats: &[],
        })
    })
}

fn views(atlases: &[wgpu::Texture; 3]) -> [wgpu::TextureView; 3] {
    atlases.each_ref().map(|t| t.create_view(&Default::default()))
}

/// GPU bytes of a single-level 2D texture.
fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    let format = texture.format();
    let (w, h) = format.block_dimensions();
    let block = format.block_copy_size(None).unwrap_or(4) as u64;
    (texture.width() / w) as u64 * (texture.height() / h) as u64 * block
}

fn table(device: &wgpu::Device, rows: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("svt page table"),
        size: wgpu::Extent3d { width: TABLE_WIDTH, height: rows, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: F::R32Uint,
        usage: U::TEXTURE_BINDING | U::COPY_DST,
        view_formats: &[],
    })
}

/// Writes one tile's texels (or blocks) into its atlas slot; `bytes_per_row`: of a row of
/// texels (or of blocks).
fn write_tile(queue: &wgpu::Queue, atlas: &wgpu::Texture, slot: u32, tiles_per_row: u32, bytes_per_row: u32, texels: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: atlas,
            mip_level: 0,
            origin: wgpu::Origin3d { x: (slot % tiles_per_row) * SPAN, y: (slot / tiles_per_row) * SPAN, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: None },
        wgpu::Extent3d { width: SPAN, height: SPAN, depth_or_array_layers: 1 },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_follow_the_device_and_the_setting() {
        let bc = wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TIMESTAMP_QUERY;
        assert_eq!(TileFormat::select(bc, true), TileFormat::Bc);
        assert_eq!(TileFormat::select(bc, false), TileFormat::Uncompressed);
        assert_eq!(TileFormat::select(wgpu::Features::TEXTURE_COMPRESSION_ETC2, true), TileFormat::Uncompressed);
        // Base colour keeps sRGB decoding; data channels are linear.
        assert!(TileFormat::Bc.formats()[0].is_srgb() && TileFormat::Uncompressed.formats()[0].is_srgb());
        assert!(!TileFormat::Bc.formats()[1].is_srgb() && !TileFormat::Bc.formats()[2].is_srgb());
    }

    #[test]
    fn tile_rows_and_slot_bytes() {
        // 144² tiles: 36 rows of 36 blocks (16 bytes each) compressed.
        assert_eq!(TileFormat::Bc.tile_rows(), [(576, 36); 3]);
        assert_eq!(TileFormat::Uncompressed.tile_rows(), [(576, 144), (288, 144), (288, 144)]);
        assert_eq!(TileFormat::Uncompressed.slot_bytes(), 144 * 144 * 8);
        assert_eq!(TileFormat::Bc.slot_bytes(), 144 * 144 * 3);
    }

    #[test]
    fn compression_fits_more_tiles_in_the_pool() {
        // Sponza: 25 layers of 64 virtual tiles.
        let slots = |format: TileFormat, mib: u64| atlas_tiles_per_row(1600, (mib << 20) / format.slot_bytes(), 113).pow(2);
        assert_eq!((slots(TileFormat::Uncompressed, 64), slots(TileFormat::Bc, 64)), (400, 1024));
        assert_eq!((slots(TileFormat::Uncompressed, 16), slots(TileFormat::Bc, 16)), (100, 256));
        // Never more slots than virtual tiles (rounded up to a square).
        assert_eq!(atlas_tiles_per_row(10, 1 << 20, 113), 4);
        assert_eq!(atlas_tiles_per_row(0, 0, 113), 1);
    }
}
