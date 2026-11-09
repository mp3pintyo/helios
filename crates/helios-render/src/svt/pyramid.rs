//! CPU mip pyramids of image texture layers: the source the streamed tiles are cut from.
//!
//! A layer's virtual texture is square, `size = 512 << levels` texels (the native resolution
//! of its images, rounded to a power of two). Level `levels` is the layer's 512² array layer,
//! which `texgen.wgsl` builds on the GPU and which stays resident as the fallback; only the
//! finer levels `0..levels` are streamed. Level 0 reads the decoded source images directly
//! (resampled only when their sizes differ); the levels between it and the array layer are
//! built once on load with the conventions of the GPU mip chain: base colour averaged in
//! linear light, normals averaged as unit vectors with the shortening folded into roughness
//! (Toksvig). A layer's ambient occlusion image takes the albedo alpha (see
//! [`ImageTexture`]).

use std::ops::Range;
use std::sync::Arc;

use glam::Vec3;
use web_time::Instant;

use super::bc;
use crate::scene::{Image, ImageTexture};
use crate::textures::TEXTURE_SIZE;

/// Texels per tile side, without the border.
pub const TILE: u32 = 128;
/// Filter border on each side of a tile: bilinear plus the anisotropic footprint the
/// shader allows (gradients are limited to `2 * (BORDER - 1)` texels).
pub const BORDER: u32 = 8;
/// Tile side in the atlas.
pub const SPAN: u32 = TILE + 2 * BORDER;
/// Largest virtual texture: 4096² (three levels above the 512² array layer).
pub const MAX_LEVELS: u32 = 3;

/// Base colour, normal, metallic-roughness and occlusion images of a layer, all `size`² (or
/// absent).
struct Sources {
    base_color: Option<Arc<Image>>,
    normal: Option<Arc<Image>>,
    metallic_roughness: Option<Arc<Image>>,
    occlusion: Option<Arc<Image>>,
}

/// The streamed levels of one image layer.
pub struct Pyramid {
    /// Side of level 0 in texels.
    pub size: u32,
    /// Streamed levels (level `levels` is the array layer).
    pub levels: u32,
    sources: Sources,
    /// Levels `1..levels`: (albedo sRGB RGBA8, data RGBA8).
    reduced: Vec<(Vec<u8>, Vec<u8>)>,
    /// Block-compressed copies of levels `0..levels` (empty for uncompressed atlases).
    encoded: Vec<EncodedLevel>,
}

impl Pyramid {
    /// The pyramid of an image layer, or `None` if its images are no larger than the array
    /// layer (nothing to stream).
    pub fn new(images: &ImageTexture) -> Option<Self> {
        let size = virtual_size(images)?;
        let levels = (size / TEXTURE_SIZE).trailing_zeros();
        let fit = |image: &Option<Arc<Image>>, srgb: bool| {
            image.as_ref().filter(|i| valid(i)).map(|i| if i.width == size && i.height == size { i.clone() } else { Arc::new(resample(i, size, srgb)) })
        };
        let sources = Sources {
            base_color: fit(&images.base_color, true),
            normal: fit(&images.normal, false),
            metallic_roughness: fit(&images.metallic_roughness, false),
            occlusion: fit(&images.occlusion, false),
        };
        let mut pyramid = Self { size, levels, sources, reduced: Vec::new(), encoded: Vec::new() };
        for level in 1..levels {
            let side = (size >> level) as usize;
            let mut albedo = Vec::with_capacity(side * side * 4);
            let mut data = Vec::with_capacity(side * side * 4);
            let (mut albedo_rows, mut data_rows) = (Vec::new(), Vec::new());
            for y in 0..side as u32 {
                albedo_rows.clear();
                data_rows.clear();
                for v in [2 * y, 2 * y + 1] {
                    pyramid.albedo_row(level - 1, v, 0, 2 * side as u32, &mut albedo_rows);
                    pyramid.data_row(level - 1, v, 0, 2 * side as u32, &mut data_rows);
                }
                for x in 0..side {
                    albedo.extend_from_slice(&average_albedo(&quad(&albedo_rows, x)));
                    data.extend_from_slice(&average_data(&quad(&data_rows, x)));
                }
            }
            pyramid.reduced.push((albedo, data));
        }
        Some(pyramid)
    }

    /// CPU memory of the reduced and compressed levels (level 0 shares the scene's images).
    pub fn cpu_bytes(&self) -> usize {
        self.reduced.iter().map(|(a, d)| a.len() + d.len()).sum::<usize>() + self.encoded.iter().map(EncodedLevel::bytes).sum::<usize>()
    }

    /// Byte range of texels `start..start + len` of row `v` of `level`.
    fn row_range(&self, level: u32, v: u32, start: u32, len: u32) -> std::ops::Range<usize> {
        let first = (v * (self.size >> level) + start) as usize;
        first * 4..(first + len as usize) * 4
    }

    /// Appends albedo texels `start..start + len` of row `v` of `level` to `out`.
    fn albedo_row(&self, level: u32, v: u32, start: u32, len: u32, out: &mut Vec<u8>) {
        let range = self.row_range(level, v, start, len);
        let first = out.len();
        match (level, &self.sources.base_color) {
            (0, Some(image)) => out.extend_from_slice(&image.pixels[range.clone()]),
            (0, None) => out.resize(out.len() + range.len(), 255),
            _ => return out.extend_from_slice(&self.reduced[level as usize - 1].0[range]),
        }
        // Level 0: the occlusion's R replaces the base colour's alpha (as `pack_images`).
        if let Some(occlusion) = &self.sources.occlusion {
            for (texel, ao) in out[first..].chunks_exact_mut(4).zip(occlusion.pixels[range].chunks_exact(4)) {
                texel[3] = ao[0];
            }
        }
    }

    /// Appends data texels `start..start + len` of row `v` of `level` to `out`.
    fn data_row(&self, level: u32, v: u32, start: u32, len: u32, out: &mut Vec<u8>) {
        let range = self.row_range(level, v, start, len);
        if level > 0 {
            out.extend_from_slice(&self.reduced[level as usize - 1].1[range]);
            return;
        }
        let normal = self.sources.normal.as_ref().map(|i| &i.pixels[range.clone()]);
        let mr = self.sources.metallic_roughness.as_ref().map(|i| &i.pixels[range]);
        let first = out.len();
        out.resize(first + len as usize * 4, 0);
        let texels = &mut out[first..];
        let neutral = |n: &'static [u8; 4]| std::iter::repeat(&n[..]);
        match (normal, mr) {
            (Some(n), Some(m)) => pack_texels(n.chunks_exact(4), m.chunks_exact(4), texels),
            (Some(n), None) => pack_texels(n.chunks_exact(4), neutral(&NEUTRAL_MR), texels),
            (None, Some(m)) => pack_texels(neutral(&FLAT_NORMAL), m.chunks_exact(4), texels),
            (None, None) => pack_texels(neutral(&FLAT_NORMAL), neutral(&NEUTRAL_MR), texels),
        }
    }

    /// Writes tile (`x`, `y`) of `level` with its border into `albedo` and `data`
    /// (`SPAN`² RGBA8 each). The texture repeats, so borders wrap around its edges.
    pub fn tile(&self, level: u32, x: u32, y: u32, albedo: &mut Vec<u8>, data: &mut Vec<u8>) {
        let side = self.size >> level;
        let wrap = |t: u32| (t + side - BORDER) % side;
        // A tile row is at most two contiguous runs of a source row.
        let first = wrap(x * TILE);
        let runs = [(first, SPAN.min(side - first)), (0, SPAN - SPAN.min(side - first))];
        albedo.clear();
        data.clear();
        for row in 0..SPAN {
            let v = wrap(y * TILE + row);
            for &(start, len) in runs.iter().filter(|r| r.1 > 0) {
                self.albedo_row(level, v, start, len, albedo);
                self.data_row(level, v, start, len, data);
            }
        }
    }

    /// Writes tile (`x`, `y`) of `level` uncompressed into the three atlas layouts: albedo
    /// (RGBA8), normal xy and roughness-metallic (RG8 each). `data`: scratch.
    pub fn uncompressed_tile(&self, level: u32, x: u32, y: u32, out: &mut [Vec<u8>; 3], data: &mut Vec<u8>) {
        let [albedo, normal, material] = out;
        self.tile(level, x, y, albedo, data);
        normal.clear();
        material.clear();
        for texel in data.chunks_exact(4) {
            normal.extend_from_slice(&texel[..2]);
            material.extend_from_slice(&texel[2..]);
        }
    }

    /// Allocates block-compressed copies of the streamed levels, to be filled by
    /// [`encode_all`] or, on demand, [`Self::encode_tile`].
    pub fn enable_compression(&mut self) {
        self.encoded = (0..self.levels).map(|level| EncodedLevel::new(self.size >> level)).collect();
    }

    /// Encodes the chunks of tile (`x`, `y`) of `level` (with its border) that are not
    /// encoded yet, stopping at `deadline` (the rest follows on the next call). Returns the
    /// number of blocks encoded if the tile is complete, `None` if the deadline came first.
    pub fn encode_tile(&mut self, level: u32, x: u32, y: u32, deadline: Option<Instant>) -> Option<usize> {
        let side = self.encoded[level as usize].chunks;
        let span = SPAN / CHUNK;
        let wrap = |c: u32| (c + side - BORDER / CHUNK) % side;
        let first = wrap(x * TILE / CHUNK);
        // A tile row is at most two runs of a chunk row.
        let runs = [(first, (first + span).min(side)), (0, span - span.min(side - first))];
        let mut encoded = 0;
        for row in 0..span {
            for &(from, to) in runs.iter().filter(|r| r.1 > r.0) {
                encoded += self.encode_chunks(level, wrap(y * TILE / CHUNK + row), from..to, deadline)?;
            }
        }
        Some(encoded)
    }

    /// Encodes chunks not encoded yet, coarse levels first, until `deadline`. Returns true when
    /// every level is encoded.
    pub fn encode_ahead(&mut self, deadline: Instant) -> bool {
        while let Some(level) = (0..self.encoded.len()).rev().find(|&l| self.encoded[l].ahead < self.encoded[l].chunks) {
            let (row, side) = (self.encoded[level].ahead, self.encoded[level].chunks);
            if self.encode_chunks(level as u32, row, 0..side, Some(deadline)).is_none() {
                return false;
            }
            self.encoded[level].ahead += 1;
        }
        true
    }

    /// Encodes the chunks `columns` of chunk row `row` of `level` that are not encoded yet.
    /// Returns the number of blocks encoded, or `None` if `deadline` passed first.
    fn encode_chunks(&mut self, level: u32, row: u32, columns: Range<u32>, deadline: Option<Instant>) -> Option<usize> {
        let mut encoded = 0;
        let mut start = columns.start;
        while start < columns.end {
            let side = self.encoded[level as usize].chunks;
            let done = |c: u32| self.encoded[level as usize].done[(row * side + c) as usize];
            if done(start) {
                start += 1;
                continue;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            let end = (start..columns.end.min(start + RUN)).find(|&c| done(c)).unwrap_or(columns.end.min(start + RUN));
            let blocks = CHUNK / BLOCK;
            let region = Region { x: start * blocks, y: row * blocks, width: (end - start) * blocks, height: blocks };
            let mut out = std::mem::take(&mut self.encoded[level as usize]);
            self.encode_region(level, region, &mut |bx, by, b| out.put(bx, by, b));
            out.done[(row * side + start) as usize..(row * side + end) as usize].fill(true);
            self.encoded[level as usize] = out;
            encoded += (region.width * region.height) as usize;
            start = end;
        }
        Some(encoded)
    }

    /// Writes the blocks of tile (`x`, `y`) of `level` (with its border, already encoded)
    /// into `out`: BC7 albedo, BC5 normal, BC5 roughness-metallic, `SPAN / 4` blocks per row.
    /// Tile origins and borders are block-aligned, so a tile is a plain copy of the level's
    /// blocks, and its border blocks are bit-identical to its neighbours' (no seams).
    pub fn compressed_tile(&self, level: u32, x: u32, y: u32, out: &mut [Vec<u8>; 3]) {
        let encoded = &self.encoded[level as usize];
        let side = encoded.side;
        let wrap = |b: u32| (b + side - BORDER / BLOCK) % side;
        let span = SPAN / BLOCK;
        let first = wrap(x * TILE / BLOCK);
        let runs = [(first, span.min(side - first)), (0, span - span.min(side - first))];
        for (k, out) in out.iter_mut().enumerate() {
            out.clear();
            for row in 0..span {
                let by = wrap(y * TILE / BLOCK + row);
                for &(start, len) in runs.iter().filter(|r| r.1 > 0) {
                    let i = (by * side + start) as usize;
                    out.extend_from_slice(encoded.blocks[k][i..i + len as usize].as_flattened());
                }
            }
        }
    }

    /// Encodes a rectangle of `level`'s blocks, passing each block's (x, y) and its BC7
    /// albedo, BC5 normal and BC5 roughness-metallic blocks to `put`.
    fn encode_region(&self, level: u32, region: Region, put: &mut dyn FnMut(u32, u32, [[u8; 16]; 3])) {
        let width = (region.width * BLOCK) as usize;
        let (mut albedo, mut data) = (Vec::with_capacity(width * 16), Vec::with_capacity(width * 16));
        for by in region.y..region.y + region.height {
            albedo.clear();
            data.clear();
            for v in by * BLOCK..(by + 1) * BLOCK {
                self.albedo_row(level, v, region.x * BLOCK, region.width * BLOCK, &mut albedo);
                self.data_row(level, v, region.x * BLOCK, region.width * BLOCK, &mut data);
            }
            for b in 0..region.width as usize {
                let texel = |rows: &[u8], i: usize| -> [u8; 4] {
                    let at = ((i / 4) * width + b * 4 + i % 4) * 4;
                    rows[at..at + 4].try_into().expect("4 bytes")
                };
                let color: bc::Block = std::array::from_fn(|i| texel(&albedo, i));
                let packed: bc::Block = std::array::from_fn(|i| texel(&data, i));
                let channel = |c: usize| packed.map(|t| t[c]);
                put(region.x + b as u32, by, [bc::encode_bc7(&color), bc::encode_bc5(&channel(0), &channel(1)), bc::encode_bc5(&channel(2), &channel(3))]);
            }
        }
    }
}

/// Texels per side of the unit of on-demand encoding: 2×2 blocks, the tile border width, so a
/// tile with its border is a whole number of chunks.
const CHUNK: u32 = 8;
/// Texels per side of a compressed block.
const BLOCK: u32 = 4;
/// Chunks encoded between deadline checks at most (~0.1 ms natively).
const RUN: u32 = 16;
const _: () = assert!(TILE.is_multiple_of(CHUNK) && BORDER.is_multiple_of(CHUNK) && CHUNK.is_multiple_of(BLOCK) && TEXTURE_SIZE.is_multiple_of(TILE));

/// A rectangle of blocks.
#[derive(Clone, Copy, Debug)]
struct Region {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// The block-compressed copy of one level.
#[derive(Default)]
struct EncodedLevel {
    /// Blocks per side.
    side: u32,
    /// Chunks per side.
    chunks: u32,
    /// Per block, row-major: BC7 albedo, BC5 normal, BC5 roughness-metallic.
    blocks: [Vec<[u8; 16]>; 3],
    /// Chunks encoded so far, row-major.
    done: Vec<bool>,
    /// Chunk rows before this one are fully encoded ([`Pyramid::encode_ahead`]'s cursor).
    ahead: u32,
}

impl EncodedLevel {
    fn new(texels: u32) -> Self {
        let (side, chunks) = (texels / BLOCK, texels / CHUNK);
        Self { side, chunks, blocks: std::array::from_fn(|_| vec![[0; 16]; (side * side) as usize]), done: vec![false; (chunks * chunks) as usize], ahead: 0 }
    }

    fn put(&mut self, x: u32, y: u32, blocks: [[u8; 16]; 3]) {
        for (k, block) in blocks.into_iter().enumerate() {
            self.blocks[k][(y * self.side + x) as usize] = block;
        }
    }

    fn bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.len() * bc::BLOCK_BYTES).sum()
    }
}

/// Block rows per job of [`encode_all`].
const BAND: u32 = 8;

/// Encodes every level of every pyramid with compression enabled, block rows spread over
/// `threads` worker threads. Returns the number of blocks encoded.
pub fn encode_all(pyramids: &mut [Option<Pyramid>], threads: usize) -> usize {
    let mut jobs = Vec::new();
    for (p, pyramid) in pyramids.iter().enumerate() {
        let Some(pyramid) = pyramid else { continue };
        for (level, encoded) in pyramid.encoded.iter().enumerate() {
            for y in (0..encoded.side).step_by(BAND as usize) {
                jobs.push((p, level as u32, Region { x: 0, y, width: encoded.side, height: BAND.min(encoded.side - y) }));
            }
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let shared: &[Option<Pyramid>] = pyramids;
    // Per worker: (job, its blocks).
    type Encoded = Vec<(usize, Vec<[[u8; 16]; 3]>)>;
    let results: Vec<Encoded> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(p, level, region)) = jobs.get(j) else { break };
                        let mut blocks = Vec::with_capacity((region.width * region.height) as usize);
                        shared[p].as_ref().expect("job of a pyramid").encode_region(level, region, &mut |_, _, b| blocks.push(b));
                        done.push((j, blocks));
                    }
                    done
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().expect("encoder thread")).collect()
    });
    let mut count = 0;
    for (j, blocks) in results.into_iter().flatten() {
        let (p, level, region) = jobs[j];
        let encoded = &mut pyramids[p].as_mut().expect("job of a pyramid").encoded[level as usize];
        for (i, b) in blocks.into_iter().enumerate() {
            encoded.put(i as u32 % region.width, region.y + i as u32 / region.width, b);
        }
        count += (region.width * region.height) as usize;
    }
    for pyramid in pyramids.iter_mut().flatten() {
        for encoded in &mut pyramid.encoded {
            encoded.done.fill(true);
            encoded.ahead = encoded.chunks;
        }
    }
    count
}

/// The 2×2 texels of output texel `x` from two consecutive source rows (concatenated).
fn quad(rows: &[u8], x: usize) -> [[u8; 4]; 4] {
    let width = rows.len() / 2;
    let texel = |i: usize| [rows[i], rows[i + 1], rows[i + 2], rows[i + 3]];
    [texel(x * 8), texel(x * 8 + 4), texel(width + x * 8), texel(width + x * 8 + 4)]
}

/// Side of an image layer's virtual texture: the largest image dimension rounded to the
/// nearest power of two, at most `512 << MAX_LEVELS`; `None` if that is not above the array
/// layer size, or if the layer holds only occlusion (material.wgsl reads such layers from the
/// array: occlusion is low-frequency).
pub fn virtual_size(images: &ImageTexture) -> Option<u32> {
    if images.base_color.is_none() && images.normal.is_none() && images.metallic_roughness.is_none() {
        return None;
    }
    let largest = [&images.base_color, &images.normal, &images.metallic_roughness, &images.occlusion]
        .into_iter()
        .flatten()
        .filter(|i| valid(i))
        .map(|i| i.width.max(i.height))
        .max()?;
    let size = 1u32 << (largest as f32).log2().round() as u32;
    (size > TEXTURE_SIZE).then_some(size.min(TEXTURE_SIZE << MAX_LEVELS))
}

fn valid(image: &Image) -> bool {
    image.width > 0 && image.height > 0 && image.pixels.len() == (image.width * image.height * 4) as usize
}

fn unorm(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    unorm(if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 })
}

/// Texels standing in for a missing normal / metallic-roughness image (roughness and
/// metallic 1: the material's factors apply unchanged, as in glTF).
const FLAT_NORMAL: [u8; 4] = [128, 128, 255, 255];
const NEUTRAL_MR: [u8; 4] = [255; 4];

fn pack_texels<'a>(normals: impl Iterator<Item = &'a [u8]>, mrs: impl Iterator<Item = &'a [u8]>, out: &mut [u8]) {
    for ((texel, n), m) in out.chunks_exact_mut(4).zip(normals).zip(mrs) {
        texel.copy_from_slice(&pack_data(n, m));
    }
}

/// Level-0 data texel, as `pack_images` in texgen.wgsl: unit normal xy, roughness (glTF G),
/// metallic (glTF B). Missing images are neutral: flat normal, roughness = metallic = 1.
fn pack_data(normal: &[u8], metallic_roughness: &[u8]) -> [u8; 4] {
    let decode = |c: u8| c as f32 * (2.0 / 255.0) - 1.0;
    let (x, y, z) = (decode(normal[0]), decode(normal[1]), decode(normal[2]) + 1e-4);
    let scale = 0.5 / (x * x + y * y + z * z).sqrt();
    [unorm(x * scale + 0.5), unorm(y * scale + 0.5), metallic_roughness[1], metallic_roughness[2]]
}

/// 2×2 average in linear light (alpha linearly), as `downsample_albedo`.
fn average_albedo(texels: &[[u8; 4]; 4]) -> [u8; 4] {
    let mut sum = [0.0f32; 4];
    for t in texels {
        for c in 0..3 {
            sum[c] += srgb_to_linear(t[c]);
        }
        sum[3] += t[3] as f32 / 255.0;
    }
    [linear_to_srgb(sum[0] * 0.25), linear_to_srgb(sum[1] * 0.25), linear_to_srgb(sum[2] * 0.25), unorm(sum[3] * 0.25)]
}

/// Shorter averaged normals mean more variance below the texel: fold it into roughness
/// (as `toksvig_roughness` in texgen.wgsl).
fn toksvig_roughness(roughness: f32, len: f32) -> f32 {
    (roughness * roughness + (1.0 - len) / len.max(1e-3) * 0.5).clamp(0.0, 1.0).sqrt()
}

/// 2×2 average of data texels, as `downsample_data`.
fn average_data(texels: &[[u8; 4]; 4]) -> [u8; 4] {
    let mut n = Vec3::ZERO;
    let (mut roughness, mut w) = (0.0, 0.0);
    for t in texels {
        let x = t[0] as f32 / 255.0 * 2.0 - 1.0;
        let y = t[1] as f32 / 255.0 * 2.0 - 1.0;
        n += Vec3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt());
        roughness += t[2] as f32 / 255.0;
        w += t[3] as f32 / 255.0;
    }
    let len = n.length() * 0.25;
    let nn = n.normalize_or(Vec3::Z);
    [unorm(nn.x * 0.5 + 0.5), unorm(nn.y * 0.5 + 0.5), unorm(toksvig_roughness(roughness * 0.25, len)), unorm(w * 0.25)]
}

/// Resamples a repeating image to `size`² (area-averaging taps when shrinking, bilinear
/// when enlarging; colour in linear light if `srgb`). Only for layers whose images differ in
/// size or are not square powers of two.
fn resample(image: &Image, size: u32, srgb: bool) -> Image {
    let (w, h) = (image.width, image.height);
    let fetch = |x: i64, y: i64, c: usize| {
        let (x, y) = (x.rem_euclid(w as i64) as u32, y.rem_euclid(h as i64) as u32);
        let value = image.pixels[((y * w + x) * 4) as usize + c];
        if srgb && c < 3 { srgb_to_linear(value) } else { value as f32 / 255.0 }
    };
    // Bilinear sample at continuous texel coordinates (texel centres at +0.5).
    let bilinear = |u: f32, v: f32, c: usize| {
        let (x, y) = (u - 0.5, v - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = fetch(x0, y0, c) * (1.0 - fx) + fetch(x0 + 1, y0, c) * fx;
        let bottom = fetch(x0, y0 + 1, c) * (1.0 - fx) + fetch(x0 + 1, y0 + 1, c) * fx;
        top * (1.0 - fy) + bottom * fy
    };
    let taps = |source: u32| source.div_ceil(size).clamp(1, 16);
    let (tx, ty) = (taps(w), taps(h));
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            for c in 0..4 {
                let mut sum = 0.0;
                for j in 0..ty {
                    for i in 0..tx {
                        let u = (x as f32 + (i as f32 + 0.5) / tx as f32) / size as f32 * w as f32;
                        let v = (y as f32 + (j as f32 + 0.5) / ty as f32) / size as f32 * h as f32;
                        sum += bilinear(u, v, c);
                    }
                }
                let mean = sum / (tx * ty) as f32;
                pixels.push(if srgb && c < 3 { linear_to_srgb(mean) } else { unorm(mean) });
            }
        }
    }
    Image { width: size, height: size, pixels }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Pyramid {
        fn albedo(&self, level: u32, x: u32, y: u32) -> [u8; 4] {
            let mut texel = Vec::new();
            self.albedo_row(level, y, x, 1, &mut texel);
            texel.try_into().unwrap()
        }

        fn data(&self, level: u32, x: u32, y: u32) -> [u8; 4] {
            let mut texel = Vec::new();
            self.data_row(level, y, x, 1, &mut texel);
            texel.try_into().unwrap()
        }
    }

    fn image(side: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Arc<Image> {
        let pixels = (0..side * side).flat_map(|i| f(i % side, i / side)).collect();
        Arc::new(Image { width: side, height: side, pixels })
    }

    #[test]
    fn virtual_size_rounds_to_powers_of_two_above_the_array_layer() {
        let with = |side| ImageTexture { normal: Some(image(side, |_, _| [128, 128, 255, 255])), ..Default::default() };
        assert_eq!(virtual_size(&with(512)), None);
        assert_eq!(virtual_size(&with(600)), None);
        assert_eq!(virtual_size(&with(1000)), Some(1024));
        assert_eq!(virtual_size(&with(2048)), Some(2048));
        assert_eq!(virtual_size(&ImageTexture::default()), None);
        let big = Arc::new(Image { width: 8192, height: 4096, pixels: vec![0; 8192 * 4096 * 4] });
        assert_eq!(virtual_size(&ImageTexture { base_color: Some(big), ..Default::default() }), Some(4096));
    }

    #[test]
    fn occlusion_takes_the_albedo_alpha() {
        // An "ORM" image: occlusion in R, roughness in G, metallic in B; base colour with alpha.
        let orm = image(2048, |x, _| [(x / 8) as u8, 200, 100, 255]);
        let base = image(2048, |_, _| [50, 60, 70, 10]);
        let images = ImageTexture { base_color: Some(base), metallic_roughness: Some(orm.clone()), occlusion: Some(orm), ..Default::default() };
        let pyramid = Pyramid::new(&images).unwrap();
        assert_eq!(pyramid.albedo(0, 800, 7), [50, 60, 70, 100]);
        assert_eq!(pyramid.data(0, 800, 7)[2..], [200, 100]);
        // Coarser levels average it like coverage (linearly).
        assert_eq!(pyramid.albedo(1, 400, 3)[3], 100);
        // Without a base colour image the colour is white.
        let normal = image(1024, |_, _| [128, 128, 255, 255]);
        let ao = image(1024, |_, _| [30, 0, 0, 255]);
        let pyramid = Pyramid::new(&ImageTexture { normal: Some(normal), occlusion: Some(ao.clone()), ..Default::default() }).unwrap();
        assert_eq!(pyramid.albedo(0, 5, 5), [255, 255, 255, 30]);
        // Occlusion-only layers are not streamed.
        assert!(Pyramid::new(&ImageTexture { occlusion: Some(ao), ..Default::default() }).is_none());
    }

    #[test]
    fn albedo_averages_in_linear_light() {
        let mixed = average_albedo(&[[0, 0, 0, 0], [255, 255, 255, 255], [0, 0, 0, 0], [255, 255, 255, 255]]);
        // Half intensity in linear light is ~188 in sRGB (not 128); alpha averages linearly.
        assert_eq!(mixed[0], 188);
        assert_eq!(mixed[3], 128);
    }

    #[test]
    fn diverging_normals_widen_roughness() {
        let flat = average_data(&[[128, 128, 100, 0]; 4]);
        assert!(flat[2].abs_diff(100) <= 1, "{flat:?}");
        let bumpy = average_data(&[[230, 128, 100, 0], [26, 128, 100, 0], [128, 230, 100, 0], [128, 26, 100, 0]]);
        assert!(bumpy[2] > 150, "{bumpy:?}");
        assert!(bumpy[0].abs_diff(128) <= 1 && bumpy[1].abs_diff(128) <= 1);
    }

    #[test]
    fn levels_and_tiles_wrap() {
        // 1024² base colour whose texels encode their coordinates; no normal or MR image.
        let base = image(1024, |x, y| [(x % 256) as u8, (y % 256) as u8, (x / 256) as u8, 255]);
        let pyramid = Pyramid::new(&ImageTexture { base_color: Some(base), ..Default::default() }).unwrap();
        assert_eq!((pyramid.size, pyramid.levels, pyramid.cpu_bytes()), (1024, 1, 0));
        let (mut albedo, mut data) = (Vec::new(), Vec::new());
        pyramid.tile(0, 0, 0, &mut albedo, &mut data);
        assert_eq!((albedo.len(), data.len()), ((SPAN * SPAN * 4) as usize, (SPAN * SPAN * 4) as usize));
        // Border texel (0, 0) wraps to texel (1024 - BORDER, 1024 - BORDER).
        assert_eq!(&albedo[..4], &[(1024 - BORDER) as u8, (1024 - BORDER) as u8, 3, 255]);
        // First interior texel is texel (0, 0).
        let interior = ((BORDER * SPAN + BORDER) * 4) as usize;
        assert_eq!(&albedo[interior..interior + 4], &[0, 0, 0, 255]);
        // Missing normal / MR images: flat normal, roughness = metallic = 1.
        assert_eq!(&data[..4], &[128, 128, 255, 255]);

        let big = image(2048, |x, _| if x % 2 == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] });
        let pyramid = Pyramid::new(&ImageTexture { base_color: Some(big), ..Default::default() }).unwrap();
        assert_eq!((pyramid.levels, pyramid.cpu_bytes()), (2, 2 * 1024 * 1024 * 4));
        assert_eq!(pyramid.albedo(1, 5, 7), [188, 188, 188, 255]);
    }

    #[test]
    fn mismatched_images_are_resampled_to_the_layer_size() {
        let base = image(1024, |_, _| [200, 100, 50, 255]);
        let normal = image(512, |_, _| [128, 128, 255, 255]);
        let mr = image(2048, |_, _| [0, 64, 32, 255]);
        let pyramid = Pyramid::new(&ImageTexture { base_color: Some(base), normal: Some(normal), metallic_roughness: Some(mr), occlusion: None }).unwrap();
        assert_eq!(pyramid.size, 2048);
        assert_eq!(pyramid.albedo(0, 77, 1500), [200, 100, 50, 255]);
        assert_eq!(pyramid.data(0, 2047, 3), [128, 128, 64, 32]);
    }

    /// A 2048² layer with texture-like base colour (with an alpha mask), a normal map and
    /// metallic-roughness: two streamed levels.
    fn textured() -> ImageTexture {
        let hash = |x: u32, y: u32, seed: u32| {
            let mut h = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ seed.wrapping_mul(83492791);
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1e995);
            (h ^ (h >> 15)) as f32 / u32::MAX as f32
        };
        // Value noise over several octaves (smooth, with detail at every scale).
        let noise = move |x: u32, y: u32, seed: u32| {
            let mut v = 0.0;
            for octave in 0..5 {
                let cell = 64 >> octave;
                let (fx, fy) = ((x % cell) as f32 / cell as f32, (y % cell) as f32 / cell as f32);
                let (cx, cy, n) = (x / cell, y / cell, 2048 / cell);
                let at = |i: u32, j: u32| hash((cx + i) % n, (cy + j) % n, seed + octave);
                let top = at(0, 0) * (1.0 - fx) + at(1, 0) * fx;
                let bottom = at(0, 1) * (1.0 - fx) + at(1, 1) * fx;
                v += (top * (1.0 - fy) + bottom * fy) / (2 << octave) as f32;
            }
            v
        };
        let base = image(2048, |x, y| {
            let v = noise(x, y, 1);
            [(v * 255.0) as u8, (v * 180.0 + 40.0) as u8, (noise(x, y, 2) * 150.0) as u8, if noise(x, y, 3) > 0.45 { 255 } else { 0 }]
        });
        let normal = image(2048, |x, y| {
            let (dx, dy) = (noise(x + 1, y, 4) - noise(x, y, 4), noise(x, y + 1, 4) - noise(x, y, 4));
            let n = Vec3::new(-dx * 40.0, -dy * 40.0, 1.0).normalize();
            [unorm(n.x * 0.5 + 0.5), unorm(n.y * 0.5 + 0.5), unorm(n.z * 0.5 + 0.5), 255]
        });
        let mr = image(2048, |x, y| [0, unorm(noise(x, y, 5)), if noise(x, y, 6) > 0.5 { 255 } else { 0 }, 255]);
        ImageTexture { base_color: Some(base), normal: Some(normal), metallic_roughness: Some(mr), occlusion: None }
    }

    /// The uncompressed tile split into its 4×4 blocks: per atlas, per block, 16 texels of
    /// up to 4 channels.
    fn source_blocks(pyramid: &Pyramid, level: u32, x: u32, y: u32) -> [Vec<[[u8; 4]; 16]>; 3] {
        let mut out: [Vec<u8>; 3] = Default::default();
        pyramid.uncompressed_tile(level, x, y, &mut out, &mut Vec::new());
        let span = (SPAN / BLOCK) as usize;
        std::array::from_fn(|k| {
            let channels = if k == 0 { 4 } else { 2 };
            (0..span * span)
                .map(|b| {
                    std::array::from_fn(|i| {
                        let (tx, ty) = ((b % span) * 4 + i % 4, (b / span) * 4 + i / 4);
                        let at = (ty * SPAN as usize + tx) * channels;
                        std::array::from_fn(|c| if c < channels { out[k][at + c] } else { 0 })
                    })
                })
                .collect()
        })
    }

    fn compressed(pyramid: &Pyramid, level: u32, x: u32, y: u32) -> [Vec<u8>; 3] {
        let mut out: [Vec<u8>; 3] = Default::default();
        pyramid.compressed_tile(level, x, y, &mut out);
        out
    }

    #[test]
    fn compressed_tiles_are_block_copies_of_the_encoded_level() {
        let mut pyramids = [Pyramid::new(&textured())];
        pyramids[0].as_mut().unwrap().enable_compression();
        let blocks = encode_all(&mut pyramids, 3);
        assert_eq!(blocks, (512 * 512 + 256 * 256) as usize);
        let pyramid = pyramids[0].as_ref().unwrap();
        // Interior, wrapping (first and last column / row) and coarser-level tiles: every
        // block equals the encoding of the same texels of the uncompressed tile, so borders
        // and wrapping are block-aligned and seamless.
        for (level, x, y) in [(0, 5, 9), (0, 0, 0), (0, 15, 15), (1, 0, 7), (1, 3, 2)] {
            let source = source_blocks(pyramid, level, x, y);
            let tile = compressed(pyramid, level, x, y);
            for k in 0..3 {
                assert_eq!(tile[k].len(), source[k].len() * bc::BLOCK_BYTES);
                for (b, block) in source[k].iter().enumerate() {
                    let expected = if k == 0 { bc::encode_bc7(block) } else { bc::encode_bc5(&block.map(|t| t[0]), &block.map(|t| t[1])) };
                    assert_eq!(&tile[k][b * 16..b * 16 + 16], &expected, "level {level} tile ({x}, {y}) atlas {k} block {b}");
                }
            }
        }
    }

    #[test]
    fn on_demand_encoding_matches_encoding_at_load() {
        let images = textured();
        let mut eager = [Pyramid::new(&images)];
        eager[0].as_mut().unwrap().enable_compression();
        encode_all(&mut eager, 2);
        let mut lazy = Pyramid::new(&images).unwrap();
        lazy.enable_compression();
        // A tile with its border is 18² chunks of 2×2 blocks; its right neighbour shares
        // 2 columns of chunks, so encodes 16 more.
        assert_eq!(lazy.encode_tile(0, 15, 4, None), Some(36 * 36));
        assert_eq!(lazy.encode_tile(0, 15, 4, None), Some(0));
        assert_eq!(lazy.encode_tile(0, 0, 4, None), Some(36 * 32));
        for (level, x, y) in [(0, 15, 4), (0, 0, 4)] {
            assert_eq!(compressed(&lazy, level, x, y), compressed(eager[0].as_ref().unwrap(), level, x, y));
        }
        // A passed deadline stops before any work; encoding ahead completes every level.
        assert_eq!(lazy.encode_tile(1, 2, 3, Some(Instant::now())), None);
        assert!(!lazy.encode_ahead(Instant::now()));
        assert!(lazy.encode_ahead(Instant::now() + std::time::Duration::from_secs(600)));
        assert_eq!((lazy.encode_tile(1, 2, 3, None), lazy.encode_tile(0, 9, 9, None)), (Some(0), Some(0)));
        assert_eq!(compressed(&lazy, 1, 2, 3), compressed(eager[0].as_ref().unwrap(), 1, 2, 3));
    }

    #[test]
    fn compressed_tiles_keep_the_source_quality() {
        // PSNR of decoded tiles against the uncompressed tiles, per atlas.
        let mut pyramid = Pyramid::new(&textured()).unwrap();
        pyramid.enable_compression();
        let mut error = [0.0f64; 4];
        let mut count = [0usize; 4];
        for (level, x, y) in [(0, 3, 11), (1, 6, 1)] {
            pyramid.encode_tile(level, x, y, None);
            let source = source_blocks(&pyramid, level, x, y);
            let tile = compressed(&pyramid, level, x, y);
            for b in 0..source[0].len() {
                let block: [u8; 16] = tile[0][b * 16..b * 16 + 16].try_into().unwrap();
                for (s, d) in source[0][b].iter().zip(bc::decode::bc7(&block)) {
                    error[0] += (0..3).map(|c| (s[c] as f64 - d[c] as f64).powi(2)).sum::<f64>();
                    error[1] += (s[3] as f64 - d[3] as f64).powi(2);
                }
                count[0] += 48;
                count[1] += 16;
                for k in 1..3 {
                    let encoded = &tile[k][b * 16..b * 16 + 16];
                    let (dx, dy) = (bc::decode::bc4(&encoded[..8]), bc::decode::bc4(&encoded[8..]));
                    for (i, s) in source[k][b].iter().enumerate() {
                        error[k + 1] += (s[0] as f64 - dx[i] as f64).powi(2) + (s[1] as f64 - dy[i] as f64).powi(2);
                    }
                    count[k + 1] += 32;
                }
            }
        }
        let psnr: Vec<f64> = (0..4).map(|k| bc::decode::psnr(error[k], count[k])).collect();
        println!("tile PSNR: BC7 colour {:.2} dB, alpha {:.2} dB; BC5 normal {:.2} dB, roughness-metallic {:.2} dB", psnr[0], psnr[1], psnr[2], psnr[3]);
        let expected = [44.0, 45.0, 45.0, 44.0];
        for k in 0..4 {
            assert!(psnr[k] > expected[k], "PSNR {psnr:.1?} dB");
        }
    }
}
