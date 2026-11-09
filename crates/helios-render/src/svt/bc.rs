//! CPU block compression of streamed texture tiles: BC7 for colour with alpha, BC5 for two
//! independent channels (normal xy, roughness-metallic).
//!
//! WebGPU cannot write compressed textures from shaders, so blocks are encoded on the CPU
//! (once per texel, when a layer's pyramid is built; see `pyramid.rs`) and copied into the
//! atlases like any other texel data. The encoders favour speed over an exhaustive search
//! but keep the quality-critical parts: a principal-axis line fit refined by least squares,
//! every p-bit choice evaluated against the block, and for BC7 the modes that matter for
//! material textures:
//!
//! * **mode 6** (one RGBA line, 16 levels, 8-bit endpoints): smooth gradients without the
//!   banding of BC1's 5:6:5 endpoints and 4 levels;
//! * **mode 1** (two RGB lines over one of 64 partitions, 8 levels) for opaque blocks that
//!   mode 6 fits badly (edges between unrelated colours), the partition chosen from a
//!   two-means clustering of the block;
//! * **mode 5** (separate RGB and alpha lines) for blocks with varying alpha, whose coverage
//!   edges rarely follow the colour.
//!
//! The error metric is the sum of squared differences of the 8-bit (sRGB-encoded) values.

/// Bytes per 4×4 block of BC5 and BC7.
pub const BLOCK_BYTES: usize = 16;

/// Texels of a 4×4 block in row order.
pub type Block = [[u8; 4]; 16];

const WEIGHTS2: [i32; 4] = [0, 21, 43, 64];
const WEIGHTS3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// Two-subset partitions (bit `i`: subset of texel `i`), from the BC7 specification.
const PARTITIONS2: [u16; 64] = [
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, 0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
];

/// Anchor texel of the second subset of each two-subset partition (the first is texel 0).
const ANCHORS2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8,
    15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];

/// Opaque blocks whose mode-6 error exceeds this (sum of squares) also try mode 1: smooth
/// blocks never need partitions, and skipping them keeps the encoder fast.
const MODE1_THRESHOLD: u32 = 16 * 3 * 4;
/// Partitions (closest to the block's two-means split) that mode 1 encodes fully.
const MODE1_CANDIDATES: usize = 2;

/// Encodes a block of RGBA texels as BC7.
pub fn encode_bc7(block: &Block) -> [u8; 16] {
    let opaque = block.iter().all(|t| t[3] == 255);
    let mut best = mode6(block);
    if !opaque {
        let alt = mode5(block);
        if alt.0 < best.0 {
            best = alt;
        }
    } else if best.0 > MODE1_THRESHOLD {
        let alt = mode1(block);
        if alt.0 < best.0 {
            best = alt;
        }
    }
    best.1
}

/// Encodes two channels (16 values each) as BC5: two BC4 blocks.
pub fn encode_bc5(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&encode_bc4(x));
    out[8..].copy_from_slice(&encode_bc4(y));
    out
}

/// Encodes one channel as BC4 in its 8-level mode (`e0 > e1`), endpoints refined by least
/// squares. (The 6-level mode with exact 0 and 1 rarely wins on smooth data such as normals.)
pub fn encode_bc4(values: &[u8; 16]) -> [u8; 8] {
    let (lo, hi) = values.iter().fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if lo == hi {
        // e0 == e1 selects the 6-level mode, whose index 0 is e0.
        return [hi, lo, 0, 0, 0, 0, 0, 0];
    }
    let mut best = bc4_fit(values, hi, lo);
    for _ in 0..2 {
        // Least squares for the palette ends given the current positions s / 7.
        let (mut a, mut b, mut c, mut xa, mut xb) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (i, &v) in values.iter().enumerate() {
            let w = bc4_position(best.indices[i]) as f32 / 7.0;
            a += (1.0 - w) * (1.0 - w);
            b += (1.0 - w) * w;
            c += w * w;
            xa += (1.0 - w) * v as f32;
            xb += w * v as f32;
        }
        let det = a * c - b * b;
        if det.abs() < 1e-6 {
            break;
        }
        let low = ((c * xa - b * xb) / det).round().clamp(0.0, 255.0) as u8;
        let high = ((a * xb - b * xa) / det).round().clamp(0.0, 255.0) as u8;
        if high <= low {
            break;
        }
        let fit = bc4_fit(values, high, low);
        if fit.error >= best.error {
            break;
        }
        best = fit;
    }
    let mut out = [best.e0, best.e1, 0, 0, 0, 0, 0, 0];
    let bits = best.indices.iter().enumerate().fold(0u64, |acc, (i, &code)| acc | ((code as u64) << (3 * i)));
    out[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
    out
}

struct Bc4Fit {
    e0: u8,
    e1: u8,
    indices: [u8; 16],
    error: f32,
}

/// Position of BC4 index `code` (8-level mode) between e1 (0) and e0 (7).
fn bc4_position(code: u8) -> u8 {
    match code {
        0 => 7,
        1 => 0,
        c => 8 - c,
    }
}

/// Nearest palette entries for `e0 > e1`. The palette is uniform, so the nearest entry is
/// the rounded position (error measured against the decoder's exact interpolation).
fn bc4_fit(values: &[u8; 16], e0: u8, e1: u8) -> Bc4Fit {
    let range = (e0 - e1) as f32;
    let scale = 7.0 / range;
    let mut fit = Bc4Fit { e0, e1, indices: [0; 16], error: 0.0 };
    for (i, &v) in values.iter().enumerate() {
        let s = ((v as f32 - e1 as f32) * scale + 0.5).clamp(0.0, 7.0) as u8;
        fit.indices[i] = match s {
            7 => 0,
            0 => 1,
            s => 8 - s,
        };
        let decoded = e1 as f32 + range * s as f32 / 7.0;
        let d = v as f32 - decoded;
        fit.error += d * d;
    }
    fit
}

/// How a BC7 mode stores its endpoints: bits per colour and alpha component (before the
/// p-bit) and whether there is a p-bit per endpoint, per subset, or none.
#[derive(Clone, Copy)]
struct Precision {
    color: u32,
    alpha: u32,
    pbits: PBits,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PBits {
    None,
    Shared,
    Unique,
}

/// One subset's quantised endpoints (codes as stored, p-bits, decoded 8-bit values) with
/// their indices and error.
#[derive(Clone, Copy)]
struct Line {
    codes: [[u8; 4]; 2],
    pbits: [u8; 2],
    ends: [[i32; 4]; 2],
    indices: [u8; 16],
    error: u32,
}

/// An endpoint component decoded to 8 bits from `code` (`bits` wide) and p-bit `p` (none if
/// `p > 1`): the bits are replicated into the low bits, as the decoder does.
const fn decode_component(code: u32, bits: u32, p: u32) -> u32 {
    let (value, n) = if p > 1 { (code, bits) } else { ((code << 1) | p, bits + 1) };
    if n >= 8 { value } else { (value << (8 - n)) | (value >> (2 * n - 8)) }
}

/// For every 8-bit value, the nearest `bits`-wide code with p-bit `p` (none if `p > 1`) and
/// its decoded value.
const fn quantization(bits: u32, p: u32) -> [[u8; 2]; 256] {
    let mut table = [[0u8; 2]; 256];
    let mut v = 0;
    while v < 256 {
        let (mut best, mut q) = ((u32::MAX, 0, 0), 0);
        while q < 1 << bits {
            let d = decode_component(q, bits, p);
            let e = d.abs_diff(v);
            if e < best.0 {
                best = (e, q, d);
            }
            q += 1;
        }
        table[v as usize] = [best.1 as u8, best.2 as u8];
        v += 1;
    }
    table
}

const Q6: [[[u8; 2]; 256]; 2] = [quantization(6, 0), quantization(6, 1)];
const Q7: [[[u8; 2]; 256]; 3] = [quantization(7, 0), quantization(7, 1), quantization(7, 2)];
const Q8: [[u8; 2]; 256] = quantization(8, 2);

/// The nearest code to `v` (0..=255) with the endpoint precision of BC7 modes 1, 5 and 6, and
/// its decoded value.
fn quantize(v: f32, p: Option<u8>, bits: u32) -> (u8, i32) {
    let v = (v.clamp(0.0, 255.0) + 0.5) as usize;
    let [code, value] = match (bits, p) {
        (6, Some(p)) => Q6[p as usize][v],
        (7, p) => Q7[p.map_or(2, usize::from)][v],
        (8, None) => Q8[v],
        _ => unreachable!("no BC7 mode emitted here has {bits}-bit endpoints with p-bit {p:?}"),
    };
    (code, value as i32)
}

fn interpolate(e0: i32, e1: i32, w: i32) -> i32 {
    ((64 - w) * e0 + w * e1 + 32) >> 6
}

/// The member texels of one subset of a block, fitted with a line in channels `0..C` whose endpoints have the given precision and whose palette has the given
/// weights.
struct Subset<'a, const C: usize> {
    /// Block index and value of each member.
    texel: [usize; 16],
    values: [[i32; 4]; 16],
    n: usize,
    precision: Precision,
    weights: &'a [i32],
}

impl<'a, const C: usize> Subset<'a, C> {
    fn new(block: &Block, members: u16, precision: Precision, weights: &'a [i32]) -> Self {
        let mut s = Self { texel: [0; 16], values: [[0; 4]; 16], n: 0, precision, weights };
        for i in (0..16).filter(|&i| members & (1 << i) != 0) {
            s.texel[s.n] = i;
            s.values[s.n] = block[i].map(i32::from);
            s.n += 1;
        }
        s
    }

    fn members(&self) -> &[[i32; 4]] {
        &self.values[..self.n]
    }

    fn fit(&self) -> Line {
        let mut mean = [0.0f32; 4];
        for v in self.members() {
            for c in 0..C {
                mean[c] += v[c] as f32;
            }
        }
        let mean = mean.map(|m| m / self.n as f32);
        let axis = self.principal_axis(&mean);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in self.members() {
            let t: f32 = (0..C).map(|c| (v[c] as f32 - mean[c]) * axis[c]).sum();
            lo = lo.min(t);
            hi = hi.max(t);
        }
        let ends = [std::array::from_fn(|c| mean[c] + axis[c] * lo), std::array::from_fn(|c| mean[c] + axis[c] * hi)];
        let mut best = self.quantized(&ends);
        for _ in 0..2 {
            let Some(ends) = self.refit(&best) else { break };
            let line = self.quantized(&ends);
            if line.error >= best.error {
                break;
            }
            best = line;
        }
        best
    }

    /// Power iteration on the covariance matrix, started from its largest column.
    fn principal_axis(&self, mean: &[f32; 4]) -> [f32; 4] {
        let mut cov = [[0.0f32; 4]; 4];
        for v in self.members() {
            let d: [f32; 4] = std::array::from_fn(|c| if c < C { v[c] as f32 - mean[c] } else { 0.0 });
            for (a, row) in cov.iter_mut().enumerate() {
                for (b, x) in row.iter_mut().enumerate() {
                    *x += d[a] * d[b];
                }
            }
        }
        let start = (0..C).max_by(|&a, &b| cov[a][a].total_cmp(&cov[b][b])).unwrap_or(0);
        let mut axis = cov[start];
        for _ in 0..6 {
            let next: [f32; 4] = std::array::from_fn(|a| (0..4).map(|b| cov[a][b] * axis[b]).sum());
            let len = next.iter().map(|v| v * v).sum::<f32>().sqrt();
            if len < 1e-9 {
                return [0.0; 4];
            }
            axis = next.map(|v| v / len);
        }
        axis
    }

    /// Endpoints minimising the squared error for fixed indices.
    fn refit(&self, line: &Line) -> Option<[[f32; 4]; 2]> {
        let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
        let (mut xa, mut xb) = ([0.0f32; 4], [0.0f32; 4]);
        for (k, v) in self.members().iter().enumerate() {
            let w = self.weights[line.indices[self.texel[k]] as usize] as f32 / 64.0;
            a += (1.0 - w) * (1.0 - w);
            b += (1.0 - w) * w;
            c += w * w;
            for ch in 0..C {
                xa[ch] += (1.0 - w) * v[ch] as f32;
                xb[ch] += w * v[ch] as f32;
            }
        }
        let det = a * c - b * b;
        (det.abs() > 1e-6).then(|| [std::array::from_fn(|ch| (c * xa[ch] - b * xb[ch]) / det), std::array::from_fn(|ch| (a * xb[ch] - b * xa[ch]) / det)])
    }

    /// The best quantisation of ideal endpoints over the mode's p-bit choices.
    fn quantized(&self, ends: &[[f32; 4]; 2]) -> Line {
        let unique: [[u8; 2]; 1];
        let choices: &[[u8; 2]] = match self.precision.pbits {
            PBits::None => &[[0, 0]],
            PBits::Shared => &[[0, 0], [1, 1]],
            PBits::Unique => {
                // Each endpoint's p-bit by its own rounding error: nearly as good as trying all
                // four pairs against the block, at a quarter of the cost.
                unique = [ends.map(|e| {
                    let error = |p: u8| -> f32 { (0..C).map(|c| (quantize(e[c], Some(p), self.bits(c)).1 as f32 - e[c]).powi(2)).sum() };
                    (error(1) < error(0)) as u8
                })];
                &unique
            }
        };
        let mut best: Option<Line> = None;
        for &pbits in choices {
            let mut line = Line { codes: [[0; 4]; 2], pbits, ends: [[0; 4]; 2], indices: [0; 16], error: 0 };
            for (e, end) in ends.iter().enumerate() {
                for (c, &v) in end.iter().enumerate().take(C) {
                    (line.codes[e][c], line.ends[e][c]) = quantize(v, self.pbit(pbits[e]), self.bits(c));
                }
            }
            self.assign(&mut line);
            if best.is_none_or(|b| line.error < b.error) {
                best = Some(line);
            }
        }
        best.expect("at least one p-bit choice")
    }

    fn bits(&self, channel: usize) -> u32 {
        if channel == 3 { self.precision.alpha } else { self.precision.color }
    }

    fn pbit(&self, p: u8) -> Option<u8> {
        (self.precision.pbits != PBits::None).then_some(p)
    }

    /// The nearest palette entry of every member: the palette is (nearly) uniform along the
    /// line, so the projection onto it gives the index up to one step either way.
    fn assign(&self, line: &mut Line) {
        let [e0, e1] = line.ends;
        let levels = self.weights.len();
        let mut palette = [[0i32; 4]; 16];
        for (p, &w) in palette.iter_mut().zip(self.weights) {
            *p = std::array::from_fn(|c| interpolate(e0[c], e1[c], w));
        }
        let d: [i32; 4] = std::array::from_fn(|c| e1[c] - e0[c]);
        let dd: i32 = (0..C).map(|c| d[c] * d[c]).sum();
        let scale = if dd > 0 { (levels - 1) as f32 / dd as f32 } else { 0.0 };
        let distance = |p: &[i32; 4], v: &[i32; 4]| (0..C).map(|c| (p[c] - v[c]) * (p[c] - v[c])).sum::<i32>() as u32;
        line.error = 0;
        for (k, v) in self.members().iter().enumerate() {
            let t: i32 = (0..C).map(|c| (v[c] - e0[c]) * d[c]).sum();
            let guess = ((t as f32 * scale + 0.5).max(0.0) as usize).min(levels - 1);
            let mut best = (distance(&palette[guess], v), guess);
            for i in [guess.wrapping_sub(1), guess + 1] {
                if i < levels {
                    let e = distance(&palette[i], v);
                    if e < best.0 {
                        best = (e, i);
                    }
                }
            }
            line.indices[self.texel[k]] = best.1 as u8;
            line.error += best.0;
        }
    }

    /// Swaps the endpoints if the anchor texel's index has its top bit set (which the format
    /// leaves implicit as 0), inverting the subset's indices.
    fn anchor(&self, line: &mut Line, anchor: usize) {
        let top = self.weights.len() as u8 - 1;
        if line.indices[anchor] > top / 2 {
            line.codes.swap(0, 1);
            line.pbits.swap(0, 1);
            line.ends.swap(0, 1);
            for &i in &self.texel[..self.n] {
                line.indices[i] = top - line.indices[i];
            }
        }
    }
}

/// Little-endian bit packer of a 128-bit block.
struct Bits {
    value: u128,
    len: u32,
}

impl Bits {
    fn new() -> Self {
        Self { value: 0, len: 0 }
    }

    fn put(&mut self, v: u32, n: u32) {
        self.value |= (v as u128) << self.len;
        self.len += n;
    }

    fn bytes(self) -> [u8; 16] {
        debug_assert_eq!(self.len, 128);
        self.value.to_le_bytes()
    }
}

const ALL: u16 = 0xffff;

/// Mode 6: one RGBA line, 7-bit endpoints with unique p-bits, 4-bit indices.
fn mode6(block: &Block) -> (u32, [u8; 16]) {
    let s = Subset::<4>::new(block, ALL, Precision { color: 7, alpha: 7, pbits: PBits::Unique }, &WEIGHTS4);
    let mut line = s.fit();
    s.anchor(&mut line, 0);
    let mut bits = Bits::new();
    bits.put(1 << 6, 7);
    for c in 0..4 {
        bits.put(line.codes[0][c] as u32, 7);
        bits.put(line.codes[1][c] as u32, 7);
    }
    bits.put(line.pbits[0] as u32, 1);
    bits.put(line.pbits[1] as u32, 1);
    for (i, &index) in line.indices.iter().enumerate() {
        bits.put(index as u32, if i == 0 { 3 } else { 4 });
    }
    (line.error, bits.bytes())
}

/// Mode 5: an RGB line (7-bit endpoints, 2-bit indices) and a separate alpha line (8-bit
/// endpoints, 2-bit indices), no channel rotation.
fn mode5(block: &Block) -> (u32, [u8; 16]) {
    let s = Subset::<3>::new(block, ALL, Precision { color: 7, alpha: 8, pbits: PBits::None }, &WEIGHTS2);
    let mut color = s.fit();
    s.anchor(&mut color, 0);
    // Alpha as a one-channel line: moved into channel 0 of a scratch block.
    let alpha_block: Block = std::array::from_fn(|i| [block[i][3], 0, 0, 0]);
    let a = Subset::<1>::new(&alpha_block, ALL, Precision { color: 8, alpha: 8, pbits: PBits::None }, &WEIGHTS2);
    let mut alpha = a.fit();
    a.anchor(&mut alpha, 0);
    let mut bits = Bits::new();
    bits.put(1 << 5, 6);
    bits.put(0, 2);
    for c in 0..3 {
        bits.put(color.codes[0][c] as u32, 7);
        bits.put(color.codes[1][c] as u32, 7);
    }
    bits.put(alpha.codes[0][0] as u32, 8);
    bits.put(alpha.codes[1][0] as u32, 8);
    for indices in [&color.indices, &alpha.indices] {
        for (i, &index) in indices.iter().enumerate() {
            bits.put(index as u32, if i == 0 { 1 } else { 2 });
        }
    }
    (color.error + alpha.error, bits.bytes())
}

/// Mode 1 (opaque blocks): two RGB lines over a partition, 6-bit endpoints with a p-bit
/// per subset, 3-bit indices. Only the partitions closest to the block's two-means split
/// are encoded.
fn mode1(block: &Block) -> (u32, [u8; 16]) {
    let split = two_means(block);
    // The partitions that differ from the split (or its complement) in the fewest texels.
    let mut candidates = [(u32::MAX, 0usize); MODE1_CANDIDATES];
    for (p, &mask) in PARTITIONS2.iter().enumerate() {
        let mismatch = (mask ^ split).count_ones().min((mask ^ !split).count_ones());
        if let Some(i) = candidates.iter().position(|c| mismatch < c.0) {
            candidates.copy_within(i..MODE1_CANDIDATES - 1, i + 1);
            candidates[i] = (mismatch, p);
        }
    }
    let precision = Precision { color: 6, alpha: 0, pbits: PBits::Shared };
    let mut best = (u32::MAX, [0; 16]);
    for (_, partition) in candidates {
        let mask = PARTITIONS2[partition];
        let subsets = [!mask, mask].map(|members| Subset::<3>::new(block, members, precision, &WEIGHTS3));
        let mut lines = [subsets[0].fit(), subsets[1].fit()];
        let error = lines[0].error + lines[1].error;
        if error >= best.0 {
            continue;
        }
        subsets[0].anchor(&mut lines[0], 0);
        subsets[1].anchor(&mut lines[1], ANCHORS2[partition] as usize);
        let mut bits = Bits::new();
        bits.put(1 << 1, 2);
        bits.put(partition as u32, 6);
        for c in 0..3 {
            for line in &lines {
                bits.put(line.codes[0][c] as u32, 6);
                bits.put(line.codes[1][c] as u32, 6);
            }
        }
        for line in &lines {
            bits.put(line.pbits[0] as u32, 1);
        }
        for i in 0..16 {
            let subset = ((mask >> i) & 1) as usize;
            let anchor = i == 0 || i == ANCHORS2[partition] as usize;
            bits.put(lines[subset].indices[i] as u32, if anchor { 2 } else { 3 });
        }
        best = (error, bits.bytes());
    }
    best
}

/// Two-means clustering of a block's colours, seeded at the extremes of its principal axis:
/// bit `i` set if texel `i` is in the second cluster.
fn two_means(block: &Block) -> u16 {
    let s = Subset::<3>::new(block, ALL, Precision { color: 8, alpha: 8, pbits: PBits::None }, &WEIGHTS2);
    let mut mean = [0.0f32; 4];
    for v in s.members() {
        for c in 0..3 {
            mean[c] += v[c] as f32 / 16.0;
        }
    }
    let axis = s.principal_axis(&mean);
    let t: [f32; 16] = std::array::from_fn(|i| (0..3).map(|c| (s.values[i][c] as f32 - mean[c]) * axis[c]).sum());
    let (lo, hi) = (0..16).fold((0, 0), |(lo, hi), i| (if t[i] < t[lo] { i } else { lo }, if t[i] > t[hi] { i } else { hi }));
    let mut centres = [s.values[lo].map(|v| v as f32), s.values[hi].map(|v| v as f32)];
    let mut mask = 0u16;
    for _ in 0..3 {
        let distance = |v: &[i32; 4], k: usize| (0..3).map(|c| (v[c] as f32 - centres[k][c]).powi(2)).sum::<f32>();
        mask = (0..16).filter(|&i| distance(&s.values[i], 1) < distance(&s.values[i], 0)).fold(0, |m, i| m | (1 << i));
        let mut sums = [[0.0f32; 4]; 2];
        let mut counts = [0.0f32; 2];
        for (i, v) in s.members().iter().enumerate() {
            let k = ((mask >> i) & 1) as usize;
            counts[k] += 1.0;
            for c in 0..3 {
                sums[k][c] += v[c] as f32;
            }
        }
        for k in 0..2 {
            if counts[k] > 0.0 {
                centres[k] = sums[k].map(|v| v / counts[k]);
            }
        }
    }
    mask
}

#[cfg(test)]
pub mod decode {
    //! Reference decoders for the modes the encoders emit (tests only).

    use super::*;

pub fn expand(code: u8, p: Option<u8>, bits: u32) -> i32 {
    decode_component(code as u32, bits, p.map_or(2, u32::from)) as i32
}

    /// PSNR in dB of 8-bit values from their summed squared error.
    pub fn psnr(squared_error: f64, count: usize) -> f64 {
        10.0 * (255.0f64 * 255.0 / (squared_error / count as f64).max(1e-12)).log10()
    }

    struct Reader {
        value: u128,
    }

    impl Reader {
        fn get(&mut self, n: u32) -> u32 {
            let v = (self.value & ((1u128 << n) - 1)) as u32;
            self.value >>= n;
            v
        }
    }

    pub fn bc4(block: &[u8]) -> [f32; 16] {
        let (e0, e1) = (block[0] as f32, block[1] as f32);
        let mut bits = [0u8; 8];
        bits[..6].copy_from_slice(&block[2..8]);
        let bits = u64::from_le_bytes(bits);
        std::array::from_fn(|i| {
            let code = ((bits >> (3 * i)) & 7) as usize;
            let palette: [f32; 8] = if e0 > e1 {
                std::array::from_fn(|k| match k {
                    0 => e0,
                    1 => e1,
                    k => ((8 - k) as f32 * e0 + (k - 1) as f32 * e1) / 7.0,
                })
            } else {
                std::array::from_fn(|k| match k {
                    0 => e0,
                    1 => e1,
                    6 => 0.0,
                    7 => 255.0,
                    k => ((6 - k) as f32 * e0 + (k - 1) as f32 * e1) / 5.0,
                })
            };
            palette[code]
        })
    }

    pub fn bc7(block: &[u8; 16]) -> Block {
        let mut r = Reader { value: u128::from_le_bytes(*block) };
        let mode = (0..8).find(|_| r.get(1) == 1).expect("reserved BC7 mode");
        let unpack = |r: &mut Reader, subsets: usize, color: u32, alpha: u32, pbits: PBits| {
            let mut ends = vec![[[0i32; 4]; 2]; subsets];
            let channels = if alpha > 0 { 4 } else { 3 };
            for c in 0..channels {
                for end in ends.iter_mut() {
                    for e in end.iter_mut() {
                        e[c] = r.get(if c == 3 { alpha } else { color }) as i32;
                    }
                }
            }
            let p: Vec<u8> = match pbits {
                PBits::None => vec![],
                PBits::Shared => (0..subsets).map(|_| r.get(1) as u8).collect(),
                PBits::Unique => (0..2 * subsets).map(|_| r.get(1) as u8).collect(),
            };
            for (s, end) in ends.iter_mut().enumerate() {
                for (k, e) in end.iter_mut().enumerate() {
                    let pbit = match pbits {
                        PBits::None => None,
                        PBits::Shared => Some(p[s]),
                        PBits::Unique => Some(p[2 * s + k]),
                    };
                    for (c, v) in e.iter_mut().enumerate() {
                        *v = if c < channels { expand(*v as u8, pbit, if c == 3 { alpha } else { color }) } else { 255 };
                    }
                }
            }
            ends
        };
        let mut out = [[0u8; 4]; 16];
        match mode {
            6 => {
                let ends = unpack(&mut r, 1, 7, 7, PBits::Unique)[0];
                for (i, t) in out.iter_mut().enumerate() {
                    let w = WEIGHTS4[r.get(if i == 0 { 3 } else { 4 }) as usize];
                    *t = std::array::from_fn(|c| interpolate(ends[0][c], ends[1][c], w) as u8);
                }
            }
            5 => {
                assert_eq!(r.get(2), 0, "rotation");
                // Components in order R0 R1 G0 G1 B0 B1 A0 A1.
                let components: Vec<i32> = (0..8).map(|k| if k < 6 { expand(r.get(7) as u8, None, 7) } else { r.get(8) as i32 }).collect();
                let color: Vec<i32> = (0..16).map(|i| WEIGHTS2[r.get(if i == 0 { 1 } else { 2 }) as usize]).collect();
                let alpha: Vec<i32> = (0..16).map(|i| WEIGHTS2[r.get(if i == 0 { 1 } else { 2 }) as usize]).collect();
                for (i, t) in out.iter_mut().enumerate() {
                    *t = std::array::from_fn(|c| interpolate(components[2 * c], components[2 * c + 1], if c == 3 { alpha[i] } else { color[i] }) as u8);
                }
            }
            1 => {
                let partition = r.get(6) as usize;
                let ends = unpack(&mut r, 2, 6, 0, PBits::Shared);
                for (i, t) in out.iter_mut().enumerate() {
                    let anchor = i == 0 || i == ANCHORS2[partition] as usize;
                    let w = WEIGHTS3[r.get(if anchor { 2 } else { 3 }) as usize];
                    let s = ((PARTITIONS2[partition] >> i) & 1) as usize;
                    *t = std::array::from_fn(|c| if c == 3 { 255 } else { interpolate(ends[s][0][c], ends[s][1][c], w) as u8 });
                }
            }
            m => panic!("BC7 mode {m} is not emitted by the encoder"),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(seed: &mut u32) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed
    }

    /// A 64×64 image as 4×4 blocks.
    fn blocks(f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<Block> {
        (0..256).map(|b| std::array::from_fn(|i| f((b % 16) * 4 + i as u32 % 4, (b / 16) * 4 + i as u32 / 4))).collect()
    }

    fn bc7_psnr(blocks: &[Block], channels: usize) -> f64 {
        let mut error = 0.0;
        for block in blocks {
            let decoded = decode::bc7(&encode_bc7(block));
            for (a, b) in block.iter().zip(&decoded) {
                for c in 0..channels {
                    error += (a[c] as f64 - b[c] as f64).powi(2);
                }
            }
        }
        decode::psnr(error, blocks.len() * 16 * channels)
    }

    /// Prints the PSNR (shown with `--nocapture`) and checks it.
    fn expect_psnr(name: &str, psnr: f64, min: f64) {
        println!("{name}: PSNR {psnr:.2} dB");
        assert!(psnr > min, "{name}: PSNR {psnr:.2} dB");
    }

    fn bc4_psnr(blocks: &[[u8; 16]]) -> f64 {
        let error: f64 = blocks
            .iter()
            .map(|b| decode::bc4(&encode_bc4(b)).iter().zip(b).map(|(d, &v)| (d - v as f32).powi(2) as f64).sum::<f64>())
            .sum();
        decode::psnr(error, blocks.len() * 16)
    }

    /// Value noise with a few octaves: texture-like detail at several scales.
    fn texture(x: u32, y: u32, seed: u32) -> f32 {
        let hash = |x: u32, y: u32| {
            let mut s = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ seed.wrapping_mul(83492791) | 1;
            random(&mut s);
            random(&mut s) as f32 / u32::MAX as f32
        };
        let mut v = 0.0;
        for octave in 0..4 {
            let cell = 16 >> octave;
            let (fx, fy) = ((x % cell) as f32 / cell as f32, (y % cell) as f32 / cell as f32);
            let (cx, cy) = (x / cell, y / cell);
            let top = hash(cx, cy) * (1.0 - fx) + hash(cx + 1, cy) * fx;
            let bottom = hash(cx, cy + 1) * (1.0 - fx) + hash(cx + 1, cy + 1) * fx;
            v += (top * (1.0 - fy) + bottom * fy) / (2 << octave) as f32;
        }
        v
    }

    #[test]
    fn bc7_round_trips_constant_and_two_colour_blocks() {
        for colour in [[0, 0, 0, 255], [255, 255, 255, 255], [200, 100, 50, 255], [17, 99, 201, 128]] {
            let decoded = decode::bc7(&encode_bc7(&[colour; 16]));
            for t in decoded {
                for c in 0..4 {
                    assert!(t[c].abs_diff(colour[c]) <= 1, "{colour:?} -> {t:?}");
                }
            }
        }
        // An edge between two unrelated colours is exact with mode 1's partitions.
        let edge = std::array::from_fn(|i| if i % 4 < 2 { [250, 20, 20, 255] } else { [20, 20, 250, 255] });
        let decoded = decode::bc7(&encode_bc7(&edge));
        let error: u32 = edge.iter().zip(&decoded).map(|(a, b)| (0..3).map(|c| a[c].abs_diff(b[c]) as u32).sum::<u32>()).sum();
        assert!(error <= 16 * 3 * 2, "{decoded:?}");
    }

    #[test]
    fn bc7_quality() {
        // Smooth ramps: 4-bit indices between 8-bit endpoints reproduce them to within one
        // step (no banding, unlike BC1's four levels between 5:6:5 endpoints).
        let ramp = blocks(|x, _| [(x * 4) as u8, (x * 3) as u8, 200 - x as u8, 255]);
        for block in &ramp {
            let decoded = decode::bc7(&encode_bc7(block));
            assert!(block.iter().zip(&decoded).all(|(a, b)| (0..4).all(|c| a[c].abs_diff(b[c]) <= 1)), "{block:?} -> {decoded:?}");
        }
        // Gradients along two axes are not on one colour line per block.
        let gradient = blocks(|x, y| [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]);
        let p = bc7_psnr(&gradient, 3);
        expect_psnr("BC7 gradient", p, 42.0);
        // Texture-like noise with correlated channels (as in most albedo maps).
        let noise = blocks(|x, y| {
            let v = texture(x, y, 1);
            let w = texture(x, y, 2);
            [(v * 230.0) as u8, (v * 180.0 + w * 40.0) as u8, (v * 120.0 + w * 60.0) as u8, 255]
        });
        let p = bc7_psnr(&noise, 3);
        expect_psnr("BC7 noise", p, 45.0);
        // Uncorrelated random colours, the worst case: the block mean alone would give 10.8 dB.
        let mut seed = 7;
        let random_blocks: Vec<Block> = (0..256).map(|_| std::array::from_fn(|_| [random(&mut seed) as u8, random(&mut seed) as u8, random(&mut seed) as u8, 255])).collect();
        let p = bc7_psnr(&random_blocks, 3);
        expect_psnr("BC7 random", p, 15.0);
        // Alpha-masked foliage: coverage edges unrelated to the colour.
        let masked = blocks(|x, y| {
            let v = texture(x, y, 3);
            [(v * 90.0) as u8, (v * 200.0) as u8, (v * 60.0) as u8, if texture(x, y, 4) > 0.5 { 255 } else { 0 }]
        });
        let p = bc7_psnr(&masked, 4);
        expect_psnr("BC7 masked", p, 44.0);
    }

    #[test]
    fn bc4_quality() {
        let gradient: Vec<[u8; 16]> = (0..64).map(|b| std::array::from_fn(|i| (b * 4 + i % 4) as u8)).collect();
        expect_psnr("BC4 gradient", bc4_psnr(&gradient), 50.0);
        let noise: Vec<[u8; 16]> = (0..256u32).map(|b| std::array::from_fn(|i| (texture((b % 16) * 4 + i as u32 % 4, (b / 16) * 4 + i as u32 / 4, 5) * 255.0) as u8)).collect();
        let p = bc4_psnr(&noise);
        expect_psnr("BC4 noise", p, 44.0);
        assert_eq!(decode::bc4(&encode_bc4(&[77; 16])), [77.0; 16]);
        // Binary data (metallic masks) is exact.
        let binary = std::array::from_fn(|i| if i % 3 == 0 { 255 } else { 0 });
        assert_eq!(decode::bc4(&encode_bc4(&binary)), binary.map(|v| v as f32));
    }

    /// Measures quality and speed on real textures: `HELIOS_BC_DIR` holds RGBA PNGs named by
    /// role (`*base*` / `*emissive*`: BC7 colour; `*normal*`: BC5 of the unit normal's xy;
    /// otherwise metallic-roughness: BC5 of G and B). Run with
    /// `cargo test --release -p helios-render png_files -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs HELIOS_BC_DIR"]
    fn png_files() {
        let Ok(dir) = std::env::var("HELIOS_BC_DIR") else { return };
        let mut paths: Vec<_> = std::fs::read_dir(dir).expect("HELIOS_BC_DIR").map(|e| e.expect("entry").path()).collect();
        paths.sort();
        for path in paths.iter().filter(|p| p.extension().is_some_and(|e| e == "png")) {
            let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).expect("png"))).read_info().expect("png");
            let mut pixels = vec![0; reader.output_buffer_size().expect("size")];
            let info = reader.next_frame(&mut pixels).expect("png");
            assert_eq!(info.color_type, png::ColorType::Rgba, "{path:?}: RGBA expected");
            let (w, h) = (info.width as usize, info.height as usize);
            let name = path.file_name().expect("name").to_string_lossy().to_string();
            let blocks: Vec<Block> = (0..(w / 4) * (h / 4))
                .map(|b| {
                    std::array::from_fn(|i| {
                        let at = (((b / (w / 4)) * 4 + i / 4) * w + (b % (w / 4)) * 4 + i % 4) * 4;
                        pixels[at..at + 4].try_into().expect("texel")
                    })
                })
                .collect();
            let start = std::time::Instant::now();
            let (error, count, format) = if name.contains("base") || name.contains("emissive") {
                let encoded: Vec<_> = blocks.iter().map(encode_bc7).collect();
                let error: f64 = blocks.iter().zip(&encoded).map(|(s, e)| s.iter().zip(decode::bc7(e)).map(|(a, b)| (0..3).map(|c| (a[c] as f64 - b[c] as f64).powi(2)).sum::<f64>()).sum::<f64>()).sum();
                // Alpha (coverage, or ambient occlusion packed into it) when it varies.
                if blocks.iter().flatten().any(|t| t[3] != blocks[0][0][3]) {
                    let alpha: f64 = blocks.iter().zip(&encoded).map(|(s, e)| s.iter().zip(decode::bc7(e)).map(|(a, b)| (a[3] as f64 - b[3] as f64).powi(2)).sum::<f64>()).sum();
                    println!("{name:40} alpha: PSNR {:.2} dB", decode::psnr(alpha, blocks.len() * 16));
                }
                (error, blocks.len() * 48, "BC7")
            } else {
                let channels: Vec<[[u8; 16]; 2]> = if name.contains("normal") {
                    blocks
                        .iter()
                        .map(|b| {
                            let n = b.map(|t| {
                                let v = glam::Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32) / 127.5 - 1.0;
                                let n = v.normalize_or(glam::Vec3::Z) * 0.5 + 0.5;
                                [(n.x * 255.0 + 0.5) as u8, (n.y * 255.0 + 0.5) as u8]
                            });
                            [n.map(|v| v[0]), n.map(|v| v[1])]
                        })
                        .collect()
                } else {
                    blocks.iter().map(|b| [b.map(|t| t[1]), b.map(|t| t[2])]).collect()
                };
                let error: f64 = channels
                    .iter()
                    .map(|[x, y]| {
                        let e = encode_bc5(x, y);
                        [(decode::bc4(&e[..8]), x), (decode::bc4(&e[8..]), y)].iter().map(|(d, s)| d.iter().zip(s.iter()).map(|(&d, &s)| (d as f64 - s as f64).powi(2)).sum::<f64>()).sum::<f64>()
                    })
                    .sum();
                (error, blocks.len() * 32, "BC5")
            };
            let seconds = start.elapsed().as_secs_f64();
            println!("{name:40} {w}x{h} {format}: PSNR {:.2} dB, {:.2} µs per block (with decoding)", decode::psnr(error, count), seconds * 1e6 / blocks.len() as f64);
        }
    }
}
