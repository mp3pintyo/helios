//! Entropy coding of page payloads for storage and transfer (`.hvm` version 7).
//!
//! Page payloads are kept in the GPU format of [`crate::encoding`] in the geometry pool: that is
//! already quantised and bit-packed (Nanite's), but every field has a fixed width per cluster, so
//! generic compressors find little to remove from it. Unreal compresses its streaming pages on
//! disk with Oodle; here a page is decoded back into its field values — the same values the
//! encoder packed — and those are coded with predictions that know what they are:
//!
//! * **Triangles** (cluster-local vertex indices, in first-use order): a FIFO of the last
//!   triangles' directed edges; a triangle that shares an edge with one of them (the usual case:
//!   neighbours follow each other) is the edge's slot and rotation plus its third vertex. A vertex
//!   is "the next new one" or a distance back from it.
//! * **Vertices that repeat** an earlier vertex of the page (cluster borders) are a reference to
//!   it — all attributes, or the position only.
//! * **Positions** are predicted from the vertex's first triangle — the mean of the parallelograms
//!   around it (Draco's multi-parallelogram), else the parallelogram over an earlier triangle
//!   sharing the edge of its two known corners, else their midpoint, else the one known corner,
//!   else the previous vertex — and the residual is coded per axis and predictor (y and z also by
//!   the size of the residuals before them).
//! * **Normals** are predicted from the decoded geometry (the area-weighted sum of the face normals
//!   around the vertex, on the octahedral grid) or from the neighbours, whichever the encoder finds
//!   cheaper for the cluster. **UVs** use the position predictors in texture space, or are mapped
//!   affinely from the neighbouring triangle's positions (planar and box mapping; again chosen per
//!   cluster). Vertex colours, plant data and skin weights are predicted from a neighbour.
//! * Voxel clusters are coded word by word (delta to the previous brick or voxel, byte by byte).
//!
//! Residuals become a *token* (exact below 8, else the bit length and the next bit) plus raw
//! extra bits. Tokens are coded with **rANS** (32-bit state, byte renormalisation) using one static
//! frequency table per context and page, stored in the page (quantised, delta-coded, shared with a
//! parent context where an own table does not pay); extra bits go to a plain bit stream.
//! The decoder rebuilds the clusters with [`encode_cluster`] — the same code that packed them — so
//! the result is bit-identical; the encoder checks that and stores a page as it is when it is not
//! (or when coding does not shrink it).
//!
//! A coded page is self-contained: it carries its cluster structure, so decoding needs nothing
//! but its bytes ([`decode_page`]), on any thread or Web Worker. Only integer arithmetic and IEEE
//! basic operations (`+ − × ÷`, `abs`, `round`; no fused or transcendental ones) enter the predictions, so every
//! platform decodes the same bits.

use std::collections::HashMap;

use glam::{IVec2, IVec3, Vec2, Vec3};

use crate::encoding::{decode_triangle, encode_cluster, read_bits, ClusterEncoding, Quantization, QuantizedVertex, VertexAttributes, PLANT_CHANNEL_BITS};
use crate::skin::{SkinWeights, MAX_INFLUENCES};
use crate::GpuCluster;

/// Why a page did not decode (static, so the hot paths return small results).
type Corrupt = &'static str;

/// First byte of a stored page: its words as they are (little-endian).
const MODE_RAW: u8 = 0;
/// First byte of a stored page: entropy coded. Any change to the model (contexts, predictors,
/// their order) changes what old pages decode to: give it a new mode (and the `.hvm` a new version).
const MODE_CODED: u8 = 1;

/// Probability scale of the rANS tables (`2^SCALE_BITS`).
const SCALE_BITS: u32 = 12;
const SCALE: u32 = 1 << SCALE_BITS;
/// Lower bound of the rANS state.
const RANS_LOW: u32 = 1 << 23;
/// Tokens: 0..8 exact, then two per bit length up to 32 bits.
const TOKENS: usize = 66;
/// Directed edges remembered for triangle coding.
const EDGE_FIFO: usize = 32;

/// Coding contexts (each has its own frequency table).
mod ctx {
    pub const CLUSTERS: u8 = 0;
    pub const KIND: u8 = 1;
    pub const VERTEX_COUNT: u8 = 2;
    pub const TRIANGLE_COUNT: u8 = 3;
    pub const PARAMS: u8 = 4;
    /// Shared-edge slot (0: none), by whether the previous triangle shared one (2); its rotation.
    pub const EDGE: u8 = 5;
    pub const EDGE_ROTATION: u8 = 7;
    /// Corners of a triangle without a shared edge (3), then the third corner after an edge.
    pub const CORNER: u8 = 8;
    pub const THIRD: u8 = 11;
    /// Repeat kind by the previous vertex's kind (3).
    pub const REPEAT: u8 = 12;
    /// Repeated vertex: distance back, or (after a repeat) the step from the previous one's source.
    pub const REPEAT_REF: u8 = 15;
    pub const REPEAT_STEP: u8 = 16;
    /// Positions: 3 axes x the first four predictors.
    pub const POSITION: u8 = 17;
    /// Positions predicted by one parallelogram, then by several: the x residual, then y by the
    /// size of x's (3), then z by the size of x's and y's (3).
    pub const PARALLELOGRAM: u8 = 29;
    pub const MULTI_PARALLELOGRAM: u8 = 36;
    /// Normals: 2 components x (geometry, previous normal, neighbours); the cluster's choice.
    pub const NORMAL: u8 = 43;
    pub const NORMAL_MODE: u8 = 49;
    /// UVs: 2 axes x 5 predictors, then the UVs mapped from the geometry; the cluster's choice.
    pub const UV: u8 = 50;
    pub const UV_MAPPED: u8 = 60;
    pub const UV_MODE: u8 = 62;
    pub const UV1: u8 = 63;
    /// Colours: red, green/blue, (unused), alpha.
    pub const COLOR: u8 = 65;
    pub const PLANT: u8 = 69;
    pub const SKIN_WEIGHT: u8 = 71;
    pub const SKIN_JOINT: u8 = 72;
    pub const VOXEL_LEN: u8 = 73;
    /// Voxel brick words: 4 lanes x 4 bytes; voxel attribute words: 4 bytes.
    pub const VOXEL_BRICK: u8 = 74;
    pub const VOXEL_ATTRIBUTE: u8 = 90;
    pub const COUNT: usize = 94;
}

// ---------------------------------------------------------------------------------------------
// Tokens and bit streams

/// Token and extra-bit count of `v`.
#[inline]
fn token(v: u32) -> (u32, u32) {
    if v < 8 {
        (v, 0)
    } else {
        let n = 31 - v.leading_zeros();
        (8 + (n - 3) * 2 + ((v >> (n - 1)) & 1), n - 1)
    }
}

/// The value of token `t` with its extra bits read by `extra(count)`.
#[inline]
fn untoken(t: u32, extra: impl FnOnce(u32) -> u32) -> u32 {
    if t < 8 {
        t
    } else {
        let n = (t - 8) / 2 + 3;
        let top = (2 | ((t - 8) & 1)) << (n - 1);
        top | extra(n - 1)
    }
}

#[inline]
fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

#[inline]
fn unzigzag(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

#[derive(Default)]
struct BitSink {
    bytes: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitSink {
    fn put(&mut self, value: u32, count: u32) {
        if count == 0 {
            return;
        }
        let value = if count == 32 { value as u64 } else { (value & ((1u32 << count) - 1)) as u64 };
        self.acc |= value << self.bits;
        self.bits += count;
        while self.bits >= 8 {
            self.bytes.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.bytes.push(self.acc as u8);
        }
        self.bytes
    }
}

struct BitSource<'a> {
    bytes: &'a [u8],
    pos: usize,
    acc: u64,
    bits: u32,
}

impl<'a> BitSource<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0, acc: 0, bits: 0 }
    }

    #[inline]
    fn get(&mut self, count: u32) -> u32 {
        if count == 0 {
            return 0;
        }
        while self.bits < count {
            // Past the end reads zeros (a corrupt page then fails the structural checks).
            let byte = self.bytes.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.acc |= (byte as u64) << self.bits;
            self.bits += 8;
        }
        let value = if count == 32 { self.acc as u32 } else { (self.acc as u32) & ((1u32 << count) - 1) };
        self.acc >>= count;
        self.bits -= count;
        value
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(bytes: &[u8], pos: &mut usize) -> Result<u32, Corrupt> {
    let mut v = 0u32;
    for shift in (0..35).step_by(7) {
        let b = *bytes.get(*pos).ok_or("truncated page")?;
        *pos += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err("bad varint".into())
}

// ---------------------------------------------------------------------------------------------
// The encoder records (context, token) pairs and extra bits; the decoder reads them back.

trait Coder {
    /// Codes `v` in context `c` (the decoder ignores `v` and returns the decoded value).
    fn value(&mut self, c: u8, v: u32) -> Result<u32, Corrupt>;
    /// Codes `v` (< [`TOKENS`]) as a symbol of its own, without extra bits.
    fn symbol(&mut self, c: u8, v: u32) -> Result<u32, Corrupt>;
    /// Raw bits.
    fn raw(&mut self, v: u32, count: u32) -> u32;

    fn signed(&mut self, c: u8, v: i32) -> Result<i32, Corrupt> {
        Ok(unzigzag(self.value(c, zigzag(v))?))
    }
}

#[derive(Default)]
struct Recorder {
    symbols: Vec<(u8, u8)>,
    extra: BitSink,
}

impl Coder for Recorder {
    fn value(&mut self, c: u8, v: u32) -> Result<u32, Corrupt> {
        let (t, n) = token(v);
        self.symbols.push((c, t as u8));
        self.extra.put(v, n);
        Ok(v)
    }

    fn symbol(&mut self, c: u8, v: u32) -> Result<u32, Corrupt> {
        if v as usize >= TOKENS {
            return Err("symbol out of range");
        }
        self.symbols.push((c, v as u8));
        Ok(v)
    }

    fn raw(&mut self, v: u32, count: u32) -> u32 {
        self.extra.put(v, count);
        v
    }
}

/// A context's normalised frequencies (summing to [`SCALE`]) and decoding table.
struct Table {
    freq: [u16; TOKENS],
    start: [u16; TOKENS],
    /// Slot -> token (built for decoding only).
    slots: Vec<u8>,
}

fn normalize(counts: &[u32; TOKENS]) -> [u16; TOKENS] {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut freq = [0u16; TOKENS];
    let mut sum = 0u32;
    for (f, &c) in freq.iter_mut().zip(counts) {
        if c > 0 {
            *f = ((c as u64 * SCALE as u64 / total) as u16).max(1);
            sum += *f as u32;
        }
    }
    let largest = (0..TOKENS).max_by_key(|&i| (freq[i], std::cmp::Reverse(i))).unwrap_or(0);
    if sum < SCALE {
        freq[largest] += (SCALE - sum) as u16;
    }
    // Rounding small counts up to 1 overshoots by at most one per token.
    while sum > SCALE {
        let i = (0..TOKENS).max_by_key(|&i| (freq[i], std::cmp::Reverse(i))).unwrap_or(0);
        freq[i] -= 1;
        sum -= 1;
    }
    freq
}

impl Table {
    fn new(freq: [u16; TOKENS], slots: bool) -> Self {
        let mut start = [0u16; TOKENS];
        let mut acc = 0u32;
        for i in 0..TOKENS {
            start[i] = acc as u16;
            acc += freq[i] as u32;
        }
        let mut table = Vec::new();
        if slots {
            table.reserve(SCALE as usize);
            for (t, &f) in freq.iter().enumerate() {
                table.extend(std::iter::repeat_n(t as u8, f as usize));
            }
        }
        Self { freq, start, slots: table }
    }
}

/// The weight of frequency level `level` (1..=31): steps of 2^(1/3) from 16 to 25 << 10.
fn level_weight(level: u32) -> u32 {
    [16, 20, 25][((level - 1) % 3) as usize] << ((level - 1) / 3)
}

/// Quantised frequency levels (0 for unused tokens), relative to the most frequent token.
fn levels(counts: &[u32; TOKENS]) -> [u8; TOKENS] {
    let max = counts.iter().copied().max().unwrap_or(0).max(1) as f64;
    counts.map(|c| if c == 0 { 0 } else { (31.0 - (3.0 * (max / c as f64).log2()).round()).clamp(1.0, 31.0) as u8 })
}

/// The coding frequencies of a context with these levels (the same in encoder and decoder).
fn level_frequencies(levels: &[u8; TOKENS]) -> [u16; TOKENS] {
    normalize(&levels.map(|l| if l == 0 { 0 } else { level_weight(l as u32) }))
}

/// The context whose table a context may share when a table of its own would not pay for itself
/// (on small pages, or for rare contexts): a related context with similar statistics.
fn parent(c: u8) -> Option<u8> {
    use ctx::*;
    let within = |base: u8, len: u8| c >= base && c < base + len;
    Some(match c {
        _ if c == EDGE + 1 => EDGE,
        _ if within(CORNER + 1, 2) => CORNER,
        _ if within(REPEAT + 1, 2) => REPEAT,
        _ if c == REPEAT_STEP => REPEAT_REF,
        // First vertex, previous vertex, one corner → midpoint, same axis → parallelogram.
        _ if within(POSITION, 9) => POSITION + 9 + (c - POSITION) % 3,
        _ if within(POSITION + 9, 3) => PARALLELOGRAM,
        // z by size → y by size → x; several parallelograms → one.
        _ if within(PARALLELOGRAM + 4, 3) => c - 3,
        _ if within(PARALLELOGRAM + 1, 3) => PARALLELOGRAM,
        _ if within(MULTI_PARALLELOGRAM, 7) => c - MULTI_PARALLELOGRAM + PARALLELOGRAM,
        _ if c == NORMAL + 1 => NORMAL,
        _ if within(NORMAL + 2, 4) => NORMAL + (c - NORMAL) % 2,
        _ if within(UV, 8) => UV + 8 + (c - UV) % 2,
        _ if c == UV + 9 || c == UV_MAPPED => UV + 8,
        _ if c == UV_MAPPED + 1 => UV_MAPPED,
        _ if c == UV1 + 1 => UV1,
        _ if within(COLOR + 1, 3) => COLOR,
        _ if c == PLANT + 1 => PLANT,
        _ if within(VOXEL_BRICK + 4, 12) => VOXEL_BRICK + (c - VOXEL_BRICK) % 4,
        _ if within(VOXEL_BRICK, 4) => VOXEL_ATTRIBUTE + (c - VOXEL_BRICK),
        _ => return None,
    })
}

/// Bits of a table with these levels in [`write_tables`]' code (after the mode bits).
fn table_bits(levels: &[u8; TOKENS]) -> u32 {
    let n = levels.iter().rposition(|&l| l > 0).map_or(0, |i| i + 1);
    let mut previous = 31i32;
    let mut bits = 7;
    for &l in &levels[..n] {
        bits += match (l as i32 - previous).abs() {
            0 => 1,
            1 => 3,
            2 | 3 => 5,
            _ => 8,
        };
        previous = l as i32;
    }
    bits
}

/// Bits to code `counts` with a table of their own, including the table.
fn context_cost(counts: &[u32; TOKENS]) -> f64 {
    if counts.iter().all(|&n| n == 0) {
        return 0.0;
    }
    let levels = levels(counts);
    let freq = level_frequencies(&levels);
    let data: f64 = counts.iter().zip(&freq).filter(|(&n, _)| n > 0).map(|(&n, &f)| n as f64 * (SCALE as f64 / f as f64).log2()).sum();
    data + table_bits(&levels) as f64
}

/// How each context's symbols are coded.
const UNUSED: u32 = 0;
const OWN: u32 = 1;
const SHARED: u32 = 2;

/// Writes the contexts' tables as a bit stream: per context its mode (2 bits: unused, own table,
/// shares its parent's); for an own table the number of tokens up to its last used one (7 bits)
/// and the tokens' levels, each coded as the step from the previous one (0: `0`, ±1: `10s`,
/// ±2..3: `110sm`, else `111` and the level).
fn write_tables(out: &mut Vec<u8>, modes: &[u32], tables: &[Option<[u8; TOKENS]>]) {
    let mut bits = BitSink::default();
    for (mode, t) in modes.iter().zip(tables) {
        bits.put(*mode, 2);
        if let (OWN, Some(levels)) = (*mode, t) {
            let n = levels.iter().rposition(|&l| l > 0).map_or(0, |i| i + 1);
            bits.put(n as u32, 7);
            let mut previous = 31i32;
            for &l in &levels[..n] {
                let d = l as i32 - previous;
                match d.abs() {
                    0 => bits.put(0, 1),
                    1 => bits.put(0b01 | ((d < 0) as u32) << 2, 3),
                    2 | 3 => bits.put(0b011 | ((d < 0) as u32) << 3 | ((d.abs() - 2) as u32) << 4, 5),
                    _ => bits.put(0b111 | (l as u32) << 3, 8),
                }
                previous = l as i32;
            }
        }
    }
    let bytes = bits.finish();
    put_varint(out, bytes.len() as u32);
    out.extend_from_slice(&bytes);
}

/// The tables a page's contexts are coded with: `which[c]` indexes `tables` (`u8::MAX`: unused).
struct Tables {
    tables: Vec<Table>,
    which: [u8; ctx::COUNT],
}

fn read_tables(bytes: &[u8], pos: &mut usize) -> Result<Tables, Corrupt> {
    let len = get_varint(bytes, pos)? as usize;
    let block = bytes.get(*pos..*pos + len).ok_or("truncated page tables")?;
    *pos += len;
    let mut bits = BitSource::new(block);
    let mut out = Tables { tables: Vec::new(), which: [u8::MAX; ctx::COUNT] };
    let mut modes = [UNUSED; ctx::COUNT];
    for c in 0..ctx::COUNT {
        modes[c] = bits.get(2);
        if modes[c] != OWN {
            continue;
        }
        let n = bits.get(7) as usize;
        if n == 0 || n > TOKENS {
            return Err("corrupt page tables".into());
        }
        let mut levels = [0u8; TOKENS];
        let mut previous = 31i32;
        for l in &mut levels[..n] {
            let level = if bits.get(1) == 0 {
                previous
            } else if bits.get(1) == 0 {
                previous + if bits.get(1) == 1 { -1 } else { 1 }
            } else if bits.get(1) == 0 {
                let negative = bits.get(1) == 1;
                let step = 2 + bits.get(1) as i32;
                previous + if negative { -step } else { step }
            } else {
                bits.get(5) as i32
            };
            if !(0..32).contains(&level) {
                return Err("corrupt page tables".into());
            }
            *l = level as u8;
            previous = level;
        }
        if levels.iter().all(|&l| l == 0) {
            return Err("corrupt page tables".into());
        }
        out.which[c] = out.tables.len() as u8;
        out.tables.push(Table::new(level_frequencies(&levels), true));
    }
    for c in 0..ctx::COUNT {
        if modes[c] == SHARED {
            out.which[c] = owner(c as u8, &modes).map(|o| out.which[o as usize]).ok_or("corrupt page tables")?;
        }
    }
    Ok(out)
}

/// The context whose table context `c` is coded with.
fn owner(c: u8, modes: &[u32]) -> Option<u8> {
    let mut c = c;
    for _ in 0..8 {
        match modes[c as usize] {
            OWN => return Some(c),
            SHARED => c = parent(c)?,
            _ => return None,
        }
    }
    None
}

struct Decoder<'a> {
    tables: Tables,
    rans: &'a [u8],
    pos: usize,
    state: u32,
    extra: BitSource<'a>,
}

impl Decoder<'_> {
    #[inline]
    fn token(&mut self, c: u8) -> Result<u32, Corrupt> {
        let index = self.tables.which[c as usize];
        let table = self.tables.tables.get(index as usize).ok_or("corrupt page: symbol in an unused context")?;
        let slot = self.state & (SCALE - 1);
        let t = table.slots[slot as usize] as usize;
        self.state = table.freq[t] as u32 * (self.state >> SCALE_BITS) + slot - table.start[t] as u32;
        while self.state < RANS_LOW {
            let byte = *self.rans.get(self.pos).ok_or("truncated page")?;
            self.pos += 1;
            self.state = (self.state << 8) | byte as u32;
        }
        Ok(t as u32)
    }
}

impl Coder for Decoder<'_> {
    #[inline]
    fn symbol(&mut self, c: u8, _: u32) -> Result<u32, Corrupt> {
        self.token(c)
    }

    #[inline]
    fn value(&mut self, c: u8, _: u32) -> Result<u32, Corrupt> {
        let t = self.token(c)?;
        let extra = &mut self.extra;
        Ok(untoken(t, |n| extra.get(n)))
    }

    fn raw(&mut self, _: u32, count: u32) -> u32 {
        self.extra.get(count)
    }
}

/// Entropy codes the recorded symbols: tables, rANS stream, extra bits.
fn finish(recorder: Recorder, out: &mut Vec<u8>) {
    let mut counts = vec![[0u32; TOKENS]; ctx::COUNT];
    for &(c, t) in &recorder.symbols {
        counts[c as usize][t as usize] += 1;
    }
    let mut modes: Vec<u32> = counts.iter().map(|c| if c.iter().any(|&n| n > 0) { OWN } else { UNUSED }).collect();
    // Contexts share their parent's table where that is cheaper, children before parents.
    let depth = |c: u8| std::iter::successors(Some(c), |&c| parent(c)).count();
    let mut order: Vec<u8> = (0..ctx::COUNT as u8).filter(|&c| modes[c as usize] == OWN && parent(c).is_some()).collect();
    order.sort_by_key(|&c| (std::cmp::Reverse(depth(c)), c));
    for c in order {
        let p = parent(c).expect("filtered") as usize;
        let merged: [u32; TOKENS] = std::array::from_fn(|t| counts[p][t] + counts[c as usize][t]);
        if context_cost(&merged) < context_cost(&counts[c as usize]) + context_cost(&counts[p]) {
            counts[p] = merged;
            modes[c as usize] = SHARED;
            if modes[p] == UNUSED {
                modes[p] = OWN;
            }
        }
    }
    let levels: Vec<Option<[u8; TOKENS]>> = (0..ctx::COUNT).map(|c| (modes[c] == OWN).then(|| levels(&counts[c]))).collect();
    write_tables(out, &modes, &levels);
    let tables: Vec<Option<Table>> = levels.iter().map(|l| l.map(|l| Table::new(level_frequencies(&l), false))).collect();
    let table = |c: u8| tables[owner(c, &modes).expect("a recorded context has a table") as usize].as_ref().expect("an own table");
    // rANS codes in reverse, so the decoder reads forward.
    let mut stream = Vec::with_capacity(recorder.symbols.len());
    let mut state = RANS_LOW;
    for &(c, t) in recorder.symbols.iter().rev() {
        let table = table(c);
        let (freq, start) = (table.freq[t as usize] as u32, table.start[t as usize] as u32);
        let max = ((RANS_LOW >> SCALE_BITS) << 8) * freq;
        while state >= max {
            stream.push(state as u8);
            state >>= 8;
        }
        state = ((state / freq) << SCALE_BITS) + (state % freq) + start;
    }
    stream.extend_from_slice(&state.to_le_bytes());
    stream.reverse();
    put_varint(out, stream.len() as u32);
    out.extend_from_slice(&stream);
    out.extend_from_slice(&recorder.extra.finish());
}

// ---------------------------------------------------------------------------------------------
// Cluster model

/// A triangle cluster's field values.
struct TriangleCluster {
    quantization: Quantization,
    vertices: Vec<QuantizedVertex>,
    triangles: Vec<[u32; 3]>,
}

/// Parses the triangle cluster `c` of `page`.
fn parse_cluster(page: &[u32], c: &GpuCluster) -> TriangleCluster {
    let words = &page[c.vertex_offset as usize..];
    let encoding = ClusterEncoding::from_words(words);
    let stream = &words[encoding.header_words() as usize..];
    let n = c.vertex_count();
    let attributes = encoding.attributes;
    let pbits = encoding.position_bits;
    let plant_bits = if attributes.plant { PLANT_CHANNEL_BITS } else { 0 };
    let pstride = pbits.x + pbits.y + pbits.z + plant_bits;
    let astride = encoding.vertex_stream_bits(n) / n.max(1) - pstride;
    let mut vertices = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut bit = i * pstride;
        let mut p = [0i32; 3];
        for a in 0..3 {
            p[a] = encoding.position_min[a].wrapping_add(read_bits(stream, bit, pbits[a]) as i32);
            bit += pbits[a];
        }
        let wind = read_bits(stream, bit, plant_bits) as u8;
        let mut bit = n * pstride + i * astride;
        let normal = read_bits(stream, bit, 2 * encoding.normal_bits);
        bit += 2 * encoding.normal_bits;
        let mut field2 = |min: IVec2, bits: glam::UVec2| {
            let x = min.x.wrapping_add(read_bits(stream, bit, bits.x) as i32);
            let y = min.y.wrapping_add(read_bits(stream, bit + bits.x, bits.y) as i32);
            bit += bits.x + bits.y;
            IVec2::new(x, y)
        };
        let uv = field2(encoding.uv_min, encoding.uv_bits);
        let uv1 = if attributes.uv1 { field2(encoding.uv1_min, encoding.uv1_bits) } else { IVec2::ZERO };
        let mut color = [255u8; 4];
        if attributes.color {
            for ch in 0..4 {
                color[ch] = encoding.color_min[ch].wrapping_add(read_bits(stream, bit, encoding.color_bits[ch]) as u8);
                bit += encoding.color_bits[ch];
            }
        }
        let occlusion = read_bits(stream, bit, plant_bits) as u8;
        let skin = if attributes.skin { encoding.skin(stream, n, i).unwrap_or_default() } else { SkinWeights::default() };
        vertices.push(QuantizedVertex {
            position: IVec3::from_array(p),
            normal,
            uv,
            uv1,
            color,
            skin,
            plant: if attributes.plant { [wind, occlusion] } else { [31, 31] },
        });
    }
    let triangles = (0..c.triangle_count()).map(|t| decode_triangle(&page[c.triangle_offset as usize..], n, t)).collect();
    let quantization = Quantization {
        position_exponent: encoding.position_exponent,
        uv_exponent: encoding.uv_exponent,
        normal_bits: encoding.normal_bits,
        attributes,
    };
    TriangleCluster { quantization, vertices, triangles }
}

fn params_word(q: &Quantization) -> u32 {
    let a = q.attributes;
    let flags = a.uv1 as u32 | (a.color as u32) << 1 | (a.skin as u32) << 2 | (a.plant as u32) << 3;
    (q.position_exponent as i8 as u8 as u32) | ((q.uv_exponent as i8 as u8 as u32) << 8) | (q.normal_bits << 16) | (flags << 24)
}

fn params_from_word(w: u32) -> Result<Quantization, Corrupt> {
    let normal_bits = (w >> 16) & 0xff;
    if !(2..=16).contains(&normal_bits) {
        return Err("corrupt page: normal bits".into());
    }
    let flags = w >> 24;
    Ok(Quantization {
        position_exponent: (w & 0xff) as u8 as i8 as i32,
        uv_exponent: ((w >> 8) & 0xff) as u8 as i8 as i32,
        normal_bits,
        attributes: VertexAttributes { uv1: flags & 1 != 0, color: flags & 2 != 0, skin: flags & 4 != 0, plant: flags & 8 != 0 },
    })
}

/// How a vertex's position (or UV) is predicted from its first triangle.
#[derive(Clone, Copy)]
enum Predictor {
    /// The cluster's first vertex: from the page's previous cluster's first vertex.
    First,
    Previous(u32),
    Corner(u32),
    Midpoint(u32, u32),
    Parallelogram(u32, u32, u32),
}

impl Predictor {
    fn kind(&self) -> u8 {
        match self {
            Predictor::First => 0,
            Predictor::Previous(_) => 1,
            Predictor::Corner(_) => 2,
            Predictor::Midpoint(..) => 3,
            Predictor::Parallelogram(..) => 4,
        }
    }

    /// The vertex the prediction leans on most (for attributes predicted from one neighbour).
    fn neighbour(&self) -> Option<usize> {
        match *self {
            Predictor::First => None,
            Predictor::Previous(a) | Predictor::Corner(a) | Predictor::Midpoint(a, _) | Predictor::Parallelogram(a, _, _) => Some(a as usize),
        }
    }

    fn apply<const N: usize>(&self, get: impl Fn(u32) -> [i32; N], first: [i32; N]) -> [i32; N] {
        match *self {
            Predictor::First => first,
            Predictor::Previous(a) | Predictor::Corner(a) => get(a),
            Predictor::Midpoint(a, b) => {
                let (a, b) = (get(a), get(b));
                std::array::from_fn(|i| ((a[i] as i64 + b[i] as i64) >> 1) as i32)
            }
            Predictor::Parallelogram(a, b, c) => {
                let (a, b, c) = (get(a), get(b), get(c));
                std::array::from_fn(|i| a[i].wrapping_add(b[i]).wrapping_sub(c[i]))
            }
        }
    }
}

/// The predictor of every vertex of a cluster with these triangles (vertices in first-use order).
/// Every vertex's parallelograms: `list[offsets[v]..offsets[v + 1]]`.
struct Parallelograms {
    offsets: Vec<u32>,
    list: Vec<[u32; 3]>,
}

impl Parallelograms {
    fn of(&self, v: usize) -> &[[u32; 3]] {
        &self.list[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

/// `edges`: a scratch table of `256 × 256` entries, all `u16::MAX`, left so.
fn predictors(vertex_count: usize, triangles: &[[u32; 3]], edges: &mut [u16]) -> (Vec<Predictor>, Parallelograms) {
    // Triangles around each vertex, in triangle order (compressed rows).
    let mut first_use = vec![u32::MAX; vertex_count];
    let mut row = vec![0u32; vertex_count + 1];
    for (t, tri) in triangles.iter().enumerate() {
        for &v in tri {
            if first_use[v as usize] == u32::MAX {
                first_use[v as usize] = t as u32;
            }
            row[v as usize + 1] += 1;
        }
    }
    for v in 0..vertex_count {
        row[v + 1] += row[v];
    }
    let mut fill = row.clone();
    let mut around = vec![0u32; triangles.len() * 3];
    for (t, tri) in triangles.iter().enumerate() {
        for &v in tri {
            around[fill[v as usize] as usize] = t as u32;
            fill[v as usize] += 1;
        }
    }
    // The triangle of every directed edge (the first one, where several share it).
    for (t, tri) in triangles.iter().enumerate().rev() {
        for k in 0..3 {
            edges[(tri[k] as usize) << 8 | tri[(k + 1) % 3] as usize] = t as u16;
        }
    }
    // The third corner of the triangle across edge x–y from triangle t, if coded before v: the
    // one that runs the edge the other way (consistent winding), else the same way.
    let across = |x: u32, y: u32, t: u32, v: u32| {
        let found = |a: u32, b: u32| Some(edges[(a as usize) << 8 | b as usize]).filter(|&o| o != u16::MAX && o as u32 != t);
        let o = found(y, x).or_else(|| found(x, y))?;
        let c = triangles[o as usize].iter().copied().find(|&z| z != x && z != y)?;
        (c < v).then_some(c)
    };
    let mut multi = Parallelograms { offsets: Vec::with_capacity(vertex_count + 1), list: Vec::with_capacity(vertex_count * 2) };
    multi.offsets.push(0);
    for v in 0..vertex_count as u32 {
        for &t in &around[row[v as usize] as usize..row[v as usize + 1] as usize] {
            let tri = triangles[t as usize];
            let k = tri.iter().position(|&x| x == v).unwrap_or(0);
            let (x, y) = (tri[(k + 1) % 3], tri[(k + 2) % 3]);
            if x < v && y < v {
                if let Some(c) = across(x, y, t, v) {
                    multi.list.push([x, y, c]);
                }
            }
        }
        multi.offsets.push(multi.list.len() as u32);
    }
    let single = (0..vertex_count as u32)
        .map(|v| {
            if v == 0 {
                return Predictor::First;
            }
            let Some(&t) = first_use.get(v as usize).filter(|&&t| t != u32::MAX) else { return Predictor::Previous(v - 1) };
            let tri = triangles[t as usize];
            let mut known = tri.iter().copied().filter(|&x| x < v);
            match (known.next(), known.next()) {
                (None, _) => Predictor::Previous(v - 1),
                (Some(a), None) => Predictor::Corner(a),
                (Some(a), Some(b)) => {
                    // A triangle across edge a–b whose third corner is coded before v.
                    match across(a, b, t, v) {
                        Some(c) => Predictor::Parallelogram(a, b, c),
                        None => Predictor::Midpoint(a, b),
                    }
                }
            }
        })
        .collect();
    for tri in triangles {
        for k in 0..3 {
            edges[(tri[k] as usize) << 8 | tri[(k + 1) % 3] as usize] = u16::MAX;
        }
    }
    (single, multi)
}

/// The octahedral normal (`bits` per component, unpacked) predicted for every vertex from the
/// area-weighted face normals around it; `None` where they cancel out. (A prediction: it rounds
/// to the nearest grid point instead of searching for the best one as the encoder did.)
fn geometry_normals(positions: &[IVec3], triangles: &[[u32; 3]], bits: u32) -> Vec<Option<[i32; 2]>> {
    let mut sums = vec![[0i64; 3]; positions.len()];
    for tri in triangles {
        let [a, b, c] = tri.map(|v| positions[v as usize].as_i64vec3());
        let n = (b - a).cross(c - a);
        for &v in tri {
            let s = &mut sums[v as usize];
            s[0] = s[0].wrapping_add(n.x);
            s[1] = s[1].wrapping_add(n.y);
            s[2] = s[2].wrapping_add(n.z);
        }
    }
    let m = ((1u32 << (bits - 1)) - 1) as f64;
    sums.iter()
        .map(|s| {
            // Integer → f64 conversions, `+ − × ÷`, `abs` and `round` are exact or IEEE-rounded.
            let [x, y, z] = s.map(|c| c as f64);
            let l1 = x.abs() + y.abs() + z.abs();
            if l1 == 0.0 {
                return None;
            }
            let (mut u, mut v) = (x / l1, y / l1);
            if z < 0.0 {
                let sign = |c: f64| if c >= 0.0 { 1.0 } else { -1.0 };
                (u, v) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
            }
            let q = |c: f64| ((c * m).round().clamp(-m, m) + m) as i32;
            Some([q(u), q(v)])
        })
        .collect()
}

/// The texture coordinates of vertex `v` mapped from triangle `t`'s (all known): `v`'s position
/// in the triangle's plane as `a + s (b − a) + t (c − a)`, applied to the UVs. Exact where the
/// UVs are an affine function of the position over the triangle's plane (planar or box mapping)
/// and `v` lies in it; `None` for a degenerate triangle.
fn map_uv(vertices: &[QuantizedVertex], [a, b, c]: [usize; 3], v: usize, uv: impl Fn(&QuantizedVertex) -> IVec2) -> Option<[i32; 2]> {
    let p = |i: usize| vertices[i].position.as_dvec3();
    let (e1, e2, d) = (p(b) - p(a), p(c) - p(a), p(v) - p(a));
    let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
    let det = d11 * d22 - d12 * d12;
    if !(det > 0.0) {
        return None;
    }
    let (q1, q2) = (d.dot(e1), d.dot(e2));
    let s = (q1 * d22 - q2 * d12) / det;
    let t = (q2 * d11 - q1 * d12) / det;
    let (ua, ub, uc) = (uv(&vertices[a]).as_dvec2(), uv(&vertices[b]).as_dvec2(), uv(&vertices[c]).as_dvec2());
    let m = ua + (ub - ua) * s + (uc - ua) * t;
    let limit = i32::MAX as f64 / 2.0;
    (m.x.abs() < limit && m.y.abs() < limit).then(|| [m.x.round() as i32, m.y.round() as i32])
}

/// What a vertex repeats of an earlier vertex of the page.
const REPEAT_NONE: u32 = 0;
const REPEAT_ALL: u32 = 1;
const REPEAT_POSITION: u32 = 2;

/// Codes (or decodes into) a cluster's triangles; its counts are set.
fn code_triangles<C: Coder>(coder: &mut C, cluster: &mut TriangleCluster, encoding: bool) -> Result<(), Corrupt> {
    let n = cluster.vertices.len() as u32;
    // Ring buffer of directed edges, newest at `head - 1`.
    let mut fifo = [(u32::MAX, u32::MAX); EDGE_FIFO];
    let mut head = 0usize;
    let mut next = 0u32;
    let mut previous_hit = 0u8;
    let corner = |coder: &mut C, c: u8, v: u32, next: &mut u32| -> Result<u32, Corrupt> {
        let code = coder.value(c, next.wrapping_sub(v))?;
        let v = next.checked_sub(code).ok_or("corrupt page: vertex reference")?;
        if v == *next {
            *next += 1;
        }
        if v >= n {
            return Err("corrupt page: vertex index".into());
        }
        Ok(v)
    };
    for t in 0..cluster.triangles.len() {
        let tri = cluster.triangles[t];
        // The FIFO slot (newest first) and rotation of a remembered edge this triangle shares.
        let mut hit = 0u32;
        if encoding {
            'search: for k in 0..EDGE_FIFO {
                let (a, b) = fifo[(head + EDGE_FIFO - 1 - k) % EDGE_FIFO];
                for r in 0..3 {
                    if tri[r] == b && tri[(r + 1) % 3] == a {
                        hit = 1 + (k * 3 + r) as u32;
                        break 'search;
                    }
                }
            }
        }
        let slot = coder.symbol(ctx::EDGE + previous_hit, hit.div_ceil(3))?;
        let hit = if slot == 0 { 0 } else { (slot - 1) * 3 + coder.symbol(ctx::EDGE_ROTATION, (hit + 2) % 3)? + 1 };
        previous_hit = (hit != 0) as u8;
        let tri = if hit == 0 {
            let mut out = [0u32; 3];
            for k in 0..3 {
                out[k] = corner(coder, ctx::CORNER + k as u8, tri[k], &mut next)?;
            }
            out
        } else {
            let (k, r) = (((hit - 1) / 3) as usize, ((hit - 1) % 3) as usize);
            if k >= EDGE_FIFO {
                return Err("corrupt page: edge reference".into());
            }
            let (a, b) = fifo[(head + EDGE_FIFO - 1 - k) % EDGE_FIFO];
            let third = corner(coder, ctx::THIRD, tri[(r + 2) % 3], &mut next)?;
            let mut out = [0u32; 3];
            out[r] = b;
            out[(r + 1) % 3] = a;
            out[(r + 2) % 3] = third;
            if a >= n || b >= n {
                return Err("corrupt page: edge reference".into());
            }
            out
        };
        cluster.triangles[t] = tri;
        for r in 0..3 {
            fifo[head] = (tri[r], tri[(r + 1) % 3]);
            head = (head + 1) % EDGE_FIFO;
        }
    }
    if next != n {
        return Err("corrupt page: unused vertices".into());
    }
    Ok(())
}

/// Page-level state shared by the clusters of a page.
#[derive(Default)]
struct PageState {
    /// Every vertex coded so far in the page.
    vertices: Vec<QuantizedVertex>,
    /// Encoder only: latest page index of each full vertex and of each position.
    full: HashMap<VertexKey, u32>,
    positions: HashMap<[i32; 3], u32>,
    first_position: [i32; 3],
    params: Option<u32>,
    /// Scratch for [`predictors`].
    edges: Vec<u16>,
}

type VertexKey = ([i32; 3], u32, [i32; 2], [i32; 2], [u8; 4], [u8; 2], [u16; MAX_INFLUENCES], [u8; MAX_INFLUENCES]);

fn key(v: &QuantizedVertex) -> VertexKey {
    (v.position.to_array(), v.normal, v.uv.to_array(), v.uv1.to_array(), v.color, v.plant, v.skin.joints, v.skin.weights)
}

fn code_vertices<C: Coder>(coder: &mut C, page: &mut PageState, cluster: &mut TriangleCluster, encoding: bool) -> Result<(), Corrupt> {
    let n = cluster.vertices.len();
    let q = cluster.quantization;
    let base = page.vertices.len() as u32;
    let (predict, multi) = predictors(n, &cluster.triangles, &mut page.edges);

    // Repeats and positions.
    let mut repeat = vec![REPEAT_NONE; n];
    let mut previous_kind = 0u8;
    let mut previous_target = 0u32;
    for v in 0..n {
        let global = base + v as u32;
        let (mut kind, mut target) = (REPEAT_NONE, 0);
        if encoding {
            let vertex = &cluster.vertices[v];
            if let Some(&t) = page.full.get(&key(vertex)) {
                (kind, target) = (REPEAT_ALL, t);
            } else if let Some(&t) = page.positions.get(&vertex.position.to_array()) {
                (kind, target) = (REPEAT_POSITION, t);
            }
        }
        kind = coder.symbol(ctx::REPEAT + previous_kind, kind)?;
        if kind > REPEAT_POSITION {
            return Err("corrupt page: repeat kind".into());
        }
        previous_kind = kind as u8;
        repeat[v] = kind;
        if kind != REPEAT_NONE {
            let target = if previous_kind_was_repeat(&repeat, v) {
                let step = coder.signed(ctx::REPEAT_STEP, target.wrapping_sub(previous_target).wrapping_sub(1) as i32)?;
                previous_target.wrapping_add(1).wrapping_add(step as u32)
            } else {
                let distance = coder.value(ctx::REPEAT_REF, global.wrapping_sub(target).wrapping_sub(1))?;
                global.wrapping_sub(distance).wrapping_sub(1)
            };
            if target >= global {
                return Err("corrupt page: repeat reference".into());
            }
            previous_target = target;
            let source = if target >= base { cluster.vertices[(target - base) as usize] } else { page.vertices[target as usize] };
            if kind == REPEAT_ALL {
                cluster.vertices[v] = source;
            } else {
                cluster.vertices[v].position = source.position;
            }
        } else {
            let p = predict[v];
            let get = |i: u32| cluster.vertices[i as usize].position.to_array();
            let (predicted, base_ctx) = match (p, multi.of(v).len()) {
                (Predictor::Parallelogram(..), 0 | 1) => (p.apply(get, page.first_position), Some(ctx::PARALLELOGRAM)),
                (_, 0) | (_, 1) => (p.apply(get, page.first_position), None),
                (_, count) => {
                    // The mean of all parallelograms around the vertex (Draco's multi-parallelogram).
                    let mut sum = [0i64; 3];
                    for &[x, y, c] in multi.of(v) {
                        let (x, y, c) = (get(x), get(y), get(c));
                        for a in 0..3 {
                            sum[a] += x[a] as i64 + y[a] as i64 - c[a] as i64;
                        }
                    }
                    let count = count as i64;
                    (sum.map(|s| (s + count / 2).div_euclid(count) as i32), Some(ctx::MULTI_PARALLELOGRAM))
                }
            };
            let actual = cluster.vertices[v].position.to_array();
            let mut out = [0i32; 3];
            let mut size = 0;
            for a in 0..3 {
                // After the first axis, the size of the residuals so far picks the statistics.
                let c = match base_ctx {
                    Some(base) => base + if a == 0 { 0 } else { 1 + (a as u8 - 1) * 3 + size },
                    None => ctx::POSITION + p.kind() * 3 + a as u8,
                };
                let residual = coder.signed(c, actual[a].wrapping_sub(predicted[a]))?;
                let bits = 32 - zigzag(residual).leading_zeros();
                size = size.max(if bits <= 2 { 0 } else if bits <= 5 { 1 } else { 2 });
                out[a] = predicted[a].wrapping_add(residual);
            }
            cluster.vertices[v].position = IVec3::from_array(out);
        }
        if v == 0 {
            page.first_position = cluster.vertices[0].position.to_array();
        }
    }
    // Normals: predicted from the geometry, or from the neighbours (the cluster's choice).
    let bits = q.normal_bits;
    let mask = (1u32 << bits) - 1;
    let positions: Vec<IVec3> = cluster.vertices.iter().map(|v| v.position).collect();
    let geometric = geometry_normals(&positions, &cluster.triangles, bits);
    let unpack = |packed: u32| [(packed & mask) as i32, ((packed >> bits) & mask) as i32];
    let predict_normal = |vertices: &[QuantizedVertex], v: usize, mode: u32, last: [i32; 2]| -> ([i32; 2], u8) {
        let get = |i: u32| unpack(vertices[i as usize].normal);
        if mode == 0 {
            if let Some(g) = geometric[v] {
                return (g, ctx::NORMAL);
            }
        } else {
            match predict[v] {
                Predictor::Corner(a) => return (get(a), ctx::NORMAL + 4),
                Predictor::Midpoint(a, b) | Predictor::Parallelogram(a, b, _) => {
                    let (a, b) = (get(a), get(b));
                    return ([(a[0] + b[0]) >> 1, (a[1] + b[1]) >> 1], ctx::NORMAL + 4);
                }
                _ => {}
            }
        }
        (last, ctx::NORMAL + 2)
    };
    let initial = [(mask / 2) as i32; 2];
    let mode = if encoding {
        let cost = |mode: u32| {
            let mut last = initial;
            let mut cost = 0;
            for v in (0..n).filter(|&v| repeat[v] != REPEAT_ALL) {
                let (p, _) = predict_normal(&cluster.vertices, v, mode, last);
                last = unpack(cluster.vertices[v].normal);
                cost += estimate(last[0] - p[0]) + estimate(last[1] - p[1]);
            }
            cost
        };
        (cost(1) < cost(0)) as u32
    } else {
        0
    };
    let mode = coder.symbol(ctx::NORMAL_MODE, mode)?;
    let mut last = initial;
    for v in 0..n {
        if repeat[v] == REPEAT_ALL {
            continue;
        }
        let (predicted, c) = predict_normal(&cluster.vertices, v, mode, last);
        let actual = unpack(cluster.vertices[v].normal);
        let mut out = [0i32; 2];
        for a in 0..2 {
            out[a] = predicted[a].wrapping_add(coder.signed(c + a as u8, actual[a].wrapping_sub(predicted[a]))?);
            if out[a] as u32 > mask {
                return Err("corrupt page: normal".into());
            }
        }
        cluster.vertices[v].normal = out[0] as u32 | ((out[1] as u32) << bits);
        last = out;
    }
    // UVs: with the position predictors in texture space, or mapped from the geometry where a
    // parallelogram has a full triangle to map from (the cluster's choice).
    let predict_uv = |vertices: &[QuantizedVertex], v: usize, mode: u32, first: [i32; 2]| -> ([i32; 2], u8) {
        let p = predict[v];
        if mode == 1 {
            if let Predictor::Parallelogram(a, b, c) = p {
                if let Some(uv) = map_uv(vertices, [a as usize, b as usize, c as usize], v, |x| x.uv) {
                    return (uv, ctx::UV_MAPPED);
                }
            }
        }
        (p.apply(|i| vertices[i as usize].uv.to_array(), first), ctx::UV + p.kind() * 2)
    };
    let mode = if encoding {
        let cost = |mode: u32| {
            let mut first = [0; 2];
            let mut cost = 0;
            for v in (0..n).filter(|&v| repeat[v] != REPEAT_ALL) {
                let (p, _) = predict_uv(&cluster.vertices, v, mode, first);
                first = cluster.vertices[v].uv.to_array();
                cost += estimate(first[0].wrapping_sub(p[0])) + estimate(first[1].wrapping_sub(p[1]));
            }
            cost
        };
        (cost(1) < cost(0)) as u32
    } else {
        0
    };
    let mode = coder.symbol(ctx::UV_MODE, mode)?;
    let mut first_uv = [0i32; 2];
    for v in 0..n {
        if repeat[v] == REPEAT_ALL {
            continue;
        }
        let (predicted, c) = predict_uv(&cluster.vertices, v, mode, first_uv);
        let actual = cluster.vertices[v].uv.to_array();
        let mut out = [0i32; 2];
        for a in 0..2 {
            out[a] = predicted[a].wrapping_add(coder.signed(c + a as u8, actual[a].wrapping_sub(predicted[a]))?);
        }
        cluster.vertices[v].uv = IVec2::from_array(out);
        first_uv = out;
        if q.attributes.uv1 {
            let predicted = predict[v].apply(|i| cluster.vertices[i as usize].uv1.to_array(), [0; 2]);
            let actual = cluster.vertices[v].uv1.to_array();
            for a in 0..2 {
                out[a] = predicted[a].wrapping_add(coder.signed(ctx::UV1 + a as u8, actual[a].wrapping_sub(predicted[a]))?);
            }
            cluster.vertices[v].uv1 = IVec2::from_array(out);
        }
    }

    // Colours, plant data and skin weights: from the vertex the predictor leans on.
    for v in 0..n {
        if repeat[v] == REPEAT_ALL {
            continue;
        }
        let from = predict[v].neighbour().map(|i| cluster.vertices[i]);
        if q.attributes.color {
            let predicted = from.map_or([255; 4], |f| f.color);
            for ch in 0..4 {
                let actual = cluster.vertices[v].color[ch];
                let c = if ch == 3 { 3 } else { ch.min(1) };
                let r = coder.signed(ctx::COLOR + c as u8, actual.wrapping_sub(predicted[ch]) as i8 as i32)?;
                cluster.vertices[v].color[ch] = predicted[ch].wrapping_add(r as u8);
            }
        }
        if q.attributes.plant {
            let predicted = from.map_or([0; 2], |f| f.plant);
            for ch in 0..2 {
                let actual = cluster.vertices[v].plant[ch];
                let r = coder.signed(ctx::PLANT + ch as u8, actual as i32 - predicted[ch] as i32)?;
                let value = (predicted[ch] as i32).wrapping_add(r);
                if !(0..32).contains(&value) {
                    return Err("corrupt page: plant data".into());
                }
                cluster.vertices[v].plant[ch] = value as u8;
            }
        }
        if q.attributes.skin {
            let predicted = from.map_or(SkinWeights::default(), |f| f.skin);
            let mut skin = cluster.vertices[v].skin;
            for slot in 0..MAX_INFLUENCES {
                let w = coder.signed(ctx::SKIN_WEIGHT, skin.weights[slot] as i32 - predicted.weights[slot] as i32)?;
                skin.weights[slot] = (predicted.weights[slot] as i32).wrapping_add(w) as u8;
                let j = coder.signed(ctx::SKIN_JOINT, skin.joints[slot] as i32 - predicted.joints[slot] as i32)?;
                skin.joints[slot] = (predicted.joints[slot] as i32).wrapping_add(j) as u16;
            }
            cluster.vertices[v].skin = skin;
        }
    }

    for (v, vertex) in cluster.vertices.iter().enumerate() {
        if encoding {
            page.full.insert(key(vertex), base + v as u32);
            page.positions.insert(vertex.position.to_array(), base + v as u32);
        }
        page.vertices.push(*vertex);
    }
    Ok(())
}

/// Rough bits to code a residual (for the encoder's choices).
fn estimate(r: i32) -> u32 {
    2 * (32 - zigzag(r).leading_zeros()) + 1
}

fn previous_kind_was_repeat(repeat: &[u32], v: usize) -> bool {
    v > 0 && repeat[v - 1] != REPEAT_NONE
}

/// Codes a voxel cluster's words.
fn code_voxels<C: Coder>(coder: &mut C, words: &mut Vec<u32>, page_words: usize) -> Result<(), Corrupt> {
    let header = crate::voxel::VOXEL_HEADER_WORDS as usize;
    let len = coder.value(ctx::VOXEL_LEN, words.len() as u32)? as usize;
    if len < header || len > page_words {
        return Err("corrupt page: voxel cluster".into());
    }
    words.resize(len, 0);
    for w in words.iter_mut().take(header) {
        *w = coder.raw(*w, 32);
    }
    let bricks = (words[4] & 0xff) as usize;
    let brick_end = (header + bricks * crate::voxel::BRICK_WORDS as usize).min(len);
    for i in header..len {
        let (c, previous) = if i < brick_end {
            let lane = (i - header) % 4;
            (ctx::VOXEL_BRICK + lane as u8 * 4, if i >= header + 4 { words[i - 4] } else { 0 })
        } else {
            (ctx::VOXEL_ATTRIBUTE, if i > brick_end { words[i - 1] } else { 0 })
        };
        let (actual, previous, mut out) = (words[i].to_le_bytes(), previous.to_le_bytes(), [0u8; 4]);
        for b in 0..4 {
            out[b] = previous[b].wrapping_add(coder.signed(c + b as u8, actual[b].wrapping_sub(previous[b]) as i8 as i32)? as u8);
        }
        words[i] = u32::from_le_bytes(out);
    }
    Ok(())
}

/// A page's clusters in coding form.
enum Cluster {
    Triangles(TriangleCluster),
    Voxels(Vec<u32>),
}

/// Codes a page of `words` words (which bounds what a corrupt page can make the decoder allocate).
fn code_page<C: Coder>(coder: &mut C, clusters: &mut Vec<Cluster>, words: usize, encoding: bool) -> Result<(), Corrupt> {
    let count = coder.value(ctx::CLUSTERS, clusters.len() as u32)? as usize;
    // Every cluster takes at least 8 words.
    if count > words / 8 + 1 {
        return Err("corrupt page: cluster count".into());
    }
    let mut page = PageState { edges: vec![u16::MAX; 1 << 16], ..Default::default() };
    for i in 0..count {
        let kind = if encoding { matches!(clusters[i], Cluster::Voxels(_)) as u32 } else { 0 };
        let kind = coder.symbol(ctx::KIND, kind)?;
        if !encoding {
            let blank = Quantization { position_exponent: 0, uv_exponent: 0, normal_bits: 2, attributes: VertexAttributes::default() };
            clusters.push(match kind {
                0 => Cluster::Triangles(TriangleCluster { quantization: blank, vertices: Vec::new(), triangles: Vec::new() }),
                1 => Cluster::Voxels(Vec::new()),
                _ => return Err("corrupt page: cluster kind".into()),
            });
        }
        match &mut clusters[i] {
            Cluster::Voxels(voxels) => code_voxels(coder, voxels, words)?,
            Cluster::Triangles(cluster) => {
                let n = coder.value(ctx::VERTEX_COUNT, cluster.vertices.len() as u32)? as usize;
                let t = coder.value(ctx::TRIANGLE_COUNT, cluster.triangles.len() as u32)? as usize;
                if n == 0 || n > 256 || t > 256 {
                    return Err("corrupt page: cluster size".into());
                }
                let params = params_word(&cluster.quantization);
                let changed = coder.symbol(ctx::PARAMS, (page.params != Some(params)) as u32)? != 0;
                let params = if changed { coder.raw(params, 32) } else { page.params.ok_or("corrupt page: parameters")? };
                page.params = Some(params);
                cluster.quantization = params_from_word(params)?;
                let blank = cluster.quantization.vertex(Vec3::ZERO, Vec3::Z, Vec2::ZERO);
                cluster.vertices.resize(n, blank);
                cluster.triangles.resize(t, [0; 3]);
                code_triangles(coder, cluster, encoding)?;
                code_vertices(coder, &mut page, cluster, encoding)?;
            }
        }
    }
    Ok(())
}

/// The clusters of a page in coding form, if the page is laid out as the builder lays it out
/// (clusters back to back, nothing else).
fn page_clusters(page: &[u32], clusters: &[GpuCluster]) -> Option<Vec<Cluster>> {
    let mut out = Vec::with_capacity(clusters.len());
    let mut offset = 0usize;
    for (i, c) in clusters.iter().enumerate() {
        let end = clusters.get(i + 1).map_or(page.len(), |n| n.vertex_offset as usize);
        if c.vertex_offset as usize != offset || end > page.len() || end < offset {
            return None;
        }
        if c.is_voxel() {
            out.push(Cluster::Voxels(page[offset..end].to_vec()));
        } else {
            if c.vertex_count() == 0 || c.triangle_offset as usize > end {
                return None;
            }
            out.push(Cluster::Triangles(parse_cluster(page, c)));
        }
        offset = end;
    }
    (offset == page.len()).then_some(out)
}

/// Whether `encode_cluster` can pack these values (a corrupt page must fail, not panic).
fn packable(cluster: &TriangleCluster) -> bool {
    let span = |f: &dyn Fn(&QuantizedVertex) -> i32| {
        let (min, max) = cluster.vertices.iter().map(f).fold((i32::MAX, i32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
        (max as i64 - min as i64) < (1 << 31)
    };
    let a = cluster.quantization.attributes;
    (0..3).all(|i| span(&|v| v.position[i]))
        && (0..2).all(|i| span(&|v| v.uv[i]) && (!a.uv1 || span(&|v| v.uv1[i])))
        && cluster.quantization.position_exponent.abs() <= 64
        && cluster.quantization.uv_exponent.abs() <= 64
}

fn rebuild(clusters: &[Cluster], words: usize) -> Result<Vec<u32>, Corrupt> {
    let mut out = Vec::with_capacity(words);
    for c in clusters {
        match c {
            Cluster::Voxels(w) => out.extend_from_slice(w),
            Cluster::Triangles(t) => {
                if !packable(t) {
                    return Err("corrupt page: values out of range".into());
                }
                encode_cluster(&t.quantization, &t.vertices, &t.triangles, &mut out);
            }
        }
        if out.len() > words {
            break;
        }
    }
    Ok(out)
}

/// Stores a page payload: entropy coded if that reproduces it exactly and is smaller, else as
/// it is. `clusters` are the page's cluster records.
pub fn encode_page(page: &[u32], clusters: &[GpuCluster]) -> Vec<u8> {
    let raw = || {
        let mut out = Vec::with_capacity(1 + page.len() * 4);
        out.push(MODE_RAW);
        out.extend_from_slice(bytemuck::cast_slice(page));
        out
    };
    let Some(mut parsed) = page_clusters(page, clusters) else { return raw() };
    let mut recorder = Recorder::default();
    if code_page(&mut recorder, &mut parsed, page.len(), true).is_err() {
        return raw();
    }
    let mut out = vec![MODE_CODED];
    put_varint(&mut out, page.len() as u32);
    finish(recorder, &mut out);
    if out.len() >= 1 + page.len() * 4 || decode_page(&out).as_deref() != Ok(page) {
        return raw();
    }
    out
}

/// Inverse of [`encode_page`]: the page payload's words.
pub fn decode_page(bytes: &[u8]) -> Result<Vec<u32>, String> {
    match bytes.first() {
        Some(&MODE_RAW) => {
            if (bytes.len() - 1) % 4 != 0 {
                return Err("truncated page".into());
            }
            Ok(bytemuck::pod_collect_to_vec(&bytes[1..]))
        }
        Some(&MODE_CODED) => {
            let mut pos = 1;
            let words = get_varint(bytes, &mut pos)? as usize;
            if words > 1 << 24 {
                return Err("corrupt page: size".into());
            }
            let tables = read_tables(bytes, &mut pos)?;
            let rans_len = get_varint(bytes, &mut pos)? as usize;
            let rans = bytes.get(pos..pos + rans_len).filter(|r| r.len() >= 4).ok_or("truncated page")?;
            let state = u32::from_be_bytes(rans[..4].try_into().unwrap());
            let mut decoder = Decoder { tables, rans, pos: 4, state, extra: BitSource::new(&bytes[pos + rans_len..]) };
            let mut clusters = Vec::new();
            code_page(&mut decoder, &mut clusters, words, false)?;
            let out = rebuild(&clusters, words)?;
            if out.len() != words {
                return Err(format!("corrupt page: {} words, expected {words}", out.len()));
            }
            Ok(out)
        }
        _ => Err("unknown page coding".into()),
    }
}


/// Whether a stored page is entropy coded (as opposed to stored as it is).
pub fn is_coded(bytes: &[u8]) -> bool {
    bytes.first() == Some(&MODE_CODED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_virtual_mesh, primitives, BuildConfig, SdfMode, VoxelConfig};

    fn check_mesh(vm: &crate::VirtualMesh, max_ratio: f32) {
        let (mut raw, mut coded) = (0, 0);
        for p in 0..vm.pages.len() {
            let page = vm.page_payload(p);
            let stored = encode_page(page, vm.page_clusters(p));
            assert_eq!(decode_page(&stored).unwrap(), page, "page {p} round-trips");
            raw += page.len() * 4;
            coded += stored.len();
        }
        let ratio = coded as f32 / raw as f32;
        assert!(ratio < max_ratio, "coded pages are {:.0} % of the GPU format", ratio * 100.0);
    }

    #[test]
    fn pages_round_trip_and_shrink() {
        let rock = build_virtual_mesh(&primitives::rock(5, 5), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        check_mesh(&rock, 0.6);
        // Voxel clusters (aggregate geometry) and plain spheres.
        let voxels = Some(VoxelConfig { voxels_per_triangle: 16.0 });
        let leaves = build_virtual_mesh(&primitives::leaf_cloud(3000, 9), &BuildConfig { sdf: SdfMode::None, voxels, ..Default::default() });
        assert!(leaves.clusters.iter().any(|c| c.is_voxel()));
        check_mesh(&leaves, 0.95);
        check_mesh(&build_virtual_mesh(&primitives::icosphere(4), &BuildConfig { sdf: SdfMode::None, ..Default::default() }), 0.6);
    }

    #[test]
    fn tokens_and_levels_round_trip() {
        for v in (0..100_000u32).chain([u32::MAX, u32::MAX - 1, 1 << 31, (1 << 31) - 1]) {
            let (t, n) = token(v);
            assert!((t as usize) < TOKENS);
            let low = v & ((1u64 << n) - 1) as u32;
            assert_eq!(
                untoken(t, |count| {
                    assert_eq!(count, n);
                    low
                }),
                v
            );
        }
        for v in [-5, 0, 7, i32::MIN, i32::MAX] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
        let mut counts = [0u32; TOKENS];
        for (i, c) in counts.iter_mut().enumerate().take(40) {
            *c = (i as u32 * 7919) % 1000;
        }
        let freq = level_frequencies(&levels(&counts));
        assert_eq!(freq.iter().map(|&f| f as u32).sum::<u32>(), SCALE);
        assert!(counts.iter().zip(&freq).all(|(&c, &f)| (c == 0) == (f == 0)));
    }

    /// Damaged pages fail to decode (or decode to something else) but never panic.
    #[test]
    fn corrupt_pages_never_panic() {
        let vm = build_virtual_mesh(&primitives::rock(3, 4), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let stored = encode_page(vm.page_payload(0), vm.page_clusters(0));
        assert!(is_coded(&stored));
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..1500 {
            let mut bytes = stored.clone();
            match round % 4 {
                0 => bytes.truncate(next() as usize % bytes.len()),
                1 => {
                    let i = 1 + next() as usize % (bytes.len() - 1);
                    bytes[i] ^= 1 << (next() % 8);
                }
                2 => {
                    for _ in 0..8 {
                        let i = 1 + next() as usize % (bytes.len() - 1);
                        bytes[i] = next() as u8;
                    }
                }
                _ => {
                    let i = 1 + next() as usize % (bytes.len() - 1);
                    for b in &mut bytes[i..] {
                        *b = next() as u8;
                    }
                }
            }
            let _ = decode_page(&bytes);
        }
        assert!(decode_page(&[]).is_err() && decode_page(&[9]).is_err() && decode_page(&[MODE_RAW, 1, 2]).is_err());
    }
}
