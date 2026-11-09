//! Compressed GPU encoding of cluster geometry (as in Nanite).
//!
//! Page payloads hold clusters in this format both in the GPU pool and in `.hvm` files; every
//! consumer (hardware and software rasterisers, shadow rasterisers, material resolve) decodes
//! it on the fly through `geometry.wgsl`, which mirrors [`ClusterEncoding`].
//!
//! ## Quantisation
//!
//! * **Positions** snap to one grid per mesh with a power-of-two step `2^e`
//!   ([`Quantization::position_exponent`]): about 1/64 of the mean shortest triangle edge (so
//!   thin triangles such as grass blades keep their shape), at most
//!   ~1 mm (2^-10), coarser only where needed to keep every grid coordinate within ±2^23.
//!   Each cluster stores the offset of every vertex from the cluster's minimum grid point
//!   with just enough bits per axis for the cluster's extent.
//! * **Normals** are octahedral, `normal_bits` per component (snorm-style, so the axes are
//!   exact), choosing the best of the four nearest grid points.
//! * **UVs** snap to a per-mesh grid of step 2^-15 (≤ 1/32 texel of error on a 2k texture) and
//!   are stored like positions: an offset from the cluster's minimum with per-axis bit widths.
//! * **Optional attributes** (imported meshes only): a second UV set on the same grid as the
//!   first, stored the same way, and RGBA8 vertex colours (RGB with a square-root curve, so
//!   dark colours keep their precision; alpha linear), each channel an offset from the
//!   cluster's minimum with just enough bits. Meshes without them encode exactly as before.
//! * **Skin weights** (skeletal meshes, [`crate::skin`]): per vertex up to four joint indices,
//!   stored as offsets from the cluster's smallest joint with just enough bits, and the
//!   quantised 8-bit weights of all but the heaviest influence (each at most half the total, so
//!   7 bits or fewer; the heaviest is the total minus the others). A cluster stores as many
//!   influences as its most-influenced vertex has: rigid clusters store one joint and no weight.
//! * **Triangles** are three cluster-local indices of `index_bits(vertex count)` bits each.
//!
//! ## Why it stays crack-free
//!
//! The grids are per mesh, not per cluster: a vertex shared by two clusters — across a cluster
//! border, or across a locked group border between LOD levels — has the same grid coordinates
//! in both. The decoder rebuilds a coordinate as `f32(min + offset) * 2^e`: an integer of at most
//! 24 bits times a power of two, which is exact in `f32`. Both clusters (and the CPU, which
//! checks watertightness on decoded positions) therefore reproduce bit-identical positions.
//! Quantising per cluster range instead would round a shared vertex differently on each side.
//!
//! ## Layout of one cluster (words, at `GpuCluster::vertex_offset` of its page)
//!
//! ```text
//! [0..3) position minimum (i32 grid units)     [3..5) uv minimum (i32 grid units)
//! [5]    bit widths: position x | y << 5 | z << 10, uv u << 15 | v << 20, normal << 25;
//!        bit 30: extended (the mesh has optional attributes)
//! [6]    grid exponents: position (i8) | uv (i8) << 8
//! only if extended, [7..11):
//! [7]    uv1 widths u | v << 5, colour widths r << 10 | g << 14 | b << 18 | a << 22,
//!        has uv1 << 26, has colour << 27, has skin << 28
//! [8..10) uv1 minimum (i32 units of the uv grid)   [10] colour minimum (r | g << 8 | b << 16 | a << 24)
//! [7..) or [11..)  vertex bit stream: every vertex's position offsets (x, y, z), then every
//!        vertex's attributes (octahedral normal x | y << normal bits, u, v, then uv1 u, v and
//!        colour r, g, b, a offsets where present)
//! skin only, word aligned after the vertex stream:
//!        [0] joint minimum | joint bits << 16 | weight bits << 21 | (influences - 1) << 25,
//!        then per vertex: `influences` joint offsets, then the weights of influences 1.. (the
//!        first is 255 minus their sum)
//! ```
//!
//! followed (word aligned, at `GpuCluster::triangle_offset`) by the triangle bit stream.
//! Positions come first so the position-only passes (rasterisers, shadows) touch only them;
//! skinning passes read the skin section too, which leaves the vertex stream's layout as it is
//! for static meshes.

use glam::{IVec2, IVec3, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};

use crate::skin::{SkinWeights, MAX_INFLUENCES, WEIGHT_TOTAL};

/// Words of the per-cluster header in front of the vertex bit stream.
pub const CLUSTER_HEADER_WORDS: u32 = 7;
/// Extra header words of clusters with optional attributes ([`VertexAttributes`]).
pub const CLUSTER_EXTENSION_WORDS: u32 = 4;
/// Bit of header word 5 marking a cluster with the extension words.
const EXTENDED_BIT: u32 = 1 << 30;
/// Bit of extension word 7 marking a skinned cluster (a skin section follows the vertex stream).
const SKIN_BIT: u32 = 1 << 28;
/// Default bits per octahedral normal component.
pub const NORMAL_BITS: u32 = 10;
/// Default UV grid: 2^-15 (half a step is 1/32 texel of a 2048² texture).
const UV_EXPONENT: i32 = -15;
/// Coarsest automatic position grid: 2^-10 ≈ 1 mm in a metre-scaled mesh.
const MAX_POSITION_EXPONENT: i32 = -10;
/// Automatic position grid steps per mean shortest triangle edge.
const STEPS_PER_EDGE: f32 = 64.0;
/// Largest grid coordinate magnitude, so `f32(coordinate)` is exact (≤ 2^24 also holds for sums).
const MAX_GRID: f32 = (1u32 << 23) as f32;
/// Largest field of the bit-width word.
const MAX_FIELD_BITS: u32 = 31;
/// Grid exponents are clamped to ±64: far inside both the i8 header field and normal floats,
/// so `2^e` and `2^-e` are both exact.
const MAX_EXPONENT: i32 = 64;

/// `2^e` (|e| ≤ [`MAX_EXPONENT`]), built bit-exactly (the shader does the same).
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Smallest exponent `e` whose grid keeps `max_abs` within ±[`MAX_GRID`] units.
fn min_exponent(max_abs: f32) -> i32 {
    if max_abs <= 0.0 || !max_abs.is_finite() {
        return -MAX_EXPONENT;
    }
    let mut e = (max_abs.log2().ceil() as i32 - 23).max(-MAX_EXPONENT);
    while max_abs * pow2(-e) > MAX_GRID {
        e += 1;
    }
    e
}

/// Bits needed to store values `0..=range`.
fn bits_for(range: u32) -> u32 {
    32 - range.leading_zeros()
}

/// Bits per cluster-local vertex index of a cluster with `vertex_count` vertices.
pub fn index_bits(vertex_count: u32) -> u32 {
    bits_for(vertex_count.max(2) - 1)
}

/// Optional vertex attributes of a mesh, encoded by all its clusters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexAttributes {
    /// A second UV set (on the grid of the first).
    pub uv1: bool,
    /// RGBA8 vertex colours ([`encode_color`]).
    pub color: bool,
    /// Skin weights ([`SkinWeights`]).
    pub skin: bool,
}

impl VertexAttributes {
    pub fn any(&self) -> bool {
        self.uv1 || self.color || self.skin
    }
}

/// The per-mesh quantisation grids. Every cluster of a mesh must use the same grids (see the
/// module documentation: that is what keeps shared vertices identical).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantization {
    /// Position grid step is `2^position_exponent` (mesh units).
    pub position_exponent: i32,
    /// UV grid step is `2^uv_exponent` (both UV sets).
    pub uv_exponent: i32,
    /// Bits per octahedral normal component (2..=16).
    pub normal_bits: u32,
    /// Optional attributes stored per vertex.
    pub attributes: VertexAttributes,
}

impl Quantization {
    /// Grids for a mesh whose triangles' shortest edges are `mean_edge` long on average, with
    /// the given coordinate magnitudes.
    /// `precision` overrides the automatic position step with `2^-precision` (Nanite's
    /// "position precision"); it is still coarsened if coordinates would not fit.
    pub fn new(precision: Option<i32>, mean_edge: f32, max_abs_position: f32, max_abs_uv: f32) -> Self {
        let automatic = || {
            let target = mean_edge / STEPS_PER_EDGE;
            let e = if target > 0.0 && target.is_finite() { target.log2().floor() as i32 } else { MAX_POSITION_EXPONENT };
            e.min(MAX_POSITION_EXPONENT)
        };
        let position = precision.map_or_else(automatic, |p| -p);
        Self {
            position_exponent: position.max(min_exponent(max_abs_position)).clamp(-MAX_EXPONENT, MAX_EXPONENT),
            uv_exponent: UV_EXPONENT.max(min_exponent(max_abs_uv)).clamp(-MAX_EXPONENT, MAX_EXPONENT),
            normal_bits: NORMAL_BITS,
            attributes: VertexAttributes::default(),
        }
    }

    /// The same grids, with optional attributes (`max_abs_uv` of [`Quantization::new`] must
    /// then cover the second UV set too).
    pub fn with_attributes(self, attributes: VertexAttributes) -> Self {
        Self { attributes, ..self }
    }

    pub fn position_step(&self) -> f32 {
        pow2(self.position_exponent)
    }

    pub fn uv_step(&self) -> f32 {
        pow2(self.uv_exponent)
    }

    /// Grid coordinates of `p` (scaling by a power of two is exact; only the rounding is lossy).
    pub fn quantize_position(&self, p: Vec3) -> IVec3 {
        (p * pow2(-self.position_exponent)).round().as_ivec3()
    }

    pub fn quantize_uv(&self, uv: Vec2) -> IVec2 {
        (uv * pow2(-self.uv_exponent)).round().as_ivec2()
    }

    /// The encoder's input for one vertex (no optional attributes: second UV at the origin,
    /// white, joint 0; see [`QuantizedVertex::with_uv1`], [`QuantizedVertex::with_color`],
    /// [`QuantizedVertex::with_skin`]).
    pub fn vertex(&self, position: Vec3, normal: Vec3, uv: Vec2) -> QuantizedVertex {
        QuantizedVertex {
            position: self.quantize_position(position),
            normal: encode_octahedral(normal, self.normal_bits),
            uv: self.quantize_uv(uv),
            uv1: IVec2::ZERO,
            color: [255; 4],
            skin: SkinWeights::default(),
        }
    }
}

/// A vertex on the mesh grids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantizedVertex {
    pub position: IVec3,
    /// [`encode_octahedral`] with the mesh's normal bits.
    pub normal: u32,
    pub uv: IVec2,
    /// Second UV set (encoded only if the mesh's [`VertexAttributes::uv1`]).
    pub uv1: IVec2,
    /// [`encode_color`] (encoded only if the mesh's [`VertexAttributes::color`]).
    pub color: [u8; 4],
    /// Encoded only if the mesh's [`VertexAttributes::skin`].
    pub skin: SkinWeights,
}

impl QuantizedVertex {
    pub fn with_uv1(self, quantization: &Quantization, uv1: Vec2) -> Self {
        Self { uv1: quantization.quantize_uv(uv1), ..self }
    }

    pub fn with_color(self, color: Vec4) -> Self {
        Self { color: encode_color(color), ..self }
    }

    pub fn with_skin(self, skin: SkinWeights) -> Self {
        Self { skin, ..self }
    }
}

/// The skin section descriptor of a cluster: joints are stored as offsets from `joint_min`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkinEncoding {
    pub joint_min: u32,
    pub joint_bits: u32,
    pub weight_bits: u32,
    /// Influences stored per vertex (1..=4).
    pub influences: u32,
}

impl SkinEncoding {
    fn fit(vertices: &[QuantizedVertex]) -> Self {
        let used = || vertices.iter().flat_map(|v| v.skin.joints.iter().zip(&v.skin.weights)).filter(|p| *p.1 > 0).map(|p| *p.0 as u32);
        let joint_min = used().min().unwrap_or(0);
        let joint_max = used().max().unwrap_or(0);
        let influences = vertices.iter().map(|v| v.skin.influences() as u32).max().unwrap_or(1).max(1);
        let weight_max = vertices.iter().flat_map(|v| v.skin.weights[1..].iter()).map(|&w| w as u32).max().unwrap_or(0);
        Self { joint_min, joint_bits: bits_for(joint_max - joint_min), weight_bits: bits_for(weight_max), influences }
    }

    fn to_word(self) -> u32 {
        self.joint_min | (self.joint_bits << 16) | (self.weight_bits << 21) | ((self.influences - 1) << 25)
    }

    fn from_word(word: u32) -> Self {
        Self { joint_min: word & 0xffff, joint_bits: (word >> 16) & 31, weight_bits: (word >> 21) & 15, influences: ((word >> 25) & 3) + 1 }
    }

    /// Bits per vertex.
    pub fn stride(&self) -> u32 {
        self.influences * self.joint_bits + (self.influences - 1) * self.weight_bits
    }

    fn write(&self, stream: &mut BitWriter, skin: &SkinWeights) {
        for slot in 0..self.influences as usize {
            let offset = if skin.weights[slot] > 0 { skin.joints[slot] as u32 - self.joint_min } else { 0 };
            stream.write(offset, self.joint_bits);
        }
        for slot in 1..self.influences as usize {
            stream.write(skin.weights[slot] as u32, self.weight_bits);
        }
    }

    /// Vertex `index` of the skin bit stream `stream` (after the descriptor word).
    fn read(&self, stream: &[u32], index: u32) -> SkinWeights {
        let mut bit = index * self.stride();
        let mut skin = SkinWeights { joints: [0; MAX_INFLUENCES], weights: [0; MAX_INFLUENCES] };
        for slot in 0..self.influences as usize {
            skin.joints[slot] = (self.joint_min + read_bits(stream, bit, self.joint_bits)) as u16;
            bit += self.joint_bits;
        }
        let mut rest = 0;
        for slot in 1..self.influences as usize {
            let w = read_bits(stream, bit, self.weight_bits);
            skin.weights[slot] = w as u8;
            rest += w;
            bit += self.weight_bits;
        }
        skin.weights[0] = (WEIGHT_TOTAL - rest) as u8;
        // Unused slots decode to the canonical joint 0.
        for slot in 0..MAX_INFLUENCES {
            if skin.weights[slot] == 0 {
                skin.joints[slot] = 0;
            }
        }
        skin
    }
}

/// RGBA8 vertex colour from linear RGBA: RGB on a square-root curve (more codes for dark
/// colours, decoded with one multiplication), alpha linear.
pub fn encode_color(color: Vec4) -> [u8; 4] {
    let c = color.clamp(Vec4::ZERO, Vec4::ONE);
    let unorm = |v: f32| (v * 255.0 + 0.5) as u8;
    [unorm(c.x.sqrt()), unorm(c.y.sqrt()), unorm(c.z.sqrt()), unorm(c.w)]
}

/// Inverse of [`encode_color`].
pub fn decode_color(color: [u8; 4]) -> Vec4 {
    let v = Vec4::from_array(color.map(|c| c as f32 / 255.0));
    Vec4::new(v.x * v.x, v.y * v.y, v.z * v.z, v.w)
}

fn octahedral(n: Vec3) -> Vec2 {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-20);
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    if n.z >= 0.0 {
        Vec2::new(n.x, n.y)
    } else {
        Vec2::new((1.0 - n.y.abs()) * sign(n.x), (1.0 - n.x.abs()) * sign(n.y))
    }
}

fn unoctahedral(e: Vec2) -> Vec3 {
    let z = 1.0 - e.x.abs() - e.y.abs();
    let t = (-z).max(0.0);
    let x = e.x + if e.x >= 0.0 { -t } else { t };
    let y = e.y + if e.y >= 0.0 { -t } else { t };
    Vec3::new(x, y, z).normalize_or(Vec3::Z)
}

/// Octahedral encoding of a unit vector with `bits` per component: `x | y << bits`, each the
/// signed value `round(c * m)` (`m = 2^(bits-1) - 1`) offset by `m`. Of the four grid points
/// around the exact projection, the one that decodes closest to `n` is chosen.
pub fn encode_octahedral(n: Vec3, bits: u32) -> u32 {
    debug_assert!((2..=16).contains(&bits));
    let m = ((1u32 << (bits - 1)) - 1) as f32;
    let n = n.normalize_or(Vec3::Z);
    let e = (octahedral(n) * m).clamp(Vec2::splat(-m), Vec2::splat(m));
    let mut best = (Vec2::ZERO, f32::NEG_INFINITY);
    for candidate in [e.floor(), Vec2::new(e.x.floor(), e.y.ceil()), Vec2::new(e.x.ceil(), e.y.floor()), e.ceil()] {
        let similarity = unoctahedral(candidate / m).dot(n);
        if similarity > best.1 {
            best = (candidate, similarity);
        }
    }
    let q = (best.0 + m).as_uvec2();
    q.x | (q.y << bits)
}

/// Inverse of [`encode_octahedral`].
pub fn decode_octahedral(packed: u32, bits: u32) -> Vec3 {
    let m = ((1u32 << (bits - 1)) - 1) as f32;
    let mask = (1u32 << bits) - 1;
    let e = Vec2::new((packed & mask) as f32, ((packed >> bits) & mask) as f32);
    unoctahedral((e - m) / m)
}

/// Appends fields of up to 32 bits to a word stream, least significant bit first.
struct BitWriter<'a> {
    words: &'a mut Vec<u32>,
    start: usize,
    bit: usize,
}

impl<'a> BitWriter<'a> {
    fn new(words: &'a mut Vec<u32>) -> Self {
        let start = words.len();
        Self { words, start, bit: 0 }
    }

    fn write(&mut self, value: u32, bits: u32) {
        debug_assert!(bits == 32 || value >> bits == 0, "{value} does not fit in {bits} bits");
        if bits == 0 {
            return;
        }
        let (word, shift) = (self.start + self.bit / 32, (self.bit % 32) as u32);
        self.words.resize(self.start + (self.bit + bits as usize).div_ceil(32), 0);
        self.words[word] |= value << shift;
        if shift + bits > 32 {
            self.words[word + 1] |= value >> (32 - shift);
        }
        self.bit += bits as usize;
    }
}

/// Reads `count` (≤ 32) bits at bit `bit` of `words`.
pub fn read_bits(words: &[u32], bit: u32, count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    let (word, shift) = ((bit / 32) as usize, bit % 32);
    let mut value = words[word] >> shift;
    if shift + count > 32 {
        value |= words[word + 1] << (32 - shift);
    }
    if count == 32 {
        value
    } else {
        value & ((1 << count) - 1)
    }
}

/// The decoded header of an encoded cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterEncoding {
    pub position_min: IVec3,
    pub position_bits: UVec3,
    pub uv_min: IVec2,
    pub uv_bits: UVec2,
    pub normal_bits: u32,
    pub position_exponent: i32,
    pub uv_exponent: i32,
    pub attributes: VertexAttributes,
    pub uv1_min: IVec2,
    pub uv1_bits: UVec2,
    /// [`encode_color`] values: per-channel minimum and bits of the offsets from it.
    pub color_min: [u8; 4],
    pub color_bits: UVec4,
}

impl ClusterEncoding {
    /// The smallest encoding of `vertices` on the grids of `quantization`.
    fn fit(quantization: &Quantization, vertices: &[QuantizedVertex]) -> Self {
        let bits = |range: i64| {
            let bits = bits_for(u32::try_from(range).expect("grid coordinates fit in 24 bits"));
            assert!(bits <= MAX_FIELD_BITS);
            bits
        };
        let range2 = |f: fn(&QuantizedVertex) -> IVec2| {
            let min = vertices.iter().map(f).reduce(IVec2::min).unwrap_or_default();
            let max = vertices.iter().map(f).reduce(IVec2::max).unwrap_or_default();
            (min, UVec2::from_array(std::array::from_fn(|i| bits(max[i] as i64 - min[i] as i64))))
        };
        let position_min = vertices.iter().map(|v| v.position).reduce(IVec3::min).unwrap_or_default();
        let position_max = vertices.iter().map(|v| v.position).reduce(IVec3::max).unwrap_or_default();
        let (uv_min, uv_bits) = range2(|v| v.uv);
        let attributes = quantization.attributes;
        let (uv1_min, uv1_bits) = if attributes.uv1 { range2(|v| v.uv1) } else { (IVec2::ZERO, UVec2::ZERO) };
        let (mut color_min, mut color_bits) = ([0u8; 4], UVec4::ZERO);
        if attributes.color {
            for c in 0..4 {
                color_min[c] = vertices.iter().map(|v| v.color[c]).min().unwrap_or(0);
                let max = vertices.iter().map(|v| v.color[c]).max().unwrap_or(0);
                color_bits[c] = bits_for((max - color_min[c]) as u32);
            }
        }
        Self {
            position_min,
            position_bits: UVec3::from_array(std::array::from_fn(|i| bits(position_max[i] as i64 - position_min[i] as i64))),
            uv_min,
            uv_bits,
            normal_bits: quantization.normal_bits,
            position_exponent: quantization.position_exponent,
            uv_exponent: quantization.uv_exponent,
            attributes,
            uv1_min,
            uv1_bits,
            color_min,
            color_bits,
        }
    }

    /// Words of the header (the vertex stream follows).
    pub fn header_words(&self) -> u32 {
        CLUSTER_HEADER_WORDS + if self.attributes.any() { CLUSTER_EXTENSION_WORDS } else { 0 }
    }

    fn to_words(self) -> Vec<u32> {
        let p = self.position_bits;
        let extended = if self.attributes.any() { EXTENDED_BIT } else { 0 };
        let widths = p.x | (p.y << 5) | (p.z << 10) | (self.uv_bits.x << 15) | (self.uv_bits.y << 20) | (self.normal_bits << 25) | extended;
        let exponents = (self.position_exponent as i8 as u8 as u32) | ((self.uv_exponent as i8 as u8 as u32) << 8);
        let mut words = vec![
            self.position_min.x as u32,
            self.position_min.y as u32,
            self.position_min.z as u32,
            self.uv_min.x as u32,
            self.uv_min.y as u32,
            widths,
            exponents,
        ];
        if self.attributes.any() {
            let c = self.color_bits;
            let flags = ((self.attributes.uv1 as u32) << 26) | ((self.attributes.color as u32) << 27);
            let flags = flags | if self.attributes.skin { SKIN_BIT } else { 0 };
            let widths = self.uv1_bits.x | (self.uv1_bits.y << 5) | (c.x << 10) | (c.y << 14) | (c.z << 18) | (c.w << 22) | flags;
            words.extend([widths, self.uv1_min.x as u32, self.uv1_min.y as u32, u32::from_le_bytes(self.color_min)]);
        }
        words
    }

    /// Parses the header at the start of `words`.
    pub fn from_words(words: &[u32]) -> Self {
        let field = |shift: u32| (words[5] >> shift) & 31;
        let mut encoding = Self {
            position_min: IVec3::new(words[0] as i32, words[1] as i32, words[2] as i32),
            uv_min: IVec2::new(words[3] as i32, words[4] as i32),
            position_bits: UVec3::new(field(0), field(5), field(10)),
            uv_bits: UVec2::new(field(15), field(20)),
            normal_bits: field(25),
            position_exponent: (words[6] & 0xff) as u8 as i8 as i32,
            uv_exponent: ((words[6] >> 8) & 0xff) as u8 as i8 as i32,
            attributes: VertexAttributes::default(),
            uv1_min: IVec2::ZERO,
            uv1_bits: UVec2::ZERO,
            color_min: [0; 4],
            color_bits: UVec4::ZERO,
        };
        if words[5] & EXTENDED_BIT != 0 {
            let e = words[7];
            encoding.attributes = VertexAttributes { uv1: (e >> 26) & 1 != 0, color: (e >> 27) & 1 != 0, skin: e & SKIN_BIT != 0 };
            encoding.uv1_bits = UVec2::new(e & 31, (e >> 5) & 31);
            encoding.color_bits = UVec4::new((e >> 10) & 15, (e >> 14) & 15, (e >> 18) & 15, (e >> 22) & 15);
            encoding.uv1_min = IVec2::new(words[8] as i32, words[9] as i32);
            encoding.color_min = words[10].to_le_bytes();
        }
        encoding
    }

    fn position_stride(&self) -> u32 {
        self.position_bits.element_sum()
    }

    fn attribute_stride(&self) -> u32 {
        2 * self.normal_bits + self.uv_bits.element_sum() + self.uv1_bits.element_sum() + self.color_bits.element_sum()
    }

    /// Bits of the vertex stream of a cluster with `vertex_count` vertices.
    pub fn vertex_stream_bits(&self, vertex_count: u32) -> u32 {
        vertex_count * (self.position_stride() + self.attribute_stride())
    }

    /// Position of vertex `index`; `stream` is the vertex bit stream (after the header).
    pub fn position(&self, stream: &[u32], index: u32) -> Vec3 {
        let mut bit = index * self.position_stride();
        let mut offset = [0u32; 3];
        for (axis, value) in offset.iter_mut().enumerate() {
            *value = read_bits(stream, bit, self.position_bits[axis]);
            bit += self.position_bits[axis];
        }
        (self.position_min + UVec3::from(offset).as_ivec3()).as_vec3() * pow2(self.position_exponent)
    }

    fn attribute_bit(&self, vertex_count: u32, index: u32) -> u32 {
        vertex_count * self.position_stride() + index * self.attribute_stride()
    }

    pub fn normal(&self, stream: &[u32], vertex_count: u32, index: u32) -> Vec3 {
        decode_octahedral(read_bits(stream, self.attribute_bit(vertex_count, index), 2 * self.normal_bits), self.normal_bits)
    }

    pub fn uv(&self, stream: &[u32], vertex_count: u32, index: u32) -> Vec2 {
        let bit = self.attribute_bit(vertex_count, index) + 2 * self.normal_bits;
        let offset = UVec2::new(read_bits(stream, bit, self.uv_bits.x), read_bits(stream, bit + self.uv_bits.x, self.uv_bits.y));
        (self.uv_min + offset.as_ivec2()).as_vec2() * pow2(self.uv_exponent)
    }

    /// Second UV of vertex `index`; the first UV where the mesh has no second set.
    pub fn uv1(&self, stream: &[u32], vertex_count: u32, index: u32) -> Vec2 {
        if !self.attributes.uv1 {
            return self.uv(stream, vertex_count, index);
        }
        let bit = self.attribute_bit(vertex_count, index) + 2 * self.normal_bits + self.uv_bits.element_sum();
        let offset = UVec2::new(read_bits(stream, bit, self.uv1_bits.x), read_bits(stream, bit + self.uv1_bits.x, self.uv1_bits.y));
        (self.uv1_min + offset.as_ivec2()).as_vec2() * pow2(self.uv_exponent)
    }

    /// Linear RGBA colour of vertex `index`; white where the mesh has no vertex colours.
    pub fn color(&self, stream: &[u32], vertex_count: u32, index: u32) -> Vec4 {
        if !self.attributes.color {
            return Vec4::ONE;
        }
        let mut bit = self.attribute_bit(vertex_count, index) + 2 * self.normal_bits + self.uv_bits.element_sum() + self.uv1_bits.element_sum();
        let mut color = self.color_min;
        for (c, value) in color.iter_mut().enumerate() {
            *value += read_bits(stream, bit, self.color_bits[c]) as u8;
            bit += self.color_bits[c];
        }
        decode_color(color)
    }

    /// The skin section's descriptor; `None` for clusters of meshes without skin.
    pub fn skin_encoding(&self, stream: &[u32], vertex_count: u32) -> Option<SkinEncoding> {
        self.attributes.skin.then(|| SkinEncoding::from_word(stream[self.vertex_stream_bits(vertex_count).div_ceil(32) as usize]))
    }

    /// Skin weights of vertex `index`; `None` for clusters of meshes without skin.
    pub fn skin(&self, stream: &[u32], vertex_count: u32, index: u32) -> Option<SkinWeights> {
        let skin = self.skin_encoding(stream, vertex_count)?;
        let section = self.vertex_stream_bits(vertex_count).div_ceil(32) as usize + 1;
        Some(skin.read(&stream[section..], index))
    }
}

/// Appends the encoding of one cluster (header, vertex stream, triangle stream) to `out`.
/// `triangles` index `vertices`. Returns the word offset of the triangle stream relative to the
/// start of the cluster.
pub fn encode_cluster(quantization: &Quantization, vertices: &[QuantizedVertex], triangles: &[[u32; 3]], out: &mut Vec<u32>) -> u32 {
    let start = out.len();
    let encoding = ClusterEncoding::fit(quantization, vertices);
    out.extend_from_slice(&encoding.to_words());

    let mut stream = BitWriter::new(out);
    for v in vertices {
        let offset = (v.position - encoding.position_min).as_uvec3();
        for axis in 0..3 {
            stream.write(offset[axis], encoding.position_bits[axis]);
        }
    }
    for v in vertices {
        let offset = (v.uv - encoding.uv_min).as_uvec2();
        stream.write(v.normal, 2 * encoding.normal_bits);
        stream.write(offset.x, encoding.uv_bits.x);
        stream.write(offset.y, encoding.uv_bits.y);
        if encoding.attributes.uv1 {
            let offset = (v.uv1 - encoding.uv1_min).as_uvec2();
            stream.write(offset.x, encoding.uv1_bits.x);
            stream.write(offset.y, encoding.uv1_bits.y);
        }
        if encoding.attributes.color {
            for c in 0..4 {
                stream.write((v.color[c] - encoding.color_min[c]) as u32, encoding.color_bits[c]);
            }
        }
    }

    if encoding.attributes.skin {
        let skin = SkinEncoding::fit(vertices);
        out.push(skin.to_word());
        let mut stream = BitWriter::new(out);
        for v in vertices {
            skin.write(&mut stream, &v.skin);
        }
    }

    let triangle_offset = (out.len() - start) as u32;
    let bits = index_bits(vertices.len() as u32);
    let mut stream = BitWriter::new(out);
    for t in triangles {
        stream.write(t[0] | (t[1] << bits) | (t[2] << (2 * bits)), 3 * bits);
    }
    triangle_offset
}

/// Corner indices of triangle `triangle` from a triangle bit stream.
pub fn decode_triangle(stream: &[u32], vertex_count: u32, triangle: u32) -> [u32; 3] {
    let bits = index_bits(vertex_count);
    let packed = read_bits(stream, triangle * 3 * bits, 3 * bits);
    let mask = (1 << bits) - 1;
    [packed & mask, (packed >> bits) & mask, (packed >> (2 * bits)) & mask]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 8) as f32 / (1 << 24) as f32
    }

    #[test]
    fn bit_streams_round_trip() {
        let mut words = Vec::new();
        let fields: Vec<(u32, u32)> = (0..200u32).map(|i| (i % 33, i.wrapping_mul(2654435761))).collect();
        let mut writer = BitWriter::new(&mut words);
        for &(bits, value) in &fields {
            let value = if bits == 32 { value } else { value & ((1u32 << bits).wrapping_sub(1)) };
            writer.write(value, bits);
        }
        let mut bit = 0;
        for &(bits, value) in &fields {
            let value = if bits == 32 { value } else { value & ((1u32 << bits).wrapping_sub(1)) };
            assert_eq!(read_bits(&words, bit, bits), value);
            bit += bits;
        }
        assert_eq!(words.len(), (bit as usize).div_ceil(32));
    }

    #[test]
    fn octahedral_normals_are_accurate() {
        let mut seed = 7;
        let mut worst = 1.0f32;
        for i in 0..20000 {
            let n = match i {
                0 => Vec3::X,
                1 => Vec3::NEG_Z,
                2 => Vec3::new(0.3, -0.5, -0.8),
                _ => Vec3::new(lcg(&mut seed) * 2.0 - 1.0, lcg(&mut seed) * 2.0 - 1.0, lcg(&mut seed) * 2.0 - 1.0),
            }
            .normalize_or(Vec3::Y);
            worst = worst.min(decode_octahedral(encode_octahedral(n, NORMAL_BITS), NORMAL_BITS).dot(n));
        }
        let degrees = worst.min(1.0).acos().to_degrees();
        assert!(degrees < 0.17, "octahedral normals with {NORMAL_BITS} bits: max error {degrees}°");
        // Axis-aligned normals (flat walls, floors) are exact.
        for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Y, Vec3::NEG_Z] {
            assert_eq!(decode_octahedral(encode_octahedral(axis, NORMAL_BITS), NORMAL_BITS), axis);
        }
    }

    #[test]
    fn automatic_grids_follow_edge_length_and_magnitude() {
        // A 1 m rock with 5 mm edges: 1/64 edge, rounded down to a power of two.
        let q = Quantization::new(None, 0.005, 1.0, 1.0);
        assert_eq!(q.position_exponent, -14);
        assert_eq!(q.uv_exponent, UV_EXPONENT);
        // Large edges are capped at ~1 mm; huge coordinates coarsen the grid to stay in 24 bits.
        assert_eq!(Quantization::new(None, 4.0, 2000.0, 0.0).position_exponent, MAX_POSITION_EXPONENT);
        let far = Quantization::new(None, 0.01, 1.0e5, 1.0e3);
        assert!(far.position_step() * MAX_GRID >= 1.0e5 && far.position_step() * MAX_GRID / 2.0 < 1.0e5);
        assert!(far.uv_step() * MAX_GRID >= 1.0e3);
        assert_eq!(Quantization::new(Some(8), 0.005, 1.0, 1.0).position_exponent, -8);
    }

    #[test]
    fn clusters_round_trip_within_half_a_step() {
        let mut seed = 99;
        let q = Quantization::new(None, 0.01, 3.0, 4.0);
        let source: Vec<(Vec3, Vec3, Vec2)> = (0..100)
            .map(|_| {
                let p = Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)) * 0.4 + Vec3::new(2.0, -1.0, 0.5);
                let n = Vec3::new(lcg(&mut seed) - 0.5, lcg(&mut seed) - 0.5, lcg(&mut seed) - 0.5).normalize();
                (p, n, Vec2::new(lcg(&mut seed) * 4.0 - 2.0, lcg(&mut seed)))
            })
            .collect();
        let vertices: Vec<QuantizedVertex> = source.iter().map(|&(p, n, uv)| q.vertex(p, n, uv)).collect();
        let triangles: Vec<[u32; 3]> = (0..128).map(|i| [i % 100, (i * 7 + 1) % 100, (i * 13 + 2) % 100]).collect();
        let mut words = vec![0xdead_beef]; // the encoder appends
        let triangle_offset = encode_cluster(&q, &vertices, &triangles, &mut words);
        let cluster = &words[1..];
        let encoding = ClusterEncoding::from_words(cluster);
        let stream = &cluster[CLUSTER_HEADER_WORDS as usize..];
        assert_eq!(triangle_offset, CLUSTER_HEADER_WORDS + encoding.vertex_stream_bits(100).div_ceil(32));
        for (i, &(p, n, uv)) in source.iter().enumerate() {
            let i = i as u32;
            assert!((encoding.position(stream, i) - p).abs().max_element() <= q.position_step() * 0.5);
            assert!(encoding.normal(stream, 100, i).dot(n) > 0.99999);
            let uv_error = (encoding.uv(stream, 100, i) - uv).abs().max_element();
            assert!(uv_error <= q.uv_step() * 0.5);
            assert!(uv_error * 2048.0 <= 1.0 / 32.0, "uv error {} texels of a 2k texture", uv_error * 2048.0);
        }
        for (t, tri) in triangles.iter().enumerate() {
            assert_eq!(decode_triangle(&cluster[triangle_offset as usize..], 100, t as u32), *tri);
        }
        // Even random vertices filling the cluster's whole range shrink to about half of 24-byte
        // vertices and 32-bit triangles (real meshes do better: see the dag tests).
        let uncompressed = 100 * 6 + 128;
        assert!(cluster.len() * 100 < uncompressed * 55, "{} words vs {uncompressed}", cluster.len());
    }

    #[test]
    fn optional_attributes_round_trip_and_cost_nothing_when_absent() {
        let mut seed = 5;
        let q = Quantization::new(None, 0.01, 2.0, 8.0);
        let source: Vec<(Vec3, Vec2, Vec2, Vec4)> = (0..60)
            .map(|_| {
                let p = Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed));
                let uv1 = Vec2::new(lcg(&mut seed) * 8.0 - 4.0, lcg(&mut seed));
                let color = Vec4::new(lcg(&mut seed), 0.2 + lcg(&mut seed) * 0.05, 0.5, 1.0);
                (p, Vec2::new(lcg(&mut seed), lcg(&mut seed)), uv1, color)
            })
            .collect();
        let triangles: Vec<[u32; 3]> = (0..40).map(|i| [i, i + 1, i + 2]).collect();
        let encode = |q: &Quantization| {
            let vertices: Vec<QuantizedVertex> =
                source.iter().map(|&(p, uv, uv1, color)| q.vertex(p, Vec3::Y, uv).with_uv1(q, uv1).with_color(color)).collect();
            let mut words = Vec::new();
            let triangle_offset = encode_cluster(q, &vertices, &triangles, &mut words);
            (words, triangle_offset)
        };

        // Without optional attributes the cluster is the plain format (the extra inputs are ignored).
        let (plain, _) = encode(&q);
        assert_eq!(ClusterEncoding::from_words(&plain).header_words(), CLUSTER_HEADER_WORDS);
        assert_eq!(plain[5] & EXTENDED_BIT, 0);

        let full = q.with_attributes(VertexAttributes { uv1: true, color: true, skin: false });
        let (words, triangle_offset) = encode(&full);
        let encoding = ClusterEncoding::from_words(&words);
        assert_eq!(encoding.attributes, full.attributes);
        assert_eq!(encoding.header_words(), CLUSTER_HEADER_WORDS + CLUSTER_EXTENSION_WORDS);
        let stream = &words[encoding.header_words() as usize..];
        for (i, &(p, uv, uv1, color)) in source.iter().enumerate() {
            let i = i as u32;
            assert!((encoding.position(stream, i) - p).abs().max_element() <= q.position_step() * 0.5);
            assert!((encoding.uv(stream, 60, i) - uv).abs().max_element() <= q.uv_step() * 0.5);
            assert!((encoding.uv1(stream, 60, i) - uv1).abs().max_element() <= q.uv_step() * 0.5);
            // 8 bits on a square-root curve: within half a code of the encoded value.
            let decoded = encoding.color(stream, 60, i);
            assert!((decoded.truncate().powf(0.5) - color.truncate().powf(0.5)).abs().max_element() <= 0.5 / 255.0 + 1e-6, "{decoded} vs {color}");
            assert_eq!(decoded.w, 1.0);
        }
        // Constant channels (blue, alpha) take no bits per vertex.
        assert_eq!((encoding.color_bits.z, encoding.color_bits.w), (0, 0));
        assert!(encoding.color_bits.y <= 4, "a narrow range needs few bits: {}", encoding.color_bits.y);
        for (t, tri) in triangles.iter().enumerate() {
            assert_eq!(decode_triangle(&words[triangle_offset as usize..], 60, t as u32), *tri);
        }

        // Absent attributes decode to their defaults.
        let only_uv1 = q.with_attributes(VertexAttributes { uv1: true, color: false, skin: false });
        let (words, _) = encode(&only_uv1);
        let encoding = ClusterEncoding::from_words(&words);
        assert_eq!(encoding.color(&words[encoding.header_words() as usize..], 60, 3), Vec4::ONE);
        let stream = &plain[CLUSTER_HEADER_WORDS as usize..];
        let plain_encoding = ClusterEncoding::from_words(&plain);
        assert_eq!(plain_encoding.uv1(stream, 60, 7), plain_encoding.uv(stream, 60, 7));
    }

    #[test]
    fn skin_weights_round_trip_with_just_enough_bits() {
        let mut seed = 11;
        let q = Quantization::new(None, 0.01, 2.0, 1.0);
        let skins: Vec<SkinWeights> = (0..90)
            .map(|i| {
                let joints = [40 + (i % 9) as u16, 41 + (i % 5) as u16, 47, 50];
                let weights = [lcg(&mut seed), lcg(&mut seed) * 0.5, if i % 3 == 0 { 0.2 } else { 0.0 }, if i % 10 == 0 { 0.1 } else { 0.0 }];
                SkinWeights::new(joints, weights)
            })
            .collect();
        let triangles: Vec<[u32; 3]> = (0..60).map(|i| [i, i + 1, i + 2]).collect();
        let encode = |q: &Quantization, skins: &[SkinWeights]| {
            let vertices: Vec<QuantizedVertex> = skins.iter().enumerate().map(|(i, &s)| q.vertex(Vec3::splat(i as f32 * 0.01), Vec3::Y, Vec2::ZERO).with_skin(s)).collect();
            let mut words = Vec::new();
            let triangle_offset = encode_cluster(q, &vertices, &triangles, &mut words);
            (words, triangle_offset)
        };
        let skinned = q.with_attributes(VertexAttributes { skin: true, ..Default::default() });
        let (words, triangle_offset) = encode(&skinned, &skins);
        let encoding = ClusterEncoding::from_words(&words);
        assert!(encoding.attributes.skin && encoding.header_words() == CLUSTER_HEADER_WORDS + CLUSTER_EXTENSION_WORDS);
        let stream = &words[encoding.header_words() as usize..];
        let descriptor = encoding.skin_encoding(stream, 90).unwrap();
        // Joints 40..=50 need 4 bits; weights after the heaviest stay below 128.
        assert_eq!((descriptor.joint_min, descriptor.joint_bits, descriptor.influences), (40, 4, 4));
        assert!(descriptor.weight_bits <= 7);
        for (i, skin) in skins.iter().enumerate() {
            assert_eq!(encoding.skin(stream, 90, i as u32), Some(*skin));
            assert!((encoding.position(stream, i as u32) - Vec3::splat(i as f32 * 0.01)).abs().max_element() <= q.position_step());
        }
        for (t, tri) in triangles.iter().enumerate() {
            assert_eq!(decode_triangle(&words[triangle_offset as usize..], 90, t as u32), *tri);
        }
        // Rigid clusters store one joint index per vertex and no weights.
        let rigid: Vec<SkinWeights> = (0..90).map(|i| SkinWeights::rigid(7 + (i % 2) as u16)).collect();
        let (words, _) = encode(&skinned, &rigid);
        let encoding = ClusterEncoding::from_words(&words);
        let stream = &words[encoding.header_words() as usize..];
        let descriptor = encoding.skin_encoding(stream, 90).unwrap();
        assert_eq!((descriptor.influences, descriptor.stride()), (1, 1));
        assert_eq!(encoding.skin(stream, 90, 3), Some(SkinWeights::rigid(8)));
        // Without the attribute nothing is stored (the inputs are ignored).
        let (plain, _) = encode(&q, &skins);
        assert_eq!(ClusterEncoding::from_words(&plain).header_words(), CLUSTER_HEADER_WORDS);
        assert_eq!(ClusterEncoding::from_words(&plain).skin(&plain[CLUSTER_HEADER_WORDS as usize..], 90, 0), None);
    }

    #[test]
    fn vertex_colors_keep_dark_values() {
        assert_eq!(encode_color(Vec4::ONE), [255; 4]);
        assert_eq!(decode_color(encode_color(Vec4::ZERO)), Vec4::ZERO);
        // A linear 0.0015 survives within 5 %; plain linear 8 bits would round it to zero.
        let dark = decode_color(encode_color(Vec4::new(0.0015, 0.25, 0.5, 0.5)));
        assert!((dark.x - 0.0015).abs() < 0.000075 && (dark.y - 0.25).abs() < 0.004 && (dark.w - 0.5).abs() < 0.003, "{dark}");
    }

    #[test]
    fn index_bits_cover_the_vertex_count() {
        assert_eq!(index_bits(3), 2);
        assert_eq!(index_bits(4), 2);
        assert_eq!(index_bits(5), 3);
        assert_eq!(index_bits(128), 7);
        assert_eq!(index_bits(129), 8);
        assert_eq!(index_bits(256), 8);
    }
}
