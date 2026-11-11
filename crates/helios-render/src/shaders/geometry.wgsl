// Streamed cluster geometry access. The including shader must declare
//   clusters: array<Cluster>, page_table: array<u32>, geometry: array<u32>
// (read-only storage).
//
// Clusters are compressed (helios_geometry::encoding, which this mirrors). At the cluster's
// vertex_offset in its page:
//   [0..3) position minimum, [3..5) uv minimum (i32 units of the mesh's grids)
//   [5]    bit widths: position x | y << 5 | z << 10, uv u << 15 | v << 20, normal << 25;
//          bit 30: extended (imported meshes with a second UV set or vertex colours, plant data);
//          bit 31: positions carry wind weights (plant data)
//   [6]    grid exponents: position (i8) | uv (i8) << 8
//   extended only: [7] uv1 widths u | v << 5, colour widths r << 10 | g << 14 | b << 18 |
//          a << 22, has uv1 << 26, has colour << 27, has skin << 28, has plant data << 29;
//          [8..10) uv1 minimum; [10] colour minimum (RGBA8: rgb on a square-root curve, alpha
//          linear)
//   [7..) or [11..)  vertex bit stream: all position offsets (x, y, z, then the wind weight of
//          plant data, PLANT_CHANNEL_BITS), then all attributes (octahedral normal x | y << normal
//          bits, u offset, v offset, then the optional uv1 u, v and colour r, g, b, a offsets and
//          the plant data's occlusion, PLANT_CHANNEL_BITS)
// At triangle_offset: 3 corner indices per triangle, index_bits(vertex count) bits each.
// A coordinate decodes as f32(minimum + offset) * 2^exponent, exact in f32, so a vertex shared
// by two clusters (or LOD levels) lands on the same bits in both: no cracks.

const CLUSTER_HEADER_WORDS: u32 = 7u;
const CLUSTER_EXTENDED: u32 = 0x40000000u;
const CLUSTER_EXTENSION_WORDS: u32 = 4u;
const CLUSTER_HAS_UV1: u32 = 0x4000000u;
const CLUSTER_HAS_COLOR: u32 = 0x8000000u;
const CLUSTER_HAS_PLANT: u32 = 0x20000000u;
// Widths word bit 31 (plant data): every vertex's position offsets are followed by its wind weight,
// so the rasterisers read it from the bits they already fetch.
const CLUSTER_WIND_WEIGHTS: u32 = 0x80000000u;
const PLANT_CHANNEL_BITS: u32 = 5u;

// `count` (<= 32) bits at bit `bit` of the stream starting at pool word `word`.
fn read_bits(word: u32, bit: u32, count: u32) -> u32 {
    let w = word + (bit >> 5u);
    let shift = bit & 31u;
    let lo = geometry[w] >> shift;
    // Shift amounts are taken modulo 32 in WGSL: shift == 0 must not add the next word.
    let hi = select(geometry[w + 1u] << (32u - shift), 0u, shift == 0u);
    let mask = select((1u << count) - 1u, 0xffffffffu, count >= 32u);
    return (lo | hi) & mask;
}

// 2^e for the signed 8-bit exponent at bit `offset` of a header word, built bit-exactly.
fn grid_step(word: u32, offset: u32) -> f32 {
    let e = extractBits(bitcast<i32>(word), offset, 8u);
    return bitcast<f32>(u32(e + 127) << 23u);
}

fn decode_octahedral(packed: u32, bits: u32) -> vec3<f32> {
    let mask = (1u << bits) - 1u;
    let m = f32((1u << (bits - 1u)) - 1u);
    let e = (vec2<f32>(f32(packed & mask), f32((packed >> bits) & mask)) - m) / m;
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// Pool word offset of a cluster's page payload (the page must be resident).
fn cluster_base(c: Cluster) -> u32 {
    return page_table[c.page];
}

fn cluster_triangle(c: Cluster, base: u32, triangle: u32) -> vec3<u32> {
    let bits = firstLeadingBit(max(cluster_vertex_count(c), 2u) - 1u) + 1u;
    let packed = read_bits(base + c.triangle_offset, triangle * 3u * bits, 3u * bits);
    let mask = (1u << bits) - 1u;
    return vec3<u32>(packed & mask, (packed >> bits) & mask, (packed >> (2u * bits)) & mask);
}

// First pool word of a cluster's vertex stream (behind the header and its extension).
fn vertex_stream(header: u32, widths: u32) -> u32 {
    return header + CLUSTER_HEADER_WORDS + select(0u, CLUSTER_EXTENSION_WORDS, (widths & CLUSTER_EXTENDED) != 0u);
}

// True for clusters of meshes with optional attributes (second UV set, vertex colours).
fn cluster_is_extended(c: Cluster, base: u32) -> bool {
    return (geometry[base + c.vertex_offset + 5u] & CLUSTER_EXTENDED) != 0u;
}

// The header extension word of an extended cluster (0 otherwise: no optional attributes).
fn cluster_extension(header: u32, widths: u32) -> u32 {
    if (widths & CLUSTER_EXTENDED) == 0u {
        return 0u;
    }
    return geometry[header + 7u];
}

fn vertex_position(c: Cluster, base: u32, index: u32) -> vec3<f32> {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let bits = vec3<u32>(widths & 31u, (widths >> 5u) & 31u, (widths >> 10u) & 31u);
    let stream = vertex_stream(header, widths);
    let bit = index * position_stride(widths);
    let offset = vec3<u32>(read_bits(stream, bit, bits.x), read_bits(stream, bit + bits.x, bits.y), read_bits(stream, bit + bits.x + bits.y, bits.z));
    let minimum = vec3<i32>(bitcast<i32>(geometry[header]), bitcast<i32>(geometry[header + 1u]), bitcast<i32>(geometry[header + 2u]));
    return vec3<f32>(minimum + vec3<i32>(offset)) * grid_step(geometry[header + 6u], 0u);
}

// Bits of a vertex's position: offsets x, y, z and, with plant data, its wind weight.
fn position_stride(widths: u32) -> u32 {
    return (widths & 31u) + ((widths >> 5u) & 31u) + ((widths >> 10u) & 31u) + select(0u, PLANT_CHANNEL_BITS, (widths & CLUSTER_WIND_WEIGHTS) != 0u);
}

// Bits of the plant data among a vertex's attributes (its occlusion), from the extension word.
fn plant_stride(extension: u32) -> u32 {
    return select(0u, PLANT_CHANNEL_BITS, (extension & CLUSTER_HAS_PLANT) != 0u);
}

// Bits of the optional attributes (uv1, colour, plant data) per vertex, from the extension word.
fn extension_stride(extension: u32) -> u32 {
    return (extension & 31u) + ((extension >> 5u) & 31u) + ((extension >> 10u) & 15u) + ((extension >> 14u) & 15u)
        + ((extension >> 18u) & 15u) + ((extension >> 22u) & 15u) + plant_stride(extension);
}

// Bit of vertex `index`'s attributes (normal, uv, then the optional ones) in the vertex stream.
fn attribute_bit(c: Cluster, widths: u32, extension: u32, index: u32) -> u32 {
    let attribute_stride = (2u * ((widths >> 25u) & 31u)) + ((widths >> 15u) & 31u) + ((widths >> 20u) & 31u) + extension_stride(extension);
    return (cluster_vertex_count(c) * position_stride(widths)) + (index * attribute_stride);
}

fn vertex_normal(c: Cluster, base: u32, index: u32) -> vec3<f32> {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let normal_bits = (widths >> 25u) & 31u;
    let bit = attribute_bit(c, widths, cluster_extension(header, widths), index);
    let packed = read_bits(vertex_stream(header, widths), bit, 2u * normal_bits);
    return decode_octahedral(packed, normal_bits);
}

fn vertex_uv(c: Cluster, base: u32, index: u32) -> vec2<f32> {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let bits = vec2<u32>((widths >> 15u) & 31u, (widths >> 20u) & 31u);
    let stream = vertex_stream(header, widths);
    let bit = attribute_bit(c, widths, cluster_extension(header, widths), index) + (2u * ((widths >> 25u) & 31u));
    let offset = vec2<u32>(read_bits(stream, bit, bits.x), read_bits(stream, bit + bits.x, bits.y));
    let minimum = vec2<i32>(bitcast<i32>(geometry[header + 3u]), bitcast<i32>(geometry[header + 4u]));
    return vec2<f32>(minimum + vec2<i32>(offset)) * grid_step(geometry[header + 6u], 8u);
}

// The second UV of vertex `index`, or `uv0` (its first UV) if the mesh has no second set.
fn vertex_uv1(c: Cluster, base: u32, index: u32, uv0: vec2<f32>) -> vec2<f32> {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let extension = cluster_extension(header, widths);
    if (extension & CLUSTER_HAS_UV1) == 0u {
        return uv0;
    }
    let bits = vec2<u32>(extension & 31u, (extension >> 5u) & 31u);
    let stream = vertex_stream(header, widths);
    let bit = attribute_bit(c, widths, extension, index) + (2u * ((widths >> 25u) & 31u)) + ((widths >> 15u) & 31u) + ((widths >> 20u) & 31u);
    let offset = vec2<u32>(read_bits(stream, bit, bits.x), read_bits(stream, bit + bits.x, bits.y));
    let minimum = vec2<i32>(bitcast<i32>(geometry[header + 8u]), bitcast<i32>(geometry[header + 9u]));
    return vec2<f32>(minimum + vec2<i32>(offset)) * grid_step(geometry[header + 6u], 8u);
}

// Linear RGBA vertex colour (white if the mesh has none).
fn vertex_color(c: Cluster, base: u32, index: u32) -> vec4<f32> {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let extension = cluster_extension(header, widths);
    if (extension & CLUSTER_HAS_COLOR) == 0u {
        return vec4<f32>(1.0);
    }
    let bits = vec4<u32>((extension >> 10u) & 15u, (extension >> 14u) & 15u, (extension >> 18u) & 15u, (extension >> 22u) & 15u);
    let bit = attribute_bit(c, widths, extension, index) + (2u * ((widths >> 25u) & 31u)) + ((widths >> 15u) & 31u) + ((widths >> 20u) & 31u)
        + (extension & 31u) + ((extension >> 5u) & 31u);
    // At most 4 x 8 bits: one read.
    let packed = read_bits(vertex_stream(header, widths), bit, bits.x + bits.y + bits.z + bits.w);
    let offset = vec4<u32>(
        extractBits(packed, 0u, bits.x),
        extractBits(packed, bits.x, bits.y),
        extractBits(packed, bits.x + bits.y, bits.z),
        extractBits(packed, bits.x + bits.y + bits.z, bits.w),
    );
    let code = (unpack4x8unorm(geometry[header + 10u]) * 255.0 + vec4<f32>(offset)) / 255.0;
    return vec4<f32>(code.rgb * code.rgb, code.a);
}

// Plant data of vertex `index` (helios_geometry::Mesh::plant): x the wind weight (0 where the
// plant is held at its root, 1 where it moves freely; wind.wgsl scales the displacement by it),
// y the baked ambient occlusion. (1, 1) for meshes without it.
fn vertex_plant(c: Cluster, base: u32, index: u32) -> vec2<f32> {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let extension = cluster_extension(header, widths);
    if (extension & CLUSTER_HAS_PLANT) == 0u {
        return vec2<f32>(1.0);
    }
    // The occlusion ends the vertex's attributes.
    let bit = attribute_bit(c, widths, extension, index + 1u) - plant_stride(extension);
    let occlusion = read_bits(vertex_stream(header, widths), bit, PLANT_CHANNEL_BITS);
    return vec2<f32>(vertex_wind_weight_of(header, widths, index), f32(occlusion) / f32((1u << PLANT_CHANNEL_BITS) - 1u));
}

// The wind weight of vertex `index` of the cluster at `header` (widths word `widths`): it follows
// the vertex's position offsets.
fn vertex_wind_weight_of(header: u32, widths: u32, index: u32) -> f32 {
    if (widths & CLUSTER_WIND_WEIGHTS) == 0u {
        return 1.0;
    }
    let stride = position_stride(widths);
    let code = read_bits(vertex_stream(header, widths), ((index + 1u) * stride) - PLANT_CHANNEL_BITS, PLANT_CHANNEL_BITS);
    return f32(code) / f32((1u << PLANT_CHANNEL_BITS) - 1u);
}

// The wind weight of vertex `index` where the instance moves in the wind (`strength` > 0, from
// wind_strength), else 1 without reading it.
fn vertex_wind_weight(c: Cluster, base: u32, index: u32, strength: f32) -> f32 {
    if strength <= 0.0 {
        return 1.0;
    }
    let header = base + c.vertex_offset;
    return vertex_wind_weight_of(header, geometry[header + 5u], index);
}
