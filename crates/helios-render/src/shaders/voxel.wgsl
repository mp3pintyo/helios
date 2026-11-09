// Voxel clusters (helios_geometry::voxel, which this mirrors): decoding and the cube
// rasterisation shared by the visibility-buffer rasteriser (voxel_raster.wgsl), the shadow page
// rasteriser (vsm_voxel_raster.wgsl) and the G-buffer resolve (voxel_resolve.wgsl), so all of
// them see the same voxels. Requires common.wgsl and geometry.wgsl (with its bindings).
//
// At the cluster's vertex_offset: [0..3) grid origin, [3] voxel edge (f32), [4] bricks | voxels
// << 8, [5..7) uv minimum, [7] uv range (f32); then 4 words per brick: occupancy bits 0..32 and
// 32..64 (bit x + 4y + 16z of its 4³ voxels), position x | y << 6 | z << 12 (in bricks) | first
// voxel << 18, mean uv u | v << 12 (12-bit fractions of the uv range) | uv spread << 24 (log2,
// eighths from 2^-16). At triangle_offset: one word per voxel, brick by brick in bit order:
// octahedral normal x | y << 8, coverage along x << 16, y << 21, z << 26 (5 bits, α = (c/31)²).
//
// A voxel is drawn as an opaque cube that covers a pixel with the probability of its coverage
// seen from that direction (stochastic transparency; the noise changes every frame, so temporal
// upsampling averages it into partial coverage).

const VOXEL_HEADER_WORDS: u32 = 8u;
const VOXEL_BRICK_WORDS: u32 = 4u;

struct VoxelCluster {
    header: u32,      // pool word of the header
    origin: vec3<f32>,
    size: f32,
    bricks: u32,
    attributes: u32,  // pool word of the first voxel's attributes
}

fn voxel_cluster(c: Cluster) -> VoxelCluster {
    var v: VoxelCluster;
    let base = cluster_base(c);
    v.header = base + c.vertex_offset;
    v.origin = vec3<f32>(bitcast<f32>(geometry[v.header]), bitcast<f32>(geometry[v.header + 1u]), bitcast<f32>(geometry[v.header + 2u]));
    v.size = bitcast<f32>(geometry[v.header + 3u]);
    v.bricks = geometry[v.header + 4u] & 0xffu;
    v.attributes = base + c.triangle_offset;
    return v;
}

// Brick `b`: occupancy (low, high), position word, uv word.
fn voxel_brick(v: VoxelCluster, b: u32) -> vec4<u32> {
    let w = v.header + VOXEL_HEADER_WORDS + (b * VOXEL_BRICK_WORDS);
    return vec4<u32>(geometry[w], geometry[w + 1u], geometry[w + 2u], geometry[w + 3u]);
}

fn voxel_brick_first(brick: vec4<u32>) -> u32 {
    return brick.z >> 18u;
}

// Voxel `bit` of a brick, in voxels from the cluster's origin.
fn voxel_cell(brick: vec4<u32>, bit: u32) -> vec3<f32> {
    let position = vec3<u32>(brick.z & 63u, (brick.z >> 6u) & 63u, (brick.z >> 12u) & 63u);
    return vec3<f32>((position * 4u) + vec3<u32>(bit & 3u, (bit >> 2u) & 3u, bit >> 4u));
}

// The brick's mean texture coordinate and its spread.
fn voxel_brick_uv(v: VoxelCluster, brick: vec4<u32>) -> vec3<f32> {
    let uv_min = vec2<f32>(bitcast<f32>(geometry[v.header + 5u]), bitcast<f32>(geometry[v.header + 6u]));
    let range = bitcast<f32>(geometry[v.header + 7u]);
    let uv = uv_min + (vec2<f32>(f32(brick.w & 4095u), f32((brick.w >> 12u) & 4095u)) / 4095.0) * range;
    return vec3<f32>(uv, exp2((f32(brick.w >> 24u) / 8.0) - 16.0));
}

// Index of the n-th set bit of a 64-bit mask (low, high).
fn voxel_nth_bit(mask: vec2<u32>, n: u32) -> u32 {
    let low = countOneBits(mask.x);
    var word = select(mask.x, mask.y, n >= low);
    var k = select(n, n - low, n >= low);
    var bit = select(0u, 32u, n >= low);
    for (var width = 16u; width > 0u; width = width >> 1u) {
        let lower = countOneBits(word & ((1u << width) - 1u));
        if k >= lower {
            k -= lower;
            word = word >> width;
            bit += width;
        }
    }
    return bit;
}

fn voxel_coverage(attributes: u32) -> vec3<f32> {
    let c = vec3<f32>(f32((attributes >> 16u) & 31u), f32((attributes >> 21u) & 31u), f32((attributes >> 26u) & 31u)) / 31.0;
    return c * c;
}

fn voxel_normal(attributes: u32) -> vec3<f32> {
    return decode_octahedral(attributes & 0xffffu, 8u);
}

// Coverage seen along `direction` (object space; helios_geometry::voxel::directional_coverage):
// the leaves' projection sqrt(Σ (d_i τ_i)²) over the cube's footprint Σ|d_i|, in optical depth.
fn voxel_directional_coverage(coverage: vec3<f32>, direction: vec3<f32>) -> f32 {
    let d = normalize(direction);
    let tau = -log(vec3<f32>(1.0) - min(coverage, vec3<f32>(0.999)));
    return 1.0 - exp(-length(d * tau) / (abs(d.x) + abs(d.y) + abs(d.z)));
}

// Uniform [0, 1) noise of a voxel at a pixel: interleaved gradient noise (animated per frame
// when `frame` changes), decorrelated between voxels by their id.
fn voxel_dither(pixel: vec2<f32>, id: u32, frame: u32) -> f32 {
    return fract(interleaved_gradient_noise(pixel, frame) + (f32(pcg_hash(id) >> 8u) * (1.0 / 16777216.0)));
}


// One front face of a projected voxel: a convex quad (raster x, y and depth at its corners) and
// the depth plane through it (depth = base + gradient · (p - corner 0)).
struct VoxelFace {
    q0: vec3<f32>,
    q1: vec3<f32>,
    q2: vec3<f32>,
    q3: vec3<f32>,
    gradient: vec2<f32>,
    // Not seen edge-on.
    valid: bool,
}

// A voxel projected into a raster (pixels or shadow page texels, y down) with reverse-Z depth:
// its three faces that look at the viewer (they tile its silhouette), their bounds, and
// whether every corner lies in front of the eye (or light) and inside the depth range. Kept in
// registers: no dynamically indexed arrays.
struct VoxelSplat {
    x: VoxelFace,
    y: VoxelFace,
    z: VoxelFace,
    lo: vec2<f32>,
    hi: vec2<f32>,
    valid: bool,
}

// Cheap conservative test before projecting a voxel's corners: whether its raster bounds may
// contain a pixel (texel) centre inside [0, size). Most distant voxels are smaller than a
// pixel and fall between pixel centres.
fn voxel_may_cover(c: vec4<f32>, ax: vec4<f32>, ay: vec4<f32>, az: vec4<f32>, size: vec2<f32>) -> bool {
    let dw = abs(ax.w) + abs(ay.w) + abs(az.w);
    if c.w - dw <= 0.0 {
        return true; // reaches the eye's plane: let the exact test decide
    }
    let p = c.xy / c.w;
    let reach = (abs(ax.xy) + abs(ay.xy) + abs(az.xy) + (abs(p) * dw)) / (c.w - dw);
    let lo = max(ceil(p - reach - 0.5), vec2<f32>(0.0));
    let hi = min(floor(p + reach - 0.5), size - 1.0);
    return all(lo <= hi);
}

fn voxel_project(h: vec4<f32>) -> vec3<f32> {
    return h.xyz / h.w;
}

fn voxel_in_range(h: vec4<f32>) -> bool {
    return h.w > 0.0 && h.z > 0.0 && h.z <= h.w;
}

// The face of the voxel on the viewer's side along `n` (the homogeneous image of half a voxel
// along one axis; `u`, `v` along the other two) of a voxel centred at homogeneous `c`.
fn voxel_face(c: vec4<f32>, n: vec4<f32>, u: vec4<f32>, v: vec4<f32>) -> VoxelFace {
    var f: VoxelFace;
    // Reverse-Z: the nearer face has the larger depth.
    let plus = c + n;
    let minus = c - n;
    let center = select(minus, plus, (plus.z / plus.w) > (minus.z / minus.w));
    f.q0 = voxel_project(center - u - v);
    f.q1 = voxel_project(center + u - v);
    f.q2 = voxel_project(center + u + v);
    f.q3 = voxel_project(center - u + v);
    let d1 = f.q1 - f.q0;
    let d2 = f.q3 - f.q0;
    let det = (d1.x * d2.y) - (d1.y * d2.x);
    f.valid = abs(det) > 1e-12;
    f.gradient = vec2<f32>((d1.z * d2.y) - (d2.z * d1.y), (d2.z * d1.x) - (d1.z * d2.x)) / select(1.0, det, f.valid);
    return f;
}

// From the homogeneous raster position (x, y, depth, w) of the voxel's centre and the
// homogeneous images of half a voxel along each object axis (a linear map, so they apply to
// every voxel of an instance).
fn voxel_splat(center: vec4<f32>, ax: vec4<f32>, ay: vec4<f32>, az: vec4<f32>) -> VoxelSplat {
    var s: VoxelSplat;
    s.x = voxel_face(center, ax, ay, az);
    s.y = voxel_face(center, ay, az, ax);
    s.z = voxel_face(center, az, ax, ay);
    // The three front faces hold 7 of the 8 corners: the hidden one projects inside them.
    s.lo = min(min(min(s.x.q0.xy, s.x.q1.xy), min(s.x.q2.xy, s.x.q3.xy)), min(min(s.y.q0.xy, s.y.q1.xy), min(s.y.q2.xy, s.y.q3.xy)));
    s.lo = min(s.lo, min(min(s.z.q0.xy, s.z.q1.xy), min(s.z.q2.xy, s.z.q3.xy)));
    s.hi = max(max(max(s.x.q0.xy, s.x.q1.xy), max(s.x.q2.xy, s.x.q3.xy)), max(max(s.y.q0.xy, s.y.q1.xy), max(s.y.q2.xy, s.y.q3.xy)));
    s.hi = max(s.hi, max(max(s.z.q0.xy, s.z.q1.xy), max(s.z.q2.xy, s.z.q3.xy)));
    // Every corner, the hidden one included.
    var valid = true;
    for (var k = 0u; k < 8u; k++) {
        let sign = vec3<f32>(select(-1.0, 1.0, (k & 1u) != 0u), select(-1.0, 1.0, (k & 2u) != 0u), select(-1.0, 1.0, (k & 4u) != 0u));
        valid = valid && voxel_in_range(center + (ax * sign.x) + (ay * sign.y) + (az * sign.z));
    }
    s.valid = valid;
    return s;
}

fn voxel_edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return ((b.x - a.x) * (p.y - a.y)) - ((b.y - a.y) * (p.x - a.x));
}

// Depth of one face at `p` (x) and its depth slope per raster unit (y); x < 0 outside it.
fn voxel_face_depth(f: VoxelFace, p: vec2<f32>) -> vec2<f32> {
    let e = vec4<f32>(voxel_edge(f.q0.xy, f.q1.xy, p), voxel_edge(f.q1.xy, f.q2.xy, p), voxel_edge(f.q2.xy, f.q3.xy, p), voxel_edge(f.q3.xy, f.q0.xy, p));
    let inside = all(e >= vec4<f32>(0.0)) || all(e <= vec4<f32>(0.0));
    if !inside || !f.valid {
        return vec2<f32>(-1.0, 0.0);
    }
    return vec2<f32>(f.q0.z + dot(f.gradient, p - f.q0.xy), max(abs(f.gradient.x), abs(f.gradient.y)));
}

// Depth of the voxel's front surface at raster point `p` (x) and that face's depth slope (y,
// for shadow bias); x < 0 where the voxel does not cover `p`. The depth of a projected plane is
// affine in raster space.
fn voxel_depth(s: VoxelSplat, p: vec2<f32>) -> vec2<f32> {
    let x = voxel_face_depth(s.x, p);
    let y = voxel_face_depth(s.y, p);
    let z = voxel_face_depth(s.z, p);
    let xy = select(y, x, x.x > y.x);
    return select(z, xy, xy.x > z.x);
}
