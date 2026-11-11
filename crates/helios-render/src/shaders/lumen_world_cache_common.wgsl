// Lumen world cache (world_cache.rs; UE 5.9's Lumen World Cache): a sparse world-space hash
// grid of cells holding the irradiance and the albedo of the surface around them, filled by
// paths traced from the pixels on screen (lumen_world_cache.wgsl). This file is the read side
// and the cell addressing, shared by the update passes and every Lumen trace that shades a hit
// (lumen_trace.wgsl, update_probes in lumen_light.wgsl).
//
// A cell is keyed by its level (cell size doubles with every doubling of the distance to the
// camera beyond WC_NEAR), the dominant axis and sign of the surface normal (the two sides of a
// thin wall, a floor and the ceiling below it, are different cells: nothing leaks between them)
// and its coordinates on that level's grid. Along the normal's axis cells are twice as deep and
// centred on the grid planes (where floors and walls usually lie), so a hit found through a
// distance field a little off the true surface still lands in the cell of the surface.
//
// Resolved cells (`world_cache_cells`, one texel per hash slot, written by `resolve`):
//   x: key checksum (0: empty slot)
//   y: irradiance r, g (f16 x 2)
//   z: irradiance b, emission as a multiple of the albedo (f16 x 2)
//   w: albedo rgb (sqrt, 8 bits each) | frames averaged << 24 (5 bits) | screen albedo << 31
//
// Includers declare `frame: Frame` and `lumen: Lumen`.

@group(0) @binding(150) var world_cache_cells: texture_2d<u32>;

// Hash slots: WC_SLOTS_X x WC_SLOTS_X texels (CAPACITY in world_cache.rs).
const WC_SLOTS_X: u32 = 512u;
const WC_CAPACITY: u32 = 262144u;
// Linear probing: a key is looked for (and inserted) in this many consecutive slots.
const WC_PROBES: u32 = 8u;
// Cell size within WC_NEAR metres of the camera; it doubles with every doubling beyond.
const WC_BASE_CELL: f32 = 0.2;
const WC_NEAR: f32 = 8.0;
const WC_MAX_LEVEL: u32 = 12u;
// A cell's irradiance is the mean of its last frames' samples, up to this many frames.
const WC_MAX_FRAMES: u32 = 16u;
// Readers trust a cell fully once it has averaged this many frames (less: blended with the
// fallback by the fraction).
const WC_TRUSTED_FRAMES: f32 = 4.0;
const WC_NONE: u32 = 0xffffffffu;
// lumen.world_cache: off, behind the mesh cards (in place of the voxel lighting), alone (in
// place of the cards and the voxel lighting).
const WC_OFF: u32 = 0u;
const WC_FALLBACK: u32 = 1u;
const WC_REPLACE: u32 = 2u;

struct WorldCacheKey {
    // First hash slot.
    slot: u32,
    // Checksum stored in the slot (never 0).
    check: u32,
}

fn wc_level_of(distance: f32) -> u32 {
    return u32(clamp(ceil(log2(max(distance, WC_NEAR) / WC_NEAR)), 0.0, f32(WC_MAX_LEVEL)));
}

fn wc_cell_size(level: u32) -> f32 {
    return WC_BASE_CELL * f32(1u << level);
}

// The level whose distance range `distance` is nearer the edge to (readers look there too,
// so a cell created on the other side of a level boundary is still found).
fn wc_other_level(distance: f32, level: u32) -> u32 {
    let x = log2(max(distance, WC_NEAR) / WC_NEAR);
    if level == 0u {
        return 1u;
    }
    let f = x - f32(level - 1u);
    return select(level + 1u, level - 1u, f < 0.5);
}

// Normal bucket: dominant axis * 2 + (1 if negative).
fn wc_bucket(n: vec3<f32>) -> u32 {
    let a = abs(n);
    var axis = 0u;
    if a.y > a.x && a.y >= a.z {
        axis = 1u;
    } else if a.z > a.x && a.z > a.y {
        axis = 2u;
    }
    return axis * 2u + select(0u, 1u, n[axis] < 0.0);
}

// The second most dominant axis' bucket, and how close the normal is to it (0: on the primary
// axis, 1: halfway between the two).
fn wc_second_bucket(n: vec3<f32>, first: u32) -> vec2<f32> {
    let a = abs(n);
    let primary = first / 2u;
    let u = (primary + 1u) % 3u;
    let v = (primary + 2u) % 3u;
    let axis = select(v, u, a[u] >= a[v]);
    let closeness = a[axis] / max(a[primary], 1e-6);
    return vec2<f32>(f32(axis * 2u + select(0u, 1u, n[axis] < 0.0)), closeness);
}

fn wc_mix(h: u32, v: u32) -> u32 {
    return pcg_hash(h ^ v);
}

fn wc_key(p: vec3<f32>, bucket: u32, level: u32) -> WorldCacheKey {
    let size = wc_cell_size(level);
    var g = floor(p / size);
    let axis = bucket / 2u;
    g[axis] = floor(p[axis] / (2.0 * size) + 0.5);
    let c = bitcast<vec3<u32>>(vec3<i32>(g));
    let tag = (level << 3u) | bucket;
    // Two independent hashes: the slot and the checksum that identifies the key in it.
    let h = wc_mix(wc_mix(wc_mix(pcg_hash(tag), c.x), c.y), c.z);
    let k = wc_mix(wc_mix(wc_mix(pcg_hash(tag + 0x9e3779b9u), c.z), c.x), c.y);
    var key: WorldCacheKey;
    key.slot = h & (WC_CAPACITY - 1u);
    key.check = select(k, 1u, k == 0u);
    return key;
}

fn wc_texel(slot: u32) -> vec2<u32> {
    return vec2<u32>(slot % WC_SLOTS_X, slot / WC_SLOTS_X);
}

fn wc_load(slot: u32) -> vec4<u32> {
    return textureLoad(world_cache_cells, wc_texel(slot), 0);
}

// Slot of `key`, or WC_NONE.
fn wc_find(key: WorldCacheKey) -> u32 {
    for (var i = 0u; i < WC_PROBES; i++) {
        let slot = (key.slot + i) & (WC_CAPACITY - 1u);
        let check = wc_load(slot).x;
        if check == key.check {
            return slot;
        }
        if check == 0u {
            return WC_NONE;
        }
    }
    return WC_NONE;
}

struct WorldCacheCell {
    irradiance: vec3<f32>,
    albedo: vec3<f32>,
    // Emitted radiance as a multiple of the albedo.
    emission: f32,
    // Frames averaged (0: no sample yet).
    frames: u32,
    // The albedo comes from pixels on screen (exact), not from an estimate at a ray hit.
    screen: bool,
}

fn wc_decode(c: vec4<u32>) -> WorldCacheCell {
    var cell: WorldCacheCell;
    let rg = unpack2x16float(c.y);
    let be = unpack2x16float(c.z);
    cell.irradiance = vec3<f32>(rg, be.x);
    cell.emission = be.y;
    let a = unpack4x8unorm(c.w).rgb;
    cell.albedo = a * a;
    cell.frames = (c.w >> 24u) & 31u;
    cell.screen = (c.w >> 31u) != 0u;
    return cell;
}

// Radiance leaving the cell's surface (diffuse).
fn wc_exitant(cell: WorldCacheCell) -> vec3<f32> {
    return cell.albedo * (cell.irradiance / PI + cell.emission);
}

fn wc_lookup_key(p: vec3<f32>, bucket: u32, level: u32) -> u32 {
    return wc_find(wc_key(p, bucket, level));
}

// Slot of the cell for surface point `p` with normal `n`, also looking on the other side of a
// level boundary and, for normals between two axes, in the second axis' bucket. WC_NONE if none.
fn wc_find_surface(p: vec3<f32>, n: vec3<f32>) -> u32 {
    let distance = length(p - frame.camera_position.xyz);
    let level = wc_level_of(distance);
    let bucket = wc_bucket(n);
    var slot = wc_lookup_key(p, bucket, level);
    if slot != WC_NONE {
        return slot;
    }
    let second = wc_second_bucket(n, bucket);
    if second.y > 0.7 {
        slot = wc_lookup_key(p, u32(second.x), level);
        if slot != WC_NONE {
            return slot;
        }
    }
    if level < WC_MAX_LEVEL {
        slot = wc_lookup_key(p, bucket, wc_other_level(distance, level));
    }
    return slot;
}

// Radiance leaving surface point `p` (normal `n`) according to the world cache; a = how much
// to trust it (0: no cell or no sample yet).
fn world_cache_radiance(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    let slot = wc_find_surface(p, n);
    if slot == WC_NONE {
        return vec4<f32>(0.0);
    }
    let cell = wc_decode(wc_load(slot));
    if cell.frames == 0u {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(wc_exitant(cell), saturate(f32(cell.frames) / WC_TRUSTED_FRAMES));
}

// `fallback` (the voxel lighting at a hit) with the world cache blended over it where a cell
// holds the surface.
fn world_cache_over(fallback: vec3<f32>, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if lumen.world_cache == WC_OFF {
        return fallback;
    }
    let c = world_cache_radiance(p, n);
    return mix(fallback, c.rgb, c.a);
}
