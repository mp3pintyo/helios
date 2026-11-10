// Global distance field composition (one clipmap level per dispatch).
//
//   cull_bricks : per 8^3 brick, the instances whose (band-expanded) bounds touch it
//   compose     : per voxel, min over the brick's instances of their mesh SDF (sampled in
//                 object space), plus the albedo of the closest surface for the voxel surface
//                 cache and that instance's index for the mesh-card surface cache.
//
// Incremental updates (lumen_dynamic.rs) recompose only the bricks listed in `brick_list`, under
// instances that moved:
//   cull_listed    : per listed brick, its instances into the brick table of its list slot
//   compose_listed : one workgroup per listed brick, recomposing its voxels; stamps the brick
//                    for the invalidation debug view

#include "common.wgsl"
#include "lumen_types.wgsl"
#include "lumen_material.wgsl"

struct SdfInstance {
    inv_world: mat4x4<f32>,
    world_min: vec4<f32>,   // w: uniform scale
    world_max: vec4<f32>,
    material: u32,
    mesh: u32,
    has_cards: u32,
    // 1 + its (surface, material) pair in the mesh cards, if that pair's material is captured
    // (cards.rs); 0: none.
    pair: u32,
}

struct MeshSdfDesc {
    origin: vec4<f32>,      // w: voxel size
    res: vec4<u32>,         // w: first word
    params: vec4<f32>,      // band, kind (0 volume, 1 heightfield), height min, height max
}

struct ComposeParams {
    origin: vec4<f32>,      // level min corner, w: voxel size
    level: u32,
    candidate_count: u32,
    level_count: u32,
    brick_count: u32,       // listed bricks (incremental updates)
}

const BRICK: u32 = 8u;
const BRICKS_PER_SIDE: u32 = 16u;
const BRICK_WORDS: u32 = 64u; // count + 63 instance indices

@group(0) @binding(0) var<uniform> params: ComposeParams;
@group(0) @binding(1) var<storage, read> sdf_instances: array<SdfInstance>;
@group(0) @binding(2) var<storage, read> descs: array<MeshSdfDesc>;
@group(0) @binding(3) var<storage, read> words: array<u32>;
@group(0) @binding(4) var<storage, read> candidates: array<u32>;
@group(0) @binding(5) var<storage, read_write> bricks: array<u32>;
@group(0) @binding(6) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(7) var material_albedo: texture_2d_array<f32>;
@group(0) @binding(8) var material_sampler: sampler;
@group(0) @binding(9) var out_sdf: texture_storage_3d<rgba8snorm, write>;
// Closest instance per voxel, 16 bits each, two voxels along x per texel (sdf_instance_at in
// lumen_cards.wgsl).
@group(0) @binding(70) var out_instances: texture_storage_3d<r32uint, write>;
// Incremental updates: the level's bricks to recompose, and per level and brick a stamp set when
// it was recomposed this frame (cleared by lumen_dynamic.rs).
@group(0) @binding(10) var<storage, read> brick_list: array<u32>;
@group(0) @binding(11) var<storage, read_write> brick_stamps: array<u32>;
// Per card pair: the sums of its captured texels (lumen_card_capture.wgsl).
@group(0) @binding(12) var<storage, read> pair_sums: array<u32>;

const PAIR_SUM_WORDS: u32 = 5u;

// The albedo the field stores for `material` (negative: glass and self-luminous surfaces), the
// mean of its card pair's captured texels where the pair is captured (graphs' colours, textures'
// true mean), else the material's average.
fn voxel_albedo(material: u32, pair: u32) -> vec3<f32> {
    let average = material_albedo_average(material);
    if pair == 0u {
        return average;
    }
    let base = (pair - 1u) * PAIR_SUM_WORDS;
    let count = pair_sums[base + 4u];
    if count == 0u {
        return average;
    }
    let mean = vec3<f32>(f32(pair_sums[base]), f32(pair_sums[base + 1u]), f32(pair_sums[base + 2u])) / (255.0 * f32(count));
    let albedo = clamp(mean, vec3<f32>(0.02), vec3<f32>(0.95));
    return select(albedo, -albedo, average.x < 0.0);
}

fn brick_coords(b: u32) -> vec3<u32> {
    return vec3<u32>(b % BRICKS_PER_SIDE, (b / BRICKS_PER_SIDE) % BRICKS_PER_SIDE, b / (BRICKS_PER_SIDE * BRICKS_PER_SIDE));
}

// Lists the candidates touching brick `b` in brick table `table`.
fn cull_brick(b: u32, table: u32) {
    let coords = brick_coords(b);
    let voxel = params.origin.w;
    let band = voxel * SDF_BAND_VOXELS;
    let lo = params.origin.xyz + vec3<f32>(coords * BRICK) * voxel - band;
    let hi = lo + f32(BRICK) * voxel + 2.0 * band;
    var count = 0u;
    for (var i = 0u; i < params.candidate_count; i++) {
        let inst = sdf_instances[candidates[i]];
        if all(inst.world_max.xyz >= lo) && all(inst.world_min.xyz <= hi) {
            if count < BRICK_WORDS - 1u {
                bricks[table * BRICK_WORDS + 1u + count] = candidates[i];
                count++;
            }
        }
    }
    bricks[table * BRICK_WORDS] = count;
}

@compute @workgroup_size(64)
fn cull_bricks(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < BRICKS_PER_SIDE * BRICKS_PER_SIDE * BRICKS_PER_SIDE {
        cull_brick(gid.x, gid.x);
    }
}

@compute @workgroup_size(64)
fn cull_listed(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < params.brick_count {
        cull_brick(brick_list[gid.x], gid.x);
    }
}

fn sdf_byte(index: u32) -> f32 {
    let word = words[index / 4u];
    let byte = (word >> ((index % 4u) * 8u)) & 0xffu;
    return f32(i32(byte << 24u) >> 24u) / 127.0;
}

fn sdf_height(desc: MeshSdfDesc, x: u32, z: u32) -> f32 {
    let i = x + desc.res.x * z;
    let h = (words[desc.res.w + i / 2u] >> ((i % 2u) * 16u)) & 0xffffu;
    return mix(desc.params.z, desc.params.w, f32(h) / 65535.0);
}

fn heightfield_height(desc: MeshSdfDesc, xz: vec2<f32>) -> f32 {
    let g = clamp((xz - desc.origin.xz) / desc.origin.w, vec2<f32>(0.0), vec2<f32>(f32(desc.res.x - 1u), f32(desc.res.z - 1u)));
    let i = vec2<u32>(floor(g));
    let j = min(i + 1u, vec2<u32>(desc.res.x - 1u, desc.res.z - 1u));
    let f = fract(g);
    let h0 = mix(sdf_height(desc, i.x, i.y), sdf_height(desc, j.x, i.y), f.x);
    let h1 = mix(sdf_height(desc, i.x, j.y), sdf_height(desc, j.x, j.y), f.x);
    return mix(h0, h1, f.y);
}

// Object-space signed distance of mesh `desc` at `q` (exact within its band).
fn mesh_distance(desc: MeshSdfDesc, q: vec3<f32>) -> f32 {
    let band = desc.params.x;
    if desc.params.y > 0.5 {
        let e = desc.origin.w;
        let h = heightfield_height(desc, q.xz);
        let gx = heightfield_height(desc, q.xz + vec2<f32>(e, 0.0)) - heightfield_height(desc, q.xz - vec2<f32>(e, 0.0));
        let gz = heightfield_height(desc, q.xz + vec2<f32>(0.0, e)) - heightfield_height(desc, q.xz - vec2<f32>(0.0, e));
        let n_y = 2.0 * e / length(vec3<f32>(gx, 2.0 * e, gz));
        return (q.y - h) * n_y;
    }
    let res = vec3<f32>(desc.res.xyz);
    let g = (q - desc.origin.xyz) / desc.origin.w;
    let clamped = clamp(g, vec3<f32>(0.0), res - 1.0);
    // Distance from the grid box, for points outside the baked volume.
    let outside = length((g - clamped) * desc.origin.w);
    let i = vec3<u32>(min(floor(clamped), res - 2.0));
    let f = clamped - vec3<f32>(i);
    var v: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) {
        let c = i + vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        v[k] = sdf_byte(desc.res.w * 4u + c.x + desc.res.x * (c.y + desc.res.y * c.z));
    }
    let x0 = mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
    let x1 = mix(mix(v[4], v[5], f.x), mix(v[6], v[7], f.x), f.y);
    return max(mix(x0, x1, f.z) * band, outside);
}

// Composes voxel `gid` from the instances in brick table `b`; returns the index of the closest
// instance with mesh cards (0xffff if none or too large). Others - heightfield terrain, which is
// solid far below its surface and often coincides with floors - would hide the cards of the
// surfaces they touch.
fn compose_voxel(gid: vec3<u32>, b: u32) -> u32 {
    let voxel = params.origin.w;
    let band = voxel * SDF_BAND_VOXELS;
    let p = params.origin.xyz + (vec3<f32>(gid) + 0.5) * voxel;
    let count = bricks[b * BRICK_WORDS];
    var best = band;
    var best_material = 0xffffffffu;
    var best_pair = 0u;
    var best_card = band;
    var best_instance = 0xffffu;
    for (var k = 0u; k < count; k++) {
        let index = bricks[b * BRICK_WORDS + 1u + k];
        let inst = sdf_instances[index];
        let q = (inst.inv_world * vec4<f32>(p, 1.0)).xyz;
        let d = mesh_distance(descs[inst.mesh], q) * inst.world_min.w;
        if d < best {
            best = d;
            best_material = inst.material;
            best_pair = inst.pair;
        }
        if d < best_card && inst.has_cards != 0u {
            best_card = d;
            best_instance = min(index, 0xffffu);
        }
    }
    var albedo = vec3<f32>(0.0);
    if best_material != 0xffffffffu {
        albedo = voxel_albedo(best_material, best_pair);
    }
    let texel = vec3<u32>(gid.xy, gid.z + params.level * u32(SDF_RES));
    textureStore(out_sdf, texel, vec4<f32>(clamp(best / band, -1.0, 1.0), albedo));
    return best_instance;
}

// Composes the pair of voxels along x at `pair` (they share an instance-index texel) from brick
// table `b`.
fn compose_pair(pair: vec3<u32>, b: u32) {
    let even = compose_voxel(vec3<u32>(pair.x * 2u, pair.yz), b);
    let odd = compose_voxel(vec3<u32>(pair.x * 2u + 1u, pair.yz), b);
    textureStore(out_instances, vec3<u32>(pair.xy, pair.z + params.level * u32(SDF_RES)), vec4<u32>(even | (odd << 16u), 0u, 0u, 0u));
}

// One thread per pair of voxels along x.
@compute @workgroup_size(4, 4, 4)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = u32(SDF_RES);
    if gid.x >= res / 2u || gid.y >= res || gid.z >= res {
        return;
    }
    let brick = vec3<u32>(gid.x * 2u, gid.y, gid.z) / BRICK;
    compose_pair(gid, brick.x + BRICKS_PER_SIDE * (brick.y + BRICKS_PER_SIDE * brick.z));
}

// One workgroup per listed brick, one thread per pair of its voxels.
@compute @workgroup_size(4, 8, 8)
fn compose_listed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let slot = wid.x;
    if slot >= params.brick_count {
        return;
    }
    let b = brick_list[slot];
    let pair = brick_coords(b) * vec3<u32>(BRICK / 2u, BRICK, BRICK) + lid;
    compose_pair(pair, slot);
    if all(lid == vec3<u32>(0u)) {
        brick_stamps[params.level * BRICKS_PER_SIDE * BRICKS_PER_SIDE * BRICKS_PER_SIDE + b] = 1u;
    }
}
