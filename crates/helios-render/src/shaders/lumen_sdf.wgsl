// Global distance field composition (one clipmap level per dispatch).
//
//   cull_bricks : per 8^3 brick, the instances whose (band-expanded) bounds touch it
//   compose     : per voxel, min over the brick's instances of their mesh SDF (sampled in
//                 object space), plus the albedo of the closest surface for the voxel surface
//                 cache and that instance's index for the mesh-card surface cache.

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
    _pad: u32,
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
    _pad: u32,
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

@compute @workgroup_size(64)
fn cull_bricks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= BRICKS_PER_SIDE * BRICKS_PER_SIDE * BRICKS_PER_SIDE {
        return;
    }
    let coords = vec3<u32>(b % BRICKS_PER_SIDE, (b / BRICKS_PER_SIDE) % BRICKS_PER_SIDE, b / (BRICKS_PER_SIDE * BRICKS_PER_SIDE));
    let voxel = params.origin.w;
    let band = voxel * SDF_BAND_VOXELS;
    let lo = params.origin.xyz + vec3<f32>(coords * BRICK) * voxel - band;
    let hi = lo + f32(BRICK) * voxel + 2.0 * band;
    var count = 0u;
    for (var i = 0u; i < params.candidate_count; i++) {
        let inst = sdf_instances[candidates[i]];
        if all(inst.world_max.xyz >= lo) && all(inst.world_min.xyz <= hi) {
            if count < BRICK_WORDS - 1u {
                bricks[b * BRICK_WORDS + 1u + count] = candidates[i];
                count++;
            }
        }
    }
    bricks[b * BRICK_WORDS] = count;
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

// Composes voxel `gid`; returns the index of the closest instance with mesh cards (0xffff if
// none or too large). Others - heightfield terrain, which is solid far below its surface and
// often coincides with floors - would hide the cards of the surfaces they touch.
fn compose_voxel(gid: vec3<u32>) -> u32 {
    let voxel = params.origin.w;
    let band = voxel * SDF_BAND_VOXELS;
    let p = params.origin.xyz + (vec3<f32>(gid) + 0.5) * voxel;
    let brick = gid / BRICK;
    let b = brick.x + BRICKS_PER_SIDE * (brick.y + BRICKS_PER_SIDE * brick.z);
    let count = bricks[b * BRICK_WORDS];
    var best = band;
    var best_material = 0xffffffffu;
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
        }
        if d < best_card && inst.has_cards != 0u {
            best_card = d;
            best_instance = min(index, 0xffffu);
        }
    }
    var albedo = vec3<f32>(0.0);
    if best_material != 0xffffffffu {
        albedo = material_albedo_average(best_material);
    }
    let texel = vec3<u32>(gid.xy, gid.z + params.level * u32(SDF_RES));
    textureStore(out_sdf, texel, vec4<f32>(clamp(best / band, -1.0, 1.0), albedo));
    return best_instance;
}

// One thread per pair of voxels along x (they share an instance-index texel).
@compute @workgroup_size(4, 4, 4)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = u32(SDF_RES);
    if gid.x >= res / 2u || gid.y >= res || gid.z >= res {
        return;
    }
    let even = compose_voxel(vec3<u32>(gid.x * 2u, gid.yz));
    let odd = compose_voxel(vec3<u32>(gid.x * 2u + 1u, gid.yz));
    textureStore(out_instances, vec3<u32>(gid.xy, gid.z + params.level * res), vec4<u32>(even | (odd << 16u), 0u, 0u, 0u));
}
