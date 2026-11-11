// Virtual shadow map rasterisation of voxel clusters: the shadow counterpart of voxel_raster.wgsl
// for (cluster, page) software items (vsm_expand.wgsl sends every voxel item here; vsm_raster.wgsl
// draws nothing for them, having no triangles). One atomicMax of the reverse-Z depth bits per
// texel into the u32 mirror of the atlas, as vsm_raster.wgsl does, with the same slope-scaled
// depth bias per cube face.
//
// Voxel cubes are projected like cluster vertices (vsm_page.wgsl: the sun's orthographic levels
// and local lights' perspective faces alike), their centres moved by the wind. A texel keeps the
// cube with the probability of the voxel's coverage along the light: the noise is a fixed
// function of the texel's absolute position in its level or face mip, so cached pages and the
// pages refreshed next to them agree.

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_local_common.wgsl"
#include "geometry.wgsl"
#include "wind.wgsl"
#include "voxel.wgsl"
#include "plant_wind.wgsl"

const DRAW_HEADER: u32 = 4u; // vec4s before the first draw entry (see vsm_expand.wgsl)
const MAX_VOXEL_TEXELS: u32 = 1024u;

// Slope-scaled depth bias, as the triangle rasterisers apply it (vsm.rs).
override SLOPE_BIAS: f32 = 1.5;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> draws: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(4) var<storage, read> page_table: array<u32>;
@group(0) @binding(5) var<storage, read> geometry: array<u32>;
// Reverse-Z depth bits per physical atlas texel (0 = nothing rasterised).
@group(0) @binding(6) var<storage, read_write> sw_depth: array<atomic<u32>>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(170) var plant_bones: texture_2d<f32>;
@group(0) @binding(171) var plant_sim: texture_2d<f32>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;

var<workgroup> brick_first: array<u32, 128>;
var<workgroup> brick_words: array<vec4<u32>, 128>;
// World displacement of each brick of a plant with bones.
var<workgroup> brick_offsets: array<vec3<f32>, 128>;

// A world point in the draw's page: (texel x, y, depth, 1) * w, as page_vertex (vsm_page.wgsl)
// maps cluster vertices. Linear in the homogeneous point, so differences map vectors.
fn page_point(draw: vec4<u32>, world: vec3<f32>) -> vec4<f32> {
    let level = draw.z >> 16u;
    let page = vec2<f32>(decode_tag(draw.w));
    if level >= LOCAL_LEVEL {
        return local_page_vertex(world, level - LOCAL_LEVEL, page);
    }
    let v = vsm_level_xy(world, level) / vsm.level_info[level].y;
    return vec4<f32>(v - page * f32(vsm.page_texels), (vsm.level_view_proj[level] * vec4<f32>(world, 1.0)).z, 1.0);
}

fn brick_of(index: u32, bricks: u32) -> u32 {
    var lo = 0u;
    var hi = bricks;
    while hi - lo > 1u {
        let mid = (lo + hi) / 2u;
        if brick_first[mid] <= index {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(128)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // The 2D dispatch may overshoot the item count (draws word 5).
    let index = wid.x + wid.y * groups.x;
    let count = draws[1].y;
    if index >= count {
        return;
    }
    let draw = draws[DRAW_HEADER + vsm.draw_capacity - 1u - index];
    let cluster = clusters[draw.y];
    if !cluster_is_voxel(cluster) {
        return; // a triangle item (uniform: one item per workgroup)
    }
    let v = voxel_cluster(cluster);
    let instance = instances[draw.x];
    let strength = wind_strength(instance);
    // A plant's bricks follow their bones rigidly (plant_wind.wgsl): one pose per brick.
    let planted = plant_animated(instance, strength);
    if lid < v.bricks {
        let brick = voxel_brick(v, lid);
        brick_words[lid] = brick;
        brick_first[lid] = voxel_brick_first(brick);
        brick_offsets[lid] = vec3<f32>(0.0);
        let bone = plant_voxel_bone(v.header, v.attributes, lid);
        if planted && bone != 0xffffffffu {
            let a = plant_air(instance, strength, wind_origin(instance), wind.time.x);
            brick_offsets[lid] = plant_brick_offset(instance, a, bone, v.origin + ((voxel_cell(brick, 0u) + 2.0) * v.size), instance.world);
        }
    }
    workgroupBarrier();

    let half = v.size * 0.5;
    let origin = (instance.world * vec4<f32>(v.origin, 1.0)).xyz;
    let base = page_point(draw, origin);
    let ax = page_point(draw, origin + instance.world[0].xyz * half) - base;
    let ay = page_point(draw, origin + instance.world[1].xyz * half) - base;
    let az = page_point(draw, origin + instance.world[2].xyz * half) - base;
    let to_object = transpose(mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz));
    let level = draw.z >> 16u;
    let local = level >= LOCAL_LEVEL;
    let slot = ((level - min(level, LOCAL_LEVEL)) / LOCAL_MIPS) / LOCAL_FACES;
    // Towards the sun (the light view's z axis), or from the local light per voxel.
    let sun_view = vsm.level_light[min(level, LOCAL_LEVEL - 1u)];
    let sun = vec3<f32>(sun_view[0].z, sun_view[1].z, sun_view[2].z);
    let n = f32(vsm.page_texels);
    let page_origin = vec2<i32>(decode_tag(draw.w)) * i32(vsm.page_texels);
    let tile = vec2<i32>(vsm_tile(draw.z & 0xffffu));
    let row = i32(vsm.atlas_size);
    let voxels = cluster_voxel_count(cluster);
    for (var i = lid; i < voxels; i += 128u) {
        let b = brick_of(i, v.bricks);
        let brick = brick_words[b];
        let object = v.origin + ((voxel_cell(brick, voxel_nth_bit(brick.xy, i - brick_first[b])) + 0.5) * v.size);
        // Voxels carry no plant data (weight 1); a plant's move with their brick's bone.
        var world = wind_world_position(instance, strength, object, 1.0);
        if planted {
            world = (instance.world * vec4<f32>(object, 1.0)).xyz + brick_offsets[b];
        }
        let center = page_point(draw, world);
        if !voxel_may_cover(center, ax, ay, az, vec2<f32>(n)) {
            continue;
        }
        let splat = voxel_splat(center, ax, ay, az);
        // Texels whose centre lies in the voxel's box, clipped to the page; voxels reaching a
        // local light's near plane are skipped.
        let lo = max(ceil(splat.lo - 0.5), vec2<f32>(0.0));
        let hi = min(floor(splat.hi - 0.5), vec2<f32>(n - 1.0));
        if !splat.valid || any(hi < lo) {
            continue;
        }
        let extent = vec2<u32>(hi - lo) + 1u;
        if extent.x * extent.y > MAX_VOXEL_TEXELS {
            continue;
        }
        let light = select(sun, world - local_vsm.slots[slot].position_range.xyz, local);
        let alpha = voxel_directional_coverage(voxel_coverage(geometry[v.attributes + i]), to_object * light);
        let id = pcg_hash(draw.y * 8192u + i) ^ draw.x;
        for (var y = 0u; y < extent.y; y++) {
            for (var x = 0u; x < extent.x; x++) {
                let texel = lo + vec2<f32>(f32(x), f32(y));
                let surface = voxel_depth(splat, texel + 0.5);
                // Absolute texel coordinates grow large: hashed rather than gradient noise.
                let absolute = vec2<u32>(page_origin + vec2<i32>(texel));
                let noise = f32(pcg_hash(id ^ level ^ pcg_hash(absolute.x ^ pcg_hash(absolute.y))) >> 8u) * (1.0 / 16777216.0);
                if surface.x <= 0.0 || noise >= alpha {
                    continue;
                }
                let z = surface.x - SLOPE_BIAS * surface.y;
                if z <= 0.0 {
                    continue; // the hardware clamps to the far plane, which never passes Greater
                }
                let t = tile + vec2<i32>(texel);
                let target_index = u32(t.y * row + t.x);
                let bits = bitcast<u32>(min(z, 1.0));
                if bits > atomicLoad(&sw_depth[target_index]) {
                    atomicMax(&sw_depth[target_index], bits);
                }
            }
        }
    }
}
