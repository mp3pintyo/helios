// Visibility-buffer rasterisation of voxel clusters (Nanite Foliage's voxel raster path), in the
// software rasteriser's two passes and target (sw_raster.wgsl): PASS 0 takes the atomicMax of
// the reverse-Z depth bits per pixel, PASS 1 writes the payload of pixels whose depth won. The
// payload is (visible cluster slot << 7 | brick), like a triangle's; the G-buffer resolve finds
// the voxel inside the brick again (voxel_resolve.wgsl). Dispatched over the software clusters
// like sw_raster.wgsl, which draws nothing for voxel clusters (they have no triangles), while
// this pass skips triangle clusters.
//
// One workgroup per cluster: the bricks' occupancy goes to shared memory, then the threads take
// the cluster's voxels in turn (up to 128 × 64) and rasterise each as a cube: its corners are
// the voxel centre (moved by the wind like a vertex there: voxels shift rigidly, they don't
// flutter) plus the instance's half-voxel axes, all projected linearly. Every covered pixel keeps
// the cube with the probability of the voxel's coverage along the view direction.

#include "common.wgsl"
#include "geometry.wgsl"
#include "wind.wgsl"
#include "voxel.wgsl"
#include "plant_wind.wgsl"

override PASS: u32 = 0u;
override LATE: bool = false;

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> page_table: array<u32>;
@group(0) @binding(6) var<storage, read> geometry: array<u32>;
// [0, w*h): depth bits, [w*h, 2*w*h): payload.
@group(0) @binding(7) var<storage, read_write> sw_target: array<atomic<u32>>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(150) var plant_bones: texture_2d<f32>;

// Pixels a voxel may cover (it is about a pixel at the LOD threshold; this guards views with a
// much coarser threshold).
const MAX_VOXEL_PIXELS: u32 = 1024u;

var<workgroup> brick_first: array<u32, 128>;
var<workgroup> brick_words: array<vec4<u32>, 128>;
// World displacement of each brick of a plant with bones.
var<workgroup> brick_offsets: array<vec3<f32>, 128>;

// Raster position (pixels, y down) of a clip-space point, homogeneous: linear in the clip
// coordinates, so it maps the half-voxel axes too.
fn clip_to_raster(clip: vec4<f32>, size: vec2<f32>) -> vec4<f32> {
    return vec4<f32>((clip.x + clip.w) * 0.5 * size.x, (clip.w - clip.y) * 0.5 * size.y, clip.z, clip.w);
}

// The brick holding voxel `index`: the last whose first voxel is not after it.
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
    let index = wid.x + wid.y * groups.x + select(0u, state[5], LATE);
    // A 2D dispatch (over 65535 clusters) overshoots: entries past the list are stale.
    if index >= software_cluster_count(LATE, state[2], state[3], state[5], view.cluster_capacity) {
        return;
    }
    let slot = view.cluster_capacity - 1u - index;
    let entry = visible_clusters[slot];
    let cluster = clusters[entry.y];
    if !cluster_is_voxel(cluster) {
        return; // a triangle cluster (uniform: one cluster per workgroup)
    }
    let v = voxel_cluster(cluster);
    let instance = instances[entry.x];
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

    let size = view.viewport.xy;
    let half = v.size * 0.5;
    let ax = clip_to_raster(view.view_proj * vec4<f32>(instance.world[0].xyz * half, 0.0), size);
    let ay = clip_to_raster(view.view_proj * vec4<f32>(instance.world[1].xyz * half, 0.0), size);
    let az = clip_to_raster(view.view_proj * vec4<f32>(instance.world[2].xyz * half, 0.0), size);
    // World directions to object space (the normal matrix is the scaled inverse transpose).
    let to_object = transpose(mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz));
    let pixels = u32(size.x);
    let payload_offset = pixels * u32(size.y);
    let count = cluster_voxel_count(cluster);
    for (var i = lid; i < count; i += 128u) {
        let b = brick_of(i, v.bricks);
        let brick = brick_words[b];
        let object = v.origin + ((voxel_cell(brick, voxel_nth_bit(brick.xy, i - brick_first[b])) + 0.5) * v.size);
        // Voxels carry no plant data (weight 1); a plant's move with their brick's bone.
        var world = wind_world_position(instance, strength, object, 1.0);
        if planted {
            world = (instance.world * vec4<f32>(object, 1.0)).xyz + brick_offsets[b];
        }
        let center = clip_to_raster(view.view_proj * vec4<f32>(world, 1.0), size);
        if !voxel_may_cover(center, ax, ay, az, size) {
            continue;
        }
        let splat = voxel_splat(center, ax, ay, az);
        // Voxels reaching the near plane are skipped (they are about a pixel: never near it).
        let lo = max(ceil(splat.lo - 0.5), vec2<f32>(0.0));
        let hi = min(floor(splat.hi - 0.5), size - 1.0);
        if !splat.valid || any(hi < lo) {
            continue;
        }
        let extent = vec2<u32>(hi - lo) + 1u;
        if extent.x * extent.y > MAX_VOXEL_PIXELS {
            continue;
        }
        let alpha = voxel_directional_coverage(voxel_coverage(geometry[v.attributes + i]), to_object * (world - view.position.xyz));
        let id = pcg_hash(entry.y * 8192u + i) ^ entry.x;
        let payload = (slot << TRIANGLE_BITS) | b;
        for (var y = 0u; y < extent.y; y++) {
            for (var x = 0u; x < extent.x; x++) {
                let pixel = lo + vec2<f32>(f32(x), f32(y));
                let depth = voxel_depth(splat, pixel + 0.5).x;
                if depth <= 0.0 || voxel_dither(pixel, id, view.frame_index) >= alpha {
                    continue;
                }
                let target_index = u32(pixel.y) * pixels + u32(pixel.x);
                let bits = bitcast<u32>(depth);
                if PASS == 0u {
                    if bits > atomicLoad(&sw_target[target_index]) {
                        atomicMax(&sw_target[target_index], bits);
                    }
                } else if bits == atomicLoad(&sw_target[target_index]) {
                    atomicStore(&sw_target[payload_offset + target_index], payload);
                }
            }
        }
    }
}
