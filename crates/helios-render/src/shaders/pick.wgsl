// Editor picking: reads the visibility buffer, depth and G-buffer normal under one pixel and
// writes what was hit (instance, world position, normal) for an asynchronous read-back.
//
// picked: [0] instance index (0xffffffff: sky / nothing opaque), [1] reverse-Z depth bits,
//         [4..7) world position, [8..11) world normal.

#include "common.wgsl"

struct PickRequest {
    pixel: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> request: PickRequest;
@group(0) @binding(2) var visibility_buffer: texture_2d<u32>;
@group(0) @binding(3) var depth_buffer: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> picked: array<u32, 12>;
@group(0) @binding(7) var<storage, read> part_owners: array<u32>;

#include "part_owner.wgsl"

@compute @workgroup_size(1)
fn pick() {
    let pixel = min(request.pixel, vec2<u32>(frame.viewport.xy) - vec2<u32>(1u));
    let id = textureLoad(visibility_buffer, pixel, 0).r;
    var instance = INVALID_ID;
    if id != INVALID_ID {
        // An assembly part picks its assembly's instance.
        instance = owner_instance(visible_clusters[id >> TRIANGLE_BITS].x);
    }
    let depth = textureLoad(depth_buffer, pixel, 0).r;
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let position = world_from_depth(uv, max(depth, 1e-7), frame.inv_view_proj);
    let normal = textureLoad(gbuffer_normal, pixel, 0).xyz;
    picked[0] = instance;
    picked[1] = bitcast<u32>(depth);
    picked[2] = 0u;
    picked[3] = 0u;
    for (var i = 0u; i < 3u; i++) {
        picked[4u + i] = bitcast<u32>(position[i]);
        picked[8u + i] = bitcast<u32>(normal[i]);
    }
    picked[7] = 0u;
    picked[11] = 0u;
}
