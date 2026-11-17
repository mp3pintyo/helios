// Merges hardware and software rasterisation into the final visibility buffer and depth.
// Reverse-Z: the larger depth wins. Every later pass reads these merged targets.

#include "common.wgsl"

override SW_PACKED: bool = false;

@group(0) @binding(0) var hw_depth: texture_depth_2d;
@group(0) @binding(1) var hw_visibility: texture_2d<u32>;
// The compute rasterisers' target (sw_target.wgsl): the depth bits of all pixels, then their
// payloads; or with 64-bit atomics (`SW_PACKED`, sw_target_u64.wgsl) payload and depth bits per pixel.
@group(0) @binding(2) var<storage, read> sw_target: array<u32>;
@group(0) @binding(3) var out_visibility: texture_storage_2d<r32uint, write>;
@group(0) @binding(4) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var<uniform> view: CullView;

@compute @workgroup_size(8, 8)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The internal resolution (the targets may be larger; the software rasteriser's rows are
    // this wide).
    let size = vec2<u32>(view.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let index = gid.y * size.x + gid.x;
    var depth = textureLoad(hw_depth, gid.xy, 0);
    var id = textureLoad(hw_visibility, gid.xy, 0).r;
    let sw_depth = bitcast<f32>(sw_target[select(index, (2u * index) + 1u, SW_PACKED)]);
    if sw_depth > depth {
        depth = sw_depth;
        id = sw_target[select((size.x * size.y) + index, 2u * index, SW_PACKED)];
    }
    textureStore(out_visibility, gid.xy, vec4<u32>(id));
    textureStore(out_depth, gid.xy, vec4<f32>(depth));
}
