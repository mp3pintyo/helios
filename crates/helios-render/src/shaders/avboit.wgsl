// AVBOIT resolve (avboit.rs): per column, the cells' optical depths become the transmittance in
// front of each cell (an exclusive product along the column), read by avboit_read.wgsl.

#include "common.wgsl"
#include "avboit_common.wgsl"
#include "avboit_write.wgsl"

@group(0) @binding(125) var front_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(126) var surface_out: texture_storage_3d<rgba8unorm, write>;

// After the bounds draw: each column holding fragments takes the bounds of its 3x3
// neighbourhood's (avboit_store_bounds).
@compute @workgroup_size(8, 8)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= avboit.grid.xy) {
        return;
    }
    let column = avboit_column_index(gid.xy);
    if avboit_raw_bounds(column).y <= 0.0 {
        return;
    }
    var b = vec2<f32>(1e30, 0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let p = vec2<i32>(gid.xy) + vec2<i32>(x, y);
            if any(p < vec2<i32>(0)) || any(p >= vec2<i32>(avboit.grid.xy)) {
                continue;
            }
            let n = avboit_raw_bounds(avboit_column_index(vec2<u32>(p)));
            if n.y > 0.0 {
                b = vec2<f32>(min(b.x, n.x), max(b.y, n.y));
            }
        }
    }
    avboit_store_bounds(column, b);
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= avboit.grid.xy) {
        return;
    }
    let column = avboit_column_index(gid.xy);
    let b = avboit_column_bounds(column);
    let p = vec2<i32>(gid.xy);
    textureStore(front_out, vec3<i32>(p, 0), vec4<f32>(b, 0.0, 0.0));
    if b.y <= b.x {
        // Readers stop at the empty bounds.
        return;
    }
    let slices = avboit.grid.z;
    var surfaces = vec3<f32>(1.0);
    var volumes = exp(-avboit_volume_in_front(column));
    for (var k = 0u; k < slices; k++) {
        let w = avboit_cell_word(column, k);
        let rg = atomicLoad(&avboit_grid[w]);
        let ba = atomicLoad(&avboit_grid[w + 1u]);
        let od = vec3<f32>(f32(rg & 0xffffu), f32(rg >> 16u), f32(ba & 0xffffu)) / AVBOIT_OD_SCALE;
        let t = exp(-od);
        textureStore(front_out, vec3<i32>(p, i32(k) + 1), vec4<f32>(surfaces * volumes, 1.0));
        textureStore(surface_out, vec3<i32>(p, i32(k)), vec4<f32>(surfaces, 1.0));
        surfaces *= t;
        volumes *= exp(-max(bitcast<f32>(atomicLoad(&avboit_grid[w + 2u])), 0.0));
    }
    textureStore(front_out, vec3<i32>(p, i32(slices) + 1), vec4<f32>(surfaces * volumes, 1.0));
    textureStore(surface_out, vec3<i32>(p, i32(slices)), vec4<f32>(surfaces, 1.0));
}
