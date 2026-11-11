// Adaptive voxel-based order-independent transparency (avboit.rs; UE 5.9's r.OIT.Method=1):
// a camera-aligned grid of light transmittance over the translucent content of the view. Each
// column of the grid covers `tile` x `tile` internal pixels; its `slices` cells are spaced
// logarithmically between the nearest and the farthest translucent fragment the column holds
// (adaptive: the cells sit where translucency actually is). Requires common.wgsl.
//
// Writers (avboit_write.wgsl) bound each column, then add every translucent fragment's optical
// depth to the cell it falls in; avboit.wgsl turns the cells into the transmittance in front of
// each cell (avboit_read.wgsl samples it). Full-resolution translucent fragments use that
// transmittance as their blend weight, so they composite order-independently.

struct Avboit {
    // x, y: columns; z: slices; w: internal pixels per column side. All 0: the grid is off.
    grid: vec4<u32>,
}

@group(0) @binding(120) var<uniform> avboit: Avboit;

fn avboit_enabled() -> bool {
    return avboit.grid.x > 0u;
}

// Slice coordinate (0 at the near bound, `slices` at the far one) of view depth `d` in a column
// bounded by `b` (near, far).
fn avboit_slice(d: f32, b: vec2<f32>) -> f32 {
    return log(max(d, 1e-4) / b.x) / log(b.y / b.x) * f32(avboit.grid.z);
}
