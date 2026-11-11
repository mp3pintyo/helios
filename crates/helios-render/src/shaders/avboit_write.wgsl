// Writing the AVBOIT grid (see avboit_common.wgsl, which must be included first). Both buffers
// are cleared to zero every frame.
//
// `avboit_bounds`: per column four words. 0, 1: the nearest and farthest view depth of the
// translucent surfaces and particles, as atomic maxima of the depth's bits (positive floats order
// as their bits): ~bits(near), bits(far). Volumes don't widen them (a smoke column or a lake's
// mist would spread the cells over metres); 2: their optical depth in front of the near bound
// (f32 bits, the volume injection); 3: the cells' bounds (two halves, `avboit_store_bounds`).
// `avboit_grid`: per cell three words. 0: optical depth of red | green, 1: blue (low half), as
// 16-bit fixed point added with atomics (no rasteriser-ordered views in WebGPU: the
// sum of optical depths is order independent), 2: the volumes' optical depth (f32 bits, stored by
// the volume injection, one thread per column).

@group(0) @binding(123) var<storage, read_write> avboit_grid: array<atomic<u32>>;
@group(0) @binding(124) var<storage, read_write> avboit_bounds_buffer: array<atomic<u32>>;

// Fixed-point units per unit of optical depth.
const AVBOIT_OD_SCALE: f32 = 512.0;
// One fragment adds at most this optical depth (transmittance 1e-7).
const AVBOIT_MAX_OD: f32 = 16.0;
// A 16-bit field above this takes no more (it would carry into its neighbour; the cell is opaque
// long before: optical depth 96).
const AVBOIT_FULL: u32 = 0xc000u;

fn avboit_column_index(column: vec2<u32>) -> u32 {
    return column.y * avboit.grid.x + column.x;
}

fn avboit_cell_word(column: u32, k: u32) -> u32 {
    return (k * avboit.grid.x * avboit.grid.y + column) * 3u;
}

// Widens a column's bounds to hold view depth `d`.
fn avboit_extend(column: u32, d: f32) {
    atomicMax(&avboit_bounds_buffer[column * 4u], ~bitcast<u32>(d));
    atomicMax(&avboit_bounds_buffer[column * 4u + 1u], bitcast<u32>(d));
}

// The bounds a column's own fragments gave it (near, far; 0, 0 for none).
fn avboit_raw_bounds(column: u32) -> vec2<f32> {
    let far_bits = atomicLoad(&avboit_bounds_buffer[column * 4u + 1u]);
    if far_bits == 0u {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(bitcast<f32>(~atomicLoad(&avboit_bounds_buffer[column * 4u])), bitcast<f32>(far_bits));
}

// Fixes a non-empty column's cells (avboit.wgsl's `dilate`): the bounds of its own and its 8
// neighbours' fragments - neighbours then space their cells alike, so a surface crossing them
// doesn't change cells from column to column (streaks) - a little wider, rounded to half
// precision (word 3, packed), the precision `avboit_front` stores them in: writers and readers
// agree on the cells.
fn avboit_store_bounds(column: u32, b: vec2<f32>) {
    let near = max(b.x * 0.98, 0.01);
    let far = min(max(b.y * 1.02, near * 1.05), 60000.0);
    atomicStore(&avboit_bounds_buffer[column * 4u + 3u], pack2x16float(vec2<f32>(near, far)));
}

// A column's cell bounds (near, far; 0, 0 for an empty column), after `dilate`.
fn avboit_column_bounds(column: u32) -> vec2<f32> {
    return unpack2x16float(atomicLoad(&avboit_bounds_buffer[column * 4u + 3u]));
}

// Stores the volumes' optical depth in front of a column's near bound.
fn avboit_set_volume_in_front(column: u32, optical_depth: f32) {
    atomicStore(&avboit_bounds_buffer[column * 4u + 2u], bitcast<u32>(optical_depth));
}

fn avboit_volume_in_front(column: u32) -> f32 {
    return max(bitcast<f32>(atomicLoad(&avboit_bounds_buffer[column * 4u + 2u])), 0.0);
}

// Adds a fragment of transmittance `t` (per channel) at view depth `d` to its cell.
fn avboit_add(column: u32, d: f32, t: vec3<f32>) {
    let b = avboit_column_bounds(column);
    if b.y <= b.x {
        return;
    }
    let k = u32(clamp(avboit_slice(d, b), 0.0, f32(avboit.grid.z) - 1.0));
    let od = min(-log(max(t, vec3<f32>(1e-7))), vec3<f32>(AVBOIT_MAX_OD));
    let q = vec3<u32>(od * AVBOIT_OD_SCALE + 0.5);
    let w = avboit_cell_word(column, k);
    if q.x + q.y > 0u {
        let current = atomicLoad(&avboit_grid[w]);
        if (current & 0xffffu) < AVBOIT_FULL && (current >> 16u) < AVBOIT_FULL {
            atomicAdd(&avboit_grid[w], q.x | (q.y << 16u));
        }
    }
    if q.z > 0u && atomicLoad(&avboit_grid[w + 1u]) < AVBOIT_FULL {
        atomicAdd(&avboit_grid[w + 1u], q.z);
    }
}
