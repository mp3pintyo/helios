// Lumen's world-space light grid (lumen_light_grid.rs): candidate lists of local lights for the
// surface caches' stochastic light sampling (lumen_light_sample.wgsl).
//
//   build : one workgroup per cell of a camera-centred clipmap (LIGHT_GRID_LEVELS levels of
//           LIGHT_GRID_CELLS^3 cells, the global distance field's extents). Every light whose
//           range reaches the cell gets a weight - its luminance times its falloff at its distance
//           from the cell (no less than half a cell), an upper bound of what it gives a point
//           there. A cell reached by at most LIGHT_GRID_ENTRIES lights lists them all (weight
//           1: exact); otherwise LIGHT_GRID_ENTRIES entries are drawn by systematic resampling
//           proportional to the weights, a fresh random offset every frame, each entry weighted
//           by 1 / (entries x its probability), so the list is an unbiased estimate of the full
//           set whatever the light count (MegaLights' candidate thinning, per world cell).

#include "common.wgsl"
#include "lights.wgsl"
#include "lumen_light_grid_common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(153) var<uniform> light_grid: LightGrid;
@group(0) @binding(154) var<storage, read_write> light_grid_words: array<u32>;

var<workgroup> partial_weight: array<f32, 64>;
var<workgroup> partial_count: array<u32, 64>;
var<workgroup> prefix: array<f32, 64>;
var<workgroup> chunk_base: f32;
var<workgroup> exact_slot: atomic<u32>;
var<workgroup> cell_total: f32;
var<workgroup> cell_in_range: u32;

// Weight of light `index` for the cell [lo, lo + size) (0: out of range).
fn cell_weight(index: u32, lo: vec3<f32>, size: f32) -> f32 {
    if index >= frame.light_count {
        return 0.0;
    }
    let light = lights[index];
    let centre = light.position_range.xyz;
    let range = light.position_range.w;
    let d = length(max(max(lo - centre, centre - (lo + vec3<f32>(size))), vec3<f32>(0.0)));
    if d >= range {
        return 0.0;
    }
    let r = max(d, size * 0.5);
    return max(luminance(light.color.rgb), 1e-6) * max(light_falloff(r * r, range), 1e-9);
}

@compute @workgroup_size(64)
fn build(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let cell = wid.x;
    let level = cell / LIGHT_GRID_LEVEL_CELLS;
    let local = cell % LIGHT_GRID_LEVEL_CELLS;
    let c = vec3<u32>(local % LIGHT_GRID_CELLS, (local / LIGHT_GRID_CELLS) % LIGHT_GRID_CELLS, local / (LIGHT_GRID_CELLS * LIGHT_GRID_CELLS));
    let l = light_grid.levels[level];
    let lo = l.xyz + vec3<f32>(c) * l.w;
    let chunks = (frame.light_count + 63u) / 64u;

    // Total weight and the number of lights in range.
    var weight = 0.0;
    var count = 0u;
    for (var k = 0u; k < chunks; k++) {
        let w = cell_weight(k * 64u + li, lo, l.w);
        weight += w;
        count += select(0u, 1u, w > 0.0);
    }
    partial_weight[li] = weight;
    partial_count[li] = count;
    if li == 0u {
        atomicStore(&exact_slot, 0u);
        chunk_base = 0.0;
    }
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if li < stride {
            partial_weight[li] += partial_weight[li + stride];
            partial_count[li] += partial_count[li + stride];
        }
        workgroupBarrier();
    }
    let base = cell * LIGHT_GRID_CELL_WORDS;
    if li == 0u {
        cell_total = partial_weight[0];
        cell_in_range = partial_count[0];
    }
    // Entries no light claims (rounding at the end of the resampling) stay empty.
    if li < LIGHT_GRID_ENTRIES {
        light_grid_words[base + 2u + li * 2u] = LIGHT_GRID_NONE;
    }
    storageBarrier();
    // Uniform values (the barriers below must be reached by every invocation).
    let total = workgroupUniformLoad(&cell_total);
    let in_range = workgroupUniformLoad(&cell_in_range);
    let exact = in_range <= LIGHT_GRID_ENTRIES;
    let jitter = f32(pcg_hash((cell * 9781u) ^ (light_grid.frame * 6271u)) >> 8u) * (1.0 / 16777216.0);
    let entries = f32(LIGHT_GRID_ENTRIES);
    for (var k = 0u; k < chunks; k++) {
        let index = k * 64u + li;
        let w = cell_weight(index, lo, l.w);
        if exact {
            if w > 0.0 {
                let slot = atomicAdd(&exact_slot, 1u);
                light_grid_words[base + 2u + slot * 2u] = index;
                light_grid_words[base + 3u + slot * 2u] = bitcast<u32>(1.0);
            }
            continue;
        }
        // Inclusive prefix sum of the chunk's weights (Hillis-Steele).
        prefix[li] = w;
        workgroupBarrier();
        for (var stride = 1u; stride < 64u; stride <<= 1u) {
            var add = 0.0;
            if li >= stride {
                add = prefix[li - stride];
            }
            workgroupBarrier();
            prefix[li] += add;
            workgroupBarrier();
        }
        let offset = chunk_base;
        let hi = offset + prefix[li];
        let lo_cdf = hi - w;
        // Systematic resampling: entry j sits at (j + jitter) / entries of the total weight.
        if w > 0.0 {
            let first = u32(max(ceil(lo_cdf / total * entries - jitter), 0.0));
            let end = min(u32(max(ceil(hi / total * entries - jitter), 0.0)), LIGHT_GRID_ENTRIES);
            let entry_weight = total / (entries * w);
            for (var j = first; j < end; j++) {
                light_grid_words[base + 2u + j * 2u] = index;
                light_grid_words[base + 3u + j * 2u] = bitcast<u32>(entry_weight);
            }
        }
        workgroupBarrier();
        if li == 63u {
            chunk_base = hi;
        }
        workgroupBarrier();
    }
    if li == 0u {
        light_grid_words[base] = min(in_range, LIGHT_GRID_ENTRIES);
        light_grid_words[base + 1u] = select(0u, 1u, exact);
    }
}
