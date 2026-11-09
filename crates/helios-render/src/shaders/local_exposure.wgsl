// Local exposure (UE5's "Local Exposure"): compress the contrast between large bright and dark
// regions around middle grey while keeping the detail within them.
//
//   build  one workgroup per grid cell splats the cell's pixels into BINS log-luminance bins
//          (linearly between the two nearest): r = Σ weight · (log luminance + RANGE), g = Σ weight,
//          in workgroup fixed-point atomics (WebGPU has no float atomics); the sum over all bins
//          is the cell's plain average, the "blurred luminance";
//   blur   separable Gaussian of the grid in x, y (space) and z (luminance), and a much wider one
//          of the blurred luminance in x and y;
//   apply  per pixel, the grid sampled at its own position *and* brightness gives the average of
//          its neighbours of similar brightness — an edge-aware base that does not bleed across
//          the rim of a window — blended with the blurred luminance. The base's distance from
//          middle grey is scaled by the highlight or shadow contrast; the detail (pixel - base)
//          stays as it is.
//
// The grid is anchored at the global exposure's metered average (adapted, without exposure
// compensation): a scene of uniform brightness is left alone, and the grid follows the exposure
// smoothly instead of the image's instantaneous content.

#include "common.wgsl"

struct Params {
    /// highlight contrast, shadow contrast, blurred luminance blend, exposure compensation (EV).
    params: vec4<f32>,
    /// Cell size in pixels, its inverse, cells in x and y.
    grid: vec4<f32>,
    /// Bin coordinate = stops from the anchor · x + y (clamped to the outermost bin centres);
    /// z: the stops the bins span on either side (values beyond are clamped to it).
    bins: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> exposure: array<f32, 4>;
@group(0) @binding(2) var color: texture_2d<f32>;
@group(0) @binding(3) var grid_in: texture_3d<f32>;
@group(0) @binding(4) var blurred_in: texture_3d<f32>;
@group(0) @binding(5) var grid_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var blurred_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var linear_sampler: sampler;

/// Blur axis: 0 = x, 1 = y, 2 = z (luminance).
override AXIS: u32 = 0u;

// Keep in sync with passes/local_exposure.rs (a unit test checks).
const BINS: u32 = 32u;
/// Fixed-point scales of the splatted values and weights (the largest cell, 64², sums to < 2^29).
const VALUE_SCALE: f32 = 64.0;
const WEIGHT_SCALE: f32 = 64.0;
/// Grid blur: in cells (space) and bins (luminance, a bin is 0.65 stops).
const GRID_SIGMA_XY: f32 = 1.0;
const GRID_SIGMA_Z: f32 = 1.0;
const GRID_RADIUS: i32 = 2;
/// Blurred luminance: a wide Gaussian (about a tenth of the screen height).
const BLURRED_SIGMA: f32 = 4.0;
const BLURRED_RADIUS: i32 = 10;
/// Weight (in cells' worth of pixels) of the blurred luminance as a prior of the bilateral
/// average: where the grid holds almost nothing at a pixel's brightness, the base falls back to
/// the blurred luminance instead of a few pixels' noisy average.
const PRIOR_WEIGHT: f32 = 0.05;

var<workgroup> sums: array<atomic<u32>, BINS>;
var<workgroup> weights: array<atomic<u32>, BINS>;

/// Log2 luminance of the metered scene average, the grid's centre.
fn anchor() -> f32 {
    return log2(0.18 / max(exposure[0], 1e-10)) + params.params.w;
}

/// Continuous bin coordinate (bin i covers [i, i + 1)) of a log luminance relative to the anchor.
fn bin_coordinate(stops: f32) -> f32 {
    return clamp(stops * params.bins.x + params.bins.y, 0.5, f32(BINS) - 0.5);
}

fn relative_stops(c: vec3<f32>, center: f32) -> f32 {
    return log2(max(luminance(c), 1e-8)) - center;
}

@compute @workgroup_size(16, 16)
fn build(
    @builtin(workgroup_id) cell: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    if index < BINS {
        atomicStore(&sums[index], 0u);
        atomicStore(&weights[index], 0u);
    }
    workgroupBarrier();

    let tile = u32(params.grid.x);
    let size = textureDimensions(color);
    let center = anchor();
    for (var y = lid.y; y < tile; y += 16u) {
        for (var x = lid.x; x < tile; x += 16u) {
            let pixel = cell.xy * tile + vec2<u32>(x, y);
            if any(pixel >= size) {
                continue;
            }
            let stops = relative_stops(textureLoad(color, pixel, 0).rgb, center);
            let z = bin_coordinate(stops) - 0.5;
            let lo = min(u32(z), BINS - 1u);
            let hi = min(lo + 1u, BINS - 1u);
            let range = params.bins.z;
            let value = u32((clamp(stops, -range, range) + range) * VALUE_SCALE + 0.5);
            let w_hi = u32((z - f32(lo)) * WEIGHT_SCALE + 0.5);
            let w_lo = u32(WEIGHT_SCALE) - w_hi;
            atomicAdd(&sums[lo], value * w_lo);
            atomicAdd(&weights[lo], w_lo);
            atomicAdd(&sums[hi], value * w_hi);
            atomicAdd(&weights[hi], w_hi);
        }
    }
    workgroupBarrier();

    // Normalised by a full cell's pixel count: a cell cut off by the screen edge weighs less.
    let norm = 1.0 / (WEIGHT_SCALE * f32(tile * tile));
    if index < BINS {
        let sum = f32(atomicLoad(&sums[index])) / VALUE_SCALE;
        let weight = f32(atomicLoad(&weights[index]));
        textureStore(grid_out, vec3<u32>(cell.xy, index), vec4<f32>(sum * norm, weight * norm, 0.0, 0.0));
    }
    if index == 0u {
        var sum = 0.0;
        var weight = 0.0;
        for (var b = 0u; b < BINS; b++) {
            sum += f32(atomicLoad(&sums[b])) / VALUE_SCALE;
            weight += f32(atomicLoad(&weights[b]));
        }
        textureStore(blurred_out, vec3<u32>(cell.xy, 0u), vec4<f32>(sum * norm, weight * norm, 0.0, 0.0));
    }
}

/// One axis of a Gaussian over (Σ value, Σ weight) pairs. Taps outside the volume count as
/// empty (zero), which is exact for these homogeneous sums: the final division by the weight
/// renormalises.
fn gaussian(source: texture_3d<f32>, p: vec3<i32>, size: vec3<i32>, sigma: f32, radius: i32) -> vec4<f32> {
    let axis = vec3<i32>(select(0, 1, AXIS == 0u), select(0, 1, AXIS == 1u), select(0, 1, AXIS == 2u));
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let w = exp(-0.5 * f32(i * i) / (sigma * sigma));
        total += w;
        let q = p + axis * i;
        if all(q >= vec3<i32>(0)) && all(q < size) {
            sum += textureLoad(source, q, 0) * w;
        }
    }
    return sum / total;
}

@compute @workgroup_size(4, 4, 4)
fn blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec3<i32>(textureDimensions(grid_in));
    let p = vec3<i32>(id);
    if any(p >= size) {
        return;
    }
    textureStore(grid_out, p, gaussian(grid_in, p, size, select(GRID_SIGMA_XY, GRID_SIGMA_Z, AXIS == 2u), GRID_RADIUS));
    if AXIS < 2u && p.z == 0 {
        textureStore(blurred_out, p, gaussian(blurred_in, p, vec3<i32>(textureDimensions(blurred_in)), BLURRED_SIGMA, BLURRED_RADIUS));
    }
}

@compute @workgroup_size(8, 8)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(color_out);
    if any(gid.xy >= size) {
        return;
    }
    let c = textureLoad(color, gid.xy, 0);
    let stops = relative_stops(c.rgb, anchor());
    let uv = (vec2<f32>(gid.xy) + 0.5) * params.grid.y / params.grid.zw;
    let grid = textureSampleLevel(grid_in, linear_sampler, vec3<f32>(uv, bin_coordinate(stops) / f32(BINS)), 0.0).xy;
    let wide = textureSampleLevel(blurred_in, linear_sampler, vec3<f32>(uv, 0.5), 0.0).xy;
    let blurred = wide.x / max(wide.y, 1e-6);
    let bilateral = (grid.x + blurred * PRIOR_WEIGHT) / (grid.y + PRIOR_WEIGHT);
    let base = mix(bilateral, blurred, params.params.z) - params.bins.z;
    let contrast = select(params.params.y, params.params.x, base > 0.0);
    textureStore(color_out, gid.xy, vec4<f32>(c.rgb * exp2(base * (contrast - 1.0)), c.a));
}
