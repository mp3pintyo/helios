// Eye adaptation from a luminance histogram (as UE's "Auto Exposure Histogram"):
//
//   * 4096 centre-weighted samples of the HDR image are binned by log2 luminance;
//   * the darkest LOW_PERCENT and brightest HIGH_PERCENT of the (weighted) samples are ignored,
//     so a sun disk, a candle flame or a black doorway cannot pull the exposure around;
//   * the mean log luminance of what remains sets the target; adaptation is smooth in log space.
//
// exposure[0] = current exposure multiplier, exposure[1] = measured average luminance.

#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<storage, read_write> exposure: array<f32, 4>;

const GRID: u32 = 64u;
const BINS: u32 = 64u;
const MIN_LOG: f32 = -12.0;
const MAX_LOG: f32 = 16.0;
const LOW_PERCENT: f32 = 0.10;
const HIGH_PERCENT: f32 = 0.90;
// Under water the brightest tenth of the view is Snell's window, the light source of the scene:
// metering without it exposes for the dim water alone and burns the window out to white. Only
// a sun-disk-sized top is ignored there.
const HIGH_PERCENT_UNDERWATER: f32 = 0.99;
/// Fixed-point scale of the per-sample weight (workgroup atomics are integer).
const WEIGHT_SCALE: f32 = 256.0;

var<workgroup> histogram: array<atomic<u32>, 64>;

fn bin_of(log_luminance: f32) -> u32 {
    let t = saturate((log_luminance - MIN_LOG) / (MAX_LOG - MIN_LOG));
    return min(u32(t * f32(BINS)), BINS - 1u);
}

fn bin_center(bin: u32) -> f32 {
    return MIN_LOG + (f32(bin) + 0.5) / f32(BINS) * (MAX_LOG - MIN_LOG);
}

@compute @workgroup_size(256)
fn adapt(@builtin(local_invocation_index) lid: u32) {
    if lid < BINS {
        atomicStore(&histogram[lid], 0u);
    }
    workgroupBarrier();
    for (var i = lid; i < GRID * GRID; i += 256u) {
        let uv = (vec2<f32>(f32(i % GRID), f32(i / GRID)) + 0.5) / f32(GRID);
        let lum = luminance(textureSampleLevel(hdr, linear_sampler, uv, 0.0).rgb);
        let centered = uv - 0.5;
        let weight = exp(-dot(centered, centered) * 4.0);
        atomicAdd(&histogram[bin_of(log2(max(lum, 1e-6)))], u32(weight * WEIGHT_SCALE + 0.5));
    }
    workgroupBarrier();
    if lid != 0u {
        return;
    }

    var total = 0.0;
    for (var b = 0u; b < BINS; b++) {
        total += f32(atomicLoad(&histogram[b]));
    }
    // Mean log luminance of the samples between the two percentiles.
    let low = total * LOW_PERCENT;
    let high = total * select(HIGH_PERCENT, HIGH_PERCENT_UNDERWATER, has_flag(frame.flags, FLAG_UNDERWATER));
    var below = 0.0;
    var sum = 0.0;
    var weight = 0.0;
    for (var b = 0u; b < BINS; b++) {
        let count = f32(atomicLoad(&histogram[b]));
        let kept = max(min(below + count, high) - max(below, low), 0.0);
        sum += bin_center(b) * kept;
        weight += kept;
        below += count;
    }
    let average = exp2(select(0.0, sum / max(weight, 1e-6), weight > 0.0));

    let compensation = exp2(frame.exposure_compensation);
    var target_exposure = compensation * 0.18 / max(average, 2e-5);
    if !has_flag(frame.flags, FLAG_AUTO_EXPOSURE) {
        target_exposure = compensation;
    }
    let previous = exposure[0];
    var adapted = target_exposure;
    if has_flag(frame.flags, FLAG_HISTORY_VALID) && previous > 0.0 && has_flag(frame.flags, FLAG_AUTO_EXPOSURE) {
        // Adapt in log space; brightening (pupil closing) is faster than darkening.
        let speed = select(1.5, 3.0, target_exposure < previous);
        let t = 1.0 - exp(-frame.misc.x * speed);
        adapted = exp2(mix(log2(previous), log2(target_exposure), t));
    }
    exposure[0] = adapted;
    exposure[1] = average;
}
