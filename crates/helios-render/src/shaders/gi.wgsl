// Temporal + spatial denoising of Lumen's half-resolution diffuse gather (see lumen_gather.wgsl):
// 3x3 bilateral spatial filter plus reprojected temporal accumulation with neighbourhood clamping.
//
// Output rgba: rgb = mean incident radiance (E / pi), a = sky visibility.

#include "common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(6) var linear_sampler: sampler;
@group(0) @binding(7) var gi_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var gi_raw: texture_2d<f32>;
@group(0) @binding(9) var gi_history: texture_2d<f32>;

fn linear_depth_at(pixel: vec2<u32>) -> f32 {
    return frame.camera_position.w / max(textureLoad(depth_texture, pixel, 0).r, 1e-9);
}

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= half_size) {
        return;
    }
    let pixel = gid.xy * 2u;
    let center_depth = linear_depth_at(pixel);
    let center_normal = textureLoad(gbuffer_normal, pixel, 0).xyz;

    // Edge-aware 3x3 spatial filter and neighbourhood statistics.
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    var m1 = vec4<f32>(0.0);
    var m2 = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = clamp(vec2<i32>(gid.xy) + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(half_size) - 1);
            let value = textureLoad(gi_raw, tap, 0);
            let tap_full = vec2<u32>(tap) * 2u;
            let depth_weight = exp(-abs(linear_depth_at(tap_full) - center_depth) / (0.03 * center_depth + 0.05));
            let normal_weight = pow(saturate(dot(textureLoad(gbuffer_normal, tap_full, 0).xyz, center_normal)), 8.0);
            let w = depth_weight * normal_weight + select(0.0, 1e-3, x == 0 && y == 0);
            sum += value * w;
            weight_sum += w;
            m1 += value;
            m2 += value * value;
        }
    }
    let current = sum / max(weight_sum, 1e-4);

    let motion = textureLoad(gbuffer_motion, pixel, 0).xy;
    let uv = (vec2<f32>(pixel) + 1.0) * frame.viewport.zw;
    let previous_uv = uv - motion;
    let valid = has_flag(frame.flags, FLAG_HISTORY_VALID)
        && all(previous_uv > vec2<f32>(0.0)) && all(previous_uv < vec2<f32>(1.0));
    var result = current;
    if valid {
        // Variance clipping (Salvi 2016): the box follows the neighbourhood's statistics rather
        // than its extremes, so a lone outlier neither drags the history nor flips it. The box
        // is wide because the input is noisy by design; the blend is slow for the same reason.
        let mean = m1 / 9.0;
        let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec4<f32>(0.0)));
        // The screen-probe noise is spatially coherent, though: all nine taps often interpolate
        // the same one to four probes (more so with adaptive probes on thin geometry), so the
        // spread of the neighbourhood underestimates how much a probe changes from frame to
        // frame, and a box of 2 sigma alone would throw valid history away there - the probe
        // noise would show as flickering blocks. The box is never narrower than a quarter of
        // the mean.
        let box = max(2.0 * sigma, 0.25 * abs(mean)) + vec4<f32>(0.002);
        let history_uv = viewport_uv(previous_uv, frame.prev_viewport.xy * 0.5, textureDimensions(gi_history));
        let history = clamp(textureSampleLevel(gi_history, linear_sampler, history_uv, 0.0), mean - box, mean + box);
        result = mix(history, current, 0.08);
    }
    textureStore(gi_output, gid.xy, select(result, current, any(result != result)));
}
