// MegaLights denoiser (see megalights.rs), on demodulated diffuse and specular lighting:
//
// temporal: reprojects last frame's accumulation with the motion vectors, rejects history whose
//   stored depth does not match the surface (disocclusion), clips it to a wide variance box of
//   the current samples' neighbourhood and blends in the new frame by 1 / (frames accumulated),
//   up to a cap (SVGF's temporal accumulation). It also accumulates each signal's luminance second
//   moment, from which the spatial filter estimates the variance per pixel.
// spatial: edge-avoiding à-trous wavelet iterations (3x3 binomial taps at doubling strides) with
//   depth-plane (the centre's reverse-Z depth extrapolated along its screen-space slope), normal
//   and luminance edge-stopping; the luminance threshold scales with the standard deviation
//   (prefiltered over 3x3), so converged, noise-free pixels keep their detail while fresh, noisy
//   ones are smoothed broadly. Variance is filtered alongside (with squared weights), as in SVGF.
//
// The last iteration modulates the result back - diffuse colour times diffuse, environment BRDF
// times specular - into one texture, which the lighting pass adds.
//
// Accumulated textures: rgb lighting, a second moment of its luminance (temporal) or its variance
// (spatial iterations); the info texture packs the frames accumulated and the linear depth.

#include "common.wgsl"
#include "brdf.wgsl"
#include "megalights_surface.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_motion: texture_2d<f32>;
// Temporal.
@group(0) @binding(4) var raw_diffuse: texture_2d<f32>;
@group(0) @binding(5) var raw_specular: texture_2d<f32>;
@group(0) @binding(6) var history_diffuse: texture_2d<f32>;
@group(0) @binding(7) var history_specular: texture_2d<f32>;
@group(0) @binding(8) var history_info: texture_2d<u32>;
@group(0) @binding(9) var out_diffuse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var out_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var out_info: texture_storage_2d<r32uint, write>;
// Spatial (out_diffuse / out_specular as above).
@group(0) @binding(12) var in_diffuse: texture_2d<f32>;
@group(0) @binding(13) var in_specular: texture_2d<f32>;
@group(0) @binding(14) var info: texture_2d<u32>;
// x: stride, y: 1 = first iteration, z: 1 = last iteration (writes modulated lighting to
// out_diffuse only).
@group(0) @binding(15) var<uniform> pass_params: vec4<u32>;
@group(0) @binding(16) var gbuffer_albedo: texture_2d<f32>;

// Frames of history at most (blend weight 1 / n): specular follows the view and lags sooner.
const MAX_DIFFUSE_FRAMES: f32 = 24.0;
const MAX_SPECULAR_FRAMES: f32 = 12.0;
// History is clipped to the current neighbourhood's mean +- this many standard deviations
// (wide: one frame's samples are very noisy).
const CLIP_SIGMA: f32 = 2.5;
// Below this many frames the temporal moments are unreliable: variance comes from the 3x3
// neighbourhood instead.
const MIN_MOMENT_FRAMES: f32 = 4.0;
// Half precision holds the moments of radiance up to ~250.
const MAX_MOMENT: f32 = 60000.0;
const LUMINANCE_SIGMA: f32 = 4.0;
const NORMAL_POWER: f32 = 64.0;
// How far a tap may lie off the centre's surface plane, relative to the depth.
const PLANE_TOLERANCE: f32 = 0.015;

fn linear_depth_at(pixel: vec2<u32>) -> f32 {
    return frame.camera_position.w / max(textureLoad(depth_texture, pixel, 0).r, 1e-9);
}

fn moment(c: vec3<f32>) -> f32 {
    let l = luminance(c);
    return min(l * l, MAX_MOMENT);
}

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = gid.xy;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        textureStore(out_diffuse, pixel, vec4<f32>(0.0));
        textureStore(out_specular, pixel, vec4<f32>(0.0));
        textureStore(out_info, pixel, vec4<u32>(0u));
        return;
    }
    let z = frame.camera_position.w / depth;
    let diffuse = textureLoad(raw_diffuse, pixel, 0).rgb;
    let specular = textureLoad(raw_specular, pixel, 0).rgb;

    // Neighbourhood statistics of this frame's samples (taps on other surfaces left out).
    var m1_d = vec3<f32>(0.0);
    var m2_d = vec3<f32>(0.0);
    var m1_s = vec3<f32>(0.0);
    var m2_s = vec3<f32>(0.0);
    var taps = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = vec2<i32>(pixel) + vec2<i32>(x, y);
            if any(tap < vec2<i32>(0)) || any(tap >= vec2<i32>(size)) {
                continue;
            }
            if abs(linear_depth_at(vec2<u32>(tap)) - z) > 0.05 * z {
                continue;
            }
            let d = textureLoad(raw_diffuse, tap, 0).rgb;
            let s = textureLoad(raw_specular, tap, 0).rgb;
            m1_d += d;
            m2_d += d * d;
            m1_s += s;
            m2_s += s * s;
            taps += 1.0;
        }
    }
    let mean_d = m1_d / taps;
    let mean_s = m1_s / taps;
    let box_d = CLIP_SIGMA * sqrt(max(m2_d / taps - mean_d * mean_d, vec3<f32>(0.0))) + 0.1 * mean_d + vec3<f32>(1e-4);
    let box_s = CLIP_SIGMA * sqrt(max(m2_s / taps - mean_s * mean_s, vec3<f32>(0.0))) + 0.1 * mean_s + vec3<f32>(1e-4);

    // Reprojection: bilinear taps of last frame's accumulation that lie on this surface.
    var history_d = vec4<f32>(0.0);
    var history_s = vec4<f32>(0.0);
    var frames = 0.0;
    var weight = 0.0;
    if has_flag(frame.flags, FLAG_HISTORY_VALID) {
        let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
        let previous_uv = uv - textureLoad(gbuffer_motion, pixel, 0).xy;
        let world = world_from_depth(uv, depth, frame.inv_view_proj);
        let expected = (frame.prev_unjittered_view_proj * vec4<f32>(world, 1.0)).w;
        let p = previous_uv * frame.prev_viewport.xy - 0.5;
        let corner = vec2<i32>(floor(p));
        let f = p - floor(p);
        let previous_size = vec2<i32>(frame.prev_viewport.xy);
        for (var i = 0; i < 4; i++) {
            let o = vec2<i32>(i & 1, i >> 1u);
            let tap = corner + o;
            if any(tap < vec2<i32>(0)) || any(tap >= previous_size) {
                continue;
            }
            let packed = unpack2x16float(textureLoad(history_info, tap, 0).r);
            if packed.x <= 0.0 || abs(packed.y - expected) > 0.03 * expected + 0.02 {
                continue;
            }
            let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) + 1e-4;
            history_d += textureLoad(history_diffuse, tap, 0) * w;
            history_s += textureLoad(history_specular, tap, 0) * w;
            frames += packed.x * w;
            weight += w;
        }
    }
    var out_d = vec4<f32>(diffuse, moment(diffuse));
    var out_s = vec4<f32>(specular, moment(specular));
    var count = 1.0;
    if weight > 0.01 {
        history_d /= weight;
        history_s /= weight;
        count = min(frames / weight, MAX_DIFFUSE_FRAMES) + 1.0;
        let clipped_d = clamp(history_d.rgb, mean_d - box_d, mean_d + box_d);
        let clipped_s = clamp(history_s.rgb, mean_s - box_s, mean_s + box_s);
        let alpha_d = 1.0 / min(count, MAX_DIFFUSE_FRAMES);
        let alpha_s = 1.0 / min(count, MAX_SPECULAR_FRAMES);
        out_d = vec4<f32>(mix(clipped_d, diffuse, alpha_d), mix(history_d.a, out_d.a, alpha_d));
        out_s = vec4<f32>(mix(clipped_s, specular, alpha_s), mix(history_s.a, out_s.a, alpha_s));
    }
    textureStore(out_diffuse, pixel, select(out_d, vec4<f32>(0.0), any(out_d != out_d)));
    textureStore(out_specular, pixel, select(out_s, vec4<f32>(0.0), any(out_s != out_s)));
    textureStore(out_info, pixel, vec4<u32>(pack2x16float(vec2<f32>(min(count, MAX_DIFFUSE_FRAMES), z)), 0u, 0u, 0u));
}

// The first iteration's luminance variance at `pixel`: from the accumulated moments - or, while
// the history is short, from the spread of the 3x3 neighbourhood if that is larger - blurred over
// 3x3 (SVGF's prefilter: one pixel's estimate is noisy). Later iterations read the variance the
// previous one filtered along.
fn first_variance(pixel: vec2<u32>, size: vec2<u32>, frames: f32) -> vec2<f32> {
    var m1 = vec2<f32>(0.0);
    var m2 = vec2<f32>(0.0);
    var blurred = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = clamp(vec2<i32>(pixel) + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(size) - 1);
            let d = textureLoad(in_diffuse, tap, 0);
            let s = textureLoad(in_specular, tap, 0);
            let l = vec2<f32>(luminance(d.rgb), luminance(s.rgb));
            blurred += max(vec2<f32>(d.a, s.a) - l * l, vec2<f32>(0.0)) * (select(0.5, 1.0, x == 0) * select(0.5, 1.0, y == 0) * 0.25);
            m1 += l;
            m2 += l * l;
        }
    }
    let spatial = max(m2 / 9.0 - (m1 / 9.0) * (m1 / 9.0), vec2<f32>(0.0));
    return select(blurred, max(blurred, spatial), frames < MIN_MOMENT_FRAMES);
}

// Binomial (1 2 1) / 4 weight of tap offset `i` (-1..1).
fn binomial(i: i32) -> f32 {
    return select(0.25, 0.5, i == 0);
}

// Screen-space slope of the reverse-Z depth (near / view depth) along `axis`, which is linear
// across a plane: the one-sided difference that changes least, so it does not straddle an edge.
fn depth_slope(pixel: vec2<i32>, axis: vec2<i32>, size: vec2<u32>, depth: f32) -> f32 {
    let limit = vec2<i32>(size) - 1;
    let ahead = textureLoad(depth_texture, clamp(pixel + axis, vec2<i32>(0), limit), 0).r - depth;
    let behind = depth - textureLoad(depth_texture, clamp(pixel - axis, vec2<i32>(0), limit), 0).r;
    return select(behind, ahead, abs(ahead) < abs(behind));
}

@compute @workgroup_size(8, 8)
fn spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = gid.xy;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        textureStore(out_diffuse, pixel, vec4<f32>(0.0));
        if pass_params.z == 0u {
            textureStore(out_specular, pixel, vec4<f32>(0.0));
        }
        return;
    }
    let stride = i32(pass_params.x);
    let center = vec2<i32>(pixel);
    let slope = vec2<f32>(depth_slope(center, vec2<i32>(1, 0), size, depth), depth_slope(center, vec2<i32>(0, 1), size, depth));
    let n = textureLoad(gbuffer_normal, pixel, 0).xyz;
    let frames = unpack2x16float(textureLoad(info, pixel, 0).r).x;
    let center_d = textureLoad(in_diffuse, pixel, 0);
    let center_s = textureLoad(in_specular, pixel, 0);
    let first = pass_params.y != 0u;
    var variance = vec2<f32>(center_d.a, center_s.a);
    if first {
        variance = first_variance(pixel, size, frames);
    }
    let l_d = luminance(center_d.rgb);
    let l_s = luminance(center_s.rgb);
    let threshold_d = LUMINANCE_SIGMA * sqrt(variance.x) + 1e-4 + 0.01 * l_d;
    let threshold_s = LUMINANCE_SIGMA * sqrt(variance.y) + 1e-4 + 0.01 * l_s;
    let plane = PLANE_TOLERANCE * depth;

    var sum_d = vec3<f32>(0.0);
    var sum_s = vec3<f32>(0.0);
    var var_d = 0.0;
    var var_s = 0.0;
    var w_sum_d = 0.0;
    var w_sum_s = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<i32>(x, y) * stride;
            let tap = center + offset;
            if any(tap < vec2<i32>(0)) || any(tap >= vec2<i32>(size)) {
                continue;
            }
            let tap_depth = textureLoad(depth_texture, tap, 0).r;
            if tap_depth <= 0.0 {
                continue;
            }
            let geometry = binomial(x) * binomial(y)
                * exp(-abs(tap_depth - depth - dot(slope, vec2<f32>(offset))) / plane)
                * pow(saturate(dot(n, textureLoad(gbuffer_normal, tap, 0).xyz)), NORMAL_POWER);
            let d = textureLoad(in_diffuse, tap, 0);
            let s = textureLoad(in_specular, tap, 0);
            let center_tap = x == 0 && y == 0;
            let w_d = select(geometry * exp(-abs(luminance(d.rgb) - l_d) / threshold_d), geometry, center_tap);
            let w_s = select(geometry * exp(-abs(luminance(s.rgb) - l_s) / threshold_s), geometry, center_tap);
            sum_d += d.rgb * w_d;
            sum_s += s.rgb * w_s;
            // The first iteration's taps hold moments: they take the centre's estimate.
            var_d += w_d * w_d * select(d.a, variance.x, first);
            var_s += w_s * w_s * select(s.a, variance.y, first);
            w_sum_d += w_d;
            w_sum_s += w_s;
        }
    }
    var out_d = vec4<f32>(sum_d / w_sum_d, var_d / (w_sum_d * w_sum_d));
    var out_s = vec4<f32>(sum_s / w_sum_s, var_s / (w_sum_s * w_sum_s));
    out_d = select(out_d, center_d, any(out_d != out_d));
    out_s = select(out_s, center_s, any(out_s != out_s));
    if pass_params.z != 0u {
        let surface = ml_surface(pixel);
        textureStore(out_diffuse, pixel, vec4<f32>(surface.diffuse_color * out_d.rgb + ml_specular_albedo(surface) * out_s.rgb, 1.0));
        return;
    }
    textureStore(out_diffuse, pixel, out_d);
    textureStore(out_specular, pixel, out_s);
}
