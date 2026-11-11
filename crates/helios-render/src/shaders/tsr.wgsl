// Temporal super resolution: reconstructs the output-resolution image from jittered frames
// rendered at the internal resolution (UE5's TSR; at 100 % it is a temporal anti-aliaser).
//
// Per output pixel:
//   * gather the 3x3 internal samples around it, each weighted by a Gaussian of its (jittered)
//     distance to the output pixel centre *in output pixels* — a sample far from the centre
//     adds little, so over the jitter cycle the history converges to the scene filtered at
//     output resolution, not at the internal one;
//   * reproject the history (Catmull–Rom) with the motion of the closest depth in the
//     neighbourhood, clip it to the neighbourhood's YCoCg variance box (wide where the image is
//     still so sub-pixel detail averages instead of flickering, tight under motion);
//   * reject history where the surface moves differently from what the history pixel moved
//     with (disocclusion; the motion each history pixel was written with is kept), or where
//     the history pixel showed another surface: each history pixel also keeps the view depth
//     of the closest sample it was written with, and the surface point it stands for must lie
//     on this pixel's surface plane (the depth history, below);
//   * blend by accumulated sample weight: history alpha counts the samples (in frames at full
//     density) it holds, capped by how much history may be trusted, and luminance (Karis)
//     weights of the *exposed* colour keep lone bright samples from flickering (on raw scene
//     radiance they would all but erase small emitters such as candle flames, which at low
//     screen percentages are only sampled in some frames);
//   * where history and this frame's nearby samples together hold less than one frame's worth
//     of samples (disocclusions, new pixels at low screen percentages), fill in a wider,
//     anti-aliased filter of the current frame (the responsive path);
//   * below 100 %, lock thin bright features: a feature narrower than an internal pixel (a
//     candle flame, a glint on a wire) is only sampled in some frames of the jitter cycle, and
//     in the others the neighbourhood clip and the luminance weights would erase it from the
//     history. Where the nearest sample is a bright ridge (much brighter than all its
//     neighbours) the output pixel is locked for one jitter cycle: while the image is still and
//     the history not disoccluded, its history is neither clipped nor luminance-weighted, so
//     it converges to the feature's true coverage (FSR 2's "locks").
//   * where particles or volumes cover the pixel (the reactive mask, FLAG_REACTIVE), keep little
//     history with a tight clip and no lock: sparks and flames change without motion vectors.

#include "common.wgsl"

struct Tsr {
    output: vec4<f32>,   // output width, height, 1 / width, 1 / height
    prev_inv_view_proj: mat4x4<f32>,   // the previous frame's, unjittered
    params: vec4<f32>,   // x: depth history on
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> tsr: Tsr;
@group(0) @binding(2) var current_color: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(5) var history_color: texture_2d<f32>;
@group(0) @binding(6) var history_state: texture_2d<f32>;
@group(0) @binding(7) var linear_sampler: sampler;
@group(0) @binding(8) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var out_state: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var<storage, read> exposure: array<f32, 4>;

// Reconstruction filter exp(-k d²), d in output pixels. Its width (sigma 0.29 px) is that of the
// one-pixel box a jittered point sample averages over, so 100 % is as sharp as plain TAA; UE's
// wider Blackman–Harris fit (k = 2.29) measurably softened the image (-12 % gradient energy in
// the forest). Its integral over the plane, pi / k, is the weight one frame contributes on
// average at full sample density; weights are expressed in those "frames".
const KERNEL: f32 = 6.0;
const KERNEL_AREA: f32 = 0.5236;
// The responsive path's filter, in internal pixels: wide enough to anti-alias a single frame.
const WIDE_KERNEL: f32 = 2.29;
// Accumulated weight caps: the blend factor is weight / (history + weight), so a still image
// converges like an exponential average with 1/12.5 = 0.08 per frame, a moving one with 0.25
// (at 100 %; at lower percentages frames carry fewer samples each and history lasts longer).
const MAX_WEIGHT_STILL: f32 = 11.5;
const MAX_WEIGHT_MOVING: f32 = 3.0;
// Fully reactive pixels (particles, flames: FLAG_REACTIVE) keep at most this much history, with
// the tight clip box of a moving image and no lock: their content changes without motion vectors.
const MAX_WEIGHT_REACTIVE: f32 = 1.5;
// Photo mode (photo.rs): output this frame's reconstruction with its sample weight, no history.
// Where no sample lies near the pixel (low screen percentages), the wide filter fills in with
// this small weight, which the samples of later frames soon outweigh.
override PHOTO: bool = false;
const PHOTO_FILL: f32 = 1e-3;

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Catmull–Rom weights of the four taps around a position with fractional part `f`.
struct CubicWeights {
    w0: vec2<f32>,
    w1: vec2<f32>,
    w2: vec2<f32>,
    w3: vec2<f32>,
}

fn catmull_rom_weights(f: vec2<f32>) -> CubicWeights {
    var w: CubicWeights;
    w.w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    w.w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    w.w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    w.w3 = f * f * (-0.5 + 0.5 * f);
    return w;
}

// History colour (rgb) and weight (a) at `uv`: 9-tap Catmull–Rom from bilinear fetches (MJP's
// formulation, 5 effective taps), keeping the history sharp under repeated resampling.
fn sample_history(uv: vec2<f32>) -> vec4<f32> {
    let size = tsr.output.xy;
    let position = uv * size;
    let center = floor(position - 0.5) + 0.5;
    let w = catmull_rom_weights(position - center);
    let w12 = w.w1 + w.w2;
    let t0 = (center - 1.0) / size;
    let t3 = (center + 2.0) / size;
    let t12 = (center + w.w2 / w12) / size;

    var result = vec4<f32>(0.0);
    result += textureSampleLevel(history_color, linear_sampler, vec2<f32>(t12.x, t0.y), 0.0) * w12.x * w.w0.y;
    result += textureSampleLevel(history_color, linear_sampler, vec2<f32>(t0.x, t12.y), 0.0) * w.w0.x * w12.y;
    result += textureSampleLevel(history_color, linear_sampler, vec2<f32>(t12.x, t12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(history_color, linear_sampler, vec2<f32>(t3.x, t12.y), 0.0) * w.w3.x * w12.y;
    result += textureSampleLevel(history_color, linear_sampler, vec2<f32>(t12.x, t3.y), 0.0) * w12.x * w.w3.y;
    let weight = w12.x * w.w0.y + w.w0.x * w12.y + w12.x * w12.y + w.w3.x * w12.y + w12.x * w.w3.y;
    return max(result / weight, vec4<f32>(0.0));
}

// Spatial-only upscale (no history: TAA off): Catmull–Rom over the internal pixels around the
// continuous internal position `position`. At 100 % without jitter it returns the pixel itself.
fn upscale_spatial(position: vec2<f32>, size: vec2<i32>) -> vec3<f32> {
    let base = floor(position);
    let w = catmull_rom_weights(position - base);
    var wx = array<f32, 4>(w.w0.x, w.w1.x, w.w2.x, w.w3.x);
    var wy = array<f32, 4>(w.w0.y, w.w1.y, w.w2.y, w.w3.y);
    var sum = vec3<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let tap = clamp(vec2<i32>(base) + vec2<i32>(x - 1, y - 1), vec2<i32>(0), size - 1);
            sum += textureLoad(current_color, tap, 0).rgb * (wx[x] * wy[y]);
        }
    }
    return max(sum, vec3<f32>(0.0));
}

// Karis weight of a colour as it will be displayed (last frame's exposure); locked pixels
// (`lock` = 1) are not weighted.
fn karis(color: vec3<f32>, lock: f32) -> f32 {
    return mix(1.0 / (1.0 + luminance(color) * exposure[0]), 1.0, lock);
}

// ---- Depth history ------------------------------------------------------------------------------
// Motion vectors alone miss a disocclusion where the uncovered surface and the one that covered
// it move by nearly the same amount per frame — a slow sideways camera move (the parallax between
// a post and the wall behind it is a fraction of a pixel per frame), a branch swaying in front of
// the sky — so the history keeps the occluder's colour inside the wide clip box of a nearly
// still image and it fades over a dozen frames: a ghost. Each history pixel therefore keeps the
// view depth of the closest sample it was written with (history_state.w; the sky is SKY_DEPTH).
// That surface point, rebuilt through the history pixel with the previous camera, must lie on
// the plane of this pixel's closest surface (its normal from the depth neighbours): the plane
// distance, unlike a plain depth comparison, does not grow on surfaces seen at grazing angles,
// where the history pixel's sub-pixel offset alone changes the depth a lot. The tolerance is
// relative to the depth (DEPTH_TOLERANCE): wind and skinned meshes move surfaces a little.
//
// A thin feature (a twig, a wire) that this frame's jittered samples miss looks exactly like a
// disocclusion: the history shows something nearer. Where the surface and the history moved
// alike, that is what it is (a jitter miss on a still image, or a camera pan, which moves near
// and far alike), so the depth test acts in proportion to the difference in motion: from
// DEPTH_MOTION_START px, fully from DEPTH_MOTION_FULL px (the motion test alone needs 1–3 px).
const SKY_DEPTH: f32 = 60000.0;
const DEPTH_TOLERANCE: f32 = 0.02;
const DEPTH_MOTION_START: f32 = 0.05;
const DEPTH_MOTION_FULL: f32 = 0.5;

// View depth of a reverse-Z depth value (SKY_DEPTH for the sky).
fn view_depth(depth: f32) -> f32 {
    return select(SKY_DEPTH, min(frame.camera_position.w / depth, SKY_DEPTH), depth > 0.0);
}

// World position of internal pixel `tap`'s (jittered) sample at reverse-Z depth `depth` > 0.
fn tap_world(tap: vec2<i32>, depth: f32) -> vec3<f32> {
    return world_from_depth((vec2<f32>(tap) + 0.5) * frame.viewport.zw, depth, frame.inv_view_proj);
}

// Normal of the surface at internal pixel `tap` from its depth neighbours: per axis the one
// nearer in depth (the other may lie across a silhouette). Faces the camera where both are sky.
fn depth_normal(tap: vec2<i32>, depth: f32, world: vec3<f32>, in_size: vec2<i32>) -> vec3<f32> {
    let z = view_depth(depth);
    var axes = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    for (var a = 0; a < 2; a++) {
        let offset = select(vec2<i32>(0, 1), vec2<i32>(1, 0), a == 0);
        let pa = clamp(tap + offset, vec2<i32>(0), in_size - 1);
        let pb = clamp(tap - offset, vec2<i32>(0), in_size - 1);
        let da = textureLoad(depth_texture, pa, 0).r;
        let db = textureLoad(depth_texture, pb, 0).r;
        let use_a = da > 0.0 && (db <= 0.0 || abs(view_depth(da) - z) <= abs(view_depth(db) - z));
        let d = select(db, da, use_a);
        if d > 0.0 {
            axes[a] = tap_world(select(pb, pa, use_a), d) - world;
        }
    }
    let n = cross(axes[0], axes[1]);
    let to_eye = normalize(frame.camera_position.xyz - world);
    if dot(n, n) < 1e-20 {
        return to_eye;
    }
    let normal = normalize(n);
    return select(-normal, normal, dot(normal, to_eye) >= 0.0);
}

// How surely the history pixel `previous_pixel` (stored view depth `history_depth`) showed a
// surface other than this pixel's closest one (internal pixel `closest` at reverse-Z `depth`):
// 0 = the same surface, 1 = another.
fn depth_mismatch(closest: vec2<i32>, depth: f32, previous_pixel: vec2<u32>, history_depth: f32, in_size: vec2<i32>) -> f32 {
    let history_sky = history_depth >= SKY_DEPTH * 0.99;
    if depth <= 0.0 || history_sky {
        return select(1.0, 0.0, depth <= 0.0 && history_sky);
    }
    let world = tap_world(closest, depth);
    let normal = depth_normal(closest, depth, world, in_size);
    // The history pixel's surface point, through its centre from the previous camera.
    let uv = (vec2<f32>(previous_pixel) + 0.5) * tsr.output.zw;
    let history_world = world_from_depth(uv, frame.camera_position.w / max(history_depth, 1e-4), tsr.prev_inv_view_proj);
    let plane_distance = abs(dot(normal, history_world - world));
    return smoothstep(1.0, 2.0, plane_distance / (DEPTH_TOLERANCE * view_depth(depth)));
}

// Clips `history` towards the centre of the box [lo, hi] (Playdead / UE).
fn clip_to_box(history: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (hi + lo);
    let extent = 0.5 * (hi - lo) + 1e-5;
    let offset = history - center;
    let units = abs(offset / extent);
    let max_unit = max(units.x, max(units.y, units.z));
    return select(history, center + offset / max_unit, max_unit > 1.0);
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_size = vec2<u32>(tsr.output.xy);
    if any(gid.xy >= out_size) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let in_size = vec2<i32>(frame.viewport.xy);
    // Internal pixels per output pixel.
    let ratio = frame.upscale.xy;
    let uv = (vec2<f32>(gid.xy) + 0.5) * tsr.output.zw;
    // Internal pixel q samples the unjittered screen position (q + 0.5) / size - jitter (uv):
    // `position` is this output pixel's centre in those sample coordinates.
    let jitter_uv = vec2<f32>(frame.jitter.x, -frame.jitter.y) * 0.5;
    let position = (uv + jitter_uv) * frame.viewport.xy - 0.5;
    let use_history = has_flag(frame.flags, FLAG_TAA) && has_flag(frame.flags, FLAG_HISTORY_VALID);
    if !has_flag(frame.flags, FLAG_TAA) && !PHOTO {
        textureStore(out_color, pixel, vec4<f32>(upscale_spatial(position, in_size), 0.0));
        textureStore(out_state, pixel, vec4<f32>(0.0));
        return;
    }

    // Neighbourhood statistics, the two reconstruction filters and the closest depth.
    let nearest = clamp(vec2<i32>(round(position)), vec2<i32>(0), in_size - 1);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var sharp = vec3<f32>(0.0);
    var sharp_weight = 0.0;
    var wide = vec3<f32>(0.0);
    var wide_weight = 0.0;
    var closest_depth = 0.0;
    var closest = nearest;
    var center_luma = 0.0;
    var neighbour_luma = 0.0;
    let reactive_mask = has_flag(frame.flags, FLAG_REACTIVE);
    var reactive = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = clamp(nearest + vec2<i32>(x, y), vec2<i32>(0), in_size - 1);
            if reactive_mask {
                reactive = max(reactive, textureLoad(gbuffer_motion, tap, 0).z);
            }
            let color = textureLoad(current_color, tap, 0).rgb;
            let c = rgb_to_ycocg(color);
            m1 += c;
            m2 += c * c;
            let luma = luminance(color) * exposure[0];
            if x == 0 && y == 0 {
                center_luma = luma;
            } else {
                neighbour_luma = max(neighbour_luma, luma);
            }
            // Offset of the tap's sample from the output pixel centre, in internal pixels.
            let offset = vec2<f32>(tap) - position;
            let offset_out = offset / ratio;
            let ws = exp(-KERNEL * dot(offset_out, offset_out));
            let ww = exp(-WIDE_KERNEL * dot(offset, offset));
            sharp += color * ws;
            sharp_weight += ws;
            wide += color * ww;
            wide_weight += ww;
            let d = textureLoad(depth_texture, tap, 0).r;
            if d > closest_depth {
                closest_depth = d;
                closest = tap;
            }
        }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let current_sharp = sharp / max(sharp_weight, 1e-6);
    let current_wide = wide / max(wide_weight, 1e-6);
    // This frame's samples near the pixel, in frames at full density (about `ratio²` on average).
    let frame_weight = sharp_weight / KERNEL_AREA;

    let motion = textureLoad(gbuffer_motion, closest, 0).xy;
    let motion_pixels = motion * tsr.output.xy;
    let speed = length(motion_pixels);
    let previous_uv = uv - motion;
    if PHOTO {
        // Photo mode (photo.rs) sums frames itself: this frame's reconstruction and its sample
        // weight alone, kept a valid one-frame history for when the photo ends.
        let w = frame_weight + PHOTO_FILL;
        let color = (current_sharp * frame_weight + current_wide * PHOTO_FILL) / w;
        textureStore(out_color, pixel, vec4<f32>(select(color, current_wide, any(color != color)), w));
        textureStore(out_state, pixel, vec4<f32>(motion_pixels, 0.0, view_depth(closest_depth)));
        return;
    }
    let max_weight = mix(mix(MAX_WEIGHT_STILL, MAX_WEIGHT_MOVING, saturate(speed / 8.0)), MAX_WEIGHT_REACTIVE, reactive);

    // Thin-feature lock: none at 100 % (every feature is sampled every frame), full from 75 %
    // down. A pixel showing a ridge now is locked whatever the motion (the feature is there);
    // carrying a lock over to the next frames, for the jitter phases that miss the feature,
    // needs a still image, or a moving feature would leave an unclipped trail. It lasts one
    // jitter cycle (`jitter_phase_count` in resolution.rs).
    let min_ratio = min(ratio.x, ratio.y);
    let still = 1.0 - saturate(speed / 1.5);
    let lock_strength = saturate((1.0 - min_ratio) * 4.0);
    let ridge = center_luma > max(2.0 * neighbour_luma, neighbour_luma + 0.5);
    let lock_decay = max(min_ratio * min_ratio / 8.0, 1.0 / 64.0);
    var lock = select(0.0, lock_strength, ridge) * (1.0 - reactive);

    var history = vec3<f32>(0.0);
    var history_weight = 0.0;
    if use_history && all(previous_uv >= vec2<f32>(0.0)) && all(previous_uv <= vec2<f32>(1.0)) {
        let h = sample_history(previous_uv);
        // Clip box size: tight under motion (no ghosting), wide where the image is still.
        // Sub-pixel detail (leaves, needles, thin tracery) lands in a different set of pixels
        // with every jitter offset; a tight box treats each of those states as "new" and the
        // detail flickers instead of being averaged.
        let gamma = mix(mix(2.25, 1.0, saturate(speed / 1.5)), 1.0, reactive);
        let clipped = ycocg_to_rgb(clip_to_box(rgb_to_ycocg(h.rgb), mean - gamma * sigma, mean + gamma * sigma));
        // Disocclusion: the history pixel was reprojected with a motion unlike this surface's,
        // so it showed something else (e.g. the background a foreground object uncovers).
        let previous_pixel = min(vec2<u32>(previous_uv * tsr.output.xy), out_size - 1u);
        let previous = textureLoad(history_state, previous_pixel, 0);
        let motion_difference = length(motion_pixels - previous.xy);
        var disocclusion = saturate((motion_difference - 1.0) * 0.5);
        // The depth history: another surface, while the motion differs only a little.
        let depth_weight = saturate((motion_difference - DEPTH_MOTION_START) / (DEPTH_MOTION_FULL - DEPTH_MOTION_START));
        if tsr.params.x > 0.0 && depth_weight > disocclusion {
            disocclusion = max(disocclusion, depth_weight * depth_mismatch(closest, closest_depth, previous_pixel, previous.w, in_size));
        }
        history_weight = min(h.a, max_weight) * (1.0 - disocclusion);
        lock = max(lock, (previous.z - lock_decay) * (1.0 - disocclusion) * still * (1.0 - reactive));
        history = mix(clipped, h.rgb, lock);
    }

    // Too few samples in history and nearby this frame: fill in the wide filter.
    let fill = max(1.0 - history_weight - frame_weight, 0.0);
    let wh = history_weight * karis(history, lock);
    let ws = frame_weight * karis(current_sharp, lock);
    let wf = fill * karis(current_wide, lock);
    var result = (history * wh + current_sharp * ws + current_wide * wf) / max(wh + ws + wf, 1e-6);
    result = select(result, current_wide, any(result != result));
    let weight = min(history_weight + frame_weight, max_weight);
    textureStore(out_color, pixel, vec4<f32>(result, weight));
    textureStore(out_state, pixel, vec4<f32>(motion_pixels, max(lock, 0.0), view_depth(closest_depth)));
}
