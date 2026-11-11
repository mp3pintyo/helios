// Camera effects after TSR, at output resolution: depth of field and motion blur. Depth and
// motion vectors are read at the internal resolution they were rendered at.
//
//   tiles   : per 16x16 tile, the largest motion-blur velocity and the largest circle of
//             confusion (both in pixels)
//   dilate  : 3x3 maximum over tiles, so blur from a neighbouring tile can reach into this one
//   focus   : autofocus, the depth at the screen centre, smoothed over time
//   dof     : scatter-as-gather bokeh (a disc of taps up to the tile's largest CoC; a tap
//             contributes where its own blur reaches this pixel, background taps are limited by
//             this pixel's own CoC so sharp foreground does not smear onto the background)
//   motion  : McGuire et al. 2012 reconstruction filter along the tile's dominant velocity, with
//             depth-aware weights so moving objects blur over the background and not vice versa

#include "common.wgsl"

struct PostFx {
    // x: shutter fraction (0 = no motion blur), y: max blur radius (px),
    // z: aperture diameter over sensor height scaled to pixels (0 = no DOF), w: focal length (m)
    params: vec4<f32>,
    // x: manual focus distance (m, 0 = autofocus), y: max CoC (px)
    focus: vec4<f32>,
    // output width, height, 1 / width, 1 / height
    output: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> fx: PostFx;
@group(0) @binding(2) var color_in: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(5) var tiles_in: texture_2d<f32>;
@group(0) @binding(6) var tiles_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> focus_state: array<f32, 4>;
@group(0) @binding(9) var linear_sampler: sampler;

const TILE: u32 = 16u;
const DOF_TAPS: u32 = 48u;
const MOTION_TAPS: u32 = 14u;

// The internal-resolution pixel under output pixel `pixel`.
fn internal_pixel(pixel: vec2<u32>) -> vec2<u32> {
    return min(vec2<u32>((vec2<f32>(pixel) + 0.5) * frame.upscale.xy), vec2<u32>(frame.viewport.xy) - 1u);
}

fn linear_depth(pixel: vec2<u32>) -> f32 {
    return frame.camera_position.w / max(textureLoad(depth_texture, internal_pixel(pixel), 0).r, 1e-9);
}

// Screen-space motion of the pixel since the last frame, in pixels, scaled by the shutter.
fn velocity(pixel: vec2<u32>) -> vec2<f32> {
    let source = internal_pixel(pixel);
    var motion = textureLoad(gbuffer_motion, source, 0).xy;
    if textureLoad(depth_texture, source, 0).r <= 0.0 {
        // Sky: only camera rotation moves it.
        let uv = (vec2<f32>(pixel) + 0.5) * fx.output.zw;
        let dir = view_direction(uv, frame.inv_view_proj, frame.camera_position.xyz);
        let clip = frame.prev_unjittered_view_proj * vec4<f32>(dir, 0.0);
        let previous = vec2<f32>(clip.x, -clip.y) / max(clip.w, 1e-6) * 0.5 + 0.5;
        motion = select(vec2<f32>(0.0), uv - previous, clip.w > 1e-6);
    }
    let v = motion * fx.output.xy * fx.params.x;
    let length_px = length(v);
    return select(v, v * (fx.params.y / length_px), length_px > fx.params.y);
}

// Signed radius of the circle of confusion in pixels (negative in front of the focus plane).
fn circle_of_confusion(depth: f32) -> f32 {
    if fx.params.z <= 0.0 {
        return 0.0;
    }
    let focus = max(focus_state[0], 0.1);
    let f = fx.params.w;
    // Thin lens: the circle's diameter is A f (d - s) / (d (s - f)), with A scaled to pixels on
    // the CPU; the taps below spread over its radius (photo mode's lens samples, photo.rs, are
    // the reference: they blur exactly this much).
    let coc = 0.5 * fx.params.z * f * (depth - focus) / (depth * max(focus - f, 1e-3));
    return clamp(coc, -fx.focus.y, fx.focus.y);
}

@compute @workgroup_size(1)
fn focus() {
    let target_depth = select(linear_depth(vec2<u32>(fx.output.xy * 0.5)), fx.focus.x, fx.focus.x > 0.0);
    let previous = focus_state[0];
    var focused = target_depth;
    if previous > 0.0 && has_flag(frame.flags, FLAG_HISTORY_VALID) && fx.focus.x <= 0.0 {
        // Pull focus smoothly in log distance, like a lens motor.
        let t = 1.0 - exp(-frame.misc.x * 5.0);
        focused = exp(mix(log(previous), log(max(target_depth, 0.05)), t));
    }
    focus_state[0] = min(focused, 1e5);
}

@compute @workgroup_size(8, 8)
fn tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(tiles_out);
    if any(gid.xy >= size) {
        return;
    }
    let full = vec2<u32>(fx.output.xy);
    var best = vec2<f32>(0.0);
    var coc = 0.0;
    for (var y = 0u; y < TILE; y += 2u) {
        for (var x = 0u; x < TILE; x += 2u) {
            let p = min(gid.xy * TILE + vec2<u32>(x, y), full - 1u);
            let v = velocity(p);
            if dot(v, v) > dot(best, best) {
                best = v;
            }
            coc = max(coc, abs(circle_of_confusion(linear_depth(p))));
        }
    }
    textureStore(tiles_out, gid.xy, vec4<f32>(best, coc, 0.0));
}

@compute @workgroup_size(8, 8)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(tiles_out);
    if any(gid.xy >= size) {
        return;
    }
    var best = vec2<f32>(0.0);
    var coc = 0.0;
    for (var i = 0u; i < 9u; i++) {
        let tap = clamp(vec2<i32>(gid.xy) + vec2<i32>(i32(i % 3u) - 1, i32(i / 3u) - 1), vec2<i32>(0), vec2<i32>(size) - 1);
        let t = textureLoad(tiles_in, tap, 0);
        if dot(t.xy, t.xy) > dot(best, best) {
            best = t.xy;
        }
        coc = max(coc, t.z);
    }
    textureStore(tiles_out, gid.xy, vec4<f32>(best, coc, 0.0));
}

@compute @workgroup_size(8, 8)
fn dof(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(fx.output.xy);
    if any(gid.xy >= size) {
        return;
    }
    let center = textureLoad(color_in, gid.xy, 0);
    let radius = textureLoad(tiles_in, gid.xy / TILE, 0).z;
    if radius < 0.5 {
        textureStore(color_out, gid.xy, center);
        return;
    }
    let own = circle_of_confusion(linear_depth(gid.xy));
    let rotation = interleaved_gradient_noise(vec2<f32>(gid.xy), frame.frame_index) * 6.2831853;
    var sum = center.rgb;
    var weight = 1.0;
    for (var i = 0u; i < DOF_TAPS; i++) {
        // Vogel disc of radius `radius`.
        let r = sqrt((f32(i) + 0.5) / f32(DOF_TAPS)) * radius;
        let theta = f32(i) * 2.39996 + rotation;
        let offset = vec2<f32>(cos(theta), sin(theta)) * r;
        let tap = clamp(vec2<i32>(vec2<f32>(gid.xy) + offset), vec2<i32>(0), vec2<i32>(size) - 1);
        let tap_coc = circle_of_confusion(linear_depth(vec2<u32>(tap)));
        // A tap behind this pixel cannot blur over it more than this pixel is blurred itself.
        let reach = select(abs(tap_coc), min(abs(tap_coc), abs(own)), tap_coc > own);
        let w = saturate(reach - r + 1.0) / max(reach * reach, 1.0);
        sum += textureLoad(color_in, tap, 0).rgb * w;
        weight += w;
    }
    textureStore(color_out, gid.xy, vec4<f32>(sum / weight, center.a));
}

fn soft_depth(a: f32, b: f32) -> f32 {
    // 1 when `a` is in front of `b`, fading over 5 % of the depth.
    return saturate(1.0 - (a - b) / (0.05 * min(a, b)));
}

@compute @workgroup_size(8, 8)
fn motion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(fx.output.xy);
    if any(gid.xy >= size) {
        return;
    }
    let center = textureLoad(color_in, gid.xy, 0);
    let dominant = textureLoad(tiles_in, gid.xy / TILE, 0).xy;
    let dominant_length = length(dominant);
    if dominant_length < 0.75 {
        textureStore(color_out, gid.xy, center);
        return;
    }
    let own_v = velocity(gid.xy);
    let own_length = max(length(own_v), 0.5);
    let own_depth = linear_depth(gid.xy);
    let jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), frame.frame_index) - 0.5;
    var weight = 1.0 / own_length;
    var sum = center.rgb * weight;
    for (var i = 0u; i < MOTION_TAPS; i++) {
        let t = ((f32(i) + 0.5 + jitter) / f32(MOTION_TAPS)) - 0.5;
        let offset = dominant * t;
        let distance = length(offset);
        let tap = vec2<u32>(clamp(vec2<i32>(vec2<f32>(gid.xy) + offset), vec2<i32>(0), vec2<i32>(size) - 1));
        let tap_depth = linear_depth(tap);
        let tap_length = max(length(velocity(tap)), 0.5);
        // Foreground taps blur over this pixel if their own motion reaches it; background taps
        // are visible through this pixel's blur.
        let front = soft_depth(tap_depth, own_depth);
        let back = soft_depth(own_depth, tap_depth);
        let w = front * saturate(1.0 - distance / tap_length) / tap_length
            + back * saturate(1.0 - distance / own_length) / own_length
            + 2.0 * min(saturate(1.0 - distance / tap_length), saturate(1.0 - distance / own_length)) / max(tap_length, own_length);
        sum += textureLoad(color_in, tap, 0).rgb * w;
        weight += w;
    }
    textureStore(color_out, gid.xy, vec4<f32>(sum / max(weight, 1e-6), center.a));
}
