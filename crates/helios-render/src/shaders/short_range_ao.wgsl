// Short-range ambient occlusion from the depth buffer, as UE's Lumen applies its screen-space
// "short range AO" on top of the screen probes: occlusion by geometry within SRAO_RADIUS of a
// pixel that the probes (one per 16² pixels, half resolution) and the distance fields (no
// alpha-masked or thin foliage) cannot see — the ground under a grass tuft or a fern, the
// crease where a rock or a log meets the ground, a stalk's base. It darkens indirect light only
// (like material AO).
//
// Computed at half resolution (as UE's `r.Lumen.ScreenProbeGather.ShortRangeAO.DownsampleFactor`
// 2): one texel per 2×2 pixels, at the even pixel the GI's half-resolution grid uses, into
// `FrameTargets::short_range_ao`; the lighting pass upsamples it with the GI's depth- and
// normal-aware weights (lighting.wgsl `sample_half`), so each pixel blends up to four
// differently rotated estimates and TSR averages the rest over frames.
//
// Per texel SRAO_DIRECTIONS screen directions (rotated per texel and frame) are marched in
// SRAO_STEPS steps within the radius, keeping each direction's horizon: the highest elevation of
// its samples above the pixel's tangent plane, sin(elevation) = cos(v, n) for a sample at
// offset v, faded out towards the radius. A horizon at elevation e hides sin(e) of that
// direction's cosine-weighted half-slice of the hemisphere (horizon-based AO, as HBAO / GTAO);
// their mean is the occlusion. The depth buffer has no thickness: samples beyond the radius (a
// blade in front of a distant ground) are ignored, so thin occluders cast no halo.

#include "common.wgsl"
#include "vsm_common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(13) var<uniform> vsm: Vsm;

const SRAO_DIRECTIONS: u32 = 4u;
// 3 steps (4 at full resolution before): each pixel blends up to four texels' estimates.
const SRAO_STEPS: u32 = 3u;
// World radius (m) and its screen cap (pixels at 900 rows).
const SRAO_RADIUS: f32 = 0.4;
const SRAO_MAX_PIXELS: f32 = 64.0;
// Tangent-plane bias (sine of the elevation: surfaces curving away do not occlude themselves)
// and strength.
const SRAO_BIAS: f32 = 0.15;
const SRAO_STRENGTH: f32 = 1.0;

fn short_range_ao(pixel: vec2<u32>, size: vec2<u32>, world: vec3<f32>, n: vec3<f32>, noise: f32) -> f32 {
    let distance_to_eye = distance(world, frame.camera_position.xyz);
    let radius_pixels = min(SRAO_RADIUS / max(distance_to_eye * vsm.pixel_angle, 1e-6), SRAO_MAX_PIXELS * f32(size.y) / 900.0);
    if radius_pixels < 1.5 {
        return 1.0;
    }
    let last = vec2<f32>(size) - 1.0;
    let r2 = SRAO_RADIUS * SRAO_RADIUS;
    let jitter = fract(noise * 7.0);
    var occlusion = 0.0;
    for (var i = 0u; i < SRAO_DIRECTIONS; i++) {
        let angle = (f32(i) + noise) * (6.2831853 / f32(SRAO_DIRECTIONS));
        let dir = vec2<f32>(cos(angle), sin(angle));
        var horizon = 0.0;
        for (var j = 0u; j < SRAO_STEPS; j++) {
            // Denser near the pixel (contact), jittered per texel and frame.
            let t = (f32(j) + jitter) / f32(SRAO_STEPS);
            let q = clamp(vec2<f32>(pixel) + 0.5 + dir * (radius_pixels * t * t + 1.5), vec2<f32>(0.5), last + 0.5);
            let q_depth = textureLoad(depth_texture, vec2<u32>(q), 0).r;
            if q_depth <= 0.0 {
                continue;
            }
            let v = world_from_depth(q * frame.viewport.zw, q_depth, frame.inv_view_proj) - world;
            let d2 = dot(v, v);
            if d2 >= r2 || d2 < 1e-8 {
                continue;
            }
            horizon = max(horizon, (dot(v, n) * inverseSqrt(d2) - SRAO_BIAS) * (1.0 - d2 / r2));
        }
        occlusion += horizon;
    }
    let mean = occlusion / (f32(SRAO_DIRECTIONS) * (1.0 - SRAO_BIAS));
    return saturate(1.0 - SRAO_STRENGTH * mean);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= half_size) {
        return;
    }
    let size = vec2<u32>(frame.viewport.xy);
    // The even pixel of the 2×2 block, as the GI's half-resolution grid (lighting.wgsl).
    let pixel = gid.xy * 2u;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    var ao = 1.0;
    if depth > 0.0 {
        let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
        let world = world_from_depth(uv, depth, frame.inv_view_proj);
        let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
        let noise = interleaved_gradient_noise(vec2<f32>(gid.xy), frame.frame_index);
        ao = short_range_ao(pixel, size, world, n, noise);
    }
    textureStore(output, gid.xy, vec4<f32>(ao, 0.0, 0.0, 0.0));
}
