// Volumetric clouds (after Schneider 2015, "The real-time volumetric cloudscapes of Horizon
// Zero Dawn", and Hillaire 2016, "Physically based sky, atmosphere and cloud rendering").
//
//   build_shape / build_detail : tileable Perlin-Worley / Worley noise volumes (once)
//   march    : per quarter-resolution pixel that sees sky, ray-march the cloud shell around the
//              planet; sun light is marched through the cloud (self-shadowing, multiple-
//              scattering octaves), ambient comes from the sky's SH irradiance
//   temporal : reprojects last frame's clouds by their distance and blends (the march jitters
//              its start every frame, so a few frames integrate many more steps)
//   shadow   : a camera-centred map of the sun's transmittance through the clouds on the ground,
//              used by the deferred lighting for moving cloud shadows
//
// Units: kilometres for cloud space (matching atmosphere.wgsl), metres for the world.

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "clouds_common.wgsl"


@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> clouds: Clouds;
@group(0) @binding(2) var shape_noise: texture_3d<f32>;
@group(0) @binding(3) var detail_noise: texture_3d<f32>;
@group(0) @binding(4) var noise_sampler: sampler;
@group(0) @binding(5) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(8) var depth_texture: texture_2d<f32>;
@group(0) @binding(9) var raw_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var distance_out: texture_storage_2d<r32float, write>;
@group(0) @binding(11) var raw_clouds: texture_2d<f32>;
@group(0) @binding(12) var raw_distance: texture_2d<f32>;
@group(0) @binding(13) var history_clouds: texture_2d<f32>;
@group(0) @binding(14) var history_distance: texture_2d<f32>;
@group(0) @binding(15) var clouds_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(16) var resolved_distance_out: texture_storage_2d<r32float, write>;
@group(0) @binding(17) var shadow_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(20) var shape_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(21) var detail_out: texture_storage_3d<rgba8unorm, write>;

const SHAPE_SIZE: u32 = 128u;
const DETAIL_SIZE: u32 = 32u;
/// Upper bound on density samples per pixel.
const MAX_STEPS: u32 = 128u;
/// Fine steps taken after the last non-empty sample before returning to coarse steps.
const FINE_STEPS: u32 = 6u;
const LIGHT_STEPS: u32 = 6u;
/// Rays are not marched further than this through the shell (km).
const MAX_MARCH: f32 = 40.0;
/// Distant clouds fade into the sky over this distance (km).
const FADE_DISTANCE: f32 = 45.0;

// ---- Tileable noise ----------------------------------------------------------------------------

fn hash33(p: vec3<u32>) -> vec3<f32> {
    var v = p * vec3<u32>(1664525u, 1013904223u, 2654435761u) + vec3<u32>(1013904223u, 1664525u, 22695477u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3<f32>(v & vec3<u32>(0xffffffu)) / f32(0xffffff);
}

fn wrap_cell(c: vec3<f32>, period: f32) -> vec3<u32> {
    return vec3<u32>(c - floor(c / period) * period);
}

// Inverted cellular noise with `period` cells per tile: 1 at feature points, 0 far from them.
fn worley(uvw: vec3<f32>, period: f32) -> f32 {
    let p = uvw * period;
    let cell = floor(p);
    let f = p - cell;
    var nearest = 1e9;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3<f32>(f32(x), f32(y), f32(z));
                let feature = o + hash33(wrap_cell(cell + o, period));
                let d = feature - f;
                nearest = min(nearest, dot(d, d));
            }
        }
    }
    return 1.0 - saturate(sqrt(nearest));
}

fn worley_fbm(uvw: vec3<f32>, period: f32) -> f32 {
    return worley(uvw, period) * 0.625 + worley(uvw, period * 2.0) * 0.25 + worley(uvw, period * 4.0) * 0.125;
}

fn gradient(c: vec3<f32>, period: f32) -> vec3<f32> {
    return normalize(hash33(wrap_cell(c, period)) * 2.0 - 1.0);
}

// Periodic gradient (Perlin) noise in [-1, 1].
fn perlin(uvw: vec3<f32>, period: f32) -> f32 {
    let p = uvw * period;
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var result = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let o = vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
        let w = mix(1.0 - u, u, o);
        result += dot(gradient(i + o, period), f - o) * w.x * w.y * w.z;
    }
    return result * 1.6;
}

fn perlin_fbm(uvw: vec3<f32>, period: f32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var p = period;
    for (var octave = 0u; octave < 5u; octave++) {
        sum += perlin(uvw, p) * amplitude;
        amplitude *= 0.5;
        p *= 2.0;
    }
    return sum;
}

fn remap(v: f32, lo: f32, hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + (v - lo) * (new_hi - new_lo) / max(hi - lo, 1e-5);
}

// r: Perlin-Worley (billowy base shape), gba: Worley fBm at three frequencies (erosion of r).
@compute @workgroup_size(4, 4, 4)
fn build_shape(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= vec3<u32>(SHAPE_SIZE)) {
        return;
    }
    let uvw = (vec3<f32>(gid) + 0.5) / f32(SHAPE_SIZE);
    let worley_low = worley_fbm(uvw, 4.0);
    let perlin_low = saturate(perlin_fbm(uvw, 4.0) * 0.5 + 0.5);
    let perlin_worley = saturate(remap(perlin_low, 0.0, 1.0, worley_low, 1.0));
    textureStore(shape_out, gid, vec4<f32>(perlin_worley, worley_low, worley_fbm(uvw, 8.0), worley_fbm(uvw, 16.0)));
}

@compute @workgroup_size(4, 4, 4)
fn build_detail(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= vec3<u32>(DETAIL_SIZE)) {
        return;
    }
    let uvw = (vec3<f32>(gid) + 0.5) / f32(DETAIL_SIZE);
    textureStore(detail_out, gid, vec4<f32>(worley_fbm(uvw, 2.0), worley_fbm(uvw, 4.0), worley_fbm(uvw, 8.0), 1.0));
}

// ---- Density ------------------------------------------------------------------------------------

// Rotates horizontal coordinates. Each noise layer is sampled in its own rotated frame: the
// tiling volumes repeat along the world axes, and with aligned layers the repeats line up into
// rows that converge towards the horizon.
fn rotate_xz(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

// Extinction (per km) at `p` (km; x/z horizontal) with height fraction `h` in the layer.
fn cloud_density(p: vec3<f32>, h: f32, detailed: bool) -> f32 {
    let q = p + vec3<f32>(clouds.wind.x, 0.0, clouds.wind.y);
    // Weather: where clouds form at all. Low-frequency Perlin-Worley thresholded by the coverage
    // gives separate cloud fields with clear sky between them.
    let w = textureSampleLevel(shape_noise, noise_sampler, rotate_xz(vec3<f32>(q.x, 0.31 / clouds.weather_scale, q.z), -0.31) * clouds.weather_scale, 0.0);
    let field = smoothstep(0.3, 0.75, w.r * 0.7 + w.g * 0.3);
    // Local coverage: the threshold the billowy shape noise has to exceed.
    let local_coverage = clouds.coverage * mix(0.35, 1.25, field);
    // Cumulus profile: a soft base, rounded tops; denser weather builds taller clouds.
    let top = mix(0.35, 1.0, field);
    let profile = saturate(remap(h, 0.0, 0.08, 0.0, 1.0)) * saturate(remap(h, top * 0.4, top, 1.0, 0.0));
    if profile <= 0.0 {
        return 0.0;
    }
    // Warp the shape domain by the weather so neighbouring tiles differ.
    let warped = rotate_xz(vec3<f32>(q.x, h * clouds.thickness, q.z), 0.52) + vec3<f32>(w.b, 0.0, w.a) * 2.0;
    let s = textureSampleLevel(shape_noise, noise_sampler, warped * clouds.shape_scale, 0.0);
    let low = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
    // Perlin-Worley eroded by its own low-frequency Worley fBm lands in a narrow band (about
    // 0.76-0.90, 5th-95th percentile); stretch it to 0-1 so that coverage reads linearly.
    let shape = saturate(remap(remap(s.r, low - 1.0, 1.0, 0.0, 1.0), 0.74, 0.92, 0.0, 1.0));
    // The weather decides how much of the billowy shape survives.
    let base = saturate(remap(shape * profile, 1.0 - local_coverage, 1.0, 0.0, 1.0));
    if base <= 0.0 || !detailed {
        return base * clouds.density;
    }
    // Erode the edges with high-frequency detail: wispy at the bottom, billowy at the top.
    let d = textureSampleLevel(detail_noise, noise_sampler, rotate_xz(vec3<f32>(q.x, h * clouds.thickness, q.z), 1.17) * clouds.detail_scale, 0.0);
    let detail = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    let modifier = mix(detail, 1.0 - detail, saturate(h * 6.0));
    return saturate(remap(base, modifier * 0.4, 1.0, 0.0, 1.0)) * clouds.density;
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Strong forward lobe (silver lining) plus a weak back lobe.
fn cloud_phase(cos_theta: f32, scale: f32) -> f32 {
    return mix(henyey_greenstein(cos_theta, -0.2 * scale), henyey_greenstein(cos_theta, 0.8 * scale), 0.7);
}

struct Shell {
    inner: f32,
    outer: f32,
}

fn shell() -> Shell {
    return Shell(GROUND_RADIUS + clouds.base, GROUND_RADIUS + clouds.base + clouds.thickness);
}

// Planet-centred position (km) of a world position (metres), keeping its horizontal offset.
fn planet_position(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(world.x * 0.001, GROUND_RADIUS + max(world.y * 0.001, 0.0) + 0.05, world.z * 0.001);
}

fn height_fraction(p: vec3<f32>) -> f32 {
    return (length(p) - (GROUND_RADIUS + clouds.base)) / clouds.thickness;
}

// Optical depth (per km · km) from `p` towards the sun through the cloud layer.
fn sun_optical_depth(p: vec3<f32>, sun: vec3<f32>, steps: u32, length_km: f32) -> f32 {
    var depth = 0.0;
    var t = 0.0;
    for (var i = 0u; i < steps; i++) {
        // Steps grow quadratically: detail near the sample, coarse far away.
        let next = length_km * pow(f32(i + 1u) / f32(steps), 2.0);
        let dt = next - t;
        let q = p + sun * (t + 0.5 * dt);
        let h = height_fraction(q);
        if h > 1.0 {
            break;
        }
        depth += cloud_density(q, h, i < 2u) * dt;
        t = next;
    }
    return depth;
}

// ---- March --------------------------------------------------------------------------------------

fn checker_active(pixel: vec2<u32>) -> bool {
    return !has_flag(frame.flags, FLAG_HISTORY_VALID) || ((pixel.x ^ pixel.y ^ frame.frame_index) & 1u) == 0u;
}

@compute @workgroup_size(8, 8)
fn march(@builtin(global_invocation_id) id: vec3<u32>) {
    // The corner of the targets the internal resolution covers (they are allocated for the
    // largest one).
    let size = viewport_extent(frame.viewport, 2u);
    // Checkerboard: each frame marches half the pixels; the temporal pass fills the other half
    // from history. Threads map to the marched pixels only (whole warps of the unused right half
    // of the dispatch retire at once). Everything is marched when there is no history.
    var gid = id;
    if has_flag(frame.flags, FLAG_HISTORY_VALID) {
        gid.x = id.x * 2u + ((id.y ^ frame.frame_index) & 1u);
    }
    if any(gid.xy >= size) {
        return;
    }
    let none = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if clouds.enabled == 0u {
        textureStore(raw_out, gid.xy, none);
        textureStore(distance_out, gid.xy, vec4<f32>(0.0));
        return;
    }
    // Skip blocks of the full-resolution image that show no sky (interiors, dense canopies).
    // Rounded up: with an odd viewport side (e.g. 603 rows at 67 % of 900), `full / size` would
    // be 1 and texel j would test pixel row j instead of 2j - wrong blocks, so clouds vanished
    // in sky that only shows lower down (a cloudless band beside foreground objects).
    let full = vec2<u32>(frame.viewport.xy);
    let block = max((full + size - 1u) / size, vec2<u32>(1u));
    let last = block - 1u;
    let corners = array<vec2<u32>, 5>(vec2<u32>(0u), vec2<u32>(last.x, 0u), vec2<u32>(0u, last.y), last, block / 2u);
    var sky = false;
    for (var i = 0u; i < 5u; i++) {
        let p = min(gid.xy * block + corners[i], full - 1u);
        sky = sky || textureLoad(depth_texture, p, 0).r <= 0.0;
    }
    if !sky {
        textureStore(raw_out, gid.xy, none);
        textureStore(distance_out, gid.xy, vec4<f32>(0.0));
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / (frame.viewport.xy * 0.5);
    let eye = frame.camera_position.xyz;
    let dir = view_direction(uv, frame.inv_view_proj, eye);
    let origin = planet_position(eye);
    let layer = shell();
    if ray_sphere(origin, dir, GROUND_RADIUS) > 0.0 {
        textureStore(raw_out, gid.xy, none);
        textureStore(distance_out, gid.xy, vec4<f32>(0.0));
        return;
    }
    let t_enter = max(ray_sphere(origin, dir, layer.inner), 0.0);
    let t_exit = ray_sphere(origin, dir, layer.outer);
    if t_exit <= t_enter {
        textureStore(raw_out, gid.xy, none);
        textureStore(distance_out, gid.xy, vec4<f32>(0.0));
        return;
    }
    let t_end = min(t_exit, t_enter + MAX_MARCH);
    // Adaptive stepping: long steps through clear air on the cheap density, then back up one step
    // and walk short steps while inside (or near) a cloud. Short steps matter: cloud extinction is
    // tens per km, so a long step would turn every cloud into an opaque, unlit slab.
    let coarse = clamp((t_end - t_enter) / 48.0, 0.12, 0.6);
    let fine = coarse * 0.25;
    let jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), frame.frame_index);

    let sun = normalize(frame.sun_direction.xyz);
    let cos_theta = dot(dir, sun);
    let sun_above = smoothstep(-0.05, 0.02, sun.y);
    let up = normalize(origin);
    // Ambient: sky irradiance from above (tops) and a darker bounce from the ground (bottoms).
    let sky_ambient = sh_irradiance(sky_sh, up) * frame.sun_illuminance.rgb / PI;
    let ground_ambient = sky_ambient * frame.ground_albedo.rgb * 0.5 + sky_ambient * 0.25;

    var transmittance = 1.0;
    var radiance = vec3<f32>(0.0);
    var distance_sum = 0.0;
    var weight_sum = 0.0;
    var t = t_enter + jitter * coarse;
    var fine_left = 0u;
    for (var i = 0u; i < MAX_STEPS; i++) {
        if t >= t_end {
            break;
        }
        let p = origin + dir * t;
        let h = height_fraction(p);
        if fine_left == 0u {
            if cloud_density(p, h, false) <= 0.0 {
                t += coarse;
                continue;
            }
            // Entered a cloud: step back and refine.
            t = max(t - coarse + fine, t_enter);
            fine_left = FINE_STEPS;
            continue;
        }
        let sigma = cloud_density(p, h, true);
        fine_left -= 1u;
        if sigma > 1e-3 {
            fine_left = FINE_STEPS;
            let optical = sun_optical_depth(p, sun, LIGHT_STEPS, clouds.thickness * 0.8);
            let sun_light = frame.sun_illuminance.rgb * sample_transmittance(transmittance_lut, lut_sampler, p, sun) * sun_above;
            // Multiple scattering as a sum of octaves with weaker extinction and a flatter phase.
            var scattered = 0.0;
            var a = 1.0;
            var b = 1.0;
            var c = 1.0;
            for (var octave = 0u; octave < 4u; octave++) {
                scattered += a * cloud_phase(cos_theta, c) * exp(-b * optical);
                a *= 0.6;
                b *= 0.45;
                c *= 0.5;
            }
            // Powder: the first few metres of a cloud facing the light scatter less towards it.
            let powder = 1.0 - exp(-2.0 * sigma * fine);
            let ambient = mix(ground_ambient, sky_ambient, saturate(h * 1.4));
            let source = (sun_light * scattered * mix(1.0, powder, 0.5) + ambient) * sigma;
            let step_transmittance = exp(-sigma * fine);
            // Energy-conserving integration of the step (Hillaire 2015).
            radiance += transmittance * (source - source * step_transmittance) / sigma;
            let absorbed = transmittance * (1.0 - step_transmittance);
            distance_sum += t * absorbed;
            weight_sum += absorbed;
            transmittance *= step_transmittance;
            if transmittance < 0.01 {
                break;
            }
        }
        t += fine;
    }
    let mean_distance = select(t_enter, distance_sum / max(weight_sum, 1e-6), weight_sum > 1e-4);
    // Aerial perspective: far clouds dissolve into the sky behind them.
    let fade = exp(-mean_distance / FADE_DISTANCE);
    textureStore(raw_out, gid.xy, vec4<f32>(radiance * fade, mix(1.0, transmittance, fade)));
    textureStore(distance_out, gid.xy, vec4<f32>(mean_distance));
}

// ---- Temporal accumulation ---------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= size) {
        return;
    }
    let current = textureLoad(raw_clouds, gid.xy, 0);
    let distance = textureLoad(raw_distance, gid.xy, 0).r;
    var lo = current;
    var hi = current;
    for (var i = 0u; i < 9u; i++) {
        let tap = clamp(vec2<i32>(gid.xy) + vec2<i32>(i32(i % 3u) - 1, i32(i / 3u) - 1), vec2<i32>(0), vec2<i32>(size) - 1);
        let v = textureLoad(raw_clouds, tap, 0);
        lo = min(lo, v);
        hi = max(hi, v);
    }
    let marched = checker_active(gid.xy);
    var result = current;
    var resolved_distance = distance;
    if has_flag(frame.flags, FLAG_HISTORY_VALID) && distance > 0.0 {
        // Reproject the point at the clouds' mean distance (the camera moves metres, they are km away).
        let uv = (vec2<f32>(gid.xy) + 0.5) / (frame.viewport.xy * 0.5);
        let eye = frame.camera_position.xyz;
        let world = eye + view_direction(uv, frame.inv_view_proj, eye) * distance * 1000.0;
        let clip = frame.prev_unjittered_view_proj * vec4<f32>(world, 1.0);
        let previous_uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
        if clip.w > 0.0 && all(previous_uv > vec2<f32>(0.0)) && all(previous_uv < vec2<f32>(1.0)) {
            // History was written at the previous frame's internal resolution.
            let p = min(vec2<u32>(previous_uv * frame.prev_viewport.xy * 0.5), viewport_extent(frame.prev_viewport, 2u) - 1u);
            let history = textureLoad(history_clouds, p, 0);
            let extent = (hi - lo) * 0.5;
            // Pixels not marched this frame keep (clamped) history; marched ones blend in.
            let blend = select(0.0, 0.2, marched);
            result = mix(clamp(history, lo - extent, hi + extent), current, blend);
            resolved_distance = mix(textureLoad(history_distance, p, 0).r, distance, blend);
        }
    }
    textureStore(clouds_out, gid.xy, select(result, current, any(result != result)));
    textureStore(resolved_distance_out, gid.xy, vec4<f32>(resolved_distance));
}

// ---- Shadow map ---------------------------------------------------------------------------------

// Sun transmittance through the clouds for ground points: texel (x, z) holds the value for the
// world point (x, 0, z); a point at height y looks up (x, z) - sun.xz * y / sun.y.
@compute @workgroup_size(8, 8)
fn shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(shadow_out);
    if any(gid.xy >= size) {
        return;
    }
    let sun = normalize(frame.sun_direction.xyz);
    if clouds.enabled == 0u || sun.y <= 0.02 {
        textureStore(shadow_out, gid.xy, vec4<f32>(1.0));
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size) - 0.5;
    let ground = vec3<f32>(clouds.shadow_center.x + uv.x * clouds.shadow_extent, 0.0, clouds.shadow_center.y + uv.y * clouds.shadow_extent);
    let origin = planet_position(ground);
    let layer = shell();
    let t_enter = ray_sphere(origin, sun, layer.inner);
    let t_exit = ray_sphere(origin, sun, layer.outer);
    if t_enter < 0.0 || t_exit <= t_enter {
        textureStore(shadow_out, gid.xy, vec4<f32>(1.0));
        return;
    }
    let steps = 16u;
    let dt = (min(t_exit, t_enter + MAX_MARCH) - t_enter) / f32(steps);
    var depth = 0.0;
    for (var i = 0u; i < steps; i++) {
        let p = origin + sun * (t_enter + (f32(i) + 0.5) * dt);
        depth += cloud_density(p, height_fraction(p), false) * dt;
    }
    let visibility = mix(1.0, exp(-depth), clouds.shadow_strength);
    textureStore(shadow_out, gid.xy, vec4<f32>(visibility));
}
