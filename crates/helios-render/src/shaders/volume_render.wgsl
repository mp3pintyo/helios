// Rendering of heterogeneous volumes (volumes.rs; UE's Heterogeneous Volumes):
//
//   light_cache : per emitter, a voxel grid at half the simulation resolution caching the light
//                 the medium scatters (UE's lighting cache): the sun's transmittance (a march
//                 through the volume towards the sun, times the virtual shadow map and cloud
//                 shadows) and the isotropically scattered ambient (Lumen's radiance cache, the
//                 sky occluded by the volume above) and local lights (with their shadow maps).
//   march       : per pixel of the volumes' screen rectangle, front to back through every
//                 visible emitter's box up to the opaque depth: density and temperature from the
//                 grid, lighting from the cache with the Henyey-Greenstein phase for the sun,
//                 blackbody emission from temperature, energy-conserving integration. The ray is
//                 split at the nearest translucent surface: the part behind it is composited
//                 before the translucency pass (glass and water show it), the part in front
//                 after. Each part is fogged at its opacity-weighted depth. Step offsets are
//                 jittered per pixel and frame; strides grow through clear air and step back
//                 where the medium begins (empty-space skipping).
//   resolve     : temporal accumulation of the march, reprojected with the depth of the volume
//                 the pixel sees and clamped to the current neighbourhood (no ghosting of the
//                 moving flames); it removes the jitter noise before TSR sees the image.

#include "common.wgsl"
#include "blackbody.wgsl"
#include "media_common.wgsl"
#include "avboit_common.wgsl"
#include "avboit_read.wgsl"
#include "avboit_write.wgsl"

const MAX_VOLUMES: u32 = 4u;
const MAX_STEPS: u32 = 192u;
// Weight of the multiple-scattering octave of sunlight.
const MS_WEIGHT: f32 = 0.5;
// Longest stride through clear air, in base steps.
const EMPTY_STRIDE: f32 = 4.0;
// Extinction (per metre) below which the medium counts as clear air. Low: thin smoke lit by a
// fire next to it still shows, and a higher cut-off draws the cut's iso-surface as visible steps.
const CLEAR_AIR: f32 = 2e-5;
const HISTORY_BLEND: f32 = 0.25;
// Debug view: particle overdraw and volume march steps (oit.wgsl draws it).
const DEBUG_VOLUMES: u32 = 21u;

// One emitter's look (volumes.rs, `VolumeShading`).
struct VolumeShading {
    origin: vec4<f32>,      // xyz box minimum (world), w: extinction per metre at density 1
    size: vec4<f32>,        // xyz box extent, w: phase anisotropy g
    albedo: vec4<f32>,      // rgb single-scattering albedo, w: emission scale
    emission: vec4<f32>,    // x kelvin at temperature 0, y at 1, z march step (m), w ambient scale
    lights: vec4<u32>,      // indices of the local lights that reach the box
    more_lights: vec4<u32>,
    params: vec4<u32>,      // x: light count, yzw: grid resolution
}

struct VolumeMarch {
    rect: vec4<u32>,          // pixels this frame: min xy, max xy (exclusive)
    previous_rect: vec4<u32>,
    count: vec4<u32>,         // x: visible volumes
    volumes: array<VolumeShading, 4>,
}

// light_cache
@group(0) @binding(70) var<uniform> shading: VolumeShading;
@group(0) @binding(71) var cache_state: texture_3d<f32>;
@group(0) @binding(72) var cache_out: texture_storage_3d<rgba16float, write>;
// march, resolve
@group(0) @binding(80) var<uniform> march_info: VolumeMarch;
@group(0) @binding(81) var state0: texture_3d<f32>;
@group(0) @binding(82) var state1: texture_3d<f32>;
@group(0) @binding(83) var state2: texture_3d<f32>;
@group(0) @binding(84) var state3: texture_3d<f32>;
@group(0) @binding(85) var cache0: texture_3d<f32>;
@group(0) @binding(86) var cache1: texture_3d<f32>;
@group(0) @binding(87) var cache2: texture_3d<f32>;
@group(0) @binding(88) var cache3: texture_3d<f32>;
@group(0) @binding(89) var scene_depth: texture_2d<f32>;
@group(0) @binding(140) var translucent_depth: texture_2d<f32>;
@group(0) @binding(141) var front_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(142) var back_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(143) var front_raw: texture_2d<f32>;
@group(0) @binding(144) var back_raw: texture_2d<f32>;
@group(0) @binding(145) var front_history: texture_2d<f32>;
@group(0) @binding(146) var back_history: texture_2d<f32>;

// `v`, or `fallback` where it is not a finite number.
fn finite(v: vec4<f32>, fallback: vec4<f32>) -> vec4<f32> {
    return select(v, fallback, any((v != v) | (abs(v) > vec4<f32>(6e4))));
}

// Entry and exit distances of the ray through the box [lo, lo + size] (t0 > t1: missed).
fn ray_box(origin: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, size: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let a = (lo - origin) * inv;
    let b = (lo + size - origin) * inv;
    let near = min(a, b);
    let far = max(a, b);
    return vec2<f32>(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
}

// ---- Lighting cache -------------------------------------------------------------------------

fn cache_density(p: vec3<f32>) -> f32 {
    return textureSampleLevel(cache_state, lut_sampler, (p - shading.origin.xyz) / shading.size.xyz, 0.0).x;
}

// Transmittance of the medium from `p` to the box's boundary along `dir`.
fn self_transmittance(p: vec3<f32>, dir: vec3<f32>, steps: u32) -> f32 {
    let exit = ray_box(p, dir, shading.origin.xyz, shading.size.xyz).y;
    let dt = max(exit, 0.0) / f32(steps);
    var optical_depth = 0.0;
    for (var i = 0u; i < steps; i++) {
        optical_depth += cache_density(p + dir * ((f32(i) + 0.5) * dt));
    }
    return exp(-optical_depth * dt * shading.origin.w);
}

fn shading_light(i: u32) -> u32 {
    if i < 4u {
        return shading.lights[i];
    }
    return shading.more_lights[i - 4u];
}

@compute @workgroup_size(4, 4, 4)
fn light_cache(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(cache_out);
    if any(gid >= dims) {
        return;
    }
    let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
    let p = shading.origin.xyz + uvw * shading.size.xyz;
    let sun = normalize(frame.sun_direction.xyz);
    var sun_transmittance = 0.0;
    if sun.y > -0.05 {
        sun_transmittance = self_transmittance(p, sun, 16u) * media_sun_visibility(p);
    }
    // Ambient light from above passes through the volume above this voxel.
    var radiance = media_ambient(p, self_transmittance(p, vec3<f32>(0.0, 1.0, 0.0), 8u)) * shading.emission.w;
    // Local lights, isotropic phase; a voxel stands for its cell.
    let cell = shading.size.xyz / vec3<f32>(dims);
    let soften = dot(cell, cell) / 12.0;
    for (var i = 0u; i < min(shading.params.x, 8u); i++) {
        let ls = media_local_light(lights[shading_light(i)], p, soften);
        radiance += ls.radiance / (4.0 * PI);
    }
    textureStore(cache_out, gid, finite(vec4<f32>(radiance, sun_transmittance), vec4<f32>(0.0)));
}

// ---- March ----------------------------------------------------------------------------------

fn sample_state(i: u32, uvw: vec3<f32>) -> vec4<f32> {
    switch i {
        case 0u: { return textureSampleLevel(state0, lut_sampler, uvw, 0.0); }
        case 1u: { return textureSampleLevel(state1, lut_sampler, uvw, 0.0); }
        case 2u: { return textureSampleLevel(state2, lut_sampler, uvw, 0.0); }
        default: { return textureSampleLevel(state3, lut_sampler, uvw, 0.0); }
    }
}

fn sample_cache(i: u32, uvw: vec3<f32>) -> vec4<f32> {
    switch i {
        case 0u: { return textureSampleLevel(cache0, lut_sampler, uvw, 0.0); }
        case 1u: { return textureSampleLevel(cache1, lut_sampler, uvw, 0.0); }
        case 2u: { return textureSampleLevel(cache2, lut_sampler, uvw, 0.0); }
        default: { return textureSampleLevel(cache3, lut_sampler, uvw, 0.0); }
    }
}

// Distance along the ray `dir` to a reverse-Z depth (`far` for the sky / nothing).
fn ray_distance(depth: f32, dir: vec3<f32>, far: f32) -> f32 {
    if depth <= 0.0 {
        return far;
    }
    return frame.camera_position.w / depth / max(dot(dir, camera_forward()), 1e-3);
}

// Radiance (rgb) and transmittance (a) of one part of the ray, plus the opacity-weighted
// distance and the weight, for fogging it.
struct Segment {
    light: vec4<f32>,
    weighted_distance: f32,
    weight: f32,
}

fn new_segment() -> Segment {
    return Segment(vec4<f32>(0.0, 0.0, 0.0, 1.0), 0.0, 0.0);
}

fn accumulate(s: ptr<function, Segment>, radiance: vec3<f32>, step_transmittance: f32, t: f32) {
    let opacity = (*s).light.a * (1.0 - step_transmittance);
    (*s).light = vec4<f32>((*s).light.rgb + (*s).light.a * radiance, (*s).light.a * step_transmittance);
    (*s).weighted_distance += opacity * t;
    (*s).weight += opacity;
}

// The segment as seen through the fog in front of it.
fn fogged(s: Segment, eye: vec3<f32>, dir: vec3<f32>, uv: vec2<f32>) -> vec4<f32> {
    if s.weight <= 1e-5 {
        return s.light;
    }
    let t = s.weighted_distance / s.weight;
    let f = media_fog(eye + dir * t, uv, dir, t * dot(dir, camera_forward()));
    return vec4<f32>(s.light.rgb * f.transmittance + (1.0 - s.light.a) * f.inscatter, s.light.a);
}

@compute @workgroup_size(8, 8)
fn march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy + march_info.rect.xy;
    if any(pixel >= march_info.rect.zw) {
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let eye = frame.camera_position.xyz;
    let dir = view_direction(uv, frame.inv_view_proj, eye);
    let t_opaque = ray_distance(textureLoad(scene_depth, pixel, 0).r, dir, 1e9);
    let t_split = min(ray_distance(textureLoad(translucent_depth, pixel, 0).r, dir, 1e9), t_opaque);
    let noise = fract(interleaved_gradient_noise(vec2<f32>(pixel), frame.frame_index) + f32(frame.frame_index % 16u) * 0.618034);
    var rng = pcg_hash(pixel.x ^ pcg_hash(pixel.y ^ pcg_hash(frame.frame_index)));
    let sun = normalize(frame.sun_direction.xyz);
    let cos_sun = dot(dir, sun);
    // Adaptive voxel-based OIT (avboit.rs): the surfaces and particles in front of a step weigh
    // its light (the march attenuates by its own medium exactly). One grid column per pixel,
    // picked at random with the bilinear weights: the temporal resolve averages the choice into
    // the bilinear filter.
    let forward = dot(dir, camera_forward());
    var grid_column = vec2<i32>(0);
    var grid_bounds = vec2<f32>(0.0);
    var split_surfaces = vec3<f32>(1.0);
    if avboit_enabled() {
        let c = (vec2<f32>(pixel) + 0.5) / f32(avboit.grid.w) - 0.5;
        rng = pcg_hash(rng);
        let r = vec2<f32>(f32(rng & 0xffffu), f32(rng >> 16u)) / 65536.0;
        let pick = vec2<i32>(floor(c)) + select(vec2<i32>(0), vec2<i32>(1), r < fract(c));
        grid_column = clamp(pick, vec2<i32>(0), vec2<i32>(avboit.grid.xy) - 1);
        grid_bounds = avboit_bounds(grid_column);
        if t_split < t_opaque {
            split_surfaces = max(avboit_surface_at(grid_column, grid_bounds, t_split * forward), vec3<f32>(1e-3));
        }
    }

    // Visible volumes by entry_t distance.
    var order = array<u32, 4>(0u, 1u, 2u, 3u);
    var entry_t = array<f32, 4>(1e9, 1e9, 1e9, 1e9);
    let count = min(march_info.count.x, MAX_VOLUMES);
    for (var i = 0u; i < count; i++) {
        let v = march_info.volumes[i];
        let hit = ray_box(eye, dir, v.origin.xyz, v.size.xyz);
        if hit.x < hit.y && hit.y > 0.0 && max(hit.x, 0.0) < t_opaque {
            entry_t[i] = max(hit.x, 0.0);
        }
    }
    for (var i = 1u; i < count; i++) {
        var j = i;
        while j > 0u && entry_t[order[j - 1u]] > entry_t[order[j]] {
            let swap = order[j];
            order[j] = order[j - 1u];
            order[j - 1u] = swap;
            j -= 1u;
        }
    }

    var front = new_segment();
    var back = new_segment();
    var steps = 0u;
    for (var k = 0u; k < count; k++) {
        let i = order[k];
        if entry_t[i] >= 1e9 {
            break;
        }
        let v = march_info.volumes[i];
        let hit = ray_box(eye, dir, v.origin.xyz, v.size.xyz);
        let t0 = max(hit.x, 0.0);
        let t1 = min(hit.y, t_opaque);
        let n = min(u32(ceil((t1 - t0) / v.emission.z)), MAX_STEPS);
        let dt = (t1 - t0) / f32(max(n, 1u));
        let sun_light = media_sun_light(v.origin.xyz + 0.5 * v.size.xyz);
        let phase = henyey_greenstein(cos_sun, v.size.w);
        let phase_ms = henyey_greenstein(cos_sun, 0.5 * v.size.w);
        var t = t0 + noise * dt;
        var stride = dt;
        for (var s = 0u; s < MAX_STEPS && t < t1; s++) {
            // Stochastic filtering: a random offset of up to half a voxel per sample turns the
            // trilinear filter's voxel-sized staircases at sharp smoke fronts into a smooth
            // falloff (the noise averages out in the temporal resolve).
            rng = pcg_hash(rng);
            let offset = vec3<f32>(f32(rng & 1023u), f32((rng >> 10u) & 1023u), f32((rng >> 20u) & 1023u)) / 1023.0 - 0.5;
            let uvw = (eye + dir * t - v.origin.xyz) / v.size.xyz + offset / vec3<f32>(v.params.yzw);
            let state = sample_state(i, uvw);
            steps += 1u;
            let extinction = state.x * v.origin.w;
            let hot = smoothstep(0.02, 0.15, state.y) * v.albedo.w;
            if extinction < CLEAR_AIR && hot <= 0.0 {
                // Clear air: lengthen the stride (empty-space skipping, as the clouds do).
                t += stride;
                stride = min(stride * 2.0, dt * EMPTY_STRIDE);
                continue;
            }
            if stride > dt {
                // Entered the medium with a long stride: go back and walk in with the base step.
                t += dt - stride;
                stride = dt;
                continue;
            }
            var radiance = blackbody(mix(v.emission.x, v.emission.y, saturate(state.y))) * hot;
            var step_transmittance = 1.0;
            if extinction >= CLEAR_AIR {
                let cache = sample_cache(i, uvw);
                // Sun: single scattering plus a second, softer and less attenuated octave that
                // stands in for multiple scattering (Wrenninge 2013), without which optically thick
                // smoke turns black away from the sun.
                let sun_scattering = phase * cache.a + MS_WEIGHT * phase_ms * sqrt(cache.a);
                let scattered = v.albedo.rgb * (cache.rgb + sun_light * sun_scattering);
                step_transmittance = exp(-extinction * dt);
                // Energy-conserving integration over the step (Hillaire 2015): emission and
                // in-scattering are both attenuated within it.
                radiance = (scattered * extinction + radiance) * (1.0 - step_transmittance) / extinction;
            } else {
                radiance *= dt;
            }
            if avboit_enabled() {
                // Behind the nearest refracting surface only what lies between it and the step
                // counts (the front part's composite dims the rest).
                var weight = avboit_surface_at(grid_column, grid_bounds, t * forward);
                if t >= t_split {
                    weight = min(weight / split_surfaces, vec3<f32>(1.0));
                }
                radiance *= weight;
            }
            if t < t_split {
                accumulate(&front, radiance, step_transmittance, t);
            } else {
                accumulate(&back, radiance, step_transmittance, t);
            }
            if front.light.a < 0.003 || (t >= t_split && back.light.a < 0.003) {
                break;
            }
            t += dt;
        }
        if front.light.a < 0.003 {
            break;
        }
    }
    var f = fogged(front, eye, dir, uv);
    var b = fogged(back, eye, dir, uv);
    if frame.debug_view == DEBUG_VOLUMES {
        f = vec4<f32>(f32(steps), 0.0, 0.0, 1.0);
        b = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // A non-finite value would never leave the temporal history.
    textureStore(front_out, pixel, finite(f, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
    textureStore(back_out, pixel, finite(b, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

// ---- AVBOIT grid (avboit.rs) -----------------------------------------------------------------
// One thread per grid column over the volumes' rectangle, along the column's centre ray.

// Index of the grid column of thread `gid` (offset to the rectangle), or none outside it.
fn grid_column_of(gid: vec2<u32>) -> vec2<u32> {
    return gid + march_info.rect.xy / avboit.grid.w;
}

fn grid_column_valid(column: vec2<u32>) -> bool {
    return all(column < avboit.grid.xy) && all(column * avboit.grid.w < march_info.rect.zw);
}

// The column's centre ray: direction and the opaque surface's distance.
fn grid_ray(column: vec2<u32>) -> vec4<f32> {
    let size = vec2<u32>(frame.viewport.xy);
    let pixel = min(column * avboit.grid.w + avboit.grid.w / 2u, size - 1u);
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let dir = view_direction(uv, frame.inv_view_proj, frame.camera_position.xyz);
    return vec4<f32>(dir, ray_distance(textureLoad(scene_depth, pixel, 0).r, dir, 1e9));
}

// The volumes' optical depth per cell of the columns the surfaces and particles bounded (4
// samples per box and cell; the cells' third word), and in front of the cells (the column's).
fn volume_optical_depth(eye: vec3<f32>, dir: vec3<f32>, ta: f32, tb: f32) -> f32 {
    var optical_depth = 0.0;
    for (var i = 0u; i < min(march_info.count.x, MAX_VOLUMES); i++) {
        let v = march_info.volumes[i];
        let hit = ray_box(eye, dir, v.origin.xyz, v.size.xyz);
        let t0 = max(hit.x, ta);
        let t1 = min(hit.y, tb);
        if t0 >= t1 {
            continue;
        }
        let dt = (t1 - t0) * 0.25;
        for (var s = 0u; s < 4u; s++) {
            let uvw = (eye + dir * (t0 + (f32(s) + 0.5) * dt) - v.origin.xyz) / v.size.xyz;
            optical_depth += sample_state(i, uvw).x * v.origin.w * dt;
        }
    }
    return optical_depth;
}

@compute @workgroup_size(8, 8)
fn avboit_volume_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let column = grid_column_of(gid.xy);
    if !avboit_enabled() || !grid_column_valid(column) {
        return;
    }
    let index = avboit_column_index(column);
    let b = avboit_column_bounds(index);
    if b.y <= b.x {
        return;
    }
    let ray = grid_ray(column);
    let eye = frame.camera_position.xyz;
    let forward = max(dot(ray.xyz, camera_forward()), 1e-3);
    let slices = avboit.grid.z;
    avboit_set_volume_in_front(index, volume_optical_depth(eye, ray.xyz, 0.0, min(b.x / forward, ray.w)));
    for (var k = 0u; k < slices; k++) {
        let ta = b.x * pow(b.y / b.x, f32(k) / f32(slices)) / forward;
        let tb = min(b.x * pow(b.y / b.x, f32(k + 1u) / f32(slices)) / forward, ray.w);
        atomicStore(&avboit_grid[avboit_cell_word(index, k) + 2u], bitcast<u32>(volume_optical_depth(eye, ray.xyz, ta, tb)));
    }
}

// ---- Temporal resolve ------------------------------------------------------------------------

// Distance to the middle of the nearest volume the ray sees (for reprojection), 0 for none.
fn volume_distance(eye: vec3<f32>, dir: vec3<f32>, t_opaque: f32) -> f32 {
    var nearest = 1e9;
    var result = 0.0;
    for (var i = 0u; i < min(march_info.count.x, MAX_VOLUMES); i++) {
        let v = march_info.volumes[i];
        let hit = ray_box(eye, dir, v.origin.xyz, v.size.xyz);
        let t0 = max(hit.x, 0.0);
        let t1 = min(hit.y, t_opaque);
        if t0 < t1 && t0 < nearest {
            nearest = t0;
            result = 0.5 * (t0 + t1);
        }
    }
    return result;
}

struct Neighbourhood {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn neighbourhood(raw: texture_2d<f32>, pixel: vec2<i32>) -> Neighbourhood {
    var n = Neighbourhood(vec4<f32>(1e9), vec4<f32>(-1e9));
    let lo = vec2<i32>(march_info.rect.xy);
    let hi = vec2<i32>(march_info.rect.zw) - 1;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = textureLoad(raw, clamp(pixel + vec2<i32>(x, y), lo, hi), 0);
            n.lo = min(n.lo, c);
            n.hi = max(n.hi, c);
        }
    }
    return n;
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy + march_info.rect.xy;
    if any(pixel >= march_info.rect.zw) {
        return;
    }
    let p = vec2<i32>(pixel);
    var f = textureLoad(front_raw, p, 0);
    var b = textureLoad(back_raw, p, 0);
    if frame.debug_view != DEBUG_VOLUMES && has_flag(frame.flags, FLAG_HISTORY_VALID) {
        let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
        let eye = frame.camera_position.xyz;
        let dir = view_direction(uv, frame.inv_view_proj, eye);
        let t = volume_distance(eye, dir, ray_distance(textureLoad(scene_depth, pixel, 0).r, dir, 1e9));
        let clip = frame.prev_unjittered_view_proj * vec4<f32>(eye + dir * t, 1.0);
        if t > 0.0 && clip.w > 0.0 {
            let previous_uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
            let previous = previous_uv * frame.prev_viewport.xy;
            let r = vec4<f32>(march_info.previous_rect);
            if all(previous >= r.xy + 0.5) && all(previous <= r.zw - 0.5) {
                let huv = viewport_uv(previous_uv, frame.prev_viewport.xy, textureDimensions(front_history));
                let nf = neighbourhood(front_raw, p);
                let nb = neighbourhood(back_raw, p);
                let hf = clamp(textureSampleLevel(front_history, lut_sampler, huv, 0.0), nf.lo, nf.hi);
                let hb = clamp(textureSampleLevel(back_history, lut_sampler, huv, 0.0), nb.lo, nb.hi);
                f = mix(hf, f, HISTORY_BLEND);
                b = mix(hb, b, HISTORY_BLEND);
            }
        }
    }
    // A non-finite value would never leave the temporal history.
    textureStore(front_out, pixel, finite(f, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
    textureStore(back_out, pixel, finite(b, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}
