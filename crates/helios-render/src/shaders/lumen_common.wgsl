// Lumen-style software ray tracing shared code.
//
// * Global distance field: a clipmap of LEVELS 128^3 volumes around the camera, stacked along
//   Z in one rgba8snorm 3D texture. r = signed distance / band (truncated), gba = albedo of the
//   nearest surface (negative = emissive colour). Built from per-mesh SDFs (lumen_sdf.wgsl).
// * Voxel lighting ("surface cache"): radiance leaving the surfaces inside each 64^3 voxel,
//   stacked the same way in an rgba16float volume (rgb premultiplied by coverage a).
// * Radiance cache: world-space irradiance probes (L1 SH) in two clipmap levels.
//
// Includers declare: lumen (uniform Lumen), global_sdf: texture_3d<f32>,
// voxel_lighting: texture_3d<f32>, lumen_sampler: sampler (linear clamp), and (for probes)
// radiance_probes: array<vec4<f32>> (4 per probe).

#include "lumen_types.wgsl"

// ---- Distance field ---------------------------------------------------------------------------

fn sdf_level_extent(level: u32) -> f32 {
    return lumen.sdf_levels[level].w * SDF_RES;
}

fn sdf_uvw(p: vec3<f32>, level: u32, res: f32) -> vec3<f32> {
    let l = lumen.sdf_levels[level];
    let local = (p - l.xyz) / (l.w * SDF_RES); // [0, 1] inside the level
    let texel = clamp(local * res, vec3<f32>(0.5), vec3<f32>(res - 0.5));
    return vec3<f32>(texel.xy / res, (texel.z + f32(level) * res) / (res * f32(lumen.level_count)));
}

fn sdf_inside_level(p: vec3<f32>, level: u32, margin_voxels: f32) -> bool {
    let l = lumen.sdf_levels[level];
    let local = (p - l.xyz) / l.w;
    return all(local > vec3<f32>(margin_voxels)) && all(local < vec3<f32>(SDF_RES - margin_voxels));
}

fn sdf_finest_level(p: vec3<f32>) -> u32 {
    for (var l = 0u; l < lumen.level_count; l++) {
        if sdf_inside_level(p, l, 2.0) {
            return l;
        }
    }
    return lumen.level_count;
}

// Signed world distance at `p` using level `level` (truncated to the band).
fn sdf_distance(p: vec3<f32>, level: u32) -> f32 {
    let s = textureSampleLevel(global_sdf, lumen_sampler, sdf_uvw(p, level, SDF_RES), 0.0).r;
    return s * SDF_BAND_VOXELS * lumen.sdf_levels[level].w;
}

fn sdf_albedo(p: vec3<f32>, level: u32) -> vec3<f32> {
    return textureSampleLevel(global_sdf, lumen_sampler, sdf_uvw(p, level, SDF_RES), 0.0).gba;
}

fn sdf_normal(p: vec3<f32>, level: u32) -> vec3<f32> {
    let e = lumen.sdf_levels[level].w * 0.75;
    let g = vec3<f32>(
        sdf_distance(p + vec3<f32>(e, 0.0, 0.0), level) - sdf_distance(p - vec3<f32>(e, 0.0, 0.0), level),
        sdf_distance(p + vec3<f32>(0.0, e, 0.0), level) - sdf_distance(p - vec3<f32>(0.0, e, 0.0), level),
        sdf_distance(p + vec3<f32>(0.0, 0.0, e), level) - sdf_distance(p - vec3<f32>(0.0, 0.0, e), level),
    );
    return normalize(g + vec3<f32>(0.0, 1e-6, 0.0));
}

struct SdfHit {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    level: u32,
    steps: u32,
}

// Sphere-traces the global distance field from the finest level containing the ray, moving to
// coarser levels as the ray leaves them.
fn trace_sdf(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> SdfHit {
    var h: SdfHit;
    h.hit = false;
    h.t = t_max;
    var t = t_min;
    var level = sdf_finest_level(origin + dir * t);
    for (var i = 0u; i < 96u; i++) {
        if level >= lumen.level_count || t >= t_max {
            break;
        }
        let p = origin + dir * t;
        if !sdf_inside_level(p, level, 1.0) {
            level++;
            continue;
        }
        let voxel = lumen.sdf_levels[level].w;
        let d = sdf_distance(p, level);
        if d < voxel * 0.35 {
            h.hit = true;
            h.t = t;
            h.position = p;
            h.level = level;
            h.steps = i;
            return h;
        }
        t += max(d, voxel * 0.4);
    }
    return h;
}

// How far to trace a shadow ray through the global field alone towards a local light (where the
// near field doesn't reach, see local_light_shadow). The ray stops two voxels short of the
// light: fixtures smaller than a voxel (a candle under its flame, a lamp housing) only exist in
// the field as a voxel-sized blob around the light, and whether the last steps land in it flips
// with sub-pixel jitter - the light would flicker by shadowing itself.
fn light_shadow_distance(light_position: vec3<f32>, distance: f32, light_radius: f32) -> f32 {
    let level = sdf_finest_level(light_position);
    let voxel = select(0.0, lumen.sdf_levels[min(level, lumen.level_count - 1u)].w, level < lumen.level_count);
    return max(distance - max(light_radius, 2.0 * voxel), 0.0);
}

// Soft shadow towards a light by cone-tracing the distance field (Quilez-style penumbra),
// starting 1.5 voxels off the surface at `origin`.
fn sdf_soft_shadow(origin: vec3<f32>, dir: vec3<f32>, t_max: f32, softness: f32) -> f32 {
    let level = sdf_finest_level(origin);
    return sdf_soft_shadow_span(origin, dir, lumen.sdf_levels[min(level, lumen.level_count - 1u)].w * 1.5, t_max, softness);
}

// Soft shadow over the stretch [t_min, t_max] of the ray.
fn sdf_soft_shadow_span(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32, softness: f32) -> f32 {
    var visibility = 1.0;
    var level = sdf_finest_level(origin);
    var t = t_min;
    for (var i = 0u; i < 48u; i++) {
        if level >= lumen.level_count || t >= t_max {
            break;
        }
        let p = origin + dir * t;
        if !sdf_inside_level(p, level, 1.0) {
            level++;
            continue;
        }
        let voxel = lumen.sdf_levels[level].w;
        let d = sdf_distance(p, level);
        visibility = min(visibility, saturate(d / (softness * t) + 0.15));
        if visibility <= 0.01 {
            return 0.0;
        }
        t += max(d, voxel * 0.5);
    }
    return visibility;
}

// ---- Voxel lighting ------------------------------------------------------------------------------

// Radiance leaving the surface near `p` (premultiplied coverage resolved).
fn voxel_radiance(p: vec3<f32>, level: u32) -> vec3<f32> {
    let v = textureSampleLevel(voxel_lighting, lumen_sampler, sdf_uvw(p, level, LIGHT_RES), 0.0);
    return v.rgb / max(v.a, 0.02);
}

// ---- Radiance cache probes -------------------------------------------------------------------------

fn probe_index(level: u32, c: vec3<u32>) -> u32 {
    return ((level * PROBE_RES + c.z) * PROBE_RES + c.y) * PROBE_RES + c.x;
}

// Each probe holds L1 SH radiance in 4 vec4: (L0.rgb, validity), (L1x.rgb, -), (L1y.rgb, -),
// (L1z.rgb, -). Irradiance applies the cosine-lobe convolution (A0 = pi, A1 = 2pi/3).
fn probe_irradiance(probe: u32, n: vec3<f32>) -> vec3<f32> {
    let base = probe * 4u;
    let l0 = radiance_probes[base].rgb;
    let l1 = radiance_probes[base + 1u].rgb * n.x + radiance_probes[base + 2u].rgb * n.y + radiance_probes[base + 3u].rgb * n.z;
    return max(l0 * (0.282095 * 3.14159265) + l1 * (0.488603 * 2.0943951), vec3<f32>(0.0));
}

fn probe_validity(probe: u32) -> f32 {
    return radiance_probes[probe * 4u].w;
}

// Trilinearly interpolated irradiance from the radiance cache (falls back to `fallback`).
fn radiance_cache_irradiance(p: vec3<f32>, n: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    for (var level = 0u; level < PROBE_LEVELS; level++) {
        let l = lumen.probe_levels[level];
        let g = (p - l.xyz) / l.w;
        if any(g < vec3<f32>(0.0)) || any(g >= vec3<f32>(f32(PROBE_RES - 1u))) {
            continue;
        }
        let base = vec3<u32>(floor(g));
        let f = fract(g);
        var sum = vec3<f32>(0.0);
        var weight = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let o = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
            let w = select(1.0 - f.x, f.x, o.x == 1u) * select(1.0 - f.y, f.y, o.y == 1u) * select(1.0 - f.z, f.z, o.z == 1u);
            let probe = probe_index(level, base + o);
            let ww = w * probe_validity(probe);
            sum += probe_irradiance(probe, n) * ww;
            weight += ww;
        }
        if weight > 1e-3 {
            return sum / weight;
        }
    }
    return fallback;
}

