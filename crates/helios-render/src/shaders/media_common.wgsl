// Lighting and fogging of participating media that live outside the froxel fog: the lighting
// cache of heterogeneous volumes (volume_render.wgsl) and GPU particles (particles.wgsl). Uses
// the same light sources as volumetric fog (fog.wgsl), at the same binding numbers (media.rs
// binds them): the sun through the virtual shadow map and cloud shadows, Lumen's radiance cache
// for ambient light, the clustered local lights with their virtual shadow maps; plus the
// integrated fog volume and aerial perspective in front of the medium.

#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "vsm_local_common.wgsl"
#include "vsm_local_sample.wgsl"
#include "lumen_common.wgsl"
#include "lights.wgsl"
#include "clouds_common.wgsl"
#include "fog_common.wgsl"
#include "glass_shadow.wgsl"
#include "volume_shadow_sample.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> fog: Fog;
@group(0) @binding(2) var<uniform> vsm: Vsm;
@group(0) @binding(3) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(4) var vsm_atlas: texture_depth_2d;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(0) @binding(6) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(7) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(8) var<storage, read> light_grid: array<u32>;
@group(0) @binding(9) var<uniform> clouds: Clouds;
@group(0) @binding(10) var cloud_shadow_map: texture_2d<f32>;
@group(0) @binding(11) var lut_sampler: sampler;
@group(0) @binding(12) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(13) var<uniform> lumen: Lumen;
@group(0) @binding(14) var global_sdf: texture_3d<f32>;
@group(0) @binding(15) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(16) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(17) var lumen_sampler: sampler;
@group(0) @binding(18) var fog_volume: texture_3d<f32>;
@group(0) @binding(19) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;

// Closer to a local light than this (m), media are not shadowed by its shadow map.
const MEDIA_SHADOW_MIN_DISTANCE: f32 = 0.1;

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

fn camera_forward() -> vec3<f32> {
    return -vec3<f32>(frame.view[0].z, frame.view[1].z, frame.view[2].z);
}

// Sun radiance arriving at `p` through the atmosphere (no shadows).
fn media_sun_light(p: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    if sun.y < -0.05 {
        return vec3<f32>(0.0);
    }
    return frame.sun_illuminance.rgb * sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(p), sun);
}

// Sun visibility at `p`: one filtered tap of the virtual shadow map at the level volumetric fog
// requests pages for (media are soft; their pages are resident wherever the fog's are) times the
// cloud shadow and the heterogeneous volumes' shadows (all but the volume box `skip`).
fn media_sun_visibility(p: vec3<f32>, skip: u32) -> f32 {
    var visibility = 1.0;
    if vsm.enabled != 0u && has_flag(frame.flags, FLAG_SHADOWS) {
        let l = vsm_lookup(p, min(vsm_level(p) + FOG_SHADOW_LEVEL_BIAS, vsm.level_count - 1u));
        if l.found {
            visibility = textureSampleCompareLevel(vsm_atlas, shadow_sampler, l.uv, l.depth);
        }
    }
    if clouds.enabled != 0u {
        let sun = normalize(frame.sun_direction.xyz);
        let ground = p.xz - sun.xz * (p.y / max(sun.y, 0.05));
        let uv = (ground - clouds.shadow_center.xy) / clouds.shadow_extent + 0.5;
        if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            visibility *= textureSampleLevel(cloud_shadow_map, lut_sampler, uv, 0.0).r;
        }
    }
    return visibility * volume_sun_transmittance(p, skip);
}

// The sun at `p` through the glass panes nearer the sun, per channel (one tap of the glass map).
fn media_sun_glass(p: vec3<f32>) -> vec3<f32> {
    return glass_transmission(p, lut_sampler);
}

// Average radiance around `p` from Lumen's radiance cache (the sky where it has none), split
// into the light from above, scaled by `up_transmittance` (the medium's own occlusion of the sky
// when known), and from below. A medium scattering it isotropically re-emits this radiance.
fn media_ambient(p: vec3<f32>, up_transmittance: f32) -> vec3<f32> {
    let sky_up = sh_irradiance(sky_sh, vec3<f32>(0.0, 1.0, 0.0)) * frame.sun_illuminance.rgb;
    var up = sky_up;
    var down = sky_up * frame.ground_albedo.rgb;
    if has_flag(frame.flags, FLAG_GI) && lumen.sdf_enabled != 0u {
        up = radiance_cache_irradiance(p, vec3<f32>(0.0, 1.0, 0.0), up) * frame.gi_intensity;
        down = radiance_cache_irradiance(p, vec3<f32>(0.0, -1.0, 0.0), down) * frame.gi_intensity;
    }
    return (up * up_transmittance + down) * (0.5 / PI);
}

// Radiance a local light scatters towards the viewer at `p` (incident radiance times the
// phase function `phase`), shadowed by one tap of its virtual shadow map. `soften` widens the
// inverse-square falloff (r² + soften) for samples that stand for a cell around them, so a
// light inside a fire does not blow up the cells next to it.
fn media_local_light(light: GpuLight, p: vec3<f32>, soften: f32) -> LightSample {
    var ls = sample_light(light, p);
    let d2 = ls.distance * ls.distance;
    ls.radiance *= d2 / (d2 + soften);
    let slot = light_shadow_slot(light);
    // Not at the light itself: the shadow map's projection degenerates there (a fire's light sits
    // inside its own volume and among its sparks).
    if slot != INVALID_ID && any(ls.radiance > vec3<f32>(0.0)) && has_flag(frame.flags, FLAG_SHADOWS) && ls.distance > MEDIA_SHADOW_MIN_DISTANCE {
        let visibility = local_vsm_visibility(slot, p, vec3<f32>(0.0), FOG_SHADOW_LEVEL_BIAS);
        ls.radiance *= select(visibility, 1.0, visibility != visibility);
    }
    return ls;
}

// The clustered local lights at an on-screen point `p` (screen `uv`, view depth `linear_depth`),
// scattered towards the viewer (`view_dir`: from the camera) with anisotropy `g`.
fn media_clustered_lights(p: vec3<f32>, uv: vec2<f32>, linear_depth: f32, view_dir: vec3<f32>, g: f32, soften: f32) -> vec3<f32> {
    if frame.light_count == 0u {
        return vec3<f32>(0.0);
    }
    var radiance = vec3<f32>(0.0);
    let base = cluster_index(uv, linear_depth) * CLUSTER_WORDS;
    let count = light_grid[base];
    for (var i = 0u; i < count; i++) {
        let index = light_grid[base + 1u + i];
        let ls = media_local_light(lights[index], p, soften);
        let volumes = volume_light_transmittance(index, lights[index].position_range.xyz, p);
        radiance += ls.radiance * volumes * henyey_greenstein(dot(ls.l, view_dir), g);
    }
    return radiance;
}

// Fog between the camera and a point: seen through it, radiance L becomes
// L * transmittance + inscatter (aerial perspective, then the integrated froxel fog, as the
// translucency pass applies them).
struct MediaFog {
    transmittance: vec3<f32>,
    inscatter: vec3<f32>,
}

fn media_fog(p: vec3<f32>, uv: vec2<f32>, view_dir: vec3<f32>, linear_depth: f32) -> MediaFog {
    var f: MediaFog;
    f.transmittance = vec3<f32>(1.0);
    f.inscatter = vec3<f32>(0.0);
    if has_flag(frame.flags, FLAG_FOG) {
        let eye = frame.camera_position.xyz;
        let distance_km = distance(p, eye) * 0.001 * frame.misc.y;
        let height_km = max(0.5 * (eye.y + p.y), 0.0) * 0.001;
        let t = exp(-sample_medium(height_km).extinction * distance_km);
        let sun = normalize(frame.sun_direction.xyz);
        let sky = sample_sky_view(sky_view_lut, lut_sampler, eye, haze_direction(view_dir), sun) * frame.sun_illuminance.rgb;
        f.transmittance = t;
        f.inscatter = sky * (vec3<f32>(1.0) - t);
    }
    if fog.enabled != 0u {
        let w = fog_depth_slice(linear_depth) - 0.5 / f32(textureDimensions(fog_volume).z);
        let v = textureSampleLevel(fog_volume, lut_sampler, vec3<f32>(uv, saturate(w)), 0.0);
        f.transmittance *= v.a;
        f.inscatter = f.inscatter * v.a + v.rgb;
    }
    return f;
}
