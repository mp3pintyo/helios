// Lumen surface lighting.
//
//   update_voxels : lights the surface inside every voxel of the voxel-lighting clipmap —
//                   sun (with distance-field soft shadows), sky (with distance-field occlusion),
//                   one bounce of indirect light from the radiance cache, and emission.
//                   A quarter of the voxels is refreshed per frame (ping-pong copies the rest).
//   update_probes : refreshes a fraction of the radiance-cache probes per frame by tracing
//                   rays through the distance field into the surface caches (mesh cards, voxel
//                   lighting where no card matches) / sky, projecting the result to L1 SH.
//                   Together they converge to multi-bounce GI.

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "lumen_common.wgsl"
#include "lumen_cards.wgsl"
#include "lights.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lumen: Lumen;
@group(0) @binding(2) var global_sdf: texture_3d<f32>;
@group(0) @binding(3) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(4) var lumen_sampler: sampler;
@group(0) @binding(5) var voxel_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<storage, read_write> radiance_probes: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(8) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(9) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(10) var lut_sampler: sampler;
@group(0) @binding(11) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(70) var<storage, read> card_data: array<vec4<u32>>;
@group(0) @binding(71) var<storage, read> card_pages: array<u32>;
@group(0) @binding(72) var card_capture: texture_2d<u32>;
@group(0) @binding(73) var card_lighting: texture_2d<f32>;
@group(0) @binding(74) var sdf_instance_ids: texture_3d<u32>;

const VOXEL_PHASES: u32 = 4u;
// Fraction of outside light a stained-glass window passes (GLASS_TRANSMISSION in
// translucency.wgsl).
const GLASS_TRANSMISSION: f32 = 0.5;

fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    return sh_irradiance(sky_sh, n) * frame.sun_illuminance.rgb * lumen.sky_intensity;
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    return sample_sky_view(sky_view_lut, lut_sampler, frame.camera_position.xyz, dir, sun) * frame.sun_illuminance.rgb * lumen.sky_intensity;
}

// Ambient occlusion from the distance field along the normal.
fn sdf_occlusion(p: vec3<f32>, n: vec3<f32>, level: u32) -> f32 {
    let voxel = lumen.sdf_levels[level].w;
    var occlusion = 0.0;
    for (var i = 1u; i <= 4u; i++) {
        let t = voxel * f32(i) * 1.5;
        let d = sdf_distance(p + n * t, min(level + i / 3u, lumen.level_count - 1u));
        occlusion += max(t - d, 0.0) / t * (1.0 / f32(i));
    }
    return saturate(1.0 - occlusion * 0.6);
}

@compute @workgroup_size(4, 4, 4)
fn update_voxels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = u32(LIGHT_RES);
    if gid.x >= res || gid.y >= res || gid.z >= res * lumen.level_count {
        return;
    }
    let level = gid.z / res;
    let local = vec3<u32>(gid.xy, gid.z % res);
    let previous = textureLoad(voxel_lighting, gid, 0);
    if ((local.x + local.y + local.z + lumen.voxel_update_phase) % VOXEL_PHASES) != 0u {
        textureStore(voxel_out, gid, previous);
        return;
    }
    let l = lumen.sdf_levels[level];
    let size = l.w * SDF_RES / LIGHT_RES;
    let p = l.xyz + (vec3<f32>(local) + 0.5) * size;
    let d = sdf_distance(p, level);
    if abs(d) > size * 1.2 {
        textureStore(voxel_out, gid, vec4<f32>(0.0));
        return;
    }
    let n = sdf_normal(p, level);
    let surface = p - n * d;
    let raw_albedo = sdf_albedo(surface, level);
    let emissive = any(raw_albedo < vec3<f32>(-0.01));
    let albedo = abs(raw_albedo);
    let sun = normalize(frame.sun_direction.xyz);

    var radiance = vec3<f32>(0.0);
    if emissive {
        // Windows: the daylight outside, tinted by the glass (as in translucency.wgsl).
        radiance = albedo * sh_irradiance(sky_sh, -n) * frame.sun_illuminance.rgb / PI * GLASS_TRANSMISSION;
    } else {
        let origin = surface + n * l.w * 1.5;
        let n_dot_l = saturate(dot(n, sun));
        var direct = vec3<f32>(0.0);
        if n_dot_l > 0.0 {
            let transmittance = sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(surface), sun);
            let shadow = sdf_soft_shadow(origin, sun, lumen.max_trace_distance, 0.05);
            direct = frame.sun_illuminance.rgb * transmittance * n_dot_l * shadow;
        }
        for (var i = 0u; i < frame.light_count; i++) {
            let light = lights[i];
            let to_light = light.position_range.xyz - surface;
            if dot(to_light, to_light) > light.position_range.w * light.position_range.w {
                continue;
            }
            let ls = sample_light(light, surface);
            let nl = saturate(dot(n, ls.l));
            if nl <= 0.0 {
                continue;
            }
            var visibility = 1.0;
            if light.params.z > 0.5 {
                visibility = sdf_soft_shadow(origin, ls.l, light_shadow_distance(light.position_range.xyz, ls.distance, light.color.w), 0.1);
            }
            direct += ls.radiance * nl * visibility;
        }
        let occlusion = sdf_occlusion(surface, n, level);
        let sky = sky_irradiance(n) * occlusion;
        let bounce = radiance_cache_irradiance(origin, n, sky);
        radiance = albedo / PI * (direct + mix(sky, bounce, 0.85));
    }
    let coverage = saturate(1.0 - abs(d) / (size * 1.2));
    let current = vec4<f32>(radiance * coverage, coverage);
    let blended = select(mix(previous, current, 0.5), current, previous.a <= 0.0);
    textureStore(voxel_out, gid, select(blended, vec4<f32>(0.0), any(blended != blended)));
}

// Fibonacci sphere direction `i` of `n`, rotated by a per-frame angle.
fn fibonacci_direction(i: u32, n: u32, rotation: f32) -> vec3<f32> {
    let golden = 2.39996323;
    let z = 1.0 - (f32(i) + 0.5) / f32(n) * 2.0;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = f32(i) * golden + rotation;
    return vec3<f32>(r * cos(phi), z, r * sin(phi));
}

const PROBE_RAYS: u32 = 32u;

// Per ray: L0, L1x, L1y, L1z radiance, summed over the workgroup.
var<workgroup> probe_sh: array<array<vec3<f32>, 4>, PROBE_RAYS>;

// One workgroup per refreshed probe, one thread per ray: a thread per probe tracing all its
// rays in turn left most of the GPU idle (a few thousand long, serial threads).
@compute @workgroup_size(32)
fn update_probes(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let total = PROBE_RES * PROBE_RES * PROBE_RES * PROBE_LEVELS;
    let fraction = lumen.probe_update_fraction;
    let probe = wid.x * fraction + (lumen.frame % fraction);
    if probe >= total {
        return;
    }
    let level = probe / (PROBE_RES * PROBE_RES * PROBE_RES);
    let local = probe % (PROBE_RES * PROBE_RES * PROBE_RES);
    let c = vec3<u32>(local % PROBE_RES, (local / PROBE_RES) % PROBE_RES, local / (PROBE_RES * PROBE_RES));
    let l = lumen.probe_levels[level];
    let position = l.xyz + vec3<f32>(c) * l.w;

    let sdf_level = sdf_finest_level(position);
    var validity = 1.0;
    if sdf_level < lumen.level_count && sdf_distance(position, sdf_level) < lumen.sdf_levels[sdf_level].w * 0.5 {
        validity = 0.0; // buried in geometry: would leak light through walls
    }

    let rotation = f32(pcg_hash(lumen.frame * 7919u + probe) & 0xffffu) / 65536.0 * 6.2831853;
    let dir = fibonacci_direction(i, PROBE_RAYS, rotation);
    let hit = trace_sdf(position, dir, l.w * 0.1, lumen.max_trace_distance);
    var radiance = sky_radiance(dir);
    if hit.hit {
        let lighting_voxel = lumen.sdf_levels[hit.level].w * SDF_RES / LIGHT_RES;
        radiance = voxel_radiance(hit.position - dir * lighting_voxel * 0.6, hit.level);
        var n = sdf_normal(hit.position, hit.level);
        n = select(n, -n, dot(n, dir) > 0.0);
        radiance = card_field_radiance(radiance, hit.position, n, hit.level, lumen.sdf_levels[hit.level].w * 0.5);
    }
    probe_sh[i] = array<vec3<f32>, 4>(radiance * 0.282095, radiance * (0.488603 * dir.x), radiance * (0.488603 * dir.y), radiance * (0.488603 * dir.z));
    workgroupBarrier();
    for (var stride = PROBE_RAYS / 2u; stride > 0u; stride >>= 1u) {
        if i < stride {
            for (var k = 0u; k < 4u; k++) {
                probe_sh[i][k] += probe_sh[i + stride][k];
            }
        }
        workgroupBarrier();
    }
    if i != 0u {
        return;
    }
    let l0 = probe_sh[0][0];
    let l1x = probe_sh[0][1];
    let l1y = probe_sh[0][2];
    let l1z = probe_sh[0][3];
    let norm = 4.0 * PI / f32(PROBE_RAYS);
    let base = probe * 4u;
    let previous_valid = radiance_probes[base].w > 0.5;
    let blend = select(1.0, 0.3, previous_valid && validity > 0.5);
    radiance_probes[base] = vec4<f32>(mix(radiance_probes[base].rgb, l0 * norm, blend), validity);
    radiance_probes[base + 1u] = vec4<f32>(mix(radiance_probes[base + 1u].rgb, l1x * norm, blend), 0.0);
    radiance_probes[base + 2u] = vec4<f32>(mix(radiance_probes[base + 2u].rgb, l1y * norm, blend), 0.0);
    radiance_probes[base + 3u] = vec4<f32>(mix(radiance_probes[base + 3u].rgb, l1z * norm, blend), 0.0);
}
