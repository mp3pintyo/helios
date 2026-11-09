// Volumetric fog in a camera-aligned froxel volume (after Wronski 2014, "Volumetric fog", and
// Hillaire 2015, as in UE's volumetric fog):
//
//   inject    : per froxel, participating-media density (height fog with drifting noise) and the
//               light it scatters towards the camera: the sun (virtual shadow map visibility,
//               cloud shadows, atmospheric transmittance, Henyey-Greenstein phase), local lights
//               from the clustered light grid, and ambient light from Lumen's radiance cache
//               (not the open sky, so interiors do not glow). Samples are jittered in depth every
//               frame and blended with the reprojected history.
//   integrate : front-to-back accumulation along each froxel column -> in-scattered radiance
//               and transmittance from the camera to every slice
//
// The deferred lighting applies the integrated volume: color * transmittance + in-scattering.

#include "common.wgsl"
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
#include "noise.wgsl"

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
@group(0) @binding(18) var history: texture_3d<f32>;
@group(0) @binding(19) var scattering_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(20) var scattering: texture_3d<f32>;
@group(0) @binding(21) var integrated_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Camera forward axis in world space.
fn camera_forward() -> vec3<f32> {
    return -vec3<f32>(frame.view[0].z, frame.view[1].z, frame.view[2].z);
}

// World position of the point at view depth `depth` along the ray through `uv`.
fn froxel_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let eye = frame.camera_position.xyz;
    let dir = view_direction(uv, frame.inv_view_proj, eye);
    return eye + dir * (depth / max(dot(dir, camera_forward()), 1e-3));
}

fn fog_density(world: vec3<f32>) -> f32 {
    let height = max(world.y - fog.height.x, 0.0);
    var density = fog.media.x * exp(-height * fog.height.y);
    if fog.media.w > 0.0 {
        // Slowly drifting low-frequency noise breaks the fog into wisps and banks.
        let p = world * 0.06 + fog.wind.xyz;
        let n = gradient_noise(p) * 0.6 + gradient_noise(p * 2.7 + 13.1) * 0.4;
        density *= saturate(1.0 + n * 2.0 * fog.media.w);
    }
    return density;
}

// Sun visibility from the virtual shadow map (one filtered tap; froxels are large).
fn sun_visibility(world: vec3<f32>) -> f32 {
    if vsm.enabled == 0u || !has_flag(frame.flags, FLAG_SHADOWS) {
        return 1.0;
    }
    let l = vsm_lookup(world, min(vsm_level(world) + FOG_SHADOW_LEVEL_BIAS, vsm.level_count - 1u));
    if !l.found {
        return 1.0;
    }
    return textureSampleCompareLevel(vsm_atlas, shadow_sampler, l.uv, l.depth);
}

fn cloud_shadow(world: vec3<f32>) -> f32 {
    if clouds.enabled == 0u {
        return 1.0;
    }
    let sun = normalize(frame.sun_direction.xyz);
    let ground = world.xz - sun.xz * (world.y / max(sun.y, 0.05));
    let uv = (ground - clouds.shadow_center.xy) / clouds.shadow_extent + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 1.0;
    }
    return textureSampleLevel(cloud_shadow_map, lut_sampler, uv, 0.0).r;
}

@compute @workgroup_size(4, 4, 4)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(scattering_out);
    if any(gid >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size.xy);
    // Jitter the sample within its slice every frame; the history integrates the slice.
    let jitter = fract(f32(frame.frame_index) * 0.618034 + interleaved_gradient_noise(vec2<f32>(gid.xy), gid.z) * 0.5);
    let depth = fog_slice_depth((f32(gid.z) + jitter) / f32(size.z));
    let world = froxel_world(uv, depth);
    let eye = frame.camera_position.xyz;
    let view_dir = normalize(world - eye);

    let density = fog_density(world);
    var radiance = vec3<f32>(0.0);
    if density > 1e-6 {
        let g = fog.media.y;
        // Sun.
        let sun = normalize(frame.sun_direction.xyz);
        let visibility = sun_visibility(world) * cloud_shadow(world);
        if visibility > 0.0 && sun.y > -0.05 {
            let transmittance = sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(world), sun);
            radiance += frame.sun_illuminance.rgb * transmittance * visibility * henyey_greenstein(dot(sun, view_dir), g);
        }
        // Ambient: the radiance cache sees walls and windows (interiors stay dark); isotropic phase.
        let sky_up = sh_irradiance(sky_sh, vec3<f32>(0.0, 1.0, 0.0)) * frame.sun_illuminance.rgb;
        let up = radiance_cache_irradiance(world, vec3<f32>(0.0, 1.0, 0.0), sky_up);
        let down = radiance_cache_irradiance(world, vec3<f32>(0.0, -1.0, 0.0), sky_up * frame.ground_albedo.rgb);
        radiance += (up + down) * (0.5 / PI) * fog.media.z;
        // Local lights (the clustered grid lists the lights reaching this froxel), shadowed by
        // one tap of their virtual shadow map where they have one: shafts between pillars and
        // under pews. Pages were requested FOG_SHADOW_LEVEL_BIAS mips coarser (vsm.wgsl).
        if frame.light_count > 0u {
            // A froxel stands for its whole cell, but the inverse-square falloff is evaluated at one
            // point: a flame a few centimetres from it would light the entire 16-pixel block. The
            // mean of 1/r² over a ball of radius R around the light is 3/R², so soften the falloff
            // to 1/(r² + R²/3) with R half the cell's extent.
            let cell_width = distance(froxel_world(uv + vec2<f32>(1.0 / f32(size.x), 0.0), depth), world);
            let cell_depth = fog_slice_depth((f32(gid.z) + 1.0) / f32(size.z)) - fog_slice_depth(f32(gid.z) / f32(size.z));
            let cell_radius = 0.5 * max(cell_width, cell_depth);
            let soften = cell_radius * cell_radius / 3.0;
            let base = cluster_index(uv, depth) * CLUSTER_WORDS;
            let count = light_grid[base];
            for (var i = 0u; i < count; i++) {
                let light = lights[light_grid[base + 1u + i]];
                var ls = sample_light(light, world);
                let d2 = ls.distance * ls.distance;
                ls.radiance *= d2 / (d2 + soften);
                var visibility = 1.0;
                let slot = light_shadow_slot(light);
                if slot != INVALID_ID && any(ls.radiance > vec3<f32>(0.0)) {
                    visibility = local_vsm_visibility(slot, world, vec3<f32>(0.0), FOG_SHADOW_LEVEL_BIAS);
                }
                radiance += ls.radiance * visibility * henyey_greenstein(dot(ls.l, view_dir), g * 0.5);
            }
        }
        radiance *= fog.albedo.rgb;
    }
    var current = vec4<f32>(radiance * density, density);

    // Temporal reprojection: where was this point in last frame's volume?
    if has_flag(frame.flags, FLAG_HISTORY_VALID) {
        let clip = frame.prev_unjittered_view_proj * vec4<f32>(world, 1.0);
        if clip.w > 0.0 {
            let previous_uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
            let w = fog_depth_slice(clip.w);
            if all(previous_uv > vec2<f32>(0.0)) && all(previous_uv < vec2<f32>(1.0)) && w < 1.0 {
                let previous = textureSampleLevel(history, lut_sampler, vec3<f32>(previous_uv, w), 0.0);
                current = mix(previous, current, FOG_HISTORY_BLEND);
            }
        }
    }
    textureStore(scattering_out, gid, select(current, vec4<f32>(0.0), any(current != current)));
}

@compute @workgroup_size(8, 8)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(integrated_out);
    if any(gid.xy >= size.xy) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size.xy);
    let eye = frame.camera_position.xyz;
    // Slices are spaced in view depth; the ray through this column is longer by 1/cos.
    let stretch = 1.0 / max(dot(view_direction(uv, frame.inv_view_proj, eye), camera_forward()), 1e-3);
    var inscattered = vec3<f32>(0.0);
    var transmittance = 1.0;
    var near = 0.0;
    for (var z = 0u; z < size.z; z++) {
        let far = fog_slice_depth(f32(z + 1u) / f32(size.z));
        let v = textureLoad(scattering, vec3<u32>(gid.xy, z), 0);
        let extinction = max(v.a, 1e-7);
        let step_transmittance = exp(-extinction * (far - near) * stretch);
        // Energy-conserving integration over the slice (Hillaire 2015).
        inscattered += transmittance * (v.rgb - v.rgb * step_transmittance) / extinction;
        transmittance *= step_transmittance;
        textureStore(integrated_out, vec3<u32>(gid.xy, z), vec4<f32>(inscattered, transmittance));
        near = far;
    }
}
