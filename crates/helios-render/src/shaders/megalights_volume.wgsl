// MegaLights for volumes (megalights_volume.rs; UE 5.8's MegaLights volume for volumetric fog and
// lit translucency): per froxel of a camera-aligned volume (the fog's grid), MLV_SAMPLES lights
// are drawn from the froxel's light list in proportion to their unshadowed radiance at a point
// jittered through the froxel every frame (weighted resampling over at most MLV_CANDIDATES,
// longer lists thinned to an unbiased stratified subset as in megalights_sample.wgsl), and each
// gets a shadow ray to a random point of its disk - shadow-map ray marching where the light has
// a virtual shadow map (pages FOG_SHADOW_LEVEL_BIAS mips coarser, as the fog requests them),
// distance fields otherwise (megalights_rays.wgsl). No screen trace: a froxel is no surface.
// The estimates are blended with the reprojected history (the jitter integrates the froxel):
//   fog_light : in-scattered light towards the camera (Henyey-Greenstein phase with half the
//               fog's anisotropy, as fog.wgsl's light loop), which fog.wgsl adds instead of its
//               loop over the clustered lights;
//   sh0, sh1  : the shadowed incident light (megalights_volume_common.wgsl), which translucent
//               surfaces shade with instead of their clustered loop.

#include "common.wgsl"
#include "lights.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "vsm_local_common.wgsl"
#include "vsm_local_sample.wgsl"
#include "fog_common.wgsl"
#include "megalights_rays.wgsl"
#include "megalights_volume_common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> ml_volume: MlVolume;
@group(0) @binding(2) var<uniform> fog: Fog;
@group(0) @binding(3) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(4) var<storage, read> light_grid: array<u32>;
@group(0) @binding(5) var<uniform> lumen: Lumen;
@group(0) @binding(6) var global_sdf: texture_3d<f32>;
@group(0) @binding(7) var lumen_sampler: sampler;
@group(0) @binding(8) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(9) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(10) var<uniform> vsm: Vsm;
@group(0) @binding(11) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(12) var vsm_atlas: texture_depth_2d;
@group(0) @binding(13) var<uniform> local_vsm: LocalVsm;
@group(0) @binding(14) var history_fog: texture_3d<f32>;
@group(0) @binding(15) var history_sh0: texture_3d<f32>;
@group(0) @binding(16) var history_sh1: texture_3d<f32>;
@group(0) @binding(17) var out_fog: texture_storage_3d<rgba16float, write>;
@group(0) @binding(18) var out_sh0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(19) var out_sh1: texture_storage_3d<rgba16float, write>;
// Declared for helpers of the included files that this entry point doesn't call; not part of
// the bind group layout.
@group(0) @binding(20) var shadow_sampler: sampler_comparison;
@group(0) @binding(21) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(22) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(23) var ml_volume_sh0: texture_3d<f32>;
@group(0) @binding(24) var ml_volume_sh1: texture_3d<f32>;

// One sample a frame: the history (HISTORY_BLEND 0.1) averages ~10; two cost 0.15 ms more at
// 1600x900 for an image within 60 dB of one.
const MLV_SAMPLES: u32 = 1u;
const MLV_CANDIDATES: u32 = 16u;

fn mlv_next(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state >> 8u) * (1.0 / 16777216.0);
}

fn mlv_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

fn camera_forward() -> vec3<f32> {
    return -vec3<f32>(frame.view[0].z, frame.view[1].z, frame.view[2].z);
}

fn froxel_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let eye = frame.camera_position.xyz;
    let dir = view_direction(uv, frame.inv_view_proj, eye);
    return eye + dir * (depth / max(dot(dir, camera_forward()), 1e-3));
}

@compute @workgroup_size(4, 4, 4)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_fog);
    if any(gid >= size) {
        return;
    }
    var rng = pcg_hash(gid.x ^ pcg_hash(gid.y ^ pcg_hash(gid.z ^ pcg_hash(frame.frame_index * 0x9e3779b9u))));
    let jitter = vec3<f32>(mlv_next(&rng), mlv_next(&rng), mlv_next(&rng));
    let uv = (vec2<f32>(gid.xy) + jitter.xy) / vec2<f32>(size.xy);
    let depth = ml_volume_depth((f32(gid.z) + jitter.z) / f32(size.z));
    let world = froxel_world(uv, depth);
    let view_dir = normalize(world - frame.camera_position.xyz);

    var fog_light = vec3<f32>(0.0);
    var sh0 = vec3<f32>(0.0);
    var sh1 = vec3<f32>(0.0);
    let list = cluster_index(uv, depth) * CLUSTER_WORDS;
    let count = select(0u, light_grid[list], frame.light_count > 0u);
    if count > 0u {
        // A froxel stands for its whole cell: soften the falloff by the mean of 1/r² over a ball
        // of half the cell's extent (as fog.wgsl), so a flame next to the sample point does not
        // light the whole cell.
        let cell_width = distance(froxel_world(uv + vec2<f32>(1.0 / f32(size.x), 0.0), depth), world);
        let cell_depth = ml_volume_depth((f32(gid.z) + 1.0) / f32(size.z)) - ml_volume_depth(f32(gid.z) / f32(size.z));
        let cell_radius = 0.5 * max(cell_width, cell_depth);
        let soften = cell_radius * cell_radius / 3.0;

        let candidates = min(count, MLV_CANDIDATES);
        let stride = f32(count) / f32(candidates);
        let offset = mlv_next(&rng);
        var weights: array<f32, MLV_CANDIDATES>;
        var ids: array<u32, MLV_CANDIDATES>;
        var total = 0.0;
        for (var i = 0u; i < candidates; i++) {
            let entry = select(i, min(u32((f32(i) + offset) * stride), count - 1u), count > MLV_CANDIDATES);
            let index = light_grid[list + 1u + entry];
            let ls = sample_light(lights[index], world);
            let d2 = ls.distance * ls.distance;
            weights[i] = luminance(ls.radiance) * d2 / (d2 + soften) * stride;
            ids[i] = index;
            total += weights[i];
        }
        if total > 0.0 {
            let start = mlv_next(&rng);
            var i = 0u;
            var running = weights[0];
            let g = fog.media.y * 0.5;
            for (var k = 0u; k < MLV_SAMPLES; k++) {
                let u = (f32(k) + start) / f32(MLV_SAMPLES) * total;
                while running < u && i + 1u < candidates {
                    i++;
                    running += weights[i];
                }
                if weights[i] <= 0.0 {
                    continue;
                }
                let light = lights[ids[i]];
                let ls = sample_light(light, world);
                let d2 = ls.distance * ls.distance;
                var visibility = 1.0;
                if light.params.z > 0.5 {
                    let center = light.position_range.xyz;
                    let radius = light.color.w;
                    let tangent = local_tangent(ls.l);
                    let disk = sqrt(mlv_next(&rng)) * radius;
                    let angle = 6.2831853 * mlv_next(&rng);
                    let point = center + (tangent * cos(angle) + cross(ls.l, tangent) * sin(angle)) * disk;
                    let slot = light_shadow_slot(light);
                    if slot != INVALID_ID {
                        visibility = vsm_ray(slot, world, vec3<f32>(0.0), point, radius, mlv_next(&rng), FOG_SHADOW_LEVEL_BIAS);
                    } else if lumen.sdf_enabled != 0u {
                        visibility = sdf_ray(world, point, center, radius);
                    }
                }
                // The pick's unbiased contribution: radiance / (its weight / stride) x total.
                let c = ls.radiance * (d2 / (d2 + soften)) * visibility * (total * stride / weights[i] / f32(MLV_SAMPLES));
                fog_light += c * mlv_phase(dot(ls.l, view_dir), g);
                sh0 += c;
                sh1 += luminance(c) * ls.l;
            }
        }
    }
    var current_fog = vec4<f32>(fog_light, 0.0);
    var current_sh0 = vec4<f32>(sh0, 0.0);
    var current_sh1 = vec4<f32>(sh1, 0.0);

    // Temporal accumulation: where was this point in last frame's volume?
    if has_flag(frame.flags, FLAG_HISTORY_VALID) {
        let clip = frame.prev_unjittered_view_proj * vec4<f32>(world, 1.0);
        if clip.w > 0.0 {
            let previous_uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
            let w = ml_volume_slice(clip.w);
            if all(previous_uv > vec2<f32>(0.0)) && all(previous_uv < vec2<f32>(1.0)) && w < 1.0 {
                let p = vec3<f32>(previous_uv, w);
                let blend = ml_volume.range.z;
                current_fog = mix(textureSampleLevel(history_fog, lumen_sampler, p, 0.0), current_fog, blend);
                current_sh0 = mix(textureSampleLevel(history_sh0, lumen_sampler, p, 0.0), current_sh0, blend);
                current_sh1 = mix(textureSampleLevel(history_sh1, lumen_sampler, p, 0.0), current_sh1, blend);
            }
        }
    }
    let bad = any(current_fog != current_fog) || any(current_sh0 != current_sh0) || any(current_sh1 != current_sh1);
    textureStore(out_fog, gid, select(current_fog, vec4<f32>(0.0), bad));
    textureStore(out_sh0, gid, select(current_sh0, vec4<f32>(0.0), bad));
    textureStore(out_sh1, gid, select(current_sh1, vec4<f32>(0.0), bad));
}
