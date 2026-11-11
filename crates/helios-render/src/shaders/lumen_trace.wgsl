// Lumen world-space ray tracing, shared by the final gathers (screen probes in lumen_gather.wgsl,
// the irradiance field of Lumen Lite in lumen_lite.wgsl): the mesh SDFs of the near field, the
// global distance field beyond, the surface caches at the hit, the sky where nothing is hit.
//
// Includers declare, besides the bindings of lumen_common.wgsl, lumen_mesh_sdf.wgsl and
// lumen_cards.wgsl: sky_view_lut: texture_2d<f32>. The world cache's cells are bound by
// lumen_world_cache_common.wgsl (binding 150).

#include "lumen_world_cache_common.wgsl"

struct Radiance {
    color: vec3<f32>,
    // Fraction of the ray that saw the sky (0, 0.5 for the far-field estimate, 1).
    sky: f32,
    // Hit distance (the trace distance for rays that hit nothing).
    t: f32,
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    return sample_sky_view(sky_view_lut, lumen_sampler, frame.camera_position.xyz, dir, sun) * frame.sun_illuminance.rgb * lumen.sky_intensity;
}

// Mesh SDFs (near field) from `origin`, then the global distance field from `t_min` on (from
// the end of the near field where it covers `origin`), then the sky.
fn trace_world_radiance(origin: vec3<f32>, dir: vec3<f32>, t_min: f32) -> Radiance {
    var r: Radiance;
    r.sky = 0.0;
    r.t = lumen.max_trace_distance;
    if lumen.sdf_enabled != 0u {
        var t_global = t_min;
        if mesh_sdf_covers(origin) {
            // The near field resolves what the screen trace cannot see (occluded or off-screen
            // geometry) down to the mesh SDFs' detail, right from the origin.
            let near = trace_mesh_sdfs(origin, dir, 0.0, lumen.mesh_sdf_trace.x);
            if near.hit {
                let p = origin + dir * near.t;
                // The world cache over the voxel estimate; the cards over both unless it replaces them.
                let fallback = world_cache_over(near_field_radiance(p, near.normal), p, near.normal);
                if lumen.world_cache == WC_REPLACE {
                    r.color = fallback;
                } else {
                    r.color = card_surface_radiance(fallback, near.instance, p, near.normal, 0.02);
                }
                r.t = near.t;
                return r;
            }
            t_global = lumen.mesh_sdf_trace.x;
        }
        let hit = trace_sdf(origin, dir, t_global, lumen.max_trace_distance);
        if hit.hit {
            // Sample the surface cache on the near side of the hit: lighting voxels are larger
            // than thin walls, so this keeps the far (e.g. sunlit exterior) side from leaking in.
            let lighting_voxel = lumen.sdf_levels[hit.level].w * SDF_RES / LIGHT_RES;
            var voxels = voxel_radiance(hit.position - dir * lighting_voxel * 0.6, hit.level);
            let normal = sdf_normal(hit.position, hit.level);
            let n = select(normal, -normal, dot(normal, dir) > 0.0);
            if lumen.world_cache != WC_OFF {
                // The world cache's cell of the surface (the hit moved onto the field's zero level).
                voxels = world_cache_over(voxels, hit.position - normal * sdf_distance(hit.position, hit.level), n);
            }
            // The surface's cards, if any.
            if lumen.world_cache == WC_REPLACE {
                r.color = voxels;
            } else {
                r.color = card_field_radiance(voxels, hit.position, n, hit.level, lumen.sdf_levels[hit.level].w * 0.5);
            }
            r.t = hit.t;
            return r;
        }
        if sdf_finest_level(origin + dir * lumen.max_trace_distance) < lumen.level_count {
            // Ray ended inside the clipmap without hitting anything nearby: far-field estimate
            // from the radiance cache.
            let far = origin + dir * lumen.max_trace_distance;
            r.color = mix(sky_radiance(dir), radiance_cache_irradiance(far, -dir, sky_radiance(dir)) / PI, 0.5);
            r.sky = 0.5;
            return r;
        }
    }
    r.color = sky_radiance(dir);
    r.sky = 1.0;
    return r;
}
