// MegaLights shadow rays towards a point of a light (megalights_sample.wgsl for pixels,
// megalights_volume.wgsl for froxels): shadow-map ray marching where the light has a virtual
// shadow map, distance fields otherwise. Requires common.wgsl, lumen_common.wgsl,
// lumen_mesh_sdf.wgsl, vsm_common.wgsl, vsm_sample.wgsl, vsm_local_common.wgsl,
// vsm_local_sample.wgsl and their bindings (vsm, vsm_entries, vsm_atlas, local_vsm, lumen,
// global_sdf, lumen_sampler, mesh_sdf_data, mesh_sdf_atlas).

// Shadow-map ray marching steps.
const ML_VSM_STEPS: u32 = 8u;
// Distance-field cones: narrow, as each ray goes to one point of the light (the source radius
// is sampled, not filtered); just wide enough to anti-alias the fields' voxels.
const ML_MESH_SDF_CONE: f32 = 0.02;
const ML_GLOBAL_SDF_CONE: f32 = 0.06;

// Shadow-map ray marching (UE's SMRT, one ray): steps from the receiver towards the light sample
// `point`, testing each against the light's virtual shadow map. The first step is the receiver
// itself (the classic shadow test, umbra and all); later steps see occluders off to the side as
// the ray converges on its point of the light - the penumbra of an area light - and count only
// within an assumed occluder thickness behind the stored depth.
fn vsm_ray(slot: u32, world: vec3<f32>, n: vec3<f32>, point: vec3<f32>, stop: f32, jitter: f32, mip_bias: u32) -> f32 {
    let r = local_receiver(slot, world, n, mip_bias);
    let near = local_vsm.slots[slot].params.y;
    let to_point = point - r.p;
    let length_point = length(to_point);
    let span = max(length_point - stop, 0.0) / max(length_point, 1e-6);
    for (var i = 0u; i < ML_VSM_STEPS; i++) {
        let f = select((f32(i) + jitter) / f32(ML_VSM_STEPS), 0.0, i == 0u);
        let p = r.p + to_point * (span * f * f);
        let s = local_lookup(slot, p, r.mip);
        if !s.found {
            continue;
        }
        let stored = textureLoad(vsm_atlas, vec2<i32>(s.uv * vsm.atlas_size), 0);
        let w = near / max(s.depth, 1e-9);
        let blocker = near / max(stored, 1e-9);
        let bias = 2.0 * local_texel(slot, w, r.mip);
        let thickness = select(1e9, 0.05 + 0.25 * w, i > 0u);
        if blocker < w - bias && w - blocker < thickness {
            return 0.0;
        }
    }
    return 1.0;
}

// Distance-field visibility of the light sample `point` (centre `center`, source radius
// `radius`) from `origin` (local_light_shadow's scheme with a narrow cone).
fn sdf_ray(origin: vec3<f32>, point: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let to_point = point - origin;
    let distance = length(to_point);
    let dir = to_point / max(distance, 1e-6);
    if !mesh_sdf_covers(origin) || !mesh_sdf_covers(center) {
        let level = sdf_finest_level(origin);
        let skip = lumen.sdf_levels[min(level, lumen.level_count - 1u)].w * 1.5;
        return sdf_soft_shadow_span(origin, dir, skip, light_shadow_distance(center, distance, radius), ML_GLOBAL_SDF_CONE);
    }
    let near = lumen.mesh_sdf_trace.x;
    // Stop short of the light's own fixture (a candle under its flame).
    let end = max(distance - radius, 0.0);
    var visibility = mesh_sdf_soft_shadow(origin, origin, dir, 0.0, min(end, near), ML_MESH_SDF_CONE, true);
    if distance > 2.0 * near && visibility > 0.01 {
        visibility = min(visibility, sdf_soft_shadow_span(origin, dir, near, distance - near, ML_GLOBAL_SDF_CONE));
    }
    if end > near && visibility > 0.01 {
        visibility = min(visibility, mesh_sdf_soft_shadow(center, origin, dir, max(near, distance - near), end, ML_MESH_SDF_CONE, false));
    }
    return visibility;
}
