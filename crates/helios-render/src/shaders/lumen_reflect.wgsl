// Lumen reflections, shared by the traced reflections (lumen_gather.wgsl) and Lumen Lite's
// screen-space ones (lumen_lite.wgsl): GGX sampling, the test for unlit screen-trace hits and
// height fog on reflection hits.
//
// Requires common.wgsl, atmosphere.wgsl, lumen_common.wgsl, lumen_trace.wgsl (sky_radiance),
// fog_common.wgsl and the bindings
//   fog: Fog, transmittance_lut: texture_2d<f32>, lumen_sampler: sampler, exposure: array<f32, 4>,
//   gbuffer_motion: texture_2d<f32>, previous_radiance: texture_2d<f32> (last frame's lit image).

// Visible-normal GGX sampling (Heitz 2018), in the local frame where n = +Z.
fn sample_ggx_vndf(v: vec3<f32>, alpha: f32, u: vec2<f32>) -> vec3<f32> {
    let vh = normalize(vec3<f32>(alpha * v.x, alpha * v.y, v.z));
    let len2 = vh.x * vh.x + vh.y * vh.y;
    let t1 = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(len2), len2 > 0.0);
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// Exposed luminance below which a pixel of last frame's image counts as unlit (black: nothing
// lit it - the first frame after a cut, a pixel the history did not cover).
const UNLIT_EXPOSED_LUMINANCE: f32 = 1e-4;

// Last frame's radiance of the surface at screen `uv` (reprojected with the pixel's motion), or
// a negative value where it has none: the reprojection leaves the screen or the pixel was unlit.
// With UE 5.8's `ScreenTraces.HZBTraversal.SkipUnlitHits` a screen trace passes such hits and
// goes on (behind them, or into the world trace) instead of returning black or the clamped
// colour of the screen's edge.
fn screen_hit_radiance(uv: vec2<f32>) -> vec3<f32> {
    let motion = textureLoad(gbuffer_motion, vec2<u32>(uv * frame.viewport.xy), 0).xy;
    let previous_uv = uv - motion;
    if any(previous_uv <= vec2<f32>(0.0)) || any(previous_uv >= vec2<f32>(1.0)) || !has_flag(frame.flags, FLAG_HISTORY_VALID) {
        return vec3<f32>(-1.0);
    }
    let color = textureSampleLevel(previous_radiance, lumen_sampler, previous_uv, 0.0).rgb;
    if luminance(color) * exposure[0] < UNLIT_EXPOSED_LUMINANCE || any(color != color) {
        return vec3<f32>(-1.0);
    }
    return color;
}

// The sun's visibility for the fog's in-scattering: a wide cone through the global field two
// levels coarser than the point's finest (fog shafts are soft; 16 steps at most).
fn fog_sun_visibility(p: vec3<f32>, sun: vec3<f32>) -> f32 {
    var level = min(sdf_finest_level(p) + 2u, lumen.level_count);
    var visibility = 1.0;
    var t = 0.0;
    for (var i = 0u; i < 16u; i++) {
        if level >= lumen.level_count || t >= lumen.max_trace_distance {
            break;
        }
        let q = p + sun * t;
        if !sdf_inside_level(q, level, 1.0) {
            level++;
            continue;
        }
        let voxel = lumen.sdf_levels[level].w;
        let d = sdf_distance(q, level);
        visibility = min(visibility, saturate(d / (0.2 * max(t, voxel)) + 0.15));
        if visibility <= 0.01 {
            return 0.0;
        }
        t += max(d, voxel);
    }
    return visibility;
}

fn reflection_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Height fog between a reflecting point `origin` and what it reflects `t` metres along `dir`
// (UE 5.8's r.Lumen.HeightFog): the volumetric fog's exponential height density integrated
// analytically along the segment (within the fog's distance; below its base height the density
// is constant, its drifting noise is left out), and the light it scatters towards the
// reflecting point - the sun shadowed by one wide distance-field cone from the segment's middle,
// ambient light from the radiance cache above and below it (as fog.wgsl injects, without local
// lights). Sky misses pass `t` = the trace distance: they get the fog of the fog's whole depth,
// as the sky behind the fog does in the main view.
fn reflection_height_fog(color: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>, t: f32) -> vec3<f32> {
    if fog.enabled == 0u {
        return color;
    }
    let len = min(t, fog.range.y);
    let k = max(fog.height.y, 1e-5);
    let h0 = max(origin.y - fog.height.x, 0.0);
    let h1 = max(origin.y + dir.y * len - fog.height.x, 0.0);
    let dh = h1 - h0;
    var integral = len * exp(-k * h0);
    if abs(dh) * k > 1e-3 {
        integral = len * (exp(-k * h0) - exp(-k * h1)) / (k * dh);
    }
    let optical_depth = fog.media.x * integral;
    if optical_depth < 1e-4 {
        return color;
    }
    let transmittance = exp(-optical_depth);
    let mid = origin + dir * (len * 0.5);
    let sun = normalize(frame.sun_direction.xyz);
    var scattered = vec3<f32>(0.0);
    if sun.y > -0.05 {
        let visibility = fog_sun_visibility(mid, sun);
        if visibility > 0.0 {
            let sun_transmittance = sample_transmittance(transmittance_lut, lumen_sampler, atmosphere_position(mid), sun);
            scattered = frame.sun_illuminance.rgb * sun_transmittance * visibility * reflection_phase(dot(sun, dir), fog.media.y);
        }
    }
    let sky_up = sky_radiance(vec3<f32>(0.0, 1.0, 0.0)) * PI;
    let up = radiance_cache_irradiance(mid, vec3<f32>(0.0, 1.0, 0.0), sky_up);
    let down = radiance_cache_irradiance(mid, vec3<f32>(0.0, -1.0, 0.0), sky_up * frame.ground_albedo.rgb);
    scattered += (up + down) * (0.5 / PI) * fog.media.z;
    return color * transmittance + scattered * fog.albedo.rgb * (1.0 - transmittance);
}
