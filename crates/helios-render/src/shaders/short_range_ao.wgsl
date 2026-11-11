// Short-range ambient occlusion from the depth buffer, as UE's Lumen applies its screen-space
// "short range AO" on top of the screen probes: occlusion by geometry within SRAO_RADIUS of a
// pixel that the probes (one per 16² pixels, half resolution) and the distance fields (no
// alpha-masked or thin foliage) cannot see — the ground under a grass tuft or a fern, the
// crease where a rock or a log meets the ground, a stalk's base. It darkens indirect light only
// (like material AO). Requires common.wgsl and, in the including shader, `frame: Frame`,
// `depth_texture`, `vsm: Vsm` (for its pixel angle).
//
// Per pixel SRAO_DIRECTIONS screen directions (rotated per pixel and frame by the caller's
// noise, so TSR averages them) are marched in SRAO_STEPS steps within the radius, keeping each
// direction's horizon: the highest elevation of its samples above the pixel's tangent plane,
// sin(elevation) = cos(v, n) for a sample at offset v, faded out towards the radius. A horizon at
// elevation e hides sin(e) of that direction's cosine-weighted half-slice of the hemisphere
// (horizon-based AO, as HBAO / GTAO); their mean is the occlusion. The depth buffer has no
// thickness: samples beyond the radius (a blade in front of a distant ground) are ignored, so
// thin occluders cast no halo.

const FLAG_SHORT_RANGE_AO: u32 = 4096u;
const SRAO_DIRECTIONS: u32 = 4u;
const SRAO_STEPS: u32 = 4u;
// World radius (m) and its screen cap (pixels at 900 rows).
const SRAO_RADIUS: f32 = 0.4;
const SRAO_MAX_PIXELS: f32 = 64.0;
// Tangent-plane bias (sine of the elevation: surfaces curving away do not occlude themselves)
// and strength.
const SRAO_BIAS: f32 = 0.15;
const SRAO_STRENGTH: f32 = 1.0;

fn short_range_ao(pixel: vec2<u32>, size: vec2<u32>, world: vec3<f32>, n: vec3<f32>, noise: f32) -> f32 {
    let distance_to_eye = distance(world, frame.camera_position.xyz);
    let radius_pixels = min(SRAO_RADIUS / max(distance_to_eye * vsm.pixel_angle, 1e-6), SRAO_MAX_PIXELS * f32(size.y) / 900.0);
    if radius_pixels < 1.5 {
        return 1.0;
    }
    let last = vec2<f32>(size) - 1.0;
    let r2 = SRAO_RADIUS * SRAO_RADIUS;
    let jitter = fract(noise * 7.0);
    var occlusion = 0.0;
    for (var i = 0u; i < SRAO_DIRECTIONS; i++) {
        let angle = (f32(i) + noise) * (6.2831853 / f32(SRAO_DIRECTIONS));
        let dir = vec2<f32>(cos(angle), sin(angle));
        var horizon = 0.0;
        for (var j = 0u; j < SRAO_STEPS; j++) {
            // Denser near the pixel (contact), jittered per pixel and frame.
            let t = (f32(j) + jitter) / f32(SRAO_STEPS);
            let q = clamp(vec2<f32>(pixel) + 0.5 + dir * (radius_pixels * t * t + 1.5), vec2<f32>(0.5), last + 0.5);
            let q_depth = textureLoad(depth_texture, vec2<u32>(q), 0).r;
            if q_depth <= 0.0 {
                continue;
            }
            let v = world_from_depth(q * frame.viewport.zw, q_depth, frame.inv_view_proj) - world;
            let d2 = dot(v, v);
            if d2 >= r2 || d2 < 1e-8 {
                continue;
            }
            horizon = max(horizon, (dot(v, n) * inverseSqrt(d2) - SRAO_BIAS) * (1.0 - d2 / r2));
        }
        occlusion += horizon;
    }
    let mean = occlusion / (f32(SRAO_DIRECTIONS) * (1.0 - SRAO_BIAS));
    return saturate(1.0 - SRAO_STRENGTH * mean);
}
