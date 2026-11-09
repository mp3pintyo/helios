// Local lights: layout, clustered grid addressing and evaluation. Requires common.wgsl.
//
// The view frustum is divided into CLUSTER_X x CLUSTER_Y screen tiles and CLUSTER_Z
// exponentially spaced depth slices; light_cull.wgsl stores, per cluster, the lights whose
// bounding spheres touch it.

const CLUSTER_X: u32 = 16u;
const CLUSTER_Y: u32 = 9u;
const CLUSTER_Z: u32 = 24u;
const CLUSTER_NEAR: f32 = 0.1;
const CLUSTER_FAR: f32 = 2000.0;
const LIGHTS_PER_CLUSTER: u32 = 63u;
const CLUSTER_WORDS: u32 = 64u; // count + indices
const LIGHT_SPOT: f32 = 1.0;

struct GpuLight {
    position_range: vec4<f32>,  // xyz, range
    color: vec4<f32>,           // rgb * intensity, source radius
    direction: vec4<f32>,       // spot direction, cos(outer)
    params: vec4<f32>,          // cos(inner), kind (0 point, 1 spot), casts shadows, shadow map slot + 1
}

// The light's virtual shadow map slot (vsm_local.rs), or INVALID_ID: lights without one fall
// back to distance-field shadows.
fn light_shadow_slot(light: GpuLight) -> u32 {
    return u32(light.params.w) - 1u;
}

fn cluster_slice(view_depth: f32) -> u32 {
    let s = log(max(view_depth, CLUSTER_NEAR) / CLUSTER_NEAR) / log(CLUSTER_FAR / CLUSTER_NEAR);
    return min(u32(s * f32(CLUSTER_Z)), CLUSTER_Z - 1u);
}

fn slice_depth(slice: f32) -> f32 {
    return CLUSTER_NEAR * pow(CLUSTER_FAR / CLUSTER_NEAR, slice / f32(CLUSTER_Z));
}

fn cluster_index(uv: vec2<f32>, view_depth: f32) -> u32 {
    let tile = min(vec2<u32>(uv * vec2<f32>(f32(CLUSTER_X), f32(CLUSTER_Y))), vec2<u32>(CLUSTER_X - 1u, CLUSTER_Y - 1u));
    return (cluster_slice(view_depth) * CLUSTER_Y + tile.y) * CLUSTER_X + tile.x;
}

// Inverse-square falloff windowed to reach zero at `range` (UE's light falloff).
fn light_falloff(distance_sq: f32, range: f32) -> f32 {
    let r = distance_sq / (range * range);
    let window = saturate(1.0 - r * r);
    return window * window / max(distance_sq, 1e-4);
}

// Incident radiance scale (before shadows and BRDF) of `light` at `p`; `l` is set to the
// unit vector towards the light and `distance` to its distance.
struct LightSample {
    l: vec3<f32>,
    distance: f32,
    radiance: vec3<f32>,
}

fn sample_light(light: GpuLight, p: vec3<f32>) -> LightSample {
    var s: LightSample;
    let to_light = light.position_range.xyz - p;
    let d2 = dot(to_light, to_light);
    s.distance = sqrt(d2);
    s.l = to_light / max(s.distance, 1e-4);
    var attenuation = light_falloff(d2, light.position_range.w);
    if light.params.y == LIGHT_SPOT {
        let cos_angle = dot(-s.l, light.direction.xyz);
        attenuation *= smoothstep(light.direction.w, light.params.x, cos_angle);
    }
    s.radiance = light.color.rgb * attenuation;
    return s;
}
