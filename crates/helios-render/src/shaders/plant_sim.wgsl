// Dynamic wind's simulated tier (plant_sim.rs; UE 5.9's promoted Dynamic Wind instances): the
// plants nearest the camera — and those a gust burst reaches — get a simulation slot. Each of
// their non-leaf bones carries a spring state: a deviation δ (a rotation vector, object space)
// from its parametric rotation (plant_wind.wgsl), driven towards contact targets — out of a push
// capsule that follows the camera (walking through grass and ferns parts them) and away from a
// radial gust front — and otherwise back to 0 with a little overshoot. The rasterisers add δ to
// the parametric rotation (plant_pose), so an undisturbed promoted plant moves exactly like an
// unpromoted one: promotion and demotion never pop. Like UE's, each bone is a spring of its own
// (no constraint solver between bones); leaves keep their parametric flutter.
//
// One workgroup per slot. The slot's row of the sim texture holds per bone δ this frame and the
// previous frame's (motion vectors).

#include "common.wgsl"
#include "wind.wgsl"

struct SimParams {
    // xyz: camera (the push capsule's top), w: capsule radius
    capsule: vec4<f32>,
    // x: time, y: time step, z: capsule length below the camera, w: largest deviation in units
    // of the bone's parametric bound
    time: vec4<f32>,
    // xyz: gust centre, w: gust start time (< 0: none)
    gust: vec4<f32>,
    // x: gust strength (× the bone's parametric bound), y: front speed (m/s), z: front width (m),
    // w: duration (s)
    gust_shape: vec4<f32>,
}

struct SimSlot {
    instance: u32,
    bones: u32,
    fresh: u32,  // just promoted: start at rest (δ = 0)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> slots: array<SimSlot>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
// Per (slot, bone): δ, δ' (2 vec4).
@group(0) @binding(3) var<storage, read_write> state: array<vec4<f32>>;
@group(0) @binding(4) var sim_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(170) var plant_bones: texture_2d<f32>;

const SIM_BONES: u32 = 512u;

fn bone_texel(t: u32) -> vec4<f32> {
    return textureLoad(plant_bones, vec2<u32>(t & 1023u, t >> 10u), 0);
}

// Closest point to `p` on the segment a–b.
fn closest_on_segment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let t = saturate(dot(p - a, ab) / max(dot(ab, ab), 1e-8));
    return a + (ab * t);
}

@compute @workgroup_size(64)
fn simulate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let slot = slots[wid.x];
    let instance = instances[slot.instance];
    let strength = wind_strength(instance);
    let response = bone_texel(instance.plant.x - 1u).xyz * strength;
    let world3 = mat3x3<f32>(instance.world[0].xyz, instance.world[1].xyz, instance.world[2].xyz);
    let to_object = transpose(mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz));
    let dt = clamp(params.time.y, 0.0, 0.05);
    let top = params.capsule.xyz;
    let bottom = top - vec3<f32>(0.0, params.time.z, 0.0);
    let radius = params.capsule.w;
    let gust_age = params.time.x - params.gust.w;
    let gusting = params.gust.w >= 0.0 && gust_age >= 0.0 && gust_age < params.gust_shape.w;
    for (var b = lid; b < min(slot.bones, SIM_BONES); b += 64u) {
        let s = (wid.x * SIM_BONES) + b;
        var d = state[2u * s];
        var v = state[2u * s + 1u];
        if slot.fresh != 0u {
            d = vec4<f32>(0.0);
            v = vec4<f32>(0.0);
        }
        let t = instance.plant.y + (b * 3u);
        let t0 = bone_texel(t);
        let t1 = bone_texel(t + 1u);
        let t2 = bone_texel(t + 2u);
        let kind = u32(t2.y);
        var target_rotation = vec3<f32>(0.0);
        // The bone's parametric bound (radians): its deviation stays within (largest − 1) times
        // it, which plant_sim.rs adds to the instance's culling bounds.
        let bound = t2.z * select(select(1.7321 * response.z, response.y, kind == 1u), 1.0441 * response.x, kind == 0u);
        let limit = params.time.w * bound;
        if kind != 2u && limit > 0.0 {
            // Contact: the bone's outer part, in world space at rest.
            let axis_world = normalize(world3 * t1.xyz);
            let tip = (instance.world * vec4<f32>(t0.xyz + (t1.xyz * (0.75 * t1.w)), 1.0)).xyz;
            let length_world = 0.75 * t1.w * instance.max_scale;
            let c = closest_on_segment(tip, bottom, top);
            let away = vec3<f32>(tip.x - c.x, 0.0, tip.z - c.z);
            let gap = length(away);
            if gap < radius {
                let dir = select(vec3<f32>(1.0, 0.0, 0.0), away / gap, gap > 1e-4);
                let angle = min((radius - gap) / max(length_world, 0.05), limit);
                target_rotation += to_object * (cross(axis_world, dir) * angle);
            }
            if gusting {
                // A ring travelling out from the centre, fading over its duration.
                let radial = vec3<f32>(tip.x - params.gust.x, 0.0, tip.z - params.gust.z);
                let r = length(radial);
                let front = params.gust_shape.y * gust_age;
                let shell = (r - front) / params.gust_shape.z;
                let envelope = exp(-(shell * shell)) * (1.0 - (gust_age / params.gust_shape.w));
                if r > 1e-3 && envelope > 1e-3 {
                    target_rotation += to_object * (cross(axis_world, radial / r) * (params.gust_shape.x * bound * envelope));
                }
            }
        }
        // A damped spring towards the target (stiffer where it touches), semi-implicit Euler in
        // four steps; its frequency follows the bone's (heavier bones respond slower).
        let hz = max(bone_texel(instance.plant.x - 1u).w * t2.w * select(1.0, 2.3, kind == 1u), 0.5);
        let omega = 6.2831853 * hz * select(2.0, 4.0, dot(target_rotation, target_rotation) > 0.0);
        let zeta = 0.3;
        let h = dt * 0.25;
        var delta = d.xyz;
        var rate = v.xyz;
        for (var k = 0u; k < 4u; k++) {
            rate += ((omega * omega * (target_rotation - delta)) - (2.0 * zeta * omega * rate)) * h;
            delta += rate * h;
        }
        let magnitude = length(delta);
        if magnitude > limit {
            delta *= limit / max(magnitude, 1e-9);
            rate *= 0.5;
        }
        // Its speed stays within as many times the class's speed bound (wind.wgsl wind_speed):
        // shadow pages under it are refreshed in time.
        let gust_rate = WIND_TAU * wind.time.z * 0.774;
        let plant_rate = WIND_TAU * bone_texel(instance.plant.x - 1u).w;
        let tau = wind.time.w;
        let class_rate = select(select((1.126 * gust_rate) + (2.0235 * WIND_TAU * WIND_FLUTTER_HZ), (0.6 * gust_rate) + (2.3 * plant_rate), kind == 1u), (gust_rate * (1.0 - (0.5 * tau))) + (plant_rate * tau * 0.713), kind == 0u);
        let max_rate = limit * class_rate;
        let speed = length(rate);
        if speed > max_rate {
            rate *= max_rate / max(speed, 1e-9);
        }
        if limit <= 0.0 {
            delta = vec3<f32>(0.0);
            rate = vec3<f32>(0.0);
        }
        let previous = select(d.xyz, delta, slot.fresh != 0u);
        state[2u * s] = vec4<f32>(delta, 0.0);
        state[2u * s + 1u] = vec4<f32>(rate, 0.0);
        textureStore(sim_out, vec2<u32>(2u * b, wid.x), vec4<f32>(delta, 0.0));
        textureStore(sim_out, vec2<u32>((2u * b) + 1u, wid.x), vec4<f32>(previous, 0.0));
    }
}
