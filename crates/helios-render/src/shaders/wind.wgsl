// Wind animation of vegetation: UE5's foliage World Position Offset (see wind.rs). Requires
// common.wgsl and a `wind: Wind` uniform at binding 65.
//
// A vertex moves by a pure function of its rest world position, its instance's transform and the
// time, so every pass computes the same position and clusters or LOD levels sharing a vertex move
// it identically (no cracks). Three layers, as in UE's hierarchical foliage wind:
//   bend    : the plant leans with the gusts and sways around that lean at its natural
//             frequency, growing with height² above the instance origin (a cantilever)
//   sway    : branches bob, growing with the distance from the plant's vertical axis, with a
//             phase that shifts along them
//   flutter : leaves and blades flutter quickly, with a short spatial wavelength
// Gusts are waves travelling along the wind, so they roll across a field of grass. A plant's own
// phase comes from its origin: wood and foliage instances sharing a transform move together.
// Normals are not rotated (the motion is subtle).

struct Wind {
    // xz: unit direction the wind blows towards, w: strength (0: calm or disabled)
    direction: vec4<f32>,
    // x: time, y: previous frame's time (s), z: gust frequency (Hz), w: turbulence (0..1)
    time: vec4<f32>,
    // xyz: camera position, w: distance beyond which nothing moves
    camera: vec4<f32>,
    // x: distance per metre of displacement bound beyond which motion is too small to see,
    // y: gust wavelength (m)
    params: vec4<f32>,
}

const WIND_TAU: f32 = 6.2831853;
const WIND_FLUTTER_HZ: f32 = 2.6;
// |wind_offset| <= strength * (WIND_BEND_BOUND * bend * height² + sway * reach
//                              + WIND_FLUTTER_BOUND * flutter); see wind_reach.
const WIND_BEND_BOUND: f32 = 1.05;
const WIND_FLUTTER_BOUND: f32 = 1.7321;

// Strength of the wind at an instance: the scene's, faded out over the last quarter of the
// instance's disable distance — where its motion becomes too small to see on screen (grass
// before trees), at most the scene's disable distance. Per instance, so a plant never deforms
// unevenly and never pops. 0 for static materials.
fn wind_strength(instance: Instance) -> f32 {
    let strength = wind.direction.w;
    if strength <= 0.0 || all(instance.wind.xyz == vec3<f32>(0.0)) {
        return 0.0;
    }
    let limit = min(wind.camera.w, wind_reach(instance, strength, instance.bounds) * wind.params.x);
    let d = distance(instance.world[3].xyz, wind.camera.xyz);
    let fade = saturate(4.0 * (limit - d) / max(limit, 1e-3));
    return strength * fade * fade * (3.0 - 2.0 * fade);
}

// Whether `instance`, with wind strength `strength` (wind_strength), is a dynamic shadow caster
// this frame: one the wind moves, a Mobility::Movable one or a skin whose pose moves
// (skin_dynamic, skin_palette.wgsl: every shader that includes this file includes it too).
// Cached shadow pages keep it out of their static layer and re-render it whenever it moved
// (vsm_invalidate.wgsl). Culling, page expansion and invalidation must agree on it.
fn dynamic_caster(instance: Instance, strength: f32) -> bool {
    return strength > 0.0 || (instance.flags & INSTANCE_MOVABLE) != 0u || skin_dynamic(instance);
}

// Gust envelope in [0, 1] at a plant's origin.
fn wind_gust(origin: vec3<f32>, time: f32) -> f32 {
    let travel = dot(origin.xz, wind.direction.xz) / wind.params.y;
    let p = WIND_TAU * (wind.time.z * time - travel);
    return saturate(0.5 + 0.3 * sin(p) + 0.2 * sin(2.37 * p + 1.9));
}

// A plant's own phase in [0, 2π) from the bits of its origin.
fn wind_phase(origin: vec3<f32>) -> f32 {
    let h = pcg_hash(bitcast<u32>(origin.x) ^ pcg_hash(bitcast<u32>(origin.z)));
    return f32(h & 0xffffu) * (WIND_TAU / 65536.0);
}

// World-space displacement at `time` of the vertex with rest position `rest` of an instance with
// transform `world` and response `response` (bend, sway, flutter, frequency).
fn wind_offset(world: mat4x4<f32>, response: vec4<f32>, strength: f32, rest: vec3<f32>, time: f32) -> vec3<f32> {
    let origin = world[3].xyz;
    let rel = rest - origin;
    let height = max(rel.y, 0.0);
    let reach = length(rel.xz);
    let dir = vec3<f32>(wind.direction.x, 0.0, wind.direction.z);
    let side = vec3<f32>(-dir.z, 0.0, dir.x);
    let gust = wind_gust(origin, time);
    let turbulence = wind.time.w;
    let plant = WIND_TAU * response.w * time + wind_phase(origin);
    // Bend: |lean| <= 1, |lateral| <= 0.3, so this term stays below WIND_BEND_BOUND.
    let lean = gust * (1.0 - 0.5 * turbulence) + 0.5 * turbulence * sin(plant);
    let lateral = 0.3 * turbulence * sin(0.71 * plant + 1.3);
    var offset = (dir * lean + side * lateral) * (response.x * height * height);
    // Sway: a unit vector, mostly up and down.
    let branch = sin(2.3 * plant + dot(rel, vec3<f32>(0.9, 0.7, 0.8)));
    offset += vec3<f32>(0.6 * dir.x, 0.8, 0.6 * dir.z) * (response.y * reach * (0.4 + 0.6 * gust) * branch);
    // Flutter: each component in [-1, 1].
    let t = WIND_TAU * WIND_FLUTTER_HZ * time;
    let q = rel * 5.0;
    let flutter = vec3<f32>(sin(t + q.x + q.y), sin(1.31 * t + q.y + q.z + 1.7), sin(1.17 * t + q.z - q.x + 3.1));
    offset += flutter * (response.z * (0.35 + 0.65 * gust));
    return offset * strength;
}

// Upper bound of the displacement of any vertex inside the world-space `sphere` (for culling).
fn wind_reach(instance: Instance, strength: f32, sphere: vec4<f32>) -> f32 {
    let origin = instance.world[3].xyz;
    let height = max(sphere.y + sphere.w - origin.y, 0.0);
    let reach = length(sphere.xz - origin.xz) + sphere.w;
    let r = instance.wind;
    return strength * ((WIND_BEND_BOUND * r.x * height * height) + (r.y * reach) + (WIND_FLUTTER_BOUND * r.z));
}

// Upper bound of the speed (m/s) of any vertex inside `sphere`: the derivative of wind_offset
// term by term (shadow pages under it are refreshed once it may have moved far enough).
fn wind_speed(instance: Instance, strength: f32, sphere: vec4<f32>) -> f32 {
    let origin = instance.world[3].xyz;
    let height = max(sphere.y + sphere.w - origin.y, 0.0);
    let reach = length(sphere.xz - origin.xz) + sphere.w;
    let r = instance.wind;
    let turbulence = wind.time.w;
    let gust_rate = WIND_TAU * wind.time.z * 0.774;  // |d gust / dt|
    let plant_rate = WIND_TAU * r.w;
    let bend = (r.x * height * height) * ((gust_rate * (1.0 - 0.5 * turbulence)) + (plant_rate * turbulence * 0.713));
    let sway = (r.y * reach) * ((0.6 * gust_rate) + (2.3 * plant_rate));
    let flutter = r.z * ((1.126 * gust_rate) + (2.0235 * WIND_TAU * WIND_FLUTTER_HZ));
    return strength * (bend + sway + flutter);
}

// A bounding sphere grown to contain its geometry wherever the wind moves it.
fn wind_padded(instance: Instance, strength: f32, sphere: vec4<f32>) -> vec4<f32> {
    if strength <= 0.0 {
        return sphere;
    }
    return vec4<f32>(sphere.xyz, sphere.w + wind_reach(instance, strength, sphere));
}

// World position of an object-space vertex this frame, wind included (`strength` from
// wind_strength).
fn wind_world_position(instance: Instance, strength: f32, object: vec3<f32>) -> vec3<f32> {
    let rest = (instance.world * vec4<f32>(object, 1.0)).xyz;
    if strength <= 0.0 {
        return rest;
    }
    return rest + wind_offset(instance.world, instance.wind, strength, rest, wind.time.x);
}

// The same vertex in the previous frame (motion vectors).
fn wind_previous_world_position(instance: Instance, strength: f32, object: vec3<f32>) -> vec3<f32> {
    let rest = (instance.prev_world * vec4<f32>(object, 1.0)).xyz;
    if strength <= 0.0 {
        return rest;
    }
    return rest + wind_offset(instance.prev_world, instance.wind, strength, rest, wind.time.y);
}
