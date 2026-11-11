// Dynamic plant wind's motion of one bone (plant_wind.rs, helios_geometry::plant_bones): what the
// pose pass (plant_pose.wgsl) and the per-vertex pose (plant_wind.wgsl) share. Requires
// common.wgsl, wind.wgsl and a `plant_bones: texture_2d<f32>` at binding 170.
//
// A bone's rotation is a closed-form function of the wind, the gust at the plant, the time, and
// its own amplitude, frequency and phase, with the time functions of the procedural wind's three
// layers (wind.wgsl) by class: bend (stems, trunks, blades, fronds lean with the gusts and sway at
// the plant's frequency), sway (branches bob, up to 2.3× faster), flutter (leaves flutter and
// twist about their stem). The bone bends about its pivot as a circular arc — its frame at
// distance s along it turns by θ·s/L — and everything attached to it rides on its frame there.
//
// Bone texels (plant_wind.rs bone_texels), four a bone from the table's first texel — a vertex
// fetches them at once, nothing it needs depends on another bone's texels — and the table's header
// texel just before it:
//   0: pivot.xyz, length
//   1: axis.xyz, class + phase fraction
//   2: amplitude, frequency fraction, attachment on the parent, parent (−1: root)
//   3: the pose entry of its carrier (−1: none: the carrier is a root), the root its chain hangs
//      from (itself for a root), where on the root (its depth-1 ancestor's attachment; for a leaf
//      on a root its own attachment; −1 for a root: each point's projection), 1 for a leaf at the
//      end of its chain (else 0)
//   header: bones, ranked bones (pose entries an instance needs), classes of the ranked bones
//      (bit per class), 0
// After the bones, the ranked bones' indices, four a texel (the pose pass's rank → bone).

const PLANT_TEXTURE_WIDTH: u32 = 1024u;
const PLANT_BONE_TEXELS: u32 = 4u;

fn plant_texel(t: u32) -> vec4<f32> {
    return textureLoad(plant_bones, vec2<u32>(t & (PLANT_TEXTURE_WIDTH - 1u), t >> 10u), 0);
}

// What every bone of one plant shares at one time, in the instance's object space.
struct PlantAir {
    wind: vec3<f32>,      // unit direction the wind blows towards
    side: vec3<f32>,      // horizontal, across it
    up: vec3<f32>,
    gust: f32,            // [0, 1] at the plant
    turbulence: f32,
    plant: f32,           // the plant's own phase: 2π f t + its phase
    flutter: f32,         // 2π · 2.6 Hz · t
    response: vec3<f32>,  // the material's bend, sway, flutter times the strength
    // Leaf flutter's level of detail: 1 near, fading to 0 where a leaf tip's flutter (2.5 × the
    // response, helios_geometry::vegetation) would cover under a pixel; at 0 leaves ride rigidly
    // on their branch and their level is skipped. A function of the instance and the camera
    // only, so every pass and view agrees.
    leaves: f32,
    // The same for branches (sway bones): about a radian per unit sway (helios_geometry::
    // vegetation) on a branch about 0.4 × the plant's radius long.
    branches: f32,
    // Promoted instances (plant_sim.wgsl): simulation slot + 1 (0: parametric only), and which
    // frame to read (0: this frame's, 1: the previous frame's) of the simulated deviations and
    // of the pose entries.
    sim: u32,
    sim_texel: u32,
    // The instance's pose entries (plant_pose.wgsl): first entry + 1, 0 when it has none (every
    // ranked bone is at rest: beyond the fades).
    pose: u32,
}

fn plant_air(instance: Instance, strength: f32, origin: vec3<f32>, time: f32) -> PlantAir {
    var a: PlantAir;
    let response = plant_texel(instance.plant.x - 1u);
    // World directions to object space: the normal matrix is the scaled inverse transpose, its
    // transpose the inverse times the scale (a rotation for uniform scale).
    let to_object = transpose(mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz));
    a.wind = normalize(to_object * vec3<f32>(wind.direction.x, 0.0, wind.direction.z));
    a.up = normalize(to_object * vec3<f32>(0.0, 1.0, 0.0));
    a.side = cross(a.wind, a.up);
    a.gust = wind_gust(origin, time);
    a.turbulence = wind.time.w;
    a.plant = (WIND_TAU * response.w * time) + wind_phase(origin);
    a.flutter = WIND_TAU * WIND_FLUTTER_HZ * time;
    a.response = response.xyz * strength;
    // wind.params.x: distance per metre of motion at which it covers WIND_MIN_PIXELS (4); the
    // leaves and branches stop where theirs would cover a pixel.
    let d = distance(origin, wind.camera.xyz);
    let leaf_limit = 2.5 * a.response.z * 4.0 * wind.params.x;
    a.leaves = saturate(4.0 * (leaf_limit - d) / max(leaf_limit, 1e-3));
    let branch_limit = 0.4 * instance.bounds.w * a.response.y * 4.0 * wind.params.x;
    a.branches = saturate(4.0 * (branch_limit - d) / max(branch_limit, 1e-3));
    a.sim = instance.plant.z;
    a.sim_texel = 0u;
    a.pose = instance.plant.w;
    return a;
}

// The same for the previous frame (motion vectors): its origin, time and simulation state.
fn plant_air_previous(instance: Instance, strength: f32) -> PlantAir {
    var a = plant_air(instance, strength, wind_previous_origin(instance), wind.time.y);
    a.sim_texel = 1u;
    return a;
}

// The rotation vector of a whole bone (axis × angle, object space), from its texel 2 fields,
// faded with distance by its class.
fn plant_bone_rotation(a: PlantAir, axis: vec3<f32>, kind_phase: f32, amplitude: f32, frequency: f32) -> vec3<f32> {
    let kind = u32(kind_phase);
    let phase = fract(kind_phase) * WIND_TAU;
    if kind == 0u {
        // Bend: lean with the gust, sway around it (|force| ≤ 1.044, as the procedural bend).
        let p = (a.plant * frequency) + phase;
        let lean = (a.gust * (1.0 - (0.5 * a.turbulence))) + (0.5 * a.turbulence * sin(p));
        let lateral = 0.3 * a.turbulence * sin((0.71 * p) + 1.3);
        return cross(axis, (a.wind * lean) + (a.side * lateral)) * (amplitude * a.response.x);
    }
    if kind == 1u {
        // Sway: pushed downwind by the gust, bobbing mostly up and down (|force| ≤ 1).
        let b = sin((2.3 * frequency * a.plant) + phase);
        let force = (a.wind * (0.3 * a.gust)) + (((a.wind * 0.6) + (a.up * 0.8)) * (0.7 * (0.4 + (0.6 * a.gust)) * b));
        return cross(axis, force) * (amplitude * a.response.y * a.branches);
    }
    // Flutter: about two axes across the leaf and a twist about its stem (each in [-1, 1]).
    let t = (a.flutter * frequency) + phase;
    let across = cross(axis, a.up);
    let e1 = select(a.side, normalize(across), dot(across, across) > 1e-6);
    let e2 = cross(axis, e1);
    let v = (e1 * sin(t)) + (axis * sin((1.31 * t) + 1.7)) + (e2 * sin((1.17 * t) + 3.1));
    return v * (amplitude * a.response.z * a.leaves * (0.35 + (0.65 * a.gust)));
}

// A point (and a direction) in its pose.
struct PlantPose {
    position: vec3<f32>,
    normal: vec3<f32>,
}

// Rodrigues: `v` turned by angle θ (sin, cos given) about unit axis `k`.
fn plant_rotate(v: vec3<f32>, k: vec3<f32>, s: f32, c: f32) -> vec3<f32> {
    return (v * c) + (cross(k, v) * s) + (k * (dot(k, v) * (1.0 - c)));
}

// One bone (its pivot, axis, length) and its rotation vector this frame.
struct PlantBoneState {
    pivot: vec3<f32>,
    axis: vec3<f32>,
    len: f32,
    omega: vec3<f32>,
}

// A bone from its texels 0..2: its rotation this frame (faded with distance by its class).
fn plant_bone_state(a: PlantAir, t0: vec4<f32>, t1: vec4<f32>, t2: vec4<f32>) -> PlantBoneState {
    return PlantBoneState(t0.xyz, t1.xyz, t0.w, plant_bone_rotation(a, t1.xyz, t1.w, t2.x, t2.y));
}

// The bone's frame at distance s along it: a rotation (k, sin, cos) about its axis point, which
// moves along the arc ∫₀ˢ R(ω u/L) a du = s·(a∥ + sinc θ·a⊥ + ((1 − cos θ)/θ)·(k × a⊥)),
// θ = |ω|·s/L.
struct PlantFrame {
    k: vec3<f32>,
    sn: f32,
    cs: f32,
    on_axis: vec3<f32>,
    arc: vec3<f32>,
    moves: bool,
}

fn plant_frame(o: PlantBoneState, s: f32) -> PlantFrame {
    var f: PlantFrame;
    let w = length(o.omega);
    let theta = w * (s / o.len);
    f.moves = theta > 1e-7;
    if f.moves {
        f.k = o.omega / w;
        f.sn = sin(theta);
        f.cs = cos(theta);
        let a_par = f.k * dot(o.axis, f.k);
        let a_perp = o.axis - a_par;
        let sinc = select(1.0 - (theta * theta / 6.0), f.sn / theta, theta > 1e-3);
        // (1 − cos θ)/θ = sin²θ / ((1 + cos θ)·θ), stable for small θ without another sine.
        let cosc = select(0.5 * theta, (f.sn * f.sn) / ((1.0 + f.cs) * theta), theta > 1e-3);
        f.on_axis = o.pivot + (o.axis * s);
        f.arc = o.pivot + ((a_par + (a_perp * sinc) + (cross(f.k, a_perp) * cosc)) * s);
    }
    return f;
}

fn plant_frame_point(f: PlantFrame, q: vec3<f32>) -> vec3<f32> {
    if !f.moves {
        return q;
    }
    return f.arc + plant_rotate(q - f.on_axis, f.k, f.sn, f.cs);
}

fn plant_frame_direction(f: PlantFrame, n: vec3<f32>) -> vec3<f32> {
    if !f.moves {
        return n;
    }
    return plant_rotate(n, f.k, f.sn, f.cs);
}

// The distance along bone `o` of point `p`'s projection.
fn plant_project(o: PlantBoneState, p: vec3<f32>) -> f32 {
    return clamp(dot(p - o.pivot, o.axis), 0.0, o.len);
}

// A rigid transform x ↦ R x + t, R a unit quaternion: the frames of a bone's ancestors at its
// attachment, composed by the pose pass.
struct PlantRigid {
    q: vec4<f32>,
    t: vec3<f32>,
}

fn plant_quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = 2.0 * cross(q.xyz, v);
    return v + (u * q.w) + cross(q.xyz, u);
}

fn plant_rigid_point(r: PlantRigid, p: vec3<f32>) -> vec3<f32> {
    return plant_quat_rotate(r.q, p) + r.t;
}
