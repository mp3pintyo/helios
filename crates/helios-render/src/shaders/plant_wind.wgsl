// Dynamic plant wind: UE 5.9's Dynamic Wind for Nanite foliage in its parametric mode (Mode 0;
// plant_wind.rs, helios_geometry::plant_bones). Requires common.wgsl, geometry.wgsl, wind.wgsl
// and a `plant_bones: texture_2d<f32>` at binding 150.
//
// A plant with bones (instance.plant.x != 0) moves by its skeleton: every vertex belongs to one
// bone (the cluster's skin section), every bone bends about its pivot as a circular arc — its
// frame at distance s along it turns by θ·s/L — and rides on its parent's frame where it is
// attached, down to the root, which the ground holds. A bone's rotation is a closed-form function
// of the wind, the gust at the plant, the time, and its own amplitude, frequency and phase, with
// the time functions of the procedural wind's three layers (wind.wgsl) by class: bend (stems,
// trunks, blades, fronds lean with the gusts and sway at the plant's frequency), sway (branches
// bob, up to 2.3× faster), flutter (leaves flutter and twist about their stem). So leaves turn
// about their stems, blades curve, nothing stretches (rotations and arcs keep lengths), normals
// turn with their bone, and the culling bounds of the procedural layers (instance.wind holds
// equivalent ones) still hold. Everything is a pure function of the rest vertex, its bone, the
// instance and the time: every pass places a vertex on the same bits, as for the procedural wind.
//
// Bone texels (plant_wind.rs bone_texels): pivot.xyz, parent (−1: root); axis.xyz, length;
// attachment on the parent, class + phase fraction, amplitude, frequency fraction.

const PLANT_TEXTURE_WIDTH: u32 = 1024u;
const PLANT_BONE_TEXELS: u32 = 3u;
// Longest bone chain walked (trunk → branch → branch → leaf is 4).
const PLANT_MAX_DEPTH: u32 = 6u;
const PLANT_HAS_SKIN: u32 = 0x10000000u;

fn plant_texel(t: u32) -> vec4<f32> {
    return textureLoad(plant_bones, vec2<u32>(t & (PLANT_TEXTURE_WIDTH - 1u), t >> 10u), 0);
}

// Whether the instance moves by its bones (with the wind blowing on it: wind_strength > 0).
fn plant_animated(instance: Instance, strength: f32) -> bool {
    return strength > 0.0 && instance.plant.x != 0u;
}

// The bone of cluster vertex `index`: the first joint of its skin (bone 0 without skin, e.g. an
// assembly part without bones of its own, which then hangs from its table's first bone).
fn plant_vertex_bone(c: Cluster, base: u32, index: u32) -> u32 {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let extension = cluster_extension(header, widths);
    if (extension & PLANT_HAS_SKIN) == 0u {
        return 0u;
    }
    let section = vertex_stream(header, widths) + ((attribute_bit(c, widths, extension, cluster_vertex_count(c)) + 31u) >> 5u);
    let d = geometry[section];
    let joint_bits = (d >> 16u) & 31u;
    let weight_bits = (d >> 21u) & 15u;
    let influences = ((d >> 25u) & 3u) + 1u;
    let stride = (influences * joint_bits) + ((influences - 1u) * weight_bits);
    return (d & 0xffffu) + read_bits(section + 1u, index * stride, joint_bits);
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
    // wind.params.x: distance per metre of motion at which it covers WIND_MIN_PIXELS (4).
    let limit = 2.5 * a.response.z * 4.0 * wind.params.x;
    let d = distance(origin, wind.camera.xyz);
    a.leaves = saturate(4.0 * (limit - d) / max(limit, 1e-3));
    return a;
}

// The rotation vector of a whole bone (axis × angle, object space), from its texel 2 fields.
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
        return cross(axis, force) * (amplitude * a.response.y);
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

// The pose of object-space point `p` (with direction `n`) bound to `bone` of the table at texel
// `table`: its own bone's arc at its projection, then each parent's frame at the attachment.
fn plant_pose(table: u32, a: PlantAir, bone: u32, p: vec3<f32>, n: vec3<f32>) -> PlantPose {
    var q = p;
    var dir = n;
    var b = bone;
    var s = 0.0;
    for (var depth = 0u; depth < PLANT_MAX_DEPTH; depth++) {
        let t = table + (b * PLANT_BONE_TEXELS);
        let t0 = plant_texel(t);
        let t2 = plant_texel(t + 2u);
        if depth == 0u && a.leaves <= 0.0 && u32(t2.y) == 2u && t0.w >= 0.0 {
            // A distant leaf: rigid on its branch at its attachment (exactly its pose at zero
            // flutter).
            s = t2.x;
            b = u32(t0.w);
            continue;
        }
        let t1 = plant_texel(t + 1u);
        let pivot = t0.xyz;
        let axis = t1.xyz;
        let len = t1.w;
        if depth == 0u {
            s = clamp(dot(p - pivot, axis), 0.0, len);
        }
        let omega = plant_bone_rotation(a, axis, t2.y, t2.z, t2.w);
        let w = length(omega);
        // The frame at s turns by θ = |ω|·s/L about k; its axis point moves along the arc
        // ∫₀ˢ R(ω u/L) a du = s·(a∥ + sinc θ·a⊥ + ((1 − cos θ)/θ)·(k × a⊥)).
        let theta = w * (s / len);
        if theta > 1e-7 {
            let k = omega / w;
            let sn = sin(theta);
            let cs = cos(theta);
            let half = sin(0.5 * theta);
            let a_par = k * dot(axis, k);
            let a_perp = axis - a_par;
            let sinc = select(1.0 - (theta * theta / 6.0), sn / theta, theta > 1e-3);
            let cosc = select(0.5 * theta, (2.0 * half * half) / theta, theta > 1e-3);
            let on_axis = pivot + (axis * s);
            let arc = pivot + ((a_par + (a_perp * sinc) + (cross(k, a_perp) * cosc)) * s);
            q = arc + plant_rotate(q - on_axis, k, sn, cs);
            dir = plant_rotate(dir, k, sn, cs);
        }
        if t0.w < 0.0 {
            break;
        }
        s = t2.x;
        b = u32(t0.w);
    }
    return PlantPose(q, dir);
}

// World position of cluster vertex `index` (object position `object`) of `instance` this frame:
// by its bones for plants, else the procedural wind (`strength` from wind_strength).
fn plant_world_position(instance: Instance, strength: f32, c: Cluster, base: u32, index: u32, object: vec3<f32>) -> vec3<f32> {
    if plant_animated(instance, strength) {
        let a = plant_air(instance, strength, wind_origin(instance), wind.time.x);
        let pose = plant_pose(instance.plant.y, a, plant_vertex_bone(c, base, index), object, vec3<f32>(0.0));
        return (instance.world * vec4<f32>(pose.position, 1.0)).xyz;
    }
    return wind_world_position(instance, strength, object, vertex_wind_weight(c, base, index, strength));
}

// Displacement (world) of a voxel brick of a plant: the pose of the brick's centre, with the bone
// the brick follows; zero for bricks without bones. Voxels move rigidly with their brick.
fn plant_brick_offset(instance: Instance, a: PlantAir, bone: u32, center: vec3<f32>, world: mat4x4<f32>) -> vec3<f32> {
    let pose = plant_pose(instance.plant.y, a, bone, center, vec3<f32>(0.0));
    return (world * vec4<f32>(pose.position - center, 0.0)).xyz;
}

// The plant bone of brick `b` of a voxel cluster (header word 4 bit 31: the bricks carry bones,
// 16 bits each after the voxels' words), or 0xffffffff.
fn plant_voxel_bone(header: u32, attributes: u32, b: u32) -> u32 {
    let w = geometry[header + 4u];
    if (w & 0x80000000u) == 0u {
        return 0xffffffffu;
    }
    let voxels = (w >> 8u) & 0x7fffu;
    let word = geometry[attributes + voxels + (b >> 1u)];
    return (word >> (16u * (b & 1u))) & 0xffffu;
}
