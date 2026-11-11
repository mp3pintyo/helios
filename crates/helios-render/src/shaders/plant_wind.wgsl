// Dynamic plant wind: UE 5.9's Dynamic Wind for Nanite foliage in its parametric mode (Mode 0;
// plant_wind.rs, helios_geometry::plant_bones), posing vertices. Requires common.wgsl,
// geometry.wgsl, wind.wgsl, a `plant_bones: texture_2d<f32>` at binding 170 and a
// `plant_motion: texture_2d<f32>` at 171 (plant_pose.rs: the simulated deviations of promoted
// plants, then the pose entries).
//
// A plant with bones (instance.plant.x != 0) moves by its skeleton: every vertex belongs to one
// bone (the cluster's skin section), every bone bends about its pivot as a circular arc and rides
// on its parent's frame where it is attached, down to the root, which the ground holds
// (plant_motion.wgsl). So leaves turn about their stems, blades curve, nothing stretches, normals
// turn with their bone, and the culling bounds of the procedural layers (instance.wind holds
// equivalent ones) still hold.
//
// Like UE, the skeleton is evaluated once per instance, not per vertex: the pose pass
// (plant_pose.wgsl) composes, for every bone that hangs from another and is not a leaf at the end
// of its chain (the instance's *ranked* bones: branches), the frames of all its ancestors at its
// attachment into one rigid transform, and stores it with the bone's own rotation (this frame's
// and the previous frame's) and its pivot, axis and length. A vertex fetches its bone's four
// texels and at most one pose entry, whatever the depth of its bone, and poses in at most three
// steps:
//   * a leaf at the end of its chain: its own arc (flutter, evaluated here: leaves are most of a
//     tree's bones, so they take no pose entry), then its carrier, the parent, at the leaf's
//     attachment;
//   * a ranked carrier: its arc with the rotation from the pose entry, then the entry's rigid
//     transform;
//   * a root carrier (grass blades, fern fronds, trunks: no ancestors) is evaluated here, and so
//     is the root of an instance without pose entries (beyond the branches' fade, where every
//     ranked bone is at rest): its frame where the chain hangs from it.
// Everything is a pure function of the rest vertex, its bone, the instance and the time, and the
// pose entries are read by every pass alike: passes, clusters and LOD levels sharing a vertex
// place it on the same bits, as for the procedural wind.

#include "plant_motion.wgsl"

const PLANT_HAS_SKIN: u32 = 0x10000000u;
// Pose entries (plant_pose.rs): after the simulation's rows, eight texels each: this frame's
// rotation vector, rotation (quaternion) and translation, the previous frame's, the bone's pivot
// and length, its axis.
const PLANT_POSE_FIRST: u32 = 256u * 1024u;
const PLANT_POSE_TEXELS: u32 = 8u;

// The scene has plants with bones (plant_pose.rs PlantPose::any): passes are compiled without the
// plant code otherwise, which costs the hot rasterisers and the G-buffer resolve registers (and
// occupancy) even where no plant is drawn.
override PLANTS: bool = true;

// Whether the instance moves by its bones (with the wind blowing on it: wind_strength > 0).
fn plant_animated(instance: Instance, strength: f32) -> bool {
    return PLANTS && strength > 0.0 && instance.plant.x != 0u;
}

fn plant_motion_texel(t: u32) -> vec4<f32> {
    return textureLoad(plant_motion, vec2<u32>(t & (PLANT_TEXTURE_WIDTH - 1u), t >> 10u), 0);
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

// Root `r` with its rotation evaluated here (its texels 0..2) and, for a promoted plant, its
// simulated deviation (plant_sim.wgsl; leaves keep their flutter).
fn plant_root(a: PlantAir, r: u32, t0: vec4<f32>, t1: vec4<f32>, t2: vec4<f32>) -> PlantBoneState {
    var o = plant_bone_state(a, t0, t1, t2);
    if a.sim != 0u && u32(t1.w) != 2u {
        o.omega += plant_motion_texel((((a.sim - 1u) * PLANT_TEXTURE_WIDTH) + (2u * r)) + a.sim_texel).xyz;
    }
    return o;
}

// A bone's four texels, fetched at once.
struct PlantBone {
    t0: vec4<f32>,
    t1: vec4<f32>,
    t2: vec4<f32>,
    t3: vec4<f32>,
}

fn plant_bone(table: u32, bone: u32) -> PlantBone {
    let t = table + (bone * PLANT_BONE_TEXELS);
    return PlantBone(plant_texel(t), plant_texel(t + 1u), plant_texel(t + 2u), plant_texel(t + 3u));
}

// The carrier of bone `b` (`bone` its index): its state and, from its pose entry, the rigid
// transform of its ancestors (`entry` false for a root carrier, evaluated here: the vertex's own
// root bone, a leaf's root parent, or the root of an instance without pose entries).
struct PlantCarrier {
    o: PlantBoneState,
    rigid: PlantRigid,
    entry: bool,
    s: f32,  // where on it (< 0: at each point's projection)
}

fn plant_carrier(table: u32, a: PlantAir, bone: u32, b: PlantBone) -> PlantCarrier {
    var c: PlantCarrier;
    c.entry = false;
    c.s = select(b.t2.z, -1.0, b.t3.w == 0.0);
    if b.t3.x >= 0.0 && a.pose != 0u {
        // A ranked carrier: rotation, ancestors and its pivot, axis and length from its pose
        // entry (this frame's or the previous frame's).
        let e = PLANT_POSE_FIRST + ((a.pose - 1u + u32(b.t3.x)) * PLANT_POSE_TEXELS);
        let half = e + (3u * a.sim_texel);
        let pivot = plant_motion_texel(e + 6u);
        c.o = PlantBoneState(pivot.xyz, plant_motion_texel(e + 7u).xyz, pivot.w, plant_motion_texel(half).xyz);
        c.rigid = PlantRigid(plant_motion_texel(half + 1u), plant_motion_texel(half + 2u).xyz);
        c.entry = true;
        return c;
    }
    // A root carrier: the bone itself, or the root its chain hangs from, where it hangs.
    let r = u32(b.t3.y);
    if r == bone {
        c.o = plant_root(a, r, b.t0, b.t1, b.t2);
        return c;
    }
    let t = table + (r * PLANT_BONE_TEXELS);
    c.o = plant_root(a, r, plant_texel(t), plant_texel(t + 1u), plant_texel(t + 2u));
    c.s = b.t3.z;
    return c;
}

// Whether the bone is a leaf at the end of its chain that flutters (within the leaves' fade):
// its own arc first (plant_bone_state), else none.
fn plant_leaf_moves(a: PlantAir, b: PlantBone) -> bool {
    return b.t3.w != 0.0 && a.leaves > 0.0;
}

// The pose of object-space point `p` (with direction `n`) bound to `bone` of the table at texel
// `table`.
fn plant_pose(table: u32, a: PlantAir, bone: u32, p: vec3<f32>, n: vec3<f32>) -> PlantPose {
    let b = plant_bone(table, bone);
    var q = p;
    var dir = n;
    if plant_leaf_moves(a, b) {
        let leaf = plant_bone_state(a, b.t0, b.t1, b.t2);
        let lf = plant_frame(leaf, plant_project(leaf, p));
        q = plant_frame_point(lf, q);
        dir = plant_frame_direction(lf, dir);
    }
    let c = plant_carrier(table, a, bone, b);
    let f = plant_frame(c.o, select(c.s, plant_project(c.o, p), c.s < 0.0));
    q = plant_frame_point(f, q);
    dir = plant_frame_direction(f, dir);
    if c.entry {
        q = plant_rigid_point(c.rigid, q);
        dir = plant_quat_rotate(c.rigid.q, dir);
    }
    return PlantPose(q, dir);
}

// A triangle's three corners and normals posed at once when they share their bone (a leaf, a
// blade, a branch): one fetch of its texels and pose entry, each corner its own arc on it, a
// carrier frame at a fixed attachment once for all three (G-buffer resolve).
struct PlantCorners {
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,
}

fn plant_pose_corners(table: u32, a: PlantAir, bone: u32, corners: PlantCorners) -> PlantCorners {
    let b = plant_bone(table, bone);
    var r = corners;
    if plant_leaf_moves(a, b) {
        let leaf = plant_bone_state(a, b.t0, b.t1, b.t2);
        var f = plant_frame(leaf, plant_project(leaf, corners.p0));
        r.p0 = plant_frame_point(f, r.p0);
        r.n0 = plant_frame_direction(f, r.n0);
        f = plant_frame(leaf, plant_project(leaf, corners.p1));
        r.p1 = plant_frame_point(f, r.p1);
        r.n1 = plant_frame_direction(f, r.n1);
        f = plant_frame(leaf, plant_project(leaf, corners.p2));
        r.p2 = plant_frame_point(f, r.p2);
        r.n2 = plant_frame_direction(f, r.n2);
    }
    let c = plant_carrier(table, a, bone, b);
    if c.s >= 0.0 {
        let f = plant_frame(c.o, c.s);
        r.p0 = plant_frame_point(f, r.p0);
        r.p1 = plant_frame_point(f, r.p1);
        r.p2 = plant_frame_point(f, r.p2);
        r.n0 = plant_frame_direction(f, r.n0);
        r.n1 = plant_frame_direction(f, r.n1);
        r.n2 = plant_frame_direction(f, r.n2);
    } else {
        var f = plant_frame(c.o, plant_project(c.o, corners.p0));
        r.p0 = plant_frame_point(f, r.p0);
        r.n0 = plant_frame_direction(f, r.n0);
        f = plant_frame(c.o, plant_project(c.o, corners.p1));
        r.p1 = plant_frame_point(f, r.p1);
        r.n1 = plant_frame_direction(f, r.n1);
        f = plant_frame(c.o, plant_project(c.o, corners.p2));
        r.p2 = plant_frame_point(f, r.p2);
        r.n2 = plant_frame_direction(f, r.n2);
    }
    if c.entry {
        r.p0 = plant_rigid_point(c.rigid, r.p0);
        r.p1 = plant_rigid_point(c.rigid, r.p1);
        r.p2 = plant_rigid_point(c.rigid, r.p2);
        r.n0 = plant_quat_rotate(c.rigid.q, r.n0);
        r.n1 = plant_quat_rotate(c.rigid.q, r.n1);
        r.n2 = plant_quat_rotate(c.rigid.q, r.n2);
    }
    return r;
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
// the brick follows. Voxels move rigidly with their brick.
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
