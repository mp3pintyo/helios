// Dynamic wind's pose pass (plant_pose.rs): each plant instance's skeleton evaluated once per
// frame, as UE evaluates bones once per instance, so that the passes placing geometry
// (plant_wind.wgsl) do one lookup and one transform per vertex instead of walking its bone chain.
//
// `allocate`: every plant instance (the scene's, and the assembly parts the expansion wrote this
// frame) whose ranked bones move — the branches within their distance fade, bones that never
// fade, any bone of a promoted plant — takes one pose entry per ranked bone from a budgeted pool;
// its first entry + 1 goes to instance.plant.w (0: none, the vertex pose then rides on the root).
// `evaluate`: one thread per entry composes the frames of the bone's ancestors at its
// attachment into one rigid transform and stores it with the bone's own rotation, for this frame
// and the previous one (motion vectors). Plants beyond the fades take no entries, and leaves —
// most of a tree's bones — never do: a vertex evaluates its own leaf's flutter itself.

#include "common.wgsl"
#include "wind.wgsl"
#include "plant_motion.wgsl"

struct PoseParams {
    plant_count: u32,     // entries of `plants`
    part_first: u32,      // the first assembly part slot (instance index)
    part_capacity: u32,
    capacity: u32,        // pose entries in the pool
}

@group(0) @binding(0) var<uniform> params: PoseParams;
// The scene's instances with plant bones.
@group(0) @binding(1) var<storage, read> plants: array<u32>;
@group(0) @binding(2) var<storage, read_write> instances: array<Instance>;
// [0]: entries handed out this frame.
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
// Per entry: the instance it belongs to.
@group(0) @binding(4) var<storage, read_write> owners: array<u32>;
// The simulation's state (plant_sim.wgsl): per (slot, bone) δ, its rate, the previous frame's δ.
@group(0) @binding(5) var<storage, read> sim_state: array<vec4<f32>>;
// [0]: the assembly part slots handed out this frame (assembly.wgsl).
@group(0) @binding(6) var<storage, read> part_counter: array<u32>;
@group(0) @binding(7) var pose_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(170) var plant_bones: texture_2d<f32>;
// The evaluation's dispatch (x, 1, 1); bound to `allocate` only (a spare buffer for `evaluate`,
// whose indirect buffer it is).
@group(0) @binding(8) var<storage, read_write> args: array<atomic<u32>>;

const POSE_FIRST: u32 = 256u * 1024u;  // after the simulation's rows (plant_wind.wgsl)
const POSE_TEXELS: u32 = 8u;
const SIM_BONES: u32 = 512u;

@compute @workgroup_size(64)
fn allocate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + (gid.y * groups.x * 64u);
    if i == 0u {
        atomicStore(&args[1], 1u);
        atomicStore(&args[2], 1u);
    }
    var index = 0u;
    if i < params.plant_count {
        index = plants[i];
    } else {
        let slot = i - params.plant_count;
        if slot >= min(part_counter[0], params.part_capacity) {
            return;
        }
        index = params.part_first + slot;
    }
    let instance = instances[index];
    if instance.plant.x == 0u {
        return;
    }
    let strength = wind_strength(instance);
    var first = 0u;
    if strength > 0.0 {
        let header = plant_texel(instance.plant.y - 1u);
        let ranks = u32(header.y);
        let classes = u32(header.z);
        let a = plant_air(instance, strength, wind_origin(instance), wind.time.x);
        let moving = ((classes & 1u) != 0u) || (((classes & 2u) != 0u) && a.branches > 0.0) || (((classes & 4u) != 0u) && a.leaves > 0.0) || a.sim != 0u;
        if ranks > 0u && moving {
            let base = atomicAdd(&counters[0], ranks);
            if base + ranks <= params.capacity {
                first = base + 1u;
                for (var r = 0u; r < ranks; r++) {
                    owners[base + r] = index;
                }
                atomicMax(&args[0], (base + ranks + 63u) / 64u);
            }
        }
    }
    instances[index].plant.w = first;
}

// Quaternion product a·b (b first, then a).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>((b.xyz * a.w) + (a.xyz * b.w) + cross(a.xyz, b.xyz), (a.w * b.w) - dot(a.xyz, b.xyz));
}

// Bone `b`'s state (texels 0..2) with its simulated deviation.
fn bone_state(a: PlantAir, b: u32, t0: vec4<f32>, t1: vec4<f32>, t2: vec4<f32>) -> PlantBoneState {
    var o = plant_bone_state(a, t0, t1, t2);
    if a.sim != 0u && u32(t1.w) != 2u && b < SIM_BONES {
        o.omega += sim_state[((((a.sim - 1u) * SIM_BONES) + b) * 3u) + (2u * a.sim_texel)].xyz;
    }
    return o;
}

fn bone_texel(table: u32, b: u32, k: u32) -> vec4<f32> {
    return plant_texel(table + (b * PLANT_BONE_TEXELS) + k);
}

// Bone `b`'s rotation vector and the rigid transform of its ancestors' frames at its attachment.
fn pose_entry(table: u32, a: PlantAir, b: u32) -> array<vec4<f32>, 3> {
    let own = bone_state(a, b, bone_texel(table, b, 0u), bone_texel(table, b, 1u), bone_texel(table, b, 2u));
    var q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var t = vec3<f32>(0.0);
    var x = b;
    for (var depth = 0u; depth < 8u; depth++) {
        let x2 = bone_texel(table, x, 2u);
        if x2.w < 0.0 {
            break;
        }
        let s = x2.z;
        x = u32(x2.w);
        let o = bone_state(a, x, bone_texel(table, x, 0u), bone_texel(table, x, 1u), bone_texel(table, x, 2u));
        // The parent's frame at the attachment, x ↦ arc + R (x − on_axis), applied after the
        // transform so far.
        let w = length(o.omega);
        let theta = w * (s / o.len);
        if theta > 1e-7 {
            let f = plant_frame(o, s);
            let fq = vec4<f32>(f.k * sin(0.5 * theta), cos(0.5 * theta));
            let ft = f.arc - plant_quat_rotate(fq, f.on_axis);
            t = plant_quat_rotate(fq, t) + ft;
            q = quat_mul(fq, q);
        }
    }
    // Renormalised: a product of a few unit quaternions drifts by rounding only.
    q = normalize(q);
    return array<vec4<f32>, 3>(vec4<f32>(own.omega, 0.0), q, vec4<f32>(t, 0.0));
}

fn store(t: u32, v: vec4<f32>) {
    textureStore(pose_out, vec2<u32>(t & (PLANT_TEXTURE_WIDTH - 1u), t >> 10u), v);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= min(atomicLoad(&counters[0]), params.capacity) {
        return;
    }
    let index = owners[e];
    let instance = instances[index];
    if instance.plant.w == 0u || e + 1u < instance.plant.w {
        return; // (an entry no allocation of this frame wrote: never read)
    }
    let rank = e + 1u - instance.plant.w;
    let table = instance.plant.y;
    let header = plant_texel(table - 1u);
    let list = plant_texel(table + (u32(header.x) * PLANT_BONE_TEXELS) + (rank >> 2u));
    let b = u32(list[rank & 3u]);
    let strength = wind_strength(instance);
    let now = pose_entry(table, plant_air(instance, strength, wind_origin(instance), wind.time.x), b);
    let before = pose_entry(table, plant_air_previous(instance, strength), b);
    let first = POSE_FIRST + (e * POSE_TEXELS);
    store(first, now[0]);
    store(first + 1u, now[1]);
    store(first + 2u, now[2]);
    store(first + 3u, before[0]);
    store(first + 4u, before[1]);
    store(first + 5u, before[2]);
    // The bone itself: pivot and length, axis.
    let t0 = bone_texel(table, b, 0u);
    store(first + 6u, t0);
    store(first + 7u, vec4<f32>(bone_texel(table, b, 1u).xyz, 0.0));
}
