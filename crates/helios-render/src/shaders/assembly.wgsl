// Assembly expansion (Nanite assemblies; see helios_geometry::assembly and assembly.rs): before
// any culling, each instance of an assembly decides — with the main view's LOD test — whether
// its base is detailed enough. If not, it is drawn as its parts: one instance record per part is
// written into the tail of the instance buffer (part slots), its transform the assembly's times
// the part's, and the base is flagged INSTANCE_EXPANDED so culling skips it. Every view
// (main, shadow levels, local lights) then culls the same instances, so a shadow never shows
// the base while the view shows the parts. The part records carry the assembly's bounds and
// plant origin, so the wind moves the parts exactly as it would move one mesh of the whole
// plant, and per-part persistent visibility bits, so two-pass occlusion works whichever slot a
// part lands in.

#include "common.wgsl"

struct AssemblyParams {
    static_count: u32,   // instances of the scene; part slots follow
    part_capacity: u32,
    instance_count: u32, // assembly instances
    // Multiplies the LOD threshold of the expansion test, raised by the CPU while the parts
    // requested approach the slots (assembly.rs), so the nearest assemblies keep theirs.
    threshold_scale: f32,
}

struct Assembly {
    first_part: u32,
    part_count: u32,
    visibility_bits: u32,
    transition_error: f32,  // error of the base's finest level
    lod_sphere: vec4<f32>,  // bounds of the parts' full detail (assembly space)
}

struct AssemblyPart {
    transform: mat4x4<f32>,
    page_offset: u32,
    page_count: u32,
    cluster_offset: u32,
    cluster_count: u32,
    visibility: u32,        // first visibility bit within the assembly's
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct AssemblyInstance {
    instance: u32,
    assembly: u32,
    visibility_base: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<uniform> params: AssemblyParams;
@group(0) @binding(2) var<storage, read_write> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> assembly_instances: array<AssemblyInstance>;
@group(0) @binding(4) var<storage, read> assemblies: array<Assembly>;
@group(0) @binding(5) var<storage, read> parts: array<AssemblyPart>;
// [0]: part slots handed out this frame; [4..7): the instance culling dispatch (cull.wgsl's
// cull_instances over the scene's instances and the part slots in use).
@group(0) @binding(6) var<storage, read_write> counter: array<atomic<u32>>;

var<workgroup> first_slot: u32;
// The assembly reserved part slots this frame, and it fits in them (expanded).
var<workgroup> reserved: bool;
var<workgroup> expanded: bool;

@compute @workgroup_size(64)
fn expand(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let index = wid.x + wid.y * groups.x;
    if index >= params.instance_count {
        return;
    }
    let a = assembly_instances[index];
    let assembly = assemblies[a.assembly];
    let base = instances[a.instance];
    if lid == 0u {
        // The LOD test of the base's finest level (cull.wgsl's projected error for the main
        // view), at the parts' nearest point.
        let sphere = transform_sphere(base.world, assembly.lod_sphere, base.max_scale);
        let d = distance(sphere.xyz, view.position.xyz) - sphere.w;
        reserved = d <= 0.0 || (assembly.transition_error * base.max_scale * view.lod_scale) > (view.lod_threshold * params.threshold_scale * d);
        expanded = false;
        if reserved {
            // Out of part slots: the base stands in (coarser, but whole).
            first_slot = atomicAdd(&counter[0], assembly.part_count);
            expanded = first_slot + assembly.part_count <= params.part_capacity;
        }
        instances[a.instance].flags = select(base.flags & ~INSTANCE_EXPANDED, base.flags | INSTANCE_EXPANDED, expanded);
    }
    workgroupBarrier();
    if !expanded {
        // Slots it reserved but cannot fill draw nothing: culling visits every slot handed out,
        // and they still hold an older frame's parts.
        for (var p = lid; reserved && first_slot + p < params.part_capacity && p < assembly.part_count; p += 64u) {
            instances[params.static_count + first_slot + p].page_count = 0u;
        }
        return;
    }
    let origin = vec3<f32>(base.normal_matrix_0.w, base.normal_matrix_1.w, base.normal_matrix_2.w);
    for (var p = lid; p < assembly.part_count; p += 64u) {
        let part = parts[assembly.first_part + p];
        var record = base;
        record.world = base.world * part.transform;
        record.prev_world = base.prev_world * part.transform;
        // Inverse transpose of the linear part from its cofactors, scaled like the CPU's.
        let m = mat3x3<f32>(record.world[0].xyz, record.world[1].xyz, record.world[2].xyz);
        let c0 = cross(m[1], m[2]);
        let c1 = cross(m[2], m[0]);
        let c2 = cross(m[0], m[1]);
        let scale = max(length(m[0]), max(length(m[1]), length(m[2])));
        let normal_scale = scale / dot(m[0], c0);
        record.normal_matrix_0 = vec4<f32>(c0 * normal_scale, origin.x);
        record.normal_matrix_1 = vec4<f32>(c1 * normal_scale, origin.y);
        record.normal_matrix_2 = vec4<f32>(c2 * normal_scale, origin.z);
        // The assembly's bounds stay (wind strength and reach are the plant's); pages and
        // clusters are culled with the part's transform.
        record.page_offset = part.page_offset;
        record.page_count = part.page_count;
        record.cluster_offset = part.cluster_offset;
        record.cluster_count = part.cluster_count;
        record.visibility_offset = a.visibility_base + part.visibility;
        record.max_scale = scale;
        record.flags = (base.flags & (INSTANCE_CAST_SHADOW | INSTANCE_ALPHA_MASKED | INSTANCE_TRANSLUCENT)) | INSTANCE_PART;
        instances[params.static_count + first_slot + p] = record;
    }
}

// Every view culls the scene's instances and only the part slots handed out.
@compute @workgroup_size(1)
fn finish() {
    let count = params.static_count + min(atomicLoad(&counter[0]), params.part_capacity);
    atomicStore(&counter[4], (count + 63u) / 64u);
    atomicStore(&counter[5], 1u);
    atomicStore(&counter[6], 1u);
}
