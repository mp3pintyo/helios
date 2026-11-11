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
//
// Culling is hierarchical: views never test the part records as instances. Each assembly
// instance owns one more record, between the scene's instances and the part slots, that
// stands for all its parts (INSTANCE_PART_LIST): the parts' shared bounds, wind and flags, and
// as its "pages" the (part, page) pairs of its parts, `page_stride` per part. Instance culling
// tests it once per view; the page pass resolves each of its lanes to a part's page
// (cull.wgsl), so the parts' pages are culled 64 to a workgroup instead of one workgroup per
// part. Non-expanded assemblies hide their record.

#include "common.wgsl"

struct AssemblyParams {
    static_count: u32,   // instances of the scene; then one record per assembly instance
    part_capacity: u32,
    instance_count: u32, // assembly instances; part slots follow their records
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
    page_stride: u32,       // the most pages of any of its parts
    part_scale: f32,        // the largest scale of any part's transform
    _pad0: u32,
    _pad1: u32,
}

struct AssemblyPart {
    transform: mat4x4<f32>,
    page_offset: u32,
    page_count: u32,
    cluster_offset: u32,
    cluster_count: u32,
    visibility: u32,        // first visibility bit within the assembly's
    bones: u32,             // first texel of its plant bone table (plant_wind.wgsl; 0: none)
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
// [0]: part slots handed out this frame (read back for the slot budget).
@group(0) @binding(6) var<storage, read_write> counter: array<atomic<u32>>;
// [0]: the scene's instance count; [1 + slot]: the scene instance a part slot belongs to (the
// editor's picking and selection outline map parts to it, part_owner.wgsl).
@group(0) @binding(7) var<storage, read_write> part_owners: array<u32>;

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
    let record_index = params.static_count + index;
    let part_base = params.static_count + params.instance_count;
    let origin = vec3<f32>(base.normal_matrix_0.w, base.normal_matrix_1.w, base.normal_matrix_2.w);
    // Movable (dynamic shadow casters) and displaced (tessellated) like the base.
    let part_flags = (base.flags & (INSTANCE_CAST_SHADOW | INSTANCE_ALPHA_MASKED | INSTANCE_TRANSLUCENT | INSTANCE_MOVABLE | INSTANCE_DISPLACED | INSTANCE_ANIMATED)) | INSTANCE_PART;
    if lid == 0u {
        // The LOD test of the base's finest level (cull.wgsl's projected error for the main
        // view), at the parts' nearest point.
        let sphere = transform_sphere(base.world, assembly.lod_sphere, base.max_scale);
        let d = distance(sphere.xyz, view.position.xyz) - sphere.w;
        // A hidden assembly (deleted in an editor) is culled whole: no parts.
        let hidden = (base.flags & INSTANCE_HIDDEN) != 0u;
        reserved = !hidden && (d <= 0.0 || (assembly.transition_error * base.max_scale * view.lod_scale) > (view.lod_threshold * params.threshold_scale * d));
        expanded = false;
        if reserved {
            // Out of part slots: the base stands in (coarser, but whole).
            first_slot = atomicAdd(&counter[0], assembly.part_count);
            expanded = first_slot + assembly.part_count <= params.part_capacity;
        }
        instances[a.instance].flags = select(base.flags & ~INSTANCE_EXPANDED, base.flags | INSTANCE_EXPANDED, expanded);
        if expanded {
            // The parts' culling record: what every part shares (bounds, wind, plant origin,
            // flags), its parts' slots and page stride, and a scale bounding theirs (for the
            // displacement padding).
            var record = base;
            record.page_offset = part_base + first_slot;
            record.page_count = assembly.part_count * assembly.page_stride;
            record.cluster_offset = assembly.page_stride;
            record.cluster_count = 0u;
            record.max_scale = base.max_scale * assembly.part_scale;
            record.flags = part_flags | INSTANCE_PART_LIST;
            instances[record_index] = record;
        } else {
            instances[record_index].flags = INSTANCE_HIDDEN;
        }
    }
    workgroupBarrier();
    if !expanded {
        // Slots it reserved but cannot fill are never culled: only expanded records list theirs.
        return;
    }
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
        record.flags = part_flags;
        // A plant's part bends with the bone it hangs from (its own table, in its space).
        if part.bones != 0u {
            record.plant.y = part.bones;
        } else {
            record.plant.x = 0u;
        }
        instances[part_base + first_slot + p] = record;
        part_owners[1u + first_slot + p] = a.instance;
    }
}
