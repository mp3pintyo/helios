// Hardware visibility-buffer rasterisation of culled clusters.
//
// One instanced draw: instance = visible cluster, vertex = triangle * 3 + corner.
// Clusters with fewer than 128 triangles emit degenerate (clipped) triangles for the rest.
// The fragment output packs (visible cluster slot << 7 | triangle) into an R32Uint target.
//
// Alpha-masked clusters are drawn by a second, "programmable" pipeline (vs_masked /
// fs_masked; Nanite bins them the same way): only its fragment shader samples and discards, so
// the opaque pipeline keeps a trivial fragment shader and full early depth testing. Culling
// lists the masked clusters' slots after the visible cluster entries; the opaque draw skips them.
// Vertices of instances in the wind are displaced (wind.wgsl) and skinned instances posed
// (skinning.wgsl) as in every other pass.

#include "common.wgsl"
#include "geometry.wgsl"
#include "alpha_mask.wgsl"
#include "wind.wgsl"
#include "skinning.wgsl"
#include "plant_wind.wgsl"

// Late-pass draws start after the clusters drawn by the early pass.
override LATE: bool = false;

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> page_table: array<u32>;
@group(0) @binding(6) var<storage, read> geometry: array<u32>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(41) var alpha_masks: texture_2d_array<f32>;
@group(0) @binding(42) var alpha_mask_sampler: sampler;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;
@group(0) @binding(170) var plant_bones: texture_2d<f32>;
@group(0) @binding(171) var plant_motion: texture_2d<f32>;

// Cull state words (CullState in cull.wgsl).
const STATE_HW_EARLY: u32 = 4u;
const STATE_MASKED_EARLY: u32 = 10u;

const CULLED: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside the clip volume

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
}

struct MaskedOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) mask: u32,
}

struct Corner {
    position: vec4<f32>,
    visible: bool,
    index: u32,  // cluster-local vertex
}

fn corner(entry: vec4<u32>, cluster: Cluster, vertex_index: u32) -> Corner {
    var c = Corner(CULLED, false, 0u);
    let triangle = vertex_index / 3u;
    if triangle >= cluster_triangle_count(cluster) {
        return c;
    }
    let base = cluster_base(cluster);
    c.index = cluster_triangle(cluster, base, triangle)[vertex_index % 3u];
    let instance = instances[entry.x];
    let strength = wind_strength(instance);
    let world = plant_world_position(instance, strength, cluster, base, c.index, skinned_vertex_position(instance, cluster, base, c.index));
    c.position = view.view_proj * vec4<f32>(world, 1.0);
    c.visible = true;
    return c;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out = VertexOutput(CULLED, INVALID_ID);
    let slot = instance_index + select(0u, state[STATE_HW_EARLY], LATE);
    let entry = visible_clusters[slot];
    if (entry.z & CLUSTER_MASKED) != 0u {
        return out; // drawn by vs_masked
    }
    let c = corner(entry, clusters[entry.y], vertex_index);
    if c.visible {
        out.position = c.position;
        out.id = (slot << TRIANGLE_BITS) | (vertex_index / 3u);
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    return in.id;
}

// Slot of the index-th alpha-masked hardware cluster: four per vec4 after the entries.
fn masked_slot(index: u32) -> u32 {
    return visible_clusters[view.cluster_capacity + index / 4u][index % 4u];
}

@vertex
fn vs_masked(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> MaskedOutput {
    var out = MaskedOutput(CULLED, INVALID_ID, vec2<f32>(0.0), 0u);
    let slot = masked_slot(instance_index + select(0u, state[STATE_MASKED_EARLY], LATE));
    if slot >= view.cluster_capacity {
        return out; // its entry did not fit the visible cluster list
    }
    let entry = visible_clusters[slot];
    let cluster = clusters[entry.y];
    let c = corner(entry, cluster, vertex_index);
    if c.visible {
        let material = instances[entry.x].material;
        out.position = c.position;
        out.id = (slot << TRIANGLE_BITS) | (vertex_index / 3u);
        out.uv = alpha_mask_vertex_uv(material, cluster, cluster_base(cluster), c.index);
        out.mask = alpha_mask_of(material);
    }
    return out;
}

@fragment
fn fs_masked(in: MaskedOutput) -> @location(0) u32 {
    if !alpha_test(in.mask, in.uv, dpdx(in.uv), dpdy(in.uv)) {
        discard;
    }
    return in.id;
}
