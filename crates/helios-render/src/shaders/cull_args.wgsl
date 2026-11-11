// Single-thread passes that turn culling counters into indirect dispatch / draw arguments.
//
// args layout (u32):
//   [0..4)   page-cull dispatch (x, y, z, -)
//   [4..8)   cluster-cull dispatch
//   [8..12)  early hardware draw (vertex_count, instance_count, first_vertex, first_instance)
//   [12..16) late hardware draw
//   [16..20) early software raster dispatch
//   [20..24) late software raster dispatch
//   [24..28) early alpha-masked hardware draw
//   [28..32) late alpha-masked hardware draw
// state layout: see CullState in cull.wgsl.

#include "common.wgsl"

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read_write> state: array<u32>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;

const MAX_DISPATCH: u32 = 65535u;

fn write_dispatch(offset: u32, groups: u32) {
    args[offset] = min(groups, MAX_DISPATCH);
    args[offset + 1u] = (groups + MAX_DISPATCH - 1u) / MAX_DISPATCH;
    args[offset + 2u] = 1u;
}

@compute @workgroup_size(1)
fn prepare_page_dispatch() {
    // A workgroup per batch, then one per 64 page candidates (cull.wgsl).
    let candidates = min(state[13], view.batch_capacity * 2u);
    write_dispatch(0u, min(state[0], view.batch_capacity) + (candidates + 63u) / 64u);
}

@compute @workgroup_size(1)
fn prepare_cluster_dispatch() {
    // A workgroup per page work item, then one per 64 cluster candidates (cull.wgsl).
    let candidates = min(state[12], view.work_capacity * 2u);
    write_dispatch(4u, min(state[1], view.work_capacity) + (candidates + 63u) / 64u);
}

fn visible_counts() -> vec2<u32> {
    let hw = min(state[2], view.cluster_capacity);
    let sw = min(state[3], view.cluster_capacity - hw);
    return vec2<u32>(hw, sw);
}

fn masked_count() -> u32 {
    return min(state[9], view.masked_capacity);
}

fn write_draw(offset: u32, instances: u32) {
    args[offset] = MAX_CLUSTER_TRIANGLES * 3u;
    args[offset + 1u] = instances;
    args[offset + 2u] = 0u;
    args[offset + 3u] = 0u;
}

@compute @workgroup_size(1)
fn prepare_early_draw() {
    let counts = visible_counts();
    state[4] = counts.x;
    state[5] = counts.y;
    state[10] = masked_count();
    write_draw(8u, counts.x);
    write_draw(24u, state[10]);
    if view.kind == 0u {
        write_dispatch(16u, counts.y);       // one software-raster workgroup per cluster
    } else {
        // Shadow views: page expansion threads over hardware and software clusters alike.
        write_dispatch(16u, (counts.x + counts.y + 63u) / 64u);
    }
}

@compute @workgroup_size(1)
fn prepare_late_draw() {
    let counts = visible_counts();
    write_draw(12u, counts.x - min(state[4], counts.x));
    let masked = masked_count();
    write_draw(28u, masked - min(state[10], masked));
    write_dispatch(20u, counts.y - min(state[5], counts.y));
}
