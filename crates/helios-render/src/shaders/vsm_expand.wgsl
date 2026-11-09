// Turns a clipmap level's visible clusters into (cluster, physical page) draws for every dirty
// page the cluster overlaps — the virtual shadow map equivalent of a per-page viewport. Only the
// part of the cluster's page rectangle inside the level's dirty rectangle is visited, and pages
// are looked up only where the dirty bit mask (vsm.wgsl) is set.
//
// Culling bins clusters by triangle size like the main view: hardware clusters become
// instanced page draws, software clusters page items for the compute rasteriser
// (vsm_raster.wgsl). Both lists share the draws buffer — hardware items grow from the front,
// software items from the back:
//   [0, 4)   hardware draw args        [4] hardware count     [5] software count
//   [8, 11)  software raster dispatch  [12] draws emitted, before clamping to the capacity
//   [13]     dynamic-only pages        [14] split pages (statistics)
//   entries (instance, cluster, physical | level << 16, tag) from word 16, `draw_capacity` of
//   them.
// A second list from word SPLIT_DRAWS (same header layout, SPLIT_CAPACITY entries) holds the
// hardware draws of dynamic casters into split pages: they are drawn after the static layer was
// captured (vsm_static.wgsl), and software clusters join them there.
//
// Static casters skip pages refreshed for their dynamic casters only; software-rasterised
// dynamic casters of those pages go to the page's scratch tile, merged over the static layer by
// the page clear (vsm_draw.wgsl). The item's physical page is then the scratch tile's index.

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_local_common.wgsl"
#include "wind.wgsl"

const DRAW_HEADER: u32 = 16u; // words before the first draw entry
const MAX_DISPATCH: u32 = 65535u;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<uniform> view: CullView;
@group(0) @binding(2) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> state: array<u32>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(6) var<storage, read> entries: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> draws: array<atomic<u32>>;
// Per-level dirty rectangles and the dirty page bit mask, and the local lights' dirty boxes and
// face mip rectangles (vsm.wgsl).
@group(0) @binding(8) var<storage, read> info: array<i32>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;
@group(0) @binding(65) var<uniform> wind: Wind;

const MAX_PAGES_PER_CLUSTER: i32 = 256;

// The split pages' list (vsm.rs).
override SPLIT_DRAWS: u32 = 0u;
override SPLIT_CAPACITY: u32 = 0u;

// Draw lists.
const LIST_HARDWARE: u32 = 0u;
const LIST_SOFTWARE: u32 = 1u;
const LIST_SPLIT: u32 = 2u;

fn emit(list: u32, item: vec4<u32>) {
    let base = select(0u, SPLIT_DRAWS, list == LIST_SPLIT);
    let slot = atomicAdd(&draws[base + select(4u, 5u, list == LIST_SOFTWARE)], 1u);
    let capacity = select(vsm.draw_capacity, SPLIT_CAPACITY, list == LIST_SPLIT);
    if slot >= capacity {
        return;
    }
    let o = base + DRAW_HEADER + select(slot, capacity - 1u - slot, list == LIST_SOFTWARE) * 4u;
    atomicStore(&draws[o], item.x);
    atomicStore(&draws[o + 1u], item.y);
    atomicStore(&draws[o + 2u], item.z);
    atomicStore(&draws[o + 3u], item.w);
}

// Emits the draw of a cluster into a dirty page (physical page `physical`, which `level` and
// `tag` address) according to the page's kind this frame (vsm_common.wgsl).
fn emit_page_draw(item: vec4<u32>, dynamic: bool, software: bool, physical: u32, level: u32, tag: u32) {
    let state = bitcast<u32>(info[INFO_PAGE_STATE + physical]);
    let kind = page_kind(state);
    var list = select(LIST_HARDWARE, LIST_SOFTWARE, software);
    var target_page = physical;
    if kind == PAGE_DYNAMIC {
        if !dynamic {
            return; // in the page's static layer
        }
        let scratch = page_scratch(state);
        if scratch == NO_SCRATCH {
            list = LIST_HARDWARE;
        } else if software {
            target_page = vsm.physical_count + scratch;
        }
    } else if kind == PAGE_SPLIT && dynamic {
        list = LIST_SPLIT;
    }
    emit(list, vec4<u32>(item.x, item.y, target_page | (level << 16u), tag));
}

@compute @workgroup_size(64)
fn expand(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = gid.x + gid.y * groups.x * 64u;
    // Visible cluster counts as clamped by prepare_early_draw (cull_args.wgsl).
    let hw = state[4];
    if index >= hw + state[5] {
        return;
    }
    let item = visible_clusters[select(view.cluster_capacity - 1u - (index - hw), index, index < hw)];
    let software = item.z != 0u;
    let instance = instances[item.x];
    // Padded like in culling: the cluster reaches every page its wind-displaced triangles touch.
    let strength = wind_strength(instance);
    let dynamic = dynamic_caster(strength);
    let sphere = wind_padded(instance, strength, transform_sphere(instance.world, clusters[item.y].bounds, instance.max_scale));
    let level = u32(view.vsm.x);
    let page_world = vsm.level_info[level].x;
    let ls = (vsm.light_view * vec4<f32>(sphere.xyz, 1.0)).xyz;
    let v = vec2<f32>(ls.x, -ls.y) / page_world;
    let r = sphere.w / page_world;
    let origin = vsm.level_origin[level].xy;
    let n = i32(vsm.level_pages);
    // Static casters only go to the pages rendered in full.
    let rect = level * 8u + select(4u, 0u, dynamic);
    let mask = select(INFO_FULL_MASK, INFO_DIRTY_MASK, dynamic);
    let lo = max(max(vec2<i32>(floor(v - r)), origin), vec2<i32>(info[rect], info[rect + 1u]));
    let hi = min(min(vec2<i32>(floor(v + r)), origin + n - 1), vec2<i32>(info[rect + 2u], info[rect + 3u]));
    if any(hi < lo) {
        return;
    }
    var visited = 0;
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            visited++;
            if visited > MAX_PAGES_PER_CLUSTER {
                return;
            }
            let wrapped = ((vec2<i32>(x, y) % n) + n) % n;
            let index = level * vsm.level_pages * vsm.level_pages + u32(wrapped.y) * vsm.level_pages + u32(wrapped.x);
            if (bitcast<u32>(info[mask + index / 32u]) & (1u << (index % 32u))) == 0u {
                continue;
            }
            let e = entries[index];
            if e.x == INVALID_ID || e.w != vsm.frame || any(decode_tag(e.y) != vec2<i32>(x, y)) {
                continue;
            }
            emit_page_draw(item, dynamic, software, e.x, level, e.y);
        }
    }
}

// Local lights (vsm_local.rs) share one culling view, restricted to boxes around each light's
// dirty pages. Each visible cluster goes to every light whose box it overlaps, and there to the
// dirty pages under its bounding square on every face and mip, within the face mip's dirty
// page rectangle. Software items are projected per vertex and bounded in size
// (vsm_raster.wgsl): clusters reaching behind a light's near plane or spanning more than
// LOCAL_SW_MAX_TEXELS take the hardware path, which clips.
const LOCAL_SW_MAX_TEXELS: f32 = 80.0;
const MAX_LOCAL_PAGES_PER_CLUSTER: i32 = 1024;

@compute @workgroup_size(64)
fn expand_local(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = gid.x + gid.y * groups.x * 64u;
    let hw = state[4];
    if index >= hw + state[5] {
        return;
    }
    let item = visible_clusters[select(view.cluster_capacity - 1u - (index - hw), index, index < hw)];
    let instance = instances[item.x];
    // Padded like in culling: the cluster reaches every page its wind-displaced triangles touch.
    let strength = wind_strength(instance);
    let dynamic = dynamic_caster(strength);
    let sphere = wind_padded(instance, strength, transform_sphere(instance.world, clusters[item.y].bounds, instance.max_scale));
    var visited = 0;
    for (var slot = 0u; slot < LOCAL_SLOTS; slot++) {
        // Static casters: the box of the light's pages rendered in full.
        let o = LOCAL_INFO_SLOTS + slot * LOCAL_INFO_SLOT_WORDS + select(12u, 0u, dynamic);
        let lo = vec3<f32>(f32(info[o]), f32(info[o + 1u]), f32(info[o + 2u])) * view.vsm.y;
        let hi = vec3<f32>(f32(info[o + 3u]), f32(info[o + 4u]), f32(info[o + 5u])) * view.vsm.y;
        if any(hi < lo) || distance(clamp(sphere.xyz, lo, hi), sphere.xyz) > sphere.w {
            continue;
        }
        visited = expand_into_light(item, dynamic, sphere, slot, visited);
        if visited > MAX_LOCAL_PAGES_PER_CLUSTER {
            return;
        }
    }
}

// Emits a cluster's draws into the dirty pages of one light; returns the running count of
// pages visited.
fn expand_into_light(item: vec4<u32>, dynamic: bool, sphere: vec4<f32>, slot: u32, visited_before: i32) -> i32 {
    var visited = visited_before;
    let software = item.z != 0u;
    let near = local_vsm.slots[slot].params.y;
    for (var face = 0u; f32(face) < local_vsm.slots[slot].params.x; face++) {
        let r = local_sphere_rect(slot, face, sphere);
        if !r.valid {
            continue;
        }
        let uv_lo = r.lo;
        let uv_hi = r.hi;
        let front = r.front;
        let view_index = slot * LOCAL_FACES + face;
        for (var mip = 0u; mip < LOCAL_MIPS; mip++) {
            let rect = LOCAL_INFO_RECTS + (view_index * LOCAL_MIPS + mip) * 4u;
            let pages = f32(local_mip_pages(mip));
            let page_lo = max(vec2<i32>(floor(uv_lo * pages)), vec2<i32>(info[rect], info[rect + 1u]));
            let page_hi = min(vec2<i32>(floor(uv_hi * pages)), vec2<i32>(info[rect + 2u], info[rect + 3u]));
            let extent = (uv_hi - uv_lo) * pages * f32(vsm.page_texels);
            let sw = software && front > 2.0 * near && all(extent <= vec2<f32>(LOCAL_SW_MAX_TEXELS));
            let level = LOCAL_LEVEL + view_index * LOCAL_MIPS + mip;
            for (var y = page_lo.y; y <= page_hi.y; y++) {
                for (var x = page_lo.x; x <= page_hi.x; x++) {
                    visited++;
                    if visited > MAX_LOCAL_PAGES_PER_CLUSTER {
                        return visited;
                    }
                    let page = vec2<u32>(u32(x), u32(y));
                    let e = entries[local_entry(slot, face, mip, page)];
                    if e.x == INVALID_ID || e.w != vsm.frame {
                        continue;
                    }
                    emit_page_draw(item, dynamic, sw, e.x, level, page.x | (page.y << 16u));
                }
            }
        }
    }
    return visited;
}

@compute @workgroup_size(1)
fn draw_args() {
    // Statistics.
    atomicStore(&draws[12], atomicLoad(&draws[4]) + atomicLoad(&draws[5]) + atomicLoad(&draws[SPLIT_DRAWS + 4u]));
    atomicStore(&draws[13], u32(info[INFO_COUNTERS]));
    atomicStore(&draws[14], u32(info[INFO_COUNTERS + 1u]));
    atomicStore(&draws[SPLIT_DRAWS], MAX_CLUSTER_TRIANGLES * 3u);
    atomicStore(&draws[SPLIT_DRAWS + 1u], min(atomicLoad(&draws[SPLIT_DRAWS + 4u]), SPLIT_CAPACITY));
    atomicStore(&draws[SPLIT_DRAWS + 2u], 0u);
    atomicStore(&draws[SPLIT_DRAWS + 3u], 0u);
    let hw = min(atomicLoad(&draws[4]), vsm.draw_capacity);
    let sw = min(atomicLoad(&draws[5]), vsm.draw_capacity - hw);
    atomicStore(&draws[0], MAX_CLUSTER_TRIANGLES * 3u);
    atomicStore(&draws[1], hw);
    atomicStore(&draws[2], 0u);
    atomicStore(&draws[3], 0u);
    // One software-raster workgroup per (cluster, page) item, 2D beyond the dispatch limit.
    atomicStore(&draws[5], sw);
    atomicStore(&draws[8], min(sw, MAX_DISPATCH));
    atomicStore(&draws[9], (sw + MAX_DISPATCH - 1u) / MAX_DISPATCH);
    atomicStore(&draws[10], 1u);
}
