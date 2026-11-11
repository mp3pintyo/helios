// Virtual shadow map passes.
//
//   mark       : every screen pixel requests the page (and filter neighbours) it will sample
//   mark_volume: a coarse froxel grid requests coarse pages for volumetric fog (as UE marks
//                coarse pages for volumetrics), so light shafts have shadow data in mid-air
//   mark_local : the same for local lights' shadow maps (vsm_local.rs): pixels request the
//                pages of the lights in their light cluster (one pixel per 2x2 quad), and the
//                coarsest page of each face they touch (a fallback for every receiver)
//   invalidate_local : frees the pages of faces whose origin moved, of lights that changed (before
//                the page cache)
//   info_begin / dirty_info : per-level bounding rectangles and bit masks of the dirty pages
//                (culling is restricted to them), dirty flags on entries, page-clear draw
//                arguments; for local lights a world box around each light's dirty pages, its
//                finest dirty mip and per face mip a dirty page rectangle. The page cache's dirty
//                pages are rendered in full (PAGE_FULL).
//   refresh_count / refresh_plan / refresh : the refresh queue — appends the pages whose
//                dynamic casters moved (vsm_invalidate.wgsl) to the dirty list, by priority within
//                a per-frame budget: PAGE_DYNAMIC over their static layer, or PAGE_SPLIT for pages
//                that have none yet (vsm_common.wgsl), PAGE_FULL for streamed ones
//   expand     : visible clusters of a level x overlapping dirty pages -> (cluster, page)
//                hardware draws and software raster items
//   draw_args  : indirect arguments of the page raster draws and software raster dispatch
//   clear_static / capture_static (vsm_static.wgsl): static layers of split pages
//   vsm_raster : compute rasteriser for (cluster, page) items with tiny triangles
//   vs_clear / fs_resolve : clears dirty pages to the software depth (0 = far in reverse-Z),
//                merged with their static layer
//   vs_raster / fs_raster : rasterises clusters into their pages of the physical atlas
//
// info buffer (word offsets in vsm_common.wgsl): per level 8 words, the bounding rectangle of
// its dirty pages (min page x, min y, max x, max y as i32), then that of its pages rendered in
// full; words [128, 132): clear draw args; [132, 135) counters (dynamic-only pages, split pages,
// scratch tiles); from word 136 one bit per clipmap page table entry, set for dirty pages, then
// another such mask for pages rendered in full (both cleared by vsm.rs every frame). Then per
// local light slot from word LOCAL_INFO_SLOTS (32 words): dirty box min xyz, max xyz (fixed
// point, LOCAL_BOX_SCALE units per metre), finest dirty mip, dirty count, light position and
// mip-0 texels per unit at distance 1 (f32 bits) - what the single local light culling view
// needs (cull.wgsl) - then the box (min xyz, max xyz) of its pages rendered in full, and per
// face its pages requested this frame and those rendered in full (read back); and per
// local face mip from word LOCAL_INFO_RECTS (4 words): its dirty page rectangle. Then per
// physical page its kind this frame, its static layer's entry and its pending refresh flag,
// then the motion bounds of local light pages' swaying casters (two words per physical page).
// draws buffer: see vsm_expand.wgsl.

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_local_common.wgsl"
#include "lights.wgsl"

// Matches BOX_UNIT in vsm_local.rs (the culling decodes the box with it).
const LOCAL_BOX_SCALE: f32 = 32.0;
const MAX_LEVELS: u32 = LOCAL_LEVEL; // sun clipmap levels; local slots follow them
// The page cache's header word holding the dirty page count (page_cache.wgsl).
const CACHE_DIRTY_COUNT: u32 = 2u;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> requests: array<u32>;
@group(0) @binding(4) var<storage, read_write> entries: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> info: array<atomic<i32>>;
// The page cache's lists; `refresh` appends to its dirty list.
@group(0) @binding(6) var<storage, read_write> cache_lists: array<atomic<u32>>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;
@group(0) @binding(61) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(62) var<storage, read> light_grid: array<u32>;
@group(0) @binding(63) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(64) var<storage, read_write> owner: array<u32>;

override DIRTY_START: u32 = 0u;
// Scratch tiles after the atlas' tiles in the software raster mirror (vsm.rs).
override SCRATCH_TILES: u32 = 0u;

fn request(world: vec3<f32>, level: u32) {
    let a = vsm_address(world, level);
    if !a.valid {
        return;
    }
    requests[a.entry] = vsm.frame;
    // Filter taps near a page edge read the neighbouring page.
    let edge = 3.0;
    let n = f32(vsm.page_texels);
    let ls = (vsm.light_view * vec4<f32>(world, 1.0)).xyz;
    let texel_world = vsm.level_info[level].y;
    let step = vec2<f32>(select(0.0, -1.0, a.texel.x < edge) + select(0.0, 1.0, a.texel.x > n - edge),
                         select(0.0, -1.0, a.texel.y < edge) + select(0.0, 1.0, a.texel.y > n - edge));
    if any(step != vec2<f32>(0.0)) {
        // Move along light X and light -Y by `edge` texels.
        let dx = vec3<f32>(vsm.light_view[0].x, vsm.light_view[1].x, vsm.light_view[2].x);
        let dy = vec3<f32>(vsm.light_view[0].y, vsm.light_view[1].y, vsm.light_view[2].y);
        let offsets = array<vec2<f32>, 3>(vec2<f32>(step.x, 0.0), vec2<f32>(0.0, step.y), step);
        for (var i = 0u; i < 3u; i++) {
            let o = offsets[i];
            if all(o == vec2<f32>(0.0)) {
                continue;
            }
            let p = world + (dx * o.x - dy * o.y) * edge * texel_world;
            let b = vsm_address(p, level);
            if b.valid {
                requests[b.entry] = vsm.frame;
            }
        }
    }
}

@compute @workgroup_size(8, 8)
fn mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) || vsm.enabled == 0u {
        return;
    }
    let depth = textureLoad(depth_texture, gid.xy, 0).r;
    if depth <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    request(world, vsm_level(world));
}

/// Grid of view-frustum sample points that request pages for volumetric effects.
const VOLUME_GRID: vec3<u32> = vec3<u32>(32u, 18u, 32u);
/// Volumetric samples cover ~16 pixels: request pages this many levels coarser than a pixel
/// would (matches FOG_SHADOW_LEVEL_BIAS in fog_common.wgsl).
const VOLUME_LEVEL_BIAS: u32 = 4u;

@compute @workgroup_size(4, 4, 4)
fn mark_volume(@builtin(global_invocation_id) gid: vec3<u32>) {
    // vsm.eye.w: view depth reached by volumetric effects (0 = none).
    if any(gid >= VOLUME_GRID) || vsm.enabled == 0u || vsm.eye.w <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(VOLUME_GRID.xy);
    let near = 0.5;
    let depth = near * pow(max(vsm.eye.w, near) / near, (f32(gid.z) + 0.5) / f32(VOLUME_GRID.z));
    let eye = frame.camera_position.xyz;
    let dir = view_direction(uv, frame.inv_view_proj, eye);
    let forward = -vec3<f32>(frame.view[0].z, frame.view[1].z, frame.view[2].z);
    let world = eye + dir * (depth / max(dot(dir, forward), 1e-3));
    request(world, min(vsm_level(world) + VOLUME_LEVEL_BIAS, vsm.level_count - 1u));
    if local_vsm.enabled != 0u && frame.light_count > 0u {
        let base = cluster_index(uv, depth) * CLUSTER_WORDS;
        for (var i = 0u; i < light_grid[base]; i++) {
            let slot = light_shadow_slot(lights[light_grid[base + 1u + i]]);
            if slot != INVALID_ID {
                request_local(slot, world, vec3<f32>(0.0), VOLUME_LEVEL_BIAS, 1.0);
            }
        }
    }
}

// Requests the local light page a receiver samples (see local_receiver) and the coarsest page
// of its face (the fallback while finer pages are missing, e.g. for translucent surfaces, which
// request nothing); where the filter footprint of `footprint` texels reaches a page border, also
// the pages under its corners (which may lie on other cube faces). A light away from its map's
// origin moves its filter by up to its search radius (local_filter_follow): twice the footprint.
fn request_local(slot: u32, world: vec3<f32>, n: vec3<f32>, mip_bias: u32, taps: f32) {
    let footprint = taps * select(1.0, 2.0, local_vsm.slots[slot].flags.y != 0u);
    let r = local_receiver(slot, world, n, mip_bias);
    let face = local_face(slot, r.p);
    let a = local_address(slot, face, r.p, r.mip);
    if !a.valid {
        return; // outside a spot light's face
    }
    requests[a.entry] = vsm.frame;
    requests[local_entry(slot, face, LOCAL_MIPS - 1u, vec2<u32>(0u))] = vsm.frame;
    let edge = footprint + 1.0;
    if all(a.texel > vec2<f32>(edge)) && all(a.texel < vec2<f32>(f32(vsm.page_texels) - edge)) {
        return;
    }
    let t = local_tangent(r.l) * (footprint * r.texel);
    let b = cross(r.l, t);
    for (var c = 0u; c < 4u; c++) {
        let q = r.p + select(-t, t, (c & 1u) != 0u) + select(-b, b, (c & 2u) != 0u);
        let corner_face = local_face(slot, q);
        let corner = local_address(slot, corner_face, q, r.mip);
        if corner.valid {
            requests[corner.entry] = vsm.frame;
            requests[local_entry(slot, corner_face, LOCAL_MIPS - 1u, vec2<u32>(0u))] = vsm.frame;
        }
    }
}

// One pixel per 2x2 quad marks for all four: their pages are nearly always the same (texels
// match the pixel footprint, and a page holds 128² of them), and requests near a page border
// include the filter footprint's neighbours anyway.
@compute @workgroup_size(8, 8)
fn mark_local(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    let pixel = gid.xy * 2u;
    if any(pixel >= size) || local_vsm.enabled == 0u || frame.light_count == 0u {
        return;
    }
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let base = cluster_index(uv, frame.camera_position.w / depth) * CLUSTER_WORDS;
    for (var i = 0u; i < light_grid[base]; i++) {
        let light = lights[light_grid[base + 1u + i]];
        let slot = light_shadow_slot(light);
        if slot == INVALID_ID {
            continue;
        }
        // Skip lights the lighting ignores (behind the surface beyond foliage wrap, out of range).
        let ls = sample_light(light, world);
        if dot(n, ls.l) <= -0.35 || all(ls.radiance <= vec3<f32>(0.0)) {
            continue;
        }
        request_local(slot, world, n, 0u, LOCAL_MAX_FILTER_TEXELS);
    }
}

@compute @workgroup_size(64)
fn invalidate_local(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= vsm.physical_count {
        return;
    }
    let entry = owner[gid.x];
    if entry == INVALID_ID || entry < local_vsm.entry_base {
        return;
    }
    let view = (entry - local_vsm.entry_base) / LOCAL_FACE_ENTRIES;
    if ((local_vsm.slots[view / LOCAL_FACES].flags.x >> (view % LOCAL_FACES)) & 1u) != 0u {
        owner[gid.x] = INVALID_ID;
        entries[entry].x = INVALID_ID;
    }
}


fn empty_rect(o: u32) {
    atomicStore(&info[o], 0x7fffffff);
    atomicStore(&info[o + 1u], 0x7fffffff);
    atomicStore(&info[o + 2u], -0x7fffffff);
    atomicStore(&info[o + 3u], -0x7fffffff);
}

fn grow_rect(o: u32, page: vec2<i32>) {
    atomicMin(&info[o], page.x);
    atomicMin(&info[o + 1u], page.y);
    atomicMax(&info[o + 2u], page.x);
    atomicMax(&info[o + 3u], page.y);
}

@compute @workgroup_size(64)
fn info_begin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < MAX_LEVELS {
        // All dirty pages, pages rendered in full.
        empty_rect(i * 8u);
        empty_rect(i * 8u + 4u);
        if i < 3u {
            atomicStore(&info[INFO_COUNTERS + i], 0);
        }
    } else if i < MAX_LEVELS + LOCAL_SLOTS {
        // Empty boxes, no dirty mip, no pages; the light's position and LOD scale.
        let slot = i - MAX_LEVELS;
        let o = LOCAL_INFO_SLOTS + slot * LOCAL_INFO_SLOT_WORDS;
        let position = local_vsm.slots[slot].position_range;
        for (var k = 0u; k < 3u; k++) {
            atomicStore(&info[o + k], 0x7fffffff);
            atomicStore(&info[o + 3u + k], -0x7fffffff);
            atomicStore(&info[o + 8u + k], bitcast<i32>(position[k]));
            atomicStore(&info[o + 12u + k], 0x7fffffff);
            atomicStore(&info[o + 15u + k], -0x7fffffff);
        }
        atomicStore(&info[o + 6u], 0x7fffffff);
        atomicStore(&info[o + 7u], 0);
        for (var k = 0u; k < 2u * LOCAL_FACES; k++) {
            atomicStore(&info[o + LOCAL_INFO_FACE_REQUESTED + k], 0);
        }
        atomicStore(&info[o + LOCAL_INFO_REQUESTED_ALL], 0);
        let texels_per_unit = local_vsm.slots[slot].params.z * f32(LOCAL_PAGES * vsm.page_texels) * 0.5;
        atomicStore(&info[o + 11u], bitcast<i32>(texels_per_unit));
    } else if i < MAX_LEVELS + LOCAL_SLOTS + LOCAL_SLOTS * LOCAL_FACES * LOCAL_MIPS {
        empty_rect(LOCAL_INFO_RECTS + (i - MAX_LEVELS - LOCAL_SLOTS) * 4u);
    }
}

fn grow_box(o: u32, a: vec3<i32>, b: vec3<i32>) {
    atomicMin(&info[o], a.x);
    atomicMin(&info[o + 1u], a.y);
    atomicMin(&info[o + 2u], a.z);
    atomicMax(&info[o + 3u], b.x);
    atomicMax(&info[o + 4u], b.y);
    atomicMax(&info[o + 5u], b.z);
}

// A dirty local light page: its face mip's dirty rectangle, and its slot's world box (and, if
// static casters are rendered too, the box of pages rendered in full), which bounds the page's
// pyramid from the light out to the face depth of the light's range, so every cluster that can
// cast a shadow into the page overlaps it.
fn local_dirty_info(entry: u32, full: bool) {
    let id = local_page_id(entry);
    let slot = id.slot;
    let face = id.face;
    let mip = id.mip;
    let page = id.page;
    let pages = local_mip_pages(mip);
    let view = slot * LOCAL_FACES + face;
    grow_rect(LOCAL_INFO_RECTS + (view * LOCAL_MIPS + mip) * 4u, vec2<i32>(page));

    let light = local_vsm.slots[slot].position_range;
    let far_depth = local_vsm.slots[slot].params.y / light.w; // reverse-Z depth at face depth = range
    let origin = light.xyz + local_vsm.slots[slot].face_offsets[face].xyz;
    var lo = origin;
    var hi = origin;
    for (var c = 0u; c < 4u; c++) {
        let uv = (vec2<f32>(page) + vec2<f32>(f32(c & 1u), f32(c >> 1u))) / f32(pages);
        let h = local_vsm.slots[slot].inv_view_proj[face] * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, far_depth, 1.0);
        let p = h.xyz / h.w;
        lo = min(lo, p);
        hi = max(hi, p);
    }
    let o = LOCAL_INFO_SLOTS + slot * LOCAL_INFO_SLOT_WORDS;
    let a = vec3<i32>(floor(lo * LOCAL_BOX_SCALE));
    let b = vec3<i32>(ceil(hi * LOCAL_BOX_SCALE));
    grow_box(o, a, b);
    atomicMin(&info[o + 6u], i32(mip));
    atomicAdd(&info[o + 7u], 1);
    if full {
        grow_box(o + 12u, a, b);
        atomicAdd(&info[o + LOCAL_INFO_FACE_FULL + face], 1);
    }
}

// Records that `entry` is rendered this frame, as `state` (PAGE_* | scratch << 8): culling and
// expansion restrict static casters to the pages rendered in full.
fn dirty_page(entry: u32, state: u32) {
    entries[entry].w = vsm.frame;
    atomicStore(&info[INFO_PAGE_STATE + entries[entry].x], bitcast<i32>(state));
    let full = page_kind(state) != PAGE_DYNAMIC;
    if entry >= local_vsm.entry_base {
        // Its expansion measures its swaying casters' motion anew.
        atomicStore(&info[INFO_PAGE_MOTION + 2u * entries[entry].x], 0);
        atomicStore(&info[INFO_PAGE_MOTION + 2u * entries[entry].x + 1u], 0);
        local_dirty_info(entry, full);
        return;
    }
    let level = entry / (vsm.level_pages * vsm.level_pages);
    let page = decode_tag(entries[entry].y);
    let bit = bitcast<i32>(1u << (entry % 32u));
    atomicOr(&info[INFO_DIRTY_MASK + entry / 32u], bit);
    grow_rect(level * 8u, page);
    if full {
        atomicOr(&info[INFO_FULL_MASK + entry / 32u], bit);
        grow_rect(level * 8u + 4u, page);
    }
}

// The page cache's dirty pages (new, or invalidated with their static casters) are rendered in
// full. A page's static layer belongs to its entry: rendering it in full drops it.
// Also counts, per local light face, the pages requested this frame (all resident now): what
// rendering the face from a new origin costs (read back, vsm_anchor.rs).
@compute @workgroup_size(64)
fn dirty_info(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < vsm.physical_count {
        let owned = owner[gid.x];
        if owned != INVALID_ID && owned >= local_vsm.entry_base && entries[owned].z == vsm.frame {
            let view = (owned - local_vsm.entry_base) / LOCAL_FACE_ENTRIES;
            let o = LOCAL_INFO_SLOTS + (view / LOCAL_FACES) * LOCAL_INFO_SLOT_WORDS;
            atomicAdd(&info[o + LOCAL_INFO_FACE_REQUESTED + view % LOCAL_FACES], 1);
        }
    }
    let dirty = atomicLoad(&cache_lists[CACHE_DIRTY_COUNT]);
    if gid.x == 0u {
        atomicStore(&info[INFO_ARGS], 6);
        atomicStore(&info[INFO_ARGS + 1u], i32(dirty));
        atomicStore(&info[INFO_ARGS + 2u], 0);
        atomicStore(&info[INFO_ARGS + 3u], 0);
    }
    if gid.x >= dirty {
        return;
    }
    let entry = atomicLoad(&cache_lists[DIRTY_START + gid.x]);
    let physical = entries[entry].x;
    var state = PAGE_FULL;
    if bitcast<u32>(atomicLoad(&info[INFO_PAGE_STATIC + physical])) != INVALID_ID {
        state |= PAGE_CLEAR;
        atomicStore(&info[INFO_PAGE_STATIC + physical], bitcast<i32>(INVALID_ID));
    }
    dirty_page(entry, state);
}

// Counts per slot the local pages requested this frame, resident or not: what a light's map
// asks of the page pool (dirty_info counts only what it got). A pool too small for every
// light's requests leaves new maps without pages, which then render nothing and would look
// cheap (vsm_local.rs). One thread per local page cache entry.
@compute @workgroup_size(64)
fn count_local_requests(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= LOCAL_SLOTS * LOCAL_FACES * LOCAL_FACE_ENTRIES {
        return;
    }
    if requests[local_vsm.entry_base + gid.x] == vsm.frame {
        let slot = gid.x / (LOCAL_FACES * LOCAL_FACE_ENTRIES);
        atomicAdd(&info[LOCAL_INFO_SLOTS + slot * LOCAL_INFO_SLOT_WORDS + LOCAL_INFO_REQUESTED_ALL], 1);
    }
}

// The refresh queue. After dirty_info, pages with a pending refresh (vsm_invalidate.wgsl) that
// are needed this frame are appended to the dirty list: a page with a static layer only
// re-renders its dynamic casters over it; one without is rendered in full once more, keeping
// its static casters as its static layer from then on; a page whose static casters streamed
// (REFRESH_FULL) is rendered in full. A pending refresh of a page that is not needed this frame
// waits until it is.
//
// Budget (UE's r.Shadow.Virtual.DeferredInvalidationBudget): with vsm.refresh.x > 0 at most that
// many pages are refreshed per frame. Pages are ranked into REFRESH_BUCKETS buckets by how far
// their casters may have moved (in their own texels: what the cached shadow is off by on screen,
// since texels match pixels) and how many frames they have waited; the cut is the first bucket
// that no longer fits (as cache_plan's). Pages past the budget keep their cached depth — stale
// by a few texels, never a hole — and wait with a growing priority; one that has waited
// vsm.refresh.y frames is refreshed regardless of the budget (bucket 0), so staleness stays
// bounded. Pages of distant levels (vsm.refresh.z and beyond) are only refreshed on their
// level's turn, every vsm.refresh.w frames (the far shadows' own rate).
//   refresh_count : per pending page its bucket (histogram)
//   refresh_plan  : the cut-off bucket and how many of its pages fit
//   refresh       : appends the pages within the budget

// Whether the page `p` with pending refresh `pending` is refreshed this frame: 0 no (waits), 1
// yes, 2 its refresh is dropped (the page was freed or rendered in full this frame).
fn refresh_needed(p: u32) -> u32 {
    let entry = owner[p];
    let valid = entry != INVALID_ID && entries[entry].x == p && entries[entry].w != vsm.frame;
    if !valid {
        return 2u;
    }
    return select(0u, 1u, entries[entry].z == vsm.frame);
}

// Frames since the page of `entry` was last rendered.
fn refresh_age(entry: u32) -> u32 {
    return vsm.frame - entries[entry].w;
}

// A distant level's page waiting for its level's turn (unless it is too stale or streamed).
fn refresh_held(entry: u32, pending: u32) -> bool {
    if entry >= local_vsm.entry_base || (pending & REFRESH_FULL) != 0u || refresh_age(entry) >= vsm.refresh.y {
        return false;
    }
    return !vsm_level_turn(entry / (vsm.level_pages * vsm.level_pages));
}

// The page's priority bucket: 0 for pages too stale to wait, then by the texels its casters may
// have moved plus a doubling for every frame it has waited (lower first).
fn refresh_bucket(entry: u32, pending: u32) -> u32 {
    let age = refresh_age(entry);
    if age >= vsm.refresh.y {
        return 0u;
    }
    let texels = f32(pending & ~REFRESH_FULL) / REFRESH_TEXEL_UNITS;
    let score = log2(max(texels, 0.0625)) + f32(age);
    let rank = i32(floor((score + 4.0) * 2.0));
    return u32(clamp(i32(REFRESH_BUCKETS) - 1 - rank, 1, i32(REFRESH_BUCKETS) - 1));
}

@compute @workgroup_size(64)
fn refresh_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= vsm.physical_count {
        return;
    }
    let pending = bitcast<u32>(atomicLoad(&info[INFO_PAGE_REFRESH + p]));
    if pending == 0u {
        return;
    }
    let needed = refresh_needed(p);
    if needed == 2u {
        atomicStore(&info[INFO_PAGE_REFRESH + p], 0);
        return;
    }
    if needed == 0u {
        return;
    }
    let entry = owner[p];
    if refresh_held(entry, pending) {
        atomicAdd(&info[INFO_REFRESH_HELD], 1);
        return;
    }
    atomicAdd(&info[INFO_REFRESH + refresh_bucket(entry, pending)], 1);
}

@compute @workgroup_size(1)
fn refresh_plan() {
    let budget = vsm.refresh.x;
    var cutoff = REFRESH_BUCKETS;
    var quota = 0u;
    // Bucket 0 goes in any case.
    let forced = u32(atomicLoad(&info[INFO_REFRESH]));
    var taken = forced;
    if budget > 0u {
        for (var b = 1u; b < REFRESH_BUCKETS; b++) {
            let pages = u32(atomicLoad(&info[INFO_REFRESH + b]));
            if taken + pages > budget {
                cutoff = b;
                quota = budget - min(taken, budget);
                break;
            }
            taken += pages;
        }
    }
    atomicStore(&info[INFO_REFRESH_CUTOFF], i32(cutoff));
    atomicStore(&info[INFO_REFRESH_QUOTA], i32(quota));
    atomicStore(&info[INFO_REFRESH_KEPT], 0);
    atomicStore(&info[INFO_REFRESH_FORCED], i32(select(0u, forced - min(forced, budget), budget > 0u)));
}

@compute @workgroup_size(64)
fn refresh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= vsm.physical_count {
        return;
    }
    let pending = bitcast<u32>(atomicLoad(&info[INFO_PAGE_REFRESH + p]));
    if pending == 0u || refresh_needed(p) != 1u {
        return;
    }
    let entry = owner[p];
    if refresh_held(entry, pending) {
        return;
    }
    let b = refresh_bucket(entry, pending);
    let cutoff = u32(atomicLoad(&info[INFO_REFRESH_CUTOFF]));
    if b > cutoff || (b == cutoff && u32(atomicAdd(&info[INFO_REFRESH_KEPT], 1)) >= u32(atomicLoad(&info[INFO_REFRESH_QUOTA]))) {
        // Past the budget: the cached depth stays one more frame.
        atomicAdd(&info[INFO_REFRESH_DEFERRED], 1);
        return;
    }
    atomicStore(&info[INFO_PAGE_REFRESH + p], 0);
    atomicStore(&cache_lists[DIRTY_START + atomicAdd(&cache_lists[CACHE_DIRTY_COUNT], 1u)], entry);
    atomicAdd(&info[INFO_ARGS + 1u], 1);
    let static_layer = bitcast<u32>(atomicLoad(&info[INFO_PAGE_STATIC + p]));
    if (pending & REFRESH_FULL) != 0u {
        // Its static casters changed: rendered in full, without a static layer (as dirty_info).
        var state = PAGE_FULL;
        if static_layer != INVALID_ID {
            state |= PAGE_CLEAR;
            atomicStore(&info[INFO_PAGE_STATIC + p], bitcast<i32>(INVALID_ID));
        }
        dirty_page(entry, state);
        return;
    }
    var state = PAGE_SPLIT;
    if static_layer == entry {
        atomicAdd(&info[INFO_COUNTERS], 1);
        let scratch = u32(atomicAdd(&info[INFO_COUNTERS + 2u], 1));
        state = PAGE_DYNAMIC | (select(NO_SCRATCH, scratch, scratch < SCRATCH_TILES) << 8u);
    } else {
        atomicAdd(&info[INFO_COUNTERS + 1u], 1);
        atomicStore(&info[INFO_PAGE_STATIC + p], bitcast<i32>(entry));
        // A static layer made stale by a caster changing layer is still in the mirror.
        state |= select(0u, PAGE_CLEAR, static_layer != INVALID_ID);
    }
    dirty_page(entry, state);
}
