// Local invalidation of cached shadow pages: geometry streaming, wind, skeletal animation and
// moving instances.
//
// Cached pages assume static geometry. Four things change it, but only inside the light-space
// footprint of the instances concerned, so only the cached pages there are re-rendered instead of
// the whole cache:
//  * streaming a page in or out changes the cluster cut of every instance of that mesh;
//  * the wind (wind.wgsl) moves vegetation every frame, up to its disable distance;
//  * animated skeletal meshes (skinning.wgsl) move whenever their pose changes;
//  * Scene::set_transform moves an instance: vsm_motion.rs writes footprints covering its old and
//    new bounds before the collect pass appends the others (the buffer then starts non-empty).
// Pages keep their static casters in a static layer (vsm_common.wgsl): a footprint of a dynamic
// caster (one the wind or its skeleton moves this frame, or a Mobility::Movable one) only
// requests a refresh of its dynamic casters, while a footprint of a static caster that changed (or moved) frees the
// page, so it is rendered again in full.
// A caster that starts or stops moving changes layer. Where it fades into or out of the wind
// (the camera approaching or leaving it) it is at rest, so the cached depth stays right and only
// the static layer of the pages under it is dropped: the next refresh renders them in full. When
// the wind is switched on or off (ABRUPT), it jumps: its pages are freed.
// Passes:
//   collect           : appends the light-space footprint of every shadow caster the wind moves
//                       or that changed layer and, in the RESIDENCY specialisation, of every
//                       instance whose mesh pages changed residency since the last check (page
//                       table vs. its copy)
//   remember_residency: copies the page table for the next check
//   invalidate_pages  : one workgroup per physical page; the page is freed if a static footprint
//                       overlaps it, loses its static layer if a layer change does, and is flagged
//                       for a dynamic refresh (vsm.wgsl, refresh) if a dynamic one overlaps it and
//                       may have changed it by a texel since it was rendered. If it is still
//                       needed, it is rendered again this frame.
// A wind footprint bounds how far its geometry moved since the page was rendered by
// min(2 · maximum displacement, speed bound · frames): pages of fine levels are refreshed every
// frame, coarse levels (distant shadows, volumetric fog) every few frames, and levels whose texels
// dwarf the displacement never — the cost stays bounded without visible lag.
// Local light pages (vsm_local.rs) apply the same rules in their light's terms: a footprint hits
// a page if its world sphere lies within the light's range and projects onto the page in its
// face and mip, once its geometry may have moved WIND_REFRESH_TEXELS of the page's texels at the
// sphere's nearest face depth (texels grow with the distance from the light). Their own
// invalidation (a light that moved, vsm.wgsl's invalidate_local) frees whole lights instead.
// If the list overflows, every cached page is freed (the old global behaviour).

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "wind.wgsl"
#include "skin_palette.wgsl"
#include "vsm_local_common.wgsl"

// Footprint capacity; set by vsm.rs.
override CAPACITY: u32 = 16383u;
// Also compare the page table with its copy (streaming changed residency).
override RESIDENCY: bool = false;
// The wind was switched on or off since the last collect: casters changing layer jump.
override ABRUPT: bool = false;

// A page is refreshed once its geometry may have moved this many of its texels.
const WIND_REFRESH_TEXELS: f32 = 2.0;
// Change of a footprint that invalidates every level at once (streaming, layer changes).
const ALWAYS: f32 = 3.0e38;
// Footprint.world.w: only the footprint's dynamic casters moved, a caster changed layer, or the
// pages' static casters changed.
const FOOTPRINT_DYNAMIC: f32 = 0.0;
const FOOTPRINT_LAYER: f32 = 1.0;
const FOOTPRINT_STATIC: f32 = 2.0;

struct Footprint {
    // World-space centre (local light pages), w: FOOTPRINT_* kind.
    world: vec4<f32>,
    // Sun light-space centre (x, -y) and radius in world units: the footprint is the circle
    // (and for local lights the world sphere) of that radius.
    center: vec2<f32>,
    radius: f32,
    // Bound of how far the geometry can move overall, and per frame.
    change: f32,
    per_frame: f32,
    // Moves (vsm_motion.rs): the half extent of a tighter light-space rectangle around `center`,
    // which replaces the circle in the sun's clipmap (0: the circle).
    rect: vec2<f32>,
}

struct Footprints {
    count: atomic<u32>,
    items: array<Footprint>,
}

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> page_table: array<u32>;
// The page table as of the previous residency check (remember_residency).
@group(0) @binding(3) var<storage, read_write> previous_page_table: array<u32>;
@group(0) @binding(4) var<storage, read_write> footprints: Footprints;
@group(0) @binding(5) var<storage, read_write> owner: array<u32>;
@group(0) @binding(6) var<storage, read_write> entries: array<vec4<u32>>;
// Per instance: the wind reach it was padded by when last collected, 0 if it was a static caster
// (its shadow is in the pages' static layers).
@group(0) @binding(7) var<storage, read_write> dynamic_reach: array<f32>;
// The VSM info buffer (vsm.wgsl): pending refresh flags per physical page.
@group(0) @binding(8) var<storage, read_write> info: array<u32>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;

fn residency_changed(instance: Instance) -> bool {
    for (var p = instance.page_offset; p < instance.page_offset + instance.page_count; p++) {
        if (page_table[p] == INVALID_ID) != (previous_page_table[p] == INVALID_ID) {
            return true;
        }
    }
    return false;
}
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;

@compute @workgroup_size(64)
fn collect(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&instances) || gid.x >= arrayLength(&dynamic_reach) {
        return;
    }
    let instance = instances[gid.x];
    if (instance.flags & INSTANCE_CAST_SHADOW) == 0u {
        return;
    }
    var f = Footprint(vec4<f32>(instance.bounds.xyz, FOOTPRINT_DYNAMIC), vec2<f32>(0.0), instance.bounds.w, 0.0, 0.0, vec2<f32>(0.0));
    let strength = wind_strength(instance);
    var reach = 0.0;
    if dynamic_caster(instance, strength) {
        // Movable instances out of the wind have no wind reach: their moves come from the CPU
        // (a moving skin adds its own below).
        let wind_bound = wind_reach(instance, strength, instance.bounds);
        reach = max(wind_bound, 1e-6);
        f.change = 2.0 * wind_bound;
        f.per_frame = wind_speed(instance, strength, instance.bounds) * max(wind.time.x - wind.time.y, 0.0);
    }
    if skin_dynamic(instance) {
        // A moving skin: every pose since the scene was loaded lies within the largest reach of
        // the rest sphere, so the footprint covers whatever a cached page may still show.
        let motion = skin_motion(instance);
        reach = max(reach, max(motion.x, 1e-6));
        f.change = max(f.change, 2.0 * motion.x);
        f.per_frame = max(f.per_frame, motion.y);
    }
    if RESIDENCY && residency_changed(instance) {
        f.change = ALWAYS;
        f.per_frame = ALWAYS;
        f.world.w = select(FOOTPRINT_STATIC, FOOTPRINT_DYNAMIC, reach > 0.0);
    }
    // Starting to move takes the caster out of the static layers, stopping puts it back: the
    // pages under everything it covered in either pose change layers.
    let previous = dynamic_reach[gid.x];
    if (previous > 0.0) != (reach > 0.0) {
        f.change = ALWAYS;
        f.per_frame = ALWAYS;
        f.world.w = max(f.world.w, select(FOOTPRINT_LAYER, FOOTPRINT_STATIC, ABRUPT));
    }
    dynamic_reach[gid.x] = reach;
    f.radius += max(reach, previous);
    if f.change <= 0.0 {
        return;
    }
    let slot = atomicAdd(&footprints.count, 1u);
    if slot < CAPACITY {
        let ls = (vsm.light_view * vec4<f32>(instance.bounds.xyz, 1.0)).xyz;
        f.center = vec2<f32>(ls.x, -ls.y);
        footprints.items[slot] = f;
    }
}

@compute @workgroup_size(64)
fn remember_residency(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&page_table) {
        previous_page_table[gid.x] = page_table[gid.x];
    }
}

// Bits of the footprints that hit the workgroup's page (HIT_STATIC ends the search).
const HIT_DYNAMIC: u32 = 1u;
const HIT_LAYER: u32 = 2u;
const HIT_STATIC: u32 = 4u;
var<workgroup> page_hit: atomic<u32>;

fn hit_bits(f: Footprint) -> u32 {
    return 1u << u32(f.world.w);
}

// Whether footprint `f` may have changed the local light page `id` by WIND_REFRESH_TEXELS of its
// texels since it was rendered `age` frames ago.
fn local_page_hit(id: LocalPageId, f: Footprint, age: f32) -> bool {
    let light = local_vsm.slots[id.slot].position_range;
    if distance(f.world.xyz, light.xyz) > f.radius + light.w {
        return false;
    }
    let r = local_sphere_rect(id.slot, id.face, vec4<f32>(f.world.xyz, f.radius));
    let depth = max(r.front, local_vsm.slots[id.slot].params.y);
    if !r.valid || min(f.change, f.per_frame * max(age, 1.0)) < local_texel(id.slot, depth, id.mip) * WIND_REFRESH_TEXELS {
        return false;
    }
    let pages = f32(local_mip_pages(id.mip));
    let page = vec2<f32>(id.page);
    return all(page >= floor(r.lo * pages)) && all(page <= floor(r.hi * pages));
}

// Dispatched with one workgroup per physical page.
@compute @workgroup_size(64)
fn invalidate_pages(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        atomicStore(&page_hit, 0u);
    }
    workgroupBarrier();
    let count = atomicLoad(&footprints.count);
    let entry = owner[wid.x];
    // Overflow: free every page.
    var hits = select(0u, HIT_STATIC, count > CAPACITY);
    if entry != INVALID_ID && entry >= local_vsm.entry_base && count > 0u {
        let id = local_page_id(entry);
        let age = f32(vsm.frame - entries[entry].w);
        for (var i = lid; i < min(count, CAPACITY) && hits < HIT_STATIC; i += 64u) {
            let f = footprints.items[i];
            if local_page_hit(id, f, age) {
                hits |= hit_bits(f);
            }
        }
    } else if entry != INVALID_ID && count > 0u {
        let level = entry / (vsm.level_pages * vsm.level_pages);
        let page_world = vsm.level_info[level].x;
        let threshold = vsm.level_info[level].y * WIND_REFRESH_TEXELS;
        // entries.w: the frame the page was last rendered in (vsm.wgsl, dirty_info).
        let age = f32(vsm.frame - entries[entry].w);
        let page = vec2<f32>(decode_tag(entries[entry].y));
        for (var i = lid; i < min(count, CAPACITY) && hits < HIT_STATIC; i += 64u) {
            let f = footprints.items[i];
            if min(f.change, f.per_frame * max(age, 1.0)) >= threshold {
                // Footprint in page units (a circle, or a rectangle) vs. the page square
                // [page, page + 1).
                let v = f.center / page_world;
                let rect = any(f.rect > vec2<f32>(0.0));
                let gap = max(abs(v - (page + 0.5)) - ((f.rect / page_world) + 0.5), vec2<f32>(0.0));
                if length(gap) <= select(f.radius / page_world, 0.0, rect) {
                    hits |= hit_bits(f);
                }
            }
        }
    }
    if hits != 0u {
        atomicOr(&page_hit, hits);
    }
    workgroupBarrier();
    let hit = atomicLoad(&page_hit);
    if lid != 0u || entry == INVALID_ID || hit == 0u {
        return;
    }
    if (hit & HIT_STATIC) != 0u {
        owner[wid.x] = INVALID_ID;
        entries[entry].x = INVALID_ID;
        return;
    }
    if (hit & HIT_LAYER) != 0u && info[INFO_PAGE_STATIC + wid.x] != INVALID_ID {
        info[INFO_PAGE_STATIC + wid.x] = STALE_STATIC_LAYER;
    }
    if (hit & HIT_DYNAMIC) != 0u {
        info[INFO_PAGE_REFRESH + wid.x] = 1u;
    }
}
