// Generic GPU page cache, shared by the runtime virtual texture and virtual shadow maps.
//
// A client marks the virtual pages it needs (requests[entry] = frame). Then:
//   cache_begin     : reset per-frame counters
//   cache_update    : per entry: keep / free / queue for allocation; stale tags are freed
//   cache_free_list : per physical page: collect unowned pages
//   cache_allocate  : per queued entry: take a free page, mark it dirty (must be rendered)
//   cache_args      : indirect dispatch for the client's page-render pass
// If the pool ran out last frame, unrequested pages are evicted immediately (retain = 0),
// otherwise they are kept `retain` frames for reuse — that is the cache.
//
// Entry: x = physical page (or INVALID), y = tag, z = last frame used, w = flags.
// CLIENT = 1 (clipmap): entries are toroidally addressed per level, and the tag is the
// absolute page coordinate, so pages that scroll out of a level are dropped. Entries after the
// levels are static pages of the client (local light shadow maps), which it invalidates
// itself: `invalidate` only applies to the levels.

override CLIENT: u32 = 0u;
const CLIENT_STATIC: u32 = 0u;
const CLIENT_CLIPMAP: u32 = 1u;
const INVALID: u32 = 0xffffffffu;

const HEADER: u32 = 16u;
const H_ALLOC: u32 = 0u;
const H_FREE: u32 = 1u;
const H_DIRTY: u32 = 2u;
const H_OVERFLOW_PREV: u32 = 3u;
const H_OVERFLOW: u32 = 4u;
const H_ARGS: u32 = 8u;    // dirty dispatch x, y, z

struct CacheParams {
    entry_count: u32,
    physical_count: u32,
    frame: u32,
    retain: u32,
    // 1: re-render every requested page (contents invalidated, e.g. the light moved).
    invalidate: u32,
    // Workgroups the client dispatches per dirty page.
    groups_per_page: u32,
    // Clipmap client: pages per level side and level count.
    level_pages: u32,
    level_count: u32,
    // Clipmap client: absolute page coordinate of each level's first page (xy), w unused.
    origins: array<vec4<i32>, 16>,
    // An always-needed entry (e.g. the coarsest mip), or INVALID.
    pinned: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: CacheParams;
@group(0) @binding(1) var<storage, read_write> entries: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> requests: array<u32>;
@group(0) @binding(3) var<storage, read_write> owner: array<u32>;
// [0, 16): header, then free list [P], allocation queue [P], dirty list [P].
@group(0) @binding(4) var<storage, read_write> lists: array<atomic<u32>>;

fn free_list(i: u32) -> u32 { return HEADER + i; }
fn alloc_list(i: u32) -> u32 { return HEADER + params.physical_count + i; }
fn dirty_list(i: u32) -> u32 { return HEADER + 2u * params.physical_count + i; }

fn expected_tag(entry: u32) -> u32 {
    if CLIENT != CLIENT_CLIPMAP {
        return 0u;
    }
    let n = params.level_pages;
    let level = entry / (n * n);
    if level >= params.level_count {
        return 0u; // a static page after the levels
    }
    let local = entry % (n * n);
    let wrapped = vec2<i32>(i32(local % n), i32(local / n));
    let origin = params.origins[level].xy;
    let ni = i32(n);
    // The absolute page this slot represents: origin + ((wrapped - origin) mod n).
    let absolute = origin + (((wrapped - origin) % ni) + ni) % ni;
    return (u32(absolute.x) & 0xffffu) | (u32(absolute.y) << 16u);
}

@compute @workgroup_size(1)
fn cache_begin() {
    atomicStore(&lists[H_OVERFLOW_PREV], atomicLoad(&lists[H_OVERFLOW]));
    atomicStore(&lists[H_OVERFLOW], 0u);
    atomicStore(&lists[H_ALLOC], 0u);
    atomicStore(&lists[H_FREE], 0u);
    atomicStore(&lists[H_DIRTY], 0u);
}

@compute @workgroup_size(64)
fn cache_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if i >= params.entry_count {
        return;
    }
    var e = entries[i];
    let tag = expected_tag(i);
    if e.x != INVALID && e.y != tag {
        owner[e.x] = INVALID;
        e.x = INVALID;
    }
    e.y = tag;
    let levels = params.level_count * params.level_pages * params.level_pages;
    let invalidate = params.invalidate != 0u && (CLIENT != CLIENT_CLIPMAP || i < levels);
    let requested = requests[i] == params.frame || i == params.pinned;
    if requested {
        if e.x != INVALID {
            e.z = params.frame;
            if invalidate {
                let slot = atomicAdd(&lists[H_DIRTY], 1u);
                atomicStore(&lists[dirty_list(slot)], i);
            }
        } else {
            let slot = atomicAdd(&lists[H_ALLOC], 1u);
            if slot < params.physical_count {
                atomicStore(&lists[alloc_list(slot)], i);
            }
        }
    } else if e.x != INVALID {
        let retain = select(params.retain, 0u, atomicLoad(&lists[H_OVERFLOW_PREV]) > 0u);
        if invalidate || params.frame - e.z > retain {
            owner[e.x] = INVALID;
            e.x = INVALID;
        }
    }
    entries[i] = e;
}

@compute @workgroup_size(64)
fn cache_free_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= params.physical_count {
        return;
    }
    if owner[p] == INVALID {
        let slot = atomicAdd(&lists[H_FREE], 1u);
        atomicStore(&lists[free_list(slot)], p);
    }
}

@compute @workgroup_size(64)
fn cache_allocate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let queued = min(atomicLoad(&lists[H_ALLOC]), params.physical_count);
    if i >= queued {
        return;
    }
    if i >= atomicLoad(&lists[H_FREE]) {
        atomicAdd(&lists[H_OVERFLOW], 1u);
        return;
    }
    let entry = atomicLoad(&lists[alloc_list(i)]);
    let page = atomicLoad(&lists[free_list(i)]);
    owner[page] = entry;
    entries[entry] = vec4<u32>(page, entries[entry].y, params.frame, 0u);
    let slot = atomicAdd(&lists[H_DIRTY], 1u);
    atomicStore(&lists[dirty_list(slot)], entry);
}

@compute @workgroup_size(1)
fn cache_args() {
    let dirty = min(atomicLoad(&lists[H_DIRTY]), params.physical_count);
    atomicStore(&lists[H_DIRTY], dirty);
    let groups = dirty * params.groups_per_page;
    atomicStore(&lists[H_ARGS], min(groups, 65535u));
    atomicStore(&lists[H_ARGS + 1u], (groups + 65534u) / 65535u);
    atomicStore(&lists[H_ARGS + 2u], 1u);
}
