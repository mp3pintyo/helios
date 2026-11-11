// Generic GPU page cache, shared by the runtime virtual texture and virtual shadow maps.
//
// A client marks the virtual pages it needs (requests[entry] = frame). Then:
//   cache_begin     : reset per-frame counters and the priority histogram
//   cache_update    : per entry: keep / free / queue for allocation; stale tags are freed
//   cache_plan      : if the pool cannot hold every request, where the priorities cut off
//   cache_reassign  : only then (dispatched indirectly): evicts retained pages and what falls
//                     below the cut, and queues the rest by priority
//   cache_free_list : per physical page: collect unowned pages
//   cache_allocate  : per queued entry: take a free page, mark it dirty (must be rendered)
//   cache_args      : indirect dispatch for the client's page-render pass
// If the pool ran out last frame, unrequested pages are evicted immediately (retain = 0),
// otherwise they are kept `retain` frames for reuse — that is the cache.
//
// Priorities. Every requested entry falls into one of BUCKETS priority buckets (bucket 0 first;
// the virtual texture has one). While the pool holds every request nothing changes: requests
// are queued as they come and pages stay where they are. When it cannot, the cut is the first
// bucket at which the requests of all buckets up to it exceed the pool: requests of better
// buckets are queued first, in bucket order, and resident pages of worse buckets are evicted to
// make room for them (a page of the cut-off bucket stays resident while the pool has room for
// it, ahead of that bucket's new requests: no page flips between two requests of equal rank).
// So what is left without a page is always the least important, not whatever was queued last.
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
// Requested entries that got no page this frame (read back: statistics).
const H_OVERFLOW: u32 = 4u;
// Entries requested this frame, resident or not (read back), and resident entries kept for reuse
// although not requested.
const H_REQUESTED: u32 = 5u;
const H_RETAINED: u32 = 6u;
const H_ARGS: u32 = 8u;    // dirty dispatch x, y, z
// The cut-off bucket (BUCKETS: none), how many of its resident pages stay, the counter of
// those that did, and the requested entries resident before the allocation.
const H_CUTOFF: u32 = 11u;
const H_QUOTA: u32 = 12u;
const H_KEPT: u32 = 13u;
const H_RESIDENT: u32 = 14u;

// Priority buckets (clipmap client; see `bucket`).
const BUCKETS: u32 = 256u;
// Each mapped local light's coarsest page per face: what a receiver falls back to while finer
// pages are missing (vsm_local_sample.wgsl). Every light that holds a shadow-map slot casts some
// shadow before any other page is placed (at most 6 pages per slot).
const BUCKET_FALLBACK: u32 = 0u;
// The sun's clipmap pages (it has no coarser fallback of its own).
const BUCKET_SUN: u32 = 1u;
// Then local light pages by the light's rank (params.ranks: most influential on screen first),
// within a light coarse mips before fine ones.
const BUCKET_LOCAL: u32 = 2u;
// Local light page entries (vsm_local_common.wgsl): per slot 6 faces of 1365 entries, mips 0..5
// starting at these offsets inside a face.
const LOCAL_FACE_ENTRIES: u32 = 1365u;
const LOCAL_SLOT_ENTRIES: u32 = 6u * 1365u;
const LOCAL_FINE_MIPS: u32 = 5u;

struct CacheParams {
    entry_count: u32,
    physical_count: u32,
    frame: u32,
    retain: u32,
    // Non-zero: re-render every requested page (contents invalidated, e.g. the light moved);
    // the clipmap client: a bit per level.
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
    // Clipmap client: the first local light page entry (INVALID: none).
    local_base: u32,
    _pad1: u32,
    _pad2: u32,
    // Clipmap client: per local light slot its rank (0 = most influential), four per vec4.
    ranks: array<vec4<u32>, 8>,
}

@group(0) @binding(0) var<uniform> params: CacheParams;
@group(0) @binding(1) var<storage, read_write> entries: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> requests: array<u32>;
@group(0) @binding(3) var<storage, read_write> owner: array<u32>;
// [0, 16): header, then free list [P], allocation queue [P], dirty list [P], then per bucket
// its resident and new requests, its first queue position and its queue cursor.
@group(0) @binding(4) var<storage, read_write> lists: array<atomic<u32>>;
// Dispatch x, y, z of cache_reassign: none unless the pool is full.
@group(0) @binding(5) var<storage, read_write> reassign_args: array<u32, 3>;

fn free_list(i: u32) -> u32 { return HEADER + i; }
fn alloc_list(i: u32) -> u32 { return HEADER + params.physical_count + i; }
fn dirty_list(i: u32) -> u32 { return HEADER + 2u * params.physical_count + i; }
fn bucket_resident(b: u32) -> u32 { return HEADER + 3u * params.physical_count + b; }
fn bucket_new(b: u32) -> u32 { return bucket_resident(b) + BUCKETS; }
fn bucket_start(b: u32) -> u32 { return bucket_resident(b) + 2u * BUCKETS; }
fn bucket_cursor(b: u32) -> u32 { return bucket_resident(b) + 3u * BUCKETS; }

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

// The priority bucket of an entry (lower first).
fn bucket(entry: u32) -> u32 {
    if CLIENT != CLIENT_CLIPMAP {
        return 0u;
    }
    if entry < params.local_base {
        return BUCKET_SUN;
    }
    let e = entry - params.local_base;
    let slot = e / LOCAL_SLOT_ENTRIES;
    let in_face = e % LOCAL_FACE_ENTRIES;
    // Mip offsets inside a face: 0, 1024, 1280, 1344, 1360, 1364 (the coarsest).
    if in_face >= 1364u {
        return BUCKET_FALLBACK;
    }
    let mip = select(0u, 1u, in_face >= 1024u) + select(0u, 1u, in_face >= 1280u)
        + select(0u, 1u, in_face >= 1344u) + select(0u, 1u, in_face >= 1360u);
    let rank = min(params.ranks[slot / 4u][slot % 4u], 31u);
    return BUCKET_LOCAL + rank * LOCAL_FINE_MIPS + (LOCAL_FINE_MIPS - 1u - mip);
}

// Clipmap client: `invalidate` is a bit mask of levels (distant levels follow a moving sun at
// their own rate, vsm.rs); the static client invalidates everything if it is set.
fn invalidated(entry: u32) -> bool {
    if CLIENT != CLIENT_CLIPMAP {
        return params.invalidate != 0u;
    }
    let level = entry / (params.level_pages * params.level_pages);
    return level < params.level_count && ((params.invalidate >> level) & 1u) != 0u;
}

@compute @workgroup_size(256)
fn cache_begin(@builtin(local_invocation_index) i: u32) {
    atomicStore(&lists[bucket_resident(i)], 0u);
    atomicStore(&lists[bucket_new(i)], 0u);
    atomicStore(&lists[bucket_cursor(i)], 0u);
    if i != 0u {
        return;
    }
    atomicStore(&lists[H_OVERFLOW_PREV], atomicLoad(&lists[H_OVERFLOW]));
    atomicStore(&lists[H_OVERFLOW], 0u);
    atomicStore(&lists[H_ALLOC], 0u);
    atomicStore(&lists[H_FREE], 0u);
    atomicStore(&lists[H_DIRTY], 0u);
    atomicStore(&lists[H_REQUESTED], 0u);
    atomicStore(&lists[H_RETAINED], 0u);
    atomicStore(&lists[H_CUTOFF], BUCKETS);
    atomicStore(&lists[H_KEPT], 0u);
    atomicStore(&lists[H_RESIDENT], 0u);
    reassign_args = array<u32, 3>(0u, 1u, 1u);
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
    let invalidate = invalidated(i);
    let requested = requests[i] == params.frame || i == params.pinned;
    if requested {
        atomicAdd(&lists[H_REQUESTED], 1u);
        if e.x != INVALID {
            e.z = params.frame;
            atomicAdd(&lists[bucket_resident(bucket(i))], 1u);
            if invalidate {
                let slot = atomicAdd(&lists[H_DIRTY], 1u);
                atomicStore(&lists[dirty_list(slot)], i);
            }
        } else {
            atomicAdd(&lists[bucket_new(bucket(i))], 1u);
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
        } else {
            atomicAdd(&lists[H_RETAINED], 1u);
        }
    }
    entries[i] = e;
}

// While requests and retained pages fit the pool, nothing to do (the queue is complete).
// Otherwise retained pages go, and if the requests alone do not fit either, the buckets are
// walked from the first: the cut is where they stop fitting.
@compute @workgroup_size(1)
fn cache_plan() {
    let pool = params.physical_count;
    let requested = atomicLoad(&lists[H_REQUESTED]);
    if requested + atomicLoad(&lists[H_RETAINED]) <= pool {
        return;
    }
    let groups = (params.entry_count + 63u) / 64u;
    reassign_args = array<u32, 3>(min(groups, 65535u), (groups + 65534u) / 65535u, 1u);
    if requested <= pool {
        return;
    }
    var placed = 0u;
    var queued = 0u;
    for (var b = 0u; b < BUCKETS; b++) {
        let resident = atomicLoad(&lists[bucket_resident(b)]);
        let fresh = atomicLoad(&lists[bucket_new(b)]);
        atomicStore(&lists[bucket_start(b)], queued);
        queued += fresh;
        if placed + resident + fresh > pool {
            atomicStore(&lists[H_CUTOFF], b);
            atomicStore(&lists[H_QUOTA], pool - placed);
            break;
        }
        placed += resident + fresh;
    }
    // The cut-off bucket's new requests are queued last: those beyond the free pages its
    // resident ones leave get none.
    atomicStore(&lists[H_ALLOC], queued);
}

@compute @workgroup_size(64)
fn cache_reassign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if i >= params.entry_count {
        return;
    }
    var e = entries[i];
    let requested = requests[i] == params.frame || i == params.pinned;
    if !requested {
        if e.x != INVALID {
            owner[e.x] = INVALID;
            e.x = INVALID;
            entries[i] = e;
        }
        return;
    }
    let cutoff = atomicLoad(&lists[H_CUTOFF]);
    if cutoff == BUCKETS {
        return;
    }
    let b = bucket(i);
    if e.x != INVALID {
        // Pages queued for re-rendering by the invalidation stay (rare: the sun or the scene
        // changed in a full pool; the queue's last requests then get no page).
        if b < cutoff || invalidated(i) || (b == cutoff && atomicAdd(&lists[H_KEPT], 1u) < atomicLoad(&lists[H_QUOTA])) {
            return;
        }
        owner[e.x] = INVALID;
        e.x = INVALID;
        entries[i] = e;
    } else if b <= cutoff {
        let slot = atomicLoad(&lists[bucket_start(b)]) + atomicAdd(&lists[bucket_cursor(b)], 1u);
        if slot < params.physical_count {
            atomicStore(&lists[alloc_list(slot)], i);
        }
    }
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
    } else if entries[owner[p]].z == params.frame {
        atomicAdd(&lists[H_RESIDENT], 1u);
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
    // Requests without a page: all requests but the resident ones and those allocated now.
    let allocated = min(min(atomicLoad(&lists[H_ALLOC]), params.physical_count), atomicLoad(&lists[H_FREE]));
    let resident = atomicLoad(&lists[H_RESIDENT]) + allocated;
    let requested = atomicLoad(&lists[H_REQUESTED]);
    atomicStore(&lists[H_OVERFLOW], requested - min(resident, requested));
}
