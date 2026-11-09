// Nanite tessellation's patch queues (tessellation.rs): a global, multipass PatchSplit as in UE.
// Requires tess_core.wgsl, wind.wgsl, an `instances` binding and the includer's
//   tess_visible(item, patch, instance, strength) -> bool   (frustum / shadow page culling)
//
// The patches buffer holds two split queues (ping-pong between passes) and a dice list per grid
// level (F = 1, 2, 4, 8 segments). A view's emit pass (tess.wgsl, vsm_tess.wgsl) routes every
// triangle of its tessellated clusters — the level-0 patches — then each split pass reads one
// queue and routes the children it splits off into the other. WebGPU has no forward-progress
// guarantee, so there are no persistent threads: a fixed number of passes (tessellation.rs raises
// it while patches still wanted splitting in the last one), each an indirect dispatch sized by a
// one-thread pass, and patches still too coarse after the last are diced as they are.
//
// state (u32 words): [0] pass, [1], [2] queue counts, [3] patches lost to full lists, [4] deepest
// pass that split, [5, 9) diced patches per level, then frame totals (cleared once per frame): [12]
// patches routed, [13] micro-triangles diced, [14] patches lost, [15] deepest splitting pass.
// dispatch (indirect arguments; only the one-thread passes bind it, as a buffer cannot be storage
// and indirect in one dispatch): [0, 3) the split pass, [4 + 4 level, 7 + 4 level) the dicing of
// grid level `level`.

@group(0) @binding(8) var<storage, read_write> patches: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> dispatch: array<u32>;

override QUEUE_CAPACITY: u32 = 262144u;
override DICE_CAPACITY: u32 = 262144u;

const STATE_PASS: u32 = 0u;
const STATE_QUEUE: u32 = 1u;
const STATE_OVERFLOW: u32 = 3u;
const STATE_DEEPEST: u32 = 4u;
const STATE_DICE: u32 = 5u;
const STATE_TOTAL_PATCHES: u32 = 12u;
const STATE_TOTAL_MICRO: u32 = 13u;
const STATE_TOTAL_OVERFLOW: u32 = 14u;
const STATE_TOTAL_DEEPEST: u32 = 15u;
const DISPATCH_DICE: u32 = 4u;
const MAX_DISPATCH: u32 = 65535u;

fn tess_queue_offset(queue: u32) -> u32 {
    return queue * QUEUE_CAPACITY;
}

fn tess_dice_offset(level: u32) -> u32 {
    return (2u * QUEUE_CAPACITY) + (level * DICE_CAPACITY);
}

// Culls a patch, then either diced (into the list of its grid level) or split.
fn tess_classify(item: u32, packed: vec3<u32>, force_dice: bool) -> TessRoute {
    var r: TessRoute;
    r.list = TESS_LISTS;
    r.count = 0u;
    let b = tess_base(item);
    let instance = instances[b.instance];
    let p = tess_patch(item, b, instance, packed);
    if !tess_visible(item, p, instance, wind_strength(instance)) {
        return r;
    }
    let mask = select(tess_split_mask(p), 0u, force_dice);
    if mask != 0u {
        return tess_children(item, p.corners, mask);
    }
    r.list = tess_dice(p).level;
    r.count = 1u;
    r.records[0] = vec4<u32>(item, packed);
    return r;
}

var<workgroup> route_counts: array<atomic<u32>, 5>;
var<workgroup> route_bases: array<u32, 5>;

// Appends every thread's route (call from uniform control flow: it synchronises the workgroup).
// One global atomic per list and workgroup.
fn tess_append(route: TessRoute, lid: u32, out_queue: u32) {
    if lid < TESS_LISTS {
        atomicStore(&route_counts[lid], 0u);
    }
    workgroupBarrier();
    var slot = 0u;
    if route.count > 0u {
        slot = atomicAdd(&route_counts[route.list], route.count);
    }
    workgroupBarrier();
    if lid < TESS_LISTS {
        let n = atomicLoad(&route_counts[lid]);
        if n > 0u {
            let counter = select(STATE_DICE + lid, STATE_QUEUE + out_queue, lid == TESS_LIST_QUEUE);
            route_bases[lid] = atomicAdd(&state[counter], n);
            atomicAdd(&state[STATE_TOTAL_PATCHES], n);
            if lid == TESS_LIST_QUEUE {
                atomicMax(&state[STATE_DEEPEST], atomicLoad(&state[STATE_PASS]) + 1u);
            }
        }
    }
    workgroupBarrier();
    if route.count == 0u {
        return;
    }
    let first = route_bases[route.list] + slot;
    let queue = route.list == TESS_LIST_QUEUE;
    let capacity = select(DICE_CAPACITY, QUEUE_CAPACITY, queue);
    let offset = select(tess_dice_offset(route.list), tess_queue_offset(out_queue), queue);
    for (var i = 0u; i < route.count; i++) {
        if first + i >= capacity {
            atomicAdd(&state[STATE_OVERFLOW], route.count - i);
            return;
        }
        patches[offset + first + i] = route.records[i];
    }
}

// The views' entry points (tess.wgsl, vsm_tess.wgsl) call these.

// Before split pass `state[0] + 1` (one thread): sizes its dispatch by the queue the previous pass filled and
// empties the queue it fills.
fn tess_prepare_split() {
    let pass_index = atomicLoad(&state[STATE_PASS]) + 1u;
    atomicStore(&state[STATE_PASS], pass_index);
    atomicStore(&state[STATE_QUEUE + (pass_index & 1u)], 0u);
    let count = min(atomicLoad(&state[STATE_QUEUE + ((pass_index - 1u) & 1u)]), QUEUE_CAPACITY);
    let groups = (count + 63u) / 64u;
    dispatch[0] = min(groups, MAX_DISPATCH);
    dispatch[1] = (groups + MAX_DISPATCH - 1u) / MAX_DISPATCH;
    dispatch[2] = 1u;
}

// One thread per patch of the queue the previous pass filled (workgroups of 64). The last pass dices everything.
fn tess_split(wid: vec3<u32>, groups: vec3<u32>, lid: u32) {
    let pass_index = atomicLoad(&state[STATE_PASS]);
    let in_queue = (pass_index - 1u) & 1u;
    let count = min(atomicLoad(&state[STATE_QUEUE + in_queue]), QUEUE_CAPACITY);
    let index = ((wid.x + (wid.y * groups.x)) * 64u) + lid;
    var route: TessRoute;
    route.count = 0u;
    if index < count {
        let queued = patches[tess_queue_offset(in_queue) + index];
        route = tess_classify(queued.x, queued.yzw, f32(pass_index) >= tess.dice.w);
    }
    tess_append(route, lid, pass_index & 1u);
}

// After the last split pass (one thread): the dice dispatches (patches per workgroup: 64 / F² for F <= 2 one
// patch per thread, then 4 and 1) and the frame's statistics.
fn tess_prepare_dice() {
    for (var level = 0u; level <= TESS_MAX_LEVEL; level++) {
        let count = min(atomicLoad(&state[STATE_DICE + level]), DICE_CAPACITY);
        let per_group = select(64u, 64u >> (2u * level), level >= 2u);
        let groups = (count + per_group - 1u) / per_group;
        let args = DISPATCH_DICE + (level * 4u);
        dispatch[args] = min(groups, MAX_DISPATCH);
        dispatch[args + 1u] = (groups + MAX_DISPATCH - 1u) / MAX_DISPATCH;
        dispatch[args + 2u] = 1u;
        atomicAdd(&state[STATE_TOTAL_MICRO], count << (2u * level));
    }
    atomicAdd(&state[STATE_TOTAL_OVERFLOW], atomicLoad(&state[STATE_OVERFLOW]));
    atomicMax(&state[STATE_TOTAL_DEEPEST], atomicLoad(&state[STATE_DEEPEST]));
}
