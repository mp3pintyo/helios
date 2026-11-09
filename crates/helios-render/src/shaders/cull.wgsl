// GPU-driven hierarchical culling, LOD selection and streaming feedback.
//
//   cull_instances : instance frustum test -> batches of up to 64 pages
//   cull_pages     : one thread per page: residency, frustum and LOD pruning -> page work items
//   cull_clusters  : one thread per cluster: LOD cut with residency, frustum, two-pass HZB
//                    occlusion, streaming feedback and hardware/software raster binning
//
// CULL_MODE selects the occlusion strategy:
//   0 = single pass: draw everything that passes LOD + frustum
//   1 = early pass: draw clusters that were visible last frame
//   2 = late pass: test against the HZB built from the early pass, draw newly visible
//       clusters and update the persistent visibility bits
//
// The work buffer holds batches in [0, batch_capacity) and page items after them.
// The atomics buffer holds page feedback in [0, page_count) and visibility bits after it.
// Visible clusters: hardware-raster entries grow from the front, software-raster entries
// from the back, so both share one slot space for the visibility buffer. In the main view,
// hardware clusters of alpha-masked materials are flagged CLUSTER_MASKED and their slots also
// listed after the entries (`masked_capacity` u32s, four per vec4) for the masked draw.
// Bounding spheres of instances the wind moves are padded by their maximum displacement
// (wind.wgsl), those of skinned instances grown to hold their current pose (skin_palette.wgsl), for
// every visibility test (frustum, shadow pages, HZB), not for the LOD cut.
// Shadow pages refreshed for their dynamic casters only (vsm_common.wgsl) keep static casters
// out from the instance test on.

#include "common.wgsl"
#include "wind.wgsl"
#include "skin_palette.wgsl"

override CULL_MODE: u32 = 0u;

const VIEW_MAIN: u32 = 0u;
const VIEW_SHADOW: u32 = 1u;
const PAGES_PER_BATCH: u32 = 64u;

struct CullState {
    batch_count: atomic<u32>,
    work_count: atomic<u32>,
    hw_count: atomic<u32>,
    sw_count: atomic<u32>,
    hw_early: u32,
    sw_early: u32,
    triangle_count: atomic<u32>,
    sw_triangles: atomic<u32>,
    overflow: atomic<u32>,
    masked_count: atomic<u32>,
    masked_early: u32,
    _pad: array<u32, 5>,
}

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> pages: array<Page>;
@group(0) @binding(3) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(4) var<storage, read> page_table: array<u32>;
@group(0) @binding(5) var<storage, read_write> work: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> visible_clusters: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> state: CullState;
@group(0) @binding(8) var<storage, read_write> atomics: array<atomic<u32>>;
@group(0) @binding(9) var hzb: texture_2d<f32>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;

fn sphere_in_frustum(s: vec4<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = view.planes[i];
        if dot(plane.xyz, s.xyz) + plane.w < -s.w {
            return false;
        }
    }
    return true;
}

fn world_error(error: f32, scale: f32) -> f32 {
    return select(error * scale, ROOT_ERROR, error >= ROOT_ERROR);
}

// Screen-space error in pixels. Mirrors `LodView::projected_error` in helios-geometry.
fn projected_error(error: f32, sphere: vec4<f32>) -> f32 {
    if error <= 0.0 {
        return 0.0;
    }
    if error >= ROOT_ERROR {
        return ROOT_ERROR;
    }
    if view.orthographic != 0u {
        return error * view.lod_scale;
    }
    if view.kind == VIEW_SHADOW {
        return error * local_light_scale(sphere);
    }
    let d = distance(sphere.xyz, view.position.xyz) - sphere.w;
    if d <= 0.0 {
        return ROOT_ERROR;
    }
    return error * view.lod_scale / d;
}

// Local light shadows (the perspective shadow view): one view covers every light with a
// virtual shadow map, view.vsm.z of them. Per light, the atomics binding holds (vsm.wgsl,
// LOCAL_SLOT_WORDS words from word view.vsm.x): a box around its dirty pages in units of
// view.vsm.y (min xyz, max xyz), its finest dirty mip, dirty page count, position and mip-0
// texels per unit at distance 1, then the box around its pages rendered in full.
const LOCAL_SLOT_WORDS: u32 = 32u;
const LOCAL_FULL_BOX: u32 = 12u;

fn local_word(light: u32, i: u32) -> u32 {
    return atomicLoad(&atomics[u32(view.vsm.x) + light * LOCAL_SLOT_WORDS + i]);
}

// Texels per world unit at `sphere` (at its nearest point) in the finest dirty mip of any light
// with dirty pages: LOD is selected for the most demanding of them. Every dirty light counts,
// whether the sphere is near it or not, so own and parent errors are measured alike.
fn local_light_scale(sphere: vec4<f32>) -> f32 {
    var scale = 0.0;
    for (var light = 0u; light < u32(view.vsm.z); light++) {
        if local_word(light, 7u) == 0u {
            continue;
        }
        let position = vec3<f32>(bitcast<f32>(local_word(light, 8u)), bitcast<f32>(local_word(light, 9u)), bitcast<f32>(local_word(light, 10u)));
        let d = distance(sphere.xyz, position) - sphere.w;
        if d <= 0.0 {
            return ROOT_ERROR;
        }
        let mip = min(local_word(light, 6u), 31u);
        scale = max(scale, bitcast<f32>(local_word(light, 11u)) / f32(1u << mip) / d);
    }
    return scale;
}

// Virtual shadow map levels only render dirty pages: the atomics binding then holds each
// level's dirty page rectangle and a bit per dirty page (the info buffer of vsm.wgsl). Anything
// outside the rectangle is culled; spheres covering a few pages are tested against their bits,
// since the dirty pages of a level may be scattered thinly over a large rectangle (wind). For
// local lights, anything outside every light's dirty box. Static casters (`dynamic` false) are
// tested against the pages rendered in full only: the others keep their static layer.
const VSM_DIRTY_MASK: u32 = 136u;
const VSM_FULL_MASK: u32 = 8328u;
const VSM_LEVEL_PAGES: i32 = 128;
const VSM_MASK_TEST_PAGES: i32 = 16;

fn vsm_overlaps(sphere: vec4<f32>, dynamic: bool) -> bool {
    if view.kind != VIEW_SHADOW {
        return true;
    }
    if view.orthographic == 0u {
        let box = select(LOCAL_FULL_BOX, 0u, dynamic);
        for (var light = 0u; light < u32(view.vsm.z); light++) {
            let lo = vec3<i32>(bitcast<i32>(local_word(light, box)), bitcast<i32>(local_word(light, box + 1u)), bitcast<i32>(local_word(light, box + 2u)));
            let hi = vec3<i32>(bitcast<i32>(local_word(light, box + 3u)), bitcast<i32>(local_word(light, box + 4u)), bitcast<i32>(local_word(light, box + 5u)));
            let nearest = clamp(sphere.xyz, vec3<f32>(lo) * view.vsm.y, vec3<f32>(hi) * view.vsm.y);
            if all(hi >= lo) && distance(nearest, sphere.xyz) <= sphere.w {
                return true;
            }
        }
        return false;
    }
    let base = u32(view.vsm.x) * 8u + select(4u, 0u, dynamic);
    let mask = select(VSM_FULL_MASK, VSM_DIRTY_MASK, dynamic);
    let lo = vec2<i32>(bitcast<i32>(atomicLoad(&atomics[base])), bitcast<i32>(atomicLoad(&atomics[base + 1u])));
    let hi = vec2<i32>(bitcast<i32>(atomicLoad(&atomics[base + 2u])), bitcast<i32>(atomicLoad(&atomics[base + 3u])));
    if any(hi < lo) {
        return false;
    }
    let ls = (view.view * vec4<f32>(sphere.xyz, 1.0)).xyz;
    let v = vec2<f32>(ls.x, -ls.y) / view.vsm.y;
    let r = sphere.w / view.vsm.y;
    let p0 = max(vec2<i32>(floor(v - r)), lo);
    let p1 = min(vec2<i32>(floor(v + r)), hi);
    if any(p1 < p0) {
        return false;
    }
    let span = p1 - p0 + 1;
    if span.x * span.y > VSM_MASK_TEST_PAGES {
        return true;
    }
    let n = VSM_LEVEL_PAGES;
    let level_base = u32(view.vsm.x) * u32(n * n);
    for (var y = p0.y; y <= p1.y; y++) {
        for (var x = p0.x; x <= p1.x; x++) {
            let wrapped = ((vec2<i32>(x, y) % n) + n) % n;
            let entry = level_base + u32((wrapped.y * n) + wrapped.x);
            if (atomicLoad(&atomics[mask + entry / 32u]) & (1u << (entry % 32u))) != 0u {
                return true;
            }
        }
    }
    return false;
}

fn is_resident(page: u32) -> bool {
    return page != INVALID_PAGE && page_table[page] != INVALID_ID;
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= view.instance_count {
        return;
    }
    let instance = instances[index];
    if view.kind == VIEW_SHADOW && (instance.flags & INSTANCE_CAST_SHADOW) == 0u {
        return;
    }
    // Translucent surfaces must not occlude the opaque scene (visibility buffer, HZB).
    if view.kind == VIEW_MAIN && (instance.flags & INSTANCE_TRANSLUCENT) != 0u {
        return;
    }
    let strength = wind_strength(instance);
    let bounds = skin_padded(instance, wind_padded(instance, strength, instance.bounds));
    if !sphere_in_frustum(bounds) || !vsm_overlaps(bounds, dynamic_caster(strength) || skin_dynamic(instance)) {
        return;
    }
    let batches = (instance.page_count + PAGES_PER_BATCH - 1u) / PAGES_PER_BATCH;
    let first = atomicAdd(&state.batch_count, batches);
    for (var b = 0u; b < batches; b++) {
        let slot = first + b;
        if slot >= view.batch_capacity {
            atomicAdd(&state.overflow, 1u);
            return;
        }
        let start = b * PAGES_PER_BATCH;
        work[slot] = vec4<u32>(index, start, min(PAGES_PER_BATCH, instance.page_count - start), 0u);
    }
}

var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(64)
fn cull_pages(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if lid == 0u {
        atomicStore(&wg_count, 0u);
    }
    workgroupBarrier();

    let batch_index = wid.x + wid.y * groups.x;
    let batch_count = min(atomicLoad(&state.batch_count), view.batch_capacity);
    var emit = false;
    var instance_id = 0u;
    var page_id = 0u;
    if batch_index < batch_count {
        let batch = work[batch_index];
        if lid < batch.z {
            instance_id = batch.x;
            let instance = instances[instance_id];
            page_id = instance.page_offset + batch.y + lid;
            let page = pages[page_id];
            let scale = instance.max_scale;
            let strength = wind_strength(instance);
            let bounds = skin_padded(instance, wind_padded(instance, strength, transform_sphere(instance.world, page.bounds, scale)));
            if page_table[page_id] != INVALID_ID && sphere_in_frustum(bounds) && vsm_overlaps(bounds, dynamic_caster(strength) || skin_dynamic(instance)) {
                // If even the coarsest parent in the page is fine enough, every cluster in it
                // would be rejected by the LOD test: skip the page entirely.
                let parent = transform_sphere(instance.world, page.parent_lod_bounds, scale);
                emit = projected_error(world_error(page.max_parent_error, scale), parent) > view.lod_threshold;
            }
        }
    }

    var local_slot = 0u;
    if emit {
        local_slot = atomicAdd(&wg_count, 1u);
    }
    workgroupBarrier();
    if lid == 0u {
        let count = atomicLoad(&wg_count);
        if count > 0u {
            wg_base = atomicAdd(&state.work_count, count);
        }
    }
    workgroupBarrier();
    if emit {
        let slot = wg_base + local_slot;
        if slot < view.work_capacity {
            work[view.batch_capacity + slot] = vec4<u32>(instance_id, page_id, 0u, 0u);
        } else {
            atomicAdd(&state.overflow, 1u);
        }
    }
}

// Conservative HZB occlusion test of a world-space sphere (reverse-Z: larger = nearer).
fn is_occluded(sphere: vec4<f32>) -> bool {
    let near = view.position.w;
    let center_view = (view.view * vec4<f32>(sphere.xyz, 1.0)).xyz;
    let z = -center_view.z;
    if z - sphere.w * 1.7320508 <= near {
        return false;
    }

    var min_uv = vec2<f32>(1.0);
    var max_uv = vec2<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let offset = vec3<f32>(
            select(-1.0, 1.0, (i & 1u) != 0u),
            select(-1.0, 1.0, (i & 2u) != 0u),
            select(-1.0, 1.0, (i & 4u) != 0u),
        );
        let clip = view.view_proj * vec4<f32>(sphere.xyz + offset * sphere.w, 1.0);
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        min_uv = min(min_uv, uv);
        max_uv = max(max_uv, uv);
    }
    min_uv = clamp(min_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    max_uv = clamp(max_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let px_min = min_uv * view.viewport.xy;
    let px_max = max_uv * view.viewport.xy;
    let extent = max(px_max.x - px_min.x, px_max.y - px_min.y);
    // HZB mip L texels cover 2^(L+1) pixels; pick the level where the rect spans <= 2 texels.
    let level = min(u32(max(ceil(log2(max(extent, 1.0))) - 1.0, 0.0)), view.hzb_mips - 1u);
    // Only the corner covered by the internal resolution holds this view's depths.
    let shift = level + 1u;
    let covered = (vec2<u32>(view.viewport.xy) + (1u << shift) - 1u) >> vec2<u32>(shift);
    let level_size = max(min(textureDimensions(hzb, level), covered), vec2<u32>(1u));
    let texel = f32(1u << (level + 1u));
    let t0 = min(vec2<u32>(px_min / texel), level_size - 1u);
    let t1 = min(vec2<u32>(px_max / texel), level_size - 1u);
    let farthest = min(
        min(textureLoad(hzb, t0, level).r, textureLoad(hzb, vec2<u32>(t1.x, t0.y), level).r),
        min(textureLoad(hzb, vec2<u32>(t0.x, t1.y), level).r, textureLoad(hzb, t1, level).r),
    );
    let nearest_depth = near / (z - sphere.w);
    return nearest_depth < farthest;
}

struct ClusterResult {
    draw: bool,
    software: bool,
    masked: bool,  // hardware cluster for the alpha-masked draw
}

fn test_cluster(instance_id: u32, cluster_id: u32) -> ClusterResult {
    var result = ClusterResult(false, false, false);
    let instance = instances[instance_id];
    let cluster = clusters[cluster_id];
    let scale = instance.max_scale;

    let own = projected_error(world_error(cluster.lod_error, scale), transform_sphere(instance.world, cluster.lod_bounds, scale));
    let parent = projected_error(world_error(cluster.parent_error, scale), transform_sphere(instance.world, cluster.parent_lod_bounds, scale));
    let bounds = transform_sphere(instance.world, cluster.bounds, scale);
    let strength = wind_strength(instance);
    let padded = skin_padded(instance, wind_padded(instance, strength, bounds));
    let own_ok = own <= view.lod_threshold;
    let candidate = parent > view.lod_threshold && sphere_in_frustum(padded) && vsm_overlaps(padded, dynamic_caster(strength) || skin_dynamic(instance));
    // Streaming feedback (main view): this cluster is on the cut but wants finer children.
    if candidate && !own_ok && cluster.child_page != INVALID_PAGE && CULL_MODE != 2u && view.kind == VIEW_MAIN {
        atomicMax(&atomics[cluster.child_page], view.frame_index);
    }
    let in_view = candidate && (own_ok || !is_resident(cluster.child_page));

    if CULL_MODE == 0u {
        result.draw = in_view;
    } else {
        let bit_index = instance.visibility_offset + (cluster_id - instance.cluster_offset);
        let word = bit_index >> 5u;
        let bit = 1u << (bit_index & 31u);
        if CULL_MODE == 1u {
            result.draw = in_view && (atomicLoad(&atomics[word]) & bit) != 0u;
        } else if !in_view || is_occluded(padded) {
            atomicAnd(&atomics[word], ~bit);
        } else {
            let previous = atomicOr(&atomics[word], bit);
            result.draw = (previous & bit) == 0u;
        }
    }

    if result.draw && view.sw_raster_threshold > 0.0 {
        result.software = prefers_software_raster(cluster, bounds);
    }
    // The software rasteriser alpha-tests per cluster, shadow rasterisers per draw: only the
    // main view's hardware path needs a separate list.
    result.masked = result.draw && !result.software && view.masked_capacity > 0u && (instance.flags & INSTANCE_ALPHA_MASKED) != 0u;
    return result;
}

// Small triangles go to the software rasteriser. In a perspective view, clusters touching the
// near plane stay on the hardware path, which clips. Sun shadow levels are orthographic: one
// texel is `1 / lod_scale` world units everywhere, and the depth range spans the scene. Local
// light pages are perspective; their expansion (vsm_expand.wgsl) keeps clusters near a light's
// near plane on the hardware path.
fn prefers_software_raster(cluster: Cluster, bounds: vec4<f32>) -> bool {
    var radius_px = bounds.w * view.lod_scale;
    if view.kind == VIEW_SHADOW && view.orthographic == 0u {
        radius_px = bounds.w * local_light_scale(vec4<f32>(bounds.xyz, 0.0));
    } else if view.orthographic == 0u {
        let d = distance(bounds.xyz, view.position.xyz);
        if d - bounds.w <= view.position.w * 4.0 {
            return false;
        }
        radius_px /= d;
    }
    let triangle_px = 2.0 * radius_px / sqrt(f32(max(cluster_triangle_count(cluster), 1u)));
    return triangle_px < view.sw_raster_threshold && radius_px < 48.0;
}

var<workgroup> hw_local: atomic<u32>;
var<workgroup> sw_local: atomic<u32>;
var<workgroup> tri_local: atomic<u32>;
var<workgroup> sw_tri_local: atomic<u32>;
var<workgroup> masked_local: atomic<u32>;
var<workgroup> hw_base: u32;
var<workgroup> sw_base: u32;
var<workgroup> masked_base: u32;

@compute @workgroup_size(64)
fn cull_clusters(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if lid == 0u {
        atomicStore(&hw_local, 0u);
        atomicStore(&sw_local, 0u);
        atomicStore(&tri_local, 0u);
        atomicStore(&sw_tri_local, 0u);
        atomicStore(&masked_local, 0u);
    }
    workgroupBarrier();

    let item_index = wid.x + wid.y * groups.x;
    let item_count = min(atomicLoad(&state.work_count), view.work_capacity);
    var result = ClusterResult(false, false, false);
    var instance_id = 0u;
    var cluster_id = 0u;
    if item_index < item_count {
        let item = work[view.batch_capacity + item_index];
        let page = pages[item.y];
        if lid < page.cluster_count {
            instance_id = item.x;
            cluster_id = page.cluster_offset + lid;
            result = test_cluster(instance_id, cluster_id);
        }
    }

    // Aggregate the workgroup's output into one global atomic per list.
    var local_slot = 0u;
    var masked_slot = 0u;
    if result.draw {
        let triangles = cluster_triangle_count(clusters[cluster_id]);
        atomicAdd(&tri_local, triangles);
        if result.software {
            local_slot = atomicAdd(&sw_local, 1u);
            atomicAdd(&sw_tri_local, triangles);
        } else {
            local_slot = atomicAdd(&hw_local, 1u);
        }
        if result.masked {
            masked_slot = atomicAdd(&masked_local, 1u);
        }
    }
    workgroupBarrier();
    if lid == 0u {
        let hw = atomicLoad(&hw_local);
        let sw = atomicLoad(&sw_local);
        if hw > 0u {
            hw_base = atomicAdd(&state.hw_count, hw);
        }
        if sw > 0u {
            sw_base = atomicAdd(&state.sw_count, sw);
        }
        let masked = atomicLoad(&masked_local);
        if masked > 0u {
            masked_base = atomicAdd(&state.masked_count, masked);
        }
        let tris = atomicLoad(&tri_local);
        if tris > 0u {
            atomicAdd(&state.triangle_count, tris);
            atomicAdd(&state.sw_triangles, atomicLoad(&sw_tri_local));
        }
    }
    workgroupBarrier();
    if result.draw {
        var entry = vec4<u32>(instance_id, cluster_id, select(0u, CLUSTER_SOFTWARE, result.software), 0u);
        if result.masked {
            let index = masked_base + masked_slot;
            if index < view.masked_capacity {
                // Every reserved list item is written; an entry that misses the cluster list
                // below is listed as INVALID_ID and skipped by the masked draw.
                let slot = hw_base + local_slot;
                visible_clusters[view.cluster_capacity + index / 4u][index % 4u] = select(INVALID_ID, slot, slot < view.cluster_capacity);
                entry.z = CLUSTER_MASKED;
            } else {
                atomicAdd(&state.overflow, 1u); // drawn by the opaque pipeline, without alpha test
            }
        }
        if result.software {
            let slot = sw_base + local_slot;
            let hw_total = atomicLoad(&state.hw_count);
            if slot + hw_total < view.cluster_capacity {
                visible_clusters[view.cluster_capacity - 1u - slot] = entry;
            }
        } else {
            let slot = hw_base + local_slot;
            if slot < view.cluster_capacity {
                visible_clusters[slot] = entry;
            }
        }
    }
}
