// Nanite tessellation of the main view (tessellation.rs): displaced clusters into the visibility
// buffer through the software rasteriser's target and two passes (sw_raster.wgsl).
//
//   emit   : one workgroup per software cluster, one thread per triangle of a tessellated
//            cluster: its level-0 patch is culled and routed (diced or split, tess_queue.wgsl)
//   split  : the split passes (tess_queue.wgsl)
//   dice   : per grid level F = 2^LEVEL, the diced patches' micro-triangles are displaced and
//            rasterised, PASS 0 the depth, PASS 1 the payload (sw_raster.wgsl). Patches with F <= 2
//            are diced by one thread each; F = 4 and 8 by 16 and 64 threads (4 patches or one per
//            workgroup): the grid's vertices are displaced and projected into shared memory, then
//            each thread rasterises one micro-triangle.
//
// The payload is the base triangle's, (visible cluster slot << 7 | triangle), as for any software
// triangle, so every pass that reads the visibility buffer (material classification, picking,
// Lumen) sees an ordinary triangle id. Next to it the payload pass stores where on the base
// triangle the pixel's point was displaced from — its barycentrics, interpolated over the
// micro-triangle — in a texture of its own (`tess_lambda`): the G-buffer resolve shades that point
// (tess_resolve.wgsl) without searching for it, which no search could do where a vector
// displacement folds the surface over itself (overhangs).

#include "common.wgsl"
#include "geometry.wgsl"
#include "wind.wgsl"
#include "displacement.wgsl"
#include "displacement_height.wgsl"
#include "tess_core.wgsl"
#include "tess_queue.wgsl"

override PASS: u32 = 0u;
override LATE: bool = false;
override LEVEL: u32 = 0u;

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> cull_state: array<u32>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> page_table: array<u32>;
@group(0) @binding(6) var<storage, read> geometry: array<u32>;
// [0, w*h): depth bits, [w*h, 2*w*h): payload (sw_raster.wgsl).
@group(0) @binding(7) var<storage, read_write> sw_target: array<atomic<u32>>;
// Per pixel: the barycentrics of the displaced point on its base triangle (tess_pack_lambda).
@group(0) @binding(11) var tess_lambda: texture_storage_2d<r32uint, write>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(43) var material_data: texture_2d_array<f32>;
@group(0) @binding(44) var material_sampler: sampler;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(67) var<uniform> tess: Tessellation;

const SUBPIXEL: f32 = 256.0;
// Micro-triangles are about the dice rate; steep displacement stretches some (UE clamps its
// software raster to a 64-pixel square the same way).
const MAX_TRIANGLE_PIXELS: u32 = 4096u;

fn tess_fetch(slot: u32) -> vec2<u32> {
    return visible_clusters[slot].xy;
}

fn tess_pixels_per_unit(item: u32, world: vec3<f32>) -> f32 {
    return tess.dice.y / max(distance(world, tess.eye.xyz), tess.dice.z);
}

// Frustum test of the patch's corners grown by the displacement and the wind.
fn tess_visible(item: u32, p: TessPatch, instance: Instance, strength: f32) -> bool {
    let center = (p.w0 + p.w1 + p.w2) / 3.0;
    let radius = max(max(distance(center, p.w0), distance(center, p.w1)), distance(center, p.w2));
    let s = wind_padded(instance, strength, displacement_padded(instance, vec4<f32>(center, radius)));
    for (var i = 0u; i < 6u; i++) {
        let plane = view.planes[i];
        if dot(plane.xyz, s.xyz) + plane.w < -s.w {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(128)
fn emit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let index = wid.x + (wid.y * groups.x) + select(0u, cull_state[5], LATE);
    var route: TessRoute;
    route.count = 0u;
    // A 2D dispatch overshoots: entries past the list are stale.
    if index < software_cluster_count(LATE, cull_state[2], cull_state[3], cull_state[5], view.cluster_capacity) {
        let slot = view.cluster_capacity - 1u - index;
        let entry = visible_clusters[slot];
        let cluster = clusters[entry.y];
        if is_tessellated(instances[entry.x].flags, cluster) && lid < cluster_triangle_count(cluster) {
            let whole = vec3<u32>(tess_pack(vec3<u32>(TESS_ONE, 0u, 0u)), tess_pack(vec3<u32>(0u, TESS_ONE, 0u)), tess_pack(vec3<u32>(0u, 0u, TESS_ONE)));
            route = tess_classify((slot << TRIANGLE_BITS) | lid, whole, f32(0u) >= tess.dice.w);
        }
    }
    tess_append(route, lid, 0u);
}

@compute @workgroup_size(1)
fn prepare_split() {
    tess_prepare_split();
}

@compute @workgroup_size(64)
fn split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    tess_split(wid, groups, lid);
}

@compute @workgroup_size(1)
fn prepare_dice() {
    tess_prepare_dice();
}

// ---- Dicing -----------------------------------------------------------------------------------

// A displaced grid vertex in the raster: pixel x, y (snapped), NDC depth; w < 0 behind the near
// plane.
fn tess_project(instance: Instance, strength: f32, object: vec3<f32>) -> vec4<f32> {
    let world = wind_world_position(instance, strength, object);
    let clip = view.view_proj * vec4<f32>(world, 1.0);
    if clip.w <= tess.dice.z {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let ndc = clip.xyz / clip.w;
    let pixel = round(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * view.viewport.xy * SUBPIXEL) / SUBPIXEL;
    return vec4<f32>(pixel, ndc.z, 1.0);
}

fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn is_top_left(a: vec2<f32>, b: vec2<f32>) -> bool {
    let d = b - a;
    return (d.y == 0.0 && d.x > 0.0) || d.y < 0.0;
}

// Rasterises a micro-triangle (sw_raster.wgsl's edge functions, top-left rule and passes). `l0`,
// `l1`, `l2`: its vertices' barycentrics on the base triangle (second and third).
fn tess_rasterize(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, l0: vec2<f32>, l1: vec2<f32>, l2: vec2<f32>, id: u32) {
    if min(min(p0.w, p1.w), p2.w) < 0.0 {
        return; // reaches behind the near plane (culled, as UE does)
    }
    let area = edge(p0.xy, p1.xy, p2.xy);
    if area >= 0.0 {
        return; // back-facing or degenerate
    }
    let v0 = p0;
    let v1 = p2;
    let v2 = p1;
    let size = view.viewport.xy;
    let lo = max(floor(min(min(v0.xy, v1.xy), v2.xy)), vec2<f32>(0.0));
    let hi = min(ceil(max(max(v0.xy, v1.xy), v2.xy)), size - 1.0);
    if any(hi < lo) {
        return;
    }
    let extent = vec2<u32>(hi - lo) + 1u;
    if extent.x * extent.y > MAX_TRIANGLE_PIXELS {
        return;
    }
    let a = (v0.xy - lo) * SUBPIXEL;
    let b = (v1.xy - lo) * SUBPIXEL;
    let c = (v2.xy - lo) * SUBPIXEL;
    let bias0 = select(-1.0, 0.0, is_top_left(b, c));
    let bias1 = select(-1.0, 0.0, is_top_left(c, a));
    let bias2 = select(-1.0, 0.0, is_top_left(a, b));
    let inv_area = 1.0 / (-area * SUBPIXEL * SUBPIXEL);
    let pixels = u32(size.x);
    let payload_offset = pixels * u32(size.y);
    for (var y = 0u; y < extent.y; y++) {
        for (var x = 0u; x < extent.x; x++) {
            let p = (vec2<f32>(f32(x), f32(y)) + 0.5) * SUBPIXEL;
            let w0 = edge(b, c, p);
            let w1 = edge(c, a, p);
            let w2 = edge(a, b, p);
            if w0 + bias0 < 0.0 || w1 + bias1 < 0.0 || w2 + bias2 < 0.0 {
                continue;
            }
            let z = (w0 * v0.z + w1 * v1.z + w2 * v2.z) * inv_area;
            if z <= 0.0 {
                continue;
            }
            let pixel = vec2<u32>(lo) + vec2<u32>(x, y);
            let target_index = pixel.y * pixels + pixel.x;
            let bits = bitcast<u32>(z);
            if PASS == 0u {
                if bits > atomicLoad(&sw_target[target_index]) {
                    atomicMax(&sw_target[target_index], bits);
                }
            } else if bits == atomicLoad(&sw_target[target_index]) {
                atomicStore(&sw_target[payload_offset + target_index], id);
                // Perspective-correct: the depth is near / w (reverse-Z, infinite far plane), so
                // the weights times it are the weights over w, and their sum is z.
                let lambda = ((l0 * (w0 * v0.z)) + (l2 * (w1 * v1.z)) + (l1 * (w2 * v2.z))) * (inv_area / z);
                textureStore(tess_lambda, pixel, vec4<u32>(tess_pack_lambda(lambda), 0u, 0u, 0u));
            }
        }
    }
}

// Patches with F <= 2 (at most 6 vertices, 4 micro-triangles): one thread each.
fn dice_patch(index: u32) {
    let record = patches[tess_dice_offset(LEVEL) + index];
    let b = tess_base(record.x);
    let instance = instances[b.instance];
    let strength = wind_strength(instance);
    let p = tess_patch(record.x, b, instance, record.yzw);
    let d = tess_dice(p);
    let f = 1u << LEVEL;
    let vertices = ((f + 1u) * (f + 2u)) / 2u;
    var own: array<vec3<f32>, 6>;
    for (var v = 0u; v < vertices; v++) {
        let ij = tess_grid_vertex(v, f);
        own[v] = tess_displaced(instance, tess_grid_point(b, p, d, ij.x, ij.y));
    }
    var screen: array<vec4<f32>, 6>;
    var lambda: array<vec2<f32>, 6>;
    for (var v = 0u; v < vertices; v++) {
        let ij = tess_grid_vertex(v, f);
        let m = tess_grid_morph(p, d, ij.x, ij.y);
        let position = tess_morphed(own[v], own[tess_grid_index(m.a.x, m.a.y, f)], own[tess_grid_index(m.c.x, m.c.y, f)], m.weight);
        screen[v] = tess_project(instance, strength, position);
        if PASS == 1u {
            lambda[v] = tess_grid_lambda(p, d, ij.x, ij.y).yz;
        }
    }
    for (var t = 0u; t < f * f; t++) {
        let tri = tess_grid_triangle(t, f);
        tess_rasterize(screen[tri.x], screen[tri.y], screen[tri.z], lambda[tri.x], lambda[tri.y], lambda[tri.z], record.x);
    }
}

// Patches with F = 4 and 8: PATCHES per workgroup.
const MAX_PATCHES: u32 = 4u;
var<workgroup> wg_base: array<TessBase, MAX_PATCHES>;
var<workgroup> wg_patch: array<TessPatch, MAX_PATCHES>;
var<workgroup> wg_dice: array<TessDice, MAX_PATCHES>;
var<workgroup> wg_item: array<u32, MAX_PATCHES>;
var<workgroup> wg_displaced: array<vec3<f32>, 64>;
var<workgroup> wg_screen: array<vec4<f32>, 64>;
var<workgroup> wg_lambda: array<vec2<f32>, 64>;

@compute @workgroup_size(64)
fn dice(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let group = wid.x + (wid.y * groups.x);
    let count = min(atomicLoad(&state[STATE_DICE + LEVEL]), DICE_CAPACITY);
    if LEVEL <= 1u {
        let index = (group * 64u) + lid;
        if index < count {
            dice_patch(index);
        }
        return;
    }
    let f = 1u << LEVEL;
    let patches_per_group = 64u >> (2u * LEVEL);
    let vertices = ((f + 1u) * (f + 2u)) / 2u;
    let first = group * patches_per_group;
    if lid < patches_per_group && first + lid < count {
        let record = patches[tess_dice_offset(LEVEL) + first + lid];
        let b = tess_base(record.x);
        let p = tess_patch(record.x, b, instances[b.instance], record.yzw);
        wg_base[lid] = b;
        wg_patch[lid] = p;
        wg_dice[lid] = tess_dice(p);
        wg_item[lid] = record.x;
    }
    workgroupBarrier();
    // Every grid vertex displaced once, then morphed from its neighbours' displaced positions.
    let local_patch = lid / vertices;
    let in_patch = lid < patches_per_group * vertices && first + local_patch < count;
    let ij = tess_grid_vertex(lid % vertices, f);
    if in_patch {
        let instance = instances[wg_base[local_patch].instance];
        wg_displaced[lid] = tess_displaced(instance, tess_grid_point(wg_base[local_patch], wg_patch[local_patch], wg_dice[local_patch], ij.x, ij.y));
    }
    workgroupBarrier();
    if in_patch {
        let instance = instances[wg_base[local_patch].instance];
        let m = tess_grid_morph(wg_patch[local_patch], wg_dice[local_patch], ij.x, ij.y);
        let base = local_patch * vertices;
        let position = tess_morphed(wg_displaced[lid], wg_displaced[base + tess_grid_index(m.a.x, m.a.y, f)], wg_displaced[base + tess_grid_index(m.c.x, m.c.y, f)], m.weight);
        wg_screen[lid] = tess_project(instance, wind_strength(instance), position);
        if PASS == 1u {
            wg_lambda[lid] = tess_grid_lambda(wg_patch[local_patch], wg_dice[local_patch], ij.x, ij.y).yz;
        }
    }
    workgroupBarrier();
    let triangle_patch = lid / (f * f);
    if first + triangle_patch < count {
        let tri = tess_grid_triangle(lid % (f * f), f) + (triangle_patch * vertices);
        tess_rasterize(wg_screen[tri.x], wg_screen[tri.y], wg_screen[tri.z], wg_lambda[tri.x], wg_lambda[tri.y], wg_lambda[tri.z], wg_item[triangle_patch]);
    }
}
