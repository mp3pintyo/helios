// Nanite tessellation of virtual shadow map pages (tessellation.rs): the shadow counterpart of
// tess.wgsl for (cluster, page) software items of displaced clusters, which vsm_raster.wgsl skips.
// Shadows of the displaced surface, in the sun's clipmap levels and local lights' faces alike.
//
// Tessellation factors are measured in the page's texels (its level's or face mip's texel size at
// the point), so a page is diced like the main view at the matching resolution; patches are culled
// against the page. Heights are the main view's (displacement_height.wgsl), so shadows fall from
// the surface the camera sees. Micro-triangles are rasterised as vsm_raster.wgsl does — 24.8
// fixed-point edge functions, clipped to the page, slope-scaled depth bias, one atomicMax of the
// reverse-Z depth bits per texel into the atlas' u32 mirror (scratch tiles included: displacement
// is static, so the static / dynamic page cache treats displaced instances like any other).

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_local_common.wgsl"
#include "geometry.wgsl"
#include "wind.wgsl"
#include "displacement.wgsl"
#include "displacement_height.wgsl"
#include "tess_core.wgsl"
#include "tess_queue.wgsl"

override LEVEL: u32 = 0u;
// Slope-scaled depth bias, as the other page rasterisers apply it (vsm.rs).
override SLOPE_BIAS: f32 = 1.5;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> draws: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> page_table: array<u32>;
@group(0) @binding(6) var<storage, read> geometry: array<u32>;
// Reverse-Z depth bits per physical atlas texel (0 = nothing rasterised).
@group(0) @binding(7) var<storage, read_write> sw_depth: array<atomic<u32>>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(43) var material_data: texture_2d_array<f32>;
@group(0) @binding(44) var material_sampler: sampler;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(67) var<uniform> tess: Tessellation;

const DRAW_HEADER: u32 = 4u; // vec4s before the first draw entry (see vsm_expand.wgsl)
const SUBPIXEL: i32 = 256;
// Micro-triangles are about the dice rate in texels; this bound keeps the fixed-point edge
// products in i32 (as in vsm_raster.wgsl).
const MAX_EXTENT: i32 = 96 * 256;

// The index-th software page item (instance, cluster, physical page | level << 16, page tag).
fn page_draw(index: u32) -> vec4<u32> {
    return draws[DRAW_HEADER + vsm.draw_capacity - 1u - index];
}

fn tess_fetch(index: u32) -> vec2<u32> {
    return page_draw(index).xy;
}

// Texels per world unit at `world` in the item's page: its clipmap level's, or its local light
// face mip's at that depth.
fn tess_pixels_per_unit(item: u32, world: vec3<f32>) -> f32 {
    let level = page_draw(item >> TRIANGLE_BITS).z >> 16u;
    if level < LOCAL_LEVEL {
        return 1.0 / vsm.level_info[level].y;
    }
    let view = (level - LOCAL_LEVEL) / LOCAL_MIPS;
    let slot = view / LOCAL_FACES;
    let depth = max(local_face_depth(slot, view % LOCAL_FACES, world), local_vsm.slots[slot].params.y);
    return 1.0 / local_texel(slot, depth, (level - LOCAL_LEVEL) % LOCAL_MIPS);
}

// A world point in the draw's page: (texel x, y, depth, 1) * w, as page_vertex (vsm_page.wgsl)
// maps cluster vertices.
fn page_point(draw: vec4<u32>, world: vec3<f32>) -> vec4<f32> {
    let level = draw.z >> 16u;
    let page = vec2<f32>(decode_tag(draw.w));
    if level >= LOCAL_LEVEL {
        return local_page_vertex(world, level - LOCAL_LEVEL, page);
    }
    let ls = (vsm.light_view * vec4<f32>(world, 1.0)).xyz;
    let v = vec2<f32>(ls.x, -ls.y) / vsm.level_info[level].y;
    return vec4<f32>(v - page * f32(vsm.page_texels), (vsm.level_view_proj[level] * vec4<f32>(world, 1.0)).z, 1.0);
}

// The patch's texel box, grown by the displacement and the wind, against the page.
fn tess_visible(item: u32, p: TessPatch, instance: Instance, strength: f32) -> bool {
    let draw = page_draw(item >> TRIANGLE_BITS);
    let center = (p.w0 + p.w1 + p.w2) / 3.0;
    let s = wind_padded(instance, strength, displacement_padded(instance, vec4<f32>(center, 0.0)));
    let pad = s.w * tess_pixels_per_unit(item, center);
    let a = page_point(draw, p.w0);
    let b = page_point(draw, p.w1);
    let c = page_point(draw, p.w2);
    if min(min(a.w, b.w), c.w) <= 0.0 {
        return true; // reaches behind a local light: keep (the micro-triangles are tested)
    }
    let lo = min(min(a.xy / a.w, b.xy / b.w), c.xy / c.w) - pad;
    let hi = max(max(a.xy / a.w, b.xy / b.w), c.xy / c.w) + pad;
    return all(hi >= vec2<f32>(0.0)) && all(lo <= vec2<f32>(f32(vsm.page_texels)));
}

@compute @workgroup_size(128)
fn emit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let index = wid.x + (wid.y * groups.x);
    var route: TessRoute;
    route.count = 0u;
    // The 2D dispatch may overshoot the item count (draws word 5).
    if index < draws[1].y {
        let draw = page_draw(index);
        let cluster = clusters[draw.y];
        if is_tessellated(instances[draw.x].flags, cluster) && lid < cluster_triangle_count(cluster) {
            let whole = vec3<u32>(tess_pack(vec3<u32>(TESS_ONE, 0u, 0u)), tess_pack(vec3<u32>(0u, TESS_ONE, 0u)), tess_pack(vec3<u32>(0u, 0u, TESS_ONE)));
            route = tess_classify((index << TRIANGLE_BITS) | lid, whole, f32(0u) >= tess.dice.w);
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

// A displaced grid vertex in its page: texel x, y, depth; w < 0 behind a local light's near plane.
fn tess_project(draw: vec4<u32>, instance: Instance, strength: f32, object: vec3<f32>) -> vec4<f32> {
    let world = wind_world_position(instance, strength, object);
    let p = page_point(draw, world);
    if p.w <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    return vec4<f32>(p.xyz / p.w, 1.0);
}

fn edge(a: vec2<i32>, b: vec2<i32>, p: vec2<i32>) -> i32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn is_top_left(a: vec2<i32>, b: vec2<i32>) -> bool {
    let d = b - a;
    return (d.y == 0 && d.x > 0) || d.y < 0;
}

// Rasterises a micro-triangle into the draw's page tile (vsm_raster.wgsl).
fn tess_rasterize(draw: vec4<u32>, p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>) {
    if min(min(p0.w, p1.w), p2.w) < 0.0 {
        return;
    }
    let n = i32(vsm.page_texels);
    let a = vec2<i32>(round(p0.xy * f32(SUBPIXEL)));
    var b = vec2<i32>(round(p1.xy * f32(SUBPIXEL)));
    var c = vec2<i32>(round(p2.xy * f32(SUBPIXEL)));
    let box_lo = min(min(a, b), c);
    let box_hi = max(max(a, b), c);
    if any(box_hi < vec2<i32>(0)) || any(box_lo > vec2<i32>(n * SUBPIXEL)) || any(box_hi - box_lo > vec2<i32>(MAX_EXTENT)) {
        return; // off the page, or stretched beyond the fixed-point range
    }
    let area = edge(a, b, c);
    if area >= 0 {
        return; // back-facing or degenerate
    }
    let tb = b;
    b = c;
    c = tb;
    let z0 = p0.z;
    let z1 = p2.z;
    let z2 = p1.z;
    let half = SUBPIXEL / 2;
    let lo = max((box_lo - half + (SUBPIXEL - 1)) >> vec2<u32>(8u), vec2<i32>(0));
    let hi = min((box_hi - half) >> vec2<u32>(8u), vec2<i32>(n - 1));
    if any(hi < lo) {
        return;
    }
    let bias0 = select(-1, 0, is_top_left(b, c));
    let bias1 = select(-1, 0, is_top_left(c, a));
    let bias2 = select(-1, 0, is_top_left(a, b));
    let inv_area = 1.0 / f32(-area);
    let s = f32(SUBPIXEL) * inv_area;
    let dzdx = (f32(b.y - c.y) * z0 + f32(c.y - a.y) * z1 + f32(a.y - b.y) * z2) * s;
    let dzdy = (f32(c.x - b.x) * z0 + f32(a.x - c.x) * z1 + f32(b.x - a.x) * z2) * s;
    let bias = SLOPE_BIAS * max(abs(dzdx), abs(dzdy));
    let tile = vec2<i32>(vsm_tile(draw.z & 0xffffu));
    let row = i32(vsm.atlas_size);
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let p = vec2<i32>(x, y) * SUBPIXEL + half;
            let w0 = edge(b, c, p);
            let w1 = edge(c, a, p);
            let w2 = edge(a, b, p);
            if w0 + bias0 < 0 || w1 + bias1 < 0 || w2 + bias2 < 0 {
                continue;
            }
            let z = (f32(w0) * z0 + f32(w1) * z1 + f32(w2) * z2) * inv_area - bias;
            if z <= 0.0 {
                continue;
            }
            let texel = tile + vec2<i32>(x, y);
            let i = u32(texel.y * row + texel.x);
            let bits = bitcast<u32>(min(z, 1.0));
            if bits > atomicLoad(&sw_depth[i]) {
                atomicMax(&sw_depth[i], bits);
            }
        }
    }
}

// Patches with F <= 2: one thread each.
fn dice_patch(index: u32) {
    let record = patches[tess_dice_offset(LEVEL) + index];
    let draw = page_draw(record.x >> TRIANGLE_BITS);
    let b = tess_base(record.x);
    let instance = instances[b.instance];
    let strength = wind_strength(instance);
    let p = tess_patch(record.x, b, instance, record.yzw);
    let d = tess_dice(p);
    let f = 1u << LEVEL;
    var page: array<vec4<f32>, 6>;
    for (var v = 0u; v < ((f + 1u) * (f + 2u)) / 2u; v++) {
        let ij = tess_grid_vertex(v, f);
        page[v] = tess_project(draw, instance, strength, tess_vertex_position(instance, tess_grid_vertex_points(b, p, d, ij.x, ij.y)));
    }
    for (var t = 0u; t < f * f; t++) {
        let tri = tess_grid_triangle(t, f);
        tess_rasterize(draw, page[tri.x], page[tri.y], page[tri.z]);
    }
}

const MAX_PATCHES: u32 = 4u;
var<workgroup> wg_base: array<TessBase, MAX_PATCHES>;
var<workgroup> wg_patch: array<TessPatch, MAX_PATCHES>;
var<workgroup> wg_dice: array<TessDice, MAX_PATCHES>;
var<workgroup> wg_draw: array<vec4<u32>, MAX_PATCHES>;
var<workgroup> wg_page: array<vec4<f32>, 64>;

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
        wg_draw[lid] = page_draw(record.x >> TRIANGLE_BITS);
    }
    workgroupBarrier();
    let local_patch = lid / vertices;
    if lid < patches_per_group * vertices && first + local_patch < count {
        let instance = instances[wg_base[local_patch].instance];
        let ij = tess_grid_vertex(lid % vertices, f);
        let vertex = tess_grid_vertex_points(wg_base[local_patch], wg_patch[local_patch], wg_dice[local_patch], ij.x, ij.y);
        wg_page[lid] = tess_project(wg_draw[local_patch], instance, wind_strength(instance), tess_vertex_position(instance, vertex));
    }
    workgroupBarrier();
    let triangle_patch = lid / (f * f);
    if first + triangle_patch < count {
        let tri = tess_grid_triangle(lid % (f * f), f) + (triangle_patch * vertices);
        tess_rasterize(wg_draw[triangle_patch], wg_page[tri.x], wg_page[tri.y], wg_page[tri.z]);
    }
}
