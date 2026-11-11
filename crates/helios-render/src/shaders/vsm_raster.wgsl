// Software rasterisation of virtual shadow map pages: the shadow counterpart of sw_raster.wgsl
// for (cluster, page) items whose triangles are tiny in the page's clipmap level.
//
// A shadow map stores depth only, so a single 32-bit atomicMax of the reverse-Z depth bits per
// texel resolves visibility: no payload pass. WebGPU has neither storage-texture atomics nor
// storage depth textures, so the rasteriser writes a u32 mirror of the physical atlas; the page
// clear (fs_resolve in vsm_draw.wgsl) then copies it into the depth atlas and resets it, before
// the hardware clusters are drawn on top with depth test Greater.
//
// One workgroup per item: threads transform one vertex each into page texel coordinates in
// shared memory, then rasterise one triangle each, clipped to the page. Everything matches the
// hardware path (vs_raster / fs_raster): vertex transform, texel centres, back-face culling and
// slope-scaled depth bias. Edge functions are evaluated exactly in 24.8 fixed point (i32), so
// triangles sharing an edge never both miss a texel. Alpha-masked materials are rasterised by
// a specialisation (MASKED, dispatched over the same items; each skips the other's) that
// alpha-tests per texel with the triangle's UV gradients per texel (as fs_raster's derivatives).

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_local_common.wgsl"
#include "geometry.wgsl"
#include "wind.wgsl"
#include "skinning.wgsl"
#include "plant_wind.wgsl"
#include "vsm_page.wgsl"
#include "alpha_mask.wgsl"

const DRAW_HEADER: u32 = 4u; // vec4s before the first draw entry (see vsm_expand.wgsl)
const SUBPIXEL: i32 = 256;
// Culling only gives this path clusters with a radius below 48 texels (cull.wgsl), so triangle
// bounds stay within 96 texels and the fixed-point edge products fit in i32.
const MAX_EXTENT: i32 = 96 * 256;

// Slope-scaled depth bias in depth units per texel of depth slope; the hardware pipeline uses
// the same value (vsm.rs).
override SLOPE_BIAS: f32 = 1.5;
override MASKED: bool = false;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> draws: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(4) var<storage, read> page_table: array<u32>;
@group(0) @binding(5) var<storage, read> geometry: array<u32>;
// Reverse-Z depth bits per physical atlas texel (0 = nothing rasterised).
@group(0) @binding(6) var<storage, read_write> sw_depth: array<atomic<u32>>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(41) var alpha_masks: texture_2d_array<f32>;
@group(0) @binding(42) var alpha_mask_sampler: sampler;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;
@group(0) @binding(170) var plant_bones: texture_2d<f32>;
@group(0) @binding(171) var plant_motion: texture_2d<f32>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;

var<workgroup> page_vertices: array<vec3<f32>, 128>; // page texel x, y, depth
var<workgroup> uvs: array<vec2<f32>, 128>;          // alpha-masked clusters only

fn edge(a: vec2<i32>, b: vec2<i32>, p: vec2<i32>) -> i32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Top-left rule in a y-down raster with positive orientation.
fn is_top_left(a: vec2<i32>, b: vec2<i32>) -> bool {
    let d = b - a;
    return (d.y == 0 && d.x > 0) || d.y < 0;
}

@compute @workgroup_size(128)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // The 2D dispatch may overshoot the item count (draws word 5).
    let index = wid.x + wid.y * groups.x;
    let count = draws[1].y;
    let valid = index < count;
    let draw = draws[DRAW_HEADER + vsm.draw_capacity - 1u - min(index, max(count, 1u) - 1u)];
    let cluster = clusters[draw.y];
    let base = cluster_base(cluster);
    let n = i32(vsm.page_texels);
    let material = instances[draw.x].material;
    let mask = alpha_mask_of(material);
    if is_alpha_masked(mask) != MASKED || is_tessellated(instances[draw.x].flags, cluster) {
        return; // the other specialisation's or a tessellated item (vsm_tess.wgsl; uniform: one item per workgroup)
    }

    if valid && lid < cluster_vertex_count(cluster) {
        // Local light pages are perspective; their software items lie in front of the light
        // (vsm_expand.wgsl), so the division is safe.
        let p = page_vertex(draw, cluster, base, lid);
        page_vertices[lid] = p.xyz / p.w;
        if MASKED {
            uvs[lid] = alpha_mask_vertex_uv(material, cluster, base, lid);
        }
    }
    workgroupBarrier();

    if !valid || lid >= cluster_triangle_count(cluster) {
        return;
    }
    let tri = cluster_triangle(cluster, base, lid);
    let v0 = page_vertices[tri.x];
    var v1 = page_vertices[tri.y];
    var v2 = page_vertices[tri.z];
    let a = vec2<i32>(round(v0.xy * f32(SUBPIXEL)));
    var b = vec2<i32>(round(v1.xy * f32(SUBPIXEL)));
    var c = vec2<i32>(round(v2.xy * f32(SUBPIXEL)));
    let box_lo = min(min(a, b), c);
    let box_hi = max(max(a, b), c);
    if any(box_hi - box_lo > vec2<i32>(MAX_EXTENT)) {
        return; // unreachable for clusters culling assigns here; keeps the i32 math exact
    }
    // Front faces are counter-clockwise in clip space, clockwise in the y-down page.
    let area = edge(a, b, c);
    if area >= 0 {
        return; // back-facing or degenerate (the hardware pipeline culls back faces too)
    }
    let tb = b;
    b = c;
    c = tb;
    let tz = v1.z;
    v1.z = v2.z;
    v2.z = tz;
    // Alpha-masked: UVs in the swapped order (a, b, c) and their gradients per texel.
    var uv0 = vec2<f32>(0.0);
    var uv1 = vec2<f32>(0.0);
    var uv2 = vec2<f32>(0.0);
    var uv_gradients = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    if MASKED {
        uv0 = uvs[tri.x];
        uv1 = uvs[tri.z];
        uv2 = uvs[tri.y];
        uv_gradients = triangle_uv_gradients(vec2<f32>(a), vec2<f32>(b), vec2<f32>(c), uv0, uv1, uv2) * f32(SUBPIXEL);
    }

    // Texels whose centre (i + 0.5) lies in the box, clipped to the page.
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
    // Depth plane slope per texel, for the same bias the hardware applies:
    // slope_scale * max(|dz/dx|, |dz/dy|).
    let s = f32(SUBPIXEL) * inv_area;
    let dzdx = (f32(b.y - c.y) * v0.z + f32(c.y - a.y) * v1.z + f32(a.y - b.y) * v2.z) * s;
    let dzdy = (f32(c.x - b.x) * v0.z + f32(a.x - c.x) * v1.z + f32(b.x - a.x) * v2.z) * s;
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
            let z = (f32(w0) * v0.z + f32(w1) * v1.z + f32(w2) * v2.z) * inv_area - bias;
            if z <= 0.0 {
                continue; // the hardware clamps to the far plane, which never passes Greater
            }
            let texel = tile + vec2<i32>(x, y);
            let i = u32(texel.y * row + texel.x);
            let bits = bitcast<u32>(min(z, 1.0));
            let uv = (f32(w0) * uv0 + f32(w1) * uv1 + f32(w2) * uv2) * inv_area;
            if bits > atomicLoad(&sw_depth[i]) && (!MASKED || alpha_test(mask, uv, uv_gradients[0], uv_gradients[1])) {
                atomicMax(&sw_depth[i], bits);
            }
        }
    }
}
