// Software rasterisation of clusters with tiny triangles (Nanite's compute raster path).
//
// WebGPU has no 64-bit atomics, so depth and payload cannot be written in one atomicMax.
// Instead every software cluster is rasterised twice:
//   PASS 0 (depth)   : atomicMax of the reverse-Z depth bits per pixel
//   PASS 1 (payload) : pixels whose depth equals the stored winner write the visibility id
// Ties write identical-depth ids, which is harmless. The merge pass then combines software
// and hardware results per pixel.
//
// One workgroup per cluster: threads transform one vertex each into shared memory, then
// rasterise one triangle each with exact sub-pixel edge functions and the top-left rule.
//
// Clusters of alpha-masked materials are rasterised by a specialisation of this shader
// (MASKED, dispatched over the same clusters; each pipeline skips the other's): it also loads
// UVs and alpha-tests every covered pixel in both passes (the payload pass must agree with the
// depth pass), sampling the mask with the triangle's UV gradients. The opaque specialisation
// compiles without that code, so it keeps its registers and occupancy.

#include "common.wgsl"
#include "geometry.wgsl"
#include "alpha_mask.wgsl"
#include "wind.wgsl"
#include "skinning.wgsl"

override PASS: u32 = 0u;
override LATE: bool = false;
override MASKED: bool = false;

@group(0) @binding(0) var<uniform> view: CullView;
@group(0) @binding(1) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> page_table: array<u32>;
@group(0) @binding(6) var<storage, read> geometry: array<u32>;
// [0, w*h): depth bits, [w*h, 2*w*h): payload.
@group(0) @binding(7) var<storage, read_write> sw_target: array<atomic<u32>>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(41) var alpha_masks: texture_2d_array<f32>;
@group(0) @binding(42) var alpha_mask_sampler: sampler;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;

const SUBPIXEL: f32 = 256.0;
const MAX_TRIANGLE_PIXELS: u32 = 4096u;

var<workgroup> screen: array<vec3<f32>, 128>; // pixel x, pixel y, ndc depth
var<workgroup> uvs: array<vec2<f32>, 128>;     // alpha-masked clusters only

fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Top-left rule in a y-down raster with positive (clockwise on screen) orientation.
fn is_top_left(a: vec2<f32>, b: vec2<f32>) -> bool {
    let d = b - a;
    return (d.y == 0.0 && d.x > 0.0) || d.y < 0.0;
}

@compute @workgroup_size(128)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let index = wid.x + wid.y * groups.x + select(0u, state[5], LATE);
    // A 2D dispatch (over 65535 clusters) overshoots: entries past the list are stale.
    if index >= software_cluster_count(LATE, state[2], state[3], state[5], view.cluster_capacity) {
        return;
    }
    let slot = view.cluster_capacity - 1u - index;
    let entry = visible_clusters[slot];
    let cluster = clusters[entry.y];
    let base = cluster_base(cluster);
    let size = view.viewport.xy;
    let instance = instances[entry.x];
    let material = instance.material;
    let mask = alpha_mask_of(material);
    if is_alpha_masked(mask) != MASKED || is_tessellated(instance.flags, cluster) {
        return; // the other specialisation's or a tessellated cluster (tess.wgsl; uniform: one cluster per workgroup)
    }

    if lid < cluster_vertex_count(cluster) {
        let strength = wind_strength(instance);
        let world = wind_world_position(instance, strength, skinned_vertex_position(instance, cluster, base, lid), vertex_wind_weight(cluster, base, lid, strength));
        let clip = view.view_proj * vec4<f32>(world, 1.0);
        let ndc = clip.xyz / clip.w;
        // Snap to the sub-pixel grid so shared edges rasterise identically in both clusters.
        let pixel = round(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size * SUBPIXEL) / SUBPIXEL;
        screen[lid] = vec3<f32>(pixel, ndc.z);
        if MASKED {
            uvs[lid] = alpha_mask_vertex_uv(material, cluster, base, lid);
        }
    }
    workgroupBarrier();

    if lid >= cluster_triangle_count(cluster) {
        return;
    }
    let tri = cluster_triangle(cluster, base, lid);
    var v0 = screen[tri.x];
    var v1 = screen[tri.y];
    var v2 = screen[tri.z];
    // Counter-clockwise in NDC (front facing) is clockwise in y-down pixels: negative area.
    let area = edge(v0.xy, v1.xy, v2.xy);
    if area >= 0.0 {
        return; // back-facing or degenerate
    }
    // Swap to positive orientation.
    let t = v1;
    v1 = v2;
    v2 = t;
    let positive_area = -area;
    // Alpha-masked: UVs in the same (swapped) order, and their per-pixel gradients.
    var uv0 = vec2<f32>(0.0);
    var uv1 = vec2<f32>(0.0);
    var uv2 = vec2<f32>(0.0);
    var uv_gradients = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    if MASKED {
        uv0 = uvs[tri.x];
        uv1 = uvs[tri.z];
        uv2 = uvs[tri.y];
        uv_gradients = triangle_uv_gradients(v0.xy, v1.xy, v2.xy, uv0, uv1, uv2);
    }

    let lo = max(floor(min(min(v0.xy, v1.xy), v2.xy)), vec2<f32>(0.0));
    let hi = min(ceil(max(max(v0.xy, v1.xy), v2.xy)), size - 1.0);
    if any(hi < lo) {
        return;
    }
    let extent = vec2<u32>(hi - lo) + 1u;
    if extent.x * extent.y > MAX_TRIANGLE_PIXELS {
        return; // never chosen for software raster in practice; guards pathological cases
    }
    // Work relative to the box origin to keep products small (and exact in f32).
    let a = (v0.xy - lo) * SUBPIXEL;
    let b = (v1.xy - lo) * SUBPIXEL;
    let c = (v2.xy - lo) * SUBPIXEL;
    let bias0 = select(-1.0, 0.0, is_top_left(b, c));
    let bias1 = select(-1.0, 0.0, is_top_left(c, a));
    let bias2 = select(-1.0, 0.0, is_top_left(a, b));
    let inv_area = 1.0 / (positive_area * SUBPIXEL * SUBPIXEL);
    let id = (slot << TRIANGLE_BITS) | lid;
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
            // Alpha-tested only where the pixel would be written (`&&` short-circuits).
            let uv = (w0 * uv0 + w1 * uv1 + w2 * uv2) * inv_area;
            if PASS == 0u {
                if bits > atomicLoad(&sw_target[target_index]) && (!MASKED || alpha_test(mask, uv, uv_gradients[0], uv_gradients[1])) {
                    atomicMax(&sw_target[target_index], bits);
                }
            } else if bits == atomicLoad(&sw_target[target_index]) && (!MASKED || alpha_test(mask, uv, uv_gradients[0], uv_gradients[1])) {
                atomicStore(&sw_target[payload_offset + target_index], id);
            }
        }
    }
}
