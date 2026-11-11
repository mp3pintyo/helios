// Virtual shadow map page rendering into the physical depth atlas.
//
//   vs_clear / fs_resolve : one quad per dirty page, depth compare Always. Writes the depth the
//               software rasteriser left in its u32 mirror of the atlas (0 = reverse-Z far
//               where it drew nothing) and resets the mirror for the next frame. Pages with a
//               static layer (vsm_common.wgsl) keep it in the mirror instead: a page refreshed
//               for its dynamic casters gets the static layer merged with its software-rasterised
//               dynamic casters (from its scratch tile, which is reset), a split page its static
//               casters so far.
//   vs_raster : instance = (cluster, page) draw, vertex = triangle * 3 + corner. Vertices are
//               projected with the level's orthographic matrix (a local light face's
//               perspective one for local pages, clipped at its near plane), then moved from
//               the page's virtual position to its physical tile; fs_raster discards fragments
//               that spill outside the tile (WebGPU has no viewport arrays) and, for
//               alpha-masked materials, texels that fail the alpha test (leaf-shaped shadows).

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_local_common.wgsl"
#include "geometry.wgsl"
#include "wind.wgsl"
#include "skinning.wgsl"
#include "plant_wind.wgsl"
#include "vsm_page.wgsl"
#include "alpha_mask.wgsl"

const DRAW_HEADER: u32 = 4u; // vec4s before the first draw entry

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> draws: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(4) var<storage, read> page_table: array<u32>;
@group(0) @binding(5) var<storage, read> geometry: array<u32>;
@group(0) @binding(6) var<storage, read> cache_lists: array<u32>;
@group(0) @binding(7) var<storage, read> entries: array<vec4<u32>>;
// Only bound for vs_clear / fs_resolve (vsm_raster.wgsl writes it).
@group(0) @binding(8) var<storage, read_write> sw_depth: array<u32>;
@group(0) @binding(9) var<storage, read> info: array<u32>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(41) var alpha_masks: texture_2d_array<f32>;
@group(0) @binding(42) var alpha_mask_sampler: sampler;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;
@group(0) @binding(150) var plant_bones: texture_2d<f32>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;

override DIRTY_START: u32 = 0u;

fn atlas_to_clip(texel: vec2<f32>) -> vec2<f32> {
    let uv = texel / vsm.atlas_size;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

struct ClearOutput {
    @builtin(position) position: vec4<f32>,
    // The page's PAGE_* state this frame.
    @location(0) @interpolate(flat) state: u32,
    // From the page's tile to its scratch tile, in texels.
    @location(1) @interpolate(flat) scratch: vec2<f32>,
}

@vertex
fn vs_clear(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ClearOutput {
    var out: ClearOutput;
    let entry = cache_lists[DIRTY_START + instance];
    let physical = entries[entry].x;
    let tile = vsm_tile(physical);
    let corner = vec2<f32>(f32((0x32u >> vertex) & 1u), f32((0x2cu >> vertex) & 1u)); // two triangles
    out.position = vec4<f32>(atlas_to_clip(tile + corner * f32(vsm.page_texels)), 0.0, 1.0);
    out.state = info[INFO_PAGE_STATE + physical];
    let scratch = page_scratch(out.state);
    if page_kind(out.state) == PAGE_DYNAMIC && scratch != NO_SCRATCH {
        out.scratch = vsm_tile(vsm.physical_count + scratch) - tile;
    }
    return out;
}

@fragment
fn fs_resolve(in: ClearOutput) -> @builtin(frag_depth) f32 {
    let texel = vec2<u32>(in.position.xy);
    let row = u32(vsm.atlas_size);
    let i = texel.y * row + texel.x;
    let bits = sw_depth[i];
    let kind = page_kind(in.state);
    if kind == PAGE_FULL {
        if bits != 0u {
            sw_depth[i] = 0u; // only touched texels need resetting: saves bandwidth
        }
        return bitcast<f32>(bits);
    }
    // The static layer, and the dynamic casters' software depth: positive floats order like
    // their bits.
    var depth = bits;
    if kind == PAGE_DYNAMIC && page_scratch(in.state) != NO_SCRATCH {
        let s = vec2<u32>(in.position.xy + in.scratch);
        let j = s.y * row + s.x;
        let dynamic_bits = sw_depth[j];
        if dynamic_bits != 0u {
            sw_depth[j] = 0u;
            depth = max(depth, dynamic_bits);
        }
    }
    return bitcast<f32>(depth);
}

struct RasterOutput {
    @builtin(position) position: vec4<f32>,
    // The page's tile in the atlas (texels); fragments outside it belong to other tiles.
    @location(0) @interpolate(flat) tile: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) mask: u32,  // alpha test (alpha_mask.wgsl), 0 if opaque
}

@vertex
fn vs_raster(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> RasterOutput {
    return raster_vertex(vertex_index, instance_index);
}

// vs_raster, callable (vsm_draw_clipped.wgsl adds clip distances to it).
fn raster_vertex(vertex_index: u32, instance_index: u32) -> RasterOutput {
    var out: RasterOutput;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside the clip volume: culled
    let draw = draws[DRAW_HEADER + instance_index];
    let cluster = clusters[draw.y];
    let triangle = vertex_index / 3u;
    if triangle >= cluster_triangle_count(cluster) {
        return out;
    }
    // A cluster is drawn once per page it overlaps, and fragments outside the page are only
    // discarded: cull triangles that miss the page entirely, so large clusters spanning many
    // pages are not rasterised over all of them for every page. Every corner of the triangle
    // makes the same decision.
    let base = cluster_base(cluster);
    let tri = cluster_triangle(cluster, base, triangle);
    let p0 = page_vertex(draw, cluster, base, tri.x);
    let p1 = page_vertex(draw, cluster, base, tri.y);
    let p2 = page_vertex(draw, cluster, base, tri.z);
    // The test is homogeneous (all corners beyond one plane of the page's frustum, including the
    // near plane), so it also holds for local light pages with corners behind the light: those
    // triangles would otherwise be clipped to the whole atlas and rasterised over all of it.
    let n = f32(vsm.page_texels);
    let w = vec3<f32>(p0.w, p1.w, p2.w);
    let x = vec3<f32>(p0.x, p1.x, p2.x);
    let y = vec3<f32>(p0.y, p1.y, p2.y);
    if all(x < vec3<f32>(0.0)) || all(y < vec3<f32>(0.0)) || all(x > n * w) || all(y > n * w) || all(vec3<f32>(p0.z, p1.z, p2.z) > w) {
        return out;
    }
    let corner = vertex_index % 3u;
    let p = select(select(p2, p1, corner == 1u), p0, corner == 0u);
    out.tile = vsm_tile(draw.z & 0xffffu);
    // atlas_to_clip(tile + p.xy / p.w) * p.w: linear in p, so the hardware clips correctly.
    let h = (out.tile * p.w + p.xy) / vsm.atlas_size;
    out.position = vec4<f32>(h.x * 2.0 - p.w, p.w - h.y * 2.0, p.z, p.w);
    let material = instances[draw.x].material;
    out.mask = alpha_mask_of(material);
    if is_alpha_masked(out.mask) {
        out.uv = alpha_mask_vertex_uv(material, cluster, base, tri[corner]);
    }
    return out;
}

@fragment
fn fs_raster(in: RasterOutput) {
    // Derivatives before any (non-uniform) branch.
    let ddx = dpdx(in.uv);
    let ddy = dpdy(in.uv);
    let n = f32(vsm.page_texels);
    let page_texel = in.position.xy - in.tile;
    if any(page_texel < vec2<f32>(0.0)) || any(page_texel > vec2<f32>(n)) {
        discard;
    }
    if is_alpha_masked(in.mask) && !alpha_test(in.mask, in.uv, ddx, ddy) {
        discard;
    }
}
