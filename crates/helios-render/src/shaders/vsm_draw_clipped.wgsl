// vsm_draw.wgsl's page raster with hardware clip distances (WebGPU's optional "clip-distances"
// feature; vsm.rs uses this variant where the device has it).
//
// Without viewport arrays a page draw is clipped to the whole atlas, and its fragments outside
// the page's tile are only discarded. Triangles larger than a page - local light pages near the
// light, and above all triangles reaching behind a light's near plane, which project across the
// whole atlas - then rasterise far more texels than they write. Four clip distances, one per tile
// edge, make the hardware clip every triangle to its tile instead.

enable clip_distances;

#include "vsm_draw.wgsl"

struct ClippedRasterOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) tile: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) mask: u32,
    @builtin(clip_distances) clip: array<f32, 4>,
}

@vertex
fn vs_raster_clipped(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> ClippedRasterOutput {
    let r = raster_vertex(vertex_index, instance_index);
    var out: ClippedRasterOutput;
    out.position = r.position;
    out.tile = r.tile;
    out.uv = r.uv;
    out.mask = r.mask;
    // Homogeneous page texel position (x, y) * w, recovered from the clip position (vs_raster).
    let w = r.position.w;
    let page = (r.position.xy * vec2<f32>(1.0, -1.0) + w) * (0.5 * vsm.atlas_size) - r.tile * w;
    let n = f32(vsm.page_texels) * w;
    out.clip = array<f32, 4>(page.x, n - page.x, page.y, n - page.y);
    return out;
}
