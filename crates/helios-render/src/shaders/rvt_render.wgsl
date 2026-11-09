// Renders dirty runtime-virtual-texture pages: one 8x8 workgroup per tile of a page,
// evaluating the layered landscape material for every texel (including the filter border).

#include "common.wgsl"
#include "noise.wgsl"
#include "material.wgsl"
#include "svt_off.wgsl"
#include "rvt.wgsl"

@group(0) @binding(0) var<uniform> rvt: RvtParams;
@group(0) @binding(1) var<storage, read> rvt_entries: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> cache_lists: array<u32>;
@group(0) @binding(3) var heightfield: texture_2d<f32>;
@group(0) @binding(4) var material_albedo: texture_2d_array<f32>;
@group(0) @binding(5) var material_data: texture_2d_array<f32>;
@group(0) @binding(6) var material_sampler: sampler;
@group(0) @binding(7) var out_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var out_detail: texture_storage_2d<rgba8unorm, write>;

override DIRTY_START: u32 = 0u;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn height_at(object_xz: vec2<f32>) -> f32 {
    let g = clamp((object_xz - rvt.hf_origin) / rvt.hf_cell, vec2<f32>(0.0), vec2<f32>(rvt.hf_resolution - 1u));
    let i = vec2<u32>(floor(g));
    let f = fract(g);
    let j = min(i + 1u, rvt.hf_resolution - 1u);
    let h00 = textureLoad(heightfield, i, 0).r;
    let h10 = textureLoad(heightfield, vec2<u32>(j.x, i.y), 0).r;
    let h01 = textureLoad(heightfield, vec2<u32>(i.x, j.y), 0).r;
    let h11 = textureLoad(heightfield, j, 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

@compute @workgroup_size(8, 8)
fn render_pages(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let span = rvt.page_texels + 2u * rvt.border;
    let tiles = span / 8u;
    let flat = wid.x + wid.y * groups.x;
    let dirty_index = flat / (tiles * tiles);
    if dirty_index >= cache_lists[2] {
        return;
    }
    let sub = flat % (tiles * tiles);
    let texel = vec2<u32>(sub % tiles, sub / tiles) * 8u + lid.xy;
    let entry = cache_lists[DIRTY_START + dirty_index];
    let physical = rvt_entries[entry].x;

    var mip = 0u;
    for (var m = 1u; m < rvt.mips; m++) {
        if entry >= rvt_mip_offset(m) {
            mip = m;
        }
    }
    let pages = rvt_pages(mip);
    let local = entry - rvt_mip_offset(mip);
    let page = vec2<u32>(local % pages, local / pages);

    let virtual_texels = f32(pages * rvt.page_texels);
    let vt = vec2<f32>(page * rvt.page_texels) + vec2<f32>(texel) - f32(rvt.border) + 0.5;
    let uv = vt / virtual_texels;
    let world_xz = rvt.world_min + uv * rvt.world_size;
    let texel_world = rvt.world_size / virtual_texels;

    let object_xz = (world_xz - rvt.hf_translation.xz) / rvt.hf_scale;
    let e = rvt.hf_cell * 0.75;
    let hx = height_at(object_xz + vec2<f32>(e, 0.0)) - height_at(object_xz - vec2<f32>(e, 0.0));
    let hz = height_at(object_xz + vec2<f32>(0.0, e)) - height_at(object_xz - vec2<f32>(0.0, e));
    let macro_normal = normalize(vec3<f32>(-hx, 2.0 * e, -hz));
    let height = height_at(object_xz) * rvt.hf_scale + rvt.hf_translation.y;
    let world = vec3<f32>(world_xz.x, height, world_xz.y);

    let s = terrain_layering(world, macro_normal, texel_world, rvt.layers);
    let detail = transpose(rvt_frame(macro_normal)) * s.normal;

    let tile = vec2<u32>(physical % rvt.tiles_per_row, physical / rvt.tiles_per_row);
    let atlas = tile * span + texel;
    textureStore(out_albedo, atlas, vec4<f32>(linear_to_srgb(saturate(s.albedo)), 1.0));
    textureStore(out_detail, atlas, vec4<f32>(detail.xy * 0.5 + 0.5, saturate(s.roughness), 1.0));
}
