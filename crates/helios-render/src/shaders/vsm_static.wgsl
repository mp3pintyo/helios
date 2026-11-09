// Static layers of virtual shadow map pages (see vsm_common.wgsl): while a page is cached, the
// u32 software raster mirror of its tile holds the depth of its static casters.
//
//   clear_static   : before rasterising, clears the mirror tile of pages rendered in full whose
//                    mirror still holds an older static layer (PAGE_CLEAR)
//   capture_static : once the static casters of a split page are drawn (software raster, page
//                    clear, first hardware raster), copies its tile of the depth atlas into its
//                    mirror tile: the static layer its dynamic refreshes start from
//
// One workgroup per dirty page, dispatched for the whole pool (the rest exit at once).

#include "vsm_common.wgsl"

override DIRTY_START: u32 = 0u;

// The page cache's header word holding the dirty page count (page_cache.wgsl).
const CACHE_DIRTY_COUNT: u32 = 2u;
const GROUP_SIZE: u32 = 256u;
const NO_PAGE: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> cache_lists: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> info: array<u32>;
@group(0) @binding(4) var<storage, read_write> sw_depth: array<u32>;
@group(0) @binding(5) var atlas: texture_depth_2d;

// The physical page of dirty page `index`, or NO_PAGE.
fn dirty_physical(index: u32) -> u32 {
    if index >= cache_lists[CACHE_DIRTY_COUNT] {
        return NO_PAGE;
    }
    return entries[cache_lists[DIRTY_START + index]].x;
}

// Texel `t` of the tile of `physical`: (atlas texel, mirror index).
fn tile_texel(physical: u32, t: u32) -> vec3<u32> {
    let texel = vec2<u32>(vsm_tile(physical)) + vec2<u32>(t % vsm.page_texels, t / vsm.page_texels);
    return vec3<u32>(texel, texel.y * u32(vsm.atlas_size) + texel.x);
}

@compute @workgroup_size(256)
fn clear_static(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let physical = dirty_physical(wid.x);
    if physical == NO_PAGE || (info[INFO_PAGE_STATE + physical] & PAGE_CLEAR) == 0u {
        return;
    }
    for (var t = lid; t < vsm.page_texels * vsm.page_texels; t += GROUP_SIZE) {
        sw_depth[tile_texel(physical, t).z] = 0u;
    }
}

@compute @workgroup_size(256)
fn capture_static(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let physical = dirty_physical(wid.x);
    if physical == NO_PAGE || page_kind(info[INFO_PAGE_STATE + physical]) != PAGE_SPLIT {
        return;
    }
    for (var t = lid; t < vsm.page_texels * vsm.page_texels; t += GROUP_SIZE) {
        let texel = tile_texel(physical, t);
        sw_depth[texel.z] = bitcast<u32>(textureLoad(atlas, texel.xy, 0));
    }
}
