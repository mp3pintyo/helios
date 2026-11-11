// Prefiltered distant shadow pages (vsm_moments.rs; UE 5.8's "Prefiltered Distant" virtual
// shadow maps). After the page rasters, every distant clipmap page rendered this frame
// (vsm.refresh.z and beyond) is converted into a moment shadow map tile: four power moments of
// the depth (Peters and Klein 2015, "Moment Shadow Mapping"), box-filtered over 2x2 depth texels
// into a 64² tile of an rgba16float atlas with the same tile layout as the depth atlas, plus two
// mips (32², 16²). The lighting pass filters the moments linearly (bilinear, across mips by the
// receiver's footprint) and reconstructs a soft visibility with the Hamburger 4MSM
// (vsm_prefiltered.wgsl): wide, alias-free filtering from a few taps.
//
// rgba16float keeps 11 significant bits: moments of the depth over the scene's whole depth range
// along the sun would blur contact shadows by metres. Each page therefore normalises its depth to
// its own range [zmin, zmax] (of the texels that hold a caster); the range is stored in the tile's
// texel of mip 6 (one texel per tile), rounded outwards to half precision, so that the writer and
// the reader normalise identically. Texels without a caster (depth 0: the far plane) count as the
// farthest depth of the range.
//
// One workgroup of 16x16 threads per physical page, dispatched for the whole pool (pages that are
// not distant or not rendered this frame exit at once); each thread owns 4x4 moment texels (8x8
// depth texels).

#include "vsm_common.wgsl"

const SUN_LEVELS: u32 = 16u;
const NO_ENTRY: u32 = 0xffffffffu;
const MOMENT_TILE: u32 = 64u;

@group(0) @binding(0) var<uniform> vsm: Vsm;
@group(0) @binding(1) var<storage, read> entries: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> owner: array<u32>;
@group(0) @binding(3) var atlas: texture_depth_2d;
@group(0) @binding(4) var moments0: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var moments1: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var moments2: texture_storage_2d<rgba16float, write>;
// Mip 6 of the same texture: per tile its depth range.
@group(0) @binding(7) var range_out: texture_storage_2d<rgba16float, write>;

var<workgroup> page_needed: u32;
var<workgroup> depth_min: atomic<u32>;
var<workgroup> depth_max: atomic<u32>;

// The moments of reverse-Z depth `d` normalised to [-1, 1] (growing away from the light) over the
// page's range: `lo` its farthest depth, `inv` 1 / (nearest - farthest).
fn depth_moments(d: f32, lo: f32, inv: f32) -> vec4<f32> {
    let z = select(1.0, clamp(1.0 - 2.0 * (d - lo) * inv, -1.0, 1.0), d > 0.0);
    let z2 = z * z;
    return vec4<f32>(z, z2, z2 * z, z2 * z2);
}

@compute @workgroup_size(16, 16)
fn build(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let p = wid.x;
    if li == 0u {
        var needed = 0u;
        let lp2 = vsm.level_pages * vsm.level_pages;
        let entry = owner[p];
        if entry != NO_ENTRY && entry < SUN_LEVELS * lp2 && entries[entry].x == p && entries[entry].w == vsm.frame {
            let level = entry / lp2;
            needed = select(0u, 1u, level >= vsm.refresh.z && level < vsm.level_count);
        }
        page_needed = needed;
        atomicStore(&depth_min, 0x7f800000u);
        atomicStore(&depth_max, 0u);
    }
    if workgroupUniformLoad(&page_needed) == 0u {
        return;
    }
    let tile = vec2<u32>(p % vsm.tiles_per_row, p / vsm.tiles_per_row);
    let base = tile * vsm.page_texels + lid.xy * 8u;

    // The page's depth range (positive floats order like their bits).
    var lo = 3.0e38;
    var hi = 0.0;
    for (var y = 0u; y < 8u; y++) {
        for (var x = 0u; x < 8u; x++) {
            let d = textureLoad(atlas, base + vec2<u32>(x, y), 0);
            if d > 0.0 {
                lo = min(lo, d);
                hi = max(hi, d);
            }
        }
    }
    if hi > 0.0 {
        atomicMin(&depth_min, bitcast<u32>(lo));
        atomicMax(&depth_max, bitcast<u32>(hi));
    }
    workgroupBarrier();
    var zmin = bitcast<f32>(atomicLoad(&depth_min));
    var zmax = bitcast<f32>(atomicLoad(&depth_max));
    if zmax <= 0.0 {
        zmin = 0.0;
        zmax = 1.0;
    }
    // Rounded outwards to half precision (whose relative error is below 2^-11), as stored.
    let range_lo = quantizeToF16(zmin * 0.999);
    let range_hi = quantizeToF16(zmax * 1.001);
    let inv = 1.0 / max(range_hi - range_lo, 1e-7);

    // Mip 0: 4x4 texels per thread, each the mean of 2x2 depth texels; mips 1 and 2 their means.
    var quads = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let out0 = tile * MOMENT_TILE + lid.xy * 4u;
    for (var my = 0u; my < 4u; my++) {
        for (var mx = 0u; mx < 4u; mx++) {
            let t = base + vec2<u32>(mx, my) * 2u;
            var m = depth_moments(textureLoad(atlas, t, 0), range_lo, inv);
            m += depth_moments(textureLoad(atlas, t + vec2<u32>(1u, 0u), 0), range_lo, inv);
            m += depth_moments(textureLoad(atlas, t + vec2<u32>(0u, 1u), 0), range_lo, inv);
            m += depth_moments(textureLoad(atlas, t + vec2<u32>(1u, 1u), 0), range_lo, inv);
            m *= 0.25;
            textureStore(moments0, out0 + vec2<u32>(mx, my), m);
            let q = (my / 2u) * 2u + mx / 2u;
            quads[q] += m;
        }
    }
    var total = vec4<f32>(0.0);
    let out1 = tile * (MOMENT_TILE / 2u) + lid.xy * 2u;
    for (var q = 0u; q < 4u; q++) {
        textureStore(moments1, out1 + vec2<u32>(q % 2u, q / 2u), quads[q] * 0.25);
        total += quads[q];
    }
    textureStore(moments2, tile * (MOMENT_TILE / 4u) + lid.xy, total * (1.0 / 16.0));
    if li == 0u {
        textureStore(range_out, tile, vec4<f32>(range_lo, range_hi, 0.0, 0.0));
    }
}
