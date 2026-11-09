// Streaming virtual texturing of image texture layers (see svt/mod.rs).
//
// An image layer's levels finer than its 512² array layer are tiles of a virtual texture,
// streamed on demand into three atlases (albedo sRGB; normal xy; roughness, metallic), which
// are BC7 / BC5 / BC5 where the device supports it (sampling is the same either way).
// The page table has one entry per virtual tile: the atlas slot of the finest resident
// ancestor-or-self (16 bits) and the minimum lod that may be sampled there (8.8 fixed point,
// raised while new tiles fade in), or SVT_INVALID if nothing finer than the array is resident.
// Every ancestor of a resident tile is resident, so the level above any sampled level is
// available for trilinear blending; the level above the coarsest streamed one is the array.
//
// Implements `sample_layer` for material.wgsl (the alternative is svt_off.wgsl) and records
// the tile each pixel wanted for the feedback pass (`svt_write_feedback`).
//
// Includers: material.wgsl and the bindings
//   svt: SvtParams (uniform), svt_page_table: texture_2d<u32>, svt_albedo (sRGB) /
//   svt_normal / svt_material: texture_2d<f32>, svt_feedback: texture_storage_2d<r32uint, write>,
//   material_sampler (linear, anisotropic; tiles are never sampled outside their border).

struct SvtParams {
    tiles_per_row: u32,
    tile_texels: u32,
    border: u32,
    base_size: u32,       // side of the array layer
    atlas_size: f32,
    table_width: u32,
    frame: u32,
    enabled: u32,
    feedback_offset: vec2<u32>,
    max_aniso: f32,
    _pad: u32,
    layers: array<vec4<u32>, 64>,  // per layer: page-table offset | streamed levels << 24
}

const SVT_INVALID: u32 = 0xffffffffu;
const SVT_FEEDBACK_BLOCK: u32 = 4u;

// Page-table entry + 1 of the tile this pixel wanted (0: none).
var<private> svt_request: u32 = 0u;

struct SvtLayer {
    offset: u32,
    levels: u32,  // 0: not streamed
}

fn svt_layer(layer: u32) -> SvtLayer {
    var packed = 0u;
    if svt.enabled != 0u && layer < 256u {
        packed = svt.layers[layer / 4u][layer % 4u];
    }
    return SvtLayer(packed & 0xffffffu, packed >> 24u);
}

// Tiles per side of a streamed level.
fn svt_tiles(info: SvtLayer, level: u32) -> u32 {
    return (svt.base_size / svt.tile_texels) << (info.levels - level);
}

// Page-table entry of the tile of `level` that contains `uv` (in [0, 1)).
fn svt_entry(info: SvtLayer, level: u32, uv: vec2<f32>) -> u32 {
    // Levels before `level` hold sum(n_m^2) entries with n_m = n_0 >> m: a geometric series
    // of ratio 1/4 (exact in integers).
    let n0 = svt_tiles(info, 0u);
    let n = svt_tiles(info, level);
    let offset = info.offset + ((n0 * n0 * 4u) - (n * n * 4u)) / 3u;
    let tile = min(vec2<u32>(uv * f32(n)), vec2<u32>(n - 1u));
    return offset + tile.y * n + tile.x;
}

fn svt_load(entry: u32) -> u32 {
    return textureLoad(svt_page_table, vec2<u32>(entry % svt.table_width, entry / svt.table_width), 0).r;
}

fn svt_limit(g: vec2<f32>, limit: f32) -> vec2<f32> {
    return g * min(1.0, limit / max(length(g), 1e-8));
}

// Samples `level` from the tile in atlas slot `slot`. `gx`, `gy`: UV derivatives in level-0
// texels. The hardware filters anisotropically inside the tile; the footprint is limited to
// what the border covers.
fn svt_sample_tile(slot: u32, info: SvtLayer, level: u32, uv: vec2<f32>, gx: vec2<f32>, gy: vec2<f32>) -> LayerTexels {
    let n = svt_tiles(info, level);
    let p = uv * f32(n);
    let tile = min(floor(p), vec2<f32>(f32(n - 1u)));
    let span = f32(svt.tile_texels + 2u * svt.border);
    let origin = vec2<f32>(f32(slot % svt.tiles_per_row), f32(slot / svt.tiles_per_row)) * span + f32(svt.border);
    let atlas = (origin + (p - tile) * f32(svt.tile_texels)) / svt.atlas_size;
    let scale = exp2(-f32(level)) / svt.atlas_size;
    let limit = f32(2u * (svt.border - 1u)) / svt.atlas_size;
    let dx = svt_limit(gx * scale, limit);
    let dy = svt_limit(gy * scale, limit);
    var t: LayerTexels;
    t.albedo = textureSampleGrad(svt_albedo, material_sampler, atlas, dx, dy);
    let normal = textureSampleGrad(svt_normal, material_sampler, atlas, dx, dy).xy;
    let material = textureSampleGrad(svt_material, material_sampler, atlas, dx, dy).xy;
    t.data = vec4<f32>(normal, material);
    return t;
}

fn svt_record(entry: u32) {
    // One request per pixel: the first streamed layer sampled on even frames, the last on odd
    // ones (e.g. base colour and emission layers).
    if svt_request == 0u || (svt.frame & 1u) == 1u {
        svt_request = entry + 1u;
    }
}

// Texels of a layer at `uv` with UV derivatives `ddx`, `ddy`: streamed tiles where finer
// detail than the array layer is wanted and resident, the array layer otherwise.
fn sample_layer(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> LayerTexels {
    let info = svt_layer(layer);
    if info.levels == 0u {
        return sample_layer_array(layer, uv, ddx, ddy);
    }
    // Level of detail in level-0 texels, resolving up to max_aniso with hardware taps.
    let size = f32(svt.base_size << info.levels);
    let gx = ddx * size;
    let gy = ddy * size;
    let major = max(length(gx), length(gy));
    let minor = min(length(gx), length(gy));
    let lod = log2(max(max(minor, major / svt.max_aniso), 1e-6));
    if lod >= f32(info.levels) {
        return sample_layer_array(layer, uv, ddx, ddy);
    }
    let wrapped = fract(uv);
    let wanted = u32(max(lod, 0.0));
    let wanted_entry = svt_entry(info, wanted, wrapped);
    svt_record(wanted_entry);
    let entry = svt_load(wanted_entry);
    if entry == SVT_INVALID {
        return sample_layer_array(layer, uv, ddx, ddy);
    }
    // Trilinear between levels `fine` and `fine + 1`, no finer than what is resident here.
    let level = max(lod, f32(entry >> 16u) / 256.0);
    let fine = u32(level);
    let t = level - f32(fine);
    if fine >= info.levels {
        return sample_layer_array(layer, uv, ddx, ddy);
    }
    var fine_entry = entry;
    if fine != wanted {
        fine_entry = svt_load(svt_entry(info, fine, wrapped));
    }
    var result = svt_sample_tile(fine_entry & 0xffffu, info, fine, wrapped, gx, gy);
    if t > 0.0 {
        var coarse: LayerTexels;
        if fine + 1u == info.levels {
            coarse = sample_layer_array(layer, uv, ddx, ddy);
        } else {
            let coarse_entry = svt_load(svt_entry(info, fine + 1u, wrapped));
            coarse = svt_sample_tile(coarse_entry & 0xffffu, info, fine + 1u, wrapped, gx, gy);
        }
        result.albedo = mix(result.albedo, coarse.albedo, t);
        result.data = mix(result.data, coarse.data, t);
    }
    return result;
}

// One pixel per feedback block (a different one each frame) stores its request. Without
// streamed layers nothing reads the feedback back.
fn svt_write_feedback(pixel: vec2<u32>, viewport: vec2<u32>) {
    if svt.enabled == 0u {
        return;
    }
    let block = pixel / SVT_FEEDBACK_BLOCK;
    let chosen = min(block * SVT_FEEDBACK_BLOCK + svt.feedback_offset, viewport - 1u);
    if all(pixel == chosen) {
        textureStore(svt_feedback, block, vec4<u32>(svt_request, 0u, 0u, 0u));
    }
}
