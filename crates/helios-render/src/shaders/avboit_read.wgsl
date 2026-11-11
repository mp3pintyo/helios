// Reading the AVBOIT transmittance grid (see avboit_common.wgsl, which must be included first).
//
// `avboit_front` (columns x rows x slices + 2): slice 0 holds the column's bounds (near, far view
// depth; far <= near: the column holds no translucency); slice k + 1 the transmittance (rgb) of
// everything in front of cell k - surfaces, particles and volumes; slice `slices` + 1 the
// column's total. `avboit_surface` (slices + 1 deep, no bounds): the same
// without the volumes, for the volume march (which attenuates by its own medium exactly).

@group(0) @binding(121) var avboit_front: texture_3d<f32>;
@group(0) @binding(122) var avboit_surface: texture_3d<f32>;

fn avboit_bounds(column: vec2<i32>) -> vec2<f32> {
    return textureLoad(avboit_front, vec3<i32>(column, 0), 0).xy;
}

// Where view depth `d` lies in a column bounded by `b`: x the cell (-1 in front of every cell,
// `slices` behind them), y the position within it (0..1).
fn avboit_cell(d: f32, b: vec2<f32>) -> vec2<f32> {
    let s = avboit_slice(d, b);
    if s < 0.0 {
        return vec2<f32>(-1.0, 0.0);
    }
    let k = min(floor(s), f32(avboit.grid.z));
    return vec2<f32>(k, saturate(s - k));
}

// Transmittance in front of view depth `d` in one column, excluding the fragment's own
// transmittance `own` (1: a depth that holds no fragment). Within a cell the medium counts as
// spread evenly: the cell's transmittance to the fragment's position, with its own share taken
// out - continuous in depth (no seam where a fragment changes cells) and, for fragments sharing a
// cell, an order by their position in it. UE's AVBOIT blends a cell's fragments as their weighted
// average instead; this keeps that for fragments at the same depth.
fn avboit_column(column: vec2<i32>, d: f32, own: vec3<f32>) -> vec3<f32> {
    let b = avboit_bounds(column);
    if b.y <= b.x {
        return vec3<f32>(1.0);
    }
    let c = avboit_cell(d, b);
    if c.x < 0.0 {
        return vec3<f32>(1.0);
    }
    let k = i32(c.x);
    let start = textureLoad(avboit_front, vec3<i32>(column, k + 1), 0).rgb;
    if k >= i32(avboit.grid.z) {
        return start;
    }
    let end = textureLoad(avboit_front, vec3<i32>(column, k + 2), 0).rgb;
    let cell = min(end / max(start, vec3<f32>(1e-4)), vec3<f32>(1.0));
    let others = pow(clamp(cell / max(own, vec3<f32>(1e-4)), vec3<f32>(1e-6), vec3<f32>(1.0)), vec3<f32>(c.y));
    return start * max(others, cell);
}

// As `avboit_column` at internal pixel position `position`, bilinear between the four nearest
// columns.
fn avboit_transmittance(position: vec2<f32>, d: f32, own: vec3<f32>) -> vec3<f32> {
    let c = position / f32(avboit.grid.w) - 0.5;
    let f = fract(c);
    let base = vec2<i32>(floor(c));
    let hi = vec2<i32>(avboit.grid.xy) - 1;
    let c00 = avboit_column(clamp(base, vec2<i32>(0), hi), d, own);
    let c10 = avboit_column(clamp(base + vec2<i32>(1, 0), vec2<i32>(0), hi), d, own);
    let c01 = avboit_column(clamp(base + vec2<i32>(0, 1), vec2<i32>(0), hi), d, own);
    let c11 = avboit_column(clamp(base + vec2<i32>(1, 1), vec2<i32>(0), hi), d, own);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// The blend weight (per channel) of a translucent fragment of transmittance `own` at view depth
// `d`: the transmittance of what lies in front of it. `split` > 0: the fragment belongs to the
// part behind the nearest refracting surface (at view depth `split`), which is composited before
// that surface is drawn and dimmed by the front part's own composite - so only what lies between
// the split and the fragment counts.
fn avboit_weight(position: vec2<f32>, d: f32, own: vec3<f32>, split: f32) -> vec3<f32> {
    var w = avboit_transmittance(position, d, own);
    if split > 0.0 {
        w = min(w / max(avboit_transmittance(position, split, vec3<f32>(1.0)), vec3<f32>(1e-3)), vec3<f32>(1.0));
    }
    return w;
}

// Transmittance of the surfaces and particles (not volumes) in front of view depth `d` in one
// column bounded by `b` (the volume march: one column per pixel, chosen stochastically), spread
// evenly within the cells.
fn avboit_surface_at(column: vec2<i32>, b: vec2<f32>, d: f32) -> vec3<f32> {
    if b.y <= b.x {
        return vec3<f32>(1.0);
    }
    let c = avboit_cell(d, b);
    if c.x < 0.0 {
        return vec3<f32>(1.0);
    }
    let k = i32(c.x);
    let start = textureLoad(avboit_surface, vec3<i32>(column, k), 0).rgb;
    if k >= i32(avboit.grid.z) {
        return start;
    }
    let end = textureLoad(avboit_surface, vec3<i32>(column, k + 1), 0).rgb;
    return start * pow(clamp(end / max(start, vec3<f32>(1e-3)), vec3<f32>(1e-6), vec3<f32>(1.0)), vec3<f32>(c.y));
}
