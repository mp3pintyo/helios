// Lumen final gather: screen probes and reflections.
//
//   place_probes : one probe per 16x16 tile, at a jittered pixel of the tile
//   request_adaptive : per 8x8 cell, up to two requests for extra probes on surfaces that none
//                  of the uniform probes around them lies on (UE5's adaptive probe placement)
//   place_adaptive : grants the requests within the capacity and places those probes
//   finish_adaptive : indirect dispatch size for the probes placed
//   trace_probes : 8x8 rays per probe over the hemisphere (hemi-octahedral layout). Each ray
//                  is first traced in screen space against the depth buffer (hits read last
//                  frame's lit image), then through the mesh SDFs of the near field and the
//                  global distance field beyond (hits read the mesh-card surface cache, the
//                  voxel surface cache where no card matches); misses see the sky.
//   probe_sh     : filters each probe's radiance with its neighbours (depth/normal aware) and
//                  projects it to order-2 SH (plus the fraction of sky rays)
//   integrate    : per half-resolution pixel, interpolates the 4 surrounding uniform probes'
//                  SH and the adaptive probes of the 4 surrounding cells, weighted by plane
//                  distance and normal agreement -> diffuse irradiance
//
// Probe indices: the uniform grid (gx x gy) first, then the adaptive probes, so probe i's
// atlas tile is (i % gx, i / gx) for both - adaptive probes fill extra tile rows below the grid.
//   reflect      : half-resolution GGX reflection rays (screen, then distance field)
//   reflect_temporal : reprojected accumulation with neighbourhood clamping

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "lumen_cards.wgsl"
#include "lumen_trace.wgsl"

const PROBE_TILE: u32 = 16u;
const PROBE_TEXELS: u32 = 8u;
const PROBE_STRIDE: u32 = 11u; // vec4s per probe: position, normal, 9 SH coefficients
/// Interpolation weight of the radiance cache next to the screen probes (a matching probe weighs ~1).
const CACHE_FALLBACK_WEIGHT: f32 = 0.03;
/// Adaptive probes are placed per cell of this many pixels (ADAPTIVE_CELL in lumen.rs).
const ADAPTIVE_CELL: u32 = 8u;
/// At most this many adaptive probes per cell (a cell at a depth edge may hold two surfaces
/// that the uniform probes miss, e.g. a window mullion and the pier behind it).
const ADAPTIVE_PER_CELL: u32 = 2u;
/// Words before the cell table in `adaptive_alloc`: indirect dispatch size (x, y, z), probes
/// placed, probes requested (ADAPTIVE_HEADER_WORDS in lumen.rs).
const ADAPTIVE_HEADER: u32 = 8u;
/// A pixel whose uniform probes' interpolation weight on its own surface sums to less than this
/// asks for an adaptive probe: the radiance cache (CACHE_FALLBACK_WEIGHT) would otherwise carry
/// a quarter or more of it, and which probe matches it changes with the per-frame jitter.
const ADAPTIVE_MATCH_THRESHOLD: f32 = 0.1;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lumen: Lumen;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(5) var previous_radiance: texture_2d<f32>;
@group(0) @binding(6) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(7) var global_sdf: texture_3d<f32>;
@group(0) @binding(8) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(9) var lumen_sampler: sampler;
@group(0) @binding(10) var<storage, read_write> probes: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(12) var probe_atlas_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var probe_atlas: texture_2d<f32>;
@group(0) @binding(14) var gi_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var reflection_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(16) var reflection_raw: texture_2d<f32>;
@group(0) @binding(17) var reflection_history: texture_2d<f32>;
@group(0) @binding(18) var<storage, read> exposure: array<f32, 4>;
@group(0) @binding(19) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(20) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(70) var<storage, read> card_data: array<vec4<u32>>;
@group(0) @binding(71) var<storage, read> card_pages: array<u32>;
@group(0) @binding(72) var card_capture: texture_2d<u32>;
@group(0) @binding(73) var card_lighting: texture_2d<f32>;
@group(0) @binding(74) var sdf_instance_ids: texture_3d<u32>;
// Adaptive probes: header (indirect dispatch of the probe passes, allocated count), then per
// 8x8 cell `first adaptive probe << 2 | count` (0: none). Written through `adaptive_alloc`,
// read through `adaptive_cells` (the same buffer).
@group(0) @binding(85) var<storage, read_write> adaptive_alloc: array<atomic<u32>>;
@group(0) @binding(86) var<storage, read> adaptive_cells: array<u32>;

// Maximum exposed luminance of a single ray (exposure maps mid grey to 0.18), as Lumen's
// MaxRayIntensity. Tiny, very bright emitters - candle flames, lamp bulbs - are hit by a
// different handful of the 64 rays of a probe every frame; unclamped, each hit tints the whole
// probe and shows up as a blotch that jumps from tile to tile. The lights that represent those
// emitters already light the scene directly, so little real energy is lost.
const MAX_PROBE_RAY_INTENSITY: f32 = 1.5;
const MAX_REFLECTION_RAY_INTENSITY: f32 = 8.0;

fn clamp_ray(color: vec3<f32>, max_exposed: f32) -> vec3<f32> {
    let limit = max_exposed / max(exposure[0], 1e-8);
    let l = luminance(color);
    return color * min(1.0, limit / max(l, 1e-8));
}

fn probe_grid() -> vec2<u32> {
    return (vec2<u32>(frame.viewport.xy) + PROBE_TILE - 1u) / PROBE_TILE;
}

fn linear_depth_at(pixel: vec2<u32>) -> f32 {
    return frame.camera_position.w / max(textureLoad(depth_texture, pixel, 0).r, 1e-9);
}

// ---- Shared tracing --------------------------------------------------------------------------------

struct ScreenHit {
    found: bool,
    uv: vec2<f32>,
    t: f32,
}

fn trace_screen(origin: vec3<f32>, dir: vec3<f32>, max_distance: f32, steps: u32, jitter: f32) -> ScreenHit {
    var hit: ScreenHit;
    hit.found = false;
    hit.t = 0.0;
    let size = frame.viewport.xy;
    // Skip the first few pixels: they mostly see the starting surface itself, and reading its
    // previous radiance there would feed the surface back into itself.
    let origin_clip = frame.view_proj * vec4<f32>(origin, 1.0);
    let t0 = max(0.05, origin_clip.w * frame.viewport.w * 6.0);
    for (var i = 0u; i < steps; i++) {
        let s = (f32(i) + jitter) / f32(steps);
        let t = t0 + max_distance * s * s;
        let p = origin + dir * t;
        let clip = frame.view_proj * vec4<f32>(p, 1.0);
        if clip.w <= frame.camera_position.w {
            return hit;
        }
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            return hit;
        }
        let scene = linear_depth_at(vec2<u32>(uv * size));
        let thickness = max(0.2, t * 0.25);
        if clip.w > scene && clip.w - scene < thickness {
            hit.found = true;
            hit.uv = uv;
            hit.t = t;
            return hit;
        }
    }
    return hit;
}

// Screen trace, then the world (lumen_trace.wgsl): near field, global distance field, sky.
fn trace_radiance(origin: vec3<f32>, dir: vec3<f32>, screen_distance: f32, jitter: f32) -> Radiance {
    let s = trace_screen(origin, dir, screen_distance, 12u, jitter);
    if s.found {
        var r: Radiance;
        let motion = textureLoad(gbuffer_motion, vec2<u32>(s.uv * frame.viewport.xy), 0).xy;
        r.color = textureSampleLevel(previous_radiance, lumen_sampler, s.uv - motion, 0.0).rgb;
        r.sky = 0.0;
        r.t = s.t;
        return r;
    }
    // Without a near field, start beyond the origin surface's own distance band (Lumen
    // continues the ray where the screen trace left off).
    let level = min(sdf_finest_level(origin), lumen.level_count - 1u);
    return trace_world_radiance(origin, dir, max(screen_distance * 0.35, lumen.sdf_levels[level].w * 2.5));
}

// ---- Screen probes ---------------------------------------------------------------------------------

fn adaptive_cell_grid() -> vec2<u32> {
    return (vec2<u32>(frame.viewport.xy) + ADAPTIVE_CELL - 1u) / ADAPTIVE_CELL;
}

// Adaptive probes fill at most half as many atlas tile rows as the uniform grid has (UE5 also
// caps them at half the uniform probe count); `adaptive_rows` in lumen.rs sizes the buffers.
fn adaptive_capacity(grid: vec2<u32>) -> u32 {
    return ((grid.y + 1u) / 2u) * grid.x;
}

// The atlas tile of probe `index` (uniform and adaptive probes alike).
fn probe_tile(index: u32) -> vec2<u32> {
    let grid = probe_grid();
    return vec2<u32>(index % grid.x, index / grid.x);
}

// The pixel a probe was placed at.
fn probe_pixel(position: vec3<f32>) -> vec2<u32> {
    let clip = frame.view_proj * vec4<f32>(position, 1.0);
    let uv = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
    return min(vec2<u32>(max(uv * frame.viewport.xy, vec2<f32>(0.0))), vec2<u32>(frame.viewport.xy) - 1u);
}

// How well a probe at `probe_position` facing `probe_normal` stands in for the surface at
// `world` (normal `n`, `linear_depth` from the camera): x = closeness to the surface's plane
// (relative to the depth), y = normal agreement.
fn probe_surface_match(probe_position: vec3<f32>, probe_normal: vec3<f32>, world: vec3<f32>, n: vec3<f32>, linear_depth: f32) -> vec2<f32> {
    let plane = abs(dot(probe_position - world, n)) / max(linear_depth * 0.02, 0.02);
    return vec2<f32>(exp(-plane * plane), pow(saturate(dot(probe_normal, n)), 2.0));
}

// One of the four corners (i = 0..3) of a regular grid (`spacing` pixels, `dims` cells)
// around `pixel`, with its bilinear weight.
struct GridCorner {
    index: u32,
    weight: f32,
}

fn grid_corner(pixel: vec2<u32>, spacing: u32, dims: vec2<u32>, i: u32) -> GridCorner {
    let g = (vec2<f32>(pixel) + 0.5) / f32(spacing) - 0.5;
    let f = fract(g);
    let o = vec2<u32>(i & 1u, i >> 1u);
    let c = clamp(vec2<i32>(floor(g)) + vec2<i32>(o), vec2<i32>(0), vec2<i32>(dims) - 1);
    var corner: GridCorner;
    corner.index = u32(c.y) * dims.x + u32(c.x);
    corner.weight = select(1.0 - f.x, f.x, o.x == 1u) * select(1.0 - f.y, f.y, o.y == 1u);
    return corner;
}

// The interpolation weight the four uniform probes around `pixel` give the surface there (the
// part of `integrate`'s weight that lies on that surface).
fn uniform_probe_match(pixel: vec2<u32>, world: vec3<f32>, n: vec3<f32>, linear_depth: f32) -> f32 {
    let grid = probe_grid();
    var sum = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let corner = grid_corner(pixel, PROBE_TILE, grid, i);
        let info = probes[corner.index * PROBE_STRIDE];
        if info.w == 0.0 {
            continue;
        }
        let m = probe_surface_match(info.xyz, probes[corner.index * PROBE_STRIDE + 1u].xyz, world, n, linear_depth);
        sum += (corner.weight + 1e-3) * m.x * m.y;
    }
    return sum;
}

@compute @workgroup_size(8, 8)
fn place_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if all(gid.xy == vec2<u32>(0u)) {
        atomicStore(&adaptive_alloc[3], 0u);
        atomicStore(&adaptive_alloc[4], 0u);
    }
    let grid = probe_grid();
    if any(gid.xy >= grid) {
        return;
    }
    let index = gid.y * grid.x + gid.x;
    let h = pcg_hash(index * 9781u + frame.frame_index * 6271u);
    let size = vec2<u32>(frame.viewport.xy);
    var pixel = min(gid.xy * PROBE_TILE + vec2<u32>(h & 15u, (h >> 4u) & 15u), size - 1u);
    var depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        // Try the tile centre before giving up (sky).
        pixel = min(gid.xy * PROBE_TILE + PROBE_TILE / 2u, size - 1u);
        depth = textureLoad(depth_texture, pixel, 0).r;
    }
    let base = index * PROBE_STRIDE;
    if depth <= 0.0 {
        probes[base] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    probes[base] = vec4<f32>(world, 1.0);
    probes[base + 1u] = vec4<f32>(n, frame.camera_position.w / depth);
}

// The integration pixel of sample `s` (0..15) of adaptive cell `cell`.
fn adaptive_sample_pixel(cell: vec2<u32>, s: u32) -> vec2<u32> {
    let samples = ADAPTIVE_CELL / 2u;
    return cell * ADAPTIVE_CELL + vec2<u32>(s % samples, s / samples) * 2u;
}

// Adaptive probe requests, per 8x8 cell. On thin or depth-discontinuous geometry (window
// mullions, gaps between pews, chandelier rings, distant detail) often none of the four uniform
// probes around a pixel lies on its surface; the pixel then falls back to the radiance cache,
// or takes whichever probe the jitter happened to put on its surface this frame - flicker in
// 8-16 px blocks. This tests the pixels `integrate` shades (every second one) against the
// uniform probes and asks for a probe on the first one they miss (in a per-frame rotating
// order), then for a second one on the first pixel that neither the uniform probes nor the
// first request match. The cell records its requests (`count | sample << 2 | sample << 6`) and
// adds them to the demand.
@compute @workgroup_size(8, 8)
fn request_adaptive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cells = adaptive_cell_grid();
    if any(gid.xy >= cells) {
        return;
    }
    let cell = gid.y * cells.x + gid.x;
    let size = vec2<u32>(frame.viewport.xy);
    let samples = (ADAPTIVE_CELL / 2u) * (ADAPTIVE_CELL / 2u);
    let start = pcg_hash(cell * 7919u + frame.frame_index * 104729u) % samples;
    var chosen = 0u;
    var request = 0u;
    var positions: array<vec4<f32>, ADAPTIVE_PER_CELL>; // xyz world, w linear depth
    var normals: array<vec3<f32>, ADAPTIVE_PER_CELL>;
    for (var k = 0u; k < samples; k++) {
        let s = (k + start) % samples;
        let pixel = adaptive_sample_pixel(gid.xy, s);
        if any(pixel >= size) {
            continue;
        }
        let depth = textureLoad(depth_texture, pixel, 0).r;
        if depth <= 0.0 {
            continue;
        }
        let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
        let world = world_from_depth(uv, depth, frame.inv_view_proj);
        let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
        let linear_depth = frame.camera_position.w / depth;
        var matched = uniform_probe_match(pixel, world, n, linear_depth);
        for (var j = 0u; j < chosen; j++) {
            let m = probe_surface_match(positions[j].xyz, normals[j], world, n, linear_depth);
            matched += m.x * m.y;
        }
        if matched >= ADAPTIVE_MATCH_THRESHOLD {
            continue;
        }
        positions[chosen] = vec4<f32>(world, linear_depth);
        normals[chosen] = n;
        request |= s << (2u + 4u * chosen);
        chosen++;
        if chosen == ADAPTIVE_PER_CELL {
            break;
        }
    }
    if chosen > 0u {
        atomicAdd(&adaptive_alloc[4], chosen);
    }
    atomicStore(&adaptive_alloc[ADAPTIVE_HEADER + cell], request | chosen);
}

// Grants the requests and places the probes. When the requests exceed the capacity (dense
// foliage, distant detail), every cell keeps its requests with the same probability, drawn
// anew each frame, so the probes thin out evenly over the screen (and the temporal filter
// averages them) rather than going to whichever cells happened to run first.
@compute @workgroup_size(8, 8)
fn place_adaptive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cells = adaptive_cell_grid();
    if any(gid.xy >= cells) {
        return;
    }
    let cell = gid.y * cells.x + gid.x;
    let request = atomicLoad(&adaptive_alloc[ADAPTIVE_HEADER + cell]);
    let grid = probe_grid();
    let capacity = adaptive_capacity(grid);
    let demand = atomicLoad(&adaptive_alloc[4]);
    let keep = f32(pcg_hash(cell * 26699u + frame.frame_index * 7907u) >> 8u) * (1.0 / 16777216.0);
    var count = request & 3u;
    if count == 0u || (demand > capacity && keep * f32(demand) >= f32(capacity)) {
        atomicStore(&adaptive_alloc[ADAPTIVE_HEADER + cell], 0u);
        return;
    }
    let first = atomicAdd(&adaptive_alloc[3], count);
    count = select(0u, min(count, capacity - first), first < capacity);
    for (var j = 0u; j < count; j++) {
        let pixel = adaptive_sample_pixel(gid.xy, (request >> (2u + 4u * j)) & 15u);
        let depth = textureLoad(depth_texture, pixel, 0).r;
        let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
        let base = (grid.x * grid.y + first + j) * PROBE_STRIDE;
        probes[base] = vec4<f32>(world_from_depth(uv, depth, frame.inv_view_proj), 1.0);
        probes[base + 1u] = vec4<f32>(normalize(textureLoad(gbuffer_normal, pixel, 0).xyz), frame.camera_position.w / depth);
    }
    atomicStore(&adaptive_alloc[ADAPTIVE_HEADER + cell], select(0u, (first << 2u) | count, count > 0u));
}

// Sizes the indirect dispatch of the probe passes: the uniform grid plus the tile rows the
// adaptive probes fill.
@compute @workgroup_size(64)
fn finish_adaptive(@builtin(local_invocation_index) li: u32) {
    let grid = probe_grid();
    let count = min(atomicLoad(&adaptive_alloc[3]), adaptive_capacity(grid));
    let rows = (count + grid.x - 1u) / grid.x;
    if li == 0u {
        atomicStore(&adaptive_alloc[0], grid.x);
        atomicStore(&adaptive_alloc[1], grid.y + rows);
        atomicStore(&adaptive_alloc[2], 1u);
    }
    // The rest of the last row holds stale probes of earlier frames; the probe passes skip
    // invalid ones.
    let first = grid.x * grid.y;
    for (var i = count + li; i < rows * grid.x; i += 64u) {
        probes[(first + i) * PROBE_STRIDE] = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(8, 8)
fn trace_probes(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let grid = probe_grid();
    let index = wid.y * grid.x + wid.x;
    let texel = wid.xy * PROBE_TEXELS + lid.xy;
    let base = index * PROBE_STRIDE;
    let info = probes[base];
    if info.w == 0.0 {
        textureStore(probe_atlas_out, texel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let normal_depth = probes[base + 1u];
    let n = normal_depth.xyz;
    let jitter = vec2<f32>(
        interleaved_gradient_noise(vec2<f32>(texel), frame.frame_index),
        interleaved_gradient_noise(vec2<f32>(texel) + vec2<f32>(37.0, 17.0), frame.frame_index),
    );
    let local = hemi_oct_decode((vec2<f32>(lid.xy) + jitter) / f32(PROBE_TEXELS));
    let dir = basis(n) * local;
    let origin = info.xyz + n * (0.03 + 0.002 * normal_depth.w);
    let r = trace_radiance(origin, dir, 2.0 + normal_depth.w * 0.05, jitter.x);
    let color = select(r.color, vec3<f32>(0.0), any(r.color != r.color));
    textureStore(probe_atlas_out, texel, vec4<f32>(clamp_ray(color, MAX_PROBE_RAY_INTENSITY), r.sky));
}

// Summed one coefficient at a time (the same additions as a 64 × 10 array): the browser
// zero-initialises workgroup memory, and Firefox's shader compiler took 1 s over the
// initialisation of the 640-element array.
var<workgroup> sh_partial: array<vec4<f32>, 64>;

// Adds what probe `other` traced in direction `dir` to the spatial filter of the probe at
// `position` (`normal_depth`), weighted by how well it lies on the same surface (`own`: the
// probe itself, weight 1).
fn filter_tap(other: u32, own: bool, position: vec3<f32>, normal_depth: vec4<f32>, dir: vec3<f32>, sum: ptr<function, vec4<f32>>, weight: ptr<function, f32>) {
    let o_info = probes[other * PROBE_STRIDE];
    if o_info.w == 0.0 {
        return;
    }
    let o_normal = probes[other * PROBE_STRIDE + 1u];
    let plane = abs(dot(o_info.xyz - position, normal_depth.xyz)) / max(normal_depth.w, 0.1);
    var w = exp(-plane * 20.0) * pow(saturate(dot(o_normal.xyz, normal_depth.xyz)), 4.0);
    // The direction must lie in the neighbour's hemisphere too.
    let d_local = transpose(basis(o_normal.xyz)) * dir;
    if d_local.z <= 0.02 {
        return;
    }
    if own {
        w = 1.0;
    }
    let t = vec2<u32>(hemi_oct_encode(normalize(d_local)) * f32(PROBE_TEXELS));
    let texel = probe_tile(other) * PROBE_TEXELS + min(t, vec2<u32>(PROBE_TEXELS - 1u));
    *sum += textureLoad(probe_atlas, texel, 0) * w;
    *weight += w;
}

@compute @workgroup_size(8, 8)
fn probe_sh(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let grid = probe_grid();
    let index = wid.y * grid.x + wid.x;
    let base = index * PROBE_STRIDE;
    let info = probes[base];
    let normal_depth = probes[base + 1u];

    // Spatial filter: the same direction texel of neighbouring probes on a similar surface.
    let uv = (vec2<f32>(lid.xy) + 0.5) / f32(PROBE_TEXELS);
    let local = hemi_oct_decode(uv);
    let dir = basis(normal_depth.xyz) * local;
    var radiance = vec4<f32>(0.0);
    var weight = 0.0;
    let uniform_count = grid.x * grid.y;
    if info.w != 0.0 && index < uniform_count {
        // Uniform probe: the 3x3 neighbouring tiles.
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let g = vec2<i32>(wid.xy) + vec2<i32>(x, y);
                if any(g < vec2<i32>(0)) || any(g >= vec2<i32>(grid)) {
                    continue;
                }
                filter_tap(u32(g.y) * grid.x + u32(g.x), x == 0 && y == 0, info.xyz, normal_depth, dir, &radiance, &weight);
            }
        }
    } else if info.w != 0.0 {
        // Adaptive probe: the uniform probes around its pixel (which rarely lie on its surface)
        // and the adaptive probes of the 3x3 cells around it (which often do: a mullion or a
        // pew edge runs through several cells).
        filter_tap(index, true, info.xyz, normal_depth, dir, &radiance, &weight);
        let pixel = probe_pixel(info.xyz);
        for (var i = 0u; i < 4u; i++) {
            filter_tap(grid_corner(pixel, PROBE_TILE, grid, i).index, false, info.xyz, normal_depth, dir, &radiance, &weight);
        }
        let cells = adaptive_cell_grid();
        let cell = vec2<i32>(pixel / ADAPTIVE_CELL);
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let c = cell + vec2<i32>(x, y);
                if any(c < vec2<i32>(0)) || any(c >= vec2<i32>(cells)) {
                    continue;
                }
                let entry = adaptive_cells[ADAPTIVE_HEADER + u32(c.y) * cells.x + u32(c.x)];
                for (var j = 0u; j < (entry & 3u); j++) {
                    let other = uniform_count + (entry >> 2u) + j;
                    if other != index {
                        filter_tap(other, false, info.xyz, normal_depth, dir, &radiance, &weight);
                    }
                }
            }
        }
    }
    radiance /= max(weight, 1e-4);

    // Project onto SH (hemisphere texels of equal solid angle, approximately 2pi / 64 each).
    // Thread k keeps the sum of coefficient k (9: the sky visibility).
    let y = sh_basis(dir);
    let solid_angle = 2.0 * PI / 64.0;
    var total = vec4<f32>(0.0);
    for (var k = 0u; k < 10u; k++) {
        sh_partial[li] = select(vec4<f32>(radiance.a / 64.0), vec4<f32>(radiance.rgb * y[min(k, 8u)] * solid_angle, 0.0), k < 9u);
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride >>= 1u) {
            if li < stride {
                sh_partial[li] += sh_partial[li + stride];
            }
            workgroupBarrier();
        }
        if li == k {
            total = sh_partial[0];
        }
        workgroupBarrier();
    }
    if li < 9u {
        probes[base + 2u + li] = select(total, vec4<f32>(0.0), info.w == 0.0);
    }
    if li == 9u {
        probes[base + 1u] = vec4<f32>(normal_depth.xyz, normal_depth.w);
        // Sky visibility rides in the position's w channel (1 + visibility, 0 = invalid).
        probes[base] = vec4<f32>(info.xyz, select(0.0, 1.0 + total.x, info.w != 0.0));
    }
}

// Adds probe `probe`'s irradiance at the surface (`world`, `n`) to the interpolation, weighted
// by `bilinear` and how well the probe lies on that surface.
fn integrate_probe(probe: u32, bilinear: f32, world: vec3<f32>, n: vec3<f32>, linear_depth: f32, irradiance: ptr<function, vec3<f32>>, sky: ptr<function, f32>, weight: ptr<function, f32>) {
    let info = probes[probe * PROBE_STRIDE];
    if info.w < 1.0 {
        return;
    }
    let m = probe_surface_match(info.xyz, probes[probe * PROBE_STRIDE + 1u].xyz, world, n, linear_depth);
    let w = (bilinear + 1e-3) * m.x * (0.1 + m.y);
    var sh: array<vec4<f32>, 9>;
    for (var k = 0u; k < 9u; k++) {
        sh[k] = probes[probe * PROBE_STRIDE + 2u + k];
    }
    *irradiance += sh_irradiance(sh, n) * w;
    *sky += (info.w - 1.0) * w;
    *weight += w;
}

@compute @workgroup_size(8, 8)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= half_size) {
        return;
    }
    let pixel = min(gid.xy * 2u, vec2<u32>(frame.viewport.xy) - 1u);
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        textureStore(gi_out, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let linear_depth = frame.camera_position.w / depth;

    var irradiance = vec3<f32>(0.0);
    var sky = 0.0;
    var weight = 0.0;
    // Sky visibility around the pixel regardless of surface (bilinear over the uniform probes):
    // the sky-visibility counterpart of the radiance-cache fallback below.
    var nearby_sky = vec2<f32>(0.0);
    let grid = probe_grid();
    for (var i = 0u; i < 4u; i++) {
        let corner = grid_corner(pixel, PROBE_TILE, grid, i);
        integrate_probe(corner.index, corner.weight, world, n, linear_depth, &irradiance, &sky, &weight);
        let w = probes[corner.index * PROBE_STRIDE].w;
        if w >= 1.0 {
            nearby_sky += vec2<f32>(w - 1.0, 1.0) * corner.weight;
        }
    }
    // The adaptive probes of the four surrounding cells, interpolated the same way on their
    // finer grid.
    let cells = adaptive_cell_grid();
    for (var i = 0u; i < 4u; i++) {
        let corner = grid_corner(pixel, ADAPTIVE_CELL, cells, i);
        let entry = adaptive_cells[ADAPTIVE_HEADER + corner.index];
        for (var j = 0u; j < (entry & 3u); j++) {
            integrate_probe(grid.x * grid.y + (entry >> 2u) + j, corner.weight, world, n, linear_depth, &irradiance, &sky, &weight);
        }
    }
    // Where no screen probe lies on this surface (adaptive probes exhausted, or a third surface
    // in a cell), normalising the poorly matching ones would let whichever probe is least bad
    // win, and with probes re-jittered every frame that flips frame to frame. The world-space
    // radiance cache is always mixed in with a small constant weight instead: negligible next
    // to a matching probe, dominant when none matches (Lumen falls back to its radiance cache
    // the same way). The sky visibility (which drives specular occlusion) falls back the same
    // way, to the visibility of the probes around the pixel.
    let fallback_sky = select(0.5, nearby_sky.x / max(nearby_sky.y, 1e-6), nearby_sky.y > 1e-4);
    let sky_fraction = (sky + fallback_sky * CACHE_FALLBACK_WEIGHT) / (weight + CACHE_FALLBACK_WEIGHT);
    let cache = radiance_cache_irradiance(world + n * 0.1, n, vec3<f32>(-1.0));
    if cache.x >= 0.0 {
        irradiance += cache * CACHE_FALLBACK_WEIGHT;
        weight += CACHE_FALLBACK_WEIGHT;
    }
    if weight <= 1e-5 {
        textureStore(gi_out, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    // Store mean incident radiance (E / pi) like the rest of the pipeline expects.
    let result = vec4<f32>(irradiance / weight / PI, saturate(sky_fraction));
    textureStore(gi_out, gid.xy, select(result, vec4<f32>(0.0, 0.0, 0.0, 1.0), any(result != result)));
}

// ---- Reflections ---------------------------------------------------------------------------------------

// Visible-normal GGX sampling (Heitz 2018), in the local frame where n = +Z.
fn sample_ggx_vndf(v: vec3<f32>, alpha: f32, u: vec2<f32>) -> vec3<f32> {
    let vh = normalize(vec3<f32>(alpha * v.x, alpha * v.y, v.z));
    let len2 = vh.x * vh.x + vh.y * vh.y;
    let t1 = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(len2), len2 > 0.0);
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

const MAX_REFLECTION_ROUGHNESS: f32 = 0.55;

@compute @workgroup_size(8, 8)
fn reflect_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= half_size) {
        return;
    }
    let pixel = min(gid.xy * 2u + vec2<u32>(frame.frame_index & 1u, (frame.frame_index >> 1u) & 1u), vec2<u32>(frame.viewport.xy) - 1u);
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let normal_rough = textureLoad(gbuffer_normal, pixel, 0);
    if depth <= 0.0 || normal_rough.w > MAX_REFLECTION_ROUGHNESS {
        textureStore(reflection_out, gid.xy, vec4<f32>(0.0));
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(normal_rough.xyz);
    let v = normalize(frame.camera_position.xyz - world);
    let tbn = basis(n);
    let alpha = max(normal_rough.w * normal_rough.w, 0.002);
    let u = vec2<f32>(
        interleaved_gradient_noise(vec2<f32>(gid.xy), frame.frame_index),
        interleaved_gradient_noise(vec2<f32>(gid.xy) + vec2<f32>(59.0, 23.0), frame.frame_index),
    );
    let h = tbn * sample_ggx_vndf(transpose(tbn) * v, alpha, u);
    var dir = reflect(-v, h);
    if dot(dir, n) <= 0.0 {
        dir = reflect(-v, n);
    }
    let linear_depth = frame.camera_position.w / depth;
    let origin = world + n * (0.02 + 0.002 * linear_depth);
    let r = trace_radiance(origin, dir, 8.0 + linear_depth * 0.2, u.x);
    let color = select(r.color, vec3<f32>(0.0), any(r.color != r.color));
    textureStore(reflection_out, gid.xy, vec4<f32>(clamp_ray(color, MAX_REFLECTION_RAY_INTENSITY), 1.0));
}

@compute @workgroup_size(8, 8)
fn reflect_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= half_size) {
        return;
    }
    // Spatial filter with Karis weights (1 / (1 + luma)): a single ray that happens to hit a
    // sunlit patch must not turn into a bright block. Moments of the same neighbourhood give
    // the variance-clipping box for the history.
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = clamp(vec2<i32>(gid.xy) + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(half_size) - 1);
            let value = textureLoad(reflection_raw, tap, 0);
            let w = 1.0 / (1.0 + luminance(value.rgb));
            sum += value * w;
            weight_sum += w;
            let compressed = value.rgb * w;
            m1 += compressed;
            m2 += compressed * compressed;
        }
    }
    let current = sum / max(weight_sum, 1e-6);
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let pixel = min(gid.xy * 2u, vec2<u32>(frame.viewport.xy) - 1u);
    let motion = textureLoad(gbuffer_motion, pixel, 0).xy;
    let uv = (vec2<f32>(pixel) + 1.0) * frame.viewport.zw;
    let previous_uv = uv - motion;
    var result = current;
    if has_flag(frame.flags, FLAG_HISTORY_VALID) && all(previous_uv > vec2<f32>(0.0)) && all(previous_uv < vec2<f32>(1.0)) {
        let history_uv = viewport_uv(previous_uv, frame.prev_viewport.xy * 0.5, textureDimensions(reflection_history));
        let history = textureSampleLevel(reflection_history, lumen_sampler, history_uv, 0.0);
        // Clip in the tone-compressed space the moments were taken in, then expand again.
        let h = history.rgb / (1.0 + luminance(history.rgb));
        let clipped = clamp(h, mean - 1.5 * sigma, mean + 1.5 * sigma);
        let restored = clipped / max(1.0 - luminance(clipped), 0.02);
        result = vec4<f32>(mix(restored, current.rgb, 0.1), mix(history.a, current.a, 0.1));
    }
    textureStore(reflection_out, gid.xy, select(result, current, any(result != result)));
}
