// Lumen Lite: an irradiance-field final gather (UE 5.8's r.Lumen.FinalGatherMethod 0, DDGI-like)
// instead of screen probes, for GPUs that cannot afford tracing every screen tile every frame.
//
// World-space probes on LITE_LEVELS camera-centred clipmap levels of GRID^3 (spacing doubling per
// level) hold an octahedral irradiance map (IRRADIANCE_TEXELS^2, rgb = cosine-weighted mean
// radiance, a = sky visibility) and an octahedral map of ray-distance moments (MOMENT_TEXELS^2,
// mean and mean square in probe spacings) for Chebyshev visibility, each with a one-texel border
// so the atlases can be sampled bilinearly. Levels scroll by whole probe spacings with toroidal
// addressing: a probe slot keeps its data while its world coordinate stays in the window, so a
// camera move only refreshes the slab of probes that entered.
//
//   schedule      : per probe slot - recognises probes that scrolled in (new), scores the others
//                   by age, whether they are on screen and their kind, and histograms the scores
//   select_probes : finds the score threshold that fills the per-frame budget
//   list_probes   : appends the probes above it (new ones always first) to the update list
//   trace_field   : per listed probe: classifies it with the global distance field (relocates it
//                   off nearby surfaces, disables it inside geometry), then traces RAYS rays
//                   through Lumen's near field, global field and surface caches (lumen_trace.wgsl)
//   update_field  : projects the rays into the probe's irradiance and moment texels, blended
//                   with hysteresis, into a scratch buffer (the atlases are read here)
//   store_field   : writes the texels and their octahedral borders into the atlases
//   integrate_field : per half-resolution pixel, interpolates the 8 surrounding probes of the
//                   finest level that holds the pixel (trilinear x backface x Chebyshev weights,
//                   a normal and view bias against self-occlusion), blending into the next level
//                   at the window edge and into the radiance cache beyond the last
//   debug_field   : probe visualisation (debug view 20)

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "lumen_cards.wgsl"
#include "lumen_trace.wgsl"

// Keep in sync with lumen_lite.rs.
const LITE_LEVELS: u32 = 4u;
const GRID: u32 = 16u;
const LEVEL_PROBES: u32 = 4096u; // GRID^3
const RAYS: u32 = 128u;
const IRRADIANCE_TEXELS: u32 = 6u;
const MOMENT_TEXELS: u32 = 14u;
// Interior texels of a probe in the scratch buffer: irradiance, then moments.
const SCRATCH_TEXELS: u32 = 232u; // 6^2 + 14^2
const BUCKETS: u32 = 64u;
// Schedule buffer: indirect dispatch (3 words), list length, threshold bucket, its quota, the
// quota counter, padding; the score histogram; the update list.
const SCHEDULE_HISTOGRAM: u32 = 8u;
const SCHEDULE_LIST: u32 = 72u;

// Probe kinds (low byte of ProbeState::status; the score bucket sits in the second byte, the
// FIRST_UPDATE flag above).
const PROBE_NEW: u32 = 0u;      // scrolled in, never traced: not interpolated
const PROBE_ACTIVE: u32 = 1u;
const PROBE_SLEEPING: u32 = 2u; // no surface in the cells around it: refreshed rarely
const PROBE_BURIED: u32 = 3u;   // inside geometry, could not be moved out: not interpolated
const FIRST_UPDATE: u32 = 0x10000u;

// Probes are kept at least this many spacings off surfaces (moved along the field's gradient,
// by at most MAX_RELOCATION), so their rays don't start in the surface's own distance band.
const CLEARANCE: f32 = 0.1;
const MAX_RELOCATION: f32 = 0.45;
// Ray distances are stored in probe spacings, clamped here: beyond the farthest shading point
// that interpolates a probe (a cell diagonal plus the bias), every direction is simply clear.
const MAX_DEPTH: f32 = 2.0;
// Moment texels average rays within a narrow cone (cos^50), as DDGI's distance sharpness.
const MOMENT_SHARPNESS: f32 = 50.0;
// Shading-point bias, in probe spacings: along the normal and towards the eye. Keeps a probe's
// Chebyshev test from seeing the receiving surface itself as an occluder.
const NORMAL_BIAS: f32 = 0.15;
const VIEW_BIAS: f32 = 0.3;
// A level hands over to the next coarser one over this many cells at its window's edge.
const BLEND_CELLS: f32 = 2.0;
// Maximum exposed luminance of a ray (MAX_PROBE_RAY_INTENSITY in lumen_gather.wgsl): probes are
// refreshed seldom, so a candle flame hit by one ray would otherwise tint a probe for seconds.
const MAX_RAY_INTENSITY: f32 = 1.5;

struct Lite {
    // Per level: xyz = world position of the window's first probe, w = probe spacing.
    levels: array<vec4<f32>, 4>,
    // Per level: xyz = the window's first probe coordinate (world probe units).
    cells: array<vec4<i32>, 4>,
    frame: u32,
    budget: u32,
    // Non-zero: forget every probe (scene change).
    reset: u32,
    // Weight of the previous value when a probe is refreshed.
    hysteresis: f32,
}

struct ProbeState {
    // World probe coordinate the slot holds.
    coord: vec3<i32>,
    // Frame of the last refresh.
    updated: u32,
    // Relocation off nearby surfaces (world units).
    offset: vec3<f32>,
    // Class | score bucket << 8 | FIRST_UPDATE.
    status: u32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lumen: Lumen;
@group(0) @binding(2) var<uniform> lite: Lite;
@group(0) @binding(3) var depth_texture: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(5) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(6) var global_sdf: texture_3d<f32>;
@group(0) @binding(7) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(8) var lumen_sampler: sampler;
@group(0) @binding(9) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> exposure: array<f32, 4>;
@group(0) @binding(11) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(12) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(13) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(20) var<storage, read_write> probe_state: array<ProbeState>;
// The schedule buffer, written through `schedule` and read through `probe_list`.
@group(0) @binding(21) var<storage, read_write> schedule_words: array<atomic<u32>>;
@group(0) @binding(22) var<storage, read> probe_list: array<u32>;
// Per listed probe and ray: rgb radiance and sky (2 x f16 pairs), distance in spacings (f32).
@group(0) @binding(23) var<storage, read_write> rays: array<vec4<u32>>;
@group(0) @binding(24) var<storage, read_write> scratch: array<vec4<f32>>;
@group(0) @binding(25) var irradiance_atlas: texture_2d<f32>;
@group(0) @binding(26) var moment_atlas: texture_2d<f32>;
@group(0) @binding(27) var irradiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(28) var moment_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(29) var gi_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(30) var gi_texture: texture_2d<f32>;
@group(0) @binding(31) var debug_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(70) var<storage, read> card_data: array<vec4<u32>>;
@group(0) @binding(71) var<storage, read> card_pages: array<u32>;
@group(0) @binding(72) var card_capture: texture_2d<u32>;
@group(0) @binding(73) var card_lighting: texture_2d<f32>;
@group(0) @binding(74) var sdf_instance_ids: texture_3d<u32>;

// ---- Addressing ------------------------------------------------------------------------------------

fn positive_mod(a: vec3<i32>, m: i32) -> vec3<i32> {
    return ((a % m) + m) % m;
}

// Slot of world probe coordinate `c` in level `level` (toroidal: the coordinate modulo GRID).
fn probe_slot(level: u32, c: vec3<i32>) -> u32 {
    let s = vec3<u32>(positive_mod(c, i32(GRID)));
    return level * LEVEL_PROBES + (s.z * GRID + s.y) * GRID + s.x;
}

fn slot_coords(index: u32) -> vec3<u32> {
    let local = index % LEVEL_PROBES;
    return vec3<u32>(local % GRID, (local / GRID) % GRID, local / (GRID * GRID));
}

// The world coordinate slot `index` holds this frame: the one in the level's window that is
// congruent to the slot modulo GRID.
fn slot_world_coord(index: u32) -> vec3<i32> {
    let level = index / LEVEL_PROBES;
    let first = lite.cells[level].xyz;
    return first + positive_mod(vec3<i32>(slot_coords(index)) - first, i32(GRID));
}

// Top-left texel of a probe's tile (`texels` interior texels plus the border) in an atlas whose
// columns are x + z * GRID and rows y + level * GRID.
fn probe_tile(index: u32, texels: u32) -> vec2<u32> {
    let s = slot_coords(index);
    let level = index / LEVEL_PROBES;
    return vec2<u32>(s.x + s.z * GRID, s.y + level * GRID) * (texels + 2u);
}

fn probe_position(index: u32, state: ProbeState) -> vec3<f32> {
    return vec3<f32>(state.coord) * lite.levels[index / LEVEL_PROBES].w + state.offset;
}

// ---- Octahedral mapping (full sphere) --------------------------------------------------------------

fn sign_not_zero(v: vec2<f32>) -> vec2<f32> {
    return select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
}

fn oct_encode(d: vec3<f32>) -> vec2<f32> {
    let p = d.xy / (abs(d.x) + abs(d.y) + abs(d.z));
    let q = select(p, (vec2<f32>(1.0) - abs(p.yx)) * sign_not_zero(p), d.z < 0.0);
    return q * 0.5 + 0.5;
}

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let e = uv * 2.0 - 1.0;
    var v = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        v = vec3<f32>((vec2<f32>(1.0) - abs(v.yx)) * sign_not_zero(v.xy), v.z);
    }
    return normalize(v);
}

// Interior texel a tile texel (border included, 0..n+1) shows: itself inside, across the
// octahedron's folded edge on the border (rows and columns mirrored, corners diagonally
// opposite), so bilinear filtering at the edge blends the right neighbours.
fn border_source(texel: vec2<u32>, n: u32) -> vec2<u32> {
    let t = vec2<i32>(texel) - 1;
    let last = i32(n) - 1;
    let low = t < vec2<i32>(0);
    let high = t > vec2<i32>(last);
    var s = t;
    if any(low | high) {
        if (low.x || high.x) && (low.y || high.y) {
            s = select(vec2<i32>(0), vec2<i32>(last), low);
        } else if low.y || high.y {
            s = vec2<i32>(last - t.x, select(last, 0, low.y));
        } else {
            s = vec2<i32>(select(last, 0, low.x), last - t.y);
        }
    }
    return vec2<u32>(s);
}

// Atlas uv of direction `d` in a probe's tile.
fn tile_uv(index: u32, texels: u32, d: vec3<f32>, atlas: vec2<u32>) -> vec2<f32> {
    let texel = vec2<f32>(probe_tile(index, texels)) + 1.0 + oct_encode(d) * f32(texels);
    return texel / vec2<f32>(atlas);
}

// ---- Scheduling --------------------------------------------------------------------------------------

fn kind_of(state: ProbeState) -> u32 {
    return state.status & 0xffu;
}

// Update priority as a histogram bucket: new probes on top (finer levels first), then the
// others by age, weighted up on screen and down for sleeping and buried probes, in quarter
// octaves. Bucket 0 (refreshed less than a weighted frame ago) is never scheduled.
fn priority_bucket(index: u32, state: ProbeState) -> u32 {
    let level = index / LEVEL_PROBES;
    let kind = kind_of(state);
    if kind == PROBE_NEW {
        return BUCKETS - 1u - level;
    }
    var weight = 1.0;
    if kind == PROBE_SLEEPING {
        weight = 1.0 / 8.0;
    } else if kind == PROBE_BURIED {
        weight = 1.0 / 64.0;
    }
    let spacing = lite.levels[level].w;
    let clip = frame.view_proj * vec4<f32>(probe_position(index, state), 1.0);
    let reach = clip.w * 0.25 + spacing * 2.0;
    if clip.w > -spacing && all(abs(clip.xy) < vec2<f32>(clip.w + reach)) {
        weight *= 4.0;
    }
    let score = f32(lite.frame - state.updated) * weight;
    if score < 1.0 {
        return 0u;
    }
    return min(u32(log2(score) * 4.0) + 1u, BUCKETS - 1u - LITE_LEVELS);
}

@compute @workgroup_size(64)
fn schedule(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= LITE_LEVELS * LEVEL_PROBES {
        return;
    }
    var state = probe_state[index];
    let c = slot_world_coord(index);
    if any(state.coord != c) || lite.reset != 0u {
        state.coord = c;
        state.offset = vec3<f32>(0.0);
        state.status = PROBE_NEW;
        state.updated = lite.frame;
    }
    let bucket = priority_bucket(index, state);
    state.status = (state.status & 0xff00ffu) | (bucket << 8u);
    probe_state[index] = state;
    atomicAdd(&schedule_words[SCHEDULE_HISTOGRAM + bucket], 1u);
}

var<workgroup> histogram: array<u32, 64>;

@compute @workgroup_size(64)
fn select_probes(@builtin(local_invocation_index) li: u32) {
    histogram[li] = atomicLoad(&schedule_words[SCHEDULE_HISTOGRAM + li]);
    // Cleared for the next frame's schedule.
    atomicStore(&schedule_words[SCHEDULE_HISTOGRAM + li], 0u);
    workgroupBarrier();
    if li == 0u {
        // The lowest bucket that still fits the budget, and how many of its probes do.
        var selected = 0u;
        var threshold = BUCKETS;
        var quota = 0u;
        for (var b = BUCKETS - 1u; b > 0u; b--) {
            let count = histogram[b];
            threshold = b;
            if selected + count >= lite.budget {
                quota = lite.budget - selected;
                selected = lite.budget;
                break;
            }
            selected += count;
            quota = count;
        }
        atomicStore(&schedule_words[0], selected);
        atomicStore(&schedule_words[1], 1u);
        atomicStore(&schedule_words[2], 1u);
        atomicStore(&schedule_words[3], 0u);
        atomicStore(&schedule_words[4], threshold);
        atomicStore(&schedule_words[5], quota);
        atomicStore(&schedule_words[6], 0u);
    }
}

@compute @workgroup_size(64)
fn list_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= LITE_LEVELS * LEVEL_PROBES {
        return;
    }
    let bucket = (probe_state[index].status >> 8u) & 0xffu;
    let threshold = atomicLoad(&schedule_words[4]);
    if bucket == 0u || bucket < threshold {
        return;
    }
    if bucket == threshold && atomicAdd(&schedule_words[6], 1u) >= atomicLoad(&schedule_words[5]) {
        return;
    }
    let slot = atomicAdd(&schedule_words[3], 1u);
    if slot < lite.budget {
        atomicStore(&schedule_words[SCHEDULE_LIST + slot], index);
        let state = probe_state[index];
        probe_state[index].updated = lite.frame;
        // A probe that holds no data of its own yet (new, or buried until now: its tile holds
        // another probe's) takes its first rays as they are.
        let kind = kind_of(state);
        probe_state[index].status = select(state.status, state.status | FIRST_UPDATE, kind == PROBE_NEW || kind == PROBE_BURIED);
    }
}

// ---- Tracing ------------------------------------------------------------------------------------------

// Classifies probe `index` with the global distance field: moves it off a surface closer than
// CLEARANCE (along the field's gradient, at most MAX_RELOCATION), buries it if that fails,
// and puts it to sleep when no surface lies within the cells around it. Writes the state and
// returns its position (w = kind).
fn classify_probe(index: u32) -> vec4<f32> {
    let state = probe_state[index];
    let level = index / LEVEL_PROBES;
    let spacing = lite.levels[level].w;
    let base = vec3<f32>(state.coord) * spacing;
    var kind = PROBE_ACTIVE;
    var offset = vec3<f32>(0.0);
    // The field level whose voxels are a quarter probe spacing, if it holds the probe: its band
    // (4 voxels) is one spacing.
    var sdf_level = min(level, lumen.level_count - 1u);
    if !sdf_inside_level(base, sdf_level, 2.0) {
        sdf_level = sdf_finest_level(base);
    }
    if lumen.sdf_enabled != 0u && sdf_level < lumen.level_count {
        let clearance = CLEARANCE * spacing;
        let d = sdf_distance(base, sdf_level);
        if d < clearance {
            var p = base;
            for (var i = 0u; i < 3u; i++) {
                let n = sdf_normal(p, sdf_level);
                p += n * (clearance * 1.5 - sdf_distance(p, sdf_level));
                let moved = p - base;
                p = base + moved * min(1.0, MAX_RELOCATION * spacing / max(length(moved), 1e-6));
            }
            if sdf_distance(p, sdf_level) < clearance * 0.5 {
                kind = PROBE_BURIED;
            } else {
                offset = p - base;
            }
        } else if sdf_level + 1u < lumen.level_count && sdf_inside_level(base, sdf_level + 1u, 2.0) {
            // A cell diagonal is 1.73 spacings: nothing shades from a probe with no surface
            // closer than that. Alpha-masked surfaces (foliage) are not in the global field, so
            // such probes are still refreshed, only less often.
            if sdf_distance(base, sdf_level + 1u) >= 1.8 * spacing {
                kind = PROBE_SLEEPING;
            }
        }
    }
    // Only these fields: the other threads read `updated` meanwhile.
    probe_state[index].offset = offset;
    probe_state[index].status = (state.status & 0xffff00u) | kind;
    return vec4<f32>(base + offset, f32(kind));
}

// Spherical Fibonacci direction `i` of `n`.
fn fibonacci_sphere(i: u32, n: u32) -> vec3<f32> {
    let z = 1.0 - (f32(i) + 0.5) / f32(n) * 2.0;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = f32(i) * 2.39996323;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// A random rotation per probe and frame (quaternion from three uniforms), so successive
// refreshes sample new directions.
fn ray_rotation(index: u32, refresh: u32) -> mat3x3<f32> {
    let h0 = pcg_hash((index * 7919u) ^ (refresh * 104729u));
    let h1 = pcg_hash(h0);
    let h2 = pcg_hash(h1);
    let u = vec3<f32>(vec3<u32>(h0, h1, h2) >> vec3<u32>(8u)) * (1.0 / 16777216.0);
    let a = sqrt(1.0 - u.x);
    let b = sqrt(u.x);
    let q = vec4<f32>(a * sin(2.0 * PI * u.y), a * cos(2.0 * PI * u.y), b * sin(2.0 * PI * u.z), b * cos(2.0 * PI * u.z));
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
    );
}

fn ray_direction(index: u32, ray: u32) -> vec3<f32> {
    return ray_rotation(index, probe_state[index].updated) * fibonacci_sphere(ray, RAYS);
}

var<workgroup> traced_probe: vec4<f32>;

// One workgroup per listed probe, RAYS / 64 rays per thread (indirect dispatch).
@compute @workgroup_size(64)
fn trace_field(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = probe_list[SCHEDULE_LIST + wid.x];
    if li == 0u {
        traced_probe = classify_probe(index);
    }
    let probe = workgroupUniformLoad(&traced_probe);
    if u32(probe.w) == PROBE_BURIED {
        return;
    }
    let spacing = lite.levels[index / LEVEL_PROBES].w;
    let rotation = ray_rotation(index, probe_state[index].updated);
    let limit = MAX_RAY_INTENSITY / max(exposure[0], 1e-8);
    for (var ray = li; ray < RAYS; ray += 64u) {
        let dir = rotation * fibonacci_sphere(ray, RAYS);
        let r = trace_world_radiance(probe.xyz, dir, 0.0);
        var color = select(r.color, vec3<f32>(0.0), any(r.color != r.color));
        color *= min(1.0, limit / max(luminance(color), 1e-8));
        let depth = min(r.t / spacing, MAX_DEPTH);
        rays[wid.x * RAYS + ray] = vec4<u32>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, r.sky)), bitcast<u32>(depth), 0u);
    }
}

// ---- Probe update --------------------------------------------------------------------------------------

var<workgroup> ray_directions: array<vec3<f32>, 128>;
var<workgroup> ray_radiance: array<vec4<f32>, 128>;
var<workgroup> ray_depth: array<f32, 128>;

@compute @workgroup_size(64)
fn update_field(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = probe_list[SCHEDULE_LIST + wid.x];
    let state = probe_state[index];
    for (var ray = li; ray < RAYS; ray += 64u) {
        let packed = rays[wid.x * RAYS + ray];
        ray_directions[ray] = ray_direction(index, ray);
        ray_radiance[ray] = vec4<f32>(unpack2x16float(packed.x), unpack2x16float(packed.y));
        ray_depth[ray] = bitcast<f32>(packed.z);
    }
    workgroupBarrier();
    if kind_of(state) == PROBE_BURIED {
        return;
    }
    // Newly entered probes take their rays as they are; a large change (lights switched,
    // sun moved) is followed faster than the usual hysteresis.
    let first = (state.status & FIRST_UPDATE) != 0u;
    let irradiance_tile = probe_tile(index, IRRADIANCE_TEXELS) + 1u;
    let moment_tile = probe_tile(index, MOMENT_TEXELS) + 1u;
    let base = wid.x * SCRATCH_TEXELS;
    for (var texel = li; texel < SCRATCH_TEXELS; texel += 64u) {
        if texel < IRRADIANCE_TEXELS * IRRADIANCE_TEXELS {
            let t = vec2<u32>(texel % IRRADIANCE_TEXELS, texel / IRRADIANCE_TEXELS);
            let n = oct_decode((vec2<f32>(t) + 0.5) / f32(IRRADIANCE_TEXELS));
            var sum = vec4<f32>(0.0);
            var weight = 0.0;
            for (var ray = 0u; ray < RAYS; ray++) {
                let w = max(dot(n, ray_directions[ray]), 0.0);
                sum += ray_radiance[ray] * w;
                weight += w;
            }
            let current = sum / max(weight, 1e-4);
            let previous = textureLoad(irradiance_atlas, irradiance_tile + t, 0);
            var hysteresis = select(lite.hysteresis, 0.0, first);
            let change = abs(luminance(current.rgb) - luminance(previous.rgb)) / max(luminance(previous.rgb), 1e-4);
            if change > 0.5 {
                hysteresis *= 0.5;
            }
            scratch[base + texel] = mix(current, previous, hysteresis);
        } else {
            let m = texel - IRRADIANCE_TEXELS * IRRADIANCE_TEXELS;
            let t = vec2<u32>(m % MOMENT_TEXELS, m / MOMENT_TEXELS);
            let d = oct_decode((vec2<f32>(t) + 0.5) / f32(MOMENT_TEXELS));
            var sum = vec2<f32>(0.0);
            var weight = 0.0;
            for (var ray = 0u; ray < RAYS; ray++) {
                let w = pow(max(dot(d, ray_directions[ray]), 0.0), MOMENT_SHARPNESS);
                let depth = ray_depth[ray];
                sum += vec2<f32>(depth, depth * depth) * w;
                weight += w;
            }
            let previous = textureLoad(moment_atlas, moment_tile + t, 0).xy;
            var current = previous;
            if weight > 1e-6 {
                current = mix(sum / weight, previous, select(lite.hysteresis, 0.0, first));
            }
            scratch[base + texel] = vec4<f32>(current, 0.0, 0.0);
        }
    }
}

@compute @workgroup_size(64)
fn store_field(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let index = probe_list[SCHEDULE_LIST + wid.x];
    let state = probe_state[index];
    if kind_of(state) == PROBE_BURIED {
        return;
    }
    if li == 0u {
        probe_state[index].status = state.status & ~FIRST_UPDATE;
    }
    let base = wid.x * SCRATCH_TEXELS;
    let irradiance_side = IRRADIANCE_TEXELS + 2u;
    let moment_side = MOMENT_TEXELS + 2u;
    let irradiance_tile = probe_tile(index, IRRADIANCE_TEXELS);
    let moment_tile = probe_tile(index, MOMENT_TEXELS);
    let irradiance_count = irradiance_side * irradiance_side;
    for (var texel = li; texel < irradiance_count + moment_side * moment_side; texel += 64u) {
        if texel < irradiance_count {
            let t = vec2<u32>(texel % irradiance_side, texel / irradiance_side);
            let s = border_source(t, IRRADIANCE_TEXELS);
            textureStore(irradiance_out, irradiance_tile + t, scratch[base + s.y * IRRADIANCE_TEXELS + s.x]);
        } else {
            let m = texel - irradiance_count;
            let t = vec2<u32>(m % moment_side, m / moment_side);
            let s = border_source(t, MOMENT_TEXELS);
            let first_moment = IRRADIANCE_TEXELS * IRRADIANCE_TEXELS;
            textureStore(moment_out, moment_tile + t, scratch[base + first_moment + s.y * MOMENT_TEXELS + s.x]);
        }
    }
}

// ---- Shading --------------------------------------------------------------------------------------------

// Interpolated irradiance field of one level at a surface (`world`, normal `n`, shading point
// `biased`, `g` = its window coordinates): rgb mean radiance, a sky visibility, plus the weight
// of the probes that contributed (0: none usable).
struct FieldSample {
    value: vec4<f32>,
    weight: f32,
}

fn sample_level(level: u32, world: vec3<f32>, n: vec3<f32>, biased: vec3<f32>, g: vec3<f32>) -> FieldSample {
    let spacing = lite.levels[level].w;
    let base = vec3<i32>(floor(g));
    let f = g - floor(g);
    let irradiance_size = textureDimensions(irradiance_atlas);
    let moment_size = textureDimensions(moment_atlas);
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let o = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let c = lite.cells[level].xyz + base + vec3<i32>(o);
        let index = probe_slot(level, c);
        let state = probe_state[index];
        let kind = kind_of(state);
        if kind == PROBE_NEW || kind == PROBE_BURIED {
            continue;
        }
        let position = probe_position(index, state);
        // Smooth backface term: probes behind the surface count little.
        let to_probe = normalize(position - world);
        let backface = (dot(to_probe, n) + 1.0) * 0.5;
        var w = backface * backface + 0.2;
        // Chebyshev visibility of the shading point from the probe.
        let to_point = biased - position;
        let r = length(to_point) / spacing;
        let moments = textureSampleLevel(moment_atlas, lumen_sampler, tile_uv(index, MOMENT_TEXELS, to_point / max(r * spacing, 1e-6), moment_size), 0.0).xy;
        if r > moments.x {
            let variance = abs(moments.y - moments.x * moments.x) + 1e-4;
            let excess = r - moments.x;
            let chebyshev = variance / (variance + excess * excess);
            w *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }
        w = max(w, 1e-6);
        // Crush tiny weights (DDGI): a mostly occluded probe should vanish, not tint.
        if w < 0.2 {
            w *= w * w * 25.0;
        }
        let trilinear = select(1.0 - f, f, o == vec3<u32>(1u));
        w *= max(trilinear.x * trilinear.y * trilinear.z, 1e-3);
        let irradiance = textureSampleLevel(irradiance_atlas, lumen_sampler, tile_uv(index, IRRADIANCE_TEXELS, n, irradiance_size), 0.0);
        sum += irradiance * w;
        weight_sum += w;
    }
    var s: FieldSample;
    s.value = sum / max(weight_sum, 1e-8);
    s.weight = weight_sum;
    return s;
}

// Window coordinates of `p` in a level (xyz; probes at integers) and how far inside the
// window it lies (w, in cells; <= 0: outside).
fn window_coords(level: u32, p: vec3<f32>) -> vec4<f32> {
    let l = lite.levels[level];
    let g = (p - l.xyz) / l.w;
    let inside = min(g, vec3<f32>(f32(GRID - 1u)) - g);
    return vec4<f32>(g, min(inside.x, min(inside.y, inside.z)));
}

// Irradiance field at a surface: the finest level that holds it, blended into the next coarser
// level near its window's edge and into the radiance cache beyond the last one.
fn field_irradiance(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> vec4<f32> {
    var result = vec4<f32>(0.0);
    var remaining = 1.0;
    for (var level = 0u; level < LITE_LEVELS && remaining > 1e-3; level++) {
        let biased = world + (n * NORMAL_BIAS + v * VIEW_BIAS) * lite.levels[level].w;
        let g = window_coords(level, biased);
        if g.w <= 0.0 {
            continue;
        }
        let s = sample_level(level, world, n, biased, g.xyz);
        if s.weight <= 1e-5 {
            continue;
        }
        let share = remaining * saturate(g.w / BLEND_CELLS);
        result += s.value * share;
        remaining -= share;
    }
    if remaining > 1e-3 {
        // Beyond the field: the radiance cache, whose sky visibility is unknown - estimated from
        // how much of the open sky's irradiance it receives.
        let sky = sh_irradiance(sky_sh, n) * frame.sun_illuminance.rgb * lumen.sky_intensity;
        let cache = radiance_cache_irradiance(world + n * 0.1, n, vec3<f32>(-1.0));
        var fallback = vec4<f32>(sky / PI, 1.0);
        if cache.x >= 0.0 {
            fallback = vec4<f32>(cache / PI, saturate(luminance(cache) / max(luminance(sky), 1e-6)));
        }
        result += fallback * remaining;
    }
    return result;
}

@compute @workgroup_size(8, 8)
fn integrate_field(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let v = normalize(frame.camera_position.xyz - world);
    let result = field_irradiance(world, n, v);
    textureStore(gi_out, gid.xy, select(result, vec4<f32>(0.0, 0.0, 0.0, 1.0), any(result != result)));
}

// ---- Debug view ------------------------------------------------------------------------------------------

// Sphere radius in probe spacings.
const DEBUG_SPHERE_RADIUS: f32 = 0.08;
// Probes closer to the eye than this many spacings are not drawn (they would fill the view).
const DEBUG_NEAR: f32 = 2.0;

struct DebugHit {
    t: f32,
    color: vec3<f32>,
}

// Ray against the sphere of probe slot `index`; keeps the nearest hit.
fn debug_probe(index: u32, origin: vec3<f32>, dir: vec3<f32>, hit: ptr<function, DebugHit>) {
    let state = probe_state[index];
    let spacing = lite.levels[index / LEVEL_PROBES].w;
    let centre = probe_position(index, state);
    let radius = DEBUG_SPHERE_RADIUS * spacing;
    let oc = origin - centre;
    let b = dot(oc, dir);
    let disc = b * b - (dot(oc, oc) - radius * radius);
    if disc < 0.0 {
        return;
    }
    let t = -b - sqrt(disc);
    if t <= DEBUG_NEAR * spacing || t >= (*hit).t {
        return;
    }
    let n = normalize(origin + dir * t - centre);
    let grey = 0.18 / max(exposure[0], 1e-8);
    var color = textureSampleLevel(irradiance_atlas, lumen_sampler, tile_uv(index, IRRADIANCE_TEXELS, n, textureDimensions(irradiance_atlas)), 0.0).rgb;
    switch kind_of(state) {
        case PROBE_NEW: {
            color = vec3<f32>(1.0, 0.0, 1.0) * grey;
        }
        case PROBE_BURIED: {
            color = vec3<f32>(1.5, 0.05, 0.05) * grey * (0.4 + 0.6 * saturate(n.y * 0.5 + 0.5));
        }
        case PROBE_SLEEPING: {
            color = mix(color, vec3<f32>(0.1, 0.2, 1.0) * grey, 0.5);
        }
        default: {}
    }
    (*hit).t = t;
    (*hit).color = color;
}

// Probes as spheres shaded with their irradiance (sleeping: blue tint, buried: red, not traced
// yet: magenta) over the irradiance the field gives each pixel. Each level's probes are found
// by stepping through the cells around them (3D DDA) where no finer level covers the ray.
@compute @workgroup_size(8, 8)
fn debug_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) * frame.viewport.zw;
    let eye = frame.camera_position.xyz;
    let dir = view_direction(uv, frame.inv_view_proj, eye);
    let depth = textureLoad(depth_texture, gid.xy, 0).r;
    var hit: DebugHit;
    hit.t = 1e6;
    hit.color = vec3<f32>(0.02, 0.025, 0.03) / max(exposure[0], 1e-8);
    if depth > 0.0 {
        hit.t = distance(world_from_depth(uv, depth, frame.inv_view_proj), eye);
        let gi = textureLoad(gi_texture, min(gid.xy / 2u, viewport_extent(frame.viewport, 2u) - 1u), 0).rgb;
        hit.color = gi * 0.6;
    }
    for (var level = 0u; level < LITE_LEVELS; level++) {
        let spacing = lite.levels[level].w;
        // Cells centred on the probes: [-0.5, GRID - 0.5) in window coordinates.
        let lo = lite.levels[level].xyz - spacing * 0.5;
        let hi = lo + spacing * f32(GRID);
        let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
        let t0 = (lo - eye) * inv;
        let t1 = (hi - eye) * inv;
        let t_enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
        let t_exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), hit.t));
        if t_enter >= t_exit {
            continue;
        }
        let start = (eye + dir * (t_enter + spacing * 1e-3) - lo) / spacing;
        var cell = clamp(vec3<i32>(floor(start)), vec3<i32>(0), vec3<i32>(i32(GRID) - 1));
        let step = select(vec3<i32>(-1), vec3<i32>(1), dir >= vec3<f32>(0.0));
        let delta = abs(inv) * spacing;
        var t_next = (lo + (vec3<f32>(cell) + select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0))) * spacing - eye) * inv;
        for (var i = 0u; i < 3u * GRID; i++) {
            let c = lite.cells[level].xyz + cell;
            let finer = level > 0u && window_coords(level - 1u, vec3<f32>(c) * spacing).w > 0.0;
            if !finer {
                debug_probe(probe_slot(level, c), eye, dir, &hit);
            }
            let t_min = min(t_next.x, min(t_next.y, t_next.z));
            if t_min >= min(t_exit, hit.t) {
                break;
            }
            if t_next.x <= t_min {
                cell.x += step.x;
                t_next.x += delta.x;
            } else if t_next.y <= t_min {
                cell.y += step.y;
                t_next.y += delta.y;
            } else {
                cell.z += step.z;
                t_next.z += delta.z;
            }
            if any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(i32(GRID))) {
                break;
            }
        }
    }
    textureStore(debug_out, gid.xy, vec4<f32>(hit.color, 1.0));
}
