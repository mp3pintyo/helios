// Lumen uniform layout, constants and binding-free helpers.

const SDF_RES: f32 = 128.0;
const LIGHT_RES: f32 = 64.0;
const SDF_BAND_VOXELS: f32 = 4.0;
const PROBE_RES: u32 = 24u;
const PROBE_LEVELS: u32 = 2u;

struct Lumen {
    // Per SDF level: xyz = world min corner, w = voxel size.
    sdf_levels: array<vec4<f32>, 4>,
    // Per probe level: xyz = world position of probe (0,0,0), w = spacing.
    probe_levels: array<vec4<f32>, 2>,
    level_count: u32,
    frame: u32,
    sdf_enabled: u32,
    max_trace_distance: f32,
    sky_intensity: f32,
    probe_update_fraction: u32,
    voxel_update_phase: u32,
    // World cache (lumen_world_cache_common.wgsl): 0 off, 1 behind the mesh cards, 2 in place of them.
    world_cache: u32,
    // Near field (lumen_mesh_sdf.wgsl). Culling grid min corner, w = cell size (0: disabled).
    mesh_sdf_grid: vec4<f32>,
    // Cells per axis, w = first vec4 of the cell table in mesh_sdf_data.
    mesh_sdf_dims: vec4<u32>,
    // x = near-field trace distance.
    mesh_sdf_trace: vec4<f32>,
    // Mesh cards (lumen_cards.wgsl): x = card instances (0: none), y = first card record in
    // card_data (in vec4s), z = card lighting atlas pages per row.
    cards: vec4<u32>,
}

// ---- Hemispherical octahedral mapping (screen probes) ---------------------------------------------

fn hemi_oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let e = uv * 2.0 - 1.0;
    let t = vec2<f32>(e.x + e.y, e.x - e.y) * 0.5;
    return normalize(vec3<f32>(t, 1.0 - abs(t.x) - abs(t.y)));
}

fn hemi_oct_encode(d: vec3<f32>) -> vec2<f32> {
    let p = d.xy / (abs(d.x) + abs(d.y) + d.z);
    return vec2<f32>(p.x + p.y, p.x - p.y) * 0.5 + 0.5;
}
