// Lumen's world-space light grid (lumen_light_grid.rs, lumen_light_grid.wgsl): layout and lookup.
// Keep in sync with lumen_light_grid.rs.

const LIGHT_GRID_LEVELS: u32 = 4u;
const LIGHT_GRID_CELLS: u32 = 8u;
const LIGHT_GRID_LEVEL_CELLS: u32 = 512u; // LIGHT_GRID_CELLS^3
// Candidate entries per cell; words per cell: entry count, exact flag, then (light, weight) pairs.
const LIGHT_GRID_ENTRIES: u32 = 32u;
const LIGHT_GRID_CELL_WORDS: u32 = 66u;
const LIGHT_GRID_NONE: u32 = 0xffffffffu;

struct LightGrid {
    // Per level: xyz = the first cell's corner (world), w = cell size.
    levels: array<vec4<f32>, 4>,
    frame: u32,
    // Non-zero: the surface caches sample lights from the grid (else every light in range).
    sampling: u32,
    // First word of the mesh cards' accumulated local light (one rgb9e5 word per atlas texel).
    card_words: u32,
    _pad: u32,
}

// The grid cell holding `p` (finest level first), or LIGHT_GRID_NONE beyond the last level.
fn light_grid_cell(p: vec3<f32>) -> u32 {
    for (var level = 0u; level < LIGHT_GRID_LEVELS; level++) {
        let l = light_grid.levels[level];
        let g = (p - l.xyz) / l.w;
        if all(g >= vec3<f32>(0.0)) && all(g < vec3<f32>(f32(LIGHT_GRID_CELLS))) {
            let c = vec3<u32>(g);
            return level * LIGHT_GRID_LEVEL_CELLS + (c.z * LIGHT_GRID_CELLS + c.y) * LIGHT_GRID_CELLS + c.x;
        }
    }
    return LIGHT_GRID_NONE;
}

// Shared-exponent rgb (9-bit mantissas, 5-bit exponent; as DXGI's R9G9B9E5) for the card
// accumulation words.
fn pack_rgb9e5(color: vec3<f32>) -> u32 {
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(65408.0));
    let largest = max(c.x, max(c.y, c.z));
    var exponent = max(-16, i32(floor(log2(max(largest, 1e-30))))) + 16;
    var scale = exp2(f32(exponent - 24));
    if floor(largest / scale + 0.5) >= 512.0 {
        exponent += 1;
        scale *= 2.0;
    }
    let m = vec3<u32>(min(floor(c / scale + 0.5), vec3<f32>(511.0)));
    return m.x | (m.y << 9u) | (m.z << 18u) | (u32(clamp(exponent, 0, 31)) << 27u);
}

fn unpack_rgb9e5(word: u32) -> vec3<f32> {
    let scale = exp2(f32(i32(word >> 27u) - 24));
    return vec3<f32>(f32(word & 511u), f32((word >> 9u) & 511u), f32((word >> 18u) & 511u)) * scale;
}
