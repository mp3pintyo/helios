// Local lights of Lumen's surface caches (mesh cards: lumen_card_light.wgsl, voxels:
// lumen_light.wgsl): every light in range with a global-distance-field shadow cone each, or a
// few lights importance-sampled from the world-space light grid (lumen_light_grid.wgsl), so the
// cost no longer grows with the light count.
//
// Requires common.wgsl, lights.wgsl, lumen_common.wgsl, lumen_light_grid_common.wgsl and the
// bindings lights: array<GpuLight>, light_grid: LightGrid, light_grid_words: array<u32>.

// Shadow rays per sample point and frame (lights picked by systematic resampling).
const LIGHT_SHADOW_SAMPLES: u32 = 2u;

struct LocalLighting {
    // Incident light times n·l (irradiance), summed over the lights.
    irradiance: vec3<f32>,
    // Whether every light the estimate stands for was evaluated and shadowed (nothing sampled):
    // no temporal accumulation needed.
    exact: bool,
}

// Global-field shadow of a local light (none for lights that cast no shadows).
fn surface_light_shadow(light: GpuLight, ls: LightSample, shadow_origin: vec3<f32>) -> f32 {
    if light.params.z <= 0.5 {
        return 1.0;
    }
    return sdf_soft_shadow(shadow_origin, ls.l, light_shadow_distance(light.position_range.xyz, ls.distance, light.color.w), 0.1);
}

// Every light in range, each shadowed by a global-field cone from `shadow_origin`.
fn local_lights_exhaustive(p: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>) -> vec3<f32> {
    var irradiance = vec3<f32>(0.0);
    for (var i = 0u; i < frame.light_count; i++) {
        let light = lights[i];
        let to_light = light.position_range.xyz - p;
        if dot(to_light, to_light) > light.position_range.w * light.position_range.w {
            continue;
        }
        let ls = sample_light(light, p);
        let nl = saturate(dot(n, ls.l));
        if nl <= 0.0 {
            continue;
        }
        irradiance += ls.radiance * nl * surface_light_shadow(light, ls, shadow_origin);
    }
    return irradiance;
}

// Unshadowed irradiance from grid entry `k` of the cell at `base` (its weight applied).
fn grid_entry_irradiance(base: u32, k: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let index = light_grid_words[base + 2u + k * 2u];
    if index == LIGHT_GRID_NONE {
        return vec3<f32>(0.0);
    }
    let light = lights[index];
    let to_light = light.position_range.xyz - p;
    if dot(to_light, to_light) > light.position_range.w * light.position_range.w {
        return vec3<f32>(0.0);
    }
    let ls = sample_light(light, p);
    return ls.radiance * (saturate(dot(n, ls.l)) * bitcast<f32>(light_grid_words[base + 3u + k * 2u]));
}

// Shadow of grid entry `k`'s light at `p`.
fn grid_entry_shadow(base: u32, k: u32, p: vec3<f32>, shadow_origin: vec3<f32>) -> f32 {
    let light = lights[light_grid_words[base + 2u + k * 2u]];
    return surface_light_shadow(light, sample_light(light, p), shadow_origin);
}

// Local light at a surface point `p` (normal `n`), shadowed from `shadow_origin`. With the grid:
// the unshadowed contribution of every candidate of p's cell (at most LIGHT_GRID_ENTRIES, each
// with its resampling weight) is evaluated; if at most LIGHT_SHADOW_SAMPLES contribute, each is
// shadowed, else LIGHT_SHADOW_SAMPLES of them are picked in proportion to their contribution
// (systematic resampling with the offset `random`) and only those are shadowed - the estimate
// is the candidates' total times the picks' mean visibility (colour from each pick), unbiased,
// noisy only where visibility differs between candidates (penumbrae, lights behind walls).
fn local_lights(p: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>, random: f32) -> LocalLighting {
    var result: LocalLighting;
    result.irradiance = vec3<f32>(0.0);
    result.exact = true;
    if frame.light_count == 0u {
        return result;
    }
    let cell = light_grid_cell(p);
    if light_grid.sampling == 0u || cell == LIGHT_GRID_NONE {
        result.irradiance = local_lights_exhaustive(p, n, shadow_origin);
        return result;
    }
    let base = cell * LIGHT_GRID_CELL_WORDS;
    let count = min(light_grid_words[base], LIGHT_GRID_ENTRIES);
    result.exact = light_grid_words[base + 1u] != 0u;
    // Two passes over the candidates (re-evaluating them is cheaper than keeping them in a
    // private array, which shader compilers handle badly).
    var total = 0.0;
    var contributing = 0u;
    for (var k = 0u; k < count; k++) {
        let c = luminance(grid_entry_irradiance(base, k, p, n));
        total += c;
        contributing += select(0u, 1u, c > 0.0);
    }
    if total <= 0.0 {
        return result;
    }
    let sampled = contributing > LIGHT_SHADOW_SAMPLES;
    let step = total / f32(LIGHT_SHADOW_SAMPLES);
    var threshold = random * step;
    var cumulative = 0.0;
    var picks = 0u;
    for (var k = 0u; k < count && picks < LIGHT_SHADOW_SAMPLES; k++) {
        let rgb = grid_entry_irradiance(base, k, p, n);
        let c = luminance(rgb);
        if c <= 0.0 {
            continue;
        }
        if !sampled {
            // Few enough to shadow each one.
            result.irradiance += rgb * grid_entry_shadow(base, k, p, shadow_origin);
            continue;
        }
        cumulative += c;
        if threshold >= cumulative {
            continue;
        }
        // Picked once or more: the thresholds (step apart) below the cumulative sum; repeated
        // picks of a light share its shadow ray.
        let times = min(u32(ceil((cumulative - threshold) / step)), LIGHT_SHADOW_SAMPLES - picks);
        picks += times;
        threshold += f32(times) * step;
        result.irradiance += rgb * (grid_entry_shadow(base, k, p, shadow_origin) * f32(times) * step / c);
    }
    result.exact = result.exact && !sampled;
    return result;
}
