// MegaLights (megalights.rs): the light samples shared by its passes, and their shading.
// Requires common.wgsl, brdf.wgsl, lights.wgsl, megalights_surface.wgsl and these bindings:
// lights: array<GpuLight>, samples: array<u32> (the samples, ML_BLOCK_WORDS per block).
//
// Samples are generated per 2x2 pixel block, at a representative pixel of the block that moves
// every frame: ML_SAMPLES lights chosen in proportion to their unshadowed contribution there
// (resampled from the froxel's light list), each with the unbiased contribution weight of its pick,
// then one shadow ray each. Every pixel of the block shades with all of the block's samples.

const ML_SAMPLES: u32 = 4u;
// Words per block: per sample the light (low 16 bits; ML_NO_LIGHT: none) and its visibility
// (unorm16, high 16 bits, written by the trace), then per sample the contribution weight (f32
// bits), then the header.
const ML_BLOCK_WORDS: u32 = 9u;
const ML_WEIGHT: u32 = 4u;
const ML_HEADER: u32 = 8u;
const ML_NO_LIGHT: u32 = 0xffffu;
// Header: index (0-3) of the representative pixel in its block | ML_VALID.
const ML_VALID: u32 = 4u;

// Visibility guiding: per tile of ML_GUIDE_TILE² pixels, a bit per light (light index modulo 256)
// set when some sample traced a ray towards it this frame, and one when such a ray reached it.
const ML_GUIDE_TILE: u32 = 8u;
const ML_GUIDE_WORDS: u32 = 16u;  // 8 words "sampled", then 8 words "visible"

fn ml_blocks() -> vec2<u32> {
    return viewport_extent(frame.viewport, 2u);
}

fn ml_block_base(block: vec2<u32>) -> u32 {
    return (block.y * ml_blocks().x + block.x) * ML_BLOCK_WORDS;
}

// Index (0-3) of the `k`-th candidate for the representative pixel of `block`: the four pixels in
// an order that starts at a different one every frame (each leads one frame in four) and is
// decorrelated between neighbouring blocks.
fn ml_candidate_index(block: vec2<u32>, k: u32) -> u32 {
    return ((pcg_hash(block.x + block.y * 65536u) + frame.frame_index) + k) & 3u;
}

fn ml_block_pixel(block: vec2<u32>, index: u32) -> vec2<u32> {
    return block * 2u + vec2<u32>(index & 1u, index >> 1u);
}

fn ml_guide_base(pixel: vec2<u32>, viewport: vec4<f32>) -> u32 {
    let tiles = viewport_extent(viewport, ML_GUIDE_TILE);
    let tile = min(pixel / ML_GUIDE_TILE, tiles - 1u);
    return (tile.y * tiles.x + tile.x) * ML_GUIDE_WORDS;
}

// Word and bit of a light in a guide mask.
fn ml_guide_word(light: u32) -> u32 {
    return (light >> 5u) & 7u;
}

fn ml_guide_bit(light: u32) -> u32 {
    return 1u << (light & 31u);
}

// A stateful random sequence (PCG hash chain) for the per-pixel sampling decisions.
fn ml_seed(pixel: vec2<u32>, stream: u32) -> u32 {
    return pcg_hash(pixel.x ^ pcg_hash(pixel.y ^ pcg_hash(frame.frame_index ^ (stream * 0x9e3779b9u))));
}

fn ml_next(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state >> 8u) * (1.0 / 16777216.0);
}

// Unshadowed direct light of `light` at a surface, demodulated: `diffuse` without the diffuse
// colour, `specular` with the Fresnel term (the shading divides it by the environment BRDF).
// `radiance` is the incident radiance scale (before the BRDF).
struct MlLighting {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    radiance: vec3<f32>,
}

fn ml_lighting(s: MlSurface, light: GpuLight) -> MlLighting {
    var c: MlLighting;
    let ls = sample_light(light, s.world);
    c.radiance = ls.radiance;
    let nl = saturate(dot(s.n, ls.l));
    if nl + s.wrap <= 0.0 {
        c.diffuse = vec3<f32>(0.0);
        c.specular = vec3<f32>(0.0);
        return c;
    }
    let brdf = evaluate_brdf(s.n, s.v, ls.l, s.n_dot_v, s.f0, s.a);
    c.diffuse = brdf.diffuse * (nl + s.wrap * s.albedo) * ls.radiance;
    c.specular = brdf.specular * nl * ls.radiance;
    return c;
}

// Demodulated, shadowed direct light at the surface of `pixel` from the samples of its block
// (zero for sky pixels and blocks without samples).
fn ml_shade_samples(pixel: vec2<u32>, s: MlSurface) -> MlLighting {
    var out: MlLighting;
    out.diffuse = vec3<f32>(0.0);
    out.specular = vec3<f32>(0.0);
    out.radiance = vec3<f32>(0.0);
    let base = ml_block_base(pixel / 2u);
    if !s.valid || (samples[base + ML_HEADER] & ML_VALID) == 0u {
        return out;
    }
    for (var k = 0u; k < ML_SAMPLES; k++) {
        let word = samples[base + k];
        let light = word & 0xffffu;
        let visibility = f32(word >> 16u) / 65535.0;
        if light == ML_NO_LIGHT || visibility <= 0.0 {
            continue;
        }
        let weight = bitcast<f32>(samples[base + ML_WEIGHT + k]) * visibility / f32(ML_SAMPLES);
        let c = ml_lighting(s, lights[light]);
        out.diffuse += c.diffuse * weight;
        out.specular += c.specular * weight;
    }
    out.specular /= ml_specular_albedo(s);
    return out;
}
