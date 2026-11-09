// Runtime virtual texture (RVT) for the landscape, in the spirit of UE's Runtime Virtual
// Texturing: the expensive layered terrain material is evaluated once per texel into cached
// pages instead of per pixel per frame.
//
// Virtual space covers the terrain's XZ rectangle with a mip chain of pages
// (page = PAGE texels + BORDER on each side for filtering). Pages live in two physical atlases
// managed by the generic page cache:
//   albedo atlas : sRGB colour
//   detail atlas : detail normal in the macro-normal frame (xy), roughness
//
// Includers: material.wgsl (texture arrays), noise.wgsl, and for sampling the bindings
//   rvt: RvtParams (uniform), rvt_entries: array<vec4<u32>>, rvt_requests: array<u32> (rw),
//   rvt_albedo / rvt_detail: texture_2d<f32>, rvt_sampler: sampler (linear clamp).

struct RvtParams {
    world_min: vec2<f32>,
    world_size: f32,
    enabled: u32,
    virtual_pages: u32,   // pages per side at mip 0
    mips: u32,
    page_texels: u32,
    border: u32,
    tiles_per_row: u32,
    atlas_size: f32,
    frame: u32,
    layers: u32,          // terrain layer textures (4 bytes)
    mip_offsets: array<vec4<u32>, 4>,
    // Heightfield (object space) and the terrain's world transform (uniform scale + translation).
    hf_origin: vec2<f32>,
    hf_cell: f32,
    hf_scale: f32,
    hf_translation: vec3<f32>,
    lod_bias: f32,
    hf_resolution: vec2<u32>,
    _pad: vec2<u32>,
}

fn rvt_mip_offset(m: u32) -> u32 {
    return rvt.mip_offsets[m / 4u][m % 4u];
}

fn rvt_pages(m: u32) -> u32 {
    return max(rvt.virtual_pages >> m, 1u);
}

// Detail normal frame around `n` (world X projected onto the tangent plane).
fn rvt_frame(n: vec3<f32>) -> mat3x3<f32> {
    var t = vec3<f32>(1.0, 0.0, 0.0) - n * n.x;
    if dot(t, t) < 1e-6 {
        t = vec3<f32>(0.0, 0.0, 1.0) - n * n.z;
    }
    t = normalize(t);
    return mat3x3<f32>(t, cross(n, t), n);
}

// ---- Terrain layering (shared by page rendering and the non-virtual fallback) -------------------

struct TerrainSample {
    albedo: vec3<f32>,
    normal: vec3<f32>,   // world
    roughness: f32,
}

const LAYER_TILING: vec4<f32> = vec4<f32>(0.33, 0.25, 0.12, 0.18); // repeats per metre

// `texel`: world size of one output sample (selects the texture mip).
fn terrain_layering(p: vec3<f32>, macro_normal: vec3<f32>, texel: f32, layers: u32) -> TerrainSample {
    let large = fbm(p * 0.015, 4u, texel * 0.015);
    let medium = fbm(p * 0.11 + 7.0, 3u, texel * 0.11);
    let slope = macro_normal.y + 0.06 * medium;

    var weights = vec4<f32>(0.0);
    weights.x = smoothstep(0.2, 0.8, 0.5 + medium * 0.9 + large * 1.2);             // grass
    weights.y = 1.0 - weights.x;                                                 // dirt
    let rock = 1.0 - smoothstep(0.62, 0.8, slope);
    let snow = smoothstep(70.0, 84.0, p.y + 14.0 * large) * smoothstep(0.65, 0.85, slope);
    weights = vec4<f32>(weights.xy * (1.0 - rock), rock, 0.0);
    weights = weights * (1.0 - snow) + vec4<f32>(0.0, 0.0, 0.0, snow);

    var result: TerrainSample;
    result.albedo = vec3<f32>(0.0);
    result.roughness = 0.0;
    var normal = vec3<f32>(0.0);
    var height_weights = vec4<f32>(0.0);
    var samples: array<TexSample, 4>;
    for (var i = 0u; i < 4u; i++) {
        let layer = texture_layer(layers, i);
        if weights[i] <= 0.001 || layer == NO_TEXTURE {
            continue;
        }
        let s = LAYER_TILING[i];
        let q = p * s;
        let d = texel * s;
        if i == 2u {
            samples[i] = sample_triplanar(layer, q, vec3<f32>(d, 0.0, 0.0), vec3<f32>(0.0, 0.0, d), macro_normal, 1.2);
        } else {
            let frame = rvt_frame(macro_normal);
            samples[i] = sample_uv(layer, q.xz, vec2<f32>(d, 0.0), vec2<f32>(0.0, d), mat3x3<f32>(frame[0], -frame[1], frame[2]), 1.0);
        }
        // Height blending: higher texels of a layer win at the transitions.
        height_weights[i] = weights[i] * (samples[i].height + 0.25);
    }
    let sharpened = pow(height_weights / max(max(max(height_weights.x, height_weights.y), max(height_weights.z, height_weights.w)), 1e-4), vec4<f32>(3.0));
    let total = max(dot(sharpened, vec4<f32>(1.0)), 1e-4);
    for (var i = 0u; i < 4u; i++) {
        let w = sharpened[i] / total;
        if w > 0.0 {
            result.albedo += samples[i].albedo * w;
            result.roughness += samples[i].roughness * w;
            normal += samples[i].normal * w;
        }
    }
    if total <= 1e-4 {
        result.albedo = vec3<f32>(0.2, 0.18, 0.12);
        result.roughness = 0.9;
        normal = macro_normal;
    }
    result.albedo *= 0.85 + 0.3 * large;
    result.normal = normalize(normal + macro_normal * 1e-3);
    return result;
}
