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

// `texel`: world size of one output sample (selects the texture mip). The landscape: grass,
// dirt and snow projected from above, rock triplanar (the mesh terrain's is
// mesh_terrain_layering below).
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

// ---- Mesh terrain (cliffs, overhangs, caves) ------------------------------------------------------

// Repeats per metre of the mesh terrain's rock: the large scale (masses, fracture planes, streaks)
// and the detail scale (chips, grain) of the same texture. Their ratio is not a whole number, so
// the two never repeat together.
const CLIFF_TILING: vec2<f32> = vec2<f32>(0.083, 0.37);
// cos, sin of the angle the detail scale's projection is turned by about the vertical.
const CLIFF_TURN: vec2<f32> = vec2<f32>(0.6, 0.8);
// Projections whose triplanar weight is below this share are skipped (on a wall one or two of
// three).
const TRIPLANAR_MIN_WEIGHT: f32 = 0.02;

// Triplanar sample with an isotropic footprint `d` (in projection units) that skips the
// projections a surface barely faces: on walls and ledges this halves the texture fetches.
// Without `colour` only the data texels are fetched (albedo stays 0).
fn triplanar_sparse(layer: u32, p: vec3<f32>, d: f32, n: vec3<f32>, strength: f32, colour: bool) -> TexSample {
    var w = pow(abs(n), vec3<f32>(4.0));
    w = select(vec3<f32>(0.0), w, w > vec3<f32>(TRIPLANAR_MIN_WEIGHT * (w.x + w.y + w.z)));
    w /= max(w.x + w.y + w.z, 1e-6);
    let gx = vec2<f32>(d, 0.0);
    let gy = vec2<f32>(0.0, d);
    var t: TexSample;
    t.albedo = vec3<f32>(0.0);
    t.roughness = 0.0;
    t.height = 0.0;
    t.alpha = 1.0;
    var tilt = vec3<f32>(0.0);
    if w.x > 0.0 {
        if colour {
            t.albedo += textureSampleGrad(material_albedo, material_sampler, p.zy, layer, gx, gy).rgb * w.x;
        }
        let e = textureSampleGrad(material_data, material_sampler, p.zy, layer, gx, gy);
        let tn = (e.xy * 2.0 - 1.0) * strength;
        t.roughness += e.z * w.x;
        t.height += e.w * w.x;
        tilt += vec3<f32>(0.0, tn.y, tn.x) * w.x;
    }
    if w.y > 0.0 {
        if colour {
            t.albedo += textureSampleGrad(material_albedo, material_sampler, p.xz, layer, gx, gy).rgb * w.y;
        }
        let e = textureSampleGrad(material_data, material_sampler, p.xz, layer, gx, gy);
        let tn = (e.xy * 2.0 - 1.0) * strength;
        t.roughness += e.z * w.y;
        t.height += e.w * w.y;
        tilt += vec3<f32>(tn.x, 0.0, tn.y) * w.y;
    }
    if w.z > 0.0 {
        if colour {
            t.albedo += textureSampleGrad(material_albedo, material_sampler, p.xy, layer, gx, gy).rgb * w.z;
        }
        let e = textureSampleGrad(material_data, material_sampler, p.xy, layer, gx, gy);
        let tn = (e.xy * 2.0 - 1.0) * strength;
        t.roughness += e.z * w.z;
        t.height += e.w * w.z;
        tilt += vec3<f32>(tn.x, tn.y, 0.0) * w.z;
    }
    t.normal = normalize(n + tilt);
    return t;
}

// The mean colour of a layer (its 1x1 mip).
fn layer_mean(layer: u32) -> vec3<f32> {
    return textureSampleLevel(material_albedo, material_sampler, vec2<f32>(0.5), layer, f32(textureNumLevels(material_albedo) - 1u)).rgb;
}

fn strata_hash(bed: f32, salt: u32) -> f32 {
    return f32(pcg_hash(bitcast<u32>(i32(bed)) ^ salt) & 0xffffu) / 65535.0;
}

// The mesh terrain's vertex attributes (MeshTerrain::shading_attributes, in its UVs).
struct RockAttributes {
    strata: f32,        // strata coordinate: bedding planes at its integers, odd beds protrude
    strata_texel: f32,  // its change across a pixel
    cavity: f32,        // 0 open, 1 deep in a crevice
}

// Natural cliff rock: the rock layer at two scales (masses and fracture planes, chips and grain),
// coloured by the strata the terrain's ledges were cut at - every bed its own tone, the hard
// (protruding) beds cleaner and smoother than the soft, recessed ones, a dark seam where beds meet
// - with moss and lichen on up-facing ledges and in sheltered hollows and dirt in crevices.
fn cliff_rock(layer: u32, dirt_layer: u32, p: vec3<f32>, n: vec3<f32>, texel: f32, a: RockAttributes, medium: f32) -> TexSample {
    let coarse = triplanar_sparse(layer, p * CLIFF_TILING.x + vec3<f32>(0.37, 0.0, 0.61), texel * CLIFF_TILING.x, n, 1.3, true);
    // The detail is projected turned about the vertical (walls keep their bedding horizontal),
    // so on ledges and undersides its grain crosses the large scale's instead of doubling it.
    let turn = mat3x3<f32>(vec3<f32>(CLIFF_TURN.x, 0.0, CLIFF_TURN.y), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(-CLIFF_TURN.y, 0.0, CLIFF_TURN.x));
    // Only its data texels: its height (mean about 0.5; mip-filtered towards it where the
    // detail is sub-pixel) modulates the large scale's colour.
    // Skipped where a pixel covers more than 1/32 of its tile (16 texels: little is left of it).
    var fine: TexSample;
    fine.height = 0.5;
    fine.normal = n;
    if texel * CLIFF_TILING.y < 1.0 / 32.0 {
        fine = triplanar_sparse(layer, (turn * p) * CLIFF_TILING.y + vec3<f32>(0.13, 0.5, 0.29), texel * CLIFF_TILING.y, turn * n, 1.1, false);
        fine.normal = transpose(turn) * fine.normal;
    }
    let detail = 0.4 + 1.2 * fine.height;
    var t: TexSample;
    t.albedo = coarse.albedo * detail;
    t.normal = normalize(coarse.normal + fine.normal - n);
    t.roughness = select(coarse.roughness, 0.5 * (coarse.roughness + fine.roughness), fine.roughness > 0.0);
    t.height = 0.6 * coarse.height + 0.4 * fine.height;
    t.alpha = 1.0;

    // Strata: a tone per bed (lighter or darker, warmer or greyer), hard beds lighter.
    let hard = -sin(PI * a.strata);
    let bed = floor(a.strata);
    let tint = mix(vec3<f32>(1.06, 0.99, 0.9), vec3<f32>(0.94, 0.97, 1.02), strata_hash(bed, 0x9e37u));
    t.albedo *= tint * ((0.86 + 0.26 * strata_hash(bed, 0x85ebu)) * (1.0 + 0.07 * hard));
    // Fine laminae within a bed, faded once they are sub-pixel.
    let laminae = sin(a.strata * PI * 5.0 + 1.7 * medium) * (1.0 - smoothstep(0.05, 0.12, a.strata_texel * 5.0));
    t.albedo *= 1.0 + 0.05 * laminae;
    // The bedding plane: a thin dark parting on the faces where the bed is resolved - open on
    // some planes, closed on others, and coming and going along each.
    let plane = abs(fract(a.strata + 0.5) - 0.5);
    let parting = strata_hash(floor(a.strata + 0.5), 0x2c1bu) * smoothstep(-0.2, 0.25, medium + 0.3 * coarse.height - 0.15);
    let seam = (1.0 - smoothstep(0.012, 0.04 + a.strata_texel, plane)) * (1.0 - smoothstep(0.08, 0.25, a.strata_texel)) * (1.0 - abs(n.y));
    t.albedo *= 1.0 - 0.45 * seam * parting;
    // Soft beds weather rougher. Undersides, out of the sun and the rain, stay darker.
    t.roughness = saturate(t.roughness + 0.05 * (1.0 - hard));
    t.albedo *= 1.0 - 0.3 * saturate(-n.y);

    // Dirt washed into crevices and onto ledges.
    let dirt = smoothstep(0.25, 0.6, a.cavity + 0.15 * medium) * smoothstep(-0.4, 0.2, t.normal.y);
    if dirt > 0.0 && dirt_layer != NO_TEXTURE {
        t.albedo = mix(t.albedo, layer_mean(dirt_layer) * (0.6 + 0.4 * detail), dirt);
        t.roughness = mix(t.roughness, 0.95, dirt);
    }
    // Moss and lichen (as on the valley's boulders): up-facing rock, more in hollows and on the
    // soft beds; undersides and the clean faces stay bare.
    let moss = smoothstep(0.4, 0.8, t.normal.y + 0.25 * medium + 0.35 * a.cavity - 0.08 * hard) * smoothstep(-0.25, 0.2, medium);
    t.albedo = mix(t.albedo, vec3<f32>(0.20, 0.21, 0.09) * (0.5 + 0.5 * detail), moss * 0.8);
    t.roughness = mix(t.roughness, 0.92, moss);
    t.height += 0.1 * moss;
    return t;
}

// The mesh terrain's layering: as the landscape's, but every layer projected triplanar (the
// top-down projection of grass and dirt would streak down its walls) and the rock is cliff rock.
fn mesh_terrain_layering(p: vec3<f32>, macro_normal: vec3<f32>, texel: f32, layers: u32, a: RockAttributes) -> TerrainSample {
    let large = fbm(p * 0.015, 4u, texel * 0.015);
    let medium = fbm(p * 0.11 + 7.0, 3u, texel * 0.11);
    let slope = macro_normal.y + 0.06 * medium;

    var weights = vec4<f32>(0.0);
    weights.x = smoothstep(0.2, 0.8, 0.5 + medium * 0.9 + large * 1.2);             // grass
    weights.y = 1.0 - weights.x;                                                 // dirt
    let rock = 1.0 - smoothstep(0.62, 0.8, slope);
    weights = vec4<f32>(weights.xy * (1.0 - rock), rock, 0.0);

    var result: TerrainSample;
    result.albedo = vec3<f32>(0.0);
    result.roughness = 0.0;
    var normal = vec3<f32>(0.0);
    var height_weights = vec4<f32>(0.0);
    var samples: array<TexSample, 4>;
    for (var i = 0u; i < 3u; i++) {
        let layer = texture_layer(layers, i);
        if weights[i] <= 0.001 || layer == NO_TEXTURE {
            continue;
        }
        if i == 2u {
            samples[i] = cliff_rock(layer, texture_layer(layers, 1u), p, macro_normal, texel, a, medium);
        } else {
            let s = LAYER_TILING[i];
            samples[i] = triplanar_sparse(layer, p * s, texel * s, macro_normal, 1.2, true);
        }
        // Height blending: higher texels of a layer win at the transitions.
        height_weights[i] = weights[i] * (samples[i].height + 0.25);
    }
    let sharpened = pow(height_weights / max(max(height_weights.x, height_weights.y), max(height_weights.z, 1e-4)), vec4<f32>(3.0));
    let total = max(dot(sharpened, vec4<f32>(1.0)), 1e-4);
    for (var i = 0u; i < 3u; i++) {
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
