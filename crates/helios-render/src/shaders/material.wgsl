// Material evaluation, specialised per shading model (MATERIAL_KIND override in the including
// shader). Vertex colours multiply the base colour (glTF): includers fold the interpolated
// colour into the Material copy's base_color (factor x vertex colour x texture). Requires common.wgsl, noise.wgsl and these bindings in the including shader:
//   material_albedo: texture_2d_array<f32> (sRGB view), material_data: texture_2d_array<f32>,
//   material_sampler: sampler (repeat, anisotropic).
// Layers are sampled through `sample_layer(layer, uv, ddx, ddy) -> LayerTexels`, which the
// includer provides: svt.wgsl (image layers streamed at native resolution) or svt_off.wgsl
// (texture arrays only).
// The Terrain model additionally needs rvt.wgsl.

const MATERIAL_STANDARD: u32 = 0u;
const MATERIAL_ROCK: u32 = 1u;
const MATERIAL_TERRAIN: u32 = 2u;
const MATERIAL_METAL: u32 = 3u;
const MATERIAL_FOLIAGE: u32 = 4u;
const MATERIAL_GLASS: u32 = 5u;
const MATERIAL_SKY: u32 = 7u;
const MATERIAL_KINDS: u32 = 8u;

const MAPPING_UV: u32 = 0u;
const MAPPING_WORLD: u32 = 1u;
const MAPPING_OBJECT: u32 = 2u;

// G-buffer shading flags (motion.w).
const SURFACE_FOLIAGE: f32 = 1.0;
const SURFACE_GLASS: f32 = 2.0;

const NO_TEXTURE: u32 = 0xffffffffu;

// The rarer texture coordinate features: the second UV set, wrap modes other than repeat, and
// occlusion layers of their own. The G-buffer resolve compiles them only for scenes that use
// them (passes/material.rs); other includers keep the default.
override UV_EXTENSIONS: bool = true;

struct Surface {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    normal: vec3<f32>,
    emissive: f32,
    flags: f32,
    occlusion: f32,     // material ambient occlusion (indirect light only), 1 = none
}

// Everything the resolve knows about the pixel's surface point.
struct SurfaceInput {
    object_position: vec3<f32>,
    world_position: vec3<f32>,
    normal: vec3<f32>,           // interpolated world normal (facing the viewer)
    tangent: vec3<f32>,          // world-space dP/du (orthonormalised)
    bitangent: vec3<f32>,
    uv: vec2<f32>,               // mesh UV set 0
    duv_dx: vec2<f32>,
    duv_dy: vec2<f32>,
    uv1: vec2<f32>,              // mesh UV set 1 (set 0 where the mesh has none)
    duv1_dx: vec2<f32>,
    duv1_dy: vec2<f32>,
    dpdx: vec3<f32>,             // world position derivatives per pixel
    dpdy: vec3<f32>,
    object_dpdx: vec3<f32>,
    object_dpdy: vec3<f32>,
    footprint: f32,              // object-space pixel size
    normal_matrix: mat3x3<f32>,
    back_facing: bool,
}

struct TexSample {
    albedo: vec3<f32>,
    normal: vec3<f32>,  // world space
    roughness: f32,
    height: f32,        // metallic for image textures (MATERIAL_FLAG_IMAGE_TEXTURE)
    alpha: f32,         // albedo alpha: coverage, or ambient occlusion (MATERIAL_FLAG_OCCLUSION_IN_TEXTURE)
}

// Texture coordinates and their screen derivatives.
struct TexCoords {
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
}

// The coordinates a texture with transform `t` samples at the pixel.
fn texture_coords(t: UvTransform, input: SurfaceInput) -> TexCoords {
    var c: TexCoords;
    var uv = input.uv;
    var ddx = input.duv_dx;
    var ddy = input.duv_dy;
    if UV_EXTENSIONS && t.uv_set == 1u {
        uv = input.uv1;
        ddx = input.duv1_dx;
        ddy = input.duv1_dy;
    }
    c.uv = uv_transform_point(t, uv);
    if UV_EXTENSIONS {
        c.uv = uv_wrap(t, c.uv);
    }
    c.ddx = uv_transform_vector(t, ddx);
    c.ddy = uv_transform_vector(t, ddy);
    return c;
}

fn decode_tangent_normal(data: vec4<f32>, strength: f32) -> vec3<f32> {
    let xy = (data.xy * 2.0 - 1.0) * strength;
    return normalize(vec3<f32>(xy, sqrt(saturate(1.0 - dot(xy, xy))) + 1e-4));
}

// Raw texels of one layer: albedo (linear rgb, alpha = coverage) and data (tangent-space
// normal xy, roughness, height or metallic).
struct LayerTexels {
    albedo: vec4<f32>,
    data: vec4<f32>,
}

// A layer from the texture arrays (512² with mips; every layer, always resident).
fn sample_layer_array(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> LayerTexels {
    var t: LayerTexels;
    t.albedo = textureSampleGrad(material_albedo, material_sampler, uv, layer, ddx, ddy);
    t.data = textureSampleGrad(material_data, material_sampler, uv, layer, ddx, ddy);
    return t;
}

// Base colour and coverage of a layer (`uv` and its derivatives in texture repeats).
fn sample_albedo(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    return sample_layer(layer, uv, ddx, ddy).albedo;
}

fn sample_uv(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, frame: mat3x3<f32>, strength: f32) -> TexSample {
    var t: TexSample;
    let s = sample_layer(layer, uv, ddx, ddy);
    t.albedo = s.albedo.rgb;
    t.normal = normalize(frame * decode_tangent_normal(s.data, strength));
    t.roughness = s.data.z;
    t.height = s.data.w;
    t.alpha = s.albedo.a;
    return t;
}

// Triplanar projection with whiteout-style normal blending. `p` and its derivatives are in the
// projection space (world or object, already scaled by the tiling). Triplanar mappings tile
// (procedural) textures over large surfaces: they sample the texture arrays only.
fn sample_triplanar(layer: u32, p: vec3<f32>, dpx: vec3<f32>, dpy: vec3<f32>, n: vec3<f32>, strength: f32) -> TexSample {
    var w = pow(abs(n), vec3<f32>(4.0));
    w /= w.x + w.y + w.z;
    let ax = textureSampleGrad(material_albedo, material_sampler, p.zy, layer, dpx.zy, dpy.zy);
    let ay = textureSampleGrad(material_albedo, material_sampler, p.xz, layer, dpx.xz, dpy.xz);
    let az = textureSampleGrad(material_albedo, material_sampler, p.xy, layer, dpx.xy, dpy.xy);
    let dx = textureSampleGrad(material_data, material_sampler, p.zy, layer, dpx.zy, dpy.zy);
    let dy = textureSampleGrad(material_data, material_sampler, p.xz, layer, dpx.xz, dpy.xz);
    let dz = textureSampleGrad(material_data, material_sampler, p.xy, layer, dpx.xy, dpy.xy);
    let tx = (dx.xy * 2.0 - 1.0) * strength;
    let ty = (dy.xy * 2.0 - 1.0) * strength;
    let tz = (dz.xy * 2.0 - 1.0) * strength;
    let nx = vec3<f32>(0.0, tx.y, tx.x);
    let ny = vec3<f32>(ty.x, 0.0, ty.y);
    let nz = vec3<f32>(tz.x, tz.y, 0.0);
    var t: TexSample;
    t.albedo = ax.rgb * w.x + ay.rgb * w.y + az.rgb * w.z;
    t.normal = normalize(n + nx * w.x + ny * w.y + nz * w.z);
    t.roughness = dx.z * w.x + dy.z * w.y + dz.z * w.z;
    t.height = dx.w * w.x + dy.w * w.y + dz.w * w.z;
    t.alpha = ax.a * w.x + ay.a * w.y + az.a * w.z;
    return t;
}

fn sample_material_texture(m: Material, input: SurfaceInput) -> TexSample {
    let strength = m.normal_strength;
    switch m.mapping {
        case MAPPING_WORLD: {
            let s = m.uv_scale;
            return sample_triplanar(m.texture, input.world_position * s, input.dpdx * s, input.dpdy * s, input.normal, strength);
        }
        case MAPPING_OBJECT: {
            let s = m.uv_scale;
            // Object-space projection axes follow the object; blend weights use the object normal.
            let object_normal = normalize(transpose(input.normal_matrix) * input.normal);
            var t = sample_triplanar(m.texture, input.object_position * s, input.object_dpdx * s, input.object_dpdy * s, object_normal, strength);
            t.normal = normalize(input.normal_matrix * t.normal);
            return t;
        }
        default: {
            // The transform includes the material's tiling (uv_scale).
            let c = texture_coords(m.uv_texture, input);
            let frame = mat3x3<f32>(input.tangent, input.bitangent, input.normal);
            return sample_uv(m.texture, c.uv, c.ddx, c.ddy, frame, strength);
        }
    }
}

fn texture_layer(packed: u32, index: u32) -> u32 {
    let layer = (packed >> (index * 8u)) & 0xffu;
    return select(layer, NO_TEXTURE, layer == 255u);
}

// Tilts `n` along the surface gradient of a height field (bump mapping without UVs).
fn perturb_normal(n: vec3<f32>, gradient: vec3<f32>, strength: f32) -> vec3<f32> {
    let surface_gradient = gradient - dot(gradient, n) * n;
    return normalize(n - surface_gradient * strength);
}

// Value and forward-difference gradient of fBm (4 evaluations).
fn fbm_with_gradient(p: vec3<f32>, octaves: u32, footprint: f32) -> vec4<f32> {
    let h = max(footprint, 0.002);
    let f0 = fbm(p, octaves, footprint);
    let fx = fbm(p + vec3<f32>(h, 0.0, 0.0), octaves, footprint);
    let fy = fbm(p + vec3<f32>(0.0, h, 0.0), octaves, footprint);
    let fz = fbm(p + vec3<f32>(0.0, 0.0, h), octaves, footprint);
    return vec4<f32>(f0, (vec3<f32>(fx, fy, fz) - f0) / h);
}

fn base_surface(m: Material, input: SurfaceInput) -> Surface {
    var s: Surface;
    s.albedo = m.base_color.rgb;
    s.roughness = m.roughness;
    s.metallic = m.metallic;
    s.normal = input.normal;
    s.emissive = m.emissive.w;
    s.flags = 0.0;
    s.occlusion = 1.0;
    return s;
}

fn evaluate_standard(m: Material, input: SurfaceInput) -> Surface {
    var s = base_surface(m, input);
    var texture_alpha = 1.0;
    if m.texture != NO_TEXTURE {
        let t = sample_material_texture(m, input);
        s.albedo *= t.albedo;
        s.normal = t.normal;
        texture_alpha = t.alpha;
        if (m.flags & MATERIAL_FLAG_IMAGE_TEXTURE) != 0u {
            // Authored maps: glTF metallic-roughness, the factors scale the texture.
            s.roughness = t.roughness * m.roughness;
            s.metallic = t.height * m.metallic;  // image layers store metallic in place of height
        } else {
            s.roughness = saturate(t.roughness * (m.roughness / 0.5));
        }
    }
    s.occlusion = material_occlusion(m, input, texture_alpha);
    return with_emission(s, m, input);
}

// Ambient occlusion of image materials: the alpha of an image layer with an occlusion image
// (the material's own texture sample when it shares its coordinates), scaled by the strength.
// An occlusion layer of its own (other coordinates than the texture) is read from its 512² array
// layer only: occlusion is low-frequency, and another streamed lookup would cost every pixel of
// the resolve pass registers and code (it is not streamed, see svt/pyramid.rs).
fn material_occlusion(m: Material, input: SurfaceInput, texture_alpha: f32) -> f32 {
    var ao = 1.0;
    if (m.flags & MATERIAL_FLAG_OCCLUSION_IN_TEXTURE) != 0u {
        ao = texture_alpha;
    } else if UV_EXTENSIONS && m.occlusion_texture != NO_TEXTURE {
        let c = texture_coords(m.uv_occlusion, input);
        ao = textureSampleGrad(material_albedo, material_sampler, c.uv, m.occlusion_texture, c.ddx, c.ddy).a;
    }
    return 1.0 + (m.occlusion_strength * (ao - 1.0));
}

// Adds emission independent of the base colour (glTF emissive). The G-buffer stores emission as
// one multiple of the albedo, so where emission dominates the albedo is tinted towards the
// emitted colour, and the multiple is chosen to reproduce the emitted luminance.
fn with_emission(surface: Surface, m: Material, input: SurfaceInput) -> Surface {
    var s = surface;
    var e = m.emissive.rgb;
    if m.emissive_texture != NO_TEXTURE && any(e > vec3<f32>(0.0)) {
        let c = texture_coords(m.uv_emissive, input);
        e *= sample_albedo(m.emissive_texture, c.uv, c.ddx, c.ddy).rgb;
    }
    let peak = max(e.r, max(e.g, e.b));
    if peak > 1e-4 {
        s.albedo = mix(s.albedo, e / peak, saturate(peak));
        s.emissive += luminance(e) / max(luminance(s.albedo), 1e-3);
    }
    return s;
}

fn evaluate_rock(m: Material, input: SurfaceInput) -> Surface {
    var s = base_surface(m, input);
    let scale = m.detail_scale;
    let footprint = input.footprint * scale;
    let p = input.object_position * scale;
    let n = fbm_with_gradient(p, 5u, footprint);
    let mottling = fbm(p * 0.23 + 4.1, 3u, footprint * 0.23);
    var albedo = m.base_color.rgb * (0.62 + 0.75 * n.x + 0.35 * mottling);
    var normal = perturb_normal(input.normal, input.normal_matrix * n.yzw, m.detail_strength);
    var roughness = clamp(m.roughness + 0.2 * n.x, 0.3, 1.0);
    if m.texture != NO_TEXTURE {
        let t = sample_material_texture(m, input);
        albedo *= t.albedo * 2.0;
        normal = normalize(normal + t.normal - input.normal);
        roughness = mix(roughness, t.roughness, 0.5);
    }
    // Lichen / moss settles on upward-facing surfaces.
    let up = normal.y + 0.25 * mottling;
    let moss_layer = texture_layer(m.layers, 0u);
    let cover = smoothstep(0.5, 0.85, up) * smoothstep(-0.15, 0.25, mottling);
    if moss_layer != NO_TEXTURE {
        let mp = input.world_position * 0.7;
        let moss = sample_triplanar(moss_layer, mp, input.dpdx * 0.7, input.dpdy * 0.7, normal, 1.0);
        let w = saturate(cover * 1.4 + (moss.height - 0.5) * 0.8 * cover);
        albedo = mix(albedo, moss.albedo, w);
        roughness = mix(roughness, moss.roughness, w);
        normal = normalize(mix(normal, moss.normal, w));
    } else {
        albedo = mix(albedo, vec3<f32>(0.20, 0.21, 0.09), cover * 0.75);
        roughness -= 0.15 * cover;
    }
    s.albedo = clamp(albedo, vec3<f32>(0.01), vec3<f32>(0.9));
    s.roughness = roughness;
    s.normal = normal;
    return s;
}

fn evaluate_metal(m: Material, input: SurfaceInput) -> Surface {
    var s = base_surface(m, input);
    let scale = m.detail_scale;
    let footprint = input.footprint * scale;
    let p = input.object_position * scale;
    let brushed = fbm(p * vec3<f32>(1.0, 1.0, 24.0), 4u, footprint * 24.0);
    let wear = fbm(p * 0.5, 4u, footprint * 0.5);
    s.albedo = m.base_color.rgb * (0.9 + 0.2 * wear);
    s.roughness = clamp(m.roughness + 0.08 * brushed + 0.12 * smoothstep(0.1, 0.4, wear), 0.05, 1.0);
    s.metallic = 1.0;
    if m.texture != NO_TEXTURE {
        let t = sample_material_texture(m, input);
        s.albedo *= mix(vec3<f32>(1.0), t.albedo, 0.5);
        s.roughness = saturate(mix(s.roughness, t.roughness, 0.6));
        s.normal = t.normal;
    }
    return s;
}

fn evaluate_foliage(m: Material, input: SurfaceInput) -> Surface {
    var s = evaluate_standard(m, input);
    s.flags = SURFACE_FOLIAGE;
    s.metallic = 0.0;
    return s;
}

fn evaluate_glass(m: Material, input: SurfaceInput) -> Surface {
    var s = base_surface(m, input);
    var glass = 1.0;
    if m.texture != NO_TEXTURE {
        let t = sample_material_texture(m, input);
        s.albedo *= t.albedo;
        glass = 1.0 - smoothstep(0.15, 0.35, t.roughness);
        s.roughness = mix(0.6, 0.08, glass);
        s.normal = t.normal;
    }
    // Lead came is dull dark metal; the glass itself transmits (translucency.wgsl reads the
    // glass coverage from `emissive`).
    s.metallic = 0.0;
    s.emissive = glass * max(m.emissive.w, 1.0);
    s.flags = SURFACE_GLASS;
    return s;
}
