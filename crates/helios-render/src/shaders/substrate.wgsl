// Substrate-lite (substrate.rs): a material is one slab - an interface whose specular layer has
// its F0 / F90 from the index of refraction and a specular colour, optionally anisotropic, under
// an optional clear coat and an optional sheen (velvet) - over a medium that is diffuse,
// scatters light beneath the surface (subsurface) or lets it through thin geometry
// (transmission). UE 5.7's Substrate mixes and layers several slabs; one slab with these
// features covers glTF's PBR extensions and UE's shading models (clear coat, cloth, subsurface
// profile, two-sided foliage).
//
// Plain materials keep the plain G-buffer. Pixels of materials with slab features also write one
// rgba32uint texel (gbuffer_substrate, only allocated while a scene has such materials) and set
// SURFACE_SUBSTRATE in the G-buffer's shading flags, so lighting reads it only there:
//   x: features (8) | material (8) | clear coat (8) | base roughness or anisotropy (8)
//   y: octahedral base normal (under a clear coat) or anisotropy tangent (16 + 16)
//   z: sheen colour (sqrt, 3 x 8) | sheen roughness (8)
//   w: dielectric F0 (sqrt, 3 x 8) | subsurface or transmission amount (8)
// With a clear coat the G-buffer's normal and roughness are the coat's (Lumen traces reflections
// for the top interface); the base's move to the slab texel. Per-material constants (F90,
// subsurface and transmission colours, radii) come from the slab table by material index.
// Requires common.wgsl.

const SLAB_CLEARCOAT: u32 = 1u;
const SLAB_SHEEN: u32 = 2u;
const SLAB_SUBSURFACE: u32 = 4u;
const SLAB_TRANSMISSION: u32 = 8u;
const SLAB_ANISOTROPY: u32 = 16u;
// F0 / F90 differ from the default dielectric (IOR 1.5, full specular).
const SLAB_SPECULAR: u32 = 32u;
const SLAB_UNLIT: u32 = 64u;
// Evaluated by a material graph (debug view only).
const SLAB_GRAPH: u32 = 128u;

// G-buffer shading flag (motion.w integer part) of pixels with a slab texel.
const SURFACE_SUBSTRATE: f32 = 4.0;
const SURFACE_SUBSTRATE_FLAG: u32 = 4u;

// Per-material slab constants (GpuSlab in substrate.rs).
struct MaterialSlab {
    f0: vec3<f32>,             // dielectric F0: IOR x specular colour x specular
    f90: f32,                  // dielectric F90: specular
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    subsurface_color: vec3<f32>, // relative scattering distance per channel
    subsurface_radius: f32,      // world-space scattering distance (m)
    transmission_color: vec3<f32>,
    transmission: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    features: u32,
    subsurface: f32,
    coat_texture: u32,         // layer: sheen colour (rgb), sheen roughness (a), coat normal, coat roughness, coat
    specular_texture: u32,     // layer: specular colour (rgb), specular (a)
    params: u32,               // first slot of the material's graph parameters
    coat_normal_strength: f32,
    _pad0: u32,
    _pad1: u32,
}

// One pixel's slab.
struct Slab {
    features: u32,
    material: u32,
    f0: vec3<f32>,
    clearcoat: f32,
    // Under a clear coat: the base layer's normal and roughness (the G-buffer has the coat's).
    base_normal: vec3<f32>,
    base_roughness: f32,
    // Anisotropy (-1..1) and its tangent (without a clear coat).
    anisotropy: f32,
    tangent: vec3<f32>,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    // Subsurface amount (SLAB_SUBSURFACE), else thin transmission amount.
    weight: f32,
}

fn slab_none() -> Slab {
    var s: Slab;
    s.f0 = vec3<f32>(0.04);
    s.base_normal = vec3<f32>(0.0, 1.0, 0.0);
    s.tangent = vec3<f32>(1.0, 0.0, 0.0);
    return s;
}

fn has_feature(s: Slab, feature: u32) -> bool {
    return (s.features & feature) != 0u;
}

fn slab_oct_encode(n: vec3<f32>) -> vec2<f32> {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        p = (1.0 - abs(p.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    }
    return p * 0.5 + 0.5;
}

fn slab_oct_decode(e: vec2<f32>) -> vec3<f32> {
    let f = e * 2.0 - 1.0;
    var n = vec3<f32>(f, 1.0 - abs(f.x) - abs(f.y));
    let t = saturate(-n.z);
    n = vec3<f32>(n.xy + select(vec2<f32>(t), vec2<f32>(-t), n.xy >= vec2<f32>(0.0)), n.z);
    return normalize(n);
}

fn encode_slab(s: Slab) -> vec4<u32> {
    let coated = has_feature(s, SLAB_CLEARCOAT);
    let aux = select(s.anisotropy * 0.5 + 0.5, s.base_roughness, coated);
    let direction = select(s.tangent, s.base_normal, coated);
    return vec4<u32>(
        pack4x8unorm(vec4<f32>(f32(s.features) / 255.0, f32(s.material) / 255.0, s.clearcoat, aux)),
        pack2x16unorm(slab_oct_encode(direction)),
        pack4x8unorm(vec4<f32>(sqrt(saturate(s.sheen_color)), s.sheen_roughness)),
        pack4x8unorm(vec4<f32>(sqrt(saturate(s.f0)), s.weight)),
    );
}

fn decode_slab(texel: vec4<u32>) -> Slab {
    var s = slab_none();
    let x = unpack4x8unorm(texel.x);
    s.features = u32(round(x.x * 255.0));
    s.material = u32(round(x.y * 255.0));
    s.clearcoat = x.z;
    let direction = slab_oct_decode(unpack2x16unorm(texel.y));
    if has_feature(s, SLAB_CLEARCOAT) {
        s.base_roughness = x.w;
        s.base_normal = direction;
    } else {
        s.anisotropy = x.w * 2.0 - 1.0;
        s.tangent = direction;
    }
    let z = unpack4x8unorm(texel.z);
    s.sheen_color = z.rgb * z.rgb;
    s.sheen_roughness = z.a;
    let w = unpack4x8unorm(texel.w);
    s.f0 = w.rgb * w.rgb;
    s.weight = w.a;
    return s;
}

// Debug view 24: one colour per feature (blended when several), brighter with more layers
// (base + clear coat + sheen); plain materials grey, graph materials striped.
fn slab_debug_color(s: Slab, pixel: vec2<u32>, substrate: bool) -> vec3<f32> {
    if !substrate {
        return vec3<f32>(0.32);
    }
    var palette = array<vec3<f32>, 7>(
        vec3<f32>(0.2, 0.45, 1.0),   // clear coat: blue
        vec3<f32>(0.95, 0.3, 0.9),   // sheen: magenta
        vec3<f32>(1.0, 0.25, 0.15),  // subsurface: red
        vec3<f32>(0.3, 0.95, 0.3),   // transmission: green
        vec3<f32>(1.0, 0.85, 0.1),   // anisotropy: yellow
        vec3<f32>(0.2, 0.9, 0.95),   // specular (IOR / colour): cyan
        vec3<f32>(1.0, 1.0, 1.0),    // unlit: white
    );
    var color = vec3<f32>(0.0);
    var count = 0.0;
    for (var i = 0u; i < 7u; i++) {
        if has_feature(s, 1u << i) {
            color += palette[i];
            count += 1.0;
        }
    }
    if count == 0.0 {
        color = vec3<f32>(0.5);
        count = 1.0;
    }
    let layers = 1.0 + f32(has_feature(s, SLAB_CLEARCOAT)) + f32(has_feature(s, SLAB_SHEEN));
    var c = color / count * (0.45 + 0.25 * layers);
    if has_feature(s, SLAB_GRAPH) && ((pixel.x + pixel.y) / 6u) % 2u == 0u {
        c *= 0.6;
    }
    return c;
}
