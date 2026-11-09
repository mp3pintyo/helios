// Material graphs (helios-material, material_graphs.rs): the interface of the generated
// `graph_eval` - its output, the parameter block, the helpers its code calls - and the
// conversion of its output into the resolve's Surface and the pixel's slab.
// Requires material.wgsl, svt.wgsl (sample_layer), noise.wgsl, substrate.wgsl and the binding
//   graph_params: array<vec4<f32>, GRAPH_PARAMETER_SLOTS> (uniform; resolve_extended.wgsl).

// vec4 slots of the parameter block shared by every graph material (16 KiB).
const GRAPH_PARAMETER_SLOTS: u32 = 1024u;
// An unset texture parameter samples the material's own texture.
const GRAPH_MATERIAL_TEXTURE: u32 = 0xfffffffeu;

// Bits of GRAPH_OUTPUTS (helios_material::OUTPUT_PINS): the outputs the graph connects.
const GRAPH_SPECULAR: u32 = 8u;
const GRAPH_CLEARCOAT: u32 = 128u;
const GRAPH_SHEEN_COLOR: u32 = 512u;
const GRAPH_SUBSURFACE: u32 = 2048u;
const GRAPH_TRANSMISSION: u32 = 4096u;
const GRAPH_ANISOTROPY: u32 = 8192u;

// The material attributes a graph sets (unconnected outputs keep the material's values).
struct GraphOutput {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    // UE's specular: a dielectric's F0 is 0.08 x specular (0.5: 0.04).
    specular: f32,
    normal: vec3<f32>,      // tangent space
    emissive: vec3<f32>,
    occlusion: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    subsurface: f32,
    transmission: f32,
    anisotropy: f32,
}

fn graph_has(output: u32) -> bool {
    return (GRAPH_OUTPUTS & output) != 0u;
}

// The texture layer of texture parameter `slot`.
fn graph_texture(m: Material, params: u32, slot: u32) -> u32 {
    let layer = bitcast<u32>(graph_params[params + slot].x);
    return select(layer, m.texture, layer == GRAPH_MATERIAL_TEXTURE);
}

// A layer's texels (streamed at native resolution where it is an image layer); white with a
// flat normal where there is no texture.
fn graph_sample(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> LayerTexels {
    if layer == NO_TEXTURE {
        var t: LayerTexels;
        t.albedo = vec4<f32>(1.0);
        t.data = vec4<f32>(0.5, 0.5, 1.0, 1.0);
        return t;
    }
    return sample_layer(layer, uv, ddx, ddy);
}

fn graph_triplanar(layer: u32, p: vec3<f32>, dpx: vec3<f32>, dpy: vec3<f32>, n: vec3<f32>) -> TexSample {
    if layer == NO_TEXTURE {
        var t: TexSample;
        t.albedo = vec3<f32>(1.0);
        t.normal = n;
        t.roughness = 1.0;
        t.height = 1.0;
        t.alpha = 1.0;
        return t;
    }
    return sample_triplanar(layer, p, dpx, dpy, n, 1.0);
}

// Reoriented normal mapping (Barré-Brisebois & Hill 2012): the detail normal rotated onto the
// base normal, as UE's BlendAngleCorrectedNormals.
fn graph_blend_normals(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * (dot(t, u) / t.z) - u);
}

// A world-space direction in the pixel's tangent frame.
fn graph_to_tangent(input: SurfaceInput, n: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(n, input.tangent), dot(n, input.bitangent), dot(n, input.normal));
}

// What unconnected outputs keep: the material's constants and slab.
fn graph_defaults(m: Material, slab: Slab, coat_roughness: f32) -> GraphOutput {
    var g: GraphOutput;
    g.base_color = m.base_color.rgb;
    g.metallic = m.metallic;
    g.roughness = m.roughness;
    g.specular = 0.5;
    g.normal = vec3<f32>(0.0, 0.0, 1.0);
    g.emissive = m.emissive.rgb;
    g.occlusion = 1.0;
    g.clearcoat = slab.clearcoat;
    g.clearcoat_roughness = coat_roughness;
    g.sheen_color = slab.sheen_color;
    g.sheen_roughness = slab.sheen_roughness;
    g.subsurface = select(0.0, slab.weight, has_feature(slab, SLAB_SUBSURFACE));
    g.transmission = select(0.0, slab.weight, has_feature(slab, SLAB_TRANSMISSION));
    g.anisotropy = slab.anisotropy;
    return g;
}

fn graph_surface(g: GraphOutput, m: Material, input: SurfaceInput) -> Surface {
    var s = base_surface(m, input);
    s.albedo = max(g.base_color, vec3<f32>(0.0));
    s.metallic = saturate(g.metallic);
    s.roughness = saturate(g.roughness);
    let frame = mat3x3<f32>(input.tangent, input.bitangent, input.normal);
    let n = frame * g.normal;
    s.normal = select(input.normal, normalize(n), dot(n, n) > 1e-8);
    s.occlusion = saturate(g.occlusion);
    // Emission as with_emission stores glTF emissive colours: a multiple of a tinted albedo.
    let e = max(g.emissive, vec3<f32>(0.0));
    let peak = max(e.r, max(e.g, e.b));
    if peak > 1e-4 {
        s.albedo = mix(s.albedo, e / peak, saturate(peak));
        s.emissive += luminance(e) / max(luminance(s.albedo), 1e-3);
    }
    return s;
}

// The slab with the graph's connected slab outputs.
fn graph_slab(g: GraphOutput, slab: Slab) -> Slab {
    var s = slab;
    s.features |= SLAB_GRAPH;
    if graph_has(GRAPH_SPECULAR) {
        s.f0 = vec3<f32>(0.08 * saturate(g.specular));
        s.features |= SLAB_SPECULAR;
    }
    if graph_has(GRAPH_CLEARCOAT) {
        s.features |= SLAB_CLEARCOAT;
    }
    if graph_has(GRAPH_SHEEN_COLOR) {
        s.features |= SLAB_SHEEN;
    }
    if graph_has(GRAPH_ANISOTROPY) {
        s.features |= SLAB_ANISOTROPY;
    }
    s.clearcoat = saturate(g.clearcoat);
    s.sheen_color = max(g.sheen_color, vec3<f32>(0.0));
    s.sheen_roughness = saturate(g.sheen_roughness);
    s.anisotropy = clamp(g.anisotropy, -1.0, 1.0);
    if graph_has(GRAPH_SUBSURFACE) {
        s.features |= SLAB_SUBSURFACE;
        s.weight = saturate(g.subsurface);
    } else if graph_has(GRAPH_TRANSMISSION) {
        s.features |= SLAB_TRANSMISSION;
        s.weight = saturate(g.transmission);
    }
    return s;
}
