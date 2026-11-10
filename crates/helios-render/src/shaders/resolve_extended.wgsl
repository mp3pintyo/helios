// Extended resolve pipelines (passes/material.rs, substrate_gpu.rs): materials with a Substrate
// slab, a material graph or both, resolved over tiles of their own (slots 8-23). Replaces
// resolve_extended_off.wgsl in gbuffer.wgsl; a graph material's compiled graph replaces
// material_graph_off.wgsl.
//
// Bind group 1: the slab table, the graph parameter block (material instances' values) and
// the slab texels of the G-buffer (substrate.wgsl).

#include "substrate.wgsl"
#include "material_graph.wgsl"
#include "material_graph_off.wgsl"

@group(1) @binding(0) var<uniform> slabs: array<MaterialSlab, MAX_MATERIALS>;
@group(1) @binding(1) var<uniform> graph_params: array<vec4<f32>, GRAPH_PARAMETER_SLOTS>;
@group(1) @binding(2) var gbuffer_substrate: texture_storage_2d<rgba32uint, write>;

// RenderSettings::substrate: off, graphs still run but no pixel keeps its slab (the A/B
// comparison against plain materials).
override SUBSTRATE: bool = true;

const EXTENDED_RESOLVE: bool = true;

// A material's slab at the pixel, with its clear coat's normal and roughness (which end up in
// the G-buffer).
struct SlabInput {
    slab: Slab,
    coat_normal: vec3<f32>,
    coat_roughness: f32,
}

// The material's slab constants times its slab textures. These are read from the 512² arrays
// with the coordinates of the material's main texture.
fn material_slab(c: MaterialSlab, m: Material, input: SurfaceInput, material: u32) -> SlabInput {
    var r: SlabInput;
    var s = slab_none();
    s.features = c.features;
    s.material = material;
    s.f0 = c.f0;
    s.clearcoat = c.clearcoat;
    s.anisotropy = c.anisotropy;
    s.sheen_color = c.sheen_color;
    s.sheen_roughness = c.sheen_roughness;
    s.weight = select(c.transmission, c.subsurface, (c.features & SLAB_SUBSURFACE) != 0u);
    r.coat_normal = input.normal;
    r.coat_roughness = c.clearcoat_roughness;
    if c.coat_texture != NO_TEXTURE || c.specular_texture != NO_TEXTURE {
        let tc = texture_coords(m.uv_texture, input);
        if c.coat_texture != NO_TEXTURE {
            let t = sample_layer_array(c.coat_texture, tc.uv, tc.ddx, tc.ddy);
            s.sheen_color *= t.albedo.rgb;
            s.sheen_roughness *= t.albedo.a;
            s.clearcoat *= t.data.w;
            r.coat_roughness *= t.data.z;
            let frame = mat3x3<f32>(input.tangent, input.bitangent, input.normal);
            r.coat_normal = normalize(frame * decode_tangent_normal(t.data, c.coat_normal_strength));
        }
        if c.specular_texture != NO_TEXTURE {
            let t = sample_layer_array(c.specular_texture, tc.uv, tc.ddx, tc.ddy);
            s.f0 *= t.albedo.rgb * t.albedo.a;
        }
    }
    r.slab = s;
    return r;
}

fn resolve_extended(pixel: vec2<u32>, m: Material, input: SurfaceInput, material: u32, vertex_color: vec4<f32>) -> Surface {
    let c = slabs[material];
    var layer = material_slab(c, m, input, material);
    var s: Surface;
    if HAS_GRAPH {
        let g = graph_eval(m, input, c.params, vertex_color, graph_defaults(m, layer.slab, layer.coat_roughness));
        s = graph_surface(g, m, input);
        layer.slab = graph_slab(g, layer.slab);
        layer.coat_roughness = saturate(g.clearcoat_roughness);
        if MATERIAL_KIND == MATERIAL_FOLIAGE {
            s.flags = SURFACE_FOLIAGE;
            s.metallic = 0.0;
        }
    } else {
        s = evaluate_kind(m, input);
    }
    var slab = layer.slab;
    // Features whose amount is zero at this pixel (textures, graphs) cost nothing here.
    if slab.clearcoat <= 0.0 {
        slab.features &= ~SLAB_CLEARCOAT;
    }
    if max(slab.sheen_color.r, max(slab.sheen_color.g, slab.sheen_color.b)) <= 0.0 {
        slab.features &= ~SLAB_SHEEN;
    }
    if abs(slab.anisotropy) < 0.004 {
        slab.features &= ~SLAB_ANISOTROPY;
    }
    // (With both, the weight is the subsurface's and transmission comes from the constants.)
    if slab.weight <= 0.0 && !(has_feature(slab, SLAB_SUBSURFACE) && has_feature(slab, SLAB_TRANSMISSION)) {
        slab.features &= ~(SLAB_SUBSURFACE | SLAB_TRANSMISSION);
    }
    if !SUBSTRATE || (slab.features & ~SLAB_GRAPH) == 0u {
        return s;
    }
    if has_feature(slab, SLAB_CLEARCOAT) {
        // The G-buffer keeps the top interface (Lumen traces its reflections); the base layer's
        // normal and roughness move to the slab texel.
        slab.base_normal = s.normal;
        slab.base_roughness = s.roughness;
        s.normal = layer.coat_normal;
        s.roughness = clamp(layer.coat_roughness, 0.02, 1.0);
    } else if has_feature(slab, SLAB_ANISOTROPY) {
        let angle = c.anisotropy_rotation;
        slab.tangent = normalize(input.tangent * cos(angle) + input.bitangent * sin(angle));
    }
    textureStore(gbuffer_substrate, pixel, encode_slab(slab));
    s.flags = SURFACE_SUBSTRATE;
    return s;
}
