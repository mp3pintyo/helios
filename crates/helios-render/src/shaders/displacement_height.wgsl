// What a displaced surface moves by (displacement.wgsl): a height read from a material texture
// layer, or a material graph's height or vector.
// Requires displacement.wgsl and the bindings `materials: array<Material, MAX_MATERIALS>`
// (uniform), `material_data: texture_2d_array<f32>` and `material_sampler: sampler`.
//
// The displacement is a pure function of a point's undisplaced attributes, the main camera and
// the time, so the tessellator, the shadow page tessellator and the G-buffer resolve displace a
// point identically, and so do both triangles of a shared edge (no cracks). A texture height is
// sampled with the material's own mapping (mesh UVs with its texture transform, or world / object
// triplanar), filtered for the size of a dice segment at the point (UE picks the displacement mip
// from the mesh's UV density too: the tessellator has no screen derivatives). Image layers are
// read from their 512² array layer: the dice rate rarely asks for finer heights than that.
//
// A material whose graph drives the displacement (its Displacement or VectorDisplacement output)
// instead evaluates the graph's code (displacement_graphs_off.wgsl's stand-in, replaced where such
// graphs exist; the same pure-function rule holds). Heights are clamped to [0, 1] and vectors to
// unit length, so the culling bound stays a bound.

#include "displacement_graphs_off.wgsl"

// TextureMapping values (MAPPING_* in material.wgsl).
const DISPLACEMENT_MAPPING_WORLD: u32 = 1u;
const DISPLACEMENT_MAPPING_OBJECT: u32 = 2u;
// Texels across a material texture array layer.
const DISPLACEMENT_LAYER_TEXELS: f32 = 512.0;
// TessMaterial.source.x of a material without a height layer (NO_TEXTURE in material.wgsl).
const DISPLACEMENT_NO_LAYER: u32 = 0xffffffffu;

// A point of a displaced surface before displacement.
struct DisplacementInput {
    object: vec3<f32>,        // object-space position
    world: vec3<f32>,         // world-space position (instance transform, no wind)
    normal: vec3<f32>,        // unit object-space normal
    world_normal: vec3<f32>,  // unit world-space normal
    uv: vec2<f32>,            // mesh UV set 0
    time: f32,                // seconds (tess.time: this frame's, or the previous one's)
}

// The undisplaced point `object` (unit normal `normal`, UV `uv`) of `instance` at time `time`.
fn displacement_input(instance: Instance, object: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>, time: f32) -> DisplacementInput {
    let normal_matrix = mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz);
    return DisplacementInput(object, (instance.world * vec4<f32>(object, 1.0)).xyz, normal, normalize(normal_matrix * normal), uv, time);
}

// What a material displaces a point by: a height along the normal, or a vector.
struct DisplacementValue {
    height: f32,          // in [0, 1]: the point moves (height - centre) * magnitude along its normal
    vector: vec3<f32>,    // world space, length <= 1, in units of magnitude * instance scale
    is_vector: bool,
}

fn displacement_from_height(height: f32) -> DisplacementValue {
    return DisplacementValue(saturate(height), vec3<f32>(0.0), false);
}

fn displacement_from_vector(v: vec3<f32>) -> DisplacementValue {
    return DisplacementValue(0.0, v / max(length(v), 1.0), true);
}

fn displacement_texel(source: vec4<u32>, uv: vec2<f32>, lod: f32) -> f32 {
    return textureSampleLevel(material_data, material_sampler, uv, source.x, lod)[source.y];
}

// World size of a dice segment at a point (main view; every pass filters heights alike).
fn displacement_segment(world: vec3<f32>) -> f32 {
    return max(distance(world, tess.eye.xyz), tess.dice.z) * (tess.dice.x / tess.dice.y);
}

// The displacement of instance material `material` (instance scale `max_scale`) at `p`.
fn displacement_value(material: u32, max_scale: f32, p: DisplacementInput) -> DisplacementValue {
    let d = tess.materials[material];
    if d.source.z != 0u {
        return displacement_graph_value(d.source.z, material, d.source.w, max_scale, p);
    }
    return displacement_from_height(displacement_texture_height(d, material, max_scale, p));
}

// Height in [0, 1] of a texture-displaced material at `p`.
fn displacement_texture_height(d: TessMaterial, material: u32, max_scale: f32, p: DisplacementInput) -> f32 {
    if d.source.x == DISPLACEMENT_NO_LAYER {
        return d.params.y;
    }
    let m = materials[material];
    let segment = displacement_segment(p.world);
    if m.mapping == DISPLACEMENT_MAPPING_WORLD || m.mapping == DISPLACEMENT_MAPPING_OBJECT {
        let world = m.mapping == DISPLACEMENT_MAPPING_WORLD;
        let q = select(p.object, p.world, world) * m.uv_scale;
        let units = select(segment / max(max_scale, 1e-6), segment, world);
        let lod = log2(max(units * m.uv_scale * DISPLACEMENT_LAYER_TEXELS, 1e-6));
        var w = pow(abs(select(p.normal, p.world_normal, world)), vec3<f32>(4.0));
        w /= w.x + w.y + w.z;
        return (displacement_texel(d.source, q.zy, lod) * w.x) + (displacement_texel(d.source, q.xz, lod) * w.y)
            + (displacement_texel(d.source, q.xy, lod) * w.z);
    }
    let t = m.uv_texture;
    let repeats = max(length(vec2<f32>(t.u.x, t.v.x)), length(vec2<f32>(t.u.y, t.v.y)));
    let lod = log2(max((segment / max(max_scale, 1e-6)) * d.params.w * repeats * DISPLACEMENT_LAYER_TEXELS, 1e-6));
    return displacement_texel(d.source, uv_transform_point(t, p.uv), lod);
}

// Object-space distance along the normal for `height`.
fn displacement_distance(material: u32, height: f32) -> f32 {
    let params = tess.materials[material].params;
    return (height - params.y) * params.x;
}

// The object-space offset of the point `p` of `instance` displaced by `value`.
fn displacement_offset(instance: Instance, p: DisplacementInput, value: DisplacementValue) -> vec3<f32> {
    return p.normal * displacement_distance(instance.material, value.height);
}

// The displaced object-space position of the point `p` of `instance`.
fn displacement_position(instance: Instance, p: DisplacementInput) -> vec3<f32> {
    return p.object + displacement_offset(instance, p, displacement_value(instance.material, instance.max_scale, p));
}
