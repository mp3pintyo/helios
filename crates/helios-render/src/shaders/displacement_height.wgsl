// The height a displaced surface moves by (displacement.wgsl), read from a material texture layer.
// Requires displacement.wgsl and the bindings `materials: array<Material, MAX_MATERIALS>`
// (uniform), `material_data: texture_2d_array<f32>` and `material_sampler: sampler`.
//
// The height is a pure function of a point's undisplaced attributes and the main camera, so the
// tessellator, the shadow page tessellator and the G-buffer resolve displace a point identically,
// and so do both triangles of a shared edge (no cracks). It is sampled with the material's own
// mapping (mesh UVs with its texture transform, or world / object triplanar), filtered for the
// size of a dice segment at the point (UE picks the displacement mip from the mesh's UV density
// too: the tessellator has no screen derivatives). Image layers are read from their 512² array
// layer: the dice rate rarely asks for finer heights than that.

// TextureMapping values (MAPPING_* in material.wgsl).
const DISPLACEMENT_MAPPING_WORLD: u32 = 1u;
const DISPLACEMENT_MAPPING_OBJECT: u32 = 2u;
// Texels across a material texture array layer.
const DISPLACEMENT_LAYER_TEXELS: f32 = 512.0;

// A point of a displaced surface before displacement.
struct DisplacementInput {
    object: vec3<f32>,        // object-space position
    world: vec3<f32>,         // world-space position (instance transform, no wind)
    normal: vec3<f32>,        // unit object-space normal
    world_normal: vec3<f32>,  // unit world-space normal
    uv: vec2<f32>,            // mesh UV set 0
}

fn displacement_texel(source: vec4<u32>, uv: vec2<f32>, lod: f32) -> f32 {
    return textureSampleLevel(material_data, material_sampler, uv, source.x, lod)[source.y];
}

// Height in [0, 1] of instance material `material` (instance scale `max_scale`) at `p`.
fn displacement_height(material: u32, max_scale: f32, p: DisplacementInput) -> f32 {
    let d = tess.materials[material];
    let m = materials[material];
    // World size of a dice segment at the point.
    let segment = max(distance(p.world, tess.eye.xyz), tess.dice.z) * (tess.dice.x / tess.dice.y);
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
fn displacement_offset(material: u32, height: f32) -> f32 {
    let params = tess.materials[material].params;
    return (height - params.y) * params.x;
}
