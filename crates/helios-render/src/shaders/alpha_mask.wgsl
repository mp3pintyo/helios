// Alpha test of alpha-masked materials (alpha_mask.rs), shared by every rasteriser that writes
// visibility or depth: the hardware and software visibility rasterisers and both virtual
// shadow map rasterisers. Requires common.wgsl, geometry.wgsl and, in the including shader,
//   materials: array<Material, MAX_MATERIALS> (uniform), alpha_masks: texture_2d_array<f32>,
//   alpha_mask_sampler: sampler                      (bindings 40, 41 and 42 by convention).

// A material's alpha test, packed for a flat varying: the bits of Material.flags above
// MATERIAL_ALPHA_CUTOFF_SHIFT (mask layer + 1, cutoff). 0 for opaque materials.
fn alpha_mask_of(material: u32) -> u32 {
    return materials[material].flags & ~((1u << MATERIAL_ALPHA_CUTOFF_SHIFT) - 1u);
}

fn is_alpha_masked(mask: u32) -> bool {
    return (mask >> MATERIAL_ALPHA_MASK_SHIFT) != 0u;
}

// The mask is sampled with the texture coordinates of the material's base colour: its UV set,
// transform and tiling (affine, so transforming the corners equals transforming per pixel; the
// sampler's wrap modes are not applied: alpha tests repeat).
fn alpha_mask_vertex_uv(material: u32, c: Cluster, base: u32, index: u32) -> vec2<f32> {
    let t = materials[material].uv_texture;
    var uv = vertex_uv(c, base, index);
    if t.uv_set == 1u {
        uv = vertex_uv1(c, base, index, uv);
    }
    return uv_transform_point(t, uv);
}

// True where the surface survives the alpha test. `ddx` / `ddy` are the UV derivatives per
// pixel (or shadow texel); they select the mip, so explicit derivatives work in compute too.
fn alpha_test(mask: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> bool {
    let layer = (mask >> MATERIAL_ALPHA_MASK_SHIFT) - 1u;
    let cutoff = f32((mask >> MATERIAL_ALPHA_CUTOFF_SHIFT) & 0xffu) / 255.0;
    return textureSampleGrad(alpha_masks, alpha_mask_sampler, uv, layer, ddx, ddy).r >= cutoff;
}

// UV derivatives of an affinely interpolated triangle per unit of its 2D (pixel or texel)
// positions p0..p2: d(uv)/dx and d(uv)/dy, for the compute rasterisers. Their triangles are
// a few pixels small, so perspective is negligible across them.
fn triangle_uv_gradients(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>) -> mat2x2<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let duv1 = uv1 - uv0;
    let duv2 = uv2 - uv0;
    let inv_det = 1.0 / (e1.x * e2.y - e1.y * e2.x);
    return mat2x2<f32>((duv1 * e2.y - duv2 * e1.y) * inv_det, (duv2 * e1.x - duv1 * e2.x) * inv_det);
}
