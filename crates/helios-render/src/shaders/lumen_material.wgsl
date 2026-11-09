// Average material albedo for Lumen's surface caches: base colour times the texture's 1x1 mip.
// Requires common.wgsl and these bindings: materials: array<Material, MAX_MATERIALS> (uniform),
// material_albedo: texture_2d_array<f32>, material_sampler: sampler.

const MATERIAL_GLASS: u32 = 5u;

// Emissive surfaces (glass, which the surface caches light with transmitted daylight, and
// self-luminous materials) are flagged with a negative albedo.
fn material_albedo_average(index: u32) -> vec3<f32> {
    let m = materials[index];
    var albedo = m.base_color.rgb;
    var layer = m.texture;
    if m.kind == 2u {
        layer = m.layers & 0xffu; // terrain: the first (dominant) layer
        if layer == 255u {
            layer = 0xffffffffu;
        }
    }
    if layer != 0xffffffffu {
        // The 1x1 mip is the texture's average colour.
        albedo *= textureSampleLevel(material_albedo, material_sampler, vec2<f32>(0.5), layer, 16.0).rgb;
    }
    albedo = clamp(albedo, vec3<f32>(0.02), vec3<f32>(0.95));
    if m.kind == MATERIAL_GLASS || m.emissive.w > 0.0 {
        return -albedo;
    }
    return albedo;
}
