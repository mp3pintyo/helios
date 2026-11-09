// MegaLights (megalights.rs): a G-buffer pixel as its passes see it - the same surface terms as
// lighting.wgsl. Requires common.wgsl, brdf.wgsl and these bindings: frame: Frame, depth_texture,
// gbuffer_albedo, gbuffer_normal, gbuffer_motion (texture_2d<f32>).
//
// The shading pass's slab variant (substrate_read.wgsl) also reads the pixel's Substrate slab and
// evaluates its lobes. Demodulation keeps using the plain terms (diffuse colour, the G-buffer's
// environment BRDF) in every pass: they only scale what the denoiser blurs, and must match
// between the shading and the denoiser's final modulation.

#include "substrate_read_off.wgsl"

struct MlSurface {
    valid: bool,
    world: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    n_dot_v: f32,
    linear_depth: f32,
    uv: vec2<f32>,
    albedo: vec3<f32>,
    diffuse_color: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    a: f32,
    // Foliage transmits light arriving from behind (lighting.wgsl's wrap term).
    wrap: f32,
    // The Substrate slab (slab variant only).
    substrate: bool,
    slab: SlabShading,
}

fn ml_surface(pixel: vec2<u32>) -> MlSurface {
    var s: MlSurface;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    s.valid = depth > 0.0;
    if !s.valid {
        return s;
    }
    s.uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    s.world = world_from_depth(s.uv, depth, frame.inv_view_proj);
    s.linear_depth = frame.camera_position.w / depth;
    s.v = normalize(frame.camera_position.xyz - s.world);
    let albedo_metal = textureLoad(gbuffer_albedo, pixel, 0);
    let normal_rough = textureLoad(gbuffer_normal, pixel, 0);
    s.n = normalize(normal_rough.xyz);
    s.n_dot_v = max(dot(s.n, s.v), 1e-4);
    s.albedo = albedo_metal.rgb * albedo_metal.rgb;
    s.diffuse_color = s.albedo * (1.0 - albedo_metal.a);
    s.f0 = mix(vec3<f32>(0.04), s.albedo, albedo_metal.a);
    s.roughness = clamp(normal_rough.w, 0.045, 1.0);
    s.a = s.roughness * s.roughness;
    let flags = surface_flags_of(textureLoad(gbuffer_motion, pixel, 0).w);
    s.wrap = select(0.0, 0.35, flags == 1u);
    s.substrate = SUBSTRATE_LIGHTING && flags >= SURFACE_SUBSTRATE_FLAG;
    if s.substrate {
        let slab = load_slab(pixel);
        s.slab = shade_slab(slab, slab_constants(slab.material), s.n, s.roughness, s.albedo, albedo_metal.a, s.v);
    }
    return s;
}

// What the lighting multiplies demodulated specular light by (UE's MegaLights demodulates by
// the environment BRDF too): smooth across texture detail, never zero.
fn ml_specular_albedo(s: MlSurface) -> vec3<f32> {
    return max(env_brdf(s.f0, s.roughness, s.n_dot_v), vec3<f32>(0.01));
}
