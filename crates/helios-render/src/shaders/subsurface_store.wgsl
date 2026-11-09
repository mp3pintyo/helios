// The deferred lighting's slab variant: subsurface pixels write their diffuse light here,
// demodulated (without the diffuse colour) and attenuated by the fog in front of them, for the
// screen-space diffusion of subsurface.wgsl, which adds it back blurred and modulated.
// Replaces subsurface_store_off.wgsl.

@group(1) @binding(2) var subsurface_light: texture_storage_2d<rgba16float, write>;

fn store_subsurface(pixel: vec2<u32>, light: vec3<f32>) {
    textureStore(subsurface_light, pixel, vec4<f32>(min(light, vec3<f32>(60000.0)), 1.0));
}
