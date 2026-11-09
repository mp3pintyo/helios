// Volumetric fog parameters and froxel depth mapping, shared by fog.wgsl and the lighting.

struct Fog {
    media: vec4<f32>,    // x: extinction per metre at the base height, y: phase anisotropy g,
                         // z: ambient scale, w: noise amount
    albedo: vec4<f32>,   // rgb scattering albedo
    height: vec4<f32>,   // x: base height (m), y: exponential falloff per metre above it
    wind: vec4<f32>,     // xyz: noise offset (drift)
    range: vec4<f32>,    // x: near, y: far (view depth, metres)
    enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const FOG_HISTORY_BLEND: f32 = 0.08;
/// Froxels are ~16 pixels wide: sample the shadow map four levels coarser than a pixel would.
const FOG_SHADOW_LEVEL_BIAS: u32 = 4u;

// Slices are spaced exponentially in view depth between range.x and range.y.
fn fog_slice_depth(w: f32) -> f32 {
    return fog.range.x * pow(fog.range.y / fog.range.x, w);
}

fn fog_depth_slice(depth: f32) -> f32 {
    return log(max(depth, fog.range.x) / fog.range.x) / log(fog.range.y / fog.range.x);
}
