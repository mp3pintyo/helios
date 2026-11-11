// The MegaLights volume (megalights_volume.rs): shadowed local light in a camera-aligned froxel
// volume, shared by volumetric fog and translucent surfaces. Requires common.wgsl and
// `ml_volume: MlVolume`; ml_volume_light also `ml_volume_sh0`, `ml_volume_sh1` (texture_3d<f32>)
// and a linear sampler `sampler_linear` passed in.
//
// Per froxel the volume holds the estimate of the sum of the lights' shadowed radiance scales
// (sh0.rgb: what sample_light returns, times visibility, summed over the froxel's lights) and its
// luminance-weighted direction (sh1.xyz: the sum of luminance x unit vector towards each light):
// SH bands 0 and 1 of the incident light, with one colour.

struct MlVolume {
    // x: near, y: far (view depth of the first and last slice's bounds), z: history blend,
    // w: 1 while the volume is on.
    range: vec4<f32>,
}

// View depth of the slice coordinate `w` (0-1), spaced exponentially as the fog's.
fn ml_volume_depth(w: f32) -> f32 {
    return ml_volume.range.x * pow(ml_volume.range.y / ml_volume.range.x, w);
}

fn ml_volume_slice(depth: f32) -> f32 {
    return log(max(depth, ml_volume.range.x) / ml_volume.range.x) / log(ml_volume.range.y / ml_volume.range.x);
}

// Shadowed local light at screen `uv` and view depth `depth` as a directional part (`radiance`
// x `directional` from `direction`) and an ambient rest (`radiance` x (1 - `directional`),
// arriving from all around).
struct MlVolumeLight {
    radiance: vec3<f32>,
    direction: vec3<f32>,
    directional: f32,
}

fn ml_volume_light(uv: vec2<f32>, depth: f32, sampler_linear: sampler) -> MlVolumeLight {
    var v: MlVolumeLight;
    let p = vec3<f32>(uv, ml_volume_slice(depth));
    v.radiance = max(textureSampleLevel(ml_volume_sh0, sampler_linear, p, 0.0).rgb, vec3<f32>(0.0));
    let d = textureSampleLevel(ml_volume_sh1, sampler_linear, p, 0.0).xyz;
    let length_d = length(d);
    v.direction = select(vec3<f32>(0.0, 1.0, 0.0), d / max(length_d, 1e-9), length_d > 1e-9);
    v.directional = saturate(length_d / max(luminance(v.radiance), 1e-9));
    return v;
}
