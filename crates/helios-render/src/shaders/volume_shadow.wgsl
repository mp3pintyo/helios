// Shadows of heterogeneous volumes (volume_shadow.rs): per voxel of an emitter's region of the
// transmittance atlas, the medium's transmittance from the voxel to the box's boundary towards
// the sun (r) and towards the fire light the emitter drives (g), 16 steps through the density
// grid each. The driven light's own fire is its source: hot gas (temperature) does not occlude
// it, nor does anything within its source radius.

#include "common.wgsl"

const STEPS: u32 = 16u;

struct ShadowBuild {
    origin: vec4<f32>,    // xyz box minimum, w extinction per metre at density 1
    size: vec4<f32>,      // xyz box extent
    offset: vec4<u32>,    // xyz the region's first voxel, w 1 if the emitter drives a light
    voxels: vec4<u32>,    // xyz the region's voxels
    light: vec4<f32>,     // xyz the driven light's position, w its source radius
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> params: ShadowBuild;
@group(0) @binding(2) var state: texture_3d<f32>;
@group(0) @binding(3) var linear_clamp: sampler;
@group(0) @binding(4) var atlas_out: texture_storage_3d<rgba8unorm, write>;

// Distance from `p` (inside the box) to its boundary along `dir`.
fn box_exit(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let a = (params.origin.xyz - p) * inv;
    let b = (params.origin.xyz + params.size.xyz - p) * inv;
    let far = max(a, b);
    return max(min(min(far.x, far.y), far.z), 0.0);
}

// Transmittance from `p` along `dir` over `span` metres; `source` skips hot (emitting) gas.
fn transmittance(p: vec3<f32>, dir: vec3<f32>, span: f32, source: bool) -> f32 {
    let dt = span / f32(STEPS);
    var optical_depth = 0.0;
    for (var i = 0u; i < STEPS; i++) {
        let q = p + dir * ((f32(i) + 0.5) * dt);
        let s = textureSampleLevel(state, linear_clamp, (q - params.origin.xyz) / params.size.xyz, 0.0);
        var density = max(s.x, 0.0);
        if source {
            density *= 1.0 - smoothstep(0.05, 0.3, s.y);
        }
        optical_depth += density;
    }
    return exp(-optical_depth * dt * params.origin.w);
}

@compute @workgroup_size(4, 4, 4)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= params.voxels.xyz) {
        return;
    }
    let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(params.voxels.xyz);
    let p = params.origin.xyz + uvw * params.size.xyz;
    let sun = normalize(frame.sun_direction.xyz);
    var sun_t = 1.0;
    if sun.y > -0.05 {
        sun_t = transmittance(p, sun, box_exit(p, sun), false);
    }
    var light_t = 1.0;
    if params.offset.w != 0u {
        let to_light = params.light.xyz - p;
        let d = length(to_light);
        if d > params.light.w {
            let dir = to_light / d;
            light_t = transmittance(p, dir, min(box_exit(p, dir), d - params.light.w), true);
        }
    }
    textureStore(atlas_out, params.offset.xyz + gid, vec4<f32>(sun_t, light_t, 0.0, 1.0));
}
