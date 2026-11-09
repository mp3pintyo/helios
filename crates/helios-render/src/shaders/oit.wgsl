// Composites heterogeneous volumes and order-independent particles over the lit scene
// (oit.rs). Twice per frame, once per bucket: `fs_back` before the translucency pass, with what
// lies behind the nearest translucent surface (so glass and water refract and tint it),
// `fs_front` after it, with what lies in front. In each bucket the particles (weighted blended
// OIT: average colour x coverage, McGuire and Bavoil 2013) go over the volume, which goes over
// the scene: premultiplied "over" with `One, OneMinusSrcAlpha` blending.
//
// `fs_front` also writes TSR's reactive mask into the motion target's z (UE's responsive AA for
// translucency): where particles or volumes cover a pixel the history is trusted less, so
// sparks and flames don't smear. In debug view 21 it shows particle overdraw (red to yellow)
// and volume march steps (blue to cyan) instead.

#include "common.wgsl"

const OIT_COLOR_SCALE: f32 = 16.0;
const DEBUG_VOLUMES: u32 = 21u;
// How much a volume's opacity makes a pixel reactive: it has a temporal filter of its own.
const VOLUME_REACTIVITY: f32 = 0.5;

struct Composite {
    volume_rect: vec4<u32>,  // pixels the volume textures hold this frame (min xy, max xy)
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> composite: Composite;
@group(0) @binding(2) var accum: texture_2d<f32>;
@group(0) @binding(3) var reveal: texture_2d<f32>;
@group(0) @binding(4) var volume: texture_2d<f32>;
// fs_front: the other bucket (reactive mask), the albedo (debug view).
@group(0) @binding(5) var other_reveal: texture_2d<f32>;
@group(0) @binding(6) var other_volume: texture_2d<f32>;
@group(0) @binding(7) var gbuffer_albedo: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

// Premultiplied colour and coverage of the particles.
fn particles(pixel: vec2<i32>, accum_texture: texture_2d<f32>, reveal_texture: texture_2d<f32>) -> vec4<f32> {
    let a = textureLoad(accum_texture, pixel, 0);
    let coverage = 1.0 - textureLoad(reveal_texture, pixel, 0).r;
    return vec4<f32>(a.rgb * OIT_COLOR_SCALE / max(a.a, 1e-5) * coverage, coverage);
}

// In-scattered radiance and transmittance of the volumes.
fn volume_at(pixel: vec2<i32>, source: texture_2d<f32>) -> vec4<f32> {
    let r = vec4<i32>(composite.volume_rect);
    if any(pixel < r.xy) || any(pixel >= r.zw) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return textureLoad(source, pixel, 0);
}

fn over(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(p.rgb + (1.0 - p.a) * v.rgb, 1.0 - (1.0 - p.a) * v.a);
}

@fragment
fn fs_back(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    return over(particles(pixel, accum, reveal), volume_at(pixel, volume));
}

struct FrontOutput {
    @location(0) color: vec4<f32>,
    @location(1) reactive: vec4<f32>,
}

fn heat(x: f32) -> vec3<f32> {
    return vec3<f32>(saturate(x * 3.0), saturate(x * 3.0 - 1.0), saturate(x * 3.0 - 2.0));
}

@fragment
fn fs_front(@builtin(position) position: vec4<f32>) -> FrontOutput {
    let pixel = vec2<i32>(position.xy);
    let p = particles(pixel, accum, reveal);
    let v = volume_at(pixel, volume);
    var out: FrontOutput;
    out.color = over(p, v);
    let other_coverage = 1.0 - textureLoad(other_reveal, pixel, 0).r;
    let opacity = 1.0 - v.a * volume_at(pixel, other_volume).a;
    out.reactive = vec4<f32>(0.0, 0.0, saturate(max(max(p.a, other_coverage), opacity * VOLUME_REACTIVITY)), 0.0);
    if frame.debug_view == DEBUG_VOLUMES {
        // Overdraw: every particle fragment added 1 to the accumulated weight; march steps.
        let overdraw = textureLoad(accum, pixel, 0).a;
        let steps = v.r;
        let albedo = textureLoad(gbuffer_albedo, pixel, 0).rgb;
        let base = vec3<f32>(luminance(albedo * albedo) * 0.35);
        var color = base + heat(overdraw / 24.0);
        color += vec3<f32>(0.0, 0.6, 1.0) * saturate(steps / 96.0) + vec3<f32>(0.0, 0.4, 0.0) * saturate(steps / 192.0 - 0.5);
        out.color = vec4<f32>(color, 1.0);
    }
    return out;
}
