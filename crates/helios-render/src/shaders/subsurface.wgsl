// Screen-space subsurface scattering (subsurface.rs; UE's subsurface profile, Jimenez et al.'s
// separable SSS). The deferred lighting writes the diffuse light of subsurface pixels apart,
// demodulated (subsurface_store.wgsl); a horizontal compute pass and a vertical full-screen pass
// blur it with the material's diffusion profile, and the vertical pass multiplies the diffuse
// colour back in and adds the result to the HDR image (additive blending), so specular light
// and texture detail stay sharp.
//
// The profile per channel is a sum of two Gaussians approximating Burley's normalised diffusion
// (a narrow peak and a wide tail), its width the material's scattering distance times its
// per-channel scattering colour (skin: red scatters furthest). Taps are spaced over the
// profile's reach in pixels, jittered per pixel and frame (TSR averages them), and skip pixels
// of other surfaces: no slab texel, or a depth step larger than the scattering can bridge.

#include "common.wgsl"
#include "substrate.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_substrate: texture_2d<u32>;
@group(0) @binding(3) var<uniform> slabs: array<MaterialSlab, MAX_MATERIALS>;
@group(0) @binding(4) var light_in: texture_2d<f32>;
@group(0) @binding(5) var light_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(7) var gbuffer_motion: texture_2d<f32>;

// Taps on each side of the centre, and the widest blur in pixels.
const TAPS: i32 = 7;
const MAX_REACH: f32 = 96.0;

struct Scatter {
    valid: bool,
    depth: f32,         // linear depth (m)
    radius: f32,        // scattering distance (m)
    width: vec3<f32>,   // profile width per channel (pixels)
}

fn scatter_at(pixel: vec2<u32>) -> Scatter {
    var s: Scatter;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 || surface_flags_of(textureLoad(gbuffer_motion, pixel, 0).w) < SURFACE_SUBSTRATE_FLAG {
        return s;
    }
    let slab = decode_slab(textureLoad(gbuffer_substrate, pixel, 0));
    if !has_feature(slab, SLAB_SUBSURFACE) {
        return s;
    }
    let c = slabs[slab.material];
    // World size of a pixel at this depth (same NDC depth: same view depth).
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let here = world_from_depth(uv, depth, frame.inv_view_proj);
    let next = world_from_depth(uv + vec2<f32>(frame.viewport.z, 0.0), depth, frame.inv_view_proj);
    let pixels_per_metre = 1.0 / max(distance(here, next), 1e-6);
    s.valid = true;
    s.depth = frame.camera_position.w / depth;
    s.radius = c.subsurface_radius;
    s.width = c.subsurface_radius * c.subsurface_color * pixels_per_metre;
    return s;
}

// The diffusion profile at distance `x` (pixels) for per-channel widths `d`.
fn profile(x: f32, d: vec3<f32>) -> vec3<f32> {
    let a = 0.4 * d + 1e-3;
    let b = 1.4 * d + 1e-3;
    let x2 = x * x;
    return 0.6 * exp(-x2 / (2.0 * a * a)) / a + 0.4 * exp(-x2 / (2.0 * b * b)) / b;
}

fn diffuse(pixel: vec2<u32>, axis: vec2<i32>) -> vec3<f32> {
    let center = scatter_at(pixel);
    let light = textureLoad(light_in, pixel, 0).rgb;
    let reach = min(3.0 * 1.4 * max(center.width.r, max(center.width.g, center.width.b)), MAX_REACH);
    if reach < 1.0 {
        return light;
    }
    let size = vec2<i32>(frame.viewport.xy);
    let step = reach / f32(TAPS);
    let jitter = interleaved_gradient_noise(vec2<f32>(pixel), frame.frame_index);
    var sum = light * profile(0.0, center.width);
    var weights = profile(0.0, center.width);
    for (var i = 1; i <= TAPS; i++) {
        for (var side = -1; side <= 1; side += 2) {
            let offset = i32(round((f32(i) - jitter) * step)) * side;
            let tap = vec2<i32>(pixel) + axis * offset;
            if any(tap < vec2<i32>(0)) || any(tap >= size) || offset == 0 {
                continue;
            }
            let t = scatter_at(vec2<u32>(tap));
            if !t.valid || abs(t.depth - center.depth) > 3.0 * center.radius {
                continue;
            }
            let w = profile(f32(abs(offset)), center.width);
            sum += textureLoad(light_in, vec2<u32>(tap), 0).rgb * w;
            weights += w;
        }
    }
    return sum / weights;
}

@compute @workgroup_size(8, 8)
fn horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= vec2<u32>(frame.viewport.xy)) || !scatter_at(gid.xy).valid {
        return;
    }
    textureStore(light_out, gid.xy, vec4<f32>(diffuse(gid.xy, vec2<i32>(1, 0)), 1.0));
}

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// Vertical pass: the diffused light times the diffuse colour, added to the lit image.
@fragment
fn vertical(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(position.xy);
    if !scatter_at(pixel).valid {
        discard;
    }
    let albedo_metal = textureLoad(gbuffer_albedo, pixel, 0);
    let diffuse_color = albedo_metal.rgb * albedo_metal.rgb * (1.0 - albedo_metal.a);
    return vec4<f32>(diffuse(pixel, vec2<i32>(0, 1)) * diffuse_color, 0.0);
}
