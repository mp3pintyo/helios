// Fog screen space scattering (UE 5.8's experimental Fog Screen Space Scattering): the light
// that fog scatters only slightly out of a pixel's view ray still reaches the camera, from a
// little beside the pixel — the glow around lamps, candles and lit windows in fog, and the
// softening of everything seen through dense fog. The volumetric fog (fog.wgsl) only removes it:
// the lighting applies `color * transmittance + in-scattering`, where the in-scattering is the
// sun's, the local lights' and the ambient light scattered *once* at froxel resolution, and the
// light coming from the visible surfaces is simply extinguished.
//
// Per output pixel, from the integrated fog volume at the pixel's depth: optical depth tau
// (transmittance T = exp(-tau)) and in-scattering S. The colour behind the fog, seen directly, is
// D = color - S = L * T. Light the fog scatters within the angle the blur below covers (a
// fraction F of what it scatters, from the Henyey-Greenstein phase function), scattered any
// number of times and attenuated along the rest of the ray, reaches the camera as
//     L * T * (exp(F * albedo * tau) - 1) = D * (exp(F * albedo * tau) - 1)
// (orders k of forward scattering add (F albedo tau)^k / k! L T). It is at most what the fog
// extinguished, L (1 - T): the scattered energy is spread, not created. The intensity setting
// scales it (default 1), never beyond all the scattered light, D * (exp(albedo * tau) - 1).
//
//   prefilter  : that scattered radiance at half resolution (alpha: its luminance times tau, for
//                the occlusion test in the composite)
//   downsample : a 13-tap pyramid down to 1/256 (as bloom.wgsl)
//   upsample   : tent upsampling, adding each level with its weight: the share of the phase
//                function's scattered light between that level's angular radius and the previous
//                one's, so the summed kernel follows the phase function's forward lobe
//   composite  : color + the spread light. A pixel only receives what scattered in front of
//                its own surface: scaled by min(1, tau_pixel / mean tau of the light's sources),
//                a near post in fog does not glow with the haze behind it.

#include "common.wgsl"
#include "fog_common.wgsl"

struct FogScattering {
    params: vec4<f32>,        // x: F, the share of scattered light the blur reaches, y: intensity
}

struct Level {
    weight: vec4<f32>,        // x: this level's weight, y: 1 = coarsest (no lower level)
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> fog: Fog;
@group(0) @binding(2) var<uniform> scattering: FogScattering;
@group(0) @binding(3) var<uniform> level: Level;
@group(0) @binding(4) var color_input: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_2d<f32>;
@group(0) @binding(6) var fog_volume: texture_3d<f32>;
@group(0) @binding(7) var linear_sampler: sampler;
@group(0) @binding(8) var source: texture_2d<f32>;
@group(0) @binding(9) var lower: texture_2d<f32>;
@group(0) @binding(10) var destination: texture_storage_2d<rgba16float, write>;

// The integrated fog between the camera and output pixel `pixel`'s surface: in-scattered
// radiance (rgb) and transmittance (a), as the lighting applies it.
fn fog_at(pixel: vec2<u32>) -> vec4<f32> {
    let internal = min(vec2<u32>((vec2<f32>(pixel) + 0.5) * frame.upscale.xy), vec2<u32>(frame.viewport.xy) - 1u);
    let depth = frame.camera_position.w / max(textureLoad(depth_texture, internal, 0).r, 1e-9);
    let uv = (vec2<f32>(pixel) + 0.5) / frame.upscale.zw;
    let w = fog_depth_slice(depth) - 0.5 / f32(textureDimensions(fog_volume).z);
    return textureSampleLevel(fog_volume, linear_sampler, vec3<f32>(uv, saturate(w)), 0.0);
}

fn optical_depth(v: vec4<f32>) -> f32 {
    return -log(clamp(v.a, 1e-4, 1.0));
}

@compute @workgroup_size(8, 8)
fn prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if any(gid.xy >= size) {
        return;
    }
    let full = vec2<u32>(frame.upscale.zw);
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let pixel = min(gid.xy * 2u + vec2<u32>(i % 2u, i / 2u), full - 1u);
        let v = fog_at(pixel);
        let tau = optical_depth(v);
        let direct = max(textureLoad(color_input, pixel, 0).rgb - v.rgb, vec3<f32>(0.0));
        // `intensity` scales the share within reach, up to all of the scattered light.
        let albedo_tau = fog.albedo.rgb * tau;
        let share = min(scattering.params.y * (exp(scattering.params.x * albedo_tau) - 1.0), exp(albedo_tau) - 1.0);
        let scattered = direct * share;
        sum += vec4<f32>(scattered, luminance(scattered) * tau);
    }
    let result = sum * 0.25;
    textureStore(destination, gid.xy, select(result, vec4<f32>(0.0), any(result != result)));
}

fn fetch(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(source, linear_sampler, uv, 0.0);
}

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    let a = fetch(uv + t * vec2<f32>(-2.0, -2.0));
    let b = fetch(uv + t * vec2<f32>(0.0, -2.0));
    let c = fetch(uv + t * vec2<f32>(2.0, -2.0));
    let d = fetch(uv + t * vec2<f32>(-2.0, 0.0));
    let e = fetch(uv);
    let f = fetch(uv + t * vec2<f32>(2.0, 0.0));
    let g = fetch(uv + t * vec2<f32>(-2.0, 2.0));
    let h = fetch(uv + t * vec2<f32>(0.0, 2.0));
    let i = fetch(uv + t * vec2<f32>(2.0, 2.0));
    let j = fetch(uv + t * vec2<f32>(-1.0, -1.0));
    let k = fetch(uv + t * vec2<f32>(1.0, -1.0));
    let l = fetch(uv + t * vec2<f32>(-1.0, 1.0));
    let m = fetch(uv + t * vec2<f32>(1.0, 1.0));
    let result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    textureStore(destination, gid.xy, max(result, vec4<f32>(0.0)));
}

// up[i] = weight_i * down[i] + tent(up[i + 1]); the coarsest level is its own weighted copy.
@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    var result = textureSampleLevel(source, linear_sampler, uv, 0.0) * level.weight.x;
    if level.weight.y < 0.5 {
        let t = 1.0 / vec2<f32>(textureDimensions(lower));
        var blurred = vec4<f32>(0.0);
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(-1.0, -1.0), 0.0);
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(0.0, -1.0), 0.0) * 2.0;
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(1.0, -1.0), 0.0);
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(-1.0, 0.0), 0.0) * 2.0;
        blurred += textureSampleLevel(lower, linear_sampler, uv, 0.0) * 4.0;
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(1.0, 0.0), 0.0) * 2.0;
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(-1.0, 1.0), 0.0);
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(0.0, 1.0), 0.0) * 2.0;
        blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(1.0, 1.0), 0.0);
        result += blurred / 16.0;
    }
    textureStore(destination, gid.xy, result);
}

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if any(gid.xy >= size) {
        return;
    }
    let input = textureLoad(color_input, gid.xy, 0);
    let color = input.rgb;
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let spread = textureSampleLevel(source, linear_sampler, uv, 0.0);
    // The light's sources lie at mean optical depth alpha / luminance; scattering behind this
    // pixel's surface is hidden by it.
    let tau = optical_depth(fog_at(gid.xy));
    let source_tau = spread.a / max(luminance(spread.rgb), 1e-8);
    let visible = saturate(tau / max(source_tau, 1e-6));
    let result = color + spread.rgb * visible;
    // Alpha passes through (TSR's sample weight, which photo mode sums by).
    textureStore(destination, gid.xy, vec4<f32>(select(result, color, any(result != result)), input.a));
}
