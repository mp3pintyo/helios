// Final composite: bloom, exposure, ACES filmic tone mapping, vignette and dithering.

#include "common.wgsl"

// True when the swapchain format applies the sRGB transfer function itself.
override OUTPUT_SRGB: bool = true;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var<storage, read> exposure: array<f32, 4>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

// Stephen Hill's fit of the ACES RRT + ODT.
fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    let input = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    let output = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602),
    );
    let v = input * color;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return saturate(output * (a / b));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn is_unlit_debug_view(view: u32) -> bool {
    return !is_lit_view(view);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    var color = textureLoad(hdr, pixel, 0).rgb;

    if is_unlit_debug_view(frame.debug_view) {
        color = saturate(color);
    } else {
        color += textureSampleLevel(bloom, linear_sampler, in.uv, 0.0).rgb * frame.bloom_intensity;
        color = aces_fitted(color * exposure[0]);
        let centered = in.uv - 0.5;
        color *= mix(1.0, 0.72, smoothstep(0.15, 0.85, dot(centered, centered) * 1.6));
    }

    if !OUTPUT_SRGB {
        color = linear_to_srgb(color);
    }
    // Triangular dither hides 8-bit banding in smooth gradients like the sky.
    let noise = interleaved_gradient_noise(in.position.xy, frame.frame_index) - interleaved_gradient_noise(in.position.xy + 31.0, frame.frame_index);
    color += noise / 255.0;
    return vec4<f32>(color, 1.0);
}
