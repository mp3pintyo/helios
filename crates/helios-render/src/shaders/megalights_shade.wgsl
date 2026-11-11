// MegaLights shading (see megalights.rs): every pixel evaluates the shared BRDF for the samples
// of its 2x2 block - light, contribution weight, traced visibility - and writes its direct light
// demodulated (diffuse without the diffuse colour, specular divided by the environment BRDF), so
// the denoiser blurs lighting, never texture detail. Plus the debug views of the samples.

#include "common.wgsl"
#include "brdf.wgsl"
#include "lights.wgsl"
#include "megalights_surface.wgsl"
#include "megalights_common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(7) var<storage, read> samples: array<u32>;
@group(0) @binding(8) var raw_diffuse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var raw_specular: texture_storage_2d<rgba16float, write>;
// Debug views: the denoised lighting (modulated) and the HDR image they replace.
@group(0) @binding(10) var denoised: texture_2d<f32>;
@group(0) @binding(12) var hdr_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= vec2<u32>(frame.viewport.xy)) {
        return;
    }
    let c = ml_shade_samples(gid.xy, ml_surface(gid.xy));
    // Lights at a few centimetres can exceed half precision; keep the sums finite.
    textureStore(raw_diffuse, gid.xy, vec4<f32>(min(c.diffuse, vec3<f32>(60000.0)), 0.0));
    textureStore(raw_specular, gid.xy, vec4<f32>(min(c.specular, vec3<f32>(60000.0)), 0.0));
}

// DEBUG_MEGALIGHTS_SAMPLES: the four pixels of a block show its four samples' lights (a saturated
// colour per light, dark where the shadow ray was blocked). The picks change every frame and TSR
// averages them: one hue where a light dominates, mixtures where many compete.
// DEBUG_MEGALIGHTS_DENOISING: the lighting MegaLights adds, as this frame's samples
// give it (left half) and denoised (right half).
@compute @workgroup_size(8, 8)
fn debug(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = gid.xy;
    let s = ml_surface(pixel);
    if !s.valid {
        return;
    }
    var color = vec3<f32>(0.0);
    if frame.debug_view == DEBUG_MEGALIGHTS_SAMPLES {
        let base = ml_block_base(pixel / 2u);
        let k = (pixel.x & 1u) + 2u * (pixel.y & 1u);
        let word = samples[base + k];
        let header = samples[base + ML_HEADER];
        if (header & ML_VALID) != 0u && (word & 0xffffu) != ML_NO_LIGHT {
            let hue = f32(pcg_hash(word & 0xffffu) & 0xffffu) / 65536.0;
            let rgb = saturate(abs(fract(hue + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
            color = rgb * (0.1 + 0.9 * ml_sample_visibility(base, header, k));
        }
    } else {
        if pixel.x < size.x / 2u {
            let c = ml_shade_samples(pixel, s);
            color = s.diffuse_color * c.diffuse + ml_specular_albedo(s) * c.specular;
        } else {
            color = textureLoad(denoised, pixel, 0).rgb;
        }
        if pixel.x == size.x / 2u {
            color = vec3<f32>(1.0);
        }
    }
    textureStore(hdr_output, pixel, vec4<f32>(color, 1.0));
}
