// Half-precision arithmetic where the device has it (WebGPU's `shader-f16`, wgpu's SHADER_F16).
//
// A pass whose math tolerates 11-bit mantissas writes it in `hf`, `hf2`, `hf3` and `hf4`. They are
// f32 here; on a device with f16 shaders `shader::create_module` swaps this file for half_f16.wgsl,
// which makes them f16 (and enables the extension). One variant is ever compiled per device.
// Values must stay within f16's range (±65504; 2^-14 is the smallest normal): exposed colours,
// weights, normalised offsets — not raw scene radiance squared, depths or world positions.

alias hf = f32;
alias hf2 = vec2<f32>;
alias hf3 = vec3<f32>;
alias hf4 = vec4<f32>;
// The largest finite value of `hf` this variant uses for clamping (f16's, so both variants clamp
// alike).
const HF_MAX: f32 = 65504.0;
