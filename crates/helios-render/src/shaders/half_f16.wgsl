// half.wgsl's f16 variant (see there); `shader::create_module` moves the `enable` to the top.
enable f16;

alias hf = f16;
alias hf2 = vec2<f16>;
alias hf3 = vec3<f16>;
alias hf4 = vec4<f16>;
const HF_MAX: f32 = 65504.0;
