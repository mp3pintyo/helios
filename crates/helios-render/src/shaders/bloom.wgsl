// Physically-plausible bloom: 13-tap downsample chain followed by a tent-filter upsample
// chain (Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare").

override FIRST_LEVEL: bool = false;

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var lower: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var destination: texture_storage_2d<rgba16float, write>;

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)));
}

fn fetch(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(source, linear_sampler, uv, 0.0).rgb;
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

    var result: vec3<f32>;
    if FIRST_LEVEL {
        // Karis average per 2x2 block suppresses fireflies (e.g. the sun disk).
        let g0 = (j + k + l + m) * 0.25;
        let g1 = (a + b + d + e) * 0.25;
        let g2 = (b + c + e + f) * 0.25;
        let g3 = (d + e + g + h) * 0.25;
        let g4 = (e + f + h + i) * 0.25;
        let w0 = karis_weight(g0) * 0.5;
        let w1 = karis_weight(g1) * 0.125;
        let w2 = karis_weight(g2) * 0.125;
        let w3 = karis_weight(g3) * 0.125;
        let w4 = karis_weight(g4) * 0.125;
        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    } else {
        result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    textureStore(destination, gid.xy, vec4<f32>(max(result, vec3<f32>(0.0)), 1.0));
}

@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(destination);
    if any(gid.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let t = 1.0 / vec2<f32>(textureDimensions(lower));
    var blurred = vec3<f32>(0.0);
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb * 1.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(0.0, -1.0), 0.0).rgb * 2.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(1.0, -1.0), 0.0).rgb * 1.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(-1.0, 0.0), 0.0).rgb * 2.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv, 0.0).rgb * 4.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(1.0, 0.0), 0.0).rgb * 2.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(-1.0, 1.0), 0.0).rgb * 1.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(0.0, 1.0), 0.0).rgb * 2.0;
    blurred += textureSampleLevel(lower, linear_sampler, uv + t * vec2<f32>(1.0, 1.0), 0.0).rgb * 1.0;
    let current = textureSampleLevel(source, linear_sampler, uv, 0.0).rgb;
    textureStore(destination, gid.xy, vec4<f32>(current + blurred / 16.0, 1.0));
}
