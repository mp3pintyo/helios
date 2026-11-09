// Gradient noise and footprint-filtered fBm for procedural materials.
// Requires common.wgsl (pcg_hash).

fn noise_gradient(cell: vec3<i32>) -> vec3<f32> {
    let h = pcg_hash(bitcast<u32>(cell.x) ^ pcg_hash(bitcast<u32>(cell.y) ^ pcg_hash(bitcast<u32>(cell.z))));
    return vec3<f32>(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) / 511.5 - 1.0;
}

// Perlin-style gradient noise, roughly in [-1, 1].
fn gradient_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = dot(noise_gradient(i), f);
    let n100 = dot(noise_gradient(i + vec3<i32>(1, 0, 0)), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = dot(noise_gradient(i + vec3<i32>(0, 1, 0)), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = dot(noise_gradient(i + vec3<i32>(1, 1, 0)), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = dot(noise_gradient(i + vec3<i32>(0, 0, 1)), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = dot(noise_gradient(i + vec3<i32>(1, 0, 1)), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = dot(noise_gradient(i + vec3<i32>(0, 1, 1)), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = dot(noise_gradient(i + vec3<i32>(1, 1, 1)), f - vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// fBm whose octaves fade out once their wavelength approaches the pixel footprint,
// which keeps procedural detail from aliasing at a distance.
fn fbm(p: vec3<f32>, octaves: u32, footprint: f32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    for (var o = 0u; o < octaves; o++) {
        let fade = 1.0 - smoothstep(0.25, 0.5, frequency * footprint);
        if fade <= 0.0 {
            break;
        }
        sum += gradient_noise(p * frequency + f32(o) * 17.17) * amplitude * fade;
        amplitude *= 0.5;
        frequency *= 2.02;
    }
    return sum;
}
