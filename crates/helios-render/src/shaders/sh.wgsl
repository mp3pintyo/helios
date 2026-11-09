// Order-2 (9 coefficient) real spherical harmonics.

fn sh_basis(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
        1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y),
    );
}

// Irradiance E(n) from SH-projected radiance (Ramamoorthi & Hanrahan cosine-lobe convolution).
fn sh_irradiance(sh: array<vec4<f32>, 9>, n: vec3<f32>) -> vec3<f32> {
    var y = sh_basis(n);
    var coefficients = sh;
    var band = array<f32, 9>(PI, 2.0 * PI / 3.0, 2.0 * PI / 3.0, 2.0 * PI / 3.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0);
    var e = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k++) {
        e += coefficients[k].rgb * y[k] * band[k];
    }
    return max(e, vec3<f32>(0.0));
}
