// Procedural night sky shared by the deferred sky and the translucent reflections.
// Needs common.wgsl (frame, pcg_hash) and atmosphere.wgsl, plus `transmittance_lut` and
// `lut_sampler` bound by the including shader.

fn star_hash(cell: vec3<i32>) -> vec4<f32> {
    let h = pcg_hash((u32(cell.x) * 73856093u) ^ pcg_hash((u32(cell.y) * 19349663u) ^ pcg_hash(u32(cell.z) * 83492791u)));
    let h2 = pcg_hash(h);
    let h3 = pcg_hash(h2);
    return vec4<f32>(f32(h & 0xffffu), f32(h >> 16u), f32(h2 & 0xffffu), f32(h3 & 0xffffu)) / 65535.0;
}

// Procedural star field: one candidate star per cell of a 3D grid around the unit sphere,
// brightness following a steep power law (few bright stars, many faint ones), colour from a
// rough temperature, gentle scintillation. Stars are about an output pixel wide (not an internal
// one: their size and brightness don't change with the screen percentage) so TSR keeps them steady.
fn star_field(dir: vec3<f32>) -> vec3<f32> {
    let scale = 150.0;
    let p = dir * scale;
    let cell = vec3<i32>(floor(p));
    var radiance = vec3<f32>(0.0);
    let pixel = 1.6 / frame.upscale.w * scale; // angular output pixel size in cell units (roughly)
    for (var i = 0; i < 8; i++) {
        let c = cell + vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1) - vec3<i32>(0, 0, 0);
        let h = star_hash(c);
        let star = normalize(vec3<f32>(c) + h.xyz) * scale;
        let d = length(p - star);
        let brightness = pow(h.w, 24.0);
        if brightness < 0.01 {
            continue;
        }
        let size = max(pixel, 0.05);
        let disk = exp(-d * d / (size * size));
        let temperature = mix(vec3<f32>(1.0, 0.72, 0.5), vec3<f32>(0.7, 0.82, 1.0), fract(h.x * 7.13));
        let twinkle = 0.8 + 0.2 * sin(frame.time * (2.0 + 4.0 * h.y) + h.z * 40.0);
        radiance += temperature * brightness * disk * twinkle;
    }
    return radiance * frame.misc.w;
}

/// Starlight arriving along `dir` through the whole atmosphere (zero in daylight, where
/// `frame.misc.w` is 0, and below the horizon).
fn starlight(dir: vec3<f32>) -> vec3<f32> {
    if frame.misc.w <= 0.0 || dir.y <= -0.02 {
        return vec3<f32>(0.0);
    }
    let transmittance = sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(frame.camera_position.xyz), dir);
    return star_field(dir) * transmittance;
}
