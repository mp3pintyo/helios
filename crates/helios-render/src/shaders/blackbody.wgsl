// Blackbody emission for fire, sparks and embers (volumes.rs, particles.rs).

// Planck's law sampled at representative red, green and blue wavelengths (µm), relative to a
// 1500 K emitter at 550 nm. The radiance rises steeply with temperature (a 1900 K flame core is
// ~40x brighter than its 1450 K edges) and shifts from dull red to yellow-white, which is what
// makes fire read as fire. Below ~700 K the result is black.
fn blackbody(kelvin: f32) -> vec3<f32> {
    let lambda = vec3<f32>(0.61, 0.55, 0.465);
    let c2 = 14388.0; // second radiation constant, µm K
    let reference = exp(c2 / (0.55 * 1500.0)) - 1.0;
    // exp() overflows f32 beyond ~88: clamp (the radiance is 0 there anyway).
    let x = min(c2 / (lambda * max(kelvin, 300.0)), vec3<f32>(80.0));
    let ratio = vec3<f32>(0.55) / lambda;
    return ratio * ratio * ratio * ratio * ratio * reference / (exp(x) - 1.0);
}
