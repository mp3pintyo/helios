// Water surface: animated wave normals, caustics, shoreline foam, the parameters of the water
// medium and the water bodies (the surfaces' extents and levels).
// Requires common.wgsl, noise.wgsl and a `water: Water` uniform in the including shader.
//
// The surface is a sum of directional waves with sharpened crests (exp(sin) profile, deep-water
// dispersion: omega = sqrt(g k)), evaluated per pixel as a normal map on flat geometry - no
// tessellation. Waves shorter than a few pixels fade out; the slope variance they would have
// added widens the specular lobe instead, so distant water turns into a soft sheen rather
// than a field of aliasing glints (the idea behind Toksvig / LEAN mapping).

struct Water {
    absorption: vec4<f32>,  // rgb: absorption per metre
    scattering: vec4<f32>,  // rgb: scattering per metre
    waves: vec4<f32>,       // xy: travel direction (unit), z: longest wavelength (m), w: steepness
    misc: vec4<f32>,        // x: foam amount, y: caustics strength, z: body count, w: body holding the camera (-1: none)
    // Water bodies (flat surfaces): xy = min corner (x, z), zw = max corner of the surface.
    bodies: array<vec4<f32>, 4>,
    levels: vec4<f32>,      // surface height of each body
    river: vec4<f32>,       // simulated river (river.rs): xy grid corner (x, z), z 1 / cell, w 1 if any
}

const WATER_WAVES: u32 = 16u;
// Wavelength ratio of successive waves: 16 waves span 60:1 (3 m down to 5 cm by default).
const WAVE_RATIO: f32 = 0.76;
const GRAVITY: f32 = 9.81;
// Refractive index of water (relative to air).
const WATER_IOR: f32 = 1.333;

// One of the directional waves: height A * exp(sin(phase) - 1) with
// phase = k * dot(dir, xz) - omega * t + offset.
struct Wave {
    dir: vec2<f32>,
    k: f32,
    omega: f32,
    offset: f32,
    // Slope amplitude A * k.
    slope: f32,
    wavelength: f32,
}

fn water_wave(i: u32) -> Wave {
    let wind = water.waves.xy;
    var w: Wave;
    w.wavelength = water.waves.z * pow(WAVE_RATIO, f32(i));
    // Directions fan out around the wind (golden-ratio sequence), wider for shorter waves: short
    // ripples run in all directions, and evenly spread ones keep the caustics they focus from
    // forming a regular lattice.
    let spread = (2.0 * fract(f32(i) * 0.618034 + 0.3) - 1.0) * (0.3 + 1.3 * f32(i) / f32(WATER_WAVES));
    let c = cos(spread);
    let s = sin(spread);
    w.dir = vec2<f32>(wind.x * c - wind.y * s, wind.x * s + wind.y * c);
    w.k = 6.2831853 / w.wavelength;
    w.omega = sqrt(GRAVITY * w.k);
    w.offset = f32(i) * 1.913;
    // Per-wave slope amplitude: the sum of all waves peaks near `steepness`.
    w.slope = water.waves.w * 2.0 / f32(WATER_WAVES) * (1.0 - 0.35 * f32(i) / f32(WATER_WAVES));
    return w;
}

// A wave's phase at `x` and its gradient over k (the local travel direction, near `dir`).
// Crests are not straight: each bends sideways by two slow sinusoids across its travel direction,
// different for every wave. The few dominant short waves then cross at angles that vary from
// place to place, so the caustics they focus form irregular cells instead of a lattice of
// parallelograms. The bend varies only across the travel direction, so a bent crest travels as
// a whole and nothing flickers; it tilts the crest by up to 0.5 rad.
struct WavePhase {
    phase: f32,
    dir: vec2<f32>,
}

fn wave_phase(w: Wave, x: vec2<f32>, time: f32) -> WavePhase {
    let side = vec2<f32>(-w.dir.y, w.dir.x);
    let across = dot(x, side) * w.k;
    let a = across * 0.27 + w.offset * 2.3;
    let b = across * 0.11 - w.offset;
    var p: WavePhase;
    p.phase = w.k * dot(w.dir, x) - w.omega * time + w.offset + 1.6 * sin(a) + 0.8 * sin(b);
    p.dir = w.dir + side * (1.6 * 0.27 * cos(a) + 0.8 * 0.11 * cos(b));
    return p;
}

// Fades a wave out where it is shorter than a few `footprint`s (metres).
fn wave_fade(w: Wave, footprint: f32) -> f32 {
    return 1.0 - smoothstep(0.15, 0.35, footprint / w.wavelength);
}

// Where the waves are evaluated: the surface position, gently warped by a slow flow. Straight
// crests of a few directional waves would tile the surface (and the caustics they focus) into
// a regular lattice; the warp bends the short crests while leaving the long swell nearly
// straight. Slopes ignore the warp's own derivative (below 0.3).
fn wave_domain(xz: vec2<f32>, time: f32) -> vec2<f32> {
    let a = sin(dot(xz, vec2<f32>(0.83, 0.55)) * 3.7 + time * 0.6);
    let b = sin(dot(xz, vec2<f32>(-0.47, 0.88)) * 2.9 - time * 0.45);
    let c = sin(dot(xz, vec2<f32>(0.96, -0.28)) * 5.3 + time * 0.8);
    return xz + vec2<f32>(a + 0.5 * c, b - 0.5 * c) * 0.05;
}

struct WaterSurface {
    normal: vec3<f32>,
    // Slope variance of the waves too small to resolve at this pixel.
    slope_variance: f32,
}

fn water_surface(xz: vec2<f32>, time: f32, footprint: f32) -> WaterSurface {
    return water_surface_from(xz, time, footprint, 0u);
}

// The waves from the `first` on (a river carries no swell: it starts at the shorter ones).
fn water_surface_from(xz: vec2<f32>, time: f32, footprint: f32, first: u32) -> WaterSurface {
    let x = wave_domain(xz, time);
    var gradient = vec2<f32>(0.0);
    var variance = 0.0;
    for (var i = first; i < WATER_WAVES; i++) {
        let w = water_wave(i);
        let p = wave_phase(w, x, time);
        // Slope along the travel direction: A k cos(phase) exp(sin(phase) - 1).
        let fade = wave_fade(w, footprint);
        let crest = exp(sin(p.phase) - 1.0);
        gradient += p.dir * (w.slope * cos(p.phase) * crest * fade);
        variance += w.slope * w.slope * 0.3 * (1.0 - fade);
    }
    var s: WaterSurface;
    s.normal = normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
    s.slope_variance = variance;
    return s;
}

// Sun light focused by the waves onto a bed `depth` metres below the surface (1 = flat water).
//
// A sun ray entering the surface at x is bent by the slope there and reaches the bed displaced
// by about depth * (1 - 1/IOR) * grad h(x). The light density on the bed is the inverse of the
// area change of that mapping, 1 / det(I + depth * (1 - 1/IOR) * H(x)) with H the Hessian of the
// same waves that shade the surface - so caustics move exactly with the visible waves. Each
// wave's curvature fades where its caustic pattern would be finer than the bed's pixel
// `footprint` or than the blur of the sun disk, which leaves the mean intensity instead of
// noise. Near-singular (focal) points are clamped.
fn water_caustics(bed: vec3<f32>, depth: f32, time: f32, footprint: f32) -> f32 {
    if water.misc.y <= 0.0 || depth <= 0.0 {
        return 1.0;
    }
    // The surface point the sun ray towards `bed` entered at (flat refraction).
    let sun = normalize(frame.sun_direction.xyz);
    let cos_t = sqrt(1.0 - (1.0 - sun.y * sun.y) / (WATER_IOR * WATER_IOR));
    let path = depth / max(cos_t, 0.2);
    var entry = bed.xz + sun.xz / WATER_IOR * path;
    // The sun disk blurs the pattern by its angular diameter over the path: a wave's lines
    // survive until that blur nears half its wavelength.
    let blur = path * 2.0 * frame.sun_direction.w;
    let focus = path * (1.0 - 1.0 / WATER_IOR);
    var fades: array<f32, WATER_WAVES>;
    var gradient = vec2<f32>(0.0);
    let x0 = wave_domain(entry, time);
    for (var i = 0u; i < WATER_WAVES; i++) {
        let w = water_wave(i);
        fades[i] = wave_fade(w, footprint) * (1.0 - smoothstep(0.25, 0.6, blur / w.wavelength));
        if fades[i] <= 0.0 {
            continue;
        }
        let p = wave_phase(w, x0, time);
        gradient += p.dir * (w.slope * cos(p.phase) * exp(sin(p.phase) - 1.0) * fades[i]);
    }
    // The ray reaching `bed` entered where the waves' slope bent it onto `bed`: one fixed-point
    // step back along the displacement. It warps the pattern as the longer waves pass, which
    // breaks up the lattice the short waves alone would draw.
    entry -= focus * gradient;
    let x1 = wave_domain(entry, time);
    var hessian = vec3<f32>(0.0); // xx, xz, zz
    for (var i = 0u; i < WATER_WAVES; i++) {
        let w = water_wave(i);
        let fade = fades[i];
        // Long waves barely curve the surface: skip those that would change det by < 2 %.
        if fade <= 0.0 || focus * w.slope * w.k < 0.02 {
            continue;
        }
        let p = wave_phase(w, x1, time);
        let s = sin(p.phase);
        let c = cos(p.phase);
        // Second derivative along the travel direction: A k^2 (cos^2 - sin) exp(sin - 1) (the
        // bend's own curvature is an order smaller).
        let curvature = w.slope * w.k * (c * c - s) * exp(s - 1.0) * fade;
        hessian += curvature * vec3<f32>(p.dir.x * p.dir.x, p.dir.x * p.dir.y, p.dir.y * p.dir.y);
    }
    let j = vec3<f32>(1.0, 0.0, 1.0) + focus * hessian;
    let det = j.x * j.z - j.y * j.y;
    let intensity = min(1.0 / max(abs(det), 0.12), 5.0);
    // Light keeps its mean: the mapping conserves energy, the clamped estimate is a bit high.
    return max(1.0 + (intensity - 1.15) * water.misc.y, 0.0);
}

// Foam where the water is shallow (`depth` in metres): a thin, broken line along the shore and
// fainter patches drifting a little further out.
fn water_foam(xz: vec2<f32>, depth: f32, time: f32, footprint: f32) -> f32 {
    if depth >= 0.35 || water.misc.x <= 0.0 {
        return 0.0;
    }
    let drift = water.waves.xy * time * 0.15;
    let scale = 2.3;
    let n = saturate(fbm(vec3<f32>((xz - drift) * scale, time * 0.15), 4u, footprint * scale) + 0.5);
    let line = (1.0 - smoothstep(0.0, 0.03 + 0.06 * n, depth)) * smoothstep(0.35, 0.6, n);
    let patches = smoothstep(0.62, 0.8, n) * (1.0 - smoothstep(0.05, 0.35, depth)) * 0.5;
    return saturate(max(line, patches)) * water.misc.x;
}

// ---- Water bodies ------------------------------------------------------------------------------------

// Index of the water body whose surface covers `xz`, or -1.
fn water_body_at(xz: vec2<f32>) -> i32 {
    for (var i = 0; i < i32(water.misc.z); i++) {
        let b = water.bodies[i];
        if all(xz >= b.xy) && all(xz <= b.zw) {
            return i;
        }
    }
    return -1;
}

// Height of the water surface above `xz` (very low where there is none).
fn water_level_at(xz: vec2<f32>) -> f32 {
    let body = water_body_at(xz);
    if body < 0 {
        return -1.0e9;
    }
    return water.levels[body];
}

// ---- The medium -----------------------------------------------------------------------------------

fn water_extinction() -> vec3<f32> {
    return water.absorption.rgb + water.scattering.rgb;
}

// Light scattered towards the eye along `path` metres of water lit by `light` (irradiance just
// below the surface): isotropic single scattering, albedo * E / (4 pi), about halved by the
// light's own extinction on the way down; it approaches this as the path gets long.
fn water_inscatter(path: f32, light: vec3<f32>, tint: vec3<f32>) -> vec3<f32> {
    let extinction = water_extinction();
    let albedo = water.scattering.rgb / max(extinction, vec3<f32>(1e-4)) * tint;
    return (vec3<f32>(1.0) - exp(-extinction * path)) * albedo * light / (8.0 * PI);
}

// Fresnel reflectance of an unpolarised dielectric interface for light arriving at `cos_i`
// from the side with refractive index `eta` relative to the other side (1 beyond the critical
// angle: total internal reflection).
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}
