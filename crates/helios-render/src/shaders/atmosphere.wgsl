// Physically based atmosphere model (Hillaire 2020, "A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique" — the technique behind UE's SkyAtmosphere).
// Distances are in kilometres. Requires common.wgsl.

const GROUND_RADIUS: f32 = 6360.0;
const ATMOSPHERE_RADIUS: f32 = 6460.0;
const RAYLEIGH_SCATTERING: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_SCALE_HEIGHT: f32 = 8.0;
const MIE_SCATTERING: f32 = 3.996e-3;
const MIE_ABSORPTION: f32 = 4.40e-3;
const MIE_SCALE_HEIGHT: f32 = 1.2;
const MIE_G: f32 = 0.8;
const OZONE_ABSORPTION: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);

const TRANSMITTANCE_LUT_SIZE: vec2<f32> = vec2<f32>(256.0, 64.0);
const MULTISCATTER_LUT_SIZE: f32 = 32.0;
const SKY_VIEW_LUT_SIZE: vec2<f32> = vec2<f32>(192.0, 108.0);

struct Medium {
    rayleigh: vec3<f32>,
    mie: f32,
    extinction: vec3<f32>,
}

fn sample_medium(height: f32) -> Medium {
    var m: Medium;
    let rayleigh_density = exp(-max(height, 0.0) / RAYLEIGH_SCALE_HEIGHT);
    let mie_density = exp(-max(height, 0.0) / MIE_SCALE_HEIGHT);
    let ozone_density = max(0.0, 1.0 - abs(height - 25.0) / 15.0);
    m.rayleigh = RAYLEIGH_SCATTERING * rayleigh_density;
    m.mie = MIE_SCATTERING * mie_density;
    m.extinction = m.rayleigh + vec3<f32>((MIE_SCATTERING + MIE_ABSORPTION) * mie_density) + OZONE_ABSORPTION * ozone_density;
    return m;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

// Cornette–Shanks phase function.
fn mie_phase(cos_theta: f32) -> f32 {
    let g = MIE_G;
    let k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5);
}

// Distance along the ray to the sphere, or -1 when missed (origin inside → far hit).
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return -1.0;
    }
    let s = sqrt(disc);
    let near = -b - s;
    let far = -b + s;
    if near >= 0.0 {
        return near;
    }
    if far >= 0.0 {
        return far;
    }
    return -1.0;
}

// Planet-centred camera position in km (the scene sits at the top of the planet).
fn atmosphere_position(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0, GROUND_RADIUS + max(world.y * 0.001, 0.0) + 0.05, 0.0);
}

// --- Transmittance LUT parameterisation (Bruneton) -----------------------------------------

fn transmittance_uv(view_height: f32, cos_zenith: f32) -> vec2<f32> {
    let h = sqrt(max(0.0, ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS - GROUND_RADIUS * GROUND_RADIUS));
    let rho = sqrt(max(0.0, view_height * view_height - GROUND_RADIUS * GROUND_RADIUS));
    let disc = view_height * view_height * (cos_zenith * cos_zenith - 1.0) + ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS;
    let d = max(0.0, -view_height * cos_zenith + sqrt(max(disc, 0.0)));
    let d_min = ATMOSPHERE_RADIUS - view_height;
    let d_max = rho + h;
    return vec2<f32>((d - d_min) / (d_max - d_min), rho / h);
}

fn transmittance_params(uv: vec2<f32>) -> vec2<f32> {
    let h = sqrt(ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS - GROUND_RADIUS * GROUND_RADIUS);
    let rho = h * uv.y;
    let view_height = sqrt(rho * rho + GROUND_RADIUS * GROUND_RADIUS);
    let d_min = ATMOSPHERE_RADIUS - view_height;
    let d_max = rho + h;
    let d = d_min + uv.x * (d_max - d_min);
    var cos_zenith = 1.0;
    if d > 0.0 {
        cos_zenith = clamp((h * h - rho * rho - d * d) / (2.0 * view_height * d), -1.0, 1.0);
    }
    return vec2<f32>(view_height, cos_zenith);
}

fn sample_transmittance(lut: texture_2d<f32>, s: sampler, position: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let height = length(position);
    let cos_zenith = dot(position / height, dir);
    // The planet itself occludes the direction.
    if ray_sphere(position, dir, GROUND_RADIUS) > 0.0 {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(lut, s, transmittance_uv(height, cos_zenith), 0.0).rgb;
}

fn sample_multiscatter(lut: texture_2d<f32>, s: sampler, position: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let height = length(position);
    let cos_sun = dot(position / height, sun);
    let uv = vec2<f32>(cos_sun * 0.5 + 0.5, (height - GROUND_RADIUS) / (ATMOSPHERE_RADIUS - GROUND_RADIUS));
    let texel = 0.5 / MULTISCATTER_LUT_SIZE;
    return textureSampleLevel(lut, s, clamp(uv, vec2<f32>(texel), vec2<f32>(1.0 - texel)), 0.0).rgb;
}

// --- Sky-view LUT parameterisation (non-linear latitude, relative sun azimuth) -----------

fn sky_view_uv(view_height: f32, dir: vec3<f32>, sun: vec3<f32>) -> vec2<f32> {
    let v_horizon = sqrt(max(view_height * view_height - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let beta = acos(clamp(v_horizon / view_height, -1.0, 1.0));
    let zenith_horizon = PI - beta;
    let view_zenith = acos(clamp(dir.y, -1.0, 1.0));

    var uv: vec2<f32>;
    if view_zenith < zenith_horizon {
        uv.y = (1.0 - sqrt(max(0.0, 1.0 - view_zenith / zenith_horizon))) * 0.5;
    } else {
        let coord = sqrt(max(0.0, (view_zenith - zenith_horizon) / beta));
        uv.y = coord * 0.5 + 0.5;
    }
    let dir_h = dir.xz;
    let sun_h = sun.xz;
    var light_view_cos = 1.0;
    if dot(dir_h, dir_h) > 1e-8 && dot(sun_h, sun_h) > 1e-8 {
        light_view_cos = dot(normalize(dir_h), normalize(sun_h));
    }
    uv.x = sqrt(max(0.0, -light_view_cos * 0.5 + 0.5));
    return uv;
}

// Inverse of sky_view_uv: returns (view zenith cos, light-view cos).
fn sky_view_params(uv: vec2<f32>, view_height: f32) -> vec2<f32> {
    let v_horizon = sqrt(max(view_height * view_height - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let beta = acos(clamp(v_horizon / view_height, -1.0, 1.0));
    let zenith_horizon = PI - beta;
    var view_zenith: f32;
    if uv.y < 0.5 {
        var coord = 1.0 - 2.0 * uv.y;
        coord = 1.0 - coord * coord;
        view_zenith = zenith_horizon * coord;
    } else {
        let coord = uv.y * 2.0 - 1.0;
        view_zenith = zenith_horizon + beta * coord * coord;
    }
    let coord = uv.x * uv.x;
    return vec2<f32>(cos(view_zenith), -(coord * 2.0 - 1.0));
}

/// Direction whose sky radiance stands in for the light scattered into a short view ray
/// (aerial perspective, height fog). Below the horizon the sky-view LUT holds the dark ground,
/// which would make haze jump at eye height; light scattered along a short path looks like
/// the horizon sky instead, so downward rays use the horizon in their azimuth.
fn haze_direction(dir: vec3<f32>) -> vec3<f32> {
    let min_y = 0.02;
    if dir.y >= min_y {
        return dir;
    }
    let len = length(dir.xz);
    let h = select(vec2<f32>(1.0, 0.0), dir.xz / max(len, 1e-6), len > 1e-4);
    return normalize(vec3<f32>(h.x, min_y, h.y));
}

// Sky radiance per unit sun illuminance in direction `dir`.
fn sample_sky_view(lut: texture_2d<f32>, s: sampler, camera_world: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let height = length(atmosphere_position(camera_world));
    let uv = sky_view_uv(height, dir, sun);
    let half_texel = 0.5 / SKY_VIEW_LUT_SIZE;
    return textureSampleLevel(lut, s, clamp(uv, half_texel, 1.0 - half_texel), 0.0).rgb;
}
