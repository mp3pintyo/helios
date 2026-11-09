// Atmosphere LUT generation and sky irradiance (SH9) projection.

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var transmittance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var multiscatter_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var multiscatter_lut: texture_2d<f32>;
@group(0) @binding(5) var sky_view_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(7) var lut_sampler: sampler;
@group(0) @binding(8) var<storage, read_write> sky_sh: array<vec4<f32>, 9>;

fn sun_direction() -> vec3<f32> {
    return normalize(frame.sun_direction.xyz);
}

// ---- Transmittance LUT (256 x 64) ------------------------------------------------------

@compute @workgroup_size(8, 8)
fn build_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(vec2<f32>(gid.xy) >= TRANSMITTANCE_LUT_SIZE) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / TRANSMITTANCE_LUT_SIZE;
    let params = transmittance_params(uv);
    let origin = vec3<f32>(0.0, params.x, 0.0);
    let dir = vec3<f32>(sqrt(max(0.0, 1.0 - params.y * params.y)), params.y, 0.0);
    let t_max = max(ray_sphere(origin, dir, ATMOSPHERE_RADIUS), 0.0);
    let steps = 40.0;
    var optical_depth = vec3<f32>(0.0);
    for (var i = 0.0; i < steps; i += 1.0) {
        let t = (i + 0.5) / steps * t_max;
        let height = length(origin + dir * t) - GROUND_RADIUS;
        optical_depth += sample_medium(height).extinction * (t_max / steps);
    }
    textureStore(transmittance_out, gid.xy, vec4<f32>(exp(-optical_depth), 1.0));
}

// ---- Multiple-scattering LUT (32 x 32, one workgroup per texel) ----------------------------

const MS_DIRECTIONS: u32 = 64u;
var<workgroup> ms_luminance: array<vec3<f32>, 64>;
var<workgroup> ms_transfer: array<vec3<f32>, 64>;

@compute @workgroup_size(1, 1, 64)
fn build_multiscatter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let uv = (vec2<f32>(wid.xy) + 0.5) / MULTISCATTER_LUT_SIZE;
    let cos_sun = uv.x * 2.0 - 1.0;
    let sun = vec3<f32>(sqrt(max(0.0, 1.0 - cos_sun * cos_sun)), cos_sun, 0.0);
    let origin = vec3<f32>(0.0, mix(GROUND_RADIUS + 0.01, ATMOSPHERE_RADIUS - 0.01, uv.y), 0.0);

    // Uniform sphere direction for this invocation (8 x 8 stratified grid).
    let cell = vec2<f32>(f32(lid % 8u), f32(lid / 8u)) + 0.5;
    let cos_theta = 1.0 - 2.0 * cell.x / 8.0;
    let phi = 2.0 * PI * cell.y / 8.0;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let dir = vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));

    let t_ground = ray_sphere(origin, dir, GROUND_RADIUS);
    let t_top = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
    let t_max = select(max(t_top, 0.0), t_ground, t_ground > 0.0);

    let steps = 20.0;
    let isotropic = 1.0 / (4.0 * PI);
    var luminance = vec3<f32>(0.0);
    var transfer = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var i = 0.0; i < steps; i += 1.0) {
        let t = (i + 0.5) / steps * t_max;
        let dt = t_max / steps;
        let p = origin + dir * t;
        let medium = sample_medium(length(p) - GROUND_RADIUS);
        let scattering = medium.rayleigh + vec3<f32>(medium.mie);
        let sample_transmittance_ = exp(-medium.extinction * dt);
        let sun_transmittance = sample_transmittance(transmittance_lut, lut_sampler, p, sun);
        let s = scattering * isotropic * sun_transmittance;
        let integral = (s - s * sample_transmittance_) / max(medium.extinction, vec3<f32>(1e-7));
        luminance += throughput * integral;
        transfer += throughput * (scattering - scattering * sample_transmittance_) / max(medium.extinction, vec3<f32>(1e-7));
        throughput *= sample_transmittance_;
    }
    if t_ground > 0.0 {
        let p = origin + dir * t_ground;
        let normal = normalize(p);
        let sun_transmittance = sample_transmittance(transmittance_lut, lut_sampler, p, sun);
        luminance += throughput * sun_transmittance * saturate(dot(normal, sun)) * frame.ground_albedo.rgb / PI;
    }

    ms_luminance[lid] = luminance / f32(MS_DIRECTIONS);
    ms_transfer[lid] = transfer * isotropic * 4.0 * PI / f32(MS_DIRECTIONS);
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            ms_luminance[lid] += ms_luminance[lid + stride];
            ms_transfer[lid] += ms_transfer[lid + stride];
        }
        workgroupBarrier();
    }
    if lid == 0u {
        let psi = ms_luminance[0] / max(vec3<f32>(1.0) - ms_transfer[0], vec3<f32>(1e-3));
        textureStore(multiscatter_out, wid.xy, vec4<f32>(psi, 1.0));
    }
}

// ---- Sky-view LUT (192 x 108), per unit sun illuminance ----------------------------------

@compute @workgroup_size(8, 8)
fn build_sky_view(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(vec2<f32>(gid.xy) >= SKY_VIEW_LUT_SIZE) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / SKY_VIEW_LUT_SIZE;
    let origin = atmosphere_position(frame.camera_position.xyz);
    let params = sky_view_params(uv, origin.y);

    // Build the view direction relative to the sun's azimuth.
    let sun = sun_direction();
    let sun_azimuth = atan2(sun.z, sun.x);
    let view_azimuth = sun_azimuth + acos(clamp(params.y, -1.0, 1.0));
    let sin_zenith = sqrt(max(0.0, 1.0 - params.x * params.x));
    let dir = vec3<f32>(sin_zenith * cos(view_azimuth), params.x, sin_zenith * sin(view_azimuth));

    let t_ground = ray_sphere(origin, dir, GROUND_RADIUS);
    let t_top = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
    let t_max = select(max(t_top, 0.0), t_ground, t_ground > 0.0);

    let cos_theta = dot(dir, sun);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta);
    let steps = 32.0;
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var t_prev = 0.0;
    for (var i = 0.0; i < steps; i += 1.0) {
        // Quadratic step distribution concentrates samples near the viewer.
        let t = t_max * pow((i + 0.5) / steps, 2.0);
        let dt = t - t_prev;
        t_prev = t;
        let p = origin + dir * t;
        let medium = sample_medium(length(p) - GROUND_RADIUS);
        let sample_transmittance_ = exp(-medium.extinction * dt);
        let sun_transmittance = sample_transmittance(transmittance_lut, lut_sampler, p, sun);
        let multiscatter = sample_multiscatter(multiscatter_lut, lut_sampler, p, sun);
        let scattering = medium.rayleigh + vec3<f32>(medium.mie);
        let s = (medium.rayleigh * phase_r + vec3<f32>(medium.mie * phase_m)) * sun_transmittance + scattering * multiscatter;
        radiance += throughput * (s - s * sample_transmittance_) / max(medium.extinction, vec3<f32>(1e-7));
        throughput *= sample_transmittance_;
    }
    textureStore(sky_view_out, gid.xy, vec4<f32>(radiance, 1.0));
}

// ---- Sky irradiance as order-2 spherical harmonics ----------------------------------------

var<workgroup> sh_partial: array<array<vec3<f32>, 9>, 64>;

@compute @workgroup_size(64)
fn project_sky_sh(@builtin(local_invocation_index) lid: u32) {
    let sun = sun_direction();
    let camera = frame.camera_position.xyz;
    let origin = atmosphere_position(camera);
    let ground_irradiance = sample_transmittance(transmittance_lut, lut_sampler, origin, sun) * saturate(sun.y);

    var acc: array<vec3<f32>, 9>;
    for (var k = 0u; k < 9u; k++) {
        acc[k] = vec3<f32>(0.0);
    }
    let grid = 32u;
    for (var i = lid; i < grid * grid; i += 64u) {
        let cell = vec2<f32>(f32(i % grid), f32(i / grid)) + 0.5;
        let cos_theta = 1.0 - 2.0 * cell.x / f32(grid);
        let phi = 2.0 * PI * cell.y / f32(grid);
        let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        let dir = vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));
        var radiance = sample_sky_view(sky_view_lut, lut_sampler, camera, dir, sun);
        if dir.y < 0.0 {
            // Light bounced off the (unseen) ground below the horizon.
            radiance += frame.ground_albedo.rgb * ground_irradiance / PI;
        }
        var y = sh_basis(dir);
        let weight = 4.0 * PI / f32(grid * grid);
        for (var k = 0u; k < 9u; k++) {
            acc[k] += radiance * y[k] * weight;
        }
    }
    sh_partial[lid] = acc;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            for (var k = 0u; k < 9u; k++) {
                sh_partial[lid][k] += sh_partial[lid + stride][k];
            }
        }
        workgroupBarrier();
    }
    if lid < 9u {
        sky_sh[lid] = vec4<f32>(sh_partial[0][lid], 0.0);
    }
}
