// Deferred lighting: sun (with virtual shadow maps and atmospheric transmittance), screen-space
// global illumination with sky fallback, image-based specular from the sky, emissive,
// aerial perspective and the sky itself.

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "vsm_prefiltered.wgsl"
#include "vsm_local_common.wgsl"
#include "vsm_local_sample.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "lights.wgsl"
#include "brdf.wgsl"
#include "substrate_read_off.wgsl"
#include "subsurface_store_off.wgsl"
#include "clouds_common.wgsl"
#include "fog_common.wgsl"
#include "stars.wgsl"
#include "glass_shadow.wgsl"
#include "glass_light.wgsl"
#include "volume_shadow_sample.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(5) var vsm_atlas: texture_depth_2d;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
@group(0) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(9) var lut_sampler: sampler;
@group(0) @binding(10) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(11) var gi_texture: texture_2d<f32>;
@group(0) @binding(12) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<uniform> vsm: Vsm;
@group(0) @binding(14) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(15) var reflection_texture: texture_2d<f32>;
@group(0) @binding(16) var<uniform> lumen: Lumen;
@group(0) @binding(17) var global_sdf: texture_3d<f32>;
@group(0) @binding(18) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(19) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(20) var lumen_sampler: sampler;
@group(0) @binding(21) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(22) var<storage, read> light_grid: array<u32>;
@group(0) @binding(28) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(29) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(23) var cloud_texture: texture_2d<f32>;
@group(0) @binding(24) var cloud_shadow_map: texture_2d<f32>;
@group(0) @binding(25) var<uniform> clouds: Clouds;
@group(0) @binding(26) var fog_volume: texture_3d<f32>;
@group(0) @binding(27) var<uniform> fog: Fog;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;
// Per pixel, the visibility (4 bits each) of its ShadowedLights::mapped lights (local_shadows).
@group(0) @binding(61) var<storage, read_write> local_shadow_mask: array<u32>;
// MegaLights (megalights.rs): the denoised direct light of every local light.
@group(0) @binding(62) var megalights_lighting: texture_2d<f32>;

// Volumetric fog between the camera and view depth `depth` (see fog.wgsl).
fn apply_volumetric_fog(color: vec3<f32>, uv: vec2<f32>, depth: f32) -> vec3<f32> {
    if fog.enabled == 0u {
        return color;
    }
    // Slice z holds the integral up to the far end of the slice.
    let w = fog_depth_slice(depth) - 0.5 / f32(textureDimensions(fog_volume).z);
    let v = textureSampleLevel(fog_volume, lut_sampler, vec3<f32>(uv, saturate(w)), 0.0);
    return color * v.a + v.rgb;
}

// fp16 render targets saturate at 65504; keep the sun disk well below that.
const MAX_SUN_RADIANCE: f32 = 20000.0;

// The lights whose shadows a pixel evaluates, picked by strength (not list order, which keeps
// the choice stable where neighbouring froxels list different lights): the MASKED_LIGHTS
// strongest with a virtual shadow map, whose shadows local_shadows computes (contact-hardening
// soft shadows for the FILTERED_LIGHTS strongest, one filtered tap for the others; any further
// light with a shadow map gets one tap in the shading), and the MAX_SHADOWED_LIGHTS strongest
// without one (distance-field soft shadows: mesh SDFs near the receiver and the light, the
// global field between).
struct ShadowedLights {
    mapped: array<u32, MASKED_LIGHTS>,
    traced: array<u32, MASKED_LIGHTS>,
}

fn select_shadowed_lights(base: u32, p: vec3<f32>, n: vec3<f32>) -> ShadowedLights {
    var mapped_lights: array<u32, MASKED_LIGHTS>;
    var traced_lights: array<u32, MASKED_LIGHTS>;
    var mapped_strength: array<f32, MASKED_LIGHTS>;
    var traced_strength: array<f32, MASKED_LIGHTS>;
    for (var k = 0u; k < MASKED_LIGHTS; k++) {
        mapped_lights[k] = NO_LIGHT;
        traced_lights[k] = NO_LIGHT;
    }
    for (var i = 0u; i < light_grid[base]; i++) {
        let index = light_grid[base + 1u + i];
        let light = lights[index];
        let mapped = light_shadow_slot(light) != INVALID_ID;
        if light.params.z <= 0.5 || (!mapped && lumen.sdf_enabled == 0u) {
            continue;
        }
        let ls = sample_light(light, p);
        let s = luminance(ls.radiance) * saturate(dot(n, ls.l) + 0.1);
        if s <= 0.05 {
            continue;
        }
        if mapped {
            insert_strongest(&mapped_lights, &mapped_strength, index, s);
        } else {
            insert_strongest(&traced_lights, &traced_strength, index, s);
        }
    }
    return ShadowedLights(mapped_lights, traced_lights);
}

// Keeps `lights` sorted by decreasing `strengths`, with `light` (of strength `s`) among them if
// it is one of the MASKED_LIGHTS strongest.
fn insert_strongest(lights: ptr<function, array<u32, MASKED_LIGHTS>>, strengths: ptr<function, array<f32, MASKED_LIGHTS>>, light: u32, s: f32) {
    var candidate = light;
    var cs = s;
    for (var k = 0u; k < MASKED_LIGHTS; k++) {
        if cs > (*strengths)[k] {
            let os = (*strengths)[k];
            let oc = (*lights)[k];
            (*strengths)[k] = cs;
            (*lights)[k] = candidate;
            cs = os;
            candidate = oc;
        }
    }
}

// Position of `light` in `lights`, or MASKED_LIGHTS.
fn listed_at(lights: array<u32, MASKED_LIGHTS>, light: u32) -> u32 {
    for (var k = 0u; k < MASKED_LIGHTS; k++) {
        if lights[k] == light {
            return k;
        }
    }
    return MASKED_LIGHTS;
}

// Visibility `k` of a local shadow mask word (4 bits each, dithered: TAA resolves the steps).
fn mask_visibility(mask: u32, k: u32) -> f32 {
    return f32((mask >> (4u * k)) & 0xfu) / 15.0;
}

const MAX_SHADOWED_LIGHTS: u32 = 4u;
// Shadow-mapped lights per pixel shadowed by local_shadows, and how many of them get
// contact-hardening soft shadows.
const MASKED_LIGHTS: u32 = 8u;
const FILTERED_LIGHTS: u32 = 4u;
const NO_LIGHT: u32 = 0xffffffffu;
// Filter taps of local light shadow maps (plus a third as many blocker-search gathers).
const LOCAL_SHADOW_TAPS: u32 = 8u;

// Sun transmittance through the cloud layer (see `shadow` in clouds.wgsl).
fn cloud_shadow(world_position: vec3<f32>) -> f32 {
    if clouds.enabled == 0u {
        return 1.0;
    }
    let sun = normalize(frame.sun_direction.xyz);
    let ground = world_position.xz - sun.xz * (world_position.y / max(sun.y, 0.05));
    let uv = (ground - clouds.shadow_center.xy) / clouds.shadow_extent + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 1.0;
    }
    return textureSampleLevel(cloud_shadow_map, lut_sampler, uv, 0.0).r;
}

fn sun_shadow(world_position: vec3<f32>, n: vec3<f32>, noise: f32) -> f32 {
    // Clouds and the shadows of smoke, fire and mist (heterogeneous volumes).
    let clouds = cloud_shadow(world_position) * volume_sun_transmittance(world_position, 0xffffffffu);
    if !has_flag(frame.flags, FLAG_SHADOWS) {
        return clouds;
    }
    return vsm_shadow_prefiltered(world_position, n, normalize(frame.sun_direction.xyz), noise) * clouds;
}

// Ray-marches the global distance field from the eye: shaded by the hit normal, red where the
// trace ran out of steps.
fn debug_distance_field(eye: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let hit = trace_sdf(eye, dir, 0.1, 400.0);
    if !hit.hit {
        return vec3<f32>(0.05, 0.06, 0.1);
    }
    let n = sdf_normal(hit.position, hit.level);
    let albedo = abs(sdf_albedo(hit.position, hit.level));
    let shade = 0.35 + 0.65 * saturate(dot(n, normalize(vec3<f32>(0.4, 0.8, 0.3))));
    let level_tint = vec3<f32>(1.0) - vec3<f32>(f32(hit.level) * 0.12, 0.0, f32(hit.level) * -0.05);
    return albedo * shade * level_tint * 2.0 + vec3<f32>(f32(hit.steps) / 96.0, 0.0, 0.0) * 0.5;
}

// Shadow pages of the local light with a virtual shadow map that contributes most here (see
// local_vsm_debug_color), shaded by its shadow; dark grey where no such light reaches.
fn debug_local_shadow_pages(pixel: vec2<u32>, linear_depth: f32, world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if frame.light_count == 0u {
        return vec3<f32>(0.08);
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let base = cluster_index(uv, linear_depth) * CLUSTER_WORDS;
    var strongest = NO_LIGHT;
    var strength = 0.0;
    for (var i = 0u; i < light_grid[base]; i++) {
        let index = light_grid[base + 1u + i];
        let ls = sample_light(lights[index], world);
        let s = luminance(ls.radiance) * saturate(dot(n, ls.l));
        if light_shadow_slot(lights[index]) != INVALID_ID && s > strength {
            strength = s;
            strongest = index;
        }
    }
    if strongest == NO_LIGHT {
        return vec3<f32>(0.08);
    }
    let slot = light_shadow_slot(lights[strongest]);
    // The mask holds its shadow if the shading picked it too (select_shadowed_lights).
    let k = listed_at(select_shadowed_lights(base, world, n).mapped, strongest);
    var shadow = local_vsm_visibility(slot, world, n, 0u);
    if k < MASKED_LIGHTS {
        shadow = mask_visibility(local_shadow_mask[pixel.y * u32(frame.viewport.x) + pixel.x], k);
    }
    return local_vsm_debug_color(slot, world, n) * (0.3 + 0.7 * shadow);
}

// ---- Indirect lighting -------------------------------------------------------------------------

// Bilateral upsample of the half-resolution GI (rgb: mean incident radiance, a: sky visibility).
fn sample_gi(pixel: vec2<u32>, linear_depth: f32, normal: vec3<f32>) -> vec4<f32> {
    let gi_size = viewport_extent(frame.viewport, 2u);
    let p = (vec2<f32>(pixel) + 0.5) * 0.5 - 0.5;
    let base = vec2<i32>(floor(p));
    let f = fract(p);
    let near = frame.camera_position.w;
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = vec2<i32>(i & 1, i >> 1);
        let tap = clamp(base + offset, vec2<i32>(0), vec2<i32>(gi_size) - 1);
        let bilinear = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1);
        let full = vec2<u32>(tap) * 2u;
        let tap_depth = near / max(textureLoad(depth_texture, full, 0).r, 1e-9);
        let tap_normal = textureLoad(gbuffer_normal, full, 0).xyz;
        let depth_weight = exp(-abs(tap_depth - linear_depth) / (0.03 * linear_depth + 0.05));
        let normal_weight = pow(saturate(dot(tap_normal, normal)), 8.0);
        let w = bilinear * depth_weight * normal_weight + 1e-4;
        sum += textureLoad(gi_texture, tap, 0) * w;
        weight_sum += w;
    }
    return sum / weight_sum;
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    return sample_sky_view(sky_view_lut, lut_sampler, frame.camera_position.xyz, dir, sun) * frame.sun_illuminance.rgb;
}

/// How much rough specular should come from gathered radiance instead of the sky: fully where
/// the sky is mostly occluded (interiors), barely in the open.
fn sky_occlusion_blend(sky_visibility: f32) -> f32 {
    return 1.0 - 0.6 * saturate(sky_visibility);
}

// Prefiltered radiance along reflection `r` for `roughness`, before traced reflections: the
// sky, blurred towards irradiance for rough surfaces, and with GI on, the radiance the screen
// probes gathered over the hemisphere (it sees walls and windows, not the open sky, which
// matters indoors).
fn specular_environment(r: vec3<f32>, roughness: f32, gi_radiance: vec3<f32>, sky_visibility: f32) -> vec3<f32> {
    let sharp = sky_radiance(r);
    let blurry = sh_irradiance(sky_sh, r) * frame.sun_illuminance.rgb / PI;
    var environment = mix(sharp, blurry, smoothstep(0.15, 0.75, roughness));
    // Below the horizon the sky LUT only holds thin air; reflect the lit ground instead.
    environment = mix(environment, blurry, 1.0 - smoothstep(-0.15, 0.02, r.y));
    if has_flag(frame.flags, FLAG_GI) {
        let rough = mix(environment, gi_radiance, sky_occlusion_blend(sky_visibility));
        environment = mix(environment, rough, smoothstep(0.2, 0.5, roughness));
    }
    return environment;
}

fn sun_disk(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    let radius = frame.sun_direction.w;
    let cos_angle = dot(dir, sun);
    let cos_radius = cos(radius);
    if cos_angle < cos_radius {
        return vec3<f32>(0.0);
    }
    let t = saturate((1.0 - cos_angle) / (1.0 - cos_radius));
    let limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - t));
    let transmittance = sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(frame.camera_position.xyz), dir);
    let radiance = frame.sun_illuminance.rgb / (PI * radius * radius) * limb * transmittance;
    return min(radiance, vec3<f32>(MAX_SUN_RADIANCE));
}

fn apply_aerial_perspective(color: vec3<f32>, world_position: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    if !has_flag(frame.flags, FLAG_FOG) {
        return color;
    }
    let eye = frame.camera_position.xyz;
    let distance_km = distance(world_position, eye) * 0.001 * frame.misc.y;
    let height_km = max(0.5 * (eye.y + world_position.y), 0.0) * 0.001;
    let transmittance = exp(-sample_medium(height_km).extinction * distance_km);
    return color * transmittance + sky_radiance(haze_direction(view_dir)) * (vec3<f32>(1.0) - transmittance);
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = gid.xy;
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let eye = frame.camera_position.xyz;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let view_dir = view_direction(uv, frame.inv_view_proj, eye);

    if frame.debug_view == DEBUG_SDF {
        textureStore(output, pixel, vec4<f32>(debug_distance_field(eye, view_dir), 1.0));
        return;
    }
    if depth <= 0.0 {
        var sky = sky_radiance(view_dir) + sun_disk(view_dir) + starlight(view_dir);
        if !is_lit_view(frame.debug_view) {
            sky = vec3<f32>(0.05, 0.06, 0.08);
        } else if clouds.enabled != 0u {
            // Clouds in front of the sky: rgb inscattered radiance, a transmittance.
            let c = textureSampleLevel(cloud_texture, lut_sampler, viewport_uv(uv, frame.viewport.xy * 0.5, textureDimensions(cloud_texture)), 0.0);
            sky = sky * c.a + c.rgb;
        }
        if is_lit_view(frame.debug_view) {
            sky = apply_volumetric_fog(sky, uv, fog.range.y);
        }
        textureStore(output, pixel, vec4<f32>(sky, 1.0));
        return;
    }

    let albedo_metal = textureLoad(gbuffer_albedo, pixel, 0);
    let normal_rough = textureLoad(gbuffer_normal, pixel, 0);
    let motion = textureLoad(gbuffer_motion, pixel, 0);
    let albedo = albedo_metal.rgb * albedo_metal.rgb;
    let metallic = albedo_metal.a;
    let n = normalize(normal_rough.xyz);
    let roughness = clamp(normal_rough.w, 0.045, 1.0);
    let a = roughness * roughness;

    let world_position = world_from_depth(uv, depth, frame.inv_view_proj);
    let near = frame.camera_position.w;
    let linear_depth = near / depth;
    let v = -view_dir;
    let n_dot_v = max(dot(n, v), 1e-4);
    let noise = interleaved_gradient_noise(vec2<f32>(pixel), frame.frame_index);

    let sun = normalize(frame.sun_direction.xyz);
    let sun_transmittance = sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(world_position), sun);
    // Through glass, the panes' colour filters the sun (glass_shadow.wgsl).
    let sun_light = frame.sun_illuminance.rgb * sun_transmittance * glass_transmission_soft(world_position, lut_sampler, noise);
    let n_dot_l = saturate(dot(n, sun));
    let surface_flags = surface_flags_of(motion.w);

    // The Substrate slab of pixels whose material has one (substrate.wgsl). Without slabs in
    // the scene this is compiled out.
    let substrate = SUBSTRATE_LIGHTING && surface_flags >= SURFACE_SUBSTRATE_FLAG;
    var slab = slab_none();
    var sh: SlabShading;
    if substrate {
        slab = load_slab(pixel);
        sh = shade_slab(slab, slab_constants(slab.material), n, roughness, albedo, metallic, v);
    }
    // Subsurface scattering lights a little past the terminator (wrapped diffuse).
    let sun_reaches = n_dot_l > 0.0 || (substrate && dot(n, sun) + sh.wrap > 0.0);
    let shadow = select(0.0, sun_shadow(world_position, n, noise), sun_reaches);

    // Debug views that only need simple shading.
    switch frame.debug_view {
        case DEBUG_CLUSTERS, DEBUG_TRIANGLES, DEBUG_LOD, DEBUG_INSTANCES, DEBUG_ALBEDO, DEBUG_RASTER_PATH, DEBUG_MATERIALS, DEBUG_SKIN_WEIGHTS: {
            let shade = 0.3 + 0.7 * n_dot_l * mix(0.4, 1.0, shadow);
            textureStore(output, pixel, vec4<f32>(albedo * shade, 1.0));
            return;
        }
        case DEBUG_NORMALS: {
            textureStore(output, pixel, vec4<f32>(n * 0.5 + 0.5, 1.0));
            return;
        }
        case DEBUG_SHADOW_PAGES: {
            textureStore(output, pixel, vec4<f32>(vsm_debug_color(world_position) * (0.25 + 0.75 * n_dot_l * shadow), 1.0));
            return;
        }
        case DEBUG_LOCAL_SHADOW_PAGES: {
            textureStore(output, pixel, vec4<f32>(debug_local_shadow_pages(pixel, linear_depth, world_position, n), 1.0));
            return;
        }
        case DEBUG_MATERIAL_FEATURES: {
            let shade = 0.3 + 0.7 * n_dot_l * mix(0.4, 1.0, shadow);
            textureStore(output, pixel, vec4<f32>(slab_debug_color(slab, pixel, substrate) * shade, 1.0));
            return;
        }
        default: {}
    }

    // Unlit slabs (KHR_materials_unlit) show their colour as bright as a white surface in full
    // sun, fogged like everything else.
    if substrate && has_feature(slab, SLAB_UNLIT) {
        let unlit = apply_aerial_perspective(albedo * (luminance(frame.sun_illuminance.rgb) / PI), world_position, view_dir);
        textureStore(output, pixel, vec4<f32>(apply_volumetric_fog(unlit, uv, linear_depth), 1.0));
        return;
    }

    let diffuse_color = albedo * (1.0 - metallic);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Direct sun.
    var color: vec3<f32>;
    // Subsurface pixels' diffuse light, demodulated: subsurface.wgsl diffuses it on screen.
    var scattered = vec3<f32>(0.0);
    if substrate {
        let c = evaluate_slab(sh, sun);
        let lit = sun_light * shadow;
        color = (diffuse_color * (c.diffuse * (1.0 - sh.subsurface)) + c.specular) * lit;
        scattered = c.diffuse * sh.subsurface * lit;
        if sh.transmission > 0.0 {
            color += diffuse_color * sh.transmission_color * (c.back * sun_shadow(world_position, -n, noise)) * sun_light;
        }
    } else {
        let h = normalize(v + sun);
        let n_dot_h = saturate(dot(n, h));
        let v_dot_h = saturate(dot(v, h));
        let fresnel = f_schlick(f0, v_dot_h);
        let specular = d_ggx(n_dot_h, a) * v_smith_ggx_correlated(n_dot_v, n_dot_l, a) * fresnel;
        let diffuse = diffuse_color / PI * (vec3<f32>(1.0) - fresnel);
        color = (diffuse + specular) * sun_light * n_dot_l * shadow;
    }

    // Local lights: MegaLights' stochastic, denoised lighting, or the clustered grid's lights
    // shadowed as select_shadowed_lights describes.
    if has_flag(frame.flags, FLAG_MEGALIGHTS) {
        color += textureLoad(megalights_lighting, pixel, 0).rgb;
    } else if frame.light_count > 0u {
        let base = cluster_index(uv, linear_depth) * CLUSTER_WORDS;
        let count = light_grid[base];
        let selected = select_shadowed_lights(base, world_position, n);
        var mask = 0u;
        if selected.mapped[0] != NO_LIGHT {
            mask = local_shadow_mask[pixel.y * size.x + pixel.x];
        }
        for (var i = 0u; i < count; i++) {
            let index = light_grid[base + 1u + i];
            let light = lights[index];
            let ls = sample_light(light, world_position);
            let nl = saturate(dot(n, ls.l));
            let wrap = select(0.0, 0.35, surface_flags == 1u); // foliage transmits
            // Slabs: wrapped subsurface diffuse and thin transmission reach behind the terminator.
            var reach = nl + wrap;
            if substrate {
                reach = max(dot(n, ls.l) + sh.wrap, sh.transmission);
            }
            if reach <= 0.0 {
                continue;
            }
            var radiance = ls.radiance;
            let slot = light_shadow_slot(light);
            if slot != INVALID_ID {
                let k = listed_at(selected.mapped, index);
                if k < MASKED_LIGHTS {
                    radiance *= mask_visibility(mask, k);
                } else {
                    radiance *= local_vsm_visibility(slot, world_position, n, 0u);
                }
            } else if listed_at(selected.traced, index) < MAX_SHADOWED_LIGHTS {
                radiance *= local_light_shadow(world_position + n * 0.04, light.position_range.xyz, light.color.w);
            }
            // Through glass between the light and the pixel, in the panes' colour.
            radiance *= glass_light_transmission(index, light.position_range.xyz, world_position);
            // Through the smoke of the fire that drives the light.
            radiance *= volume_light_transmittance(index, light.position_range.xyz, world_position);
            if substrate {
                let c = evaluate_slab(sh, ls.l);
                color += (diffuse_color * (c.diffuse * (1.0 - sh.subsurface) + sh.transmission_color * c.back) + c.specular) * radiance;
                scattered += c.diffuse * sh.subsurface * radiance;
            } else {
                let brdf = evaluate_brdf(n, v, ls.l, n_dot_v, f0, a);
                color += (diffuse_color * brdf.diffuse * (nl + wrap * albedo) + brdf.specular * nl) * radiance;
            }
        }
    }
    if surface_flags == 1u {
        // Foliage: thin leaves transmit sunlight arriving from behind.
        let back = saturate(dot(-n, sun));
        let back_shadow = sun_shadow(world_position, -n, noise);
        color += albedo * sun_light * (back * 0.35 + 0.08) * back_shadow / PI;
    }

    // Indirect diffuse: screen-space GI (with sky fallback) or plain sky irradiance.
    var sky_visibility = 1.0;
    var indirect = vec3<f32>(0.0);
    var gi_radiance = vec3<f32>(0.0);
    if has_flag(frame.flags, FLAG_GI) {
        let gi = sample_gi(pixel, linear_depth, n);
        gi_radiance = gi.rgb * frame.gi_intensity;
        indirect = diffuse_color * gi_radiance;
        sky_visibility = gi.a;
    } else {
        indirect = diffuse_color * sh_irradiance(sky_sh, n) * frame.sun_illuminance.rgb / PI;
    }
    // Material ambient occlusion (baked creases) darkens indirect light only, as in UE.
    let occlusion = surface_occlusion_of(motion.w);
    indirect *= occlusion;
    // Slabs: the coat and the sheen take their share; subsurface pixels scatter theirs.
    var irradiance = vec3<f32>(0.0);
    if substrate {
        irradiance = select(sh_irradiance(sky_sh, n) * frame.sun_illuminance.rgb / PI, gi_radiance, has_flag(frame.flags, FLAG_GI));
        let under = irradiance * (occlusion * sh.coat_view * sh.sheen_view);
        indirect = diffuse_color * under * (1.0 - sh.subsurface);
        scattered += under * sh.subsurface;
        if sh.transmission > 0.0 {
            // Thin surfaces pass on the sky light reaching their back.
            let back = sh_irradiance(sky_sh, -n) * frame.sun_illuminance.rgb / PI * (sky_visibility * occlusion);
            indirect += diffuse_color * sh.transmission_color * back * sh.transmission;
        }
    }

    // Indirect specular from the sky, blurred towards irradiance for rough surfaces.
    let r = reflect(-v, n);
    var environment = specular_environment(r, roughness, gi_radiance, sky_visibility);
    var reflection = vec4<f32>(0.0);
    var reflection_weight = 0.0;
    if has_flag(frame.flags, FLAG_GI) {
        // Lumen: traced reflections for glossy surfaces. A single ray per pixel cannot resolve a
        // wide lobe without fireflies, so tracing stops at MAX_REFLECTION_ROUGHNESS in
        // lumen_gather.wgsl.
        reflection = textureSampleLevel(reflection_texture, lut_sampler, viewport_uv(uv, frame.viewport.xy * 0.5, textureDimensions(reflection_texture)), 0.0);
        reflection_weight = (1.0 - smoothstep(0.35, 0.55, roughness)) * saturate(reflection.a);
        environment = mix(environment, reflection.rgb, reflection_weight);
    }
    let visibility = sky_visibility * occlusion;
    let specular_occlusion = saturate(pow(n_dot_v + visibility, exp2(-16.0 * a - 1.0)) - 1.0 + visibility);
    var indirect_specular = environment * env_brdf(f0, roughness, n_dot_v) * specular_occlusion;
    if substrate {
        // Lumen traced the G-buffer's interface: the coat where there is one. The base layer
        // under a coat, or an anisotropic one, reflects about its own (bent) normal.
        var base_environment = environment;
        if (sh.features & (SLAB_CLEARCOAT | SLAB_ANISOTROPY)) != 0u {
            base_environment = specular_environment(reflect(-v, slab_reflection_normal(sh)), sh.roughness, gi_radiance, sky_visibility);
            if !has_feature(slab, SLAB_CLEARCOAT) {
                base_environment = mix(base_environment, reflection.rgb, reflection_weight);
            }
        }
        indirect_specular = slab_ambient_specular(sh, base_environment, environment, irradiance * occlusion) * specular_occlusion;
    }

    if frame.debug_view == DEBUG_SURFACE_CACHE {
        let level = sdf_finest_level(world_position);
        var cache = vec3<f32>(0.0);
        if level < lumen.level_count {
            cache = voxel_radiance(world_position + n * lumen.sdf_levels[level].w, level);
        }
        textureStore(output, pixel, vec4<f32>(cache, 1.0));
        return;
    }
    if frame.debug_view == DEBUG_INDIRECT {
        color = indirect / max(diffuse_color, vec3<f32>(0.02));
    } else if frame.debug_view == DEBUG_REFLECTIONS {
        color = indirect_specular;
    } else {
        let emitted = albedo * motion.z;
        color += indirect + indirect_specular + emitted;
        color = apply_aerial_perspective(color, world_position, view_dir);
        color = apply_volumetric_fog(color, uv, linear_depth);
        if substrate && sh.subsurface > 0.0 {
            // Both fogs apply affinely (c T + inscatter): their transmittance in front of the
            // pixel, which the scattered light still has to pass.
            let clear = apply_volumetric_fog(apply_aerial_perspective(vec3<f32>(0.0), world_position, view_dir), uv, linear_depth);
            let transmittance = apply_volumetric_fog(apply_aerial_perspective(vec3<f32>(1.0), world_position, view_dir), uv, linear_depth) - clear;
            store_subsurface(pixel, scattered * transmittance);
        }
    }
    textureStore(output, pixel, vec4<f32>(color, 1.0));
}

// Shadows of each pixel's ShadowedLights::mapped lights, into local_shadow_mask. A pass of its own (like UE's shadow projection into a screen-space mask):
// the filter's registers do not weigh on the occupancy of the shading.
@compute @workgroup_size(8, 8)
fn local_shadows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) || frame.light_count == 0u {
        return;
    }
    let pixel = gid.xy;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let noise = interleaved_gradient_noise(vec2<f32>(pixel), frame.frame_index);
    let selected = select_shadowed_lights(cluster_index(uv, frame.camera_position.w / depth) * CLUSTER_WORDS, world, n);
    var mask = 0u;
    for (var k = 0u; k < MASKED_LIGHTS; k++) {
        if selected.mapped[k] == NO_LIGHT {
            break;
        }
        let light = lights[selected.mapped[k]];
        let slot = light_shadow_slot(light);
        var shadow = 1.0;
        if k < FILTERED_LIGHTS {
            shadow = local_vsm_shadow(slot, world, n, light.color.w, noise, LOCAL_SHADOW_TAPS);
        } else {
            shadow = local_vsm_visibility(slot, world, n, 0u);
        }
        let dither = fract(noise + f32(k) * 0.618034);
        mask |= min(u32(saturate(shadow) * 15.0 + dither), 15u) << (4u * k);
    }
    local_shadow_mask[pixel.y * size.x + pixel.x] = mask;
}
