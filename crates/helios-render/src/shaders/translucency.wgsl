// Forward translucency: glass, water and alpha-blended surfaces, drawn after the deferred
// lighting over the lit HDR image (UE5's separate translucency), in per-pixel depth order where
// surfaces overlap: `fs_peel` finds the nearest layers' depths (depth peeling), and each shading
// pass composites the fragments of one depth range (`layer_range`) - one peeled layer, or
// everything behind them in per-instance order (see translucency.rs).
//
// Inputs are the opaque results: depth (tested here, in the fragment shader, since the merged
// hardware + software depth is not a depth attachment), the lit HDR image copied into
// `scene_color` with a few mips (refraction, blurred transmission, screen-space reflections),
// shadows (virtual shadow map, cloud shadows), clustered local lights, Lumen's radiance cache and
// distance field (reflection rays that leave the screen), clouds, volumetric fog and aerial
// perspective.
//
// SHADING selects the model (one pipeline each):
//   WATER : waves, refraction of the bottom, absorption + scattering by depth, caustics on the
//           bed, shore foam, reflections, sun glints. Output replaces the pixel (it already
//           contains the refracted background). Two-sided: from below (camera under water) the
//           surface shows the world above through Snell's window, total internal reflection
//           around it, and the water between the camera and the surface.
//   GLASS : stained / tinted glass: lead came is opaque, glass reflects and transmits the
//           (blurred, slightly refracted) scene behind it, tinted, plus light it diffuses.
//   BLEND : alpha-blended PBR (glTF BLEND), premultiplied "over" blending.
//   UNDERWATER : full-screen, drawn first when the camera is in a water body: the opaque scene
//           seen through the water (absorption + scattering along the view, caustics on the
//           bed). Water surfaces drawn after it replace what lies beyond them.
//
// Fog and aerial perspective: the background already carries the fog between the camera and
// itself. With A, B the fog transmittance / inscattering up to the surface and Tb the surface's
// transmittance for the background, out = A * surface + B * (1 - Tb) + Tb * background is exact
// for fog in front of the surface (and needs no blending with a separate fog term).

#include "common.wgsl"
#include "noise.wgsl"
#include "material.wgsl"
#include "svt_off.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "vsm_local_common.wgsl"
#include "vsm_local_sample.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "lights.wgsl"
#include "brdf.wgsl"
#include "clouds_common.wgsl"
#include "fog_common.wgsl"
#include "stars.wgsl"
#include "water.wgsl"
#include "glass_shadow.wgsl"
#include "megalights_volume_common.wgsl"
#include "avboit_common.wgsl"
#include "avboit_read.wgsl"
#include "avboit_write.wgsl"

override SHADING: u32 = 0u;
// The AVBOIT grid draw of refracting glass behind the nearest refracting layer (fs_grid_*).
override OIT_BEHIND_NEAREST: bool = false;
const SHADING_WATER: u32 = 0u;
const SHADING_GLASS: u32 = 1u;
const SHADING_BLEND: u32 = 2u;
const SHADING_UNDERWATER: u32 = 3u;

// Fraction of the light behind a stained-glass pane that passes (GLASS_TRANSMISSION in
// lumen_light.wgsl lights the interior with the same amount).
const GLASS_TRANSMISSION: f32 = 0.5;
// Share of the light through stained glass that the textured glass diffuses; the rest shows the
// scene behind, blurred (scene-colour mip GLASS_BLUR).
const GLASS_DIFFUSION: f32 = 0.7;
const GLASS_BLUR: f32 = 3.0;
const GLASS_PROBE_OFFSET: f32 = 2.5;
const WATER_F0: f32 = 0.02;
const GLASS_F0: f32 = 0.04;
// Period (seconds) of the river's two-phase flow map: how far the waves drift before a phase
// restarts.
const RIVER_FLOW_PERIOD: f32 = 1.6;
// A river's waves start at the fifth (1 m and shorter: no lake swell on a stream).
const RIVER_FIRST_WAVE: u32 = 4u;
// Glass reflectance above which reflection rays trace the distance field (grazing angles).
const GLASS_TRACED_REFLECTANCE: f32 = 0.1;
// fp16 render targets saturate at 65504; sun glints stay well below that.
const MAX_SPECULAR: f32 = 400.0;
// Screen-space reflections: march length (metres) and steps.
const SSR_DISTANCE: f32 = 600.0;
const SSR_STEPS: u32 = 40u;
// Background behind a translucent surface where only sky is visible (metres of water).
const OPEN_DEPTH: f32 = 200.0;
// Filter taps of local light shadow maps (lighting.wgsl uses 8).
const TRANSLUCENT_LOCAL_SHADOW_TAPS: u32 = 6u;
// `min_height` of reflection rays that may hit anything on screen.
const NO_FLOOR: f32 = -1.0e30;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform> water: Water;
@group(0) @binding(4) var depth_texture: texture_2d<f32>;
@group(0) @binding(5) var scene_color: texture_2d<f32>;
@group(0) @binding(6) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(7) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(8) var lut_sampler: sampler;
@group(0) @binding(9) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(10) var<uniform> vsm: Vsm;
@group(0) @binding(11) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(12) var vsm_atlas: texture_depth_2d;
@group(0) @binding(13) var shadow_sampler: sampler_comparison;
@group(0) @binding(14) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(15) var<storage, read> light_grid: array<u32>;
@group(0) @binding(16) var cloud_texture: texture_2d<f32>;
@group(0) @binding(17) var cloud_shadow_map: texture_2d<f32>;
@group(0) @binding(18) var<uniform> clouds: Clouds;
@group(0) @binding(19) var fog_volume: texture_3d<f32>;
@group(0) @binding(20) var<uniform> fog: Fog;
@group(0) @binding(21) var<uniform> lumen: Lumen;
@group(0) @binding(22) var global_sdf: texture_3d<f32>;
@group(0) @binding(23) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(24) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(25) var lumen_sampler: sampler;
@group(0) @binding(26) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(27) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(28) var material_albedo: texture_2d_array<f32>;
@group(0) @binding(29) var material_data: texture_2d_array<f32>;
@group(0) @binding(30) var material_sampler: sampler;
@group(0) @binding(31) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(60) var<uniform> local_vsm: LocalVsm;
// MegaLights' shadowed local light in froxels (megalights_volume.wgsl; range.w = 0 while off).
@group(0) @binding(130) var<uniform> ml_volume: MlVolume;
@group(0) @binding(131) var ml_volume_sh0: texture_3d<f32>;
@group(0) @binding(132) var ml_volume_sh1: texture_3d<f32>;
// The peeled layers' depths, nearest first, and the range of them this shading pass composites.
@group(0) @binding(105) var layers: texture_2d_array<f32>;
@group(0) @binding(106) var<uniform> layer_range: LayerRange;
// fs_peel: the previous layer's depth (1x1 for none: the reads are clamped).
@group(0) @binding(108) var peel_near: texture_2d<f32>;

// Fragments with layers[far] <= depth < layers[near] (reverse-Z), over a background whose nearest
// composited surface is layers[behind]; -1 stands for the opaque depth (far, behind) or no bound
// (near).
struct LayerRange {
    far: i32,
    near: i32,
    behind: i32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) color: vec4<f32>,
}

struct VertexOutput {
    // Invariant: the peeling and the shading pipelines must rasterise identical depths.
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) object_position: vec3<f32>,
    @location(2) previous_world: vec3<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) @interpolate(flat) instance: u32,
    @location(6) uv1: vec2<f32>,
    @location(7) color: vec4<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    // xy: motion vector, w: 1 where the surface should own the pixel's motion (blend weight).
    @location(1) motion: vec4<f32>,
}

// One draw per instance: `first_instance` is the instance index.
@vertex
fn vs_main(v: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    return transform_vertex(v, instance_index, frame.view_proj);
}

fn transform_vertex(v: VertexInput, instance_index: u32, view_proj: mat4x4<f32>) -> VertexOutput {
    let instance = instances[instance_index];
    let world = instance.world * vec4<f32>(v.position, 1.0);
    let normal_matrix = mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz);
    var out: VertexOutput;
    out.position = view_proj * world;
    out.world_position = world.xyz;
    out.object_position = v.position;
    out.previous_world = (instance.prev_world * vec4<f32>(v.position, 1.0)).xyz;
    out.normal = normal_matrix * v.normal;
    out.uv = v.uv;
    out.uv1 = v.uv1;
    out.color = v.color;
    out.instance = instance_index;
    return out;
}

// ---- Shared lighting -----------------------------------------------------------------------------

fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn screen_pixel(uv: vec2<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(uv * frame.viewport.xy), vec2<i32>(0), vec2<i32>(frame.viewport.xy) - 1);
}

// Texture coordinates of screen `uv` in the opaque scene copy at mip `lod`: the copy covers
// the internal viewport's corner of its targets (targets.rs); stay half a texel of the mip
// inside it.
fn scene_color_uv(uv: vec2<f32>, lod: f32) -> vec2<f32> {
    let margin = 0.5 * exp2(lod);
    let texel = clamp(uv * frame.viewport.xy, vec2<f32>(margin), max(frame.viewport.xy - margin, vec2<f32>(margin)));
    return texel / vec2<f32>(textureDimensions(scene_color));
}

fn opaque_depth(uv: vec2<f32>) -> f32 {
    return textureLoad(depth_texture, screen_pixel(uv), 0).r;
}

// Depth of peeled layer `index` at `pixel`, `fallback` for index -1.
fn layer_depth(index: i32, pixel: vec2<i32>, fallback: f32) -> f32 {
    if index < 0 {
        return fallback;
    }
    return textureLoad(layers, pixel, index, 0).r;
}

// Depth of the nearest surface composited at `uv` before this pass: the opaque scene or a
// translucent layer behind the ones being drawn.
fn background_depth(uv: vec2<f32>) -> f32 {
    let pixel = screen_pixel(uv);
    return max(textureLoad(depth_texture, pixel, 0).r, layer_depth(layer_range.behind, pixel, 0.0));
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    return sample_sky_view(sky_view_lut, lut_sampler, frame.camera_position.xyz, dir, sun) * frame.sun_illuminance.rgb;
}

fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    return sh_irradiance(sky_sh, n) * frame.sun_illuminance.rgb;
}

// Irradiance at `p` for normal `n`: Lumen's radiance cache (it knows walls and interiors),
// the sky without GI.
fn ambient_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let sky = sky_irradiance(n);
    if !has_flag(frame.flags, FLAG_GI) || lumen.sdf_enabled == 0u {
        return sky;
    }
    return radiance_cache_irradiance(p + n * 0.2, n, sky) * frame.gi_intensity;
}

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
    let clouds_visibility = cloud_shadow(world_position);
    if !has_flag(frame.flags, FLAG_SHADOWS) {
        return clouds_visibility;
    }
    return vsm_shadow(world_position, n, normalize(frame.sun_direction.xyz), noise) * clouds_visibility;
}

fn sun_light_at(p: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    return frame.sun_illuminance.rgb * sample_transmittance(transmittance_lut, lut_sampler, atmosphere_position(p), sun);
}

// Fog between the camera and a surface: out = color * transmittance + inscatter.
struct FogTerms {
    transmittance: vec3<f32>,
    inscatter: vec3<f32>,
}

fn fog_terms(p: vec3<f32>, uv: vec2<f32>, view_dir: vec3<f32>, linear_depth: f32) -> FogTerms {
    var f: FogTerms;
    f.transmittance = vec3<f32>(1.0);
    f.inscatter = vec3<f32>(0.0);
    // Aerial perspective (as in lighting.wgsl).
    if has_flag(frame.flags, FLAG_FOG) {
        let eye = frame.camera_position.xyz;
        let distance_km = distance(p, eye) * 0.001 * frame.misc.y;
        let height_km = max(0.5 * (eye.y + p.y), 0.0) * 0.001;
        let t = exp(-sample_medium(height_km).extinction * distance_km);
        f.transmittance = t;
        f.inscatter = sky_radiance(haze_direction(view_dir)) * (vec3<f32>(1.0) - t);
    }
    // Volumetric fog: slice z holds the integral up to the far end of the slice.
    if fog.enabled != 0u {
        let w = fog_depth_slice(linear_depth) - 0.5 / f32(textureDimensions(fog_volume).z);
        let v = textureSampleLevel(fog_volume, lut_sampler, vec3<f32>(uv, saturate(w)), 0.0);
        f.transmittance *= v.a;
        f.inscatter = f.inscatter * v.a + v.rgb;
    }
    return f;
}

// Direct light (sun + clustered local lights) on a GGX / Lambert surface.
struct DirectInput {
    p: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    diffuse: vec3<f32>,
    f0: vec3<f32>,
    roughness: f32,
    uv: vec2<f32>,
    linear_depth: f32,
    noise: f32,
    // Trace a distance-field shadow for the strongest local light.
    light_shadows: bool,
    // Sun visibility (virtual shadow map x cloud shadow), shared with the caller's other uses.
    sun_visibility: f32,
}

fn brdf(input: DirectInput, l: vec3<f32>, a: f32) -> vec3<f32> {
    let n_dot_l = saturate(dot(input.n, l));
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(input.n, input.v), 1e-4);
    let h = normalize(input.v + l);
    let f = f_schlick(input.f0, saturate(dot(input.v, h)));
    let specular = d_ggx(saturate(dot(input.n, h)), a) * v_smith_ggx_correlated(n_dot_v, n_dot_l, a) * f;
    let diffuse = input.diffuse / PI * (vec3<f32>(1.0) - f);
    return (diffuse + min(specular, vec3<f32>(MAX_SPECULAR))) * n_dot_l;
}

fn direct_light(input: DirectInput) -> vec3<f32> {
    let a = max(input.roughness * input.roughness, 0.002);
    let sun = normalize(frame.sun_direction.xyz);
    var color = vec3<f32>(0.0);
    if dot(input.n, sun) > 0.0 {
        // Through glass panes nearer the sun, in their colour (glass_shadow.wgsl).
        color = brdf(input, sun, a) * sun_light_at(input.p) * input.sun_visibility * glass_transmission_soft(input.p, lut_sampler, input.noise);
    }
    if frame.light_count == 0u {
        return color;
    }
    if ml_volume.range.w > 0.0 && input.linear_depth < ml_volume.range.y {
        // MegaLights: every light shadowed, from the froxel volume - a directional part through
        // the BRDF, the rest as diffuse light from all around (a quarter of it on average).
        let v = ml_volume_light(input.uv, input.linear_depth, lut_sampler);
        color += brdf(input, v.direction, a) * v.radiance * v.directional;
        return color + input.diffuse * (0.25 / PI) * v.radiance * (1.0 - v.directional);
    }
    // Local lights: those with a virtual shadow map use it (with fewer taps than lighting.wgsl;
    // translucent surfaces request no pages and fall back to coarser resident mips); of the
    // others only the strongest shadow-casting one is traced against the distance field
    // (translucent surfaces are a small part of the screen; lighting.wgsl traces four).
    let base = cluster_index(input.uv, input.linear_depth) * CLUSTER_WORDS;
    let count = light_grid[base];
    var strongest = INVALID_ID;
    if input.light_shadows && lumen.sdf_enabled != 0u {
        var strength = 0.0;
        for (var i = 0u; i < count; i++) {
            let index = light_grid[base + 1u + i];
            let light = lights[index];
            let ls = sample_light(light, input.p);
            let s = luminance(ls.radiance) * saturate(dot(input.n, ls.l));
            if light.params.z > 0.5 && light_shadow_slot(light) == INVALID_ID && s > strength {
                strength = s;
                strongest = index;
            }
        }
    }
    for (var i = 0u; i < count; i++) {
        let index = light_grid[base + 1u + i];
        let light = lights[index];
        let ls = sample_light(light, input.p);
        var radiance = ls.radiance;
        let slot = light_shadow_slot(light);
        if input.light_shadows && slot != INVALID_ID && dot(input.n, ls.l) > 0.0 {
            radiance *= local_vsm_shadow(slot, input.p, input.n, light.color.w, input.noise, TRANSLUCENT_LOCAL_SHADOW_TAPS);
        } else if index == strongest {
            radiance *= local_light_shadow(input.p + input.n * 0.04, light.position_range.xyz, light.color.w);
        }
        color += brdf(input, ls.l, a) * radiance;
    }
    return color;
}

// ---- Reflections -----------------------------------------------------------------------------------

struct ScreenHit {
    color: vec3<f32>,
    weight: f32,
}

// Screen-space reflection against the opaque depth: marches the ray's projection (depth is
// linear in screen space for a perspective projection) and returns the opaque scene's radiance
// where it passes behind a surface, faded towards the screen border. Hits below `min_height`
// are rejected: a ray leaving a water surface upwards never meets what lies under it,
// although the depth buffer shows the lake bed and the submerged rocks right there.
fn trace_screen(origin: vec3<f32>, dir: vec3<f32>, lod: f32, noise: f32, min_height: f32) -> ScreenHit {
    var hit: ScreenHit;
    hit.color = vec3<f32>(0.0);
    hit.weight = 0.0;
    let near = frame.camera_position.w;
    let c0 = frame.view_proj * vec4<f32>(origin, 1.0);
    let cd = frame.view_proj * vec4<f32>(dir, 0.0);
    var ray_length = SSR_DISTANCE;
    if cd.w < 0.0 {
        // Stop in front of the near plane.
        ray_length = min(ray_length, (c0.w - near * 2.0) / -cd.w);
    }
    if ray_length <= 0.0 {
        return hit;
    }
    let c1 = c0 + cd * ray_length;
    let s0 = vec3<f32>(clip_to_uv(c0), c0.z / c0.w);
    let s1 = vec3<f32>(clip_to_uv(c1), c1.z / c1.w);
    let d = s1 - s0;
    // Clip the segment to the screen.
    var t_end = 1.0;
    if abs(d.x) > 1e-7 {
        t_end = min(t_end, select(-s0.x / d.x, (1.0 - s0.x) / d.x, d.x > 0.0));
    }
    if abs(d.y) > 1e-7 {
        t_end = min(t_end, select(-s0.y / d.y, (1.0 - s0.y) / d.y, d.y > 0.0));
    }
    if t_end <= 0.0 {
        return hit;
    }
    // Steps grow towards the end of the segment: precision near the origin, reach far away.
    var previous = 0.0;
    for (var i = 0u; i < SSR_STEPS; i++) {
        let x = (f32(i) + noise) / f32(SSR_STEPS);
        let t = t_end * x * x;
        let s = s0 + d * t;
        let scene = opaque_depth(s.xy);
        if scene > s.z && s.z > 0.0 {
            // The ray passed behind an opaque surface; accept it if it is not far behind and the
            // surface faces the ray (otherwise it went behind a foreground object and goes on).
            let ray_linear = near / s.z;
            let scene_linear = near / scene;
            let step_linear = abs(ray_linear - near / max(s0.z + d.z * previous, 1e-9));
            let facing = dot(textureLoad(gbuffer_normal, screen_pixel(s.xy), 0).xyz, dir) < 0.1;
            let above = world_from_depth(s.xy, scene, frame.inv_view_proj).y >= min_height;
            if facing && above && ray_linear - scene_linear < max(step_linear, 0.1 + 0.02 * scene_linear) {
                // Refine between the last two samples.
                var lo = previous;
                var hi = t;
                for (var k = 0u; k < 5u; k++) {
                    let mid = 0.5 * (lo + hi);
                    let m = s0 + d * mid;
                    if opaque_depth(m.xy) > m.z {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let uv = (s0 + d * hi).xy;
                let border = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                hit.color = textureSampleLevel(scene_color, lut_sampler, scene_color_uv(uv, lod), lod).rgb;
                hit.weight = saturate(border / 0.06);
                return hit;
            }
        }
        previous = t;
    }
    return hit;
}

// The sky (and at night the stars) in direction `dir`, with clouds where that part of the sky is visible on screen
// (the cloud texture is only computed over sky pixels).
fn sky_with_clouds(dir: vec3<f32>) -> vec3<f32> {
    var sky = sky_radiance(dir) + starlight(dir);
    if clouds.enabled == 0u || dir.y <= 0.0 {
        return sky;
    }
    let c = frame.view_proj * vec4<f32>(dir, 0.0);
    if c.w <= 0.0 {
        return sky;
    }
    let uv = clip_to_uv(c);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || opaque_depth(uv) > 0.0 {
        return sky;
    }
    let cloud = textureSampleLevel(cloud_texture, lut_sampler, viewport_uv(uv, frame.viewport.xy * 0.5, textureDimensions(cloud_texture)), 0.0);
    return sky * cloud.a + cloud.rgb;
}

// Radiance arriving along a reflection ray: screen space first, then Lumen's global distance
// field and surface cache (or, cheaper, for faint reflections, the radiance cache's blurry
// view in that direction), then the sky. Rough reflections blend towards the ambient
// irradiance.
fn reflection(p: vec3<f32>, n: vec3<f32>, r: vec3<f32>, roughness: f32, noise: f32, distance_field: bool, min_height: f32) -> vec3<f32> {
    let lod = clamp(roughness * 12.0, 0.0, 4.0);
    let screen = trace_screen(p, r, lod, noise, min_height);
    if screen.weight >= 1.0 {
        return screen.color;
    }
    var far = sky_with_clouds(r);
    // Below the horizon the sky LUT only holds thin air: the lit ground instead.
    let ground = sky_irradiance(r) / PI;
    far = mix(far, ground, 1.0 - smoothstep(-0.12, 0.02, r.y));
    if has_flag(frame.flags, FLAG_GI) && lumen.sdf_enabled != 0u {
        if distance_field {
            let level = sdf_finest_level(p);
            let voxel = select(1.0, lumen.sdf_levels[min(level, lumen.level_count - 1u)].w, level < lumen.level_count);
            let sdf_hit = trace_sdf(p + n * voxel * 1.5, r, voxel, lumen.max_trace_distance);
            if sdf_hit.hit {
                let lighting_voxel = lumen.sdf_levels[sdf_hit.level].w * SDF_RES / LIGHT_RES;
                far = voxel_radiance(sdf_hit.position - r * lighting_voxel * 0.6, sdf_hit.level);
            }
        } else {
            far = radiance_cache_irradiance(p + n * 0.3, r, far * PI) / PI;
        }
    }
    if roughness > 0.3 {
        far = mix(far, ambient_irradiance(p, r) / PI, smoothstep(0.3, 0.7, roughness));
    }
    return mix(far, screen.color, screen.weight);
}

// ---- Surface reconstruction ------------------------------------------------------------------------

// Material inputs from the interpolated attributes and screen derivatives (computed up front,
// in uniform control flow).
fn surface_input(in: VertexOutput, front_facing: bool) -> SurfaceInput {
    // Derivatives per output pixel rather than per internal one: the mip bias of temporal
    // super resolution (texture detail as at the output resolution; TSR resolves it).
    return surface_input_seen(in, front_facing, frame.upscale.xy, normalize(frame.camera_position.xyz - in.world_position));
}

// `surface_input` for a view from direction `to_eye`, derivatives scaled by `gradient_scale`.
fn surface_input_seen(in: VertexOutput, front_facing: bool, gradient_scale: vec2<f32>, to_eye: vec3<f32>) -> SurfaceInput {
    let instance = instances[in.instance];
    var s: SurfaceInput;
    s.object_position = in.object_position;
    s.world_position = in.world_position;
    s.dpdx = dpdx(in.world_position) * gradient_scale.x;
    s.dpdy = dpdy(in.world_position) * gradient_scale.y;
    s.object_dpdx = dpdx(in.object_position) * gradient_scale.x;
    s.object_dpdy = dpdy(in.object_position) * gradient_scale.y;
    s.uv = in.uv;
    s.duv_dx = dpdx(in.uv) * gradient_scale.x;
    s.duv_dy = dpdy(in.uv) * gradient_scale.y;
    s.uv1 = in.uv1;
    s.duv1_dx = dpdx(in.uv1) * gradient_scale.x;
    s.duv1_dy = dpdy(in.uv1) * gradient_scale.y;
    s.footprint = max(length(s.dpdx), length(s.dpdy)) / max(instance.max_scale, 1e-4);
    s.normal_matrix = mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz);
    s.back_facing = !front_facing;
    var n = normalize(in.normal);
    if s.back_facing {
        n = -n;
    }
    n = normalize(n + to_eye * max(0.0, -dot(n, to_eye) + 0.05));
    s.normal = n;
    // Cotangent frame from screen derivatives of the material texture's coordinates (no stored
    // tangents); tangent-space +Y points to decreasing v, as in gbuffer.wgsl.
    let texture_uv = materials[instance.material].uv_texture;
    let duv_dx = uv_transform_vector(texture_uv, uv_set_of(texture_uv, s.duv_dx, s.duv1_dx));
    let duv_dy = uv_transform_vector(texture_uv, uv_set_of(texture_uv, s.duv_dy, s.duv1_dy));
    let dp2_perp = cross(s.dpdy, n);
    let dp1_perp = cross(n, s.dpdx);
    var tangent = dp2_perp * duv_dx.x + dp1_perp * duv_dy.x;
    var bitangent = dp2_perp * duv_dx.y + dp1_perp * duv_dy.y;
    tangent -= n * dot(n, tangent);
    if dot(tangent, tangent) < 1e-20 {
        tangent = basis(n)[0];
    }
    tangent = normalize(tangent);
    let handedness = select(1.0, -1.0, dot(cross(n, tangent), bitangent) < 0.0);
    s.tangent = tangent;
    s.bitangent = cross(n, tangent) * -handedness;
    return s;
}

// ---- Shading models ----------------------------------------------------------------------------------

struct Shaded {
    // Radiance leaving the surface itself (reflection, emission, scattering), before fog.
    surface: vec3<f32>,
    // Transmittance for the background (seen through the surface).
    transmittance: vec3<f32>,
    // Where to look up the background.
    background_uv: vec2<f32>,
    background_lod: f32,
    // Coverage for BLEND (premultiplied over); 1 for the refracting models.
    alpha: f32,
}

struct Pixel {
    uv: vec2<f32>,
    depth: f32,          // reverse-Z depth of the surface
    linear_depth: f32,
    opaque_depth: f32,   // reverse-Z depth of the opaque scene behind
    view_dir: vec3<f32>, // camera -> surface
    noise: f32,
}

// Background lookup displaced by `offset` (world space, at the background), unless that would
// sample something in front of the surface (opaque, or a translucent layer already composited).
fn refracted_uv(px: Pixel, behind: vec3<f32>, offset: vec3<f32>) -> vec2<f32> {
    let clip = frame.view_proj * vec4<f32>(behind + offset, 1.0);
    if clip.w <= 0.0 {
        return px.uv;
    }
    let uv = clamp(clip_to_uv(clip), vec2<f32>(0.0), vec2<f32>(1.0));
    if background_depth(uv) > px.depth {
        return px.uv;
    }
    return uv;
}

// Caustics on the opaque bed at `bed` (screen position `uv`, `depth` metres under the surface).
// They focus the direct sun only: the bed's lit radiance albedo / pi * (E_sun + E_ambient)
// becomes albedo / pi * (E_sun * caustics + E_ambient), i.e. it is scaled by
// (E_sun * caustics + E_ambient) / (E_sun + E_ambient) - no albedo needed. E_sun includes the
// bed's virtual-shadow-map and cloud shadows, so nothing is focused into a shadow.
fn bed_caustics(bed: vec3<f32>, uv: vec2<f32>, depth: f32, footprint: f32, ambient: vec3<f32>, noise: f32) -> f32 {
    if water.misc.y <= 0.0 || depth <= 0.0 {
        return 1.0;
    }
    let n = textureLoad(gbuffer_normal, screen_pixel(uv), 0).xyz;
    let sun = normalize(frame.sun_direction.xyz);
    let n_dot_l = dot(n, sun);
    if n_dot_l <= 0.0 || sun.y <= 0.0 {
        return 1.0;
    }
    let e_sun = luminance(sun_light_at(bed)) * n_dot_l * sun_shadow(bed, n, noise);
    let e_ambient = luminance(ambient);
    // In shadow there is nothing to focus (and the waves need not be evaluated).
    if e_sun <= 0.01 * e_ambient {
        return 1.0;
    }
    let c = water_caustics(bed, depth, frame.time, footprint);
    return (e_sun * c + e_ambient) / max(e_sun + e_ambient, 1e-6);
}

// Sun and sky light entering the water at `p` (on the surface); `sky`: the ambient irradiance there.
fn water_light(p: vec3<f32>, sun_visibility: f32, sky: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    return sun_light_at(p) * sun_visibility * saturate(sun.y) + sky;
}

// Light entering a water surface at `surface` (on it): the direct sun - its irradiance across
// the beam, through the surface's Fresnel transmission and the shadows there - and the sky and
// bounce light (Lumen's radiance cache, or the sky). They scatter differently, so they stay apart.
struct WaterLight {
    sun: vec3<f32>,
    sky: vec3<f32>,
}

fn water_light_entering(surface: vec3<f32>) -> WaterLight {
    let sun = normalize(frame.sun_direction.xyz);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var l: WaterLight;
    l.sun = vec3<f32>(0.0);
    if sun.y > 0.0 {
        l.sun = sun_light_at(surface) * sun_shadow(surface, up, 0.5) * (1.0 - fresnel_dielectric(sun.y, 1.0 / WATER_IOR));
    }
    l.sky = ambient_irradiance(surface, up);
    return l;
}

// The light lighting a view ray under water from `a` to `b`: what enters the surface above the
// ray's middle (one shadow and radiance-cache lookup per pixel), so the water darkens where the
// shore or a rock shades it, and the fog stays continuous between the surface draw and the
// full-screen draw, whose rays end at the same points.
fn water_light_along(a: vec3<f32>, b: vec3<f32>, level: f32) -> WaterLight {
    let middle = 0.5 * (a + b);
    return water_light_entering(vec3<f32>(middle.x, level, middle.z));
}

// The integral over t in [0, dist] of exp(-k * (depth - rise * t)) * exp(-extinction * t):
// light fading with depth below the surface (k per metre), seen through the water along a ray
// starting `depth` metres down and rising `rise` metres per metre.
fn water_depth_integral(k: vec3<f32>, depth: f32, rise: f32, dist: f32) -> vec3<f32> {
    let c = water_extinction() - k * rise;
    let span = select((vec3<f32>(1.0) - exp(-c * dist)) / c, vec3<f32>(dist), abs(c) < vec3<f32>(1e-4));
    return exp(-k * depth) * span;
}

// Phase function of lake water: a quarter isotropic (molecules, fine particles), the rest
// forward-peaked (larger particles; Henyey-Greenstein with asymmetry `g`). Natural waters scatter
// only a few percent backwards, so water seen sideways or downwards stays clear while looking
// towards the light glows.
fn water_phase(cos_theta: f32, g: f32) -> f32 {
    let hg = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
    return 0.25 / (4.0 * PI) + 0.75 * hg;
}

// Light scattered towards the eye along `dist` metres of a ray from `origin` in direction `dir`
// inside a water body whose surface is at `level`: single scattering of `light`, which fades
// with depth (sky light by the extinction per metre, the refracted sun beam along its slanted
// path). The depth changes linearly along the ray, so the integral has a closed form: looking up
// is brighter than looking down, and deep water is dark. The diffuse sky light comes down from
// the whole window above: it scatters as a vertical beam with a broad lobe; the sun's beam with
// a sharp one. `tint`: the water material's colour.
fn underwater_inscatter(origin: vec3<f32>, dir: vec3<f32>, dist: f32, level: f32, light: WaterLight, tint: vec3<f32>) -> vec3<f32> {
    let depth = max(level - origin.y, 0.0);
    let extinction = water_extinction();
    let sun = normalize(frame.sun_direction.xyz);
    let beam = refract(-sun, vec3<f32>(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
    let beam_cos = max(-beam.y, 0.2);
    let sky = light.sky * water_depth_integral(extinction, depth, dir.y, dist) * water_phase(dir.y, 0.5);
    let direct = light.sun * water_depth_integral(extinction / beam_cos, depth, dir.y, dist) * water_phase(dot(beam, -dir), 0.8);
    return water.scattering.rgb * tint * (sky + direct);
}

// Radiance arriving along the reflection ray of a glass or blended surface: what `reflection`
// traces, or under water the deep water around it (as shade_water_below shows it by total
// internal reflection), not the sky.
fn surface_reflection(p: vec3<f32>, n: vec3<f32>, r: vec3<f32>, roughness: f32, noise: f32, distance_field: bool) -> vec3<f32> {
    let level = water_level_at(p.xz);
    if p.y < level {
        return underwater_inscatter(p, r, OPEN_DEPTH, level, water_light_entering(vec3<f32>(p.x, level, p.z)), vec3<f32>(1.0));
    }
    return reflection(p, n, r, roughness, noise, distance_field, NO_FLOOR);
}

fn shade_water(in: VertexOutput, input: SurfaceInput, m: Material, px: Pixel) -> Shaded {
    if input.back_facing {
        return shade_water_below(in, input, m, px);
    }
    let p = in.world_position;
    let v = -px.view_dir;
    let footprint = max(length(input.dpdx.xz), length(input.dpdy.xz));
    let up = input.normal;
    var waves: WaterSurface;
    var n: vec3<f32>;
    // A simulated river (river.rs) marks its vertices with alpha 0 and carries its velocity in
    // the second UV set and its foam in red.
    let simulated = in.color.a < 0.5;
    let flow = select(vec2<f32>(0.0), in.uv1, simulated);
    let flow_phase = fract(frame.time / RIVER_FLOW_PERIOD);
    // Weight of the second phase: 0 where it restarts, 1 where the first does.
    let flow_blend = abs(2.0 * flow_phase - 1.0);
    if simulated {
        // Two-phase flow map: the waves drift with the current, each phase restarting from
        // the undisplaced pattern while the other carries the look; ripples grow choppier with
        // the speed and calm down in pools.
        let w0 = water_surface_from(p.xz - flow * (flow_phase * RIVER_FLOW_PERIOD), frame.time, footprint, RIVER_FIRST_WAVE);
        let w1 = water_surface_from(p.xz - flow * (fract(flow_phase + 0.5) * RIVER_FLOW_PERIOD), frame.time, footprint, RIVER_FIRST_WAVE);
        let wn = normalize(mix(w0.normal, w1.normal, flow_blend));
        let chop = clamp(0.35 + 0.5 * length(flow), 0.35, 1.4);
        waves.normal = normalize(vec3<f32>(wn.x * chop, wn.y, wn.z * chop));
        waves.slope_variance = mix(w0.slope_variance, w1.slope_variance, flow_blend) * chop * chop;
        // The surface slopes downstream: tilt its normal by the waves.
        n = normalize(up * waves.normal.y + vec3<f32>(waves.normal.x, 0.0, waves.normal.z));
    } else {
        waves = water_surface(p.xz, frame.time, footprint);
        // Flat geometry: the wave normal lives in the surface's frame.
        let n_waves = normalize(waves.normal.x * input.tangent - waves.normal.z * input.bitangent + waves.normal.y * up);
        n = select(waves.normal, n_waves, abs(up.y) < 0.99);
    }
    // Never let a wave face away from the viewer (grazing views).
    n = normalize(n + v * max(0.0, 0.02 - dot(n, v)));
    // Running water is rougher than its resolved waves: the faster, the more small-scale chop
    // breaks up its reflections (a still pool stays a mirror).
    let speed = length(flow);
    let roughness = sqrt(m.roughness * m.roughness + waves.slope_variance * 2.0 + 0.03 * speed * speed);
    let n_dot_v = max(dot(n, v), 1e-4);

    // Water column behind the surface: refract the lookup, then measure the path to the bottom.
    let bottom0 = select(p + px.view_dir * OPEN_DEPTH, world_from_depth(px.uv, px.opaque_depth, frame.inv_view_proj), px.opaque_depth > 0.0);
    let depth0 = clamp(dot(p - bottom0, up), 0.0, OPEN_DEPTH);
    let bend = (n - up) * min(depth0, 4.0) * 0.6;
    let uv = refracted_uv(px, bottom0, bend);
    let behind_depth = opaque_depth(uv);
    let bottom = select(p + px.view_dir * OPEN_DEPTH, world_from_depth(uv, behind_depth, frame.inv_view_proj), behind_depth > 0.0);
    let path = min(distance(p, bottom), OPEN_DEPTH);
    let vertical = clamp(dot(p - bottom, up), 0.0, OPEN_DEPTH);
    // Light reaching the bottom crossed the water too (roughly vertically).
    let to_bottom = exp(-water_extinction() * (path + vertical));

    // Light scattered back out of the water column.
    let sun_visibility = sun_shadow(p, up, px.noise);
    let sky_in = ambient_irradiance(p, up);
    let light = water_light(p, sun_visibility, sky_in);
    let column = water_inscatter(path, light, m.base_color.rgb);

    // Caustics on the bed, at the bed's pixel footprint (the surface's, spread over the path).
    var caustics = 1.0;
    if behind_depth > 0.0 {
        let bed_footprint = footprint * (1.0 + path / max(distance(p, frame.camera_position.xyz), 1e-3));
        caustics = bed_caustics(bottom, uv, vertical, bed_footprint, sky_in, px.noise);
    }

    // Shore foam: a white, rough layer floating on top (on a river only its thin line).
    var foam = water_foam(p.xz, depth0, frame.time, footprint) * select(1.0, 0.4, simulated);
    if simulated && in.color.r > 0.01 {
        // The simulation's foam as a lace of thin streaks drifting with the current (the ridges
        // of a noise), patches only where it is thick.
        let s0 = fbm(vec3<f32>((p.xz - flow * (flow_phase * RIVER_FLOW_PERIOD)) * 1.8, 5.1), 3u, footprint * 1.8);
        let s1 = fbm(vec3<f32>((p.xz - flow * (fract(flow_phase + 0.5) * RIVER_FLOW_PERIOD)) * 1.8, 9.7), 3u, footprint * 1.8);
        let pattern = mix(s0, s1, flow_blend);
        let amount = saturate(in.color.r);
        // Lines thicken with the amount; sub-pixel lines fade to their coverage.
        let width = 0.02 + 0.1 * amount + footprint * 0.5;
        let lace = (1.0 - smoothstep(0.0, width, abs(pattern))) * smoothstep(0.1, 0.5, amount);
        let patches = smoothstep(0.8, 1.0, amount) * smoothstep(0.1, 0.3, pattern);
        foam = max(foam, max(lace * 0.6, patches * 0.75));
    }
    let foam_light = light / PI * 0.8;

    let fresnel = f_schlick(vec3<f32>(WATER_F0), n_dot_v).x;
    let r = reflect(-v, n);
    var specular = reflection(p, n, r, roughness, px.noise, true, p.y - 0.05) * env_brdf(vec3<f32>(WATER_F0), roughness, n_dot_v);
    // Sun glints.
    var direct: DirectInput;
    direct.p = p;
    direct.n = n;
    direct.v = v;
    direct.diffuse = vec3<f32>(0.0);
    direct.f0 = vec3<f32>(WATER_F0);
    direct.roughness = roughness;
    direct.uv = px.uv;
    direct.linear_depth = px.linear_depth;
    direct.noise = px.noise;
    direct.light_shadows = true;
    direct.sun_visibility = sun_visibility;
    specular += direct_light(direct);

    // Fade in from the shore line so the intersection with the ground stays soft.
    let edge = smoothstep(0.0, 0.15, depth0);
    var s: Shaded;
    let open = (1.0 - fresnel) * (1.0 - foam);
    s.surface = (specular * (1.0 - foam) + column * open + foam * foam_light) * edge;
    s.transmittance = mix(vec3<f32>(1.0), to_bottom * open * caustics, edge);
    s.background_uv = uv;
    s.background_lod = clamp(log2(1.0 + path * 0.5), 0.0, 3.0) * edge;
    s.alpha = 1.0;
    return s;
}

// The surface seen from below (the camera is in the water). Light from above refracts into
// Snell's window - the whole sky and shore squeezed into a cone of 48.6 degrees around the
// normal (radiance would also rise by IOR^2 as it is compressed into the denser medium; that is
// left out, it only saturates the window) - and beyond the critical angle the surface totally
// reflects the water beneath. Then the water between the
// camera and the surface absorbs and scatters (no air fog: fs_main skips it).
fn shade_water_below(in: VertexOutput, input: SurfaceInput, m: Material, px: Pixel) -> Shaded {
    let p = in.world_position;
    let footprint = max(length(input.dpdx.xz), length(input.dpdy.xz));
    let waves = water_surface(p.xz, frame.time, footprint);
    let n = waves.normal;
    let roughness = sqrt(m.roughness * m.roughness + waves.slope_variance * 2.0);
    let i = px.view_dir;
    let reflectance = fresnel_dielectric(saturate(dot(i, n)), WATER_IOR);

    let eye = frame.camera_position.xyz;
    let level = p.y;
    let light = water_light_along(eye, p, level);
    // Reflected: the water body below, seen through a long path (its scattering only).
    var color = underwater_inscatter(p, reflect(i, -n), OPEN_DEPTH, level, light, m.base_color.rgb) * reflectance;
    if reflectance < 1.0 {
        let t = refract(i, -n, WATER_IOR);
        color += reflection(p, n, t, roughness, px.noise, true, p.y - 0.05) * (1.0 - reflectance);
    }

    let d = distance(p, eye);
    var s: Shaded;
    s.surface = color * exp(-water_extinction() * d) + underwater_inscatter(eye, i, d, level, light, m.base_color.rgb);
    s.transmittance = vec3<f32>(0.0);
    s.background_uv = px.uv;
    s.background_lod = 0.0;
    s.alpha = 1.0;
    return s;
}

fn shade_glass(in: VertexOutput, input: SurfaceInput, m: Material, px: Pixel) -> Shaded {
    let p = in.world_position;
    let v = -px.view_dir;
    let g = evaluate_glass(m, input);
    let coverage = saturate(g.emissive);
    let n = g.normal;
    let n_dot_v = max(dot(n, v), 1e-4);
    // Stained (textured) glass is hammered: it diffuses most of the light it transmits and
    // blurs the rest. Untextured glass is a clear pane transmitting its base colour, blurred by
    // its roughness.
    let stained = m.texture != NO_TEXTURE;
    // Hammered glass normals alias under sharp highlights; keep the lobe a little wide.
    let roughness = select(g.roughness, max(g.roughness, 0.12), stained);

    var direct: DirectInput;
    direct.p = p;
    direct.n = n;
    direct.v = v;
    direct.diffuse = g.albedo * (1.0 - coverage);
    direct.f0 = vec3<f32>(GLASS_F0);
    direct.roughness = roughness;
    direct.uv = px.uv;
    direct.linear_depth = px.linear_depth;
    direct.noise = px.noise;
    // Panes sit in walls, away from the lights: no light shadows.
    direct.light_shadows = false;
    direct.sun_visibility = sun_shadow(p, n, px.noise);
    var color = direct_light(direct);
    // Lead came: diffuse ambient; both: reflections.
    color += direct.diffuse * ambient_irradiance(p, n) / PI;
    let r = reflect(-v, n);
    // Reflections are faint (a few percent) where the pane is seen face on: the radiance
    // cache's blurry view stands in for the distance-field trace. At grazing angles they are
    // not: a pane's edge seen from just above (its top face, a few centimetres wide) reflects
    // half the light, and the cache's hemisphere around a near-horizontal ray is half sky -
    // it drew a bright line across the landscape behind the pane where the reflection should
    // show that landscape. Strong reflections trace the distance field like water's.
    let reflectance = env_brdf(vec3<f32>(GLASS_F0), roughness, n_dot_v);
    let traced = reflectance.x > GLASS_TRACED_REFLECTANCE;
    color += surface_reflection(p, n, r, roughness, px.noise, traced) * reflectance;

    let fresnel = f_schlick(vec3<f32>(GLASS_F0), n_dot_v).x;
    let tint = g.albedo * select(1.0, GLASS_TRANSMISSION, stained) * (1.0 - fresnel) * coverage;
    let diffusion = select(0.0, GLASS_DIFFUSION, stained);
    if diffusion > 0.0 {
        // Textured glass diffuses most of the light arriving on its far side: skylight, as
        // Lumen's surface cache assumes for the window (lumen_light.wgsl), scaled by how open
        // the far side is - the radiance cache a little beyond the wall (not averaged with the
        // near side's probes) against the open sky: dark from outside a lit interior - plus
        // direct sun. The rest shows the scene behind, blurred.
        let back = -input.normal;
        let sun = normalize(frame.sun_direction.xyz);
        let sky_behind = sky_irradiance(back);
        let cached = ambient_irradiance(p + back * GLASS_PROBE_OFFSET, back);
        var behind = sky_behind * saturate(2.0 * luminance(cached) / max(luminance(sky_behind), 1e-4));
        if dot(back, sun) > 0.0 {
            behind += sun_light_at(p) * sun_shadow(p, back, px.noise) * dot(back, sun) * glass_transmission_soft(p, lut_sampler, px.noise);
        }
        color += tint * diffusion * behind / PI;
    }

    var s: Shaded;
    s.surface = color;
    s.transmittance = tint * (1.0 - diffusion);
    // Thin pane: the texture's relief displaces the view a little; hammered glass blurs it.
    s.background_uv = refracted_uv(px, p + px.view_dir * 0.5, (n - input.normal) * 0.05);
    s.background_lod = select(clamp(roughness * 16.0 - 1.0, 0.0, 4.0), GLASS_BLUR, stained);
    s.alpha = 1.0;
    return s;
}

fn shade_blend(in: VertexOutput, input: SurfaceInput, m: Material, px: Pixel) -> Shaded {
    let p = in.world_position;
    let v = -px.view_dir;
    let surface = evaluate_standard(m, input);
    var alpha = m.base_color.a;
    if m.texture != NO_TEXTURE && m.mapping == MAPPING_UV {
        let c = texture_coords(m.uv_texture, input);
        alpha *= textureSampleGrad(material_albedo, material_sampler, c.uv, m.texture, c.ddx, c.ddy).a;
    }
    let n = surface.normal;
    let n_dot_v = max(dot(n, v), 1e-4);
    let roughness = clamp(surface.roughness, 0.045, 1.0);
    let diffuse = surface.albedo * (1.0 - surface.metallic);
    let f0 = mix(vec3<f32>(0.04), surface.albedo, surface.metallic);

    var direct: DirectInput;
    direct.p = p;
    direct.n = n;
    direct.v = v;
    direct.diffuse = diffuse;
    direct.f0 = f0;
    direct.roughness = roughness;
    direct.uv = px.uv;
    direct.linear_depth = px.linear_depth;
    direct.noise = px.noise;
    direct.light_shadows = true;
    direct.sun_visibility = sun_shadow(p, n, px.noise);
    var color = direct_light(direct);
    // Material ambient occlusion darkens the indirect terms only.
    color += diffuse * ambient_irradiance(p, n) * surface.occlusion / PI;
    color += surface_reflection(p, n, reflect(-v, n), roughness, px.noise, true) * env_brdf(f0, roughness, n_dot_v) * surface.occlusion;
    color += surface.albedo * surface.emissive;

    var s: Shaded;
    s.surface = color;
    s.transmittance = vec3<f32>(0.0);
    s.background_uv = px.uv;
    s.background_lod = 0.0;
    s.alpha = saturate(alpha);
    return s;
}

// A translucent fragment shaded: the model's result, the fog in front of it, the pixel.
struct Fragment {
    s: Shaded,
    f: FogTerms,
    px: Pixel,
}

// Shades the fragment of this pass's depth range (discards the others).
fn shade_fragment(in: VertexOutput, front_facing: bool) -> Fragment {
    // Derivatives first, while control flow is uniform.
    let input = surface_input(in, front_facing);

    var px: Pixel;
    let pixel = vec2<i32>(in.position.xy);
    px.uv = in.position.xy * frame.viewport.zw;
    px.depth = in.position.z;
    px.opaque_depth = textureLoad(depth_texture, pixel, 0).r;
    // Reverse-Z depth test against the opaque scene, and this pass's depth range.
    if px.depth < max(px.opaque_depth, layer_depth(layer_range.far, pixel, 0.0)) || px.depth >= layer_depth(layer_range.near, pixel, 2.0) {
        discard;
    }
    let eye = frame.camera_position.xyz;
    px.view_dir = normalize(in.world_position - eye);
    px.linear_depth = frame.camera_position.w / max(px.depth, 1e-9);
    px.noise = interleaved_gradient_noise(in.position.xy, frame.frame_index);

    // glTF: the vertex colour multiplies the base colour (and its alpha the coverage).
    var m = materials[instances[in.instance].material];
    m.base_color *= in.color;
    var s: Shaded;
    switch SHADING {
        case SHADING_WATER: { s = shade_water(in, input, m, px); }
        case SHADING_GLASS: { s = shade_glass(in, input, m, px); }
        default: { s = shade_blend(in, input, m, px); }
    }

    var f = fog_terms(in.world_position, px.uv, px.view_dir, px.linear_depth);
    if SHADING == SHADING_WATER && !front_facing {
        // Seen from below: water, not air, lies between the camera and the surface.
        f.transmittance = vec3<f32>(1.0);
        f.inscatter = vec3<f32>(0.0);
    } else if water.misc.w >= 0.0 && in.world_position.y < water.levels[i32(water.misc.w)] {
        // The camera and the surface are under water. As fs_underwater does for the opaque
        // scene: the light reaching the surface crossed the water above it, and the water
        // between the surface and the camera fogs it.
        let level = water.levels[i32(water.misc.w)];
        let below = level - in.world_position.y;
        s.surface *= exp(-water_extinction() * below);
        let d = distance(in.world_position, eye);
        f.transmittance = exp(-water_extinction() * d);
        f.inscatter = underwater_inscatter(eye, px.view_dir, d, level, water_light_along(eye, in.world_position, level), vec3<f32>(1.0));
    }
    return Fragment(s, f, px);
}

// The surface's own motion, owning the pixel's where it is mostly opaque.
fn surface_motion(in: VertexOutput, alpha: f32) -> vec4<f32> {
    let motion = clip_to_uv(frame.unjittered_view_proj * vec4<f32>(in.world_position, 1.0))
        - clip_to_uv(frame.prev_unjittered_view_proj * vec4<f32>(in.previous_world, 1.0));
    return vec4<f32>(motion, 0.0, select(0.0, 1.0, alpha >= 0.5));
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
    let fragment = shade_fragment(in, front_facing);
    let s = fragment.s;
    let f = fragment.f;
    var out: FragmentOutput;
    if SHADING == SHADING_BLEND {
        out.color = vec4<f32>((s.surface * f.transmittance + f.inscatter) * s.alpha, s.alpha);
    } else {
        let background = textureSampleLevel(scene_color, lut_sampler, scene_color_uv(s.background_uv, s.background_lod), s.background_lod).rgb;
        let t = s.transmittance;
        out.color = vec4<f32>(s.surface * f.transmittance + f.inscatter * (vec3<f32>(1.0) - t) + background * t, 1.0);
    }
    out.motion = surface_motion(in, s.alpha);
    return out;
}

// ---- Adaptive voxel-based OIT (avboit.rs) ----------------------------------------------------------
//
// THE AVBOIT MATERIAL CONTRACT (as UE's): a surface composited order-independently contributes
// only its OWN radiance - lighting, reflections, emission, the fog in front of it - and its
// transmittance. It never reads the scene colour behind it (`s.background_uv`, refraction, the
// blurred view through hammered glass): the composite multiplies the background by the summed
// transmittance once, and a surface that also added its own read of the background would add it
// twice (washed out). Water, whose model folds the refracted bed into its radiance, and the
// nearest refracting glass layer therefore stay on the depth-peeled path (translucency.rs);
// glass and blended surfaces composited here keep only `surface` and `transmittance`.

// Water (SHADING_WATER) is never drawn with these pipelines.
fn oit_light(fragment: Fragment) -> vec4<f32> {
    let s = fragment.s;
    let f = fragment.f;
    if SHADING == SHADING_BLEND {
        return vec4<f32>((s.surface * f.transmittance + f.inscatter) * s.alpha, 1.0 - s.alpha);
    }
    let t = s.transmittance;
    return vec4<f32>(s.surface * f.transmittance + f.inscatter * (vec3<f32>(1.0) - t), 0.0);
}

fn oit_transmittance(fragment: Fragment) -> vec3<f32> {
    if SHADING == SHADING_BLEND {
        return vec3<f32>(1.0 - fragment.s.alpha);
    }
    return fragment.s.transmittance;
}

// The view depth a surface fragment looks its weight up at: the grid holds the surface at the
// depth of its column's centre (low resolution), so a surface steep in depth reaches cells in
// front of and behind the fragment within the columns the bilinear lookup reads. Looking up in
// front of the nearest of them keeps the surface from dimming itself (streaks along the columns).
fn oit_depth(fragment: Fragment, slope: f32) -> f32 {
    let d = fragment.px.linear_depth;
    return max(d - slope * f32(avboit.grid.w) * 1.5, d * 0.5);
}

struct OitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) transmittance: vec4<f32>,
    @location(2) motion: vec4<f32>,
}

// Full resolution, one bucket of surfaces, chosen by the pass's `layer_range` (one pipeline for
// both buckets: the browser compiles one per shading model): the own light weighted by the
// transmittance in front of the fragment (added), the transmittance (multiplied; alpha 1: not a
// particle), the motion.
// * Front bucket (`near` -1: at or in front of the nearest refracting surface, layers[0]):
//   weighted by everything in front of the fragment; owns the pixel's motion where mostly opaque.
// * Back bucket (`near` 0: behind it): weighted by what lies between the split and the fragment
//   only (the front composite dims the rest); motion weight 0 (the blend keeps the target's).
@fragment
fn fs_oit(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> OitOutput {
    let slope = fwidth(frame.camera_position.w / max(in.position.z, 1e-9));
    let fragment = shade_fragment(in, front_facing);
    let back = layer_range.near >= 0;
    var split = 0.0;
    if back {
        split = frame.camera_position.w / max(layer_depth(layer_range.near, vec2<i32>(in.position.xy), 1.0), 1e-9);
    }
    let w = avboit_weight(in.position.xy, oit_depth(fragment, slope), oit_transmittance(fragment), split);
    var out: OitOutput;
    out.accum = vec4<f32>(oit_light(fragment).rgb * w, 0.0);
    out.transmittance = vec4<f32>(oit_transmittance(fragment), 1.0);
    out.motion = surface_motion(in, fragment.s.alpha);
    if back {
        out.motion.w = 0.0;
    }
    return out;
}

// Low resolution, one fragment per grid column (its centre pixel tests the opaque depth):
// whether the fragment is in the grid. With OIT_BEHIND_NEAREST (refracting glass behind the
// nearest refracting layer), the nearest layer itself is not: the depth-peeled pass draws it.
fn grid_fragment(position: vec4<f32>) -> bool {
    let column = vec2<u32>(position.xy);
    let pixel = vec2<i32>(column * avboit.grid.w + avboit.grid.w / 2u);
    if position.z < textureLoad(depth_texture, pixel, 0).r {
        return false;
    }
    if OIT_BEHIND_NEAREST && position.z >= textureLoad(peel_near, pixel, 0).r * 0.999 {
        return false;
    }
    return true;
}

@fragment
fn fs_grid_bounds(in: VertexOutput) {
    if grid_fragment(in.position) {
        avboit_extend(avboit_column_index(vec2<u32>(in.position.xy)), frame.camera_position.w / max(in.position.z, 1e-9));
    }
}

// The transmittance `shade_glass` / `shade_blend` give the background, without the lighting.
@fragment
fn fs_grid_transmittance(in: VertexOutput, @builtin(front_facing) front_facing: bool) {
    let input = surface_input(in, front_facing);
    if !grid_fragment(in.position) {
        return;
    }
    var m = materials[instances[in.instance].material];
    m.base_color *= in.color;
    var t = vec3<f32>(1.0);
    if SHADING == SHADING_GLASS {
        let g = evaluate_glass(m, input);
        let stained = m.texture != NO_TEXTURE;
        let n_dot_v = max(dot(g.normal, normalize(frame.camera_position.xyz - in.world_position)), 1e-4);
        let fresnel = f_schlick(vec3<f32>(GLASS_F0), n_dot_v).x;
        let tint = g.albedo * select(1.0, GLASS_TRANSMISSION, stained) * (1.0 - fresnel) * saturate(g.emissive);
        t = tint * (1.0 - select(0.0, GLASS_DIFFUSION, stained));
    } else {
        var alpha = m.base_color.a;
        if m.texture != NO_TEXTURE && m.mapping == MAPPING_UV {
            let c = texture_coords(m.uv_texture, input);
            alpha *= textureSampleGrad(material_albedo, material_sampler, c.uv, m.texture, c.ddx, c.ddy).a;
        }
        t = vec3<f32>(1.0 - saturate(alpha));
    }
    avboit_add(avboit_column_index(vec2<u32>(in.position.xy)), frame.camera_position.w / max(in.position.z, 1e-9), t);
}

// ---- Layer peeling ---------------------------------------------------------------------------------

// Depth-only: the nearest fragment in front of the opaque scene and behind the previous layer
// (`peel_near`) wins the depth test. Writing the interpolated depth explicitly makes the stored
// value exactly what the shading passes compare against.
@fragment
fn fs_peel(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    let pixel = vec2<i32>(position.xy);
    let near = textureLoad(peel_near, min(pixel, vec2<i32>(textureDimensions(peel_near)) - 1), 0).r;
    if position.z < textureLoad(depth_texture, pixel, 0).r || position.z >= near {
        discard;
    }
    return position.z;
}

// ---- Underwater view -------------------------------------------------------------------------------

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

// The opaque scene seen from inside the water body holding the camera: absorption and
// scattering along the view up to the opaque surface or to where the ray leaves through the
// water surface (that surface's own draw shades what lies beyond it), and on the bed the light
// that came down through the water, focused into caustics.
@fragment
fn fs_underwater(@builtin(position) position: vec4<f32>) -> FragmentOutput {
    let pixel = vec2<i32>(position.xy);
    let uv = position.xy * frame.viewport.zw;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let eye = frame.camera_position.xyz;
    let level = water.levels[max(i32(water.misc.w), 0)];
    let view_dir = view_direction(uv, frame.inv_view_proj, eye);
    let noise = interleaved_gradient_noise(position.xy, frame.frame_index);
    let extinction = water_extinction();
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sky_in = ambient_irradiance(vec3<f32>(eye.x, level, eye.z), up);

    var background = textureLoad(scene_color, pixel, 0).rgb;
    var out: FragmentOutput;
    // Weight 0: the opaque scene keeps its own motion vectors.
    out.motion = vec4<f32>(0.0);
    var dist = OPEN_DEPTH;
    var p = eye + view_dir * dist;
    if depth > 0.0 {
        p = world_from_depth(uv, depth, frame.inv_view_proj);
        dist = distance(p, eye);
    }
    if view_dir.y > 1e-4 {
        let exit = (level - eye.y) / view_dir.y;
        let body = water.bodies[max(i32(water.misc.w), 0)];
        let crossing = eye.xz + view_dir.xz * exit;
        if exit < dist && all(crossing >= body.xy) && all(crossing <= body.zw) {
            // The view leaves the water through its surface, whose own draw replaces this pixel
            // with the world above. The scene stays unfogged here for that draw's screen trace
            // (which reads a scene-colour copy taken after this pass when layers are peeled).
            out.color = vec4<f32>(background, 1.0);
            return out;
        }
        dist = min(dist, exit);
    }
    let below = level - p.y;
    if depth > 0.0 && below > 0.0 {
        // An output pixel's footprint at this distance (from the neighbouring pixel's view ray;
        // output, not internal, pixels: the mip bias of TSR, as for the surface derivatives).
        let footprint = dist * length(view_direction(uv + vec2<f32>(0.0, 1.0 / frame.upscale.w), frame.inv_view_proj, eye) - view_dir);
        background *= exp(-extinction * below) * bed_caustics(p, uv, below, footprint, sky_in, noise);
    }
    let light = water_light_along(eye, eye + view_dir * dist, level);
    out.color = vec4<f32>(background * exp(-extinction * dist) + underwater_inscatter(eye, view_dir, dist, level, light, vec3<f32>(1.0)), 1.0);
    return out;
}

// ---- Sunlight through glass ------------------------------------------------------------------------

// The glass panes seen from the sun (glass_shadow.rs): one draw per pane, as `vs_main`.
@vertex
fn vs_glass_shadow(v: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    return transform_vertex(v, instance_index, glass_shadow.view_proj);
}

// The fraction of the sunlight a pane passes, as `shade_glass` transmits it (the light the
// hammered glass diffuses included: it still arrives, a little blurred - see
// GLASS_SHADOW_SPREAD), and its depth along the sunlight. Lead came passes nothing.
@fragment
fn fs_glass_shadow(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    let input = surface_input_seen(in, front_facing, vec2<f32>(1.0), sun);
    var m = materials[instances[in.instance].material];
    m.base_color *= in.color;
    let g = evaluate_glass(m, input);
    let coverage = saturate(g.emissive);
    let stained = m.texture != NO_TEXTURE;
    let cos_incidence = max(abs(dot(normalize(in.normal), sun)), 1e-4);
    let fresnel = f_schlick(vec3<f32>(GLASS_F0), cos_incidence).x;
    let tint = g.albedo * select(1.0, GLASS_TRANSMISSION, stained) * (1.0 - fresnel) * coverage;
    return vec4<f32>(tint, in.position.z);
}

// ---- Local lights through glass ----------------------------------------------------------------

// One face of a light's glass cube map (glass_light.rs).
struct GlassLightFace {
    view_proj: mat4x4<f32>,
    // xyz: the light's position.
    light: vec4<f32>,
}

@group(0) @binding(95) var<uniform> glass_face: GlassLightFace;

@vertex
fn vs_glass_light(v: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    return transform_vertex(v, instance_index, glass_face.view_proj);
}

// As `fs_glass_shadow`, for the light at `glass_face.light`: the pane's transmittance towards it
// and its distance from it.
@fragment
fn fs_glass_light(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let to_light = glass_face.light.xyz - in.world_position;
    let l = normalize(to_light);
    let input = surface_input_seen(in, front_facing, vec2<f32>(1.0), l);
    var m = materials[instances[in.instance].material];
    m.base_color *= in.color;
    let g = evaluate_glass(m, input);
    let coverage = saturate(g.emissive);
    let stained = m.texture != NO_TEXTURE;
    let cos_incidence = max(abs(dot(normalize(in.normal), l)), 1e-4);
    let fresnel = f_schlick(vec3<f32>(GLASS_F0), cos_incidence).x;
    let tint = g.albedo * select(1.0, GLASS_TRANSMISSION, stained) * (1.0 - fresnel) * coverage;
    return vec4<f32>(tint, length(to_light));
}
