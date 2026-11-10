// Lumen mesh-card surface cache: lighting and debug view (cards.rs, lumen_cards.wgsl).
//
//   light_cards : lights the card pages scheduled this frame (one workgroup per 8x8 page): each
//                 texel's captured surface gets direct sunlight (virtual shadow map where the
//                 texel's shadow page is resident, distance-field soft shadows otherwise), local
//                 lights with global distance-field shadows, sky light with distance-field occlusion,
//                 one bounce from the radiance cache (which is fed by the surface caches, so
//                 bounces accumulate) and emission, reflected with the texel's captured albedo
//                 (lumen_card_capture.wgsl; the material's average until it is captured).
//   debug_cards : the card lighting each pixel's surface would read (DebugView::SurfaceCards),
//                 purple-tinted voxel lighting where no card matches.

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "sh.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "lumen_cards.wgsl"
#include "lumen_material.wgsl"
#include "lights.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lumen: Lumen;
@group(0) @binding(2) var global_sdf: texture_3d<f32>;
@group(0) @binding(3) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(4) var lumen_sampler: sampler;
@group(0) @binding(5) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> sky_sh: array<vec4<f32>, 9>;
@group(0) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(8) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(9) var<storage, read> card_data: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read> card_pages: array<u32>;
@group(0) @binding(11) var card_capture: texture_2d<u32>;
@group(0) @binding(12) var card_lighting: texture_2d<f32>;
@group(0) @binding(13) var card_lighting_out: texture_storage_2d<rgba16float, write>;
// Pages to light: (physical page, instance, slot | mip << 8, page x | page y << 16).
@group(0) @binding(14) var<storage, read> card_updates: array<vec4<u32>>;
@group(0) @binding(15) var<uniform> vsm: Vsm;
@group(0) @binding(16) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(17) var vsm_atlas: texture_depth_2d;
@group(0) @binding(18) var shadow_sampler: sampler_comparison;
@group(0) @binding(19) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(20) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(21) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(22) var material_albedo: texture_2d_array<f32>;
@group(0) @binding(23) var material_sampler: sampler;
@group(0) @binding(24) var sdf_instance_ids: texture_3d<u32>;
@group(0) @binding(25) var depth_texture: texture_2d<f32>;
@group(0) @binding(26) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(27) var debug_out: texture_storage_2d<rgba16float, write>;
// Captured materials (cards.rs): rgb = sqrt(albedo), a = emission e / (1 + e) as a multiple of
// the albedo; a card record's texels.z is its origin (0xffffffff: not captured).
@group(0) @binding(28) var card_albedo: texture_2d<f32>;
#include "glass_shadow.wgsl"

// Fraction of outside light a stained-glass window passes (as in lumen_light.wgsl).
const GLASS_TRANSMISSION: f32 = 0.5;
// Sun penumbra cone of distance-field shadows (the voxel cache's softness).
const SUN_SOFTNESS: f32 = 0.05;

fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    return sh_irradiance(sky_sh, n) * frame.sun_illuminance.rgb * lumen.sky_intensity;
}

// The point where the global distance field's version of the surface at `p` lies (along n).
// Large meshes have coarse distance fields whose surface can sit a voxel or more off the true
// one; queries of the global field (occlusion, shadows, radiance probes) that start at the true
// surface would begin inside the field and see themselves as fully occluded. The voxel cache
// is lit at the field's surface, so this also keeps both caches consistent.
fn field_surface(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let level = sdf_finest_level(p);
    if level >= lumen.level_count {
        return p;
    }
    let voxel = lumen.sdf_levels[level].w;
    return p + n * clamp(voxel * 0.25 - sdf_distance(p, level), 0.0, voxel * 3.0);
}

// Ambient occlusion from the global distance field along the normal (as the voxel cache).
fn sdf_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let level = sdf_finest_level(p);
    if level >= lumen.level_count {
        return 1.0;
    }
    let voxel = lumen.sdf_levels[level].w;
    var occlusion = 0.0;
    for (var i = 1u; i <= 4u; i++) {
        let t = voxel * f32(i) * 1.5;
        let d = sdf_distance(p + n * t, min(level + i / 3u, lumen.level_count - 1u));
        occlusion += max(t - d, 0.0) / t * (1.0 / f32(i));
    }
    return saturate(1.0 - occlusion * 0.6);
}

// Sun visibility from the virtual shadow map, if the texel's shadow page (or a coarser one) is
// resident: 4 taps around the biased position (cf. vsm_shadow). -1 if not.
fn vsm_visibility(p: vec3<f32>, n: vec3<f32>, sun: vec3<f32>) -> f32 {
    if vsm.enabled == 0u || !has_flag(frame.flags, FLAG_SHADOWS) {
        return -1.0;
    }
    let level = vsm_level(p);
    let texel = vsm.level_info[level].y;
    let q = p + n * texel * (1.0 + 2.5 * (1.0 - saturate(dot(n, sun)))) + sun * texel * 1.5;
    let dx = vec3<f32>(vsm.light_view[0].x, vsm.light_view[1].x, vsm.light_view[2].x);
    let dy = vec3<f32>(vsm.light_view[0].y, vsm.light_view[1].y, vsm.light_view[2].y);
    var lit = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = (vec2<f32>(f32(i & 1u), f32(i >> 1u)) - 0.5) * 1.2 * texel;
        let l = vsm_lookup(q + dx * o.x - dy * o.y, level);
        if !l.found {
            return -1.0;
        }
        lit += textureSampleCompareLevel(vsm_atlas, shadow_sampler, l.uv, l.depth);
    }
    return lit * 0.25;
}

// Sun visibility at `p` (`field`: the global field's surface there).
fn sun_visibility(p: vec3<f32>, field: vec3<f32>, n: vec3<f32>, sun: vec3<f32>) -> f32 {
    let vsm_lit = vsm_visibility(p, n, sun);
    if vsm_lit >= 0.0 {
        return vsm_lit;
    }
    // Mesh SDFs for the first metres (contact shadows), the global field beyond.
    let origin = p + n * 0.05;
    if !mesh_sdf_covers(origin) {
        return sdf_soft_shadow(field, sun, lumen.max_trace_distance, SUN_SOFTNESS);
    }
    let near = lumen.mesh_sdf_trace.x;
    let visibility = mesh_sdf_soft_shadow(origin, origin, sun, 0.0, near, SUN_SOFTNESS, true);
    if visibility <= 0.01 {
        return 0.0;
    }
    return min(visibility, sdf_soft_shadow_span(origin, sun, near, lumen.max_trace_distance, SUN_SOFTNESS));
}

// A card texel's reflectance and emitted radiance.
struct TexelMaterial {
    albedo: vec3<f32>,
    emitted: vec3<f32>,
}

// The material's average (base colour x the texture's 1x1 mip, its emission).
fn average_texel_material(material: u32) -> TexelMaterial {
    let raw = material_albedo_average(material);
    var t: TexelMaterial;
    t.albedo = abs(raw);
    let m = materials[material];
    t.emitted = m.emissive.rgb;
    if raw.x < 0.0 {
        t.emitted += t.albedo * m.emissive.w;
    }
    return t;
}

// Radiance leaving a card texel's surface at world position `p` with normal `n`.
fn card_texel_radiance(p: vec3<f32>, n: vec3<f32>, material: u32, surface: TexelMaterial) -> vec3<f32> {
    let albedo = surface.albedo;
    let m = materials[material];
    if m.kind == MATERIAL_GLASS {
        // Windows: the daylight outside, tinted by the glass (as the voxel cache).
        return albedo * sh_irradiance(sky_sh, -n) * frame.sun_illuminance.rgb / PI * GLASS_TRANSMISSION;
    }
    let sun = normalize(frame.sun_direction.xyz);
    let field = field_surface(p, n);
    var direct = vec3<f32>(0.0);
    let n_dot_l = dot(n, sun);
    if n_dot_l > 0.0 {
        let transmittance = sample_transmittance(transmittance_lut, lumen_sampler, atmosphere_position(p), sun);
        // Through glass, coloured by the panes: the coloured patches bounce coloured light.
        let glass = glass_transmission(p, lumen_sampler);
        direct = frame.sun_illuminance.rgb * transmittance * glass * n_dot_l * sun_visibility(p, field, n, sun);
    }
    for (var i = 0u; i < frame.light_count; i++) {
        let light = lights[i];
        let to_light = light.position_range.xyz - p;
        if dot(to_light, to_light) > light.position_range.w * light.position_range.w {
            continue;
        }
        let ls = sample_light(light, p);
        let nl = saturate(dot(n, ls.l));
        if nl <= 0.0 {
            continue;
        }
        // Global-field shadows only (as the voxel cache): mesh-SDF traces for every light in
        // range would cost more than the rest of the texel's lighting together.
        var visibility = 1.0;
        if light.params.z > 0.5 {
            let distance = light_shadow_distance(light.position_range.xyz, ls.distance, light.color.w);
            visibility = sdf_soft_shadow(field, ls.l, distance, 0.1);
        }
        direct += ls.radiance * nl * visibility;
    }
    let sky = sky_irradiance(n) * sdf_occlusion(field, n);
    let bounce = radiance_cache_irradiance(field + n * 0.1, n, sky);
    return albedo / PI * (direct + mix(sky, bounce, 0.85)) + surface.emitted;
}

@compute @workgroup_size(8, 8)
fn light_cards(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let update = card_updates[wid.x];
    let out_texel = vec2<u32>(update.x % lumen.cards.z, update.x / lumen.cards.z) * CARD_PAGE_TEXELS + lid.xy;
    let inst = card_instance_load(update.y);
    let slot = update.z & 0xffu;
    let mip = update.z >> 8u;
    let direction = slot / CARD_LAYERS;
    let card = card_load(inst.info.x + slot);
    let resolution = card_unpack(card.texels.x);
    let texel = card_unpack(update.w) * CARD_PAGE_TEXELS + lid.xy;
    if any(texel >= card_mip_resolution(resolution, mip)) {
        textureStore(card_lighting_out, out_texel, vec4<f32>(0.0));
        return;
    }
    // A coarse texel covers 2^mip x 2^mip captured texels: light the first of up to four
    // inside it that sees a surface, with the average captured material of those that do.
    let size = 1u << mip;
    var captured = 0xffffffffu;
    var source = vec2<u32>(0u);
    let has_albedo = card.texels.z != 0xffffffffu;
    var surface = TexelMaterial(vec3<f32>(0.0), vec3<f32>(0.0));
    var surfaces = 0.0;
    for (var k = 0u; k < select(4u, 1u, mip == 0u); k++) {
        let offset = (vec2<u32>(k & 1u, k >> 1u) * 2u + 1u) * size / 4u;
        let candidate = min(texel * size + offset, resolution - 1u);
        let t = textureLoad(card_capture, card_unpack(card.texels.y) + candidate, 0).r;
        if (t & 0xffffu) != 0xffffu {
            if (captured & 0xffffu) == 0xffffu {
                captured = t;
                source = candidate;
            }
            if has_albedo {
                let a = textureLoad(card_albedo, card_unpack(card.texels.z) + candidate, 0);
                let reflectance = a.rgb * a.rgb;
                surface.albedo += reflectance;
                surface.emitted += reflectance * (a.a / max(1.0 - a.a, 1.0 / 255.0));
                surfaces += 1.0;
            }
        }
    }
    if (captured & 0xffffu) == 0xffffu {
        textureStore(card_lighting_out, out_texel, vec4<f32>(0.0));
        return;
    }
    if has_albedo {
        surface.albedo = clamp(surface.albedo / surfaces, vec3<f32>(0.02), vec3<f32>(0.95));
        surface.emitted /= surfaces;
    } else {
        surface = average_texel_material(inst.info.y);
    }
    let q = card_texel_position(card, direction, vec2<f32>(source) + 0.5, captured);
    let p = card_point(q, inst.to_world);
    let n = card_normal(card_texel_normal(captured >> 16u), inst.to_local);
    let radiance = card_texel_radiance(p, n, inst.info.y, surface);
    // Alpha: the lit surface's depth, which card_sample checks its taps against.
    let lit = vec4<f32>(radiance, 1.0 + f32(captured & 0xffffu) / CARD_DEPTH_STEPS);
    textureStore(card_lighting_out, out_texel, select(lit, vec4<f32>(0.0), any(radiance != radiance)));
}

@compute @workgroup_size(8, 8)
fn debug_cards(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let depth = textureLoad(depth_texture, gid.xy, 0).r;
    if depth <= 0.0 {
        textureStore(debug_out, gid.xy, vec4<f32>(0.05, 0.06, 0.08, 1.0));
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) * frame.viewport.zw;
    let p = world_from_depth(uv, depth, frame.inv_view_proj);
    let n = normalize(textureLoad(gbuffer_normal, gid.xy, 0).xyz);
    let level = sdf_finest_level(p);
    if level >= lumen.level_count {
        textureStore(debug_out, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let voxel = lumen.sdf_levels[level].w;
    let fallback = voxel_radiance(p + n * voxel, level) * vec3<f32>(0.8, 0.2, 0.8);
    let color = card_field_radiance(fallback, p, n, level, 0.02);
    textureStore(debug_out, gid.xy, vec4<f32>(color, 1.0));
}
