// Lumen world cache update (world_cache.rs; UE 5.9's Lumen World Cache, traced against Helios's
// distance fields instead of hardware ray tracing). See lumen_world_cache_common.wgsl for the
// cells.
//
//   trace_paths : one path per 8x8 tile of the screen, from a random pixel of it: the pixel's
//                 surface is the first vertex (its G-buffer albedo, normal and emission are exact,
//                 so the cell it lands in learns the material of whatever is on screen: textured,
//                 skinned, displaced, wind-blown), then the vertex where a cosine ray hits the
//                 mesh SDFs (near field) or the global distance field, and a second ray whose
//                 hit is shaded by the caches (the world cache itself, so light keeps bouncing
//                 over frames; the mesh cards in front of the cache unless it replaces them; the
//                 voxel lighting where neither has the surface). Every vertex gets direct light
//                 (sun through the virtual shadow map where resident, distance-field shadows
//                 otherwise, coloured by glass; one local light chosen by its contribution, with
//                 distance-field shadows); the irradiance each
//                 vertex receives along the path is added to its cell (fixed point, atomics),
//                 with the vertex's albedo: exact on screen, elsewhere the albedo the cell
//                 learnt from the screen, else the mesh cards' captured texel, else the global
//                 field's mean of the instance's material.
//   resolve     : per hash slot: this frame's samples blended into the cell (a running mean of
//                 up to WC_MAX_FRAMES frames), cells no path visited for WC_EVICT_FRAMES frames
//                 evicted, the result written to the texture the Lumen traces read.
//   debug       : the world cache's radiance at every pixel (DebugView::WorldCache; purple-tinted
//                 voxel lighting where no cell holds the surface).
//   reference   : brute-force path tracing for comparisons (HELIOS_WORLD_CACHE_REFERENCE, native
//                 only): per half-resolution pixel and frame one path of up to 8 bounces through
//                 the same distance fields, materials and lights, without any cache, accumulated
//                 while the camera rests, written in place of the final gather.

#include "common.wgsl"
#include "atmosphere.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "lumen_cards.wgsl"
#include "lumen_trace.wgsl"
#include "lights.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> lumen: Lumen;
@group(0) @binding(2) var global_sdf: texture_3d<f32>;
@group(0) @binding(3) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(4) var lumen_sampler: sampler;
@group(0) @binding(5) var<storage, read> radiance_probes: array<vec4<f32>>;
@group(0) @binding(6) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(9) var<storage, read> exposure: array<f32, 4>;
@group(0) @binding(15) var<uniform> vsm: Vsm;
@group(0) @binding(16) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(17) var vsm_atlas: texture_depth_2d;
@group(0) @binding(18) var shadow_sampler: sampler_comparison;
@group(0) @binding(19) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(20) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(25) var depth_texture: texture_2d<f32>;
@group(0) @binding(26) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(27) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(28) var gbuffer_motion: texture_2d<f32>;
// Captured materials of the mesh cards (cards.rs): rgb = sqrt(albedo), a = emission e / (1 + e).
@group(0) @binding(29) var card_albedo: texture_2d<f32>;
@group(0) @binding(70) var<storage, read> card_data: array<vec4<u32>>;
@group(0) @binding(71) var<storage, read> card_pages: array<u32>;
@group(0) @binding(72) var card_capture: texture_2d<u32>;
@group(0) @binding(73) var card_lighting: texture_2d<f32>;
@group(0) @binding(74) var sdf_instance_ids: texture_3d<u32>;
// Per hash slot, WC_RECORD words (see below).
@group(0) @binding(151) var<storage, read_write> records: array<atomic<u32>>;
@group(0) @binding(152) var cells_out: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(153) var debug_out: texture_storage_2d<rgba16float, write>;
// Statistics (HELIOS_WORLD_CACHE_STATS): live cells, cells sampled this frame, debug-view pixels
// whose surface has a cell, debug-view pixels.
@group(0) @binding(154) var<storage, read_write> stats: array<atomic<u32>, 4>;
// Reference: per half-resolution pixel (radiance sum, paths), (sky sum, -, -, -).
@group(0) @binding(155) var<storage, read_write> reference_sums: array<vec4<f32>>;
@group(0) @binding(156) var gi_out: texture_storage_2d<rgba16float, write>;
#include "glass_shadow.wgsl"

// A slot's record: key checksum; this frame's irradiance sum (rgb) and sample count; albedo sum
// (rgb, weighted) and weight; emission sum (weighted like the albedo); screen samples; the last
// frame a path visited the cell; the resolved cell (as world_cache_cells' y, z, w); spare.
const WC_RECORD: u32 = 16u;
const R_KEY: u32 = 0u;
const R_IRRADIANCE: u32 = 1u;
const R_COUNT: u32 = 4u;
const R_ALBEDO: u32 = 5u;
const R_WEIGHT: u32 = 8u;
const R_EMISSION: u32 = 9u;
const R_SCREEN: u32 = 10u;
const R_TOUCHED: u32 = 11u;
const R_CELL: u32 = 12u;
// Cells no path visited for this many frames are evicted.
const WC_EVICT_FRAMES: u32 = 64u;
// Fixed-point steps per unit of exposed irradiance (exposure maps mid grey to 0.18), and the
// largest sample (about 4096 samples of it fit a word).
const WC_FIXED: f32 = 256.0;
const WC_MAX_EXPOSED: f32 = 4096.0;
// Weights of an albedo seen on screen and of an estimate at a ray hit.
const WC_SCREEN_WEIGHT: u32 = 16u;
const WC_ESTIMATE_WEIGHT: u32 = 1u;
// Largest exposed radiance a path carries between two vertices (as the screen probes' rays):
// tiny bright emitters hit by a few paths would otherwise speckle the cells.
const WC_MAX_RAY_INTENSITY: f32 = 4.0;
// Sun penumbra cone of distance-field shadows (the surface caches').
const WC_SUN_SOFTNESS: f32 = 0.05;
// Vertices a path shades (the screen pixel and one hit); the ray from the last one reads the
// caches. A third shaded vertex cost 0.2 ms more (cathedral, 1600x900) and adds what the cells
// already carry over from frame to frame.
const WC_VERTICES: u32 = 2u;

fn wc_random(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state >> 8u) / 16777216.0;
}

fn wc_clamp_ray(color: vec3<f32>) -> vec3<f32> {
    let limit = WC_MAX_RAY_INTENSITY / max(exposure[0], 1e-8);
    let l = luminance(color);
    let c = color * min(1.0, limit / max(l, 1e-8));
    return select(c, vec3<f32>(0.0), any(c != c));
}

// ---- Cells -------------------------------------------------------------------------------------

// Slot of `key`, inserted if new (WC_NONE if its probe sequence is full); marks it visited.
fn wc_insert(key: WorldCacheKey) -> u32 {
    for (var i = 0u; i < WC_PROBES; i++) {
        let slot = (key.slot + i) & (WC_CAPACITY - 1u);
        let word = slot * WC_RECORD;
        for (var attempt = 0u; attempt < 4u; attempt++) {
            let r = atomicCompareExchangeWeak(&records[word + R_KEY], 0u, key.check);
            if r.exchanged || r.old_value == key.check {
                atomicStore(&records[word + R_TOUCHED], lumen.frame);
                return slot;
            }
            if r.old_value != 0u {
                break;
            }
        }
    }
    return WC_NONE;
}

fn wc_surface_key(p: vec3<f32>, n: vec3<f32>) -> WorldCacheKey {
    return wc_key(p, wc_bucket(n), wc_level_of(length(p - frame.camera_position.xyz)));
}

fn wc_add_irradiance(slot: u32, irradiance: vec3<f32>, rng: ptr<function, u32>) {
    let word = slot * WC_RECORD;
    let scaled = min(irradiance * max(exposure[0], 1e-8), vec3<f32>(WC_MAX_EXPOSED)) * WC_FIXED;
    let dither = wc_random(rng);
    atomicAdd(&records[word + R_IRRADIANCE], u32(scaled.r + dither));
    atomicAdd(&records[word + R_IRRADIANCE + 1u], u32(scaled.g + dither));
    atomicAdd(&records[word + R_IRRADIANCE + 2u], u32(scaled.b + dither));
    atomicAdd(&records[word + R_COUNT], 1u);
}

fn wc_add_albedo(slot: u32, albedo: vec3<f32>, emission: f32, weight: u32) {
    let word = slot * WC_RECORD;
    let a = vec3<u32>(saturate(albedo) * 255.0 * f32(weight) + 0.5);
    atomicAdd(&records[word + R_ALBEDO], a.r);
    atomicAdd(&records[word + R_ALBEDO + 1u], a.g);
    atomicAdd(&records[word + R_ALBEDO + 2u], a.b);
    atomicAdd(&records[word + R_WEIGHT], weight);
    atomicAdd(&records[word + R_EMISSION], u32(clamp(emission, 0.0, 64.0) * 256.0 * f32(weight) + 0.5));
    if weight == WC_SCREEN_WEIGHT {
        atomicAdd(&records[word + R_SCREEN], 1u);
    }
}

// ---- Direct light at a path vertex (as the mesh cards' texels are lit, lumen_card_light.wgsl) ----

// The point where the global distance field's version of the surface at `p` lies (along n).
fn wc_field_surface(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let level = sdf_finest_level(p);
    if level >= lumen.level_count {
        return p;
    }
    let voxel = lumen.sdf_levels[level].w;
    return p + n * clamp(voxel * 0.25 - sdf_distance(p, level), 0.0, voxel * 3.0);
}

// Sun visibility from the virtual shadow map where `p`'s page (or a coarser one) is resident,
// -1 if not.
fn wc_vsm_visibility(p: vec3<f32>, n: vec3<f32>, sun: vec3<f32>) -> f32 {
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

fn wc_sun_visibility(p: vec3<f32>, field: vec3<f32>, n: vec3<f32>, sun: vec3<f32>) -> f32 {
    let vsm_lit = wc_vsm_visibility(p, n, sun);
    if vsm_lit >= 0.0 {
        return vsm_lit;
    }
    let origin = p + n * 0.05;
    if !mesh_sdf_covers(origin) {
        return sdf_soft_shadow(field, sun, lumen.max_trace_distance, WC_SUN_SOFTNESS);
    }
    let near = lumen.mesh_sdf_trace.x;
    let visibility = mesh_sdf_soft_shadow(origin, origin, sun, 0.0, near, WC_SUN_SOFTNESS, true);
    if visibility <= 0.01 {
        return 0.0;
    }
    return min(visibility, sdf_soft_shadow_span(origin, sun, near, lumen.max_trace_distance, WC_SUN_SOFTNESS));
}

// Irradiance at `p` (normal `n`) from the sun and the local lights. One local light is shadowed
// per call, chosen among those in reach in proportion to its unshadowed irradiance (weighted
// reservoir sampling, an unbiased estimate of their sum): a shadow ray per light in reach - a
// dozen candles around a pew - cost more than the rest of the path; the cells average it out.
fn wc_direct(p: vec3<f32>, n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let sun = normalize(frame.sun_direction.xyz);
    let field = wc_field_surface(p, n);
    var direct = vec3<f32>(0.0);
    let n_dot_l = dot(n, sun);
    if n_dot_l > 0.0 {
        let transmittance = sample_transmittance(transmittance_lut, lumen_sampler, atmosphere_position(p), sun);
        let glass = glass_transmission(p, lumen_sampler);
        direct = frame.sun_illuminance.rgb * transmittance * glass * n_dot_l * wc_sun_visibility(p, field, n, sun);
    }
    var total = 0.0;
    var chosen = WC_NONE;
    var chosen_weight = 0.0;
    var chosen_irradiance = vec3<f32>(0.0);
    var chosen_sample: LightSample;
    for (var i = 0u; i < frame.light_count; i++) {
        let light = lights[i];
        let to_light = light.position_range.xyz - p;
        if dot(to_light, to_light) > light.position_range.w * light.position_range.w {
            continue;
        }
        let ls = sample_light(light, p);
        let nl = saturate(dot(n, ls.l));
        let irradiance = ls.radiance * nl;
        let weight = luminance(irradiance);
        if weight <= 0.0 {
            continue;
        }
        total += weight;
        if wc_random(rng) * total < weight {
            chosen = i;
            chosen_weight = weight;
            chosen_irradiance = irradiance;
            chosen_sample = ls;
        }
    }
    if chosen != WC_NONE {
        let light = lights[chosen];
        var visibility = 1.0;
        if light.params.z > 0.5 {
            let distance = light_shadow_distance(light.position_range.xyz, chosen_sample.distance, light.color.w);
            visibility = sdf_soft_shadow(field, chosen_sample.l, distance, 0.1);
        }
        direct += chosen_irradiance * (visibility * total / chosen_weight);
    }
    return direct;
}

// ---- Tracing ---------------------------------------------------------------------------------------

const HIT_SKY: u32 = 0u;
const HIT_NEAR: u32 = 1u;
const HIT_GLOBAL: u32 = 2u;
const HIT_FAR: u32 = 3u;

struct WcHit {
    kind: u32,
    // The hit on the surface (global field: moved onto the field's zero level), its normal
    // facing the ray, and the raw sphere-tracing hit (global field).
    p: vec3<f32>,
    n: vec3<f32>,
    raw: vec3<f32>,
    // Near field: the card instance hit; global field: the level.
    instance: u32,
    level: u32,
}

// Near field (mesh SDFs) from the surface, then the global field: from beyond the near field,
// or where the near field doesn't reach, from just off the field's own version of the surface.
fn wc_trace(p: vec3<f32>, n: vec3<f32>, dir: vec3<f32>) -> WcHit {
    var h: WcHit;
    h.kind = HIT_SKY;
    h.instance = CARD_NONE;
    h.level = 0u;
    if lumen.sdf_enabled == 0u {
        return h;
    }
    let origin = p + n * 0.03;
    var global_origin = origin;
    var t_global = 0.0;
    if mesh_sdf_covers(origin) {
        let near = trace_mesh_sdfs(origin, dir, 0.0, lumen.mesh_sdf_trace.x);
        if near.hit {
            h.kind = HIT_NEAR;
            h.p = origin + dir * near.t;
            h.raw = h.p;
            h.n = near.normal;
            h.instance = near.instance;
            return h;
        }
        t_global = lumen.mesh_sdf_trace.x;
    } else {
        let level = sdf_finest_level(p);
        if level < lumen.level_count {
            let voxel = lumen.sdf_levels[level].w;
            global_origin = wc_field_surface(p, n) + n * voxel * 0.5;
            t_global = voxel;
        }
    }
    let hit = trace_sdf(global_origin, dir, t_global, lumen.max_trace_distance);
    if hit.hit {
        let normal = sdf_normal(hit.position, hit.level);
        h.kind = HIT_GLOBAL;
        h.raw = hit.position;
        h.p = hit.position - normal * sdf_distance(hit.position, hit.level);
        h.n = select(normal, -normal, dot(normal, dir) > 0.0);
        h.level = hit.level;
        return h;
    }
    if sdf_finest_level(global_origin + dir * lumen.max_trace_distance) < lumen.level_count {
        h.kind = HIT_FAR;
        h.raw = global_origin + dir * lumen.max_trace_distance;
    }
    return h;
}

// The card instance a global-field hit lies on (as card_field_radiance chooses it).
fn wc_global_instance(h: WcHit) -> u32 {
    let offset = h.n * lumen.sdf_levels[h.level].w * 0.5;
    let inside = sdf_instance_at(h.raw - offset, h.level);
    if inside != CARD_NONE {
        return inside;
    }
    return sdf_instance_at(h.raw + offset, h.level);
}

fn wc_card_tolerance(h: WcHit) -> f32 {
    return select(0.02, lumen.sdf_levels[h.level].w * 0.5, h.kind == HIT_GLOBAL);
}

// The mesh cards' lighting at a hit (Fallback mode: the cards come before the world cache).
fn wc_hit_cards(h: WcHit) -> CardRadiance {
    if h.kind == HIT_NEAR {
        return card_radiance(h.instance, h.p, h.n, 0.02);
    }
    let offset = h.n * lumen.sdf_levels[h.level].w * 0.5;
    let tolerance = lumen.sdf_levels[h.level].w * 0.5;
    let inside = sdf_instance_at(h.raw - offset, h.level);
    var cards = card_radiance(inside, h.raw, h.n, tolerance);
    if cards.weight < 0.25 {
        let outside = sdf_instance_at(h.raw + offset, h.level);
        if outside != inside {
            let other = card_radiance(outside, h.raw, h.n, tolerance);
            if other.weight > cards.weight {
                cards = other;
            }
        }
    }
    return cards;
}

// Radiance leaving a surface hit according to the voxel lighting (near field: the estimate from it).
fn wc_voxel_radiance(h: WcHit, dir: vec3<f32>) -> vec3<f32> {
    if h.kind == HIT_NEAR {
        return near_field_radiance(h.p, h.n);
    }
    let lighting_voxel = lumen.sdf_levels[h.level].w * SDF_RES / LIGHT_RES;
    return voxel_radiance(h.raw - dir * lighting_voxel * 0.6, h.level);
}

// Radiance leaving a hit according to the surface caches, as the screen probes' rays see it
// (lumen_trace.wgsl): the world cache over the voxel lighting (near field: over its estimate),
// the cards over both unless the world cache replaces them; sky or far field for misses.
fn wc_cached_radiance(h: WcHit, dir: vec3<f32>) -> vec3<f32> {
    if h.kind == HIT_SKY {
        return sky_radiance(dir);
    }
    if h.kind == HIT_FAR {
        // The ray ended inside the clipmap without a hit: far-field estimate (as lumen_trace.wgsl).
        let sky = sky_radiance(dir);
        return mix(sky, radiance_cache_irradiance(h.raw, -dir, sky) / PI, 0.5);
    }
    let base = world_cache_over(wc_voxel_radiance(h, dir), h.p, h.n);
    if lumen.world_cache == WC_REPLACE {
        return base;
    }
    let cards = wc_hit_cards(h);
    return mix(base, cards.radiance, saturate(cards.weight * 2.0));
}

// ---- Material at a hit ---------------------------------------------------------------------------

struct HitMaterial {
    albedo: vec3<f32>,
    emission: f32,
    weight: f32,
}

// The captured albedo of the card texel that sees mesh-space hit `p` (as card_sample matches
// the hit against the captured depth and normal), weighted by how squarely the cards face it.
fn wc_card_albedo(instance: u32, p: vec3<f32>, n: vec3<f32>, tolerance: f32) -> HitMaterial {
    var r = HitMaterial(vec3<f32>(0.0), 0.0, 0.0);
    if instance >= lumen.cards.x {
        return r;
    }
    let inst = card_instance_load(instance);
    let q = card_point(p, inst.to_local);
    let nl = card_normal(n, inst.to_world);
    let local_tolerance = tolerance / bitcast<f32>(inst.info.w);
    for (var c = 0u; c < 3u; c++) {
        let facing = nl[c];
        if abs(facing) < 0.3 {
            continue;
        }
        let direction = c * 2u + select(0u, 1u, facing < 0.0);
        let sign = select(1.0, -1.0, facing < 0.0);
        for (var layer = 0u; layer < card_layer_count(inst.info, direction); layer++) {
            let card = card_load(inst.info.x + direction * CARD_LAYERS + layer);
            if card.texels.z == CARD_NONE {
                break;
            }
            let texel = (vec2<f32>(q[(c + 1u) % 3u], q[(c + 2u) % 3u]) - card.frame.xy) / card.frame.z;
            let resolution = card_unpack(card.texels.x);
            if any(texel < vec2<f32>(0.0)) || any(texel >= vec2<f32>(resolution)) {
                continue;
            }
            let captured = textureLoad(card_capture, card_unpack(card.texels.y) + vec2<u32>(texel), 0).r;
            if (captured & 0xffffu) == 0xffffu {
                continue;
            }
            let depth = card.depth.x + f32(captured & 0xffffu) * card.depth.y;
            let gap = abs(sign * (card.frame.w - q[c]) - depth);
            let range = local_tolerance + card.depth.z + 2.0 * card.frame.z;
            let agreement = saturate(dot(card_texel_normal(captured >> 16u), nl) * 1.5 - 0.25);
            let confidence = saturate(1.0 - gap / range) * agreement;
            if confidence <= 0.0 {
                continue;
            }
            let a = textureLoad(card_albedo, card_unpack(card.texels.z) + vec2<u32>(texel), 0);
            let w = facing * facing * confidence;
            r.albedo += a.rgb * a.rgb * w;
            r.emission += a.a / max(1.0 - a.a, 1.0 / 255.0) * w;
            r.weight += w;
            break;
        }
    }
    if r.weight > 0.0 {
        r.albedo /= r.weight;
        r.emission /= r.weight;
    }
    return r;
}

// ---- Paths -----------------------------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn trace_paths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    let tiles = (size + 7u) / 8u;
    if any(gid.xy >= tiles) || lumen.world_cache == WC_OFF {
        return;
    }
    var rng = pcg_hash((gid.y * tiles.x + gid.x) ^ pcg_hash(lumen.frame + 0x51ed270bu));
    let pick = pcg_hash(rng);
    let pixel = min(gid.xy * 8u + vec2<u32>(pick & 7u, (pick >> 3u) & 7u), size - 1u);
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    var p = world_from_depth(uv, depth, frame.inv_view_proj);
    var n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let base_colour = textureLoad(gbuffer_albedo, pixel, 0);
    var albedo = clamp(base_colour.rgb * base_colour.rgb, vec3<f32>(0.02), vec3<f32>(0.95));
    var emission = textureLoad(gbuffer_motion, pixel, 0).z;
    var slot = wc_insert(wc_surface_key(p, n));
    if slot != WC_NONE {
        wc_add_albedo(slot, albedo, emission, WC_SCREEN_WEIGHT);
    }

    var slots: array<u32, 3>;
    var directs: array<vec3<f32>, 3>;
    var albedos: array<vec3<f32>, 3>;
    var emissions: array<f32, 3>;
    var vertices = 0u;
    // Radiance arriving at the last vertex along its ray.
    var tail = vec3<f32>(0.0);
    for (var k = 0u; k < WC_VERTICES; k++) {
        slots[k] = slot;
        albedos[k] = albedo;
        emissions[k] = emission;
        directs[k] = wc_direct(p, n, &rng);
        vertices = k + 1u;
        let dir = basis(n) * cosine_sample_hemisphere(vec2<f32>(wc_random(&rng), wc_random(&rng)));
        let h = wc_trace(p, n, dir);
        if k + 1u == WC_VERTICES || h.kind == HIT_SKY || h.kind == HIT_FAR {
            tail = wc_cached_radiance(h, dir);
            break;
        }
        // Behind the cards (Fallback), a surface the cards hold ends the path: its light is what
        // the screen probes' rays will read there anyway.
        if lumen.world_cache == WC_FALLBACK {
            let cards = wc_hit_cards(h);
            if cards.weight >= 0.5 {
                tail = cards.radiance;
                break;
            }
        }
        // The hit's material: the albedo its cell learnt from the screen, else the cards'
        // captured texel, else the global field's mean of the instance's material.
        let level = select(sdf_finest_level(h.p), h.level, h.kind == HIT_GLOBAL);
        if level >= lumen.level_count {
            tail = wc_cached_radiance(h, dir);
            break;
        }
        let raw_albedo = sdf_albedo(h.p, level);
        if any(raw_albedo < vec3<f32>(-0.01)) {
            // Glass and emissive voxels: their light is in the voxel lighting (translucent
            // surfaces are not cached, as in UE).
            tail = wc_cached_radiance(h, dir);
            break;
        }
        var material = HitMaterial(abs(raw_albedo), 0.0, 0.0);
        var learnt = false;
        let known = wc_find_surface(h.p, h.n);
        if known != WC_NONE {
            let cell = wc_decode(wc_load(known));
            if cell.screen {
                material = HitMaterial(cell.albedo, cell.emission, 1.0);
                learnt = true;
            }
        }
        if !learnt {
            let instance = select(wc_global_instance(h), h.instance, h.kind == HIT_NEAR);
            let card = wc_card_albedo(instance, h.p, h.n, wc_card_tolerance(h));
            if card.weight > 0.25 {
                material = card;
            }
        }
        p = h.p;
        n = h.n;
        albedo = clamp(material.albedo, vec3<f32>(0.02), vec3<f32>(0.95));
        emission = material.emission;
        slot = wc_insert(wc_surface_key(p, n));
        if slot != WC_NONE && !learnt {
            wc_add_albedo(slot, albedo, emission, WC_ESTIMATE_WEIGHT);
        }
    }
    // Back along the path: each vertex receives its direct light and the radiance leaving the
    // next one towards it (cosine-sampled: irradiance = pi x radiance).
    var incoming = wc_clamp_ray(tail);
    for (var j = 0u; j < vertices; j++) {
        let k = vertices - 1u - j;
        let irradiance = directs[k] + PI * incoming;
        if slots[k] != WC_NONE {
            wc_add_irradiance(slots[k], irradiance, &rng);
        }
        incoming = wc_clamp_ray(albedos[k] * (irradiance / PI + emissions[k]));
    }
}

// ---- Resolve -----------------------------------------------------------------------------------------

fn wc_pack(cell: WorldCacheCell, check: u32) -> vec4<u32> {
    let albedo = pack4x8unorm(vec4<f32>(sqrt(saturate(cell.albedo)), 0.0)) & 0xffffffu;
    let w = albedo | (min(cell.frames, 31u) << 24u) | (select(0u, 1u, cell.screen) << 31u);
    return vec4<u32>(check, pack2x16float(cell.irradiance.rg), pack2x16float(vec2<f32>(cell.irradiance.b, cell.emission)), w);
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if slot >= WC_CAPACITY {
        return;
    }
    let word = slot * WC_RECORD;
    let check = atomicLoad(&records[word + R_KEY]);
    if check == 0u {
        return;
    }
    if lumen.frame - atomicLoad(&records[word + R_TOUCHED]) > WC_EVICT_FRAMES {
        for (var i = 0u; i < WC_RECORD; i++) {
            atomicStore(&records[word + i], 0u);
        }
        textureStore(cells_out, wc_texel(slot), vec4<u32>(0u));
        return;
    }
    let packed = vec4<u32>(check, atomicLoad(&records[word + R_CELL]), atomicLoad(&records[word + R_CELL + 1u]), atomicLoad(&records[word + R_CELL + 2u]));
    var cell = wc_decode(packed);
    let has_albedo = (packed.w & 0xffffffu) != 0u || cell.screen;

    let count = atomicExchange(&records[word + R_COUNT], 0u);
    atomicAdd(&stats[0], 1u);
    if count > 0u {
        atomicAdd(&stats[1], 1u);
    }
    let sum = vec3<f32>(
        f32(atomicExchange(&records[word + R_IRRADIANCE], 0u)),
        f32(atomicExchange(&records[word + R_IRRADIANCE + 1u], 0u)),
        f32(atomicExchange(&records[word + R_IRRADIANCE + 2u], 0u)),
    );
    if count > 0u {
        let mean = sum / (f32(count) * WC_FIXED * max(exposure[0], 1e-8));
        var frames = min(cell.frames + 1u, WC_MAX_FRAMES);
        // A large change seen by many samples at once (a light switched, the sun moved behind
        // a wall) shortens the history, so the cell follows within a few frames.
        let before = luminance(cell.irradiance);
        let after = luminance(mean);
        if count >= 8u && abs(after - before) > 0.6 * max(before, after) {
            frames = min(frames, 4u);
        }
        if all(mean == mean) {
            cell.irradiance = min(mix(cell.irradiance, mean, 1.0 / f32(frames)), vec3<f32>(60000.0));
            cell.frames = frames;
        }
    }
    let weight = atomicExchange(&records[word + R_WEIGHT], 0u);
    let albedo_sum = vec3<f32>(
        f32(atomicExchange(&records[word + R_ALBEDO], 0u)),
        f32(atomicExchange(&records[word + R_ALBEDO + 1u], 0u)),
        f32(atomicExchange(&records[word + R_ALBEDO + 2u], 0u)),
    );
    let emission_sum = f32(atomicExchange(&records[word + R_EMISSION], 0u));
    let screen = atomicExchange(&records[word + R_SCREEN], 0u);
    if weight > 0u {
        let albedo = albedo_sum / (f32(weight) * 255.0);
        let emission = emission_sum / (f32(weight) * 256.0);
        if screen > 0u && !cell.screen {
            // First seen on screen: the exact albedo replaces any estimate at once.
            cell.albedo = albedo;
            cell.emission = emission;
            cell.screen = true;
        } else if screen > 0u || !cell.screen {
            let rate = select(1.0, 0.25, has_albedo);
            cell.albedo = mix(cell.albedo, albedo, rate);
            cell.emission = mix(cell.emission, emission, rate);
        }
    }
    let resolved = wc_pack(cell, check);
    atomicStore(&records[word + R_CELL], resolved.y);
    atomicStore(&records[word + R_CELL + 1u], resolved.z);
    atomicStore(&records[word + R_CELL + 2u], resolved.w);
    textureStore(cells_out, wc_texel(slot), resolved);
}

// ---- Reference ---------------------------------------------------------------------------------------

const WC_REFERENCE_BOUNCES: u32 = 8u;

@compute @workgroup_size(8, 8)
fn reference(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_size = viewport_extent(frame.viewport, 2u);
    if any(gid.xy >= half_size) {
        return;
    }
    let index = gid.y * half_size.x + gid.x;
    let pixel = min(gid.xy * 2u, vec2<u32>(frame.viewport.xy) - 1u);
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        textureStore(gi_out, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    var p = world_from_depth(uv, depth, frame.inv_view_proj);
    var n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    var rng = pcg_hash(index ^ pcg_hash(lumen.frame * 0x2c1b3c6du + 0x7f4a7c15u));
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var sky = 0.0;
    for (var b = 0u; b < WC_REFERENCE_BOUNCES; b++) {
        let dir = basis(n) * cosine_sample_hemisphere(vec2<f32>(wc_random(&rng), wc_random(&rng)));
        let h = wc_trace(p, n, dir);
        if h.kind == HIT_SKY || h.kind == HIT_FAR {
            radiance += throughput * wc_cached_radiance(h, dir);
            if b == 0u {
                sky = select(0.5, 1.0, h.kind == HIT_SKY);
            }
            break;
        }
        let level = select(sdf_finest_level(h.p), h.level, h.kind == HIT_GLOBAL);
        if level >= lumen.level_count {
            break;
        }
        let raw_albedo = sdf_albedo(h.p, level);
        if any(raw_albedo < vec3<f32>(-0.01)) {
            // Glass: the daylight through it, as every Lumen cache models it.
            radiance += throughput * wc_voxel_radiance(h, dir);
            break;
        }
        var material = HitMaterial(abs(raw_albedo), 0.0, 0.0);
        let instance = select(wc_global_instance(h), h.instance, h.kind == HIT_NEAR);
        let card = wc_card_albedo(instance, h.p, h.n, wc_card_tolerance(h));
        if card.weight > 0.25 {
            material = card;
        }
        let albedo = clamp(material.albedo, vec3<f32>(0.02), vec3<f32>(0.95));
        p = h.p;
        n = h.n;
        radiance += throughput * albedo * (wc_direct(p, n, &rng) / PI + material.emission);
        throughput *= albedo;
        if b >= 2u {
            let survive = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
            if wc_random(&rng) > survive {
                break;
            }
            throughput /= survive;
        }
    }
    radiance = min(radiance, vec3<f32>(64.0 / max(exposure[0], 1e-8)));
    if any(radiance != radiance) {
        radiance = vec3<f32>(0.0);
    }
    let sums = reference_sums[index * 2u] + vec4<f32>(radiance, 1.0);
    let sky_sum = reference_sums[index * 2u + 1u].x + sky;
    reference_sums[index * 2u] = sums;
    reference_sums[index * 2u + 1u] = vec4<f32>(sky_sum, 0.0, 0.0, 0.0);
    textureStore(gi_out, gid.xy, vec4<f32>(sums.rgb / sums.w, sky_sum / sums.w));
}

// ---- Debug view ----------------------------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn debug(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    var fallback = vec3<f32>(0.0);
    let level = sdf_finest_level(p);
    if level < lumen.level_count {
        fallback = voxel_radiance(p + n * lumen.sdf_levels[level].w, level) * vec3<f32>(0.8, 0.2, 0.8);
    }
    let c = world_cache_radiance(p, n);
    atomicAdd(&stats[3], 1u);
    if c.a > 0.0 {
        atomicAdd(&stats[2], 1u);
    }
    textureStore(debug_out, gid.xy, vec4<f32>(mix(fallback, c.rgb, c.a), 1.0));
}
