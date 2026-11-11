// MegaLights light sampling and shadow rays (see megalights.rs and megalights_common.wgsl).
//
// generate: per 2x2 block, picks ML_SAMPLES lights at the block's representative pixel by
//   resampling the froxel's light list: up to ML_CANDIDATES candidates, each weighted by its
//   unshadowed contribution (the shared BRDF times the light's radiance, plus a small floor so a
//   light reaching the block is never impossible to pick); lights whose rays were all blocked in
//   this tile last frame keep a tenth of it (visibility guiding). Longer lists are thinned to a
//   stratified random subset, each light with the same chance of being a candidate, whose weights
//   are scaled by the inverse of that chance: the result stays unbiased and the cost per pixel
//   stays constant whatever the number of lights. The four picks are made by systematic
//   resampling over the candidates' running sum, so they are spread over the lights (each pick
//   alone is confined to a quarter of the sum; their average is unbiased).
//   Where several picks are the same light and last frame's rays to it in this tile all agreed
//   (all reached it or all were blocked: outside its penumbrae), they share one ray (UE 5.8).
//   Each workgroup appends the rays its blocks need to a compact list (ray_args counts them and
//   sizes the trace's indirect dispatch), so rays saved cost nothing.
// trace: one thread per ray of the list. The ray ends on a random point of the light's disk (soft shadows
//   from the source radius): a short screen trace against the depth buffer for contact shadows,
//   and over the whole ray either the light's virtual shadow map (shadow-map ray marching where it
//   has one) or the distance fields - mesh SDFs within the near-field distance of either end, the
//   global field between - as the classic path traces the lights beyond its shadow-map budget.

#include "common.wgsl"
#include "brdf.wgsl"
#include "lights.wgsl"
#include "lumen_common.wgsl"
#include "lumen_mesh_sdf.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "vsm_local_common.wgsl"
#include "vsm_local_sample.wgsl"
#include "megalights_surface.wgsl"
#include "megalights_common.wgsl"
#include "megalights_rays.wgsl"
#include "hzb_trace.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_motion: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(6) var<storage, read> light_grid: array<u32>;
@group(0) @binding(7) var<storage, read_write> samples: array<u32>;
// Visibility guiding masks, two halves: this frame's (frame index parity) and last frame's.
@group(0) @binding(9) var<storage, read_write> guide_masks: array<atomic<u32>>;
@group(0) @binding(10) var<uniform> lumen: Lumen;
@group(0) @binding(11) var global_sdf: texture_3d<f32>;
@group(0) @binding(12) var lumen_sampler: sampler;
@group(0) @binding(13) var<storage, read> mesh_sdf_data: array<vec4<u32>>;
@group(0) @binding(14) var mesh_sdf_atlas: texture_3d<f32>;
@group(0) @binding(15) var<uniform> vsm: Vsm;
@group(0) @binding(16) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(17) var vsm_atlas: texture_depth_2d;
@group(0) @binding(18) var<uniform> local_vsm: LocalVsm;
@group(0) @binding(22) var hzb_closest: texture_2d<f32>;
// The rays to trace: [0] their count (written by ray_args), then (block index * 4 + sample).
@group(0) @binding(23) var<storage, read_write> rays: array<u32>;
// generate and ray_args only (the trace's indirect arguments): workgroups x, y, z, ray count.
@group(1) @binding(0) var<storage, read_write> ray_args: array<atomic<u32>, 4>;
// Declared for helpers of the included files that these entry points don't call (filtered
// shadow-map sampling, surface-cache lookups); not part of the bind group layout.
@group(0) @binding(19) var shadow_sampler: sampler_comparison;
@group(0) @binding(20) var voxel_lighting: texture_3d<f32>;
@group(0) @binding(21) var<storage, read> radiance_probes: array<vec4<f32>>;

// Permutations: shared rays outside penumbrae (RenderSettings::mega_lights_shared_rays) and HZB
// screen traces (mega_lights_hzb_traces; off: ML_SCREEN_STEPS linear steps).
override ML_SHARED_RAYS: bool = true;
override ML_HZB_TRACE: bool = true;
const ML_TRACE_GROUP: u32 = 64u;
const ML_MAX_GROUPS_X: u32 = 32768u;
// Iterations of an HZB screen trace.
const ML_HZB_STEPS: u32 = 32u;

// Candidate lights per block (the whole list when it is no longer).
const ML_CANDIDATES: u32 = 32u;
// Weight left to a light whose rays in a tile were all blocked last frame: low enough to spend
// the rays elsewhere, high enough to notice when it comes back into view.
const ML_HIDDEN_WEIGHT: f32 = 0.1;
// Floor of the sampling weight relative to the light's incident radiance: lights behind the
// representative pixel's normal may still light the other pixels of its block.
const ML_WEIGHT_FLOOR: f32 = 0.004;

// Screen trace: steps and maximum length (world units) of the stretch next to the receiver, and
// the assumed thickness of what the depth buffer shows (grown by 10 % per metre of depth).
const ML_SCREEN_STEPS: u32 = 8u;
const ML_SCREEN_DISTANCE: f32 = 1.0;
const CONTACT_THICKNESS: f32 = 0.03;

// ---- Light sampling --------------------------------------------------------------------------------

// First word of this frame's (or last frame's) half of the guide buffer.
fn guide_half(previous: bool) -> u32 {
    let parity = frame.frame_index & 1u;
    return select(parity, 1u - parity, previous) * (arrayLength(&guide_masks) / 2u);
}

// The guide masks of the tile that held `world` last frame, as the word index of its first word
// (INVALID_ID: no history).
fn previous_guide(world: vec3<f32>) -> u32 {
    if !has_flag(frame.flags, FLAG_HISTORY_VALID) {
        return INVALID_ID;
    }
    let clip = frame.prev_unjittered_view_proj * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        return INVALID_ID;
    }
    let uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return INVALID_ID;
    }
    return guide_half(true) + ml_guide_base(vec2<u32>(uv * frame.prev_viewport.xy), frame.prev_viewport);
}

// Whether last frame's rays to `light` in the guide tile at `guide` all agreed: some were traced
// and they all reached it or were all blocked.
fn ml_agreed(guide: u32, light: u32) -> bool {
    if guide == INVALID_ID {
        return false;
    }
    let word = ml_guide_word(light);
    let bit = ml_guide_bit(light);
    let sampled = (atomicLoad(&guide_masks[guide + word]) & bit) != 0u;
    let visible = (atomicLoad(&guide_masks[guide + 8u + word]) & bit) != 0u;
    let shadowed = (atomicLoad(&guide_masks[guide + 16u + word]) & bit) != 0u;
    return sampled && visible != shadowed;
}

// Picks the samples of `block`; returns the mask of the samples that need a ray of their own.
fn generate_block(block: vec2<u32>) -> u32 {
    let base = ml_block_base(block);
    let size = vec2<u32>(frame.viewport.xy);
    // The representative pixel: the first of this frame's order that is on screen and not sky.
    var header = 0u;
    for (var k = 0u; k < 4u; k++) {
        let index = ml_candidate_index(block, k);
        let p = ml_block_pixel(block, index);
        if all(p < size) && textureLoad(depth_texture, p, 0).r > 0.0 {
            header = index | ML_VALID;
            break;
        }
    }
    for (var k = 0u; k < ML_SAMPLES; k++) {
        samples[base + k] = ML_NO_LIGHT;
    }
    // Every sample its own ray's source until shared (0, 1, 2, 3).
    let own = 0xe4u << ML_SOURCE_SHIFT;
    samples[base + ML_HEADER] = header | own;
    if header == 0u {
        return 0u;
    }
    let pixel = ml_block_pixel(block, header & 3u);
    let s = ml_surface(pixel);
    let list = cluster_index(s.uv, s.linear_depth) * CLUSTER_WORDS;
    let count = light_grid[list];
    if count == 0u {
        return 0u;
    }
    var rng = ml_seed(pixel, 0u);
    let guide = previous_guide(s.world);

    // Candidates: the whole list, or a stratified subset in which every light has the chance
    // ML_CANDIDATES / count (its weight scaled by the inverse).
    let candidates = min(count, ML_CANDIDATES);
    let stride = f32(count) / f32(candidates);
    let offset = ml_next(&rng);
    var weights: array<f32, ML_CANDIDATES>;
    var ids: array<u32, ML_CANDIDATES>;
    var total = 0.0;
    for (var i = 0u; i < candidates; i++) {
        let entry = select(i, min(u32((f32(i) + offset) * stride), count - 1u), count > ML_CANDIDATES);
        let index = light_grid[list + 1u + entry];
        let c = ml_lighting(s, lights[index]);
        var w = luminance(s.diffuse_color * c.diffuse + c.specular) + ML_WEIGHT_FLOOR * luminance(c.radiance);
        if guide != INVALID_ID {
            let word = ml_guide_word(index);
            let bit = ml_guide_bit(index);
            let sampled = (atomicLoad(&guide_masks[guide + word]) & bit) != 0u;
            let visible = (atomicLoad(&guide_masks[guide + 8u + word]) & bit) != 0u;
            if sampled && !visible {
                w *= ML_HIDDEN_WEIGHT;
            }
        }
        weights[i] = w * stride;
        ids[i] = index;
        total += weights[i];
    }
    if total <= 0.0 {
        return 0u;
    }

    // Systematic resampling: ML_SAMPLES evenly spaced points of the weights' running sum, so each
    // pick has the probability weight / total and the picks spread over the lights.
    let start = ml_next(&rng);
    var i = 0u;
    var running = weights[0];
    var traced = 0u;
    var sources = 0u;
    var last_light = ML_NO_LIGHT;
    var last_source = 0u;
    for (var k = 0u; k < ML_SAMPLES; k++) {
        let u = (f32(k) + start) / f32(ML_SAMPLES) * total;
        while running < u && i + 1u < candidates {
            i++;
            running += weights[i];
        }
        var source = k;
        if weights[i] > 0.0 {
            samples[base + k] = ids[i];
            // Unbiased contribution weight of the pick: total / (its weight / stride).
            samples[base + ML_WEIGHT + k] = bitcast<u32>(total * stride / weights[i]);
            // The picks run through the candidates in order: a repeated light follows itself.
            if ML_SHARED_RAYS && ids[i] == last_light && ml_agreed(guide, ids[i]) {
                source = last_source;
            } else {
                traced |= 1u << k;
                last_light = ids[i];
                last_source = k;
            }
        }
        sources |= source << (2u * k);
    }
    samples[base + ML_HEADER] = header | (sources << ML_SOURCE_SHIFT);
    return traced;
}

var<workgroup> group_rays: atomic<u32>;
var<workgroup> group_base: u32;

@compute @workgroup_size(8, 8)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        atomicStore(&group_rays, 0u);
    }
    workgroupBarrier();
    var traced = 0u;
    if all(gid.xy < ml_blocks()) {
        traced = generate_block(gid.xy);
    }
    // Compact this workgroup's rays into one range of the list (rays stay near their blocks).
    let offset = atomicAdd(&group_rays, countOneBits(traced));
    workgroupBarrier();
    if lid == 0u {
        group_base = atomicAdd(&ray_args[3], atomicLoad(&group_rays));
    }
    workgroupBarrier();
    let block_index = gid.y * ml_blocks().x + gid.x;
    var j = group_base + offset;
    while traced != 0u {
        let k = firstTrailingBit(traced);
        traced &= traced - 1u;
        rays[1u + j] = block_index * 4u + k;
        j++;
    }
}

// The trace's indirect dispatch for the rays generate listed.
@compute @workgroup_size(1)
fn ray_args_main() {
    let count = atomicLoad(&ray_args[3]);
    let groups = (count + ML_TRACE_GROUP - 1u) / ML_TRACE_GROUP;
    atomicStore(&ray_args[0], min(groups, ML_MAX_GROUPS_X));
    atomicStore(&ray_args[1], (groups + ML_MAX_GROUPS_X - 1u) / ML_MAX_GROUPS_X);
    atomicStore(&ray_args[2], 1u);
    rays[0] = count;
}

// ---- Shadow rays -----------------------------------------------------------------------------------

// Whether the ray from `origin` along `dir` passes through a surface of the depth buffer within
// `max_distance`: contact shadows of details the shadow maps and fields are too coarse for. Only
// the depth buffer's front faces are known, so surfaces count as CONTACT_THICKNESS-thin shells:
// a ray passing behind thicker geometry is left to the shadow map or the fields, which trace the
// whole ray (a thicker shell would shadow walls behind any object in front of them).
fn screen_trace(origin: vec3<f32>, dir: vec3<f32>, max_distance: f32, jitter: f32) -> bool {
    if ML_HZB_TRACE {
        // Every pixel along the ray, through the closest-depth HZB (the same shells), from two
        // pixels out as below.
        return hzb_trace(origin, origin + dir * max_distance, 2.0, vec2<f32>(CONTACT_THICKNESS, 0.1), 0.002, ML_HZB_STEPS).hit;
    }
    let size = frame.viewport.xy;
    let origin_clip = frame.view_proj * vec4<f32>(origin, 1.0);
    // Start a couple of pixels out, where the receiver's own depth no longer masks the ray.
    let t0 = max(0.01, origin_clip.w * frame.viewport.w * 2.0);
    if max_distance <= t0 {
        return false;
    }
    for (var i = 0u; i < ML_SCREEN_STEPS; i++) {
        let f = (f32(i) + jitter) / f32(ML_SCREEN_STEPS);
        let t = t0 + (max_distance - t0) * f * f;
        let clip = frame.view_proj * vec4<f32>(origin + dir * t, 1.0);
        if clip.w <= frame.camera_position.w {
            return false;
        }
        let uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            return false;
        }
        let pixel = vec2<u32>(uv * size);
        let scene = frame.camera_position.w / max(textureLoad(depth_texture, pixel, 0).r, 1e-9);
        let behind = clip.w - scene;
        // Emissive surfaces are the lights' own visible shapes (candle flames, which cast no
        // shadows): the depth buffer holds them, but they must not block their own light.
        if behind > 0.002 * scene && behind < CONTACT_THICKNESS * (1.0 + 0.1 * scene) && textureLoad(gbuffer_motion, pixel, 0).z <= 0.0 {
            return true;
        }
    }
    return false;
}

// Emissive surfaces are the lights' own visible shapes (candle flames, which cast no shadows):
// the depth buffer holds them, but they must not block their own light.
fn hzb_trace_accept(pixel: vec2<u32>) -> bool {
    return textureLoad(gbuffer_motion, pixel, 0).z <= 0.0;
}

@compute @workgroup_size(64)
fn trace(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let id = (wg.y * groups.x + wg.x) * ML_TRACE_GROUP + lid;
    if id >= rays[0] {
        return;
    }
    let ray = rays[1u + id];
    let k = ray & 3u;
    let block_index = ray >> 2u;
    let blocks_x = ml_blocks().x;
    let block = vec2<u32>(block_index % blocks_x, block_index / blocks_x);
    let base = ml_block_base(block);
    let header = samples[base + ML_HEADER];
    let index = samples[base + k] & 0xffffu;
    if (header & ML_VALID) == 0u || index == ML_NO_LIGHT {
        return;
    }
    let pixel = ml_block_pixel(block, header & 3u);
    let s = ml_surface(pixel);
    let light = lights[index];
    var rng = ml_seed(pixel, 1u + k);

    var visibility = 1.0;
    if light.params.z > 0.5 {
        // A point on the disk the light's sphere shows the receiver.
        let center = light.position_range.xyz;
        let radius = light.color.w;
        let l = normalize(center - s.world);
        let tangent = local_tangent(l);
        let disk = sqrt(ml_next(&rng)) * radius;
        let angle = 6.2831853 * ml_next(&rng);
        let point = center + (tangent * cos(angle) + cross(l, tangent) * sin(angle)) * disk;
        let origin = s.world + s.n * (0.01 + 0.002 * s.linear_depth);
        let dir = normalize(point - origin);
        let jitter = ml_next(&rng);
        if screen_trace(origin, dir, min(ML_SCREEN_DISTANCE, max(distance(point, origin) - radius, 0.0)), jitter) {
            visibility = 0.0;
        } else {
            let slot = light_shadow_slot(light);
            if slot != INVALID_ID {
                visibility = vsm_ray(slot, s.world, s.n, point, radius, jitter, 0u);
            } else if lumen.sdf_enabled != 0u {
                visibility = sdf_ray(origin, point, center, radius);
            }
        }
    }
    samples[base + k] = index | (u32(saturate(visibility) * 65535.0 + 0.5) << 16u);

    let guide = guide_half(false) + ml_guide_base(pixel, frame.viewport);
    let word = ml_guide_word(index);
    let bit = ml_guide_bit(index);
    atomicOr(&guide_masks[guide + word], bit);
    if visibility > 0.25 {
        atomicOr(&guide_masks[guide + 8u + word], bit);
    }
    if visibility < 0.75 {
        atomicOr(&guide_masks[guide + 16u + word], bit);
    }
}
