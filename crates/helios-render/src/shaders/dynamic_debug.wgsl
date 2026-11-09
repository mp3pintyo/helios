// Invalidation debug view (DEBUG_INVALIDATION, dynamic_debug.rs): what moving instances
// invalidated this frame, over the sun-lit G-buffer (movable instances are orange, gbuffer.wgsl):
//   red hatching : the pixel's sun shadow page was rendered this frame (new, or invalidated by
//                  a move, the wind or streaming)
//   cyan tint    : the global distance field brick around the pixel (finest clipmap level
//                  holding it) was recomposed this frame for a moved instance, with the brick's
//                  outline

#include "common.wgsl"
#include "vsm_common.wgsl"
#include "vsm_sample.wgsl"
#include "lumen_types.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var<uniform> vsm: Vsm;
@group(0) @binding(5) var<storage, read> vsm_entries: array<vec4<u32>>;
@group(0) @binding(6) var vsm_atlas: texture_depth_2d;
@group(0) @binding(7) var shadow_sampler: sampler_comparison;
@group(0) @binding(8) var<uniform> lumen: Lumen;
// Per SDF level and brick: recomposed this frame (lumen_sdf.wgsl, compose_listed).
@group(0) @binding(9) var<storage, read> brick_stamps: array<u32>;
@group(0) @binding(10) var output: texture_storage_2d<rgba16float, write>;

const BRICKS_PER_SIDE: i32 = 16;
const BRICK_VOXELS: f32 = 8.0;

// Whether the sun shadow page sampled at `world` was rendered this frame.
fn page_rendered(world: vec3<f32>) -> bool {
    let a = vsm_address(world, vsm_level(world));
    if !a.valid {
        return false;
    }
    let e = vsm_entries[a.entry];
    return e.x != INVALID_ID && e.w == vsm.frame;
}

// Distance (in voxels) to the edge of the recomposed brick holding `world`, or -1.
fn recomposed_brick(world: vec3<f32>) -> f32 {
    for (var level = 0u; level < 4u; level++) {
        let l = lumen.sdf_levels[level];
        let g = (world - l.xyz) / (l.w * BRICK_VOXELS);
        let brick = vec3<i32>(floor(g));
        if any(brick < vec3<i32>(0)) || any(brick >= vec3<i32>(BRICKS_PER_SIDE)) {
            continue;
        }
        let index = u32(brick.x + (BRICKS_PER_SIDE * (brick.y + (BRICKS_PER_SIDE * brick.z))));
        if brick_stamps[level * u32(BRICKS_PER_SIDE * BRICKS_PER_SIDE * BRICKS_PER_SIDE) + index] == 0u {
            return -1.0;
        }
        let f = fract(g) * BRICK_VOXELS;
        let edge = min(f, vec3<f32>(BRICK_VOXELS) - f);
        return min(edge.x, min(edge.y, edge.z));
    }
    return -1.0;
}

@compute @workgroup_size(8, 8)
fn invalidation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if any(gid.xy >= size) {
        return;
    }
    let pixel = gid.xy;
    let depth = textureLoad(depth_texture, pixel, 0).r;
    if depth <= 0.0 {
        return;  // the lighting pass drew the (debug) sky
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world = world_from_depth(uv, depth, frame.inv_view_proj);
    let albedo = textureLoad(gbuffer_albedo, pixel, 0).rgb;
    let n = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let sun = normalize(frame.sun_direction.xyz);
    let n_dot_l = saturate(dot(n, sun));
    var shadow = 0.0;
    if n_dot_l > 0.0 {
        shadow = vsm_shadow(world, n, sun, interleaved_gradient_noise(vec2<f32>(pixel), frame.frame_index));
    }
    var color = (albedo * albedo) * (0.25 + (0.75 * n_dot_l * shadow));
    // Hatched, so movable instances keep their tint underneath.
    if page_rendered(world) && ((pixel.x + pixel.y) / 4u) % 3u == 0u {
        color = mix(color, vec3<f32>(1.0, 0.1, 0.06), 0.7);
    }
    let brick = recomposed_brick(world);
    if brick >= 0.0 {
        color = mix(color, vec3<f32>(0.1, 0.85, 1.0), select(0.2, 0.8, brick < 0.35));
    }
    textureStore(output, pixel, vec4<f32>(color, 1.0));
}
