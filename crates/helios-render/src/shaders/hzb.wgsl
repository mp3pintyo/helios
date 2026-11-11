// Hierarchical-Z pyramid (reverse-Z, so each texel stores the *farthest* = minimum depth).
// Mip 0 is half resolution; texel (x, y) of mip L covers pixels [x, x+1) * 2^(L+1).
// Only the corner covered by the internal resolution is built (the targets are allocated for
// the largest one, see targets.rs); reads clamp to the source's covered corner, so texels
// beyond it, left over from larger frames, never leak in.

struct HzbLevel {
    source_limit: vec2<u32>,  // last covered texel of the source
    size: vec2<u32>,          // covered texels of the destination
}

@group(0) @binding(0) var depth: texture_2d<f32>;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var destination: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> hzb_level: HzbLevel;

@compute @workgroup_size(8, 8)
fn from_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = hzb_level.size;
    if any(gid.xy >= size) {
        return;
    }
    let limit = hzb_level.source_limit;
    let base = gid.xy * 2u;
    let d = min(
        min(textureLoad(depth, min(base, limit), 0).r, textureLoad(depth, min(base + vec2<u32>(1u, 0u), limit), 0).r),
        min(textureLoad(depth, min(base + vec2<u32>(0u, 1u), limit), 0).r, textureLoad(depth, min(base + vec2<u32>(1u, 1u), limit), 0).r),
    );
    textureStore(destination, gid.xy, vec4<f32>(d));
}

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = hzb_level.size;
    if any(gid.xy >= size) {
        return;
    }
    let limit = hzb_level.source_limit;
    let base = gid.xy * 2u;
    let d = min(
        min(textureLoad(source, min(base, limit), 0).r, textureLoad(source, min(base + vec2<u32>(1u, 0u), limit), 0).r),
        min(textureLoad(source, min(base + vec2<u32>(0u, 1u), limit), 0).r, textureLoad(source, min(base + vec2<u32>(1u, 1u), limit), 0).r),
    );
    textureStore(destination, gid.xy, vec4<f32>(d));
}

// Closest-depth pyramid (reverse-Z: the *maximum*), for screen traces (hzb_trace.wgsl): a ray
// in front of a texel's closest depth passes everything under it. Built from the final depth
// buffer (after the late pass), where the farthest-depth pyramid above is built from the
// early pass for occlusion culling.
@compute @workgroup_size(8, 8)
fn from_depth_closest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = hzb_level.size;
    if any(gid.xy >= size) {
        return;
    }
    let limit = hzb_level.source_limit;
    let base = gid.xy * 2u;
    let d = max(
        max(textureLoad(depth, min(base, limit), 0).r, textureLoad(depth, min(base + vec2<u32>(1u, 0u), limit), 0).r),
        max(textureLoad(depth, min(base + vec2<u32>(0u, 1u), limit), 0).r, textureLoad(depth, min(base + vec2<u32>(1u, 1u), limit), 0).r),
    );
    textureStore(destination, gid.xy, vec4<f32>(d));
}

@compute @workgroup_size(8, 8)
fn downsample_closest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = hzb_level.size;
    if any(gid.xy >= size) {
        return;
    }
    let limit = hzb_level.source_limit;
    let base = gid.xy * 2u;
    let d = max(
        max(textureLoad(source, min(base, limit), 0).r, textureLoad(source, min(base + vec2<u32>(1u, 0u), limit), 0).r),
        max(textureLoad(source, min(base + vec2<u32>(0u, 1u), limit), 0).r, textureLoad(source, min(base + vec2<u32>(1u, 1u), limit), 0).r),
    );
    textureStore(destination, gid.xy, vec4<f32>(d));
}
