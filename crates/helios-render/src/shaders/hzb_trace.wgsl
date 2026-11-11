// Screen traces over the closest-depth HZB (hzb.rs `ClosestHzb`; after Uludag 2014, "Hi-Z
// Screen-Space Cone-Traced Reflections", and UE's HZB screen traces). Requires common.wgsl and:
//   frame: Frame, depth_texture: texture_2d<f32> (reverse-Z depth, internal resolution),
//   hzb_closest: texture_2d<f32> (mip L: the maximum depth of 2^(L+1)-pixel squares),
//   fn hzb_trace_accept(pixel: vec2<u32>) -> bool  (the caller's filter of hit pixels, e.g.
//   emissive surfaces that must not occlude their own light).
//
// The ray is walked in screen space, where reverse-Z depth (near / w) is linear along it. Level
// L > 0 reads cells of 2^L pixels from the pyramid: if the ray stays in front of a cell's closest
// depth over its whole span through the cell, it passes everything there and moves on to the
// cell's exit one level coarser; otherwise it goes one level finer. At the pixel level (the
// depth buffer itself) a hit is a ray that crosses the surface's shell - the visible surface and
// `thickness` behind it (the depth buffer knows only front faces) - over its span through the
// pixel; a ray passing further behind continues pixel by pixel. Every pixel the ray crosses is
// considered (a linear march's steps skip thin occluders), in O(log) steps through empty space.

struct HzbTrace {
    hit: bool,
    // The pixel hit and the fraction of the segment where the ray entered it.
    pixel: vec2<u32>,
    t: f32,
    // Iterations used (for budgets and debugging).
    steps: u32,
}

// Linear view depth of a reverse-Z depth value.
fn hzb_linear_depth(depth: f32) -> f32 {
    return frame.camera_position.w / max(depth, 1e-9);
}

// The closest depth of `cell` at `level` (0: the depth buffer's pixel).
fn hzb_closest_depth(cell: vec2<i32>, level: u32) -> f32 {
    if level == 0u {
        return textureLoad(depth_texture, cell, 0).r;
    }
    return textureLoad(hzb_closest, cell, i32(level) - 1).r;
}

// Traces the world-space segment `origin` -> `end` from `start_pixels` pixels out (where the
// receiver's own depth no longer masks the ray). `thickness`: x the shell's thickness in metres
// at the camera, growing by the factor (1 + y · depth); `bias`: the fraction of the depth a hit
// must lie behind the surface (self-intersection). At most `max_steps` iterations: an
// unfinished trace reports no hit.
fn hzb_trace(origin: vec3<f32>, end_point: vec3<f32>, start_pixels: f32, thickness: vec2<f32>, bias: f32, max_steps: u32) -> HzbTrace {
    var result: HzbTrace;
    result.hit = false;
    result.steps = 0u;
    let near = frame.camera_position.w;
    let size = frame.viewport.xy;
    let clip0 = frame.view_proj * vec4<f32>(origin, 1.0);
    var clip1 = frame.view_proj * vec4<f32>(end_point, 1.0);
    if clip0.w <= near {
        return result;
    }
    // Clip the segment in front of the near plane (with a margin: depth grows as 1 / w).
    let w_min = near * 1.5;
    if clip1.w < w_min {
        let s = (clip0.w - w_min) / max(clip0.w - clip1.w, 1e-6);
        clip1 = mix(clip0, clip1, s);
    }
    let p0 = vec3<f32>((vec2<f32>(clip0.x, -clip0.y) / clip0.w * 0.5 + 0.5) * size, near / clip0.w);
    let p1 = vec3<f32>((vec2<f32>(clip1.x, -clip1.y) / clip1.w * 0.5 + 0.5) * size, near / clip1.w);
    let delta = p1 - p0;
    let span = max(abs(delta.x), abs(delta.y));
    if span < start_pixels {
        return result;
    }
    // A hundredth of a pixel past a cell's border.
    let nudge = 0.01 / span;
    let inverse = vec2<f32>(select(1e9, 1.0 / delta.x, abs(delta.x) > 1e-6), select(1e9, 1.0 / delta.y, abs(delta.y) > 1e-6));
    let towards = vec2<f32>(select(0.0, 1.0, delta.x > 0.0), select(0.0, 1.0, delta.y > 0.0));
    let max_level = min(textureNumLevels(hzb_closest), 8u);
    var level = 0u;
    var t = start_pixels / span;
    for (var i = 0u; i < max_steps; i++) {
        result.steps = i + 1u;
        if t >= 1.0 {
            break;
        }
        let p = p0 + delta * t;
        if any(p.xy < vec2<f32>(0.0)) || any(p.xy >= size) {
            break;
        }
        let cell_size = f32(1u << level);
        let cell = floor(p.xy / cell_size);
        // Where the ray leaves the cell (or the segment ends).
        let exits = ((cell + towards) * cell_size - p0.xy) * inverse;
        let t_exit = min(max(min(exits.x, exits.y), t + nudge), 1.0);
        let z_exit = p0.z + delta.z * t_exit;
        let closest = hzb_closest_depth(vec2<i32>(cell), level);
        if min(p.z, z_exit) > closest {
            // In front of everything in the cell: skip it, and look coarser.
            t = t_exit + nudge;
            level = min(level + 1u, max_level);
            continue;
        }
        if level > 0u {
            level -= 1u;
            continue;
        }
        // The pixel: does the ray's depth span through it cross the surface's shell?
        let scene = hzb_linear_depth(closest);
        let w_a = near / max(p.z, 1e-9);
        let w_b = near / max(z_exit, 1e-9);
        let shell_front = scene * (1.0 + bias);
        let shell_back = scene + thickness.x * (1.0 + thickness.y * scene);
        let pixel = vec2<u32>(cell);
        if max(w_a, w_b) > shell_front && min(w_a, w_b) < shell_back && hzb_trace_accept(pixel) {
            result.hit = true;
            result.pixel = pixel;
            result.t = t;
            return result;
        }
        t = t_exit + nudge;
    }
    return result;
}
