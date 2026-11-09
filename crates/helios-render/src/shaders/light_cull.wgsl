// Clustered light culling: one thread per froxel cluster tests every light's bounding sphere
// against the cluster's view-space bounding box.

#include "common.wgsl"
#include "lights.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read_write> light_grid: array<u32>;

fn view_ray(uv: vec2<f32>) -> vec3<f32> {
    let eye = frame.camera_position.xyz;
    let world_dir = view_direction(uv, frame.inv_view_proj, eye);
    let d = (frame.view * vec4<f32>(world_dir, 0.0)).xyz;
    return d / max(-d.z, 1e-6); // scaled so that z = -1
}

@compute @workgroup_size(64)
fn cull_lights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= CLUSTER_X * CLUSTER_Y * CLUSTER_Z {
        return;
    }
    let x = index % CLUSTER_X;
    let y = (index / CLUSTER_X) % CLUSTER_Y;
    let z = index / (CLUSTER_X * CLUSTER_Y);
    let uv0 = vec2<f32>(f32(x), f32(y)) / vec2<f32>(f32(CLUSTER_X), f32(CLUSTER_Y));
    let uv1 = vec2<f32>(f32(x + 1u), f32(y + 1u)) / vec2<f32>(f32(CLUSTER_X), f32(CLUSTER_Y));
    let near = slice_depth(f32(z));
    let far = slice_depth(f32(z + 1u));
    var lo = vec3<f32>(1e30);
    var hi = vec3<f32>(-1e30);
    for (var c = 0u; c < 4u; c++) {
        let uv = vec2<f32>(select(uv0.x, uv1.x, (c & 1u) != 0u), select(uv0.y, uv1.y, (c & 2u) != 0u));
        let ray = view_ray(uv);
        lo = min(lo, min(ray * near, ray * far));
        hi = max(hi, max(ray * near, ray * far));
    }

    var count = 0u;
    let base = index * CLUSTER_WORDS;
    for (var i = 0u; i < frame.light_count; i++) {
        let light = lights[i];
        let center = (frame.view * vec4<f32>(light.position_range.xyz, 1.0)).xyz;
        let closest = clamp(center, lo, hi);
        let delta = center - closest;
        if dot(delta, delta) <= light.position_range.w * light.position_range.w && count < LIGHTS_PER_CLUSTER {
            light_grid[base + 1u + count] = i;
            count++;
        }
    }
    light_grid[base] = count;
}
