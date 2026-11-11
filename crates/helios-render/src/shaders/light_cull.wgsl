// Clustered light culling, hierarchical: one workgroup per coarse cell of COARSE_X x COARSE_Y
// screen tiles x COARSE_Z depth slices (48 froxels). The workgroup streams the scene's lights
// through shared memory 64 at a time: each thread moves one light into view space and tests its
// bounding sphere against the cell's box, setting a bit of the chunk's mask; then each froxel's
// thread tests only the lights of the mask against its own box. Lights are visited in index
// order, so every froxel lists them in the same order as a brute-force loop would (the cap of
// LIGHTS_PER_CLUSTER drops the same ones), at a fraction of the tests: a light far from a cell
// costs one test for its 48 froxels.
//
// Importance cull (`cull.x` > 0, MegaLights only; UE 5.8's light attenuation falloff): a light's
// bounding sphere shrinks to the distance where its incident radiance (power x inverse-square
// falloff), at last frame's exposure, falls below `cull.x` - a faint candle ten metres away
// lights nothing visible, though its range reaches - so sampling spends its candidates on lights
// that matter. Specular highlights of a culled light on glossy surfaces are lost with it.

#include "common.wgsl"
#include "lights.wgsl"

struct LightCullParams {
    // x: importance threshold on exposed incident radiance (0: off).
    cull: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read_write> light_grid: array<u32>;
@group(0) @binding(3) var<uniform> params: LightCullParams;
// [0]: last frame's exposure multiplier (exposure.rs).
@group(0) @binding(4) var<storage, read> exposure: array<f32, 4>;

const COARSE_X: u32 = 4u;
const COARSE_Y: u32 = 3u;
const COARSE_Z: u32 = 4u;
const CELLS_X: u32 = CLUSTER_X / COARSE_X;
const CELLS_Y: u32 = CLUSTER_Y / COARSE_Y;
const FROXELS_PER_CELL: u32 = COARSE_X * COARSE_Y * COARSE_Z;
const CHUNK: u32 = 64u;

// The lights of the current chunk in view space (centre, range) and, per parity, the mask of
// those touching the coarse cell.
var<workgroup> spheres: array<vec4<f32>, CHUNK>;
var<workgroup> chunk_mask: array<atomic<u32>, 4>;

fn view_ray(uv: vec2<f32>) -> vec3<f32> {
    let eye = frame.camera_position.xyz;
    let world_dir = view_direction(uv, frame.inv_view_proj, eye);
    let d = (frame.view * vec4<f32>(world_dir, 0.0)).xyz;
    return d / max(-d.z, 1e-6); // scaled so that z = -1
}

struct Box {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

// View-space bounding box of tiles [t0, t1) and slices [z0, z1).
fn froxel_box(t0: vec2<u32>, t1: vec2<u32>, z0: u32, z1: u32) -> Box {
    let grid = vec2<f32>(f32(CLUSTER_X), f32(CLUSTER_Y));
    let uv0 = vec2<f32>(t0) / grid;
    let uv1 = vec2<f32>(t1) / grid;
    let near = slice_depth(f32(z0));
    let far = slice_depth(f32(z1));
    var b: Box;
    b.lo = vec3<f32>(1e30);
    b.hi = vec3<f32>(-1e30);
    for (var c = 0u; c < 4u; c++) {
        let uv = vec2<f32>(select(uv0.x, uv1.x, (c & 1u) != 0u), select(uv0.y, uv1.y, (c & 2u) != 0u));
        let ray = view_ray(uv);
        b.lo = min(b.lo, min(ray * near, ray * far));
        b.hi = max(b.hi, max(ray * near, ray * far));
    }
    return b;
}

fn sphere_touches(b: Box, sphere: vec4<f32>) -> bool {
    let delta = sphere.xyz - clamp(sphere.xyz, b.lo, b.hi);
    return dot(delta, delta) <= sphere.w * sphere.w;
}

@compute @workgroup_size(64)
fn cull_lights(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cell = vec3<u32>(wg.x % CELLS_X, (wg.x / CELLS_X) % CELLS_Y, wg.x / (CELLS_X * CELLS_Y));
    let first = cell * vec3<u32>(COARSE_X, COARSE_Y, COARSE_Z);
    let coarse = froxel_box(first.xy, first.xy + vec2<u32>(COARSE_X, COARSE_Y), first.z, first.z + COARSE_Z);
    // This thread's froxel (threads beyond FROXELS_PER_CELL only load lights).
    let owner = t < FROXELS_PER_CELL;
    let offset = vec3<u32>(t % COARSE_X, (t / COARSE_X) % COARSE_Y, t / (COARSE_X * COARSE_Y));
    let froxel = first + select(vec3<u32>(0u), offset, owner);
    let fine = froxel_box(froxel.xy, froxel.xy + 1u, froxel.z, froxel.z + 1u);
    let index = (froxel.z * CLUSTER_Y + froxel.y) * CLUSTER_X + froxel.x;
    let base = index * CLUSTER_WORDS;
    // Importance cull: the radiance scale at distance d is power / d² (the window only lowers it).
    let cull_scale = select(0.0, exposure[0] / params.cull.x, params.cull.x > 0.0);

    if t < 4u {
        atomicStore(&chunk_mask[t], 0u);
    }
    workgroupBarrier();
    var count = 0u;
    let chunks = (frame.light_count + CHUNK - 1u) / CHUNK;
    for (var chunk = 0u; chunk < chunks; chunk++) {
        let parity = (chunk & 1u) * 2u;
        let i = chunk * CHUNK + t;
        if i < frame.light_count {
            let light = lights[i];
            var range = light.position_range.w;
            if cull_scale > 0.0 {
                range = min(range, sqrt(luminance(light.color.rgb) * cull_scale));
            }
            let sphere = vec4<f32>((frame.view * vec4<f32>(light.position_range.xyz, 1.0)).xyz, range);
            spheres[t] = sphere;
            if sphere_touches(coarse, sphere) {
                atomicOr(&chunk_mask[parity + t / 32u], 1u << (t & 31u));
            }
        }
        workgroupBarrier();
        if owner {
            for (var w = 0u; w < 2u; w++) {
                var m = atomicLoad(&chunk_mask[parity + w]);
                while m != 0u {
                    let bit = firstTrailingBit(m);
                    m &= m - 1u;
                    let k = w * 32u + bit;
                    let sphere = spheres[k];
                    if !sphere_touches(fine, sphere) || count >= LIGHTS_PER_CLUSTER {
                        continue;
                    }
                    light_grid[base + 1u + count] = chunk * CHUNK + k;
                    count++;
                }
            }
        } else if t < FROXELS_PER_CELL + 2u {
            // Two idle threads clear the other parity's mask for the next chunk.
            atomicStore(&chunk_mask[(2u - parity) + (t - FROXELS_PER_CELL)], 0u);
        }
        workgroupBarrier();
    }
    if owner {
        light_grid[base] = count;
    }
}
