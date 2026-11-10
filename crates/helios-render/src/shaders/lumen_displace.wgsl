// Lumen's view of displaced surfaces (lumen_displaced.rs): moves points of a mesh's surface by its
// material's displacement with the tessellator's own functions — displacement_height.wgsl, and for
// material graphs the same generated dispatch module the dicing passes run in place of
// displacement_graphs_off.wgsl — so the distance fields and mesh cards baked from the result
// show the surface Nanite tessellation renders.
//
// The points come from a refinement of the mesh's triangles (shared points once each). Every
// job dispatch displaces the points of one (mesh, displacement) variant through a scene
// instance that has it: its transform, scale and material, exactly as the tessellator reads
// them. Textures and noise are filtered for the spacing of the points (the `tess` header of the
// job, see lumen_displaced.rs), not for a camera. An animated displacement is averaged over
// `samples` times: Lumen represents its mean surface; w holds the RMS distance of the moving
// surface from it.

#include "common.wgsl"
#include "displacement.wgsl"
#include "displacement_height.wgsl"

struct DisplaceJob {
    instance: u32,     // the scene instance whose transform and material displace the points
    first: u32,        // first point of the variant
    count: u32,
    samples: u32,      // time samples averaged (1: a static displacement)
    time: vec4<f32>,   // x: the first sample's time, y: the spacing of the samples (seconds)
}

struct SurfacePoint {
    object_u: vec4<f32>,   // undisplaced object-space position, mesh U
    normal_v: vec4<f32>,   // interpolated object-space normal (not normalised), mesh V
}

@group(0) @binding(0) var<uniform> job: DisplaceJob;
@group(0) @binding(1) var<storage, read> points: array<SurfacePoint>;
@group(0) @binding(2) var<storage, read_write> displaced: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(40) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(43) var material_data: texture_2d_array<f32>;
@group(0) @binding(44) var material_sampler: sampler;
@group(0) @binding(67) var<uniform> tess: Tessellation;

@compute @workgroup_size(64)
fn displace(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= job.count {
        return;
    }
    let index = job.first + id.x;
    let point = points[index];
    let instance = instances[job.instance];
    let normal = normalize(point.normal_v.xyz);
    let uv = vec2<f32>(point.object_u.w, point.normal_v.w);
    // Offsets from the first sample, so the variance does not cancel against large coordinates.
    let first = displacement_position(instance, displacement_input(instance, point.object_u.xyz, normal, uv, job.time.x));
    var sum = vec3<f32>(0.0);
    var squares = 0.0;
    for (var k = 1u; k < job.samples; k++) {
        let t = job.time.x + (f32(k) * job.time.y);
        let d = displacement_position(instance, displacement_input(instance, point.object_u.xyz, normal, uv, t)) - first;
        sum += d;
        squares += dot(d, d);
    }
    let n = f32(job.samples);
    let mean = sum / n;
    displaced[index] = vec4<f32>(first + mean, sqrt(max((squares / n) - dot(mean, mean), 0.0)));
}
