// What a material graph's Displacement output reads (helios-material's displacement context,
// tessellation/graphs.rs): the undisplaced point, its texture filtering and the helpers the
// generated `graph_displacement` code calls. Shared by the tessellation passes (through the
// generated dispatch module) and the extended G-buffer resolve of a displacing graph
// (displacement_graph_resolve.wgsl), so every pass computes the same height.
// Requires common.wgsl, displacement.wgsl, displacement_height.wgsl and the bindings
// `materials`, `material_albedo`, `material_data`, `material_sampler` and
//   graph_params: array<vec4<f32>, 1024> (uniform: the material instances' parameter values).

#include "noise.wgsl"

// The undisplaced point as graph code sees it: the field names of the resolve's SurfaceInput, so
// one node emits the same code in both contexts. The "derivatives" are not screen-space: the
// tessellator has none. They span one dice segment at the point (along two directions of the
// tangent plane), so textures and noise are filtered for the micro-triangles that sample them, as
// the texture-driven height is (displacement_height.wgsl).
struct DisplacementSurface {
    uv: vec2<f32>,               // mesh UV set 0
    duv_dx: vec2<f32>,
    duv_dy: vec2<f32>,
    world_position: vec3<f32>,   // instance transform, no wind
    dpdx: vec3<f32>,
    dpdy: vec3<f32>,
    object_position: vec3<f32>,
    object_dpdx: vec3<f32>,
    object_dpdy: vec3<f32>,
    normal: vec3<f32>,           // interpolated world normal
}

fn displacement_surface(material: u32, max_scale: f32, p: DisplacementInput) -> DisplacementSurface {
    let segment = displacement_segment(p.world);
    let object_segment = segment / max(max_scale, 1e-6);
    let uv_segment = object_segment * tess.materials[material].params.w;
    let world_frame = basis(p.world_normal);
    let object_frame = basis(p.normal);
    var s: DisplacementSurface;
    s.uv = p.uv;
    s.duv_dx = vec2<f32>(uv_segment, 0.0);
    s.duv_dy = vec2<f32>(0.0, uv_segment);
    s.world_position = p.world;
    s.dpdx = world_frame[0] * segment;
    s.dpdy = world_frame[1] * segment;
    s.object_position = p.object;
    s.object_dpdx = object_frame[0] * object_segment;
    s.object_dpdy = object_frame[1] * object_segment;
    s.normal = p.world_normal;
    return s;
}

// The texture layer of texture parameter `slot` (unset: the material's own texture).
fn displacement_graph_texture(m: Material, params: u32, slot: u32) -> u32 {
    let layer = bitcast<u32>(graph_params[params + slot].x);
    return select(layer, m.texture, layer == 0xfffffffeu);
}

struct DisplacementTexels {
    albedo: vec4<f32>,
    data: vec4<f32>,
}

// A layer's texels from the 512² arrays (streamed tiles are not consulted: their residency
// differs between frames and passes, the height must not); white and flat without a texture.
fn displacement_graph_sample(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> DisplacementTexels {
    var t: DisplacementTexels;
    if layer == DISPLACEMENT_NO_LAYER {
        t.albedo = vec4<f32>(1.0);
        t.data = vec4<f32>(0.5, 0.5, 1.0, 1.0);
        return t;
    }
    t.albedo = textureSampleGrad(material_albedo, material_sampler, uv, layer, ddx, ddy);
    t.data = textureSampleGrad(material_data, material_sampler, uv, layer, ddx, ddy);
    return t;
}

struct DisplacementTriplanar {
    albedo: vec3<f32>,
    normal: vec3<f32>,   // world space
    roughness: f32,
    height: f32,
    alpha: f32,
}

fn displacement_graph_triplanar(layer: u32, p: vec3<f32>, dpx: vec3<f32>, dpy: vec3<f32>, n: vec3<f32>) -> DisplacementTriplanar {
    var t: DisplacementTriplanar;
    var w = pow(abs(n), vec3<f32>(4.0));
    w /= w.x + w.y + w.z;
    let x = displacement_graph_sample(layer, p.zy, dpx.zy, dpy.zy);
    let y = displacement_graph_sample(layer, p.xz, dpx.xz, dpy.xz);
    let z = displacement_graph_sample(layer, p.xy, dpx.xy, dpy.xy);
    let tx = (x.data.xy * 2.0) - 1.0;
    let ty = (y.data.xy * 2.0) - 1.0;
    let tz = (z.data.xy * 2.0) - 1.0;
    t.albedo = (x.albedo.rgb * w.x) + (y.albedo.rgb * w.y) + (z.albedo.rgb * w.z);
    t.normal = normalize(n + (vec3<f32>(0.0, tx.y, tx.x) * w.x) + (vec3<f32>(ty.x, 0.0, ty.y) * w.y) + (vec3<f32>(tz.x, tz.y, 0.0) * w.z));
    t.roughness = (x.data.z * w.x) + (y.data.z * w.y) + (z.data.z * w.z);
    t.height = (x.data.w * w.x) + (y.data.w * w.y) + (z.data.w * w.z);
    t.alpha = (x.albedo.a * w.x) + (y.albedo.a * w.y) + (z.albedo.a * w.z);
    return t;
}

// As decode_tangent_normal (material.wgsl) and graph_blend_normals (material_graph.wgsl), which
// the tessellation passes do not include.
fn displacement_decode_normal(data: vec4<f32>, strength: f32) -> vec3<f32> {
    let xy = ((data.xy * 2.0) - 1.0) * strength;
    return normalize(vec3<f32>(xy, sqrt(saturate(1.0 - dot(xy, xy))) + 1e-4));
}

fn displacement_blend_normals(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize((t * (dot(t, u) / t.z)) - u);
}
