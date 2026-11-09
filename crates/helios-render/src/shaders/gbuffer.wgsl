// Visibility buffer -> G-buffer, with material classification (as in Nanite's material passes).
//
//   classify (classify.wgsl): per 8x8 tile, append the tile to the list of every shading model
//              it contains, then write per-model indirect arguments.
//   resolve  : one pipeline per shading model (MATERIAL_KIND override), dispatched indirectly
//              over that model's tiles. Each pixel re-fetches its triangle, computes analytic
//              perspective-correct barycentrics with screen derivatives (Schied & Dachsbacher),
//              interpolates attributes and evaluates its material exactly once. Corners of
//              instances in the wind are displaced and skinned instances posed exactly as the
//              rasterisers placed them (wind.wgsl, skinning.wgsl); motion vectors evaluate the
//              previous frame's offset and palette.
//
// tiles buffer: [0, 8) per-model counts, [8, 40) per-model dispatch args (x, y, z, -),
// [64 + kind * tile_capacity ...) per-model tile lists.

#include "common.wgsl"
#include "noise.wgsl"
#include "material.wgsl"
#include "geometry.wgsl"
#include "rvt.wgsl"
#include "svt.wgsl"
#include "wind.wgsl"
#include "skinning.wgsl"
#include "water.wgsl"
#include "shore.wgsl"

override MATERIAL_KIND: u32 = 0u;
// The scene has skinned instances (passes/material.rs): without them the skinning code is
// compiled out, so it costs other scenes no registers.
override SKINNING: bool = true;

const TILE_LISTS: u32 = 64u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(2) var visibility_buffer: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;
@group(0) @binding(7) var<storage, read> geometry: array<u32>;
@group(0) @binding(8) var<storage, read> tiles: array<u32>;
@group(0) @binding(9) var material_albedo: texture_2d_array<f32>;
@group(0) @binding(10) var material_data: texture_2d_array<f32>;
@group(0) @binding(11) var material_sampler: sampler;
@group(0) @binding(12) var<uniform> rvt: RvtParams;
@group(0) @binding(13) var<storage, read> rvt_entries: array<vec4<u32>>;
@group(0) @binding(14) var<storage, read_write> rvt_requests: array<u32>;
@group(0) @binding(15) var rvt_albedo: texture_2d<f32>;
@group(0) @binding(16) var rvt_detail: texture_2d<f32>;
@group(0) @binding(17) var rvt_sampler: sampler;
@group(0) @binding(18) var gbuffer_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(19) var gbuffer_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(20) var gbuffer_motion: texture_storage_2d<rgba16float, write>;
@group(0) @binding(45) var<uniform> svt: SvtParams;
@group(0) @binding(46) var svt_page_table: texture_2d<u32>;
@group(0) @binding(47) var svt_albedo: texture_2d<f32>;
@group(0) @binding(48) var svt_normal: texture_2d<f32>;
@group(0) @binding(49) var svt_feedback: texture_storage_2d<r32uint, write>;
@group(0) @binding(80) var svt_material: texture_2d<f32>;
@group(0) @binding(65) var<uniform> wind: Wind;
@group(0) @binding(66) var skin_palette: texture_2d<f32>;
@group(0) @binding(90) var<uniform> water: Water;

fn tile_capacity() -> u32 {
    let size = vec2<u32>(frame.viewport.xy);
    return ((size.x + 7u) / 8u) * ((size.y + 7u) / 8u);
}

fn pixel_kind(pixel: vec2<u32>) -> u32 {
    let id = textureLoad(visibility_buffer, pixel, 0).r;
    if id == INVALID_ID {
        return MATERIAL_SKY;
    }
    let entry = visible_clusters[id >> TRIANGLE_BITS];
    return min(materials[instances[entry.x].material].kind, MATERIAL_KINDS - 1u);
}

// ---- Runtime virtual texture sampling --------------------------------------------------------------

struct RvtSample {
    found: bool,
    albedo: vec3<f32>,
    detail: vec3<f32>,  // detail normal (frame-local xy, z), roughness in w of the atlas
    roughness: f32,
}

fn rvt_lookup(uv: vec2<f32>, mip: u32) -> RvtSample {
    var r: RvtSample;
    r.found = false;
    for (var m = mip; m < rvt.mips; m++) {
        let pages = rvt_pages(m);
        let p = min(vec2<u32>(uv * f32(pages)), vec2<u32>(pages - 1u));
        let physical = rvt_entries[rvt_mip_offset(m) + p.y * pages + p.x].x;
        if physical == INVALID_ID {
            continue;
        }
        let span = f32(rvt.page_texels + 2u * rvt.border);
        let tile = vec2<f32>(f32(physical % rvt.tiles_per_row), f32(physical / rvt.tiles_per_row));
        let within = uv * f32(pages) - vec2<f32>(p);
        let atlas = (tile * span + f32(rvt.border) + within * f32(rvt.page_texels)) / rvt.atlas_size;
        let a = textureSampleLevel(rvt_albedo, rvt_sampler, atlas, 0.0);
        let d = textureSampleLevel(rvt_detail, rvt_sampler, atlas, 0.0);
        r.found = true;
        r.albedo = a.rgb;
        let xy = d.xy * 2.0 - 1.0;
        r.detail = vec3<f32>(xy, sqrt(saturate(1.0 - dot(xy, xy))));
        r.roughness = d.z;
        return r;
    }
    return r;
}

// The landscape: its layered material, turning into lake bed and wet shore by water bodies.
fn evaluate_terrain(m: Material, input: SurfaceInput) -> Surface {
    let footprint = max(length(input.dpdx.xz), length(input.dpdy.xz));
    return shore_surface(evaluate_landscape(m, input), input.world_position, input.normal, footprint);
}

fn evaluate_landscape(m: Material, input: SurfaceInput) -> Surface {
    var s = base_surface(m, input);
    let footprint = max(length(input.dpdx.xz), length(input.dpdy.xz));
    if rvt.enabled != 0u {
        let uv = (input.world_position.xz - rvt.world_min) / rvt.world_size;
        let texel0 = rvt.world_size / f32(rvt.virtual_pages * rvt.page_texels);
        let level = clamp(log2(max(footprint / texel0, 1e-6)) + rvt.lod_bias, 0.0, f32(rvt.mips - 1u));
        let mip = u32(level);
        // Feedback: request the page this pixel wants (benign races: all write the same frame).
        let pages = rvt_pages(mip);
        let p = min(vec2<u32>(saturate(uv) * f32(pages)), vec2<u32>(pages - 1u));
        rvt_requests[rvt_mip_offset(mip) + p.y * pages + p.x] = rvt.frame;
        let sample = rvt_lookup(saturate(uv), mip);
        if sample.found {
            s.albedo = sample.albedo * m.base_color.rgb;
            s.roughness = sample.roughness;
            s.normal = normalize(rvt_frame(input.normal) * sample.detail);
            return s;
        }
    }
    // No virtual texture (or nothing resident yet): evaluate the layers directly.
    let t = terrain_layering(input.world_position, input.normal, footprint, m.layers);
    s.albedo = t.albedo * m.base_color.rgb;
    s.roughness = t.roughness;
    s.normal = t.normal;
    return s;
}

// ---- Resolve ----------------------------------------------------------------------------------------

// Debug view 17: each joint's colour, blended by the corners' weights and the barycentrics (grey:
// unskinned).
fn skin_weight_color(instance: Instance, cluster: Cluster, base: u32, corners: vec3<u32>, lambda: vec3<f32>) -> vec3<f32> {
    if !SKINNING || instance_skin(instance) == 0u {
        return vec3<f32>(0.5);
    }
    let section = cluster_skin_section(cluster, base);
    if section == INVALID_ID {
        return vec3<f32>(0.5);
    }
    var color = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i++) {
        let s = vertex_skin(section, corners[i]);
        for (var k = 0u; k < 4u; k++) {
            color += hash_color(s.joints[k] * 2654435761u) * (s.weights[k] * lambda[i]);
        }
    }
    return color;
}

struct Barycentrics {
    lambda: vec3<f32>,
    ddx: vec3<f32>,
    ddy: vec3<f32>,
}

// Perspective-correct barycentrics and their screen derivatives (Schied & Dachsbacher 2015).
fn compute_barycentrics(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, ndc: vec2<f32>, size: vec2<f32>) -> Barycentrics {
    var b: Barycentrics;
    let inv_w = 1.0 / vec3<f32>(c0.w, c1.w, c2.w);
    let n0 = c0.xy * inv_w.x;
    let n1 = c1.xy * inv_w.y;
    let n2 = c2.xy * inv_w.z;
    let a = n2 - n1;
    let c = n0 - n1;
    let inv_det = 1.0 / (a.x * c.y - a.y * c.x);
    var ddx = vec3<f32>(n1.y - n2.y, n2.y - n0.y, n0.y - n1.y) * inv_det * inv_w;
    var ddy = vec3<f32>(n2.x - n1.x, n0.x - n2.x, n1.x - n0.x) * inv_det * inv_w;
    var ddx_sum = dot(ddx, vec3<f32>(1.0));
    var ddy_sum = dot(ddy, vec3<f32>(1.0));
    let delta = ndc - n0;
    let interp_inv_w = inv_w.x + delta.x * ddx_sum + delta.y * ddy_sum;
    let interp_w = 1.0 / interp_inv_w;
    b.lambda = interp_w * (vec3<f32>(inv_w.x, 0.0, 0.0) + delta.x * ddx + delta.y * ddy);

    ddx *= 2.0 / size.x;
    ddy *= -2.0 / size.y;
    ddx_sum *= 2.0 / size.x;
    ddy_sum *= -2.0 / size.y;
    let interp_w_ddx = 1.0 / (interp_inv_w + ddx_sum);
    let interp_w_ddy = 1.0 / (interp_inv_w + ddy_sum);
    b.ddx = interp_w_ddx * (b.lambda * interp_inv_w + ddx) - b.lambda;
    b.ddy = interp_w_ddy * (b.lambda * interp_inv_w + ddy) - b.lambda;
    return b;
}

fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn lod_color(level: u32) -> vec3<f32> {
    var palette = array<vec3<f32>, 8>(
        vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.55, 0.15), vec3<f32>(0.95, 0.9, 0.2), vec3<f32>(0.35, 0.85, 0.25),
        vec3<f32>(0.2, 0.8, 0.8), vec3<f32>(0.25, 0.4, 0.95), vec3<f32>(0.6, 0.3, 0.9), vec3<f32>(0.9, 0.35, 0.75),
    );
    return palette[level % 8u];
}

fn interpolate3(v: array<vec3<f32>, 3>, l: vec3<f32>) -> vec3<f32> {
    return v[0] * l.x + v[1] * l.y + v[2] * l.z;
}

fn interpolate2(v: array<vec2<f32>, 3>, l: vec3<f32>) -> vec2<f32> {
    return v[0] * l.x + v[1] * l.y + v[2] * l.z;
}

fn interpolate4(v: array<vec4<f32>, 3>, l: vec3<f32>) -> vec4<f32> {
    return v[0] * l.x + v[1] * l.y + v[2] * l.z;
}

fn write_sky(pixel: vec2<u32>, uv: vec2<f32>) {
    // Motion comes from camera rotation only.
    let eye = frame.camera_position.xyz;
    let far_point = eye + view_direction(uv, frame.inv_view_proj, eye) * 1.0e5;
    let motion = clip_to_uv(frame.unjittered_view_proj * vec4<f32>(far_point, 1.0))
        - clip_to_uv(frame.prev_unjittered_view_proj * vec4<f32>(far_point, 1.0));
    textureStore(gbuffer_albedo, pixel, vec4<f32>(0.0));
    textureStore(gbuffer_normal, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    textureStore(gbuffer_motion, pixel, vec4<f32>(motion, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn resolve(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile_index = wid.x + wid.y * groups.x;
    if tile_index >= tiles[MATERIAL_KIND] {
        return;
    }
    let tile = tiles[TILE_LISTS + MATERIAL_KIND * tile_capacity() + tile_index];
    let pixel = vec2<u32>(tile & 0xffffu, tile >> 16u) * 8u + lid.xy;
    let size = vec2<u32>(frame.viewport.xy);
    if any(pixel >= size) || pixel_kind(pixel) != MATERIAL_KIND {
        return;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    if MATERIAL_KIND == MATERIAL_SKY {
        write_sky(pixel, uv);
        svt_write_feedback(pixel, size);
        return;
    }

    let id = textureLoad(visibility_buffer, pixel, 0).r;
    let slot = id >> TRIANGLE_BITS;
    let triangle = id & TRIANGLE_MASK;
    let entry = visible_clusters[slot];
    let instance = instances[entry.x];
    let cluster = clusters[entry.y];
    let base = cluster_base(cluster);
    let corners = cluster_triangle(cluster, base, triangle);
    let wind_force = wind_strength(instance);

    var object_positions: array<vec3<f32>, 3>;
    var world_positions: array<vec3<f32>, 3>;
    var normals: array<vec3<f32>, 3>;
    var uvs: array<vec2<f32>, 3>;
    var uv1s: array<vec2<f32>, 3>;
    var colors: array<vec4<f32>, 3>;
    var clips: array<vec4<f32>, 3>;
    // Skinned instances: the corners' previous positions (motion vectors).
    var previous_worlds: array<vec3<f32>, 3>;
    let skinned = SKINNING && instance_skin(instance) != 0u;
    let extended = cluster_is_extended(cluster, base);
    for (var i = 0u; i < 3u; i++) {
        let local = corners[i];
        // Object-space attributes stay those of the rest pose (object triplanar mapping sticks
        // to the surface); world positions and normals are posed.
        object_positions[i] = vertex_position(cluster, base, local);
        normals[i] = vertex_normal(cluster, base, local);
        var posed = object_positions[i];
        if skinned {
            let v = skinned_vertex(instance, cluster, base, local, object_positions[i], normals[i]);
            posed = v.position;
            normals[i] = v.normal;
            previous_worlds[i] = wind_previous_world_position(instance, wind_force, v.previous);
        }
        world_positions[i] = wind_world_position(instance, wind_force, posed);
        uvs[i] = vertex_uv(cluster, base, local);
        // Optional attributes (imported meshes only).
        uv1s[i] = uvs[i];
        colors[i] = vec4<f32>(1.0);
        if extended {
            if UV_EXTENSIONS {
                uv1s[i] = vertex_uv1(cluster, base, local, uvs[i]);
            }
            colors[i] = vertex_color(cluster, base, local);
        }
        clips[i] = frame.view_proj * vec4<f32>(world_positions[i], 1.0);
    }

    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    var b = compute_barycentrics(clips[0], clips[1], clips[2], ndc, frame.viewport.xy);
    // Texture derivatives per output pixel rather than per internal one: the negative mip bias
    // of temporal super resolution, log2(internal / output). Texture detail (and the streamed
    // texture tiles it requests) is that of the output resolution; TSR resolves it over the
    // jitter cycle.
    b.ddx *= frame.upscale.x;
    b.ddy *= frame.upscale.y;
    var input: SurfaceInput;
    input.object_position = interpolate3(object_positions, b.lambda);
    // The wind offset varies smoothly: evaluated at the pixel, it matches the interpolated corners
    // to far below a pixel, which is all motion vectors need. Skinning is affine per triangle: the
    // corners' previous positions interpolate exactly. (Resolved here, before the material, so
    // the corner arrays need not live through it.)
    var previous_world = wind_previous_world_position(instance, wind_force, input.object_position);
    if skinned {
        previous_world = interpolate3(previous_worlds, b.lambda);
    }
    input.world_position = interpolate3(world_positions, b.lambda);
    input.dpdx = interpolate3(world_positions, b.ddx);
    input.dpdy = interpolate3(world_positions, b.ddy);
    input.object_dpdx = interpolate3(object_positions, b.ddx);
    input.object_dpdy = interpolate3(object_positions, b.ddy);
    input.footprint = max(length(input.dpdx), length(input.dpdy)) / max(instance.max_scale, 1e-4);
    input.uv = interpolate2(uvs, b.lambda);
    input.duv_dx = interpolate2(uvs, b.ddx);
    input.duv_dy = interpolate2(uvs, b.ddy);
    input.uv1 = interpolate2(uv1s, b.lambda);
    input.duv1_dx = interpolate2(uv1s, b.ddx);
    input.duv1_dy = interpolate2(uv1s, b.ddy);

    let normal_matrix = mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz);
    input.normal_matrix = normal_matrix;
    var normal = normalize(normal_matrix * interpolate3(normals, b.lambda));
    let eye = frame.camera_position.xyz;
    let to_eye = normalize(eye - input.world_position);
    let face = cross(world_positions[1] - world_positions[0], world_positions[2] - world_positions[0]);
    input.back_facing = dot(face, to_eye) < 0.0;
    if input.back_facing {
        normal = -normal; // two-sided geometry (leaves) seen from behind
    }
    // Guard against interpolated normals facing away from the viewer at silhouettes.
    normal = normalize(normal + to_eye * max(0.0, -dot(normal, to_eye) + 0.05));
    input.normal = normal;

    // Per-triangle tangent frame from positions and the UVs of the material's texture (no stored
    // tangents needed; a rotated or mirrored texture transform turns the frame with the image).
    var m = materials[instance.material];
    m.base_color *= interpolate4(colors, b.lambda);
    let texture_uv = m.uv_texture;
    let e1 = world_positions[1] - world_positions[0];
    let e2 = world_positions[2] - world_positions[0];
    let t1 = uv_transform_vector(texture_uv, uv_set_of(texture_uv, uvs[1] - uvs[0], uv1s[1] - uv1s[0]));
    let t2 = uv_transform_vector(texture_uv, uv_set_of(texture_uv, uvs[2] - uvs[0], uv1s[2] - uv1s[0]));
    let det = t1.x * t2.y - t2.x * t1.y;
    var tangent = (e1 * t2.y - e2 * t1.y) * select(1.0, 1.0 / det, abs(det) > 1e-12);
    var bitangent = (e2 * t1.x - e1 * t2.x) * select(1.0, 1.0 / det, abs(det) > 1e-12);
    tangent = tangent - normal * dot(normal, tangent);
    if dot(tangent, tangent) < 1e-12 {
        tangent = basis(normal)[0];
    }
    tangent = normalize(tangent);
    let handedness = select(1.0, -1.0, dot(cross(normal, tangent), bitangent) < 0.0);
    input.tangent = tangent;
    // Tangent-space +Y points to decreasing v, the top of the texture image: the OpenGL / glTF
    // normal-map convention, which texgen.wgsl follows too.
    input.bitangent = cross(normal, tangent) * -handedness;

    var surface: Surface;
    switch MATERIAL_KIND {
        case MATERIAL_ROCK: { surface = evaluate_rock(m, input); }
        case MATERIAL_TERRAIN: { surface = evaluate_terrain(m, input); }
        case MATERIAL_METAL: { surface = evaluate_metal(m, input); }
        case MATERIAL_FOLIAGE: { surface = evaluate_foliage(m, input); }
        default: { surface = evaluate_standard(m, input); }
    }

    switch frame.debug_view {
        case DEBUG_CLUSTERS: { surface.albedo = hash_color(entry.y * 7919u + entry.x * 104729u); }
        case DEBUG_TRIANGLES: { surface.albedo = hash_color(entry.y * 128u + triangle + entry.x * 15485863u); }
        case DEBUG_LOD: { surface.albedo = lod_color(cluster_lod_level(cluster)); }
        case DEBUG_INSTANCES: { surface.albedo = hash_color(entry.x); }
        case DEBUG_RASTER_PATH: {
            // Green: hardware, red: software, blue: hardware alpha-masked.
            var path = vec3<f32>(0.15, 0.6, 0.2);
            if entry.z == CLUSTER_SOFTWARE {
                path = vec3<f32>(0.9, 0.25, 0.15);
            } else if entry.z == CLUSTER_MASKED {
                path = vec3<f32>(0.2, 0.35, 0.95);
            }
            surface.albedo = path * (0.6 + 0.4 * hash_color(entry.y).x);
        }
        case DEBUG_MATERIALS: { surface.albedo = hash_color(MATERIAL_KIND * 31u + 7u); }
        case DEBUG_SKIN_WEIGHTS: { surface.albedo = skin_weight_color(instance, cluster, base, corners, b.lambda); }
        default: {}
    }

    let motion = clip_to_uv(frame.unjittered_view_proj * vec4<f32>(input.world_position, 1.0))
        - clip_to_uv(frame.prev_unjittered_view_proj * vec4<f32>(previous_world, 1.0));

    // Albedo is stored with a sqrt curve to spend 8-bit precision on dark values.
    textureStore(gbuffer_albedo, pixel, vec4<f32>(sqrt(saturate(surface.albedo)), surface.metallic));
    textureStore(gbuffer_normal, pixel, vec4<f32>(surface.normal, surface.roughness));
    textureStore(gbuffer_motion, pixel, vec4<f32>(motion, surface.emissive, pack_surface_flags(surface.flags, surface.occlusion)));
    svt_write_feedback(pixel, size);
}
