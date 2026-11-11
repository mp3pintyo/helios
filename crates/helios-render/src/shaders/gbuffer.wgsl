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
// Materials with a Substrate slab or a material graph resolve in extended pipelines of their own
// tiles (slots 8-23): the shader composed with resolve_extended.wgsl in place of
// resolve_extended_off.wgsl. Plain pipelines compile that path out.
//
// tiles buffer: [0, 24) per-slot counts, [24, 120) per-slot dispatch args (x, y, z, -),
// [120 + slot * tile_capacity ...) per-slot tile lists.

#include "common.wgsl"
#include "noise.wgsl"
#include "material.wgsl"
#include "geometry.wgsl"
#include "rvt.wgsl"
#include "svt.wgsl"
#include "wind.wgsl"
#include "skinning.wgsl"
#include "plant_wind.wgsl"
#include "water.wgsl"
#include "shore.wgsl"
#include "voxel.wgsl"
#include "voxel_resolve.wgsl"
#include "resolve_extended_off.wgsl"
#include "displacement.wgsl"
#include "displacement_height.wgsl"
#include "tess_core.wgsl"
#include "tess_resolve.wgsl"

override MATERIAL_KIND: u32 = 0u;
// The tile list this pipeline resolves: MATERIAL_KIND for plain pipelines, 8-23 for extended ones.
override RESOLVE_SLOT: u32 = 0u;
// The scene has skinned instances (passes/material.rs): without them the skinning code is
// compiled out, so it costs other scenes no registers.
override SKINNING: bool = true;
// The scene has displaced instances (tessellation.rs): without them the displaced-surface resolve
// (tess_resolve.wgsl) is compiled out.
override TESSELLATION: bool = true;
// A debug view recolours the G-buffer (passes/material.rs): the lit view compiles the debug
// colouring out, as keeping its inputs (cluster, triangle, instance, ...) alive across the material
// evaluation costs registers — the resolve ran 1.4x slower with it (cathedral, 1440p: 0.3 ms).
override DEBUG_VIEWS: bool = true;

const RESOLVE_SLOTS: u32 = 24u;
const TILE_LISTS: u32 = 120u;

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
@group(0) @binding(170) var plant_bones: texture_2d<f32>;
@group(0) @binding(171) var plant_sim: texture_2d<f32>;
@group(0) @binding(90) var<uniform> water: Water;
// The simulated river's water surface per cell (river.rs; shore.wgsl).
@group(0) @binding(150) var river_surface: texture_2d<f32>;

fn tile_capacity() -> u32 {
    let size = vec2<u32>(frame.viewport.xy);
    return ((size.x + 7u) / 8u) * ((size.y + 7u) / 8u);
}

fn pixel_slot(pixel: vec2<u32>) -> u32 {
    let id = textureLoad(visibility_buffer, pixel, 0).r;
    if id == INVALID_ID {
        return MATERIAL_SKY;
    }
    let entry = visible_clusters[id >> TRIANGLE_BITS];
    return min(materials[instances[entry.x].material].resolve, RESOLVE_SLOTS - 1u);
}

// The pipeline's shading model.
fn evaluate_kind(m: Material, input: SurfaceInput) -> Surface {
    switch MATERIAL_KIND {
        case MATERIAL_ROCK: { return evaluate_rock(m, input); }
        case MATERIAL_TERRAIN: { return evaluate_terrain(m, input); }
        case MATERIAL_METAL: { return evaluate_metal(m, input); }
        case MATERIAL_FOLIAGE: { return evaluate_foliage(m, input); }
        default: { return evaluate_standard(m, input); }
    }
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
    // A mesh terrain (world triplanar mapping) holds cliffs and overhangs the landscape's
    // top-down virtual texture can't: it is always layered directly, its texel footprint
    // measured in 3D (on a wall the horizontal one vanishes).
    let volumetric = m.mapping == MAPPING_WORLD;
    let footprint = select(max(length(input.dpdx.xz), length(input.dpdy.xz)), max(length(input.dpdx), length(input.dpdy)), volumetric);
    if rvt.enabled != 0u && !volumetric {
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
    // No virtual texture (or nothing resident yet): evaluate the layers directly. A mesh
    // terrain's vertices carry its strata and cavity in their UVs.
    var t: TerrainSample;
    if volumetric {
        let a = RockAttributes(input.uv.x, max(abs(input.duv_dx.x), abs(input.duv_dy.x)), saturate(input.uv.y));
        t = mesh_terrain_layering(input.world_position, input.normal, footprint, m.layers, a);
    } else {
        t = terrain_layering(input.world_position, input.normal, footprint, m.layers);
    }
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
    if tile_index >= tiles[RESOLVE_SLOT] {
        return;
    }
    let tile = tiles[TILE_LISTS + RESOLVE_SLOT * tile_capacity() + tile_index];
    let pixel = vec2<u32>(tile & 0xffffu, tile >> 16u) * 8u + lid.xy;
    let size = vec2<u32>(frame.viewport.xy);
    if any(pixel >= size) || pixel_slot(pixel) != RESOLVE_SLOT {
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
    if cluster_is_voxel(cluster) {
        resolve_voxel(pixel, id);
        return;
    }
    let base = cluster_base(cluster);
    let corners = cluster_triangle(cluster, base, triangle);
    let wind_force = wind_strength(instance);

    var object_positions: array<vec3<f32>, 3>;
    var world_positions: array<vec3<f32>, 3>;
    var normals: array<vec3<f32>, 3>;
    var uvs: array<vec2<f32>, 3>;
    var uv1s: array<vec2<f32>, 3>;
    var colors: array<vec4<f32>, 3>;
    // Plant data (wind weight, baked occlusion; vegetation only).
    var plants: array<vec2<f32>, 3>;
    var clips: array<vec4<f32>, 3>;
    // Skinned instances: the corners' previous positions (motion vectors).
    var previous_worlds: array<vec3<f32>, 3>;
    let skinned = SKINNING && instance_skin(instance) != 0u;
    // A plant with bones (plant_wind.wgsl): corners and normals posed by their bones (the
    // parents' frames once for the triangle when its corners share a bone), motion vectors from
    // the previous pose at the pixel.
    let planted = plant_animated(instance, wind_force);
    var bones: array<u32, 3>;
    let extended = cluster_is_extended(cluster, base);
    // The landscape needs no texture coordinates (its virtual texture and layers are mapped in
    // world space); only the mesh terrain reads its vertex attributes from them.
    let fetch_uvs = MATERIAL_KIND != MATERIAL_TERRAIN || materials[instance.material].mapping == MAPPING_WORLD;
    for (var i = 0u; i < 3u; i++) {
        let local = corners[i];
        // Object-space attributes stay those of the rest pose (object triplanar mapping sticks
        // to the surface); world positions and normals are posed.
        object_positions[i] = vertex_position(cluster, base, local);
        normals[i] = vertex_normal(cluster, base, local);
        plants[i] = vec2<f32>(1.0);
        if extended {
            plants[i] = vertex_plant(cluster, base, local);
        }
        var posed = object_positions[i];
        if skinned {
            let v = skinned_vertex(instance, cluster, base, local, object_positions[i], normals[i]);
            posed = v.position;
            normals[i] = v.normal;
            previous_worlds[i] = wind_previous_world_position(instance, wind_force, v.previous, plants[i].x);
        }
        if planted {
            bones[i] = plant_vertex_bone(cluster, base, local);
        } else {
            world_positions[i] = wind_world_position(instance, wind_force, posed, plants[i].x);
        }
        if fetch_uvs {
            uvs[i] = vertex_uv(cluster, base, local);
        }
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
    if planted {
        let air = plant_air(instance, wind_force, wind_origin(instance), wind.time.x);
        if bones[0] == bones[1] && bones[0] == bones[2] {
            let posed = plant_pose_corners(instance.plant.y, air, bones[0], PlantCorners(object_positions[0], object_positions[1], object_positions[2], normals[0], normals[1], normals[2]));
            world_positions[0] = posed.p0;
            world_positions[1] = posed.p1;
            world_positions[2] = posed.p2;
            normals[0] = posed.n0;
            normals[1] = posed.n1;
            normals[2] = posed.n2;
        } else {
            for (var i = 0u; i < 3u; i++) {
                let posed = plant_pose(instance.plant.y, air, bones[i], object_positions[i], normals[i]);
                world_positions[i] = posed.position;
                normals[i] = posed.normal;
            }
        }
        for (var i = 0u; i < 3u; i++) {
            world_positions[i] = (instance.world * vec4<f32>(world_positions[i], 1.0)).xyz;
            clips[i] = frame.view_proj * vec4<f32>(world_positions[i], 1.0);
        }
    }

    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    var b = compute_barycentrics(clips[0], clips[1], clips[2], ndc, frame.viewport.xy);
    // Texture derivatives per output pixel rather than per internal one: the negative mip bias
    // of temporal super resolution, log2(internal / output). Texture detail (and the streamed
    // texture tiles it requests) is that of the output resolution; TSR resolves it over the
    // jitter cycle.
    b.ddx *= frame.upscale.x;
    b.ddy *= frame.upscale.y;
    // Tessellated: the pixel's point on the displaced surface (position, normal; the barycentrics
    // of the base point it was displaced from). The material is evaluated at that base point —
    // positions, UVs and their derivatives are the undisplaced surface's — as the displacement
    // itself is, so one pattern can drive a material's displacement and its colours and stays
    // attached to the surface however far a vector displacement moves it.
    let displaced = TESSELLATION && is_tessellated(instance.flags, cluster);
    var hit: DisplacedHit;
    if displaced {
        let depth_hit = world_from_depth(uv, textureLoad(scene_depth, pixel, 0).r, frame.inv_view_proj);
        hit = resolve_displaced(pixel, instance, object_positions, normals, uvs, world_positions, depth_hit);
        b.lambda = hit.lambda;
    }
    var input: SurfaceInput;
    input.object_position = interpolate3(object_positions, b.lambda);
    // The wind offset varies smoothly: evaluated at the pixel, it matches the interpolated corners
    // to far below a pixel, which is all motion vectors need. Skinning is affine per triangle: the
    // corners' previous positions interpolate exactly. (Resolved here, before the material, so
    // the corner arrays need not live through it.)
    let plant = interpolate2(plants, b.lambda);
    var previous_world = wind_previous_world_position(instance, wind_force, input.object_position, plant.x);
    if skinned {
        previous_world = interpolate3(previous_worlds, b.lambda);
    }
    if planted {
        // The pixel's point on its triangle's (first corner's) bone a frame ago.
        let before = plant_pose(instance.plant.y, plant_air_previous(instance, wind_force), bones[0], input.object_position, vec3<f32>(0.0));
        previous_world = (instance.prev_world * vec4<f32>(before.position, 1.0)).xyz;
    }
    if displaced {
        // The hit's own motion: where its point of the surface was a frame ago (an animated
        // displacement moves it). The displaced point is a hair off the micro-triangle the depth
        // came from, which must not read as motion (TSR and motion blur would smear a still
        // surface).
        previous_world = wind_previous_world_position(instance, wind_force, hit.previous, 1.0) + (hit.world - wind_world_position(instance, wind_force, hit.object, 1.0));
    }
    input.world_position = interpolate3(world_positions, b.lambda);
    // Where the pixel's surface is.
    let surface_position = select(input.world_position, hit.world, displaced);
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
    var normal = select(normalize(normal_matrix * interpolate3(normals, b.lambda)), hit.normal, displaced);
    let eye = frame.camera_position.xyz;
    let to_eye = normalize(eye - surface_position);
    let face = cross(world_positions[1] - world_positions[0], world_positions[2] - world_positions[0]);
    // (Displaced surfaces are single-sided: the tessellator culls back-facing micro-triangles, and
    // a displaced surface may face the viewer where its base triangle does not.)
    input.back_facing = !displaced && dot(face, to_eye) < 0.0;
    if input.back_facing {
        normal = -normal; // two-sided geometry (leaves) seen from behind
    }
    // Guard against interpolated normals facing away from the viewer at silhouettes.
    normal = normalize(normal + to_eye * max(0.0, -dot(normal, to_eye) + 0.05));
    input.normal = normal;

    // Per-triangle tangent frame from positions and the UVs of the material's texture (no stored
    // tangents needed; a rotated or mirrored texture transform turns the frame with the image).
    var m = materials[instance.material];
    let vertex_color = interpolate4(colors, b.lambda);
    m.base_color *= vertex_color;
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
    if EXTENDED_RESOLVE {
        surface = resolve_extended(pixel, m, input, instance.material, vertex_color);
    } else {
        surface = evaluate_kind(m, input);
    }

    // A plant's baked occlusion darkens its indirect light like the material's ambient occlusion.
    surface.occlusion *= plant.y;

    if DEBUG_VIEWS {
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
            case DEBUG_MATERIALS: { surface.albedo = hash_color(RESOLVE_SLOT * 31u + 7u); }
            // Movable instances in orange, the rest grey (dynamic_debug.wgsl adds the invalidation).
            case DEBUG_INVALIDATION: { surface.albedo = select(vec3<f32>(0.2 + 0.6 * luminance(surface.albedo)), vec3<f32>(1.0, 0.45, 0.08), (instance.flags & INSTANCE_MOVABLE) != 0u); }
            case DEBUG_SKIN_WEIGHTS: { surface.albedo = skin_weight_color(instance, cluster, base, corners, b.lambda); }
            // Triangles grey (voxel_resolve.wgsl colours voxel clusters).
            case DEBUG_VOXELS: { surface.albedo = vec3<f32>(0.3 + 0.1 * hash_color(entry.y).x); }
            case DEBUG_ASSEMBLY_PARTS: { surface.albedo = assembly_part_color(instance); }
            // Tessellated surfaces coloured by depth and micro-triangle, the rest grey.
            case DEBUG_TESSELLATION: {
                surface.albedo = vec3<f32>(0.25 + 0.3 * luminance(surface.albedo));
                if displaced {
                    surface.albedo = tessellation_debug_color(id, instance, b.lambda);
                }
            }
            default: {}
        }
    }

    let motion = clip_to_uv(frame.unjittered_view_proj * vec4<f32>(surface_position, 1.0))
        - clip_to_uv(frame.prev_unjittered_view_proj * vec4<f32>(previous_world, 1.0));

    // Albedo is stored with a sqrt curve to spend 8-bit precision on dark values.
    textureStore(gbuffer_albedo, pixel, vec4<f32>(sqrt(saturate(surface.albedo)), surface.metallic));
    textureStore(gbuffer_normal, pixel, vec4<f32>(surface.normal, surface.roughness));
    textureStore(gbuffer_motion, pixel, vec4<f32>(motion, surface.emissive, pack_surface_flags(surface.flags, surface.occlusion)));
    svt_write_feedback(pixel, size);
}
