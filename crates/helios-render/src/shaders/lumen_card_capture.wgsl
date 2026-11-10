// Lumen mesh cards: material capture (cards.rs; UE5's card capture into the surface cache's
// albedo and emissive atlases).
//
//   capture_albedo : evaluates the material of every texel of the scheduled cards (one
//                    workgroup per 8x8 tile) into the card albedo atlas, which light_cards
//                    multiplies the texels' lighting by. The material is evaluated as the G-buffer
//                    resolve evaluates it (material.wgsl's shading models, a material graph's
//                    generated `graph_eval`), from what the CPU capture recorded per texel: the
//                    surface point (card depth + normal), its texture coordinates and, for a
//                    displaced variant, the base point it was displaced from - the resolve, too,
//                    evaluates a displaced surface's material at its base point.
//
// One pipeline per shading model (MATERIAL_KIND) and per material graph (the graph's surface
// code in place of material_graph_off.wgsl). Image layers are read from the 512² texture arrays
// (svt_off.wgsl), filtered for the card texel's footprint.
//
// Tile records: x: the card record (vec4 index in card_data), y: the card instance whose
// placement and material are captured, z: the card's first texel in `attributes`,
// w: tile x | tile y << 8 | card direction << 16 (x = 0xffffffff: padding).
// Albedo texels: rgb = sqrt(albedo), a = emission e / (1 + e), e = the emitted radiance as a
// multiple of the albedo (the G-buffer's emissive channel).
// Each pair's captured texels are also summed (albedo rgb and the emission byte, 8 bits each, and
// the texel count; PAIR_SUM_WORDS per pair): the global distance field gives its voxels the
// pair's mean (lumen_sdf.wgsl), so the voxel surface cache reflects the captured colours too.

#include "common.wgsl"
#include "noise.wgsl"
#include "material.wgsl"
#include "svt_off.wgsl"
#include "substrate.wgsl"
#include "material_graph.wgsl"
#include "material_graph_off.wgsl"

override MATERIAL_KIND: u32 = 0u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(2) var material_albedo: texture_2d_array<f32>;
@group(0) @binding(3) var material_data: texture_2d_array<f32>;
@group(0) @binding(4) var material_sampler: sampler;
@group(0) @binding(5) var<storage, read> card_data: array<vec4<u32>>;
@group(0) @binding(6) var card_capture: texture_2d<u32>;
// Per card texel: uv (2 x f32), the base point's offset from the texel's surface point (mesh
// units, 3 x f16) and the texture coordinate change across the texel (f16).
@group(0) @binding(7) var<storage, read> attributes: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> tiles: array<vec4<u32>>;
@group(0) @binding(9) var albedo_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var<storage, read_write> pair_sums: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> slabs: array<MaterialSlab, MAX_MATERIALS>;
@group(1) @binding(1) var<uniform> graph_params: array<vec4<f32>, GRAPH_PARAMETER_SLOTS>;

const NONE: u32 = 0xffffffffu;
const INSTANCE_VEC4S: u32 = 7u;
const CARD_SLOTS: u32 = 18u;
const PAIR_SUM_WORDS: u32 = 5u;

// The workgroup's sums, added to the pair's once.
var<workgroup> tile_sums: array<atomic<u32>, 5>;
// A sloped surface's depth change per texel is clamped to this many texels (cards hold surfaces
// within ~73 degrees of their direction: at most ~3.2).
const MAX_SLOPE: f32 = 4.0;

fn unpack16(v: u32) -> vec2<u32> {
    return vec2<u32>(v & 0xffffu, v >> 16u);
}

fn rows_point(rows: array<vec4<f32>, 3>, p: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(rows[0], p), dot(rows[1], p), dot(rows[2], p));
}

// Octahedral normal of a captured texel (as card_texel_normal in lumen_cards.wgsl).
fn texel_normal(packed: u32) -> vec3<f32> {
    let e = vec2<f32>(f32(packed & 0xffu), f32((packed >> 8u) & 0xffu)) / 127.0 - 1.0;
    let z = 1.0 - abs(e.x) - abs(e.y);
    let t = max(-z, 0.0);
    return normalize(vec3<f32>(e.x + select(t, -t, e.x >= 0.0), e.y + select(t, -t, e.y >= 0.0), z));
}

// The pipeline's shading model (the G-buffer resolve's evaluate_kind; terrain has no cards'
// albedo, glass evaluates its tint and lead came).
fn evaluate_kind(m: Material, input: SurfaceInput) -> Surface {
    switch MATERIAL_KIND {
        case MATERIAL_ROCK: { return evaluate_rock(m, input); }
        case MATERIAL_METAL: { return evaluate_metal(m, input); }
        case MATERIAL_FOLIAGE: { return evaluate_foliage(m, input); }
        case MATERIAL_GLASS: { return evaluate_glass(m, input); }
        default: { return evaluate_standard(m, input); }
    }
}

@compute @workgroup_size(8, 8)
fn capture_albedo(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let tile = tiles[wid.x];
    if li < PAIR_SUM_WORDS {
        atomicStore(&tile_sums[li], 0u);
    }
    workgroupBarrier();
    let value = capture_texel(tile, lid.xy);
    if value.a >= 0.0 {
        let bytes = vec4<u32>(round(saturate(value) * 255.0));
        atomicAdd(&tile_sums[0], bytes.r);
        atomicAdd(&tile_sums[1], bytes.g);
        atomicAdd(&tile_sums[2], bytes.b);
        atomicAdd(&tile_sums[3], bytes.a);
        atomicAdd(&tile_sums[4], 1u);
    }
    workgroupBarrier();
    if li < PAIR_SUM_WORDS && tile.x != NONE {
        // The capturing instance's records are its pair's.
        let pair = card_data[tile.y * INSTANCE_VEC4S + 6u].x / CARD_SLOTS;
        let sum = atomicLoad(&tile_sums[li]);
        if sum > 0u {
            atomicAdd(&pair_sums[pair * PAIR_SUM_WORDS + li], sum);
        }
    }
}

// Captures one texel of a tile; returns its linear albedo and emission byte (a < 0: no surface).
fn capture_texel(tile: vec4<u32>, lid: vec2<u32>) -> vec4<f32> {
    if tile.x == NONE {
        return vec4<f32>(-1.0);
    }
    let card_frame = bitcast<vec4<f32>>(card_data[tile.x]);
    let card_depth = bitcast<vec4<f32>>(card_data[tile.x + 1u]);
    let card_texels = card_data[tile.x + 2u];
    let resolution = unpack16(card_texels.x);
    let texel = vec2<u32>(tile.w & 0xffu, (tile.w >> 8u) & 0xffu) * 8u + lid;
    if any(texel >= resolution) || card_texels.z == NONE {
        return vec4<f32>(-1.0);
    }
    let dst = unpack16(card_texels.z) + texel;
    let captured = textureLoad(card_capture, unpack16(card_texels.y) + texel, 0).r;
    if (captured & 0xffffu) == 0xffffu {
        textureStore(albedo_out, dst, vec4<f32>(0.0));
        return vec4<f32>(-1.0);
    }

    // The texel's surface point and normal in mesh space (card_texel_position).
    let direction = tile.w >> 16u;
    let c = direction / 2u;
    let cu = (c + 1u) % 3u;
    let cv = (c + 2u) % 3u;
    let sign = select(1.0, -1.0, (direction & 1u) != 0u);
    let size = card_frame.z;
    let uv_card = card_frame.xy + (vec2<f32>(texel) + 0.5) * size;
    var q = vec3<f32>(0.0);
    q[cu] = uv_card.x;
    q[cv] = uv_card.y;
    q[c] = card_frame.w - sign * (card_depth.x + f32(captured & 0xffffu) * card_depth.y);
    let object_normal = texel_normal(captured >> 16u);

    let a = attributes[tile.z + texel.y * resolution.x + texel.x];
    let offset_xy = unpack2x16float(a.z);
    let offset_z = unpack2x16float(a.w);
    let base = q + vec3<f32>(offset_xy, offset_z.x);
    let uv_footprint = offset_z.y;

    // The capturing instance: placement and material.
    let ib = tile.y * INSTANCE_VEC4S;
    var to_local: array<vec4<f32>, 3>;
    var to_world: array<vec4<f32>, 3>;
    for (var i = 0u; i < 3u; i++) {
        to_local[i] = bitcast<vec4<f32>>(card_data[ib + i]);
        to_world[i] = bitcast<vec4<f32>>(card_data[ib + 3u + i]);
    }
    let info = card_data[ib + 6u];
    let material = info.y;
    let max_scale = bitcast<f32>(info.w);
    // The resolve's normal matrix: the inverse transpose of the linear part, times the scale.
    let normal_matrix = mat3x3<f32>(to_local[0].xyz, to_local[1].xyz, to_local[2].xyz) * max_scale;
    let normal = normalize(normal_matrix * object_normal + vec3<f32>(0.0, 1e-7, 0.0));

    // "Screen" derivatives: one texel along the card's u and v, on the surface's tangent plane.
    var du = vec3<f32>(0.0);
    var dv = vec3<f32>(0.0);
    let nc = max(abs(object_normal[c]), 1e-3) * select(-1.0, 1.0, object_normal[c] >= 0.0);
    du[cu] = size;
    du[c] = -clamp(object_normal[cu] / nc, -MAX_SLOPE, MAX_SLOPE) * size;
    dv[cv] = size;
    dv[c] = -clamp(object_normal[cv] / nc, -MAX_SLOPE, MAX_SLOPE) * size;

    var input: SurfaceInput;
    input.object_position = base;
    input.world_position = rows_point(to_world, vec4<f32>(base, 1.0));
    input.normal = normal;
    let frame_ws = basis(normal);
    input.tangent = frame_ws[0];
    input.bitangent = frame_ws[1];
    input.uv = bitcast<vec2<f32>>(a.xy);
    input.duv_dx = vec2<f32>(uv_footprint, 0.0);
    input.duv_dy = vec2<f32>(0.0, uv_footprint);
    // (The second UV set is not captured: set 0 stands in for it.)
    input.uv1 = input.uv;
    input.duv1_dx = input.duv_dx;
    input.duv1_dy = input.duv_dy;
    input.object_dpdx = du;
    input.object_dpdy = dv;
    input.dpdx = rows_point(to_world, vec4<f32>(du, 0.0));
    input.dpdy = rows_point(to_world, vec4<f32>(dv, 0.0));
    input.footprint = max(length(du), length(dv));
    input.normal_matrix = normal_matrix;
    input.back_facing = false;

    let m = materials[material];
    var s: Surface;
    if HAS_GRAPH {
        let slab = slabs[material];
        let g = graph_eval(m, input, slab.params, vec4<f32>(1.0), graph_defaults(m, slab_none(), slab.clearcoat_roughness));
        s = graph_surface(g, m, input);
    } else {
        s = evaluate_kind(m, input);
    }
    // Glass keeps its coverage in `emissive` (translucency.wgsl): it does not glow.
    let e = select(max(s.emissive, 0.0), 0.0, MATERIAL_KIND == MATERIAL_GLASS);
    let albedo = select(s.albedo, vec3<f32>(0.0), any(s.albedo != s.albedo));
    let emission = select(e / (1.0 + e), 0.0, e != e);
    textureStore(albedo_out, dst, vec4<f32>(sqrt(saturate(albedo)), emission));
    return vec4<f32>(saturate(albedo), emission);
}
