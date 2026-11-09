// G-buffer resolve of voxel pixels (included by gbuffer.wgsl, whose bindings, material
// evaluation and MATERIAL_KIND specialisation it uses; requires voxel.wgsl).
//
// The visibility buffer names the voxel's cluster slot and brick, the depth buffer where the
// pixel ray hit its cube: the voxel is the one of the brick whose (wind-displaced) cube that
// point lies on, found among the cells around it. The shading normal is the voxel's (the
// orientation of its leaves blended with the canopy's outward direction, facing the viewer like
// a two-sided leaf), the texture coordinates the brick's mean with its spread as the sampling
// footprint, so the material's colour is the average of the leaves it stands for.

@group(0) @binding(91) var scene_depth: texture_2d<f32>;

// Debug view 23: a colour per assembly part (by its persistent visibility bits, which stay with
// the part whichever instance slot it takes), assembly bases light grey, the rest dark grey.
fn assembly_part_color(instance: Instance) -> vec3<f32> {
    if (instance.flags & INSTANCE_PART) != 0u {
        return hash_color(instance.visibility_offset);
    }
    return vec3<f32>(select(0.2, 0.55, (instance.flags & INSTANCE_ASSEMBLY) != 0u));
}

fn resolve_voxel(pixel: vec2<u32>, id: u32) {
    let slot = id >> TRIANGLE_BITS;
    let b = id & TRIANGLE_MASK;
    let entry = visible_clusters[slot];
    let instance = instances[entry.x];
    let cluster = clusters[entry.y];
    let v = voxel_cluster(cluster);
    let brick = voxel_brick(v, b);
    let size = frame.viewport.xy;
    let normal_matrix = mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz);
    let to_object = transpose(normal_matrix);
    // The normal matrix is the inverse transpose scaled by max_scale, so its transpose is the
    // inverse times max_scale.
    let inverse_scale = 1.0 / max(instance.max_scale, 1e-6);
    let strength = wind_strength(instance);
    let eye = frame.camera_position.xyz;
    let uv = (vec2<f32>(pixel) + 0.5) * frame.viewport.zw;
    let world_position = world_from_depth(uv, textureLoad(scene_depth, pixel, 0).r, frame.inv_view_proj);

    // The hit in voxels from the cluster origin, the wind moved back by its offset there; the
    // voxel is the nearest set cell around it (Chebyshev distance to its displaced centre: ½ on
    // its own cube).
    let rest_estimate = world_position - (wind_world_position(instance, strength, (to_object * (world_position - instance.world[3].xyz)) * inverse_scale) - (instance.world * vec4<f32>((to_object * (world_position - instance.world[3].xyz)) * inverse_scale, 1.0)).xyz);
    let estimate = (((to_object * (rest_estimate - instance.world[3].xyz)) * inverse_scale) - v.origin) / v.size;
    let brick_cell = voxel_cell(brick, 0u);
    let mask = vec2<u32>(brick.x, brick.y);
    var best = 0u;
    var best_center = vec3<f32>(0.0);
    var best_distance = 3.0e38;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let local = vec3<i32>(floor(estimate - brick_cell)) + vec3<i32>(dx, dy, dz);
                if any(local < vec3<i32>(0)) || any(local > vec3<i32>(3)) {
                    continue;
                }
                let bit = u32(local.x + (4 * local.y) + (16 * local.z));
                let word = select(mask.x, mask.y, bit >= 32u);
                if ((word >> (bit & 31u)) & 1u) == 0u {
                    continue;
                }
                let object = v.origin + ((brick_cell + vec3<f32>(local) + 0.5) * v.size);
                let offset = wind_world_position(instance, strength, object) - (instance.world * vec4<f32>(object, 1.0)).xyz;
                let d = abs((((to_object * (world_position - offset - instance.world[3].xyz)) * inverse_scale) - object) / v.size);
                let distance = max(d.x, max(d.y, d.z));
                if distance < best_distance {
                    best_distance = distance;
                    best_center = object;
                    // Voxels are stored in bit order: those below it in the brick come first.
                    let below = select(countOneBits(mask.x & ((1u << bit) - 1u)), countOneBits(mask.x) + countOneBits(mask.y & ((1u << (bit - 32u)) - 1u)), bit >= 32u);
                    best = voxel_brick_first(brick) + below;
                }
            }
        }
    }
    let attributes = geometry[v.attributes + best];
    let center_world = wind_world_position(instance, strength, best_center);
    // The cube moves rigidly with the wind at its centre.
    let rest = world_position - (center_world - (instance.world * vec4<f32>(best_center, 1.0)).xyz);
    let object_position = (to_object * (rest - instance.world[3].xyz)) * inverse_scale;

    var input: SurfaceInput;
    input.object_position = object_position;
    input.world_position = world_position;
    // Per-pixel position derivatives on the plane facing the viewer at the hit.
    let hit_depth = textureLoad(scene_depth, pixel, 0).r;
    input.dpdx = (world_from_depth(uv + vec2<f32>(frame.viewport.z, 0.0), hit_depth, frame.inv_view_proj) - world_position) * frame.upscale.x;
    input.dpdy = (world_from_depth(uv + vec2<f32>(0.0, frame.viewport.w), hit_depth, frame.inv_view_proj) - world_position) * frame.upscale.y;
    input.object_dpdx = (to_object * input.dpdx) * inverse_scale;
    input.object_dpdy = (to_object * input.dpdy) * inverse_scale;
    input.footprint = max(length(input.dpdx), length(input.dpdy)) / max(instance.max_scale, 1e-4);
    let texture = voxel_brick_uv(v, brick);
    input.uv = texture.xy;
    input.duv_dx = vec2<f32>(texture.z, 0.0);
    input.duv_dy = vec2<f32>(0.0, texture.z);
    input.uv1 = input.uv;
    input.duv1_dx = input.duv_dx;
    input.duv1_dy = input.duv_dy;
    input.normal_matrix = normal_matrix;
    let to_eye = normalize(eye - world_position);
    var normal = normalize(normal_matrix * voxel_normal(attributes));
    // Like a two-sided leaf: the side facing the viewer.
    input.back_facing = dot(normal, to_eye) < 0.0;
    if input.back_facing {
        normal = -normal;
    }
    input.normal = normal;
    let frame_basis = basis(normal);
    input.tangent = frame_basis[0];
    input.bitangent = frame_basis[1];

    let m = materials[instance.material];
    var surface: Surface;
    switch MATERIAL_KIND {
        case MATERIAL_ROCK: { surface = evaluate_rock(m, input); }
        case MATERIAL_TERRAIN: { surface = evaluate_terrain(m, input); }
        case MATERIAL_METAL: { surface = evaluate_metal(m, input); }
        case MATERIAL_FOLIAGE: { surface = evaluate_foliage(m, input); }
        default: { surface = evaluate_standard(m, input); }
    }
    switch frame.debug_view {
        case DEBUG_CLUSTERS, DEBUG_VOXELS: { surface.albedo = hash_color(entry.y * 7919u + entry.x * 104729u); }
        case DEBUG_TRIANGLES: { surface.albedo = hash_color(entry.y * 8192u + best + entry.x * 15485863u); }
        case DEBUG_LOD: { surface.albedo = lod_color(cluster_lod_level(cluster)); }
        case DEBUG_INSTANCES: { surface.albedo = hash_color(entry.x); }
        // Voxels in yellow (red: software triangles, green: hardware, blue: alpha-masked).
        case DEBUG_RASTER_PATH: { surface.albedo = vec3<f32>(0.95, 0.8, 0.15) * (0.6 + 0.4 * hash_color(entry.y).x); }
        case DEBUG_MATERIALS: { surface.albedo = hash_color(MATERIAL_KIND * 31u + 7u); }
        case DEBUG_ASSEMBLY_PARTS: { surface.albedo = assembly_part_color(instance); }
        default: {}
    }

    let previous = wind_previous_world_position(instance, strength, best_center) + ((instance.prev_world * vec4<f32>(object_position - best_center, 0.0)).xyz);
    let motion = clip_to_uv(frame.unjittered_view_proj * vec4<f32>(world_position, 1.0)) - clip_to_uv(frame.prev_unjittered_view_proj * vec4<f32>(previous, 1.0));
    textureStore(gbuffer_albedo, pixel, vec4<f32>(sqrt(saturate(surface.albedo)), surface.metallic));
    textureStore(gbuffer_normal, pixel, vec4<f32>(surface.normal, surface.roughness));
    textureStore(gbuffer_motion, pixel, vec4<f32>(motion, surface.emissive, pack_surface_flags(surface.flags, surface.occlusion)));
    svt_write_feedback(pixel, vec2<u32>(size));
}
