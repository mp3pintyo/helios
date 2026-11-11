// Cluster vertices of a (cluster, page) shadow draw, shared by the hardware (vsm_draw.wgsl) and
// software (vsm_raster.wgsl) page rasterisers so both see bit-identical positions — wind and
// skinning included, as in the main view. Requires vsm_common.wgsl, vsm_local_common.wgsl,
// geometry.wgsl, wind.wgsl, skinning.wgsl and an `instances: array<Instance>` binding.
//
// draw = (instance, cluster, physical page | level << 16, page tag).

// The scene has skinned instances: without them the hardware page raster is compiled without the
// skinning code (vsm.rs), which otherwise costs its vertex shader ~5 % in the forest.
override SKINNING: bool = true;

// Position of a cluster vertex inside the draw's page (texels, y down) and its depth, in
// homogeneous form (x, y, depth, 1) * w: w = 1 on the sun's orthographic levels, the face
// depth on local lights' perspective pages (level >= LOCAL_LEVEL).
fn page_vertex(draw: vec4<u32>, cluster: Cluster, base: u32, local: u32) -> vec4<f32> {
    let instance = instances[draw.x];
    var object = vertex_position(cluster, base, local);
    if SKINNING {
        object = skinned_position(instance, cluster, base, local, object);
    }
    let strength = wind_strength(instance);
    let world = wind_world_position(instance, strength, object, vertex_wind_weight(cluster, base, local, strength));
    let level = draw.z >> 16u;
    let page = vec2<f32>(decode_tag(draw.w));
    if level >= LOCAL_LEVEL {
        return local_page_vertex(world, level - LOCAL_LEVEL, page);
    }
    let a = vsm_address(world, level);
    let ls = (vsm.light_view * vec4<f32>(world, 1.0)).xyz;
    let v = vec2<f32>(ls.x, -ls.y) / vsm.level_info[level].y; // absolute texel coordinates
    return vec4<f32>(v - page * f32(vsm.page_texels), a.depth, 1.0);
}
