// Skeletal meshes on virtual geometry: linear blend skinning (UE 5.5's Nanite skinned meshes; see
// skin.rs). Requires common.wgsl, geometry.wgsl and a `skin_palette: texture_2d<f32>` at
// binding 66.
//
// A vertex blends its joints' matrices (skin_palette.wgsl) by its quantised weights (the cluster's
// skin section, encoding.rs), skipping zero weights, in slot order: a pure function of the rest
// vertex, its weights and the palette. Every pass and every cluster or LOD level that shares a
// vertex therefore puts it on the same bits (no cracks), as wind.wgsl does for the wind.

#include "skin_palette.wgsl"

const CLUSTER_HAS_SKIN: u32 = 0x10000000u;

// A vertex's influences; unused slots have weight 0.
struct VertexSkin {
    joints: vec4<u32>,
    weights: vec4<f32>,
}

// Three rows of an affine matrix.
struct SkinMatrix {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

// Pool word of a cluster's skin section (its descriptor word), INVALID_ID if it has none. The
// section follows the vertex stream, word aligned.
fn cluster_skin_section(c: Cluster, base: u32) -> u32 {
    let header = base + c.vertex_offset;
    let widths = geometry[header + 5u];
    let extension = cluster_extension(header, widths);
    if (extension & CLUSTER_HAS_SKIN) == 0u {
        return INVALID_ID;
    }
    let bits = attribute_bit(c, widths, extension, cluster_vertex_count(c));
    return vertex_stream(header, widths) + ((bits + 31u) >> 5u);
}

// Influences of vertex `index` from the skin section at `section`: descriptor
// joint minimum | joint bits << 16 | weight bits << 21 | (influences - 1) << 25, then per vertex
// the joint offsets and the weights of influences 1.. (the first is 255 minus their sum).
fn vertex_skin(section: u32, index: u32) -> VertexSkin {
    let d = geometry[section];
    let joint_min = d & 0xffffu;
    let joint_bits = (d >> 16u) & 31u;
    let weight_bits = (d >> 21u) & 15u;
    let influences = ((d >> 25u) & 3u) + 1u;
    var bit = index * ((influences * joint_bits) + ((influences - 1u) * weight_bits));
    var s = VertexSkin(vec4<u32>(0u), vec4<f32>(0.0));
    for (var i = 0u; i < influences; i++) {
        s.joints[i] = joint_min + read_bits(section + 1u, bit, joint_bits);
        bit += joint_bits;
    }
    var rest = 0u;
    for (var i = 1u; i < influences; i++) {
        let w = read_bits(section + 1u, bit, weight_bits);
        s.weights[i] = f32(w) / 255.0;
        rest += w;
        bit += weight_bits;
    }
    s.weights[0] = f32(255u - rest) / 255.0;
    return s;
}

// Σ weight · joint matrix over the non-zero weights, in slot order (helios_geometry::skin::
// skin_matrix); `offset` 0 for this frame's palette, SKIN_PREVIOUS for the previous frame's.
fn skin_matrix(palette: u32, s: VertexSkin, offset: u32) -> SkinMatrix {
    var m = SkinMatrix(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var i = 0u; i < 4u; i++) {
        let w = s.weights[i];
        if w > 0.0 {
            let t = palette + SKIN_HEADER_TEXELS + (s.joints[i] * SKIN_JOINT_TEXELS) + offset;
            m.r0 += skin_texel(t) * w;
            m.r1 += skin_texel(t + 1u) * w;
            m.r2 += skin_texel(t + 2u) * w;
        }
    }
    return m;
}

fn skin_point(m: SkinMatrix, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(m.r0, h), dot(m.r1, h), dot(m.r2, h));
}

// Normals transform with the cofactor of the blended matrix (inverse transpose up to scale), so
// non-uniformly scaled joints keep them perpendicular.
fn skin_direction(m: SkinMatrix, n: vec3<f32>) -> vec3<f32> {
    let a = m.r0.xyz;
    let b = m.r1.xyz;
    let c = m.r2.xyz;
    let skinned = vec3<f32>(dot(cross(b, c), n), dot(cross(c, a), n), dot(cross(a, b), n));
    return select(n, normalize(skinned), dot(skinned, skinned) > 1e-20);
}

// Object-space position of cluster vertex `index` (decoded rest position `rest`) in its
// instance's current pose; `rest` itself for unskinned instances and clusters.
fn skinned_position(instance: Instance, c: Cluster, base: u32, index: u32, rest: vec3<f32>) -> vec3<f32> {
    let palette = instance_skin(instance);
    if palette == 0u {
        return rest;
    }
    let section = cluster_skin_section(c, base);
    if section == INVALID_ID {
        return rest;
    }
    return skin_point(skin_matrix(palette - 1u, vertex_skin(section, index), 0u), rest);
}

// vertex_position (geometry.wgsl), posed: what every rasteriser places.
fn skinned_vertex_position(instance: Instance, c: Cluster, base: u32, index: u32) -> vec3<f32> {
    return skinned_position(instance, c, base, index, vertex_position(c, base, index));
}

// Everything the G-buffer resolve needs of a skinned vertex.
struct SkinnedVertex {
    position: vec3<f32>,
    previous: vec3<f32>,  // the previous frame's position (motion vectors)
    normal: vec3<f32>,
}

fn skinned_vertex(instance: Instance, c: Cluster, base: u32, index: u32, rest: vec3<f32>, normal: vec3<f32>) -> SkinnedVertex {
    var v = SkinnedVertex(rest, rest, normal);
    let palette = instance_skin(instance);
    let section = select(INVALID_ID, cluster_skin_section(c, base), palette != 0u);
    if section == INVALID_ID {
        return v;
    }
    let skin = vertex_skin(section, index);
    let m = skin_matrix(palette - 1u, skin, 0u);
    v.position = skin_point(m, rest);
    v.normal = skin_direction(m, normal);
    v.previous = skin_point(skin_matrix(palette - 1u, skin, SKIN_PREVIOUS), rest);
    return v;
}
