// Skin palettes of skeletal meshes (skin.rs): the per-instance palette lookup and the culling
// bounds of posed instances, for passes that cull or invalidate without decoding geometry.
// Requires common.wgsl and a `skin_palette: texture_2d<f32>` at binding 66; skinning.wgsl adds
// the deformation of vertices.
//
// A skinned instance holds its palette's first texel + 1 in flags bits 12..32. From there (skin.rs):
// texel 0 the posed object-space bounding sphere, texel 1 (reach, largest reach so far, motion
// since the previous frame, -), then per joint 6 texels: the three rows of its current matrix and
// of the previous frame's.

const SKIN_TEXTURE_WIDTH: u32 = 1024u;  // texel t is at (t & 1023, t >> 10)
const SKIN_HEADER_TEXELS: u32 = 2u;
const SKIN_JOINT_TEXELS: u32 = 6u;
const SKIN_PREVIOUS: u32 = 3u;  // texel offset of the previous frame's rows
const INSTANCE_SKIN_SHIFT: u32 = 12u;

fn skin_texel(t: u32) -> vec4<f32> {
    return textureLoad(skin_palette, vec2<u32>(t & (SKIN_TEXTURE_WIDTH - 1u), t >> 10u), 0);
}

// The instance's palette (first texel + 1), 0 if it is not skinned.
fn instance_skin(instance: Instance) -> u32 {
    return instance.flags >> INSTANCE_SKIN_SHIFT;
}

// A world-space bounding sphere of an instance's geometry (rest pose) grown to hold it in the
// current pose: padded by the largest displacement of any vertex, or the posed sphere of the
// whole instance where that is smaller (both conservative, skin.rs). For every visibility test
// (frustum, HZB, shadow pages); LOD keeps the rest spheres.
fn skin_padded(instance: Instance, sphere: vec4<f32>) -> vec4<f32> {
    let palette = instance_skin(instance);
    if palette == 0u {
        return sphere;
    }
    let posed = transform_sphere(instance.world, skin_texel(palette - 1u), instance.max_scale);
    let padded = vec4<f32>(sphere.xyz, sphere.w + (skin_texel(palette).x * instance.max_scale));
    return select(padded, posed, posed.w < padded.w);
}

// Whether the instance's skin moved since the previous frame: a dynamic shadow caster
// (vsm_invalidate.wgsl), like an instance in the wind. Culling, page expansion and invalidation
// must agree on it.
fn skin_dynamic(instance: Instance) -> bool {
    let palette = instance_skin(instance);
    return palette != 0u && skin_texel(palette).z > 0.0;
}

// World-space bounds of a moving skin's displacement for shadow invalidation: x the largest
// reach so far (every pose a cached page may still show is within it), y the motion this frame.
fn skin_motion(instance: Instance) -> vec2<f32> {
    let palette = instance_skin(instance);
    if palette == 0u {
        return vec2<f32>(0.0);
    }
    return skin_texel(palette).yz * instance.max_scale;
}
