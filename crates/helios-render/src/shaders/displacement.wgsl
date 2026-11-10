// Material displacement as Nanite tessellation sees it (tessellation.rs). Requires common.wgsl
// and a `tess: Tessellation` uniform at binding 67.
//
// A displaced instance (INSTANCE_DISPLACED) moves every point of its surface along the
// interpolated normal by magnitude * (height - centre) object units. Displacement is applied at
// render time, so the bounds culling tests are padded by its bound, as for the wind (UE: the
// material's maximum displacement).

struct TessMaterial {
    // x: magnitude (object units for heights 0..1), y: centre (the height left in place), z: bound
    // of |displacement| (object units), w: mesh UV units per object unit (picks the height's mip)
    params: vec4<f32>,
    // x: texture layer holding the height, y: its channel in the layer's data texel; z: the
    // material graph driving the height instead (graph index + 1, 0: none), w: its parameters'
    // offset in the graph parameter block
    source: vec4<u32>,
}

struct Tessellation {
    // x: dice rate (pixels per micro-triangle edge), y: main view pixels per world unit at
    // distance 1 (internal resolution), z: main view near plane, w: unused
    dice: vec4<f32>,
    // xyz: main camera position (every view picks height mips from it)
    eye: vec4<f32>,
    materials: array<TessMaterial, MAX_MATERIALS>,
}

// A bounding sphere of a displaced instance's geometry grown by its displacement bound.
fn displacement_padded(instance: Instance, sphere: vec4<f32>) -> vec4<f32> {
    if (instance.flags & INSTANCE_DISPLACED) == 0u {
        return sphere;
    }
    return vec4<f32>(sphere.xyz, sphere.w + tess.materials[instance.material].params.z * instance.max_scale);
}
