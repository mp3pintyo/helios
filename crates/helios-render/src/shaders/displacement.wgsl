// Material displacement as Nanite tessellation sees it (tessellation.rs). Requires common.wgsl
// and a `tess: Tessellation` uniform at binding 67.
//
// A displaced instance (INSTANCE_DISPLACED) moves every point of its surface along the
// interpolated normal by magnitude * (height - centre) object units, or — a material graph's
// VectorDisplacement output — by a world-space vector of at most magnitude * instance scale.
// Displacement is applied at render time, so the bounds culling tests are padded by its bound, as
// for the wind (UE: the material's maximum displacement). A graph may read the time: its instances
// (INSTANCE_ANIMATED) are then dynamic shadow casters (dynamic_caster.wgsl, vsm_invalidate.wgsl),
// whose cached pages are refreshed by the displacement's speed bound where it has one.

struct TessMaterial {
    // x: magnitude (object units for heights 0..1 or a unit vector), y: centre (the height left in
    // place), z: bound of |displacement| (object units), w: mesh UV units per object unit (picks
    // the height's mip)
    params: vec4<f32>,
    // x: texture layer holding the height, y: its channel in the layer's data texel; z: the
    // material graph driving the height instead (graph index + 1, 0: none), w: its parameters'
    // offset in the graph parameter block
    source: vec4<u32>,
    // x: bound of the displacement's speed (object units per second; 0: static, negative: an
    // animated graph without a bound), yzw: unused
    motion: vec4<f32>,
}

struct Tessellation {
    // x: dice rate (pixels per micro-triangle edge), y: main view pixels per world unit at
    // distance 1 (internal resolution), z: main view near plane, w: unused
    dice: vec4<f32>,
    // xyz: main camera position (every view picks height mips from it)
    eye: vec4<f32>,
    // x: the time animated displacement is evaluated at (seconds), y: the previous frame's,
    // z: 1 to ignore the materials' speed bounds (RenderSettings::displacement_speed_bound off)
    time: vec4<f32>,
    materials: array<TessMaterial, MAX_MATERIALS>,
}

// A bounding sphere of a displaced instance's geometry grown by its displacement bound.
fn displacement_padded(instance: Instance, sphere: vec4<f32>) -> vec4<f32> {
    if (instance.flags & INSTANCE_DISPLACED) == 0u {
        return sphere;
    }
    return vec4<f32>(sphere.xyz, sphere.w + displacement_reach(instance));
}

// World-space bound of how far the displacement moves a point of a displaced instance.
fn displacement_reach(instance: Instance) -> f32 {
    return tess.materials[instance.material].params.z * instance.max_scale;
}

// World-space bound of how fast an animated displacement moves a point of a displaced instance
// (units per second), negative if its graph has none (helios-material's motion analysis).
fn displacement_speed(instance: Instance) -> f32 {
    let speed = tess.materials[instance.material].motion.x;
    return select(-1.0, speed * instance.max_scale, speed >= 0.0 && tess.time.z == 0.0);
}
