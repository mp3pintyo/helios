// Lighting passes of scenes without Substrate slabs (the default): no slab texel to read, and
// every `if SUBSTRATE_LIGHTING` branch compiles away, so they cost what they did before slabs.
// Where some material has slab features the renderer composes the shader with
// substrate_read.wgsl in place of this file (shader::compose_variant).
// Requires substrate.wgsl.

const SUBSTRATE_LIGHTING: bool = false;

fn load_slab(pixel: vec2<u32>) -> Slab {
    return slab_none();
}

fn slab_constants(material: u32) -> MaterialSlab {
    var c: MaterialSlab;
    return c;
}
