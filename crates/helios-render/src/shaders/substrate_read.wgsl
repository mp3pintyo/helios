// Lighting passes of scenes with Substrate slabs (substrate_gpu.rs): bind group 1 holds the
// per-material slab table and the slab texels of the G-buffer (written by the extended resolve
// pipelines, resolve_extended.wgsl). Replaces substrate_read_off.wgsl.
// Requires common.wgsl (MAX_MATERIALS) and substrate.wgsl.

const SUBSTRATE_LIGHTING: bool = true;

@group(1) @binding(0) var<uniform> slabs: array<MaterialSlab, MAX_MATERIALS>;
@group(1) @binding(1) var gbuffer_substrate: texture_2d<u32>;

fn load_slab(pixel: vec2<u32>) -> Slab {
    return decode_slab(textureLoad(gbuffer_substrate, pixel, 0));
}

fn slab_constants(material: u32) -> MaterialSlab {
    return slabs[material];
}
