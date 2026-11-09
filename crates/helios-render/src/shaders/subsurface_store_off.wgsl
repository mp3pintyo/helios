// The deferred lighting without Substrate slabs: no subsurface light to separate. Replaced by
// subsurface_store.wgsl in the lighting pass's slab variant.

fn store_subsurface(pixel: vec2<u32>, light: vec3<f32>) {}
