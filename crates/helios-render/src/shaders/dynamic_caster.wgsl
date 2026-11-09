// Dynamic shadow casters, for the passes that cull, expand and invalidate shadow pages. Requires
// common.wgsl and the bindings of wind.wgsl (`wind`) and skin_palette.wgsl (`skin_palette`).

#include "wind.wgsl"
#include "skin_palette.wgsl"

// Whether the scene has skinned instances. Culling is specialised on it (culling.rs): without
// them the palette lookups compile away, which matters where every cluster of every instance
// and assembly part is tested (they cost registers even when never taken).
override SKINNING: bool = true;

// Whether `instance`, with wind strength `strength` (wind_strength), is a dynamic shadow caster
// this frame: one the wind moves, a Mobility::Movable one or a skin whose pose moves
// (skin_dynamic). Cached shadow pages keep it out of their static layer and re-render it
// whenever it moved (vsm_invalidate.wgsl). Culling, page expansion and invalidation must agree
// on it.
fn dynamic_caster(instance: Instance, strength: f32) -> bool {
    return strength > 0.0 || (instance.flags & INSTANCE_MOVABLE) != 0u || (SKINNING && skin_dynamic(instance));
}

// `sphere` grown to hold a skinned instance in its current pose (skin_padded).
fn posed_bounds(instance: Instance, sphere: vec4<f32>) -> vec4<f32> {
    if SKINNING {
        return skin_padded(instance, sphere);
    }
    return sphere;
}
