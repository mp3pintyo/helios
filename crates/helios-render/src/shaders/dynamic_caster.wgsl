// Dynamic shadow casters, for the passes that cull, expand and invalidate shadow pages. Requires
// common.wgsl and the bindings of wind.wgsl (`wind`) and skin_palette.wgsl (`skin_palette`).

#include "wind.wgsl"
#include "skin_palette.wgsl"

// Whether `instance`, with wind strength `strength` (wind_strength), is a dynamic shadow caster
// this frame: one the wind moves, a Mobility::Movable one or a skin whose pose moves
// (skin_dynamic). Cached shadow pages keep it out of their static layer and re-render it
// whenever it moved (vsm_invalidate.wgsl). Culling, page expansion and invalidation must agree
// on it.
fn dynamic_caster(instance: Instance, strength: f32) -> bool {
    return strength > 0.0 || (instance.flags & INSTANCE_MOVABLE) != 0u || skin_dynamic(instance);
}
