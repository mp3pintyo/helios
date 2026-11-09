// Plain resolve pipelines (passes/material.rs): no Substrate slab and no material graph, so the
// extended branch of the resolve compiles away. The extended pipelines compose gbuffer.wgsl with
// resolve_extended.wgsl in place of this file.

const EXTENDED_RESOLVE: bool = false;

fn resolve_extended(pixel: vec2<u32>, m: Material, input: SurfaceInput, material: u32, vertex_color: vec4<f32>) -> Surface {
    return evaluate_kind(m, input);
}
