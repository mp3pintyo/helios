// `sample_layer` for material.wgsl in passes without the streaming virtual texture bindings
// (svt.wgsl): every layer is sampled from the texture arrays.

fn sample_layer(layer: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> LayerTexels {
    return sample_layer_array(layer, uv, ddx, ddy);
}
