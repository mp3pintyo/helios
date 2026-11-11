// Local lights through glass (glass_light.rs): the glass panes seen from a few local lights near
// glass, one cube map each - per texel the transmittance (rgb) of the pane nearest the light and
// its distance from the light (a). A point behind its texel's pane gets the light filtered by the
// pane's colour: a chandelier's candles shine through a stained-glass screen in its colours.
//
// The hook is `glass_light_transmission` (four texels, each depth-compared: a 2x2 box filter);
// lights without a map (and every light while no map exists) pass unfiltered. Texels without
// glass hold 1 and distance 0.

struct GlassLights {
    // Per light index (up to 1024), the light's cube map layer + 1 (0: none), 8 bits each, 4 per
    // word, 4 words per vector.
    layers: array<vec4<u32>, 64>,
    // x: 1 while any light has a map, y: depth bias (metres).
    params: vec4<f32>,
}

@group(0) @binding(92) var<uniform> glass_lights: GlassLights;
@group(0) @binding(93) var glass_light_maps: texture_cube_array<f32>;
@group(0) @binding(94) var glass_light_sampler: sampler;

// The cube map layer of light `index`, or -1.
fn glass_light_layer(index: u32) -> i32 {
    if index >= 1024u {
        return -1;
    }
    let word = glass_lights.layers[index / 16u][(index / 4u) % 4u];
    return i32((word >> (8u * (index % 4u))) & 0xffu) - 1;
}

// Fraction (per channel) of local light `index`'s light (at `light_position`) reaching `p`
// through the glass between them.
fn glass_light_transmission(index: u32, light_position: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    if glass_lights.params.x == 0.0 {
        return vec3<f32>(1.0);
    }
    let layer = glass_light_layer(index);
    if layer < 0 {
        return vec3<f32>(1.0);
    }
    // The four texels around the direction, each compared on its own (a filtered distance would
    // blend a pane's edge with the empty texels beside it and colour what lies in front of the
    // pane), their transmittances averaged.
    let d = p - light_position;
    let distance = length(d) - glass_lights.params.y;
    let r = textureGather(0, glass_light_maps, glass_light_sampler, d, layer);
    let g = textureGather(1, glass_light_maps, glass_light_sampler, d, layer);
    let b = textureGather(2, glass_light_maps, glass_light_sampler, d, layer);
    let a = textureGather(3, glass_light_maps, glass_light_sampler, d, layer);
    let behind = (a > vec4<f32>(0.0)) & (vec4<f32>(distance) > a);
    let one = vec4<f32>(1.0);
    let t = vec3<f32>(dot(select(one, r, behind), one), dot(select(one, g, behind), one), dot(select(one, b, behind), one));
    return t * 0.25;
}
