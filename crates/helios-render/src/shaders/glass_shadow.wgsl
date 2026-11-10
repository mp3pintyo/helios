// Sunlight through glass (glass_shadow.rs): the glass panes seen from the sun, an orthographic
// map along the sunlight of the transmittance (rgb) and depth (a) of the pane nearest the sun.
// Where a point lies behind that pane, the sun reaching it is filtered by the pane's colour:
// stained glass throws coloured light onto the floor, the piers and the dusty air.
//
// Texels without glass hold transmittance 1 and depth 0, so being "behind" them changes
// nothing and bilinear filtering across a pane's edge blends towards white light.

struct GlassShadow {
    // World -> map: x, y in clip space (orthographic along the sunlight), z the depth along it,
    // 0..1 over the panes.
    view_proj: mat4x4<f32>,
    // x: 1 while the map holds panes, y: depth bias (map depth units), z: map depth units per
    // metre, w: the map's width in metres.
    params: vec4<f32>,
}

@group(0) @binding(90) var<uniform> glass_shadow: GlassShadow;
@group(0) @binding(91) var glass_shadow_map: texture_2d<f32>;

// Spread of the projected light, radians: the sun's disc and a little of the hammered glass's
// scattering. The light through a pane 10 m up lands blurred by ~10 cm.
const GLASS_SHADOW_SPREAD: f32 = 0.012;
const GLASS_SHADOW_TAPS: u32 = 6u;

// Map coordinates (uv, depth) of `world`.
fn glass_shadow_coords(world: vec3<f32>) -> vec3<f32> {
    let c = glass_shadow.view_proj * vec4<f32>(world, 1.0);
    return vec3<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5, c.z);
}

fn glass_shadow_inside(uv: vec2<f32>) -> bool {
    return all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
}

// One filtered tap: the transmittance of the pane at `uv` if `depth` lies behind it.
fn glass_shadow_tap(uv: vec2<f32>, depth: f32, s: sampler) -> vec3<f32> {
    let t = textureSampleLevel(glass_shadow_map, s, uv, 0.0);
    return select(vec3<f32>(1.0), t.rgb, depth > t.a + glass_shadow.params.y);
}

// Fraction of the sunlight (per channel) reaching `world` through glass: one tap (froxels,
// surface-cache texels).
fn glass_transmission(world: vec3<f32>, s: sampler) -> vec3<f32> {
    if glass_shadow.params.x == 0.0 {
        return vec3<f32>(1.0);
    }
    let c = glass_shadow_coords(world);
    if !glass_shadow_inside(c.xy) {
        return vec3<f32>(1.0);
    }
    return glass_shadow_tap(c.xy, c.z, s);
}

// As `glass_transmission`, blurred by the distance behind the pane (rotated by `noise` per
// pixel and frame; TSR resolves it).
fn glass_transmission_soft(world: vec3<f32>, s: sampler, noise: f32) -> vec3<f32> {
    if glass_shadow.params.x == 0.0 {
        return vec3<f32>(1.0);
    }
    let c = glass_shadow_coords(world);
    if !glass_shadow_inside(c.xy) {
        return vec3<f32>(1.0);
    }
    let center = textureSampleLevel(glass_shadow_map, s, c.xy, 0.0);
    // Metres behind the pane (texels without glass: the far end of the map).
    let behind = max(c.z - center.a, 0.0) / glass_shadow.params.z;
    let texel = 1.0 / f32(textureDimensions(glass_shadow_map).x);
    let radius = clamp(behind * GLASS_SHADOW_SPREAD / glass_shadow.params.w, texel, 8.0 * texel);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < GLASS_SHADOW_TAPS; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(GLASS_SHADOW_TAPS));
        let theta = f32(i) * 2.4 + noise * 6.2831853;
        sum += glass_shadow_tap(c.xy + r * radius * vec2<f32>(cos(theta), sin(theta)), c.z, s);
    }
    return sum / f32(GLASS_SHADOW_TAPS);
}
