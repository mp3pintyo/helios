// Cloud layer parameters shared by clouds.wgsl and the deferred lighting.

struct Clouds {
    coverage: f32,
    density: f32,        // extinction per km at full density
    base: f32,           // km above the ground
    thickness: f32,      // km
    wind: vec2<f32>,     // km travelled so far (x, z)
    shape_scale: f32,    // noise repeats per km
    detail_scale: f32,
    weather_scale: f32,
    shadow_strength: f32,
    shadow_extent: f32,  // metres covered by the shadow map
    enabled: u32,
    shadow_center: vec4<f32>, // xz: world centre of the shadow map (metres)
}
