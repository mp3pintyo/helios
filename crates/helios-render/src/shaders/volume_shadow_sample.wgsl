// Shadows of heterogeneous volumes on what lies in or around them (volume_shadow.rs). Requires
// `lut_sampler` (linear clamp) and these bindings: the boxes (151) and the transmittance atlas
// (152; r: towards the sun, g: towards the box's driven light, per voxel to the box's boundary).

struct VolumeShadowBox {
    origin: vec4<f32>,    // xyz box minimum, w the driven light's index (bits; 0xffffffff none)
    size: vec4<f32>,      // xyz extent, w 1 while it casts shadows
    offset: vec4<f32>,    // xyz the region's first voxel in the atlas
    voxels: vec4<f32>,    // xyz the region's voxels
}

struct VolumeShadows {
    count: vec4<u32>,     // x boxes, y 1 if a casting box drives a light
    atlas: vec4<f32>,     // xyz the atlas' voxels
    boxes: array<VolumeShadowBox, 8>,
}

@group(0) @binding(151) var<uniform> volume_shadows: VolumeShadows;
@group(0) @binding(152) var volume_shadow_atlas: texture_3d<f32>;

// Entry and exit distances of the ray through box `k` (x > y: missed).
fn volume_shadow_span(k: u32, origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let b = volume_shadows.boxes[k];
    let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let a = (b.origin.xyz - origin) * inv;
    let c = (b.origin.xyz + b.size.xyz - origin) * inv;
    let near = min(a, c);
    let far = max(a, c);
    return vec2<f32>(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
}

// The atlas at world point `q` of box `k` (clamped into its region).
fn volume_shadow_at(k: u32, q: vec3<f32>) -> vec4<f32> {
    let b = volume_shadows.boxes[k];
    let local = clamp((q - b.origin.xyz) / b.size.xyz * b.voxels.xyz, vec3<f32>(0.5), b.voxels.xyz - 0.5);
    return textureSampleLevel(volume_shadow_atlas, lut_sampler, (b.offset.xyz + local) / volume_shadows.atlas.xyz, 0.0);
}

// The sun's transmittance at `p` through every casting volume but `skip`: each box the ray
// towards the sun crosses, at the point where it enters the box (or `p` inside it).
fn volume_sun_transmittance(p: vec3<f32>, skip: u32) -> f32 {
    var t = 1.0;
    let sun = normalize(frame.sun_direction.xyz);
    for (var k = 0u; k < volume_shadows.count.x; k++) {
        if k == skip || volume_shadows.boxes[k].size.w == 0.0 {
            continue;
        }
        let span = volume_shadow_span(k, p, sun);
        if span.x > span.y || span.y <= 0.0 {
            continue;
        }
        t *= volume_shadow_at(k, p + sun * max(span.x, 0.0)).r;
    }
    return t;
}

// The transmittance from light `index` at `light` to `p` through the volume whose fire drives
// that light: looked up where the segment from the light leaves its box towards `p` (or at `p`
// inside the box).
fn volume_light_transmittance(index: u32, light: vec3<f32>, p: vec3<f32>) -> f32 {
    if volume_shadows.count.y == 0u {
        return 1.0;
    }
    var t = 1.0;
    for (var k = 0u; k < volume_shadows.count.x; k++) {
        let b = volume_shadows.boxes[k];
        if bitcast<u32>(b.origin.w) != index || b.size.w == 0.0 {
            continue;
        }
        let to_p = p - light;
        let d = length(to_p);
        let dir = to_p / max(d, 1e-4);
        let span = volume_shadow_span(k, light, dir);
        if span.x > span.y || span.y <= 0.0 || span.x >= d {
            continue;
        }
        t *= volume_shadow_at(k, light + dir * min(span.y, d)).g;
    }
    return t;
}
