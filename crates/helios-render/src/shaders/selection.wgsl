// Editor selection outline, drawn over the tone-mapped output: pixels whose visibility-buffer
// instance is selected get a faint tint, pixels next to them an outline (as UE's editor
// selection outline). The visibility buffer is at the internal resolution; each output pixel
// reads the internal pixel under it, and the outline is found in output pixels so its width
// does not change with the screen percentage.

#include "common.wgsl"

override OUTPUT_SRGB: bool = true;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var visibility_buffer: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> visible_clusters: array<vec4<u32>>;
// One bit per scene instance.
@group(0) @binding(3) var<storage, read> selected: array<u32>;
@group(0) @binding(4) var<storage, read> part_owners: array<u32>;

#include "part_owner.wgsl"

struct VsOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

fn is_selected(output_pixel: vec2<f32>) -> bool {
    let output_size = frame.upscale.zw;
    let p = clamp(output_pixel, vec2<f32>(0.0), output_size - vec2<f32>(1.0));
    let internal = min(vec2<u32>((p + 0.5) * frame.upscale.xy), vec2<u32>(frame.viewport.xy) - vec2<u32>(1u));
    let id = textureLoad(visibility_buffer, internal, 0).r;
    if id == INVALID_ID {
        return false;
    }
    // Assembly parts are outlined with their assembly.
    let instance = owner_instance(visible_clusters[id >> TRIANGLE_BITS].x);
    let word = instance >> 5u;
    if word >= arrayLength(&selected) {
        return false;
    }
    return ((selected[word] >> (instance & 31u)) & 1u) != 0u;
}

fn to_output(c: vec3<f32>) -> vec3<f32> {
    // Colours are authored in sRGB; an sRGB target encodes them itself.
    if OUTPUT_SRGB {
        return pow(c, vec3<f32>(2.2));
    }
    return c;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pixel = floor(in.position.xy);
    let orange = to_output(vec3<f32>(1.0, 0.6, 0.12));
    if is_selected(pixel) {
        return vec4<f32>(orange, 0.12);
    }
    // Outline: a selected pixel up to 2 output pixels away. Eight taps (the axes at 2 px, the
    // diagonals at 1 px) cover that ring; each costs a visibility load and two buffer reads.
    var hits = 0u;
    let taps = array<vec2<f32>, 8>(
        vec2<f32>(2.0, 0.0), vec2<f32>(-2.0, 0.0), vec2<f32>(0.0, 2.0), vec2<f32>(0.0, -2.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, -1.0),
    );
    for (var i = 0u; i < 8u; i++) {
        if is_selected(pixel + taps[i]) {
            hits += 1u;
        }
    }
    if hits == 0u {
        discard;
    }
    return vec4<f32>(orange, min(1.0, 0.5 + f32(hits) * 0.2));
}
