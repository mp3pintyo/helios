// Photo mode (photo.rs): accumulates the frames of a still. Each frame TSR's photo permutation
// writes the output pixel's Gaussian reconstruction of this frame's jittered samples alone
// (colour, sample weight); summed with their weights over all frames they give the scene
// filtered at output resolution from every sample, through every lens position.

struct Photo {
    params: vec4<f32>,   // x: 1 for the first sample (start a new sum)
}

@group(0) @binding(0) var<uniform> photo: Photo;
@group(0) @binding(1) var color_in: texture_2d<f32>;
@group(0) @binding(2) var sum_in: texture_2d<f32>;
@group(0) @binding(3) var sum_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var mean_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(mean_out);
    if any(gid.xy >= size) {
        return;
    }
    let taken = textureLoad(color_in, gid.xy, 0);
    var sum = vec4<f32>(taken.rgb * taken.a, taken.a);
    if any(sum != sum) {
        sum = vec4<f32>(0.0);
    }
    if photo.params.x < 0.5 {
        sum += textureLoad(sum_in, gid.xy, 0);
    }
    textureStore(sum_out, gid.xy, sum);
    textureStore(mean_out, gid.xy, vec4<f32>(sum.rgb / max(sum.a, 1e-8), 1.0));
}
