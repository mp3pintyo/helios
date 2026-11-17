// The target of the compute rasterisers (sw_raster.wgsl, voxel_raster.wgsl, tess.wgsl) and how a
// fragment gets into it. This file is the two-pass variant for devices without 64-bit atomics
// (every browser): depth and payload cannot be written in one atomicMax, so every cluster is
// rasterised twice,
//   PASS 0 (depth)   : atomicMax of the reverse-Z depth bits per pixel
//   PASS 1 (payload) : pixels whose depth equals the stored winner write the visibility id
// Ties write identical-depth ids, which is harmless. sw_target_u64.wgsl replaces this file where
// the device has 64-bit atomics (shader.rs, `single_pass_raster`). The including shader declares
// the `PASS` override.

// [0, w*h): depth bits, [w*h, 2*w*h): payload.
@group(0) @binding(7) var<storage, read_write> sw_target: array<atomic<u32>>;

// Whether a fragment of depth `bits` would be written to pixel `i` in this pass: the cheap test
// before an alpha test.
fn sw_wanted(i: u32, bits: u32) -> bool {
    let stored = atomicLoad(&sw_target[i]);
    return select(bits == stored, bits > stored, PASS == 0u);
}

// Writes a wanted fragment: its depth in the depth pass, its id in the payload pass. `pixels`:
// the target's pixel count.
fn sw_write(i: u32, pixels: u32, bits: u32, id: u32) {
    if PASS == 0u {
        atomicMax(&sw_target[i], bits);
    } else {
        atomicStore(&sw_target[pixels + i], id);
    }
}

// Payload pass of a rasteriser that stores more than the id per pixel (tess.wgsl's barycentrics):
// whether the fragment won the pixel. One step of slack in the depth: that pass has code the
// depth pass lacks, and a compiler may round z differently next to it — a pixel that matched no
// triangle would keep a stale payload.
fn sw_won(i: u32, bits: u32, id: u32) -> bool {
    return atomicLoad(&sw_target[i]) - bits <= 1u;
}
