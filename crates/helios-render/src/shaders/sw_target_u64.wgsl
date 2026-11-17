// sw_target.wgsl for devices with 64-bit atomics (native Vulkan and D3D12; shader.rs,
// `single_pass_raster`): depth and id are one value per pixel, depth in the high word, so one
// atomicMax writes both and every cluster is rasterised once (Nanite's own scheme). Of two
// fragments of equal depth the larger id wins, whatever the order of the threads.
//
// Only `PASS` 0 pipelines of sw_raster.wgsl and voxel_raster.wgsl exist on such a device.
// tess.wgsl keeps its second pass for the barycentrics it stores next to the id (`sw_won`).

// Per pixel: depth bits << 32 | payload (as u32 words: payload, then depth bits; visbuffer.wgsl).
@group(0) @binding(7) var<storage, read_write> sw_target: array<atomic<u64>>;

// Every fragment is written (the max sorts them out). Reading the depth first, to spare hidden
// fragments their alpha test as the two-pass variant does, measured slower than not reading it
// (forest 1.14 → 1.09 ms, the cathedral's alpha-masked screens 0.053 → 0.048 ms).
fn sw_wanted(i: u32, bits: u32) -> bool {
    return PASS == 0u;
}

// Writes a wanted fragment. `pixels`: the target's pixel count (the two-pass layout's offset).
fn sw_write(i: u32, pixels: u32, bits: u32, id: u32) {
    if PASS == 0u {
        atomicMax(&sw_target[i], (u64(bits) << 32u) | u64(id));
    }
}

// tess.wgsl's second pass: whether the fragment won the pixel (see sw_target.wgsl for the step of
// slack in the depth). The value is read with a max that changes nothing, not with atomicLoad:
// naga's HLSL loads a 64-bit atomic with the 32-bit InterlockedOr, which returns its low word
// only (the terrace's barycentrics were never written on D3D12).
fn sw_won(i: u32, bits: u32, id: u32) -> bool {
    let stored = atomicMax(&sw_target[i], 0lu);
    return (u32(stored >> 32u) - bits) + 1u <= 2u && u32(stored & 0xfffffffflu) == id;
}
