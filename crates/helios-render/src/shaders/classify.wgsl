// Material classification: bins 8x8 screen tiles by the resolve pipelines their pixels need -
// the shading models (slots 0-7, 7: sky) and the extended pipelines of Substrate-slab and
// material-graph materials (slots 8-23, see passes/material.rs).
// tiles buffer: [0, 24) per-slot counts, [24, 120) per-slot dispatch args, [120 + slot * capacity)
// per-slot tile lists (tile = x | y << 16).

#include "common.wgsl"

const MATERIAL_SKY: u32 = 7u;
const RESOLVE_SLOTS: u32 = 24u;
const TILE_ARGS: u32 = 24u;
const TILE_LISTS: u32 = 120u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> materials: array<Material, MAX_MATERIALS>;
@group(0) @binding(2) var visibility_buffer: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> visible_clusters: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read_write> tiles: array<atomic<u32>>;

fn tile_capacity() -> u32 {
    let size = vec2<u32>(frame.viewport.xy);
    return ((size.x + 7u) / 8u) * ((size.y + 7u) / 8u);
}

fn pixel_slot(pixel: vec2<u32>) -> u32 {
    let id = textureLoad(visibility_buffer, pixel, 0).r;
    if id == INVALID_ID {
        return MATERIAL_SKY;
    }
    let entry = visible_clusters[id >> TRIANGLE_BITS];
    return min(materials[instances[entry.x].material].resolve, RESOLVE_SLOTS - 1u);
}

// ---- Classification -------------------------------------------------------------------------------

var<workgroup> tile_slots: atomic<u32>;

@compute @workgroup_size(8, 8)
fn classify(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if all(lid.xy == vec2<u32>(0u)) {
        atomicStore(&tile_slots, 0u);
    }
    workgroupBarrier();
    let size = vec2<u32>(frame.viewport.xy);
    let pixel = wid.xy * 8u + lid.xy;
    if all(pixel < size) {
        atomicOr(&tile_slots, 1u << pixel_slot(pixel));
    }
    workgroupBarrier();
    if all(lid.xy == vec2<u32>(0u)) {
        var slots = atomicLoad(&tile_slots);
        let tile = wid.x | (wid.y << 16u);
        while slots != 0u {
            let slot = firstTrailingBit(slots);
            slots &= slots - 1u;
            let index = atomicAdd(&tiles[slot], 1u);
            atomicStore(&tiles[TILE_LISTS + slot * tile_capacity() + index], tile);
        }
    }
}

@compute @workgroup_size(32)
fn prepare_tiles(@builtin(local_invocation_index) slot: u32) {
    if slot >= RESOLVE_SLOTS {
        return;
    }
    let count = atomicLoad(&tiles[slot]);
    atomicStore(&tiles[TILE_ARGS + slot * 4u], min(count, 65535u));
    atomicStore(&tiles[TILE_ARGS + 1u + slot * 4u], (count + 65534u) / 65535u);
    atomicStore(&tiles[TILE_ARGS + 2u + slot * 4u], 1u);
}
