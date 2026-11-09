// Material classification: bins 8x8 screen tiles by the shading models they contain.
// tiles buffer: [0, 8) per-model counts, [8, 40) per-model dispatch args, [64 + kind * capacity)
// per-model tile lists (tile = x | y << 16).

#include "common.wgsl"

const MATERIAL_SKY: u32 = 7u;
const MATERIAL_KINDS: u32 = 8u;
const TILE_LISTS: u32 = 64u;

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

fn pixel_kind(pixel: vec2<u32>) -> u32 {
    let id = textureLoad(visibility_buffer, pixel, 0).r;
    if id == INVALID_ID {
        return MATERIAL_SKY;
    }
    let entry = visible_clusters[id >> TRIANGLE_BITS];
    return min(materials[instances[entry.x].material].kind, MATERIAL_KINDS - 1u);
}

// ---- Classification -------------------------------------------------------------------------------

var<workgroup> tile_kinds: atomic<u32>;

@compute @workgroup_size(8, 8)
fn classify(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if all(lid.xy == vec2<u32>(0u)) {
        atomicStore(&tile_kinds, 0u);
    }
    workgroupBarrier();
    let size = vec2<u32>(frame.viewport.xy);
    let pixel = wid.xy * 8u + lid.xy;
    if all(pixel < size) {
        atomicOr(&tile_kinds, 1u << pixel_kind(pixel));
    }
    workgroupBarrier();
    if all(lid.xy == vec2<u32>(0u)) {
        var kinds = atomicLoad(&tile_kinds);
        let tile = wid.x | (wid.y << 16u);
        while kinds != 0u {
            let kind = firstTrailingBit(kinds);
            kinds &= kinds - 1u;
            let slot = atomicAdd(&tiles[kind], 1u);
            atomicStore(&tiles[TILE_LISTS + kind * tile_capacity() + slot], tile);
        }
    }
}

@compute @workgroup_size(8)
fn prepare_tiles(@builtin(local_invocation_index) kind: u32) {
    let count = atomicLoad(&tiles[kind]);
    atomicStore(&tiles[8u + kind * 4u], min(count, 65535u));
    atomicStore(&tiles[9u + kind * 4u], (count + 65534u) / 65535u);
    atomicStore(&tiles[10u + kind * 4u], 1u);
}

