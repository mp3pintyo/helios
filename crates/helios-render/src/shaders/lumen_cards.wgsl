// Lumen surface cache made of mesh cards (cards.rs; UE5 Lumen's card-based surface cache).
//
// Each instance with a volume distance field has up to 6 directions x 3 depth layers of cards:
// orthographic captures of its mesh's surfaces (depth + normal per texel, captured once per
// mesh). The instance's cards are lit per texel into pages of a lighting atlas, at a resolution
// (mip) that follows the instance's distance to the camera. A ray hit is shaded by projecting
// the hit point into the cards facing its normal: a card texel whose captured depth matches the
// hit holds the radiance leaving that very surface - not an average over a voxel that also
// covers the other side of a thin wall. Where no card matches (uncaptured surfaces, cards not
// allocated or not lit yet), callers keep the voxel surface cache.
//
// Includers declare, besides lumen_common.wgsl's bindings:
//   card_data: array<vec4<u32>>       instance records, then card records (see below)
//   card_pages: array<u32>            allocation word per instance card, then page tables
//   card_capture: texture_2d<u32>     captured texels of every mesh card (depth16 | normal16)
//   card_lighting: texture_2d<f32>    lit pages (rgb: exitant radiance, a: 0 without a surface, else
//                                     1 + the lit surface's stored depth / CARD_DEPTH_STEPS)
//   sdf_instance_ids: texture_3d<u32> closest instance per global-SDF voxel (2 x 16 bits)

const CARD_INSTANCE_VEC4S: u32 = 7u;
const CARD_RECORD_VEC4S: u32 = 3u;
const CARD_LAYERS: u32 = 3u;
const CARD_SLOTS: u32 = 18u; // directions x layers
const CARD_PAGE_TEXELS: u32 = 8u;
const CARD_NONE: u32 = 0xffffffffu;
const CARD_DEPTH_STEPS: f32 = 65534.0;

// Instance record: world -> mesh and mesh -> world (rows), then info.
struct CardInstance {
    to_local: array<vec4<f32>, 3>,
    to_world: array<vec4<f32>, 3>,
    // x: first card record (the 18 slots, direction-major, of its (surface, material) pair), y: material,
    // z: layer count per direction (2 bits each), w: world units per mesh unit (f32 bits).
    info: vec4<u32>,
}

// Card record (mesh space). Slot = direction * CARD_LAYERS + layer; direction 2c (+) or 2c + 1
// (-) along coordinate c, card (u, v) = coordinates (c + 1, c + 2) mod 3.
struct Card {
    // xy: (u, v) of texel (0, 0)'s corner, z: texel size, w: card plane along c.
    frame: vec4<f32>,
    // x: depth of stored 0, y: depth per stored step, z: the mesh SDF's voxel size.
    depth: vec4<f32>,
    // x: resolution (x | y << 16), y: texel (0, 0) in the capture atlas (x | y << 16), z: in the
    // albedo atlas (0xffffffff: the material is not captured; light_cards only).
    texels: vec4<u32>,
}

fn card_instance_load(index: u32) -> CardInstance {
    let base = index * CARD_INSTANCE_VEC4S;
    var c: CardInstance;
    for (var i = 0u; i < 3u; i++) {
        c.to_local[i] = bitcast<vec4<f32>>(card_data[base + i]);
        c.to_world[i] = bitcast<vec4<f32>>(card_data[base + 3u + i]);
    }
    c.info = card_data[base + 6u];
    return c;
}

fn card_load(index: u32) -> Card {
    let base = lumen.cards.y + index * CARD_RECORD_VEC4S;
    var c: Card;
    c.frame = bitcast<vec4<f32>>(card_data[base]);
    c.depth = bitcast<vec4<f32>>(card_data[base + 1u]);
    c.texels = card_data[base + 2u];
    return c;
}

fn card_unpack(v: u32) -> vec2<u32> {
    return vec2<u32>(v & 0xffffu, v >> 16u);
}

fn card_layer_count(info: vec4<u32>, direction: u32) -> u32 {
    return (info.z >> (direction * 2u)) & 3u;
}

// Texels of a card at lighting mip `mip`.
fn card_mip_resolution(resolution: vec2<u32>, mip: u32) -> vec2<u32> {
    return max((resolution + ((1u << mip) - 1u)) >> vec2<u32>(mip), vec2<u32>(1u));
}

fn card_point(p: vec3<f32>, rows: array<vec4<f32>, 3>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(rows[0], h), dot(rows[1], h), dot(rows[2], h));
}

// A normal through `rows` (the inverse transpose of the transform the rows' inverse describes).
fn card_normal(n: vec3<f32>, inverse_rows: array<vec4<f32>, 3>) -> vec3<f32> {
    return normalize(n.x * inverse_rows[0].xyz + n.y * inverse_rows[1].xyz + n.z * inverse_rows[2].xyz + vec3<f32>(0.0, 1e-7, 0.0));
}

// Octahedral normal of a captured texel (8 bits per component, encode_octahedral(n, 8)).
fn card_texel_normal(packed: u32) -> vec3<f32> {
    let e = vec2<f32>(f32(packed & 0xffu), f32((packed >> 8u) & 0xffu)) / 127.0 - 1.0;
    let z = 1.0 - abs(e.x) - abs(e.y);
    let t = max(-z, 0.0);
    return normalize(vec3<f32>(e.x + select(t, -t, e.x >= 0.0), e.y + select(t, -t, e.y >= 0.0), z));
}

// Mesh-space position of the surface a captured texel (`packed`) at mip-0 texel `texel` sees.
fn card_texel_position(card: Card, direction: u32, texel: vec2<f32>, packed: u32) -> vec3<f32> {
    let c = direction / 2u;
    let sign = select(1.0, -1.0, (direction & 1u) != 0u);
    let uv = card.frame.xy + texel * card.frame.z;
    var q = vec3<f32>(0.0);
    q[(c + 1u) % 3u] = uv.x;
    q[(c + 2u) % 3u] = uv.y;
    q[c] = card.frame.w - sign * (card.depth.x + f32(packed & 0xffffu) * card.depth.y);
    return q;
}

// Lit texel `texel` (at the card's mip) through its page table.
fn card_lighting_texel(table: u32, pages_x: u32, texel: vec2<u32>) -> vec4<f32> {
    let page = card_pages[table + (texel.y / CARD_PAGE_TEXELS) * pages_x + texel.x / CARD_PAGE_TEXELS];
    if page == CARD_NONE {
        return vec4<f32>(0.0);
    }
    let origin = vec2<u32>(page % lumen.cards.z, page / lumen.cards.z) * CARD_PAGE_TEXELS;
    return textureLoad(card_lighting, origin + texel % CARD_PAGE_TEXELS, 0);
}

// Radiance of mesh-space point `q` (normal `n`) in one card, a = confidence (0: no match).
// `tolerance` is the uncertainty of the hit position in mesh units.
fn card_sample(instance: u32, slot: u32, card: Card, direction: u32, q: vec3<f32>, n: vec3<f32>, tolerance: f32) -> vec4<f32> {
    let c = direction / 2u;
    let sign = select(1.0, -1.0, (direction & 1u) != 0u);
    let texel = (vec2<f32>(q[(c + 1u) % 3u], q[(c + 2u) % 3u]) - card.frame.xy) / card.frame.z;
    let resolution = card_unpack(card.texels.x);
    if any(texel < vec2<f32>(0.0)) || any(texel >= vec2<f32>(resolution)) {
        return vec4<f32>(0.0);
    }
    let captured = textureLoad(card_capture, card_unpack(card.texels.y) + vec2<u32>(texel), 0).r;
    if (captured & 0xffffu) == 0xffffu {
        return vec4<f32>(0.0);
    }
    // The hit must lie on the captured surface: within the hit's own uncertainty, the
    // distance field's (hits come from it) and the depth change across a sloped texel.
    let depth = card.depth.x + f32(captured & 0xffffu) * card.depth.y;
    let hit_depth = sign * (card.frame.w - q[c]);
    let gap = abs(hit_depth - depth);
    let range = tolerance + card.depth.z + 2.0 * card.frame.z;
    let agreement = saturate(dot(card_texel_normal(captured >> 16u), n) * 1.5 - 0.25);
    let confidence = saturate(1.0 - gap / range) * agreement;
    if confidence <= 0.0 {
        return vec4<f32>(0.0);
    }

    let allocation = card_pages[instance * CARD_SLOTS + slot];
    if allocation == CARD_NONE {
        return vec4<f32>(0.0);
    }
    let mip = allocation & 15u;
    let table = allocation >> 4u;
    let mip_resolution = card_mip_resolution(resolution, mip);
    let pages_x = (mip_resolution.x + CARD_PAGE_TEXELS - 1u) / CARD_PAGE_TEXELS;
    // Bilinear over the lit texels, skipping texels without a surface and those whose surface
    // is not the hit's: a texel's light belongs to the surface it was lit at (its stored depth),
    // which must lie on the hit's tangent plane (extrapolated along the hit normal), so the light
    // of a mushroom cap does not bleed onto the ground beside it, nor a wall's onto the floor.
    let scale = f32(1u << mip);
    let t = texel / scale - 0.5;
    let base = vec2<i32>(floor(t));
    let f = t - floor(t);
    let slope = vec2<f32>(n[(c + 1u) % 3u], n[(c + 2u) % 3u]) / max(sign * n[c], 0.3);
    let depth_range = tolerance + 0.5 * card.depth.z + 0.5 * card.frame.z * scale;
    var sum = vec4<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let o = vec2<i32>(i & 1, i >> 1);
        let tap = clamp(base + o, vec2<i32>(0), vec2<i32>(mip_resolution) - 1);
        let lit = card_lighting_texel(table, pages_x, vec2<u32>(tap));
        if lit.a <= 0.0 {
            continue;
        }
        let tap_depth = card.depth.x + (lit.a - 1.0) * CARD_DEPTH_STEPS * card.depth.y;
        let offset = ((vec2<f32>(tap) + 0.5) * scale - texel) * card.frame.z;
        let expected = hit_depth + dot(clamp(slope, vec2<f32>(-4.0), vec2<f32>(4.0)), offset);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * saturate(1.0 - abs(tap_depth - expected) / depth_range);
        sum += vec4<f32>(lit.rgb, 1.0) * w;
    }
    if sum.a < 1e-3 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum.rgb / sum.a, confidence * saturate(sum.a * 4.0));
}

struct CardRadiance {
    radiance: vec3<f32>,
    // Summed confidence of the cards that matched (~1 for a well-covered surface, 0: none).
    weight: f32,
}

// Radiance leaving world point `p` with normal `n` on card instance `instance`, from the cards
// facing `n` (weighted by how squarely they face it). `tolerance`: world-space uncertainty of p.
fn card_radiance(instance: u32, p: vec3<f32>, n: vec3<f32>, tolerance: f32) -> CardRadiance {
    var r: CardRadiance;
    r.radiance = vec3<f32>(0.0);
    r.weight = 0.0;
    if instance >= lumen.cards.x {
        return r;
    }
    let inst = card_instance_load(instance);
    let q = card_point(p, inst.to_local);
    let nl = card_normal(n, inst.to_world);
    let local_tolerance = tolerance / bitcast<f32>(inst.info.w);
    var sum = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let facing = nl[c];
        // Cards capture surfaces within ~73 degrees of their direction (MIN_FACING in cards.rs).
        if abs(facing) < 0.3 {
            continue;
        }
        let direction = c * 2u + select(0u, 1u, facing < 0.0);
        for (var layer = 0u; layer < card_layer_count(inst.info, direction); layer++) {
            let slot = direction * CARD_LAYERS + layer;
            let s = card_sample(instance, slot, card_load(inst.info.x + slot), direction, q, nl, local_tolerance);
            if s.a > 0.0 {
                let w = facing * facing * s.a;
                sum += s.rgb * w;
                r.weight += w;
                break;
            }
        }
    }
    r.radiance = sum / max(r.weight, 1e-6);
    return r;
}

// Blends card radiance over `fallback` (the voxel surface cache) by the cards' confidence.
fn card_surface_radiance(fallback: vec3<f32>, instance: u32, p: vec3<f32>, n: vec3<f32>, tolerance: f32) -> vec3<f32> {
    let cards = card_radiance(instance, p, n, tolerance);
    return mix(fallback, cards.radiance, saturate(cards.weight * 2.0));
}

// Card instance closest to `p` according to the global distance field's level `level`
// (CARD_NONE outside the level or where no instance is near).
fn sdf_instance_at(p: vec3<f32>, level: u32) -> u32 {
    let l = lumen.sdf_levels[level];
    let g = vec3<i32>(floor((p - l.xyz) / l.w));
    let res = i32(SDF_RES);
    if any(g < vec3<i32>(0)) || any(g >= vec3<i32>(res)) {
        return CARD_NONE;
    }
    let word = textureLoad(sdf_instance_ids, vec3<i32>(g.x >> 1u, g.y, g.z + i32(level) * res), 0).r;
    let id = (word >> ((u32(g.x) & 1u) * 16u)) & 0xffffu;
    return select(id, CARD_NONE, id == 0xffffu);
}

// card_surface_radiance for a point found through the global distance field (level `level`):
// the instance is the closest one to a point just inside the surface or, failing that (another
// instance's interior dominates there), to a point just outside it.
fn card_field_radiance(fallback: vec3<f32>, p: vec3<f32>, n: vec3<f32>, level: u32, tolerance: f32) -> vec3<f32> {
    let offset = n * lumen.sdf_levels[level].w * 0.5;
    let inside = sdf_instance_at(p - offset, level);
    var cards = card_radiance(inside, p, n, tolerance);
    if cards.weight < 0.25 {
        let outside = sdf_instance_at(p + offset, level);
        if outside != inside {
            let other = card_radiance(outside, p, n, tolerance);
            if other.weight > cards.weight {
                cards = other;
            }
        }
    }
    return mix(fallback, cards.radiance, saturate(cards.weight * 2.0));
}
