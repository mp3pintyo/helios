// Nanite tessellation: patches, tessellation factors and the dicing pattern (tessellation.rs).
// Requires common.wgsl, geometry.wgsl, displacement.wgsl and two functions of the includer's view:
//   tess_fetch(item) -> vec2<u32>             (instance, cluster) of a rasteriser item
//   tess_pixels_per_unit(item, world) -> f32  pixels (or shadow texels) per world unit at `world`
//
// Every triangle of a displaced cluster is a *patch*. A patch is a triangle in the barycentric
// space of its base triangle: its corners' barycentrics are fixed point with 15 fractional bits,
// so midpoints are exact and both sides of an edge see bit-identical corners.
//
// Tessellation factors are per edge and depend only on the edge: its undisplaced world length
// projected to pixels as if it faced the camera (so displacement towards the viewer still gets
// diced), divided by the dice rate (UE's TessFactor). Edges are oriented canonically (by their
// end points' object positions), so the two triangles sharing one compute the same factor and the
// same vertices on it — across clusters and LOD levels too, whose shared vertices are
// bit-identical.
//
// Patches whose factors exceed TESS_MAX_DICE are split (tess_queue.wgsl): every edge over it at
// its midpoint, the others not (one, two or three edges: two, three or four children, the
// red-green refinement of adaptive meshes), so a split again depends only on the edge. The rest are
// diced: an edge with factor c gets 2^ceil(log2 c) segments (at most 8), the inside a uniform grid
// of F = 2^level segments per side with the largest edge level. Grid vertices on an edge with
// fewer segments than F snap to that edge's segments (a zipper of degenerate triangles), and
// vertices that are new at a level *morph* out of the coarser segment they split, to their
// displaced place, as the factor grows from 2^(level-1) to 2^level (geomorphing): the micro-mesh
// changes continuously as the camera moves, instead of popping when a factor crosses a power of
// two, and both sides of an edge morph its vertices alike.

const TESS_ONE: u32 = 32768u;              // barycentric 1.0 in fixed point
const TESS_MAX_DICE: f32 = 8.0;           // largest factor an edge is diced with
const TESS_MAX_LEVEL: u32 = 3u;           // log2(TESS_MAX_DICE)

// Corner barycentrics (b0, b1; b2 implied) packed into a u32.
fn tess_unpack(packed: u32) -> vec3<u32> {
    let b0 = packed & 0xffffu;
    let b1 = packed >> 16u;
    return vec3<u32>(b0, b1, TESS_ONE - b0 - b1);
}

fn tess_pack(b: vec3<u32>) -> u32 {
    return b.x | (b.y << 16u);
}

// A point of the undisplaced surface.
struct TessPoint {
    object: vec3<f32>,
    normal: vec3<f32>,   // unit, object space
    uv: vec2<f32>,
}

// The base triangle of a patch.
struct TessBase {
    instance: u32,
    cluster: u32,
    v0: TessPoint,
    v1: TessPoint,
    v2: TessPoint,
}

// Patch item: rasteriser item << 7 | triangle.
fn tess_base(item: u32) -> TessBase {
    let ids = tess_fetch(item >> TRIANGLE_BITS);
    let cluster = clusters[ids.y];
    let base = cluster_base(cluster);
    let corners = cluster_triangle(cluster, base, item & TRIANGLE_MASK);
    var b: TessBase;
    b.instance = ids.x;
    b.cluster = ids.y;
    b.v0 = TessPoint(vertex_position(cluster, base, corners.x), vertex_normal(cluster, base, corners.x), vertex_uv(cluster, base, corners.x));
    b.v1 = TessPoint(vertex_position(cluster, base, corners.y), vertex_normal(cluster, base, corners.y), vertex_uv(cluster, base, corners.y));
    b.v2 = TessPoint(vertex_position(cluster, base, corners.z), vertex_normal(cluster, base, corners.z), vertex_uv(cluster, base, corners.z));
    return b;
}

fn tess_vertex(b: TessBase, k: u32) -> TessPoint {
    if k == 0u {
        return b.v0;
    }
    if k == 1u {
        return b.v1;
    }
    return b.v2;
}

// Lexicographic order of positions: orients a shared edge the same way from both sides.
fn tess_precedes(a: vec3<f32>, b: vec3<f32>) -> bool {
    if a.x != b.x {
        return a.x < b.x;
    }
    if a.y != b.y {
        return a.y < b.y;
    }
    return a.z < b.z;
}

// Corners (first, second) of base edge k (opposite corner k) in canonical order.
fn tess_base_edge(b: TessBase, k: u32) -> vec2<u32> {
    let e = vec2<u32>((k + 1u) % 3u, (k + 2u) % 3u);
    return select(e, e.yx, tess_precedes(tess_vertex(b, e.y).object, tess_vertex(b, e.x).object));
}

// The point at parameter s from the canonical first corner of base edge k: the triangle on the
// other side computes the same.
fn tess_edge_point(b: TessBase, k: u32, s: f32) -> TessPoint {
    let e = tess_base_edge(b, k);
    let lo = tess_vertex(b, e.x);
    let hi = tess_vertex(b, e.y);
    return TessPoint(mix(lo.object, hi.object, s), normalize(mix(lo.normal, hi.normal, s)), mix(lo.uv, hi.uv, s));
}

fn tess_interior_point(b: TessBase, l: vec3<f32>) -> TessPoint {
    let object = (b.v0.object * l.x) + (b.v1.object * l.y) + (b.v2.object * l.z);
    let normal = normalize((b.v0.normal * l.x) + (b.v1.normal * l.y) + (b.v2.normal * l.z));
    return TessPoint(object, normal, (b.v0.uv * l.x) + (b.v1.uv * l.y) + (b.v2.uv * l.z));
}

// A patch corner: a base corner, a point of a base edge or a point inside.
fn tess_corner_point(b: TessBase, c: vec3<u32>) -> TessPoint {
    for (var k = 0u; k < 3u; k++) {
        if c[k] == TESS_ONE {
            return tess_vertex(b, k);
        }
    }
    for (var k = 0u; k < 3u; k++) {
        if c[k] == 0u {
            return tess_edge_point(b, k, f32(c[tess_base_edge(b, k).y]) / f32(TESS_ONE));
        }
    }
    return tess_interior_point(b, vec3<f32>(c) / f32(TESS_ONE));
}

// The point at parameter t of the patch edge from corner `lo` to corner `hi` (canonical order).
// On a base edge it is placed along that edge, as the neighbouring base triangle places it.
fn tess_patch_edge_point(b: TessBase, lo: vec3<u32>, hi: vec3<u32>, t: f32) -> TessPoint {
    for (var k = 0u; k < 3u; k++) {
        if lo[k] == 0u && hi[k] == 0u {
            let far = tess_base_edge(b, k).y;
            return tess_edge_point(b, k, mix(f32(lo[far]), f32(hi[far]), t) / f32(TESS_ONE));
        }
    }
    return tess_interior_point(b, mix(vec3<f32>(lo), vec3<f32>(hi), t) / f32(TESS_ONE));
}

fn tess_world(instance: Instance, object: vec3<f32>) -> vec3<f32> {
    return (instance.world * vec4<f32>(object, 1.0)).xyz;
}

// Tessellation factor of the edge between world points a and b given in canonical order. Edges
// are diced into 2^ceil(log2 factor) segments: the 1/sqrt(2) centres segment lengths on the dice
// rate (within a factor sqrt(2) either way) instead of between half of it and all of it.
fn tess_edge_factor(item: u32, a: vec3<f32>, b: vec3<f32>) -> f32 {
    return distance(a, b) * tess_pixels_per_unit(item, (a + b) * 0.5) * (0.70710678 / tess.dice.x);
}

// Dicing level (2^level segments, at most TESS_MAX_LEVEL) of an edge with factor c, and the morph
// weight of the vertices new at that level: 0 when c just passed 2^(level-1) (they lie on their
// coarser neighbour), 1 at c = 2^level.
struct TessLevel {
    level: u32,
    weight: f32,
}

fn tess_level(c: f32) -> TessLevel {
    let l = log2(max(c, 1.0));
    let level = min(u32(ceil(l)), TESS_MAX_LEVEL);
    return TessLevel(level, select(1.0, saturate(l - f32(level) + 1.0), level > 0u));
}

// A patch, ready to dice or split.
struct TessPatch {
    corners: array<vec3<u32>, 3>,
    // Undisplaced corner points and their world positions (no wind).
    c0: TessPoint,
    c1: TessPoint,
    c2: TessPoint,
    w0: vec3<f32>,
    w1: vec3<f32>,
    w2: vec3<f32>,
    // Per edge e (opposite corner e, from corner e + 1 to e + 2): factor and whether its canonical
    // first end is corner e + 2.
    factors: vec3<f32>,
    flipped: vec3<bool>,
}

fn tess_patch(item: u32, b: TessBase, instance: Instance, packed: vec3<u32>) -> TessPatch {
    var p: TessPatch;
    p.corners = array<vec3<u32>, 3>(tess_unpack(packed.x), tess_unpack(packed.y), tess_unpack(packed.z));
    p.c0 = tess_corner_point(b, p.corners[0]);
    p.c1 = tess_corner_point(b, p.corners[1]);
    p.c2 = tess_corner_point(b, p.corners[2]);
    p.w0 = tess_world(instance, p.c0.object);
    p.w1 = tess_world(instance, p.c1.object);
    p.w2 = tess_world(instance, p.c2.object);
    p.flipped = vec3<bool>(tess_precedes(p.c2.object, p.c1.object), tess_precedes(p.c0.object, p.c2.object), tess_precedes(p.c1.object, p.c0.object));
    p.factors = vec3<f32>(
        tess_edge_factor(item, select(p.w1, p.w2, p.flipped.x), select(p.w2, p.w1, p.flipped.x)),
        tess_edge_factor(item, select(p.w2, p.w0, p.flipped.y), select(p.w0, p.w2, p.flipped.y)),
        tess_edge_factor(item, select(p.w0, p.w1, p.flipped.z), select(p.w1, p.w0, p.flipped.z)),
    );
    return p;
}

// Whether the edge between fixed-point corners a and b may be split: its midpoint must be exact.
fn tess_splittable(a: vec3<u32>, b: vec3<u32>) -> bool {
    let sum = a + b;
    return ((sum.x | sum.y) & 1u) == 0u;
}

// Edges of the patch to split (bit e: edge e).
fn tess_split_mask(p: TessPatch) -> u32 {
    var mask = 0u;
    for (var e = 0u; e < 3u; e++) {
        if p.factors[e] > TESS_MAX_DICE && tess_splittable(p.corners[(e + 1u) % 3u], p.corners[(e + 2u) % 3u]) {
            mask |= 1u << e;
        }
    }
    return mask;
}

// Dicing of a patch: per edge its level and morph weight, the grid level F = 2^level and the
// morph weight of the grid's inner vertices (from its finest edge).
struct TessDice {
    edges: vec3<u32>,
    weights: vec3<f32>,
    level: u32,
    weight: f32,
}

fn tess_dice(p: TessPatch) -> TessDice {
    var d: TessDice;
    let l0 = tess_level(p.factors.x);
    let l1 = tess_level(p.factors.y);
    let l2 = tess_level(p.factors.z);
    d.edges = vec3<u32>(l0.level, l1.level, l2.level);
    d.weights = vec3<f32>(l0.weight, l1.weight, l2.weight);
    let finest = max(max(p.factors.x, p.factors.y), p.factors.z);
    let inner = tess_level(finest);
    d.level = inner.level;
    d.weight = inner.weight;
    return d;
}

// ---- Splitting ----------------------------------------------------------------------------------

// Lists a patch can go to: the dice lists of levels 0..3, the next split queue.
const TESS_LISTS: u32 = 5u;
const TESS_LIST_QUEUE: u32 = 4u;

// Where a patch goes: up to four records into one list.
struct TessRoute {
    list: u32,
    count: u32,
    records: array<vec4<u32>, 4>,
}

// Splits the patch along the edges in `mask` (bit e: edge e at its midpoint) into its children,
// each wound like the patch: one edge gives two, two edges three, three edges four.
fn tess_children(item: u32, c: array<vec3<u32>, 3>, mask: u32) -> TessRoute {
    var r: TessRoute;
    r.list = TESS_LIST_QUEUE;
    let m = array<vec3<u32>, 3>((c[1] + c[2]) / 2u, (c[2] + c[0]) / 2u, (c[0] + c[1]) / 2u);
    let split = countOneBits(mask);
    if split == 3u {
        r.count = 4u;
        r.records[0] = vec4<u32>(item, tess_pack(c[0]), tess_pack(m[2]), tess_pack(m[1]));
        r.records[1] = vec4<u32>(item, tess_pack(m[2]), tess_pack(c[1]), tess_pack(m[0]));
        r.records[2] = vec4<u32>(item, tess_pack(m[1]), tess_pack(m[0]), tess_pack(c[2]));
        r.records[3] = vec4<u32>(item, tess_pack(m[0]), tess_pack(m[1]), tess_pack(m[2]));
    } else if split == 2u {
        // The unsplit edge u is opposite corner o = c[u]; a, b follow o.
        let u = firstTrailingBit(~mask & 7u);
        let o = c[u];
        let a = c[(u + 1u) % 3u];
        let b = c[(u + 2u) % 3u];
        let oa = m[(u + 2u) % 3u];
        let bo = m[(u + 1u) % 3u];
        r.count = 3u;
        r.records[0] = vec4<u32>(item, tess_pack(o), tess_pack(oa), tess_pack(bo));
        r.records[1] = vec4<u32>(item, tess_pack(oa), tess_pack(a), tess_pack(b));
        r.records[2] = vec4<u32>(item, tess_pack(oa), tess_pack(b), tess_pack(bo));
    } else {
        let e = firstTrailingBit(mask);
        let o = c[e];
        let a = c[(e + 1u) % 3u];
        let b = c[(e + 2u) % 3u];
        r.count = 2u;
        r.records[0] = vec4<u32>(item, tess_pack(o), tess_pack(a), tess_pack(m[e]));
        r.records[1] = vec4<u32>(item, tess_pack(o), tess_pack(m[e]), tess_pack(b));
    }
    return r;
}

// ---- The dicing grid ------------------------------------------------------------------------------
// Grid vertex (i, j) of a patch diced with F segments: corner 0 + (corner 1 - corner 0) i / F +
// (corner 2 - corner 0) j / F, i + j <= F, numbered row by row (row j has F + 1 - j vertices).
// Row j holds 2 (F - j) - 1 triangles, alternately "up" (i, j), (i+1, j), (i, j+1) and "down"
// (i+1, j), (i+1, j+1), (i, j+1), all wound like the patch.

fn tess_grid_index(i: u32, j: u32, f: u32) -> u32 {
    return (j * (f + 1u)) - ((j * (j - min(j, 1u))) / 2u) + i;
}

fn tess_grid_vertex(index: u32, f: u32) -> vec2<u32> {
    var j = 0u;
    var first = 0u;
    loop {
        let row = f + 1u - j;
        if index < first + row || j >= f {
            break;
        }
        first += row;
        j++;
    }
    return vec2<u32>(index - first, j);
}

// Grid vertex indices of micro-triangle t of a patch with F segments.
fn tess_grid_triangle(t: u32, f: u32) -> vec3<u32> {
    var j = 0u;
    var first = 0u;
    loop {
        let row = (2u * (f - j)) - 1u;
        if t < first + row || j + 1u >= f {
            break;
        }
        first += row;
        j++;
    }
    let r = t - first;
    let i = r / 2u;
    if (r & 1u) == 0u {
        return vec3<u32>(tess_grid_index(i, j, f), tess_grid_index(i + 1u, j, f), tess_grid_index(i, j + 1u, f));
    }
    return vec3<u32>(tess_grid_index(i + 1u, j, f), tess_grid_index(i + 1u, j + 1u, f), tess_grid_index(i, j + 1u, f));
}

// The edge of a grid vertex on the patch boundary: (edge e, index k from its first corner e + 1),
// e = 3 inside.
fn tess_grid_edge(i: u32, j: u32, f: u32) -> vec2<u32> {
    if j == 0u {
        return vec2<u32>(2u, i);
    }
    if i == 0u {
        return vec2<u32>(1u, f - j);
    }
    if i + j == f {
        return vec2<u32>(0u, j);
    }
    return vec2<u32>(3u, 0u);
}

// The point at index `index` of 2^edge_level segments along patch edge e, counted from its
// canonical first end.
fn tess_edge_grid_point(b: TessBase, p: TessPatch, e: u32, index: u32, edge_level: u32) -> TessPoint {
    let flipped = p.flipped[e];
    let first = p.corners[(e + 1u) % 3u];
    let second = p.corners[(e + 2u) % 3u];
    return tess_patch_edge_point(b, select(first, second, flipped), select(second, first, flipped), f32(index) / f32(1u << edge_level));
}

// The undisplaced point of grid vertex (i, j) of a diced patch; vertices on an edge with fewer
// segments than the grid snap to the edge's own segments.
fn tess_grid_point(b: TessBase, p: TessPatch, d: TessDice, i: u32, j: u32) -> TessPoint {
    let f = 1u << d.level;
    if i == f {
        return p.c1;
    }
    if j == f {
        return p.c2;
    }
    if i == 0u && j == 0u {
        return p.c0;
    }
    let edge = tess_grid_edge(i, j, f);
    if edge.x < 3u {
        let canonical = select(edge.y, f - edge.y, p.flipped[edge.x]);
        let edge_level = d.edges[edge.x];
        return tess_edge_grid_point(b, p, edge.x, canonical >> (d.level - edge_level), edge_level);
    }
    let c0 = vec3<f32>(p.corners[0]);
    let l = (c0 + ((vec3<f32>(p.corners[1]) - c0) * (f32(i) / f32(f))) + ((vec3<f32>(p.corners[2]) - c0) * (f32(j) / f32(f)))) / f32(TESS_ONE);
    return tess_interior_point(b, l);
}

// A grid vertex and, if it is new at its level, the two vertices of the coarser level whose
// segment it splits and how far it has grown out of that segment (morph 1: in place).
struct TessVertex {
    point: TessPoint,
    morph: f32,
    a: TessPoint,
    c: TessPoint,
}

fn tess_grid_vertex_points(b: TessBase, p: TessPatch, d: TessDice, i: u32, j: u32) -> TessVertex {
    let f = 1u << d.level;
    var v: TessVertex;
    v.point = tess_grid_point(b, p, d, i, j);
    v.morph = 1.0;
    let edge = tess_grid_edge(i, j, f);
    if edge.x < 3u && i < f && j < f {
        let canonical = select(edge.y, f - edge.y, p.flipped[edge.x]);
        let edge_level = d.edges[edge.x];
        let snapped = canonical >> (d.level - edge_level);
        if (snapped & 1u) == 1u {
            v.morph = d.weights[edge.x];
            v.a = tess_edge_grid_point(b, p, edge.x, snapped - 1u, edge_level);
            v.c = tess_edge_grid_point(b, p, edge.x, snapped + 1u, edge_level);
        }
    } else if edge.x == 3u && ((i | j) & 1u) == 1u {
        // The coarser grid's segment through it: along i, along j or the diagonal.
        v.morph = d.weight;
        let odd = vec2<bool>((i & 1u) == 1u, (j & 1u) == 1u);
        let da = select(vec2<i32>(0, -1), select(vec2<i32>(-1, 0), vec2<i32>(-1, 1), odd.y), odd.x);
        let a = vec2<u32>(vec2<i32>(i32(i), i32(j)) + da);
        let c = vec2<u32>(vec2<i32>(i32(i), i32(j)) - da);
        v.a = tess_grid_point(b, p, d, a.x, a.y);
        v.c = tess_grid_point(b, p, d, c.x, c.y);
    }
    return v;
}

// ---- Displacement ---------------------------------------------------------------------------------

// The displaced object-space position of an undisplaced point of instance `instance`.
fn tess_displaced(instance: Instance, point: TessPoint) -> vec3<f32> {
    let normal_matrix = mat3x3<f32>(instance.normal_matrix_0.xyz, instance.normal_matrix_1.xyz, instance.normal_matrix_2.xyz);
    let input = DisplacementInput(point.object, tess_world(instance, point.object), point.normal, normalize(normal_matrix * point.normal), point.uv);
    let height = displacement_height(instance.material, instance.max_scale, input);
    return point.object + (point.normal * displacement_offset(instance.material, height));
}

// The displaced object-space position of a grid vertex: a vertex new at its level grows out of the
// coarser segment it splits, from the segment's midpoint to its displaced place as the morph weight
// goes from 0 to 1 (geomorphing in displaced space, as CDLOD does for terrain: interpolating its
// parameter instead would slide the displacement across the surface, a visible swim).
fn tess_vertex_position(instance: Instance, v: TessVertex) -> vec3<f32> {
    let own = tess_displaced(instance, v.point);
    if v.morph >= 1.0 {
        return own;
    }
    return mix((tess_displaced(instance, v.a) + tess_displaced(instance, v.c)) * 0.5, own, v.morph);
}
