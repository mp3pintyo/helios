//! A CPU mirror of the tessellator's patch splitting and dicing (`tess_core.wgsl`,
//! `tess_queue.wgsl`) on the undisplaced surface, to check the properties the shaders rely on:
//! the two triangles sharing an edge place exactly the same vertices on it, displaced and morphed
//! (no cracks or T-junctions, whatever their other edges, split depths and grid levels), and the
//! displaced edge moves continuously as the camera does (geomorphing: no pops when a factor
//! crosses a power of two or an edge starts splitting).

use glam::Vec3;

const ONE: u32 = 32768;
const MAX_DICE: f32 = 8.0;
const MAX_LEVEL: u32 = 3;
const PASSES: u32 = 14;

/// The view: pixels per unit at distance 1, near plane, dice rate, eye.
struct View {
    scale: f32,
    near: f32,
    dice: f32,
    eye: Vec3,
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t))
}

fn precedes(a: Vec3, b: Vec3) -> bool {
    if a.x != b.x {
        return a.x < b.x;
    }
    if a.y != b.y {
        return a.y < b.y;
    }
    a.z < b.z
}

/// Canonical (first, second) corners of base edge k.
fn base_edge(p: &[Vec3; 3], k: usize) -> (usize, usize) {
    let e = ((k + 1) % 3, (k + 2) % 3);
    if precedes(p[e.1], p[e.0]) {
        (e.1, e.0)
    } else {
        e
    }
}

/// A vertex: its position and, per base edge it lies on, its parameter from the edge's canonical
/// first corner.
#[derive(Clone, Copy)]
struct Point {
    position: Vec3,
    on_edge: [Option<f32>; 3],
}

fn edge_point(p: &[Vec3; 3], k: usize, s: f32) -> Point {
    let (lo, hi) = base_edge(p, k);
    let mut on_edge = [None; 3];
    on_edge[k] = Some(s);
    // A base corner lies on its other edge too.
    for (end, corner) in [(0.0, lo), (1.0, hi)] {
        if s == end {
            let other = 3 - k - corner;
            let (first, _) = base_edge(p, other);
            on_edge[other] = Some(if first == corner { 0.0 } else { 1.0 });
        }
    }
    Point { position: mix3(p[lo], p[hi], s), on_edge }
}

fn interior_point(p: &[Vec3; 3], l: [f32; 3]) -> Point {
    Point { position: p[0] * l[0] + p[1] * l[1] + p[2] * l[2], on_edge: [None; 3] }
}

fn corner_point(p: &[Vec3; 3], c: [u32; 3]) -> Point {
    if let Some(k) = c.iter().position(|&x| x == ONE) {
        // On both edges through corner k.
        return edge_point(p, (k + 1) % 3, if base_edge(p, (k + 1) % 3).0 == k { 0.0 } else { 1.0 });
    }
    if let Some(k) = c.iter().position(|&x| x == 0) {
        return edge_point(p, k, c[base_edge(p, k).1] as f32 / ONE as f32);
    }
    interior_point(p, c.map(|x| x as f32 / ONE as f32))
}

fn patch_edge_point(p: &[Vec3; 3], lo: [u32; 3], hi: [u32; 3], t: f32) -> Point {
    for k in 0..3 {
        if lo[k] == 0 && hi[k] == 0 {
            let far = base_edge(p, k).1;
            return edge_point(p, k, mix(lo[far] as f32, hi[far] as f32, t) / ONE as f32);
        }
    }
    interior_point(p, [0, 1, 2].map(|i| mix(lo[i] as f32, hi[i] as f32, t) / ONE as f32))
}

fn edge_factor(view: &View, a: Vec3, b: Vec3) -> f32 {
    let mid = (a + b) * 0.5;
    a.distance(b) * (view.scale / mid.distance(view.eye).max(view.near)) * (0.70710678 / view.dice)
}

fn level(c: f32) -> (u32, f32) {
    let l = c.max(1.0).log2();
    let level = (l.ceil() as u32).min(MAX_LEVEL);
    (level, if level > 0 { (l - level as f32 + 1.0).clamp(0.0, 1.0) } else { 1.0 })
}

struct Patch {
    corners: [[u32; 3]; 3],
    points: [Point; 3],
    factors: [f32; 3],
    flipped: [bool; 3],
}

fn patch(view: &View, p: &[Vec3; 3], corners: [[u32; 3]; 3]) -> Patch {
    let points = corners.map(|c| corner_point(p, c));
    let pos = points.map(|q| q.position);
    let flipped = [precedes(pos[2], pos[1]), precedes(pos[0], pos[2]), precedes(pos[1], pos[0])];
    let factors = [0, 1, 2].map(|e| {
        let (a, b) = (pos[(e + 1) % 3], pos[(e + 2) % 3]);
        if flipped[e] {
            edge_factor(view, b, a)
        } else {
            edge_factor(view, a, b)
        }
    });
    Patch { corners, points, factors, flipped }
}

fn split_mask(p: &Patch) -> u32 {
    (0..3)
        .filter(|&e| {
            let (a, b) = (p.corners[(e + 1) % 3], p.corners[(e + 2) % 3]);
            p.factors[e] > MAX_DICE && ((a[0] + b[0]) | (a[1] + b[1])) & 1 == 0
        })
        .fold(0, |m, e| m | 1 << e)
}

fn children(c: [[u32; 3]; 3], mask: u32) -> Vec<[[u32; 3]; 3]> {
    let mid = |a: [u32; 3], b: [u32; 3]| [0, 1, 2].map(|i| (a[i] + b[i]) / 2);
    let m = [mid(c[1], c[2]), mid(c[2], c[0]), mid(c[0], c[1])];
    match mask.count_ones() {
        3 => vec![[c[0], m[2], m[1]], [m[2], c[1], m[0]], [m[1], m[0], c[2]], [m[0], m[1], m[2]]],
        2 => {
            let u = (!mask & 7).trailing_zeros() as usize;
            let (o, a, b) = (c[u], c[(u + 1) % 3], c[(u + 2) % 3]);
            let (oa, bo) = (m[(u + 2) % 3], m[(u + 1) % 3]);
            vec![[o, oa, bo], [oa, a, b], [oa, b, bo]]
        }
        _ => {
            let e = mask.trailing_zeros() as usize;
            let (o, a, b) = (c[e], c[(e + 1) % 3], c[(e + 2) % 3]);
            vec![[o, a, m[e]], [o, m[e], b]]
        }
    }
}

/// A synthetic height: a pure function of the undisplaced point, as a texture's or a graph's is.
type Height = fn(Vec3) -> f32;

/// Smooth waves (a texture-like height).
fn waves(q: Vec3) -> f32 {
    0.15 * (5.0 * q.x).sin() * (4.0 * q.z).cos()
}

/// A material graph's kind of height: the brick pavement preset's running bond on world xz —
/// floors, fractions and a steep bevelled step into the mortar, whose slope is discontinuous.
fn bricks(q: Vec3) -> f32 {
    let (x, z) = (q.x / 0.44, q.z / 0.22);
    let row = z.floor();
    let x = x + (row * 0.5).fract();
    let edge = (x.fract().min(1.0 - x.fract()) * 0.44).min(z.fract().min(1.0 - z.fract()) * 0.22);
    let t = ((edge - 0.02) / 0.03).clamp(0.0, 1.0);
    let brick = t * t * (3.0 - 2.0 * t);
    0.12 * (brick * (1.0 - 0.3 * ((x.floor() * 0.613 + row * 0.37).sin() * 0.5 + 0.5)) - 0.9)
}

fn displaced(point: &Point, height: Height) -> Vec3 {
    point.position + Vec3::Y * height(point.position)
}

/// The grid point (i, j) of a diced patch, snapped to its edge's segments (tess_grid_point).
fn grid_point(base: &[Vec3; 3], p: &Patch, i: u32, j: u32) -> Point {
    let levels = p.factors.map(level);
    let finest = level(p.factors.iter().copied().fold(0.0, f32::max)).0;
    let f = 1 << finest;
    if i == f {
        return p.points[1];
    }
    if j == f {
        return p.points[2];
    }
    if i == 0 && j == 0 {
        return p.points[0];
    }
    if let Some((e, k)) = grid_edge(i, j, f) {
        let canonical = if p.flipped[e] { f - k } else { k };
        return edge_grid_point(base, p, e, canonical >> (finest - levels[e].0), levels[e].0);
    }
    let c = p.corners.map(|c| c.map(|x| x as f32));
    let (u, v) = (i as f32 / f as f32, j as f32 / f as f32);
    interior_point(base, [0, 1, 2].map(|k| (c[0][k] + (c[1][k] - c[0][k]) * u + (c[2][k] - c[0][k]) * v) / ONE as f32))
}

fn grid_edge(i: u32, j: u32, f: u32) -> Option<(usize, u32)> {
    if j == 0 {
        Some((2, i))
    } else if i == 0 {
        Some((1, f - j))
    } else if i + j == f {
        Some((0, j))
    } else {
        None
    }
}

fn edge_grid_point(base: &[Vec3; 3], p: &Patch, e: usize, index: u32, edge_level: u32) -> Point {
    let (first, second) = (p.corners[(e + 1) % 3], p.corners[(e + 2) % 3]);
    let (lo, hi) = if p.flipped[e] { (second, first) } else { (first, second) };
    patch_edge_point(base, lo, hi, index as f32 / (1 << edge_level) as f32)
}

/// The grid vertex at index k from the first corner of patch edge e (tess_edge_grid_vertex).
fn edge_grid_vertex(e: usize, k: u32, f: u32) -> (u32, u32) {
    match e {
        2 => (k, 0),
        1 => (0, f - k),
        _ => (f - k, k),
    }
}

/// The morph of grid vertex (i, j), if it is new at its level: its weight and the two grid
/// vertices whose segment it splits (tess_grid_morph).
fn grid_morph(p: &Patch, i: u32, j: u32) -> Option<(f32, (u32, u32), (u32, u32))> {
    let levels = p.factors.map(level);
    let finest = level(p.factors.iter().copied().fold(0.0, f32::max));
    let f = 1 << finest.0;
    match grid_edge(i, j, f) {
        Some((e, k)) if i < f && j < f => {
            let canonical = if p.flipped[e] { f - k } else { k };
            let shift = finest.0 - levels[e].0;
            let snapped = canonical >> shift;
            let back = |c: u32| edge_grid_vertex(e, if p.flipped[e] { f - c } else { c }, f);
            (snapped & 1 == 1).then(|| (levels[e].1, back((snapped - 1) << shift), back((snapped + 1) << shift)))
        }
        None if (i | j) & 1 == 1 => {
            let (di, dj): (i32, i32) = match (i & 1 == 1, j & 1 == 1) {
                (true, true) => (-1, 1),
                (true, false) => (-1, 0),
                _ => (0, -1),
            };
            let at = |s: i32| ((i as i32 + s * di) as u32, (j as i32 + s * dj) as u32);
            Some((finest.1, at(1), at(-1)))
        }
        _ => None,
    }
}

/// A micro-triangle vertex: its undisplaced point and displaced position.
type MicroVertex = (Point, Vec3);

/// Every micro-triangle of a base triangle.
fn tessellate(view: &View, base: &[Vec3; 3], height: Height) -> Vec<[MicroVertex; 3]> {
    let whole = [[ONE, 0, 0], [0, ONE, 0], [0, 0, ONE]];
    let mut queue = vec![whole];
    let mut triangles = Vec::new();
    for pass in 0..=PASSES {
        let mut next = Vec::new();
        for corners in queue {
            let p = patch(view, base, corners);
            let mask = if pass == PASSES { 0 } else { split_mask(&p) };
            if mask != 0 {
                next.extend(children(p.corners, mask));
                continue;
            }
            let f = 1 << level(p.factors.iter().copied().fold(0.0, f32::max)).0;
            let index = |i: u32, j: u32| (j * (f + 1) - (j * j.saturating_sub(1)) / 2 + i) as usize;
            // Every grid vertex displaced once, then morphed from its neighbours' (tess.wgsl dice).
            let grid: Vec<(u32, u32)> = (0..=f).flat_map(|j| (0..=f - j).map(move |i| (i, j))).collect();
            let own: Vec<MicroVertex> = grid.iter().map(|&(i, j)| grid_point(base, &p, i, j)).map(|q| (q, displaced(&q, height))).collect();
            let vertices: Vec<MicroVertex> = grid
                .iter()
                .zip(&own)
                .map(|(&(i, j), &(q, position))| match grid_morph(&p, i, j) {
                    Some((weight, a, c)) => (q, ((own[index(a.0, a.1)].1 + own[index(c.0, c.1)].1) * 0.5).lerp(position, weight)),
                    None => (q, position),
                })
                .collect();
            for j in 0..f {
                for i in 0..f - j {
                    triangles.push([vertices[index(i, j)], vertices[index(i + 1, j)], vertices[index(i, j + 1)]]);
                    if i + 1 < f - j {
                        triangles.push([vertices[index(i + 1, j)], vertices[index(i + 1, j + 1)], vertices[index(i, j + 1)]]);
                    }
                }
            }
        }
        queue = next;
    }
    triangles
}

/// A micro-edge on a base edge: its end points' parameters along the edge and displaced positions.
type Segment = ((f32, Vec3), (f32, Vec3));

/// The non-degenerate micro-edges a triangle puts on its base edge k, ordered along it.
fn boundary(triangles: &[[MicroVertex; 3]], k: usize) -> Vec<Segment> {
    let mut segments: Vec<Segment> = triangles
        .iter()
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .filter_map(|(a, b)| Some(((a.0.on_edge[k]?, a.1), (b.0.on_edge[k]?, b.1))))
        .filter(|(a, b)| a.0 != b.0)
        .map(|(a, b)| if a.0 < b.0 { (a, b) } else { (b, a) })
        .collect();
    segments.sort_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());
    segments.dedup_by(|a, b| a.0 .0 == b.0 .0 && a.1 .0 == b.1 .0);
    segments
}

fn bits(segments: &[Segment]) -> Vec<[u32; 8]> {
    segments
        .iter()
        .map(|((s, a), (t, b))| [s.to_bits(), a.x.to_bits(), a.y.to_bits(), a.z.to_bits(), t.to_bits(), b.x.to_bits(), b.y.to_bits(), b.z.to_bits()])
        .collect()
}

/// A jittered grid of triangles (alternating diagonals) and the edges two of them share:
/// (triangle, its edge, other triangle, its edge).
fn grid() -> (Vec<[Vec3; 3]>, Vec<(usize, usize, usize, usize)>) {
    let n = 3;
    let vertex = |x: usize, z: usize| {
        let h = (x * 7 + z * 13) as f32;
        Vec3::new(x as f32 + 0.17 * (h * 1.3).sin(), 0.3 * (h * 0.7).cos(), z as f32 + 0.21 * (h * 2.1).cos())
    };
    let mut triangles: Vec<[(usize, usize); 3]> = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let (a, b, c, d) = ((x, z), (x + 1, z), (x + 1, z + 1), (x, z + 1));
            if (x + z) % 2 == 0 {
                triangles.extend([[a, c, b], [a, d, c]]);
            } else {
                triangles.extend([[a, d, b], [b, d, c]]);
            }
        }
    }
    let edge = |t: &[(usize, usize); 3], k: usize| {
        let (a, b) = (t[(k + 1) % 3], t[(k + 2) % 3]);
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    };
    let mut shared = Vec::new();
    for (i, ti) in triangles.iter().enumerate() {
        for (j, tj) in triangles.iter().enumerate().skip(i + 1) {
            for ki in 0..3 {
                for kj in 0..3 {
                    if edge(ti, ki) == edge(tj, kj) {
                        shared.push((i, ki, j, kj));
                    }
                }
            }
        }
    }
    (triangles.iter().map(|t| t.map(|(x, z)| vertex(x, z))).collect(), shared)
}

fn view(eye: Vec3) -> View {
    // A coarse view (factors up to ~100 next to the surface) keeps the test quick.
    View { scale: 60.0, near: 0.05, dice: 2.0, eye }
}

#[test]
fn shared_edges_get_identical_vertices() {
    shared_edges_agree(waves);
}

/// A graph's height (non-smooth, cells and cliffs) keeps the edges closed alike: the tessellator
/// needs the height to be a pure function of the point, not smooth.
#[test]
fn shared_edges_agree_under_a_graph_height() {
    shared_edges_agree(bricks);
}

fn shared_edges_agree(height: Height) {
    let (triangles, shared) = grid();
    // Close to the surface (deep splits next to coarse dicing), far away, and in between.
    for eye in [Vec3::new(0.3, 0.4, 0.2), Vec3::new(1.6, 0.25, 1.2), Vec3::new(-3.0, 2.0, 6.0), Vec3::new(7.0, 3.0, -4.0)] {
        let v = view(eye);
        let micro: Vec<_> = triangles.iter().map(|t| tessellate(&v, t, height)).collect();
        for &(a, ka, b, kb) in &shared {
            let (sa, sb) = (boundary(&micro[a], ka), boundary(&micro[b], kb));
            // The same vertices, bit for bit (parameters and displaced, morphed positions).
            assert_eq!(bits(&sa), bits(&sb), "triangles {a} and {b} disagree on their shared edge (eye {eye})");
            // The segments tile the edge.
            assert_eq!(sa.first().map(|s| s.0 .0), Some(0.0));
            assert_eq!(sa.last().map(|s| s.1 .0), Some(1.0));
            assert!(sa.windows(2).all(|w| w[0].1 .0 == w[1].0 .0), "gaps or overlaps on edge {a}/{b}");
        }
        let count: usize = micro.iter().map(Vec::len).sum();
        assert!(count > triangles.len(), "eye {eye}: nothing was tessellated");
    }
}

#[test]
fn edge_vertices_move_continuously_with_the_camera() {
    let (triangles, _) = grid();
    // Walk the eye towards the surface in small steps, through many factor doublings and splits:
    // every displaced vertex on an edge must start on (or next to) the edge's micro-polyline of
    // the step before — a vertex new at a level grows out of the segment it splits.
    let path = |s: f32| Vec3::new(1.4, 0.2, 1.2) + Vec3::new(3.0, 4.0, 2.5) * (1.0 - s);
    let polylines = |s: f32| {
        let v = view(path(s));
        triangles.iter().map(|t| (0..3).map(|k| boundary(&tessellate(&v, t, waves), k)).collect::<Vec<_>>()).collect::<Vec<_>>()
    };
    let distance = |p: Vec3, (a, b): (Vec3, Vec3)| {
        let d = b - a;
        let t = if d.length_squared() > 0.0 { ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        p.distance(a + d * t)
    };
    let steps = 200;
    let mut previous = polylines(0.0);
    let mut worst = 0.0f32;
    for step in 1..=steps {
        let current = polylines(step as f32 / steps as f32);
        for (t_now, t_before) in current.iter().zip(&previous) {
            for (now, before) in t_now.iter().zip(t_before) {
                for &((_, a), (_, b)) in now {
                    for p in [a, b] {
                        let nearest = before.iter().map(|&((_, x), (_, y))| distance(p, (x, y))).fold(f32::MAX, f32::min);
                        worst = worst.max(nearest);
                    }
                }
            }
        }
        previous = current;
    }
    // Edges are about a unit long and displaced by up to 0.15: popping vertices (no morph) would
    // leave the previous polyline by a sizeable part of that.
    assert!(worst < 0.01, "an edge vertex jumped by {worst} off the surface it had");
}
