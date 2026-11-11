//! Procedural modelling: the building blocks for architecture and vegetation.
//!
//! Every generator returns a [`Mesh`] with normals and world-scale UVs (one UV unit per metre
//! unless stated otherwise), so tiling textures look consistent across pieces. Hard edges are
//! separate wedges; call [`Mesh::weld`] before baking (the generators already do).

use glam::{Mat4, Vec2, Vec3};

use crate::noise::fbm;
use crate::{Mesh, Vertex};

// ---- Small builder --------------------------------------------------------------------------------

/// Accumulates vertices and triangles.
#[derive(Default)]
pub struct MeshBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Plant data ([`Mesh::plant`]): empty until a vertex is added with [`MeshBuilder::plant_vertex`].
    pub plant: Vec<Vec2>,
}

impl MeshBuilder {
    pub fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2) -> u32 {
        self.vertices.push(Vertex::new(p, n, uv));
        if !self.plant.is_empty() {
            self.plant.push(Vec2::ONE);
        }
        self.vertices.len() as u32 - 1
    }

    /// A vertex with plant data (wind weight, ambient occlusion; see [`Mesh::plant`]). Vertices
    /// added without it get (1, 1).
    pub fn plant_vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2, plant: Vec2) -> u32 {
        self.plant.resize(self.vertices.len(), Vec2::ONE);
        self.plant.push(plant);
        self.vertices.push(Vertex::new(p, n, uv));
        self.vertices.len() as u32 - 1
    }

    pub fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }

    /// Planar quad `a b c d` (counter-clockwise seen from the front) with UVs projected onto
    /// the quad's own axes in metres.
    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        let n = (b - a).cross(d - a).normalize_or(Vec3::Y);
        let u_axis = (b - a).normalize_or(Vec3::X);
        let v_axis = n.cross(u_axis);
        let uv = |p: Vec3| Vec2::new(p.dot(u_axis), p.dot(v_axis));
        let i = [a, b, c, d].map(|p| self.vertex(p, n, uv(p)));
        self.triangle(i[0], i[1], i[2]);
        self.triangle(i[0], i[2], i[3]);
    }

    pub fn append(&mut self, mesh: &Mesh) {
        let base = self.vertices.len() as u32;
        if !self.plant.is_empty() || !mesh.plant.is_empty() {
            self.plant.resize(self.vertices.len(), Vec2::ONE);
            self.plant.extend((0..mesh.vertices.len()).map(|i| mesh.plant_of(i)));
        }
        self.vertices.extend_from_slice(&mesh.vertices);
        self.indices.extend(mesh.indices.iter().map(|i| i + base));
    }

    pub fn build(mut self) -> Mesh {
        if !self.plant.is_empty() {
            self.plant.resize(self.vertices.len(), Vec2::ONE);
        }
        let mut mesh = Mesh { plant: self.plant, ..Mesh::new(self.vertices, self.indices) };
        mesh.weld();
        mesh
    }
}

/// Axis-aligned box with hard edges and per-face world-scale UVs.
pub fn cuboid(min: Vec3, max: Vec3) -> Mesh {
    let mut b = MeshBuilder::default();
    let c = |x: bool, y: bool, z: bool| Vec3::new(if x { max.x } else { min.x }, if y { max.y } else { min.y }, if z { max.z } else { min.z });
    b.quad(c(false, false, true), c(true, false, true), c(true, true, true), c(false, true, true)); // +Z
    b.quad(c(true, false, false), c(false, false, false), c(false, true, false), c(true, true, false)); // -Z
    b.quad(c(true, false, true), c(true, false, false), c(true, true, false), c(true, true, true)); // +X
    b.quad(c(false, false, false), c(false, false, true), c(false, true, true), c(false, true, false)); // -X
    b.quad(c(false, true, true), c(true, true, true), c(true, true, false), c(false, true, false)); // +Y
    b.quad(c(false, false, false), c(true, false, false), c(true, false, true), c(false, false, true)); // -Y
    b.build()
}

// ---- Polygon triangulation and extrusion -------------------------------------------------------------

fn signed_area(poly: &[Vec2]) -> f32 {
    let mut a = 0.0;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}

fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = (p - b).perp_dot(a - b);
    let d2 = (p - c).perp_dot(b - c);
    let d3 = (p - a).perp_dot(c - a);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// Ear-clipping triangulation of a simple polygon with holes. The outline is made
/// counter-clockwise and holes clockwise, then each hole is bridged to the outline.
/// Returns triangles as indices into the concatenation `outline ++ holes...`.
pub fn triangulate(outline: &[Vec2], holes: &[Vec<Vec2>]) -> Vec<[u32; 3]> {
    // Working polygon of (point, original index).
    let mut points: Vec<Vec2> = outline.to_vec();
    let mut ring: Vec<u32> = (0..outline.len() as u32).collect();
    if signed_area(outline) < 0.0 {
        ring.reverse();
    }
    let mut hole_list: Vec<Vec<u32>> = Vec::new();
    for hole in holes {
        let base = points.len() as u32;
        points.extend_from_slice(hole);
        let mut h: Vec<u32> = (base..base + hole.len() as u32).collect();
        if signed_area(hole) > 0.0 {
            h.reverse();
        }
        hole_list.push(h);
    }
    // Bridge holes, rightmost first.
    hole_list.sort_by(|a, b| {
        let ma = a.iter().map(|&i| points[i as usize].x).fold(f32::MIN, f32::max);
        let mb = b.iter().map(|&i| points[i as usize].x).fold(f32::MIN, f32::max);
        mb.total_cmp(&ma)
    });
    for hole in hole_list {
        let (hk, &hi) = hole.iter().enumerate().max_by(|a, b| points[*a.1 as usize].x.total_cmp(&points[*b.1 as usize].x)).unwrap();
        let hp = points[hi as usize];
        // Closest visible outline vertex to the right of the hole's rightmost point.
        let mut best: Option<(usize, f32)> = None;
        for (k, &ri) in ring.iter().enumerate() {
            let rp = points[ri as usize];
            if rp.x < hp.x - 1e-6 {
                continue;
            }
            let d = rp.distance_squared(hp);
            // Reject bridges crossing the ring.
            let crosses = (0..ring.len()).any(|e| {
                let (a, b) = (points[ring[e] as usize], points[ring[(e + 1) % ring.len()] as usize]);
                if ring[e] == ri || ring[(e + 1) % ring.len()] == ri {
                    return false;
                }
                segments_intersect(hp, rp, a, b)
            });
            if !crosses && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        let k = best.map(|b| b.0).unwrap_or(0);
        let mut merged = Vec::with_capacity(ring.len() + hole.len() + 2);
        merged.extend_from_slice(&ring[..=k]);
        for j in 0..=hole.len() {
            merged.push(hole[(hk + j) % hole.len()]);
        }
        merged.push(ring[k]);
        merged.extend_from_slice(&ring[k + 1..]);
        ring = merged;
    }

    let mut tris = Vec::new();
    let mut guard = 0;
    while ring.len() > 3 && guard < 100_000 {
        guard += 1;
        let n = ring.len();
        let mut clipped = false;
        for i in 0..n {
            let (ia, ib, ic) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
            let (a, b, c) = (points[ia as usize], points[ib as usize], points[ic as usize]);
            if (b - a).perp_dot(c - b) <= 1e-12 {
                continue; // reflex or degenerate
            }
            let blocked = ring.iter().any(|&j| {
                j != ia && j != ib && j != ic && {
                    let p = points[j as usize];
                    p != a && p != b && p != c && point_in_triangle(p, a, b, c)
                }
            });
            if blocked {
                continue;
            }
            tris.push([ia, ib, ic]);
            ring.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            // Numerical trouble: fan out the rest.
            for i in 1..ring.len() - 1 {
                tris.push([ring[0], ring[i], ring[i + 1]]);
            }
            ring.clear();
        }
    }
    if ring.len() == 3 {
        tris.push([ring[0], ring[1], ring[2]]);
    }
    tris
}

fn segments_intersect(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool {
    let d1 = (q2 - q1).perp_dot(p1 - q1);
    let d2 = (q2 - q1).perp_dot(p2 - q1);
    let d3 = (p2 - p1).perp_dot(q1 - p1);
    let d4 = (p2 - p1).perp_dot(q2 - p1);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// Extrudes a 2D shape (XY plane, holes allowed) along +Z by `depth`: front and back caps
/// plus side walls (hard edges). UVs are world-scale.
pub fn extrude(outline: &[Vec2], holes: &[Vec<Vec2>], depth: f32) -> Mesh {
    let mut b = MeshBuilder::default();
    let mut all: Vec<Vec2> = outline.to_vec();
    for h in holes {
        all.extend_from_slice(h);
    }
    let tris = triangulate(outline, holes);
    for (z, front) in [(depth, true), (0.0, false)] {
        let n = if front { Vec3::Z } else { Vec3::NEG_Z };
        let base = b.vertices.len() as u32;
        for p in &all {
            b.vertex(p.extend(z), n, *p);
        }
        for t in &tris {
            if front {
                b.triangle(base + t[0], base + t[1], base + t[2]);
            } else {
                b.triangle(base + t[0], base + t[2], base + t[1]);
            }
        }
    }
    let mut ring = |poly: &[Vec2], outward_ccw: bool| {
        let ccw = signed_area(poly) > 0.0;
        let flip = ccw != outward_ccw;
        for i in 0..poly.len() {
            let (mut p, mut q) = (poly[i], poly[(i + 1) % poly.len()]);
            if flip {
                std::mem::swap(&mut p, &mut q);
            }
            b.quad(p.extend(0.0), q.extend(0.0), q.extend(depth), p.extend(depth));
        }
    };
    ring(outline, true);
    for h in holes {
        ring(h, false);
    }
    b.build()
}

// ---- Curves -------------------------------------------------------------------------------------------

/// Points of a pointed (gothic) arch of `span` and `rise` above the springing line, from the
/// left springer over the apex to the right springer (`segments` per side). With
/// `rise = span * 0.866` it is an equilateral arch.
pub fn pointed_arch(span: f32, rise: f32, segments: usize) -> Vec<Vec2> {
    // Each side is a circle arc through the springer and the apex, centred on the springing line.
    let half = span * 0.5;
    // Centre at x = c (right arc for the left side): (c + half)^2 = (c)^2 + rise^2 solved for radius.
    let r = (half * half + rise * rise) / span;
    let c = r - half; // distance of the left side's centre right of the axis
    let mut left = Vec::with_capacity(segments + 1);
    let start = std::f32::consts::PI;
    let end = (rise / r).asin();
    let end = std::f32::consts::PI - end;
    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let a = start + (end - start) * t;
        left.push(Vec2::new(c + r * a.cos(), r * a.sin().abs()));
    }
    let mut points = left.clone();
    for p in left.iter().rev().skip(1) {
        points.push(Vec2::new(-p.x, p.y));
    }
    points
}

/// Semicircular (round) arch points from left to right.
pub fn round_arch(span: f32, segments: usize) -> Vec<Vec2> {
    let r = span * 0.5;
    (0..=segments)
        .map(|i| {
            let a = std::f32::consts::PI * (1.0 - i as f32 / segments as f32);
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// Wall panel `width × height` with a pointed-arch opening (`opening` wide, sill at
/// `sill`, springing at `spring`, apex at `apex`), extruded `thickness`, centred on x = 0.
pub fn arched_wall(width: f32, height: f32, opening: f32, sill: f32, spring: f32, apex: f32, thickness: f32) -> Mesh {
    let outline = vec![Vec2::new(-width * 0.5, 0.0), Vec2::new(width * 0.5, 0.0), Vec2::new(width * 0.5, height), Vec2::new(-width * 0.5, height)];
    let mut hole = vec![Vec2::new(-opening * 0.5, sill), Vec2::new(opening * 0.5, sill)];
    let arch = pointed_arch(opening, apex - spring, 10);
    for p in arch.iter().rev() {
        hole.push(Vec2::new(p.x, p.y + spring));
    }
    // Remove duplicate springer points.
    hole.dedup_by(|a, b| a.distance(*b) < 1e-4);
    extrude(&outline, &[hole], thickness)
}

// ---- Surfaces of revolution and sweeps ----------------------------------------------------------------

/// Lathe: revolves `profile` (x = radius, y = height, bottom to top) around +Y.
/// `radius_fn(angle, height)` may modulate the radius (fluting, carving). UV: u around
/// (circumference in metres), v = arc length along the profile.
pub fn lathe(profile: &[Vec2], segments: u32, radius_fn: &dyn Fn(f32, f32) -> f32, caps: bool) -> Mesh {
    let tau = std::f32::consts::TAU;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let columns = segments + 1;
    let max_r = profile.iter().map(|p| p.x).fold(0.0f32, f32::max).max(1e-3);
    let mut arc = 0.0;
    for (j, p) in profile.iter().enumerate() {
        if j > 0 {
            arc += p.distance(profile[j - 1]);
        }
        for s in 0..columns {
            let a = (s % segments) as f32 / segments as f32 * tau;
            let r = p.x * radius_fn(a, p.y);
            let pos = Vec3::new(a.cos() * r, p.y, a.sin() * r);
            vertices.push(Vertex::new(pos, Vec3::Y, Vec2::new(s as f32 / segments as f32 * tau * max_r, arc)));
        }
    }
    let rows = profile.len() as u32;
    for j in 0..rows - 1 {
        for s in 0..segments {
            let a = j * columns + s;
            let b = a + columns;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    if caps {
        for (row, up) in [(0u32, false), (rows - 1, true)] {
            let p = profile[row as usize];
            if p.x <= 1e-5 {
                continue;
            }
            let center = vertices.len() as u32;
            let n = if up { Vec3::Y } else { Vec3::NEG_Y };
            vertices.push(Vertex::new(Vec3::new(0.0, p.y, 0.0), n, Vec2::ZERO));
            for s in 0..segments {
                let src = vertices[(row * columns + s) as usize];
                vertices.push(Vertex::new(src.position(), n, Vec2::new(src.position[0], src.position[2])));
            }
            for s in 0..segments {
                let (i, k) = (center + 1 + s, center + 1 + (s + 1) % segments);
                if up {
                    indices.extend_from_slice(&[center, k, i]);
                } else {
                    indices.extend_from_slice(&[center, i, k]);
                }
            }
        }
    }
    let mut mesh = Mesh::new(vertices, indices);
    mesh.compute_normals_creased(50.0);
    mesh.weld();
    mesh
}

/// Sweeps a closed 2D cross-section along a 3D path using rotation-minimising frames.
/// `scale(t)` scales the section along the path (t in [0, 1]); `caps` closes both ends.
/// UV: u around the section (metres), v along the path (metres).
pub fn sweep(section: &[Vec2], path: &[Vec3], up_hint: Vec3, scale: &dyn Fn(f32) -> f32, caps: bool) -> Mesh {
    assert!(path.len() >= 2 && section.len() >= 3);
    let n = section.len();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    // Rotation-minimising frames (double reflection, Wang et al. 2008).
    let tangent = |i: usize| {
        let a = path[i.saturating_sub(1)];
        let b = path[(i + 1).min(path.len() - 1)];
        (b - a).normalize_or(Vec3::Y)
    };
    let t0 = tangent(0);
    let mut normal = up_hint.cross(t0).cross(t0).normalize_or_zero();
    if normal == Vec3::ZERO {
        normal = t0.any_orthonormal_vector();
    }
    normal = -normal;
    let mut frames = vec![(t0, normal)];
    for i in 1..path.len() {
        let (t_prev, r_prev) = frames[i - 1];
        let v1 = path[i] - path[i - 1];
        let c1 = v1.dot(v1).max(1e-12);
        let r_l = r_prev - v1 * (2.0 / c1 * v1.dot(r_prev));
        let t_l = t_prev - v1 * (2.0 / c1 * v1.dot(t_prev));
        let t_i = tangent(i);
        let v2 = t_i - t_l;
        let c2 = v2.dot(v2);
        let r_i = if c2 > 1e-12 { r_l - v2 * (2.0 / c2 * v2.dot(r_l)) } else { r_l };
        frames.push((t_i, r_i.normalize_or(r_prev)));
    }
    let perimeter: Vec<f32> = {
        let mut acc = vec![0.0];
        for i in 0..n {
            acc.push(acc[i] + section[i].distance(section[(i + 1) % n]));
        }
        acc
    };
    let mut along = 0.0;
    let total: f32 = path.windows(2).map(|w| w[0].distance(w[1])).sum::<f32>().max(1e-6);
    for (i, &p) in path.iter().enumerate() {
        if i > 0 {
            along += p.distance(path[i - 1]);
        }
        let (t, r) = frames[i];
        let b = t.cross(r);
        let s = scale(along / total);
        for k in 0..=n {
            let q = section[k % n] * s;
            let pos = p + r * q.x + b * q.y;
            vertices.push(Vertex::new(pos, Vec3::Y, Vec2::new(perimeter[k], along)));
        }
    }
    let ring = (n + 1) as u32;
    let ccw = signed_area(section) > 0.0;
    for i in 0..path.len() as u32 - 1 {
        for k in 0..n as u32 {
            let a = i * ring + k;
            let c = a + ring;
            if ccw {
                indices.extend_from_slice(&[a, a + 1, c, a + 1, c + 1, c]);
            } else {
                indices.extend_from_slice(&[a, c, a + 1, a + 1, c, c + 1]);
            }
        }
    }
    if caps {
        let tris = triangulate(section, &[]);
        for (end, sign) in [(0usize, -1.0f32), (path.len() - 1, 1.0)] {
            let (t, r) = frames[end];
            let b = t.cross(r);
            let s = scale(if end == 0 { 0.0 } else { 1.0 });
            let base = vertices.len() as u32;
            for q in section {
                let q = *q * s;
                vertices.push(Vertex::new(path[end] + r * q.x + b * q.y, t * sign, q));
            }
            for tri in &tris {
                // Section triangles are CCW in (r, b); (r, b, t) is right-handed, so they face +t.
                if sign > 0.0 {
                    indices.extend_from_slice(&[base + tri[0], base + tri[1], base + tri[2]]);
                } else {
                    indices.extend_from_slice(&[base + tri[0], base + tri[2], base + tri[1]]);
                }
            }
        }
    }
    let mut mesh = Mesh::new(vertices, indices);
    mesh.compute_normals_creased(55.0);
    mesh.weld();
    mesh
}

/// Circle section with `sides` vertices (counter-clockwise).
pub fn circle(radius: f32, sides: usize) -> Vec<Vec2> {
    (0..sides)
        .map(|i| {
            let a = i as f32 / sides as f32 * std::f32::consts::TAU;
            Vec2::new(a.cos(), a.sin()) * radius
        })
        .collect()
}

/// Generalised cylinder along `path` with per-point radii (branches, roots, ribs).
pub fn tube(path: &[Vec3], radii: &[f32], sides: usize, caps: bool) -> Mesh {
    let section = circle(1.0, sides);
    let count = path.len();
    let radius = |t: f32| {
        let x = t * (count - 1) as f32;
        let i = (x.floor() as usize).min(count - 2);
        let f = x - i as f32;
        radii[i] * (1.0 - f) + radii[i + 1] * f
    };
    sweep(&section, path, Vec3::Y, &radius, caps)
}

/// Samples a 3D arc (circle in the plane spanned by `a` and `b` around `center`) from angle
/// `start` to `end` (radians), with `segments` steps.
pub fn arc(center: Vec3, a: Vec3, b: Vec3, start: f32, end: f32, segments: usize) -> Vec<Vec3> {
    (0..=segments)
        .map(|i| {
            let t = start + (end - start) * i as f32 / segments as f32;
            center + a * t.cos() + b * t.sin()
        })
        .collect()
}

// ---- Detail ----------------------------------------------------------------------------------------------

/// Splits every triangle into four (shared edge midpoints), `levels` times. Attributes are
/// interpolated; seams stay seams.
pub fn subdivide(mesh: &Mesh, levels: u32) -> Mesh {
    let mut current = mesh.clone();
    for _ in 0..levels {
        let mut vertices = current.vertices.clone();
        let mut midpoints: crate::hash::FxHashMap<(u32, u32), u32> = crate::hash::FxHashMap::default();
        let mut mid = |a: u32, b: u32, vertices: &mut Vec<Vertex>| {
            let key = if a < b { (a, b) } else { (b, a) };
            *midpoints.entry(key).or_insert_with(|| {
                let (va, vb) = (vertices[a as usize], vertices[b as usize]);
                vertices.push(Vertex::new(
                    (va.position() + vb.position()) * 0.5,
                    (va.normal() + vb.normal()).normalize_or(va.normal()),
                    (va.uv() + vb.uv()) * 0.5,
                ));
                vertices.len() as u32 - 1
            })
        };
        let mut indices = Vec::with_capacity(current.indices.len() * 4);
        for [a, b, c] in current.triangles() {
            let ab = mid(a, b, &mut vertices);
            let bc = mid(b, c, &mut vertices);
            let ca = mid(c, a, &mut vertices);
            indices.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
        }
        current = Mesh::new(vertices, indices);
    }
    current.weld();
    current
}

/// Adaptively splits edges longer than `max_edge` until none remain (red-green refinement:
/// every long edge gets a midpoint shared by both of its triangles, and each triangle is
/// split according to how many of its edges were split, so no cracks appear). Small
/// details are left alone, large faces become evenly dense.
pub fn subdivide_to(mesh: &Mesh, max_edge: f32) -> Mesh {
    let mut current = mesh.clone();
    let canonical_key = |v: &Vertex| v.position.map(f32::to_bits);
    for _ in 0..16 {
        let positions: Vec<Vec3> = current.vertices.iter().map(Vertex::position).collect();
        let mut vertices = current.vertices.clone();
        // Midpoints are keyed by position-level edges so seams split consistently; every
        // wedge pair gets its own midpoint vertex with interpolated attributes.
        let mut splits: crate::hash::FxHashMap<(u32, u32), u32> = crate::hash::FxHashMap::default();
        let classes: crate::hash::FxHashMap<[u32; 3], u32> = {
            let mut m = crate::hash::FxHashMap::default();
            for (i, v) in current.vertices.iter().enumerate() {
                m.entry(canonical_key(v)).or_insert(i as u32);
            }
            m
        };
        let class = |i: u32| classes[&canonical_key(&current.vertices[i as usize])];
        let long = |a: u32, b: u32| positions[a as usize].distance(positions[b as usize]) > max_edge;
        let mut midpoint = |a: u32, b: u32, vertices: &mut Vec<Vertex>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            *splits.entry(key).or_insert_with(|| {
                let (va, vb) = (vertices[a as usize], vertices[b as usize]);
                // Positions must be bit-identical for both wedges of a seam edge: compute from
                // the canonical (position-class) endpoints in a fixed order.
                let (ca, cb) = (class(a), class(b));
                let (pa, pb) = if ca < cb { (positions[ca as usize], positions[cb as usize]) } else { (positions[cb as usize], positions[ca as usize]) };
                vertices.push(Vertex::new((pa + pb) * 0.5, (va.normal() + vb.normal()).normalize_or(va.normal()), (va.uv() + vb.uv()) * 0.5));
                vertices.len() as u32 - 1
            })
        };
        let mut indices = Vec::with_capacity(current.indices.len() * 2);
        let mut changed = false;
        for [a, b, c] in current.triangles() {
            let marks = [long(a, b), long(b, c), long(c, a)];
            let count = marks.iter().filter(|&&m| m).count();
            if count == 0 {
                indices.extend_from_slice(&[a, b, c]);
                continue;
            }
            changed = true;
            // Rotate so the split edges come first in a canonical arrangement.
            let (v, m) = match marks {
                [true, false, false] | [true, true, true] | [true, true, false] => ([a, b, c], marks),
                [false, true, false] | [false, true, true] => ([b, c, a], [marks[1], marks[2], marks[0]]),
                _ => ([c, a, b], [marks[2], marks[0], marks[1]]),
            };
            let [p, q, r] = v;
            match m {
                [true, false, false] => {
                    let pq = midpoint(p, q, &mut vertices);
                    indices.extend_from_slice(&[p, pq, r, pq, q, r]);
                }
                [true, true, false] => {
                    let pq = midpoint(p, q, &mut vertices);
                    let qr = midpoint(q, r, &mut vertices);
                    indices.extend_from_slice(&[pq, q, qr, p, pq, qr, p, qr, r]);
                }
                _ => {
                    let pq = midpoint(p, q, &mut vertices);
                    let qr = midpoint(q, r, &mut vertices);
                    let rp = midpoint(r, p, &mut vertices);
                    indices.extend_from_slice(&[p, pq, rp, q, qr, pq, r, rp, qr, pq, qr, rp]);
                }
            }
        }
        current = Mesh::new(vertices, indices);
        if !changed {
            break;
        }
    }
    current.weld();
    current
}

/// Displaces vertices along their normals by `amount * fbm(position * frequency)` — chisel
/// marks, weathering, bark ridges. Seams stay closed (positions are a function of position).
pub fn displace(mesh: &mut Mesh, amount: f32, frequency: f32, octaves: u32, seed: u32) {
    for v in &mut mesh.vertices {
        let p = v.position();
        let d = fbm(p * frequency, octaves, seed) * amount;
        v.position = (p + v.normal() * d).into();
    }
    mesh.compute_normals_creased(60.0);
}

/// Mesh transformed by `m` (convenience for assembling pieces).
pub fn place(mesh: &Mesh, m: Mat4) -> Mesh {
    mesh.transformed(m)
}

/// Concatenates meshes (welded).
pub fn merge(meshes: &[Mesh]) -> Mesh {
    let mut b = MeshBuilder::default();
    for m in meshes {
        b.append(m);
    }
    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_closed(mesh: &Mesh) -> bool {
        let canonical = mesh.canonical_ids();
        let mut uses: crate::hash::FxHashMap<u64, u32> = crate::hash::FxHashMap::default();
        for t in mesh.triangles() {
            let c = t.map(|v| canonical[v as usize]);
            for (a, b) in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])] {
                *uses.entry(crate::hash::edge_key(a, b)).or_insert(0) += 1;
            }
        }
        uses.values().all(|&n| n == 2)
    }

    #[test]
    fn triangulation_with_hole_covers_the_area() {
        let outline = vec![Vec2::new(0.0, 0.0), Vec2::new(4.0, 0.0), Vec2::new(4.0, 4.0), Vec2::new(0.0, 4.0)];
        let hole = vec![Vec2::new(1.0, 1.0), Vec2::new(3.0, 1.0), Vec2::new(3.0, 3.0), Vec2::new(1.0, 3.0)];
        let all: Vec<Vec2> = outline.iter().chain(&hole).copied().collect();
        let tris = triangulate(&outline, &[hole]);
        let area: f32 = tris.iter().map(|t| (all[t[1] as usize] - all[t[0] as usize]).perp_dot(all[t[2] as usize] - all[t[0] as usize]).abs() * 0.5).sum();
        assert!((area - 12.0).abs() < 1e-3, "area {area}");
    }

    #[test]
    fn extruded_arched_wall_is_closed() {
        let wall = arched_wall(6.0, 10.0, 2.5, 2.0, 6.0, 8.0, 0.8);
        assert!(is_closed(&wall), "arched wall has open edges");
    }

    #[test]
    fn cuboid_lathe_and_tube_are_closed() {
        assert!(is_closed(&cuboid(Vec3::ZERO, Vec3::ONE)));
        let profile = [Vec2::new(0.5, 0.0), Vec2::new(0.4, 1.0), Vec2::new(0.45, 2.0)];
        assert!(is_closed(&lathe(&profile, 24, &|_, _| 1.0, true)));
        let path = [Vec3::ZERO, Vec3::new(0.0, 1.0, 0.2), Vec3::new(0.3, 2.0, 0.5)];
        assert!(is_closed(&tube(&path, &[0.2, 0.15, 0.1], 8, true)));
    }

    #[test]
    fn pointed_arch_is_symmetric_and_reaches_the_apex() {
        let arch = pointed_arch(4.0, 3.0, 8);
        let apex = arch.iter().map(|p| p.y).fold(0.0f32, f32::max);
        assert!((apex - 3.0).abs() < 1e-3, "apex {apex}");
        assert!((arch[0].x + 2.0).abs() < 1e-3 && (arch.last().unwrap().x - 2.0).abs() < 1e-3);
    }
}
