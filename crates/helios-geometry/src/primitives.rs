//! Procedural, welded, high-density meshes for testing and demos.

use glam::{Vec2, Vec3};

use crate::hash::FxHashMap;
use crate::noise::{fbm, ridged};
use crate::{Mesh, Vertex};

/// Unit icosphere with `20 * 4^subdivisions` triangles.
pub fn icosphere(subdivisions: u32) -> Mesh {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut positions: Vec<Vec3> = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|&p| Vec3::from(p).normalize())
    .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        let mut midpoint: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut mid = |a: u32, b: u32, positions: &mut Vec<Vec3>| {
            let key = if a < b { (a, b) } else { (b, a) };
            *midpoint.entry(key).or_insert_with(|| {
                positions.push(((positions[a as usize] + positions[b as usize]) * 0.5).normalize());
                (positions.len() - 1) as u32
            })
        };
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let ab = mid(a, b, &mut positions);
            let bc = mid(b, c, &mut positions);
            let ca = mid(c, a, &mut positions);
            next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }

    Mesh::from_positions(&positions, tris.into_iter().flatten().collect())
}

/// A weathered boulder: an icosphere displaced by layered ridged/fBm noise with strata.
pub fn rock(seed: u32, subdivisions: u32) -> Mesh {
    let mut mesh = icosphere(subdivisions);
    let s = seed as f32 * 17.13;
    let offset = Vec3::new(s, s * 0.37, s * 0.71);
    let stretch = Vec3::new(1.0 + 0.25 * fbm(offset, 2, seed), 0.75, 1.0 + 0.25 * fbm(-offset, 2, seed + 1));

    for v in &mut mesh.vertices {
        let dir = v.position();
        let p = dir + offset;
        let large = 0.28 * fbm(p * 0.9, 4, seed);
        let crags = 0.16 * ridged(p * 2.3, 5, seed + 7);
        let detail = 0.025 * fbm(p * 11.0, 4, seed + 13);
        let strata = 0.012 * ((dir.y * 38.0 + 6.0 * fbm(p * 1.7, 2, seed + 3)).sin());
        let mut pos = dir * (1.0 + large + crags + detail + strata) * stretch;
        // Settle the bottom so it rests on the ground.
        if pos.y < -0.35 {
            pos.y = -0.35 + (pos.y + 0.35) * 0.25;
        }
        v.position = pos.into();
    }
    mesh.compute_normals();
    mesh
}

/// Terrain heightfield of `size × size` world units with `resolution²` quads, centred at the origin.
pub fn terrain(size: f32, resolution: u32, height: f32, seed: u32) -> Mesh {
    let n = resolution + 1;
    let mut vertices = Vec::with_capacity((n * n) as usize);
    for z in 0..n {
        for x in 0..n {
            let uv = Vec2::new(x as f32, z as f32) / resolution as f32;
            let xz = (uv - 0.5) * size;
            let y = terrain_height(xz, size, height, seed);
            vertices.push(Vertex::new(Vec3::new(xz.x, y, xz.y), Vec3::Y, uv));
        }
    }
    let mut indices = Vec::with_capacity((resolution * resolution * 6) as usize);
    for z in 0..resolution {
        for x in 0..resolution {
            let i = z * n + x;
            indices.extend_from_slice(&[i, i + n, i + 1, i + 1, i + n, i + n + 1]);
        }
    }
    let mut mesh = Mesh::new(vertices, indices);
    mesh.compute_normals();
    mesh
}

/// Height of [`terrain`] at world position `xz` (for placing objects on the ground).
pub fn terrain_height(xz: Vec2, size: f32, height: f32, seed: u32) -> f32 {
    let p = Vec3::new(xz.x, 0.0, xz.y) / size * 4.0;
    let mountains = ridged(p * 1.3, 7, seed).powf(1.6);
    let hills = 0.5 + 0.5 * fbm(p * 2.0 + Vec3::splat(3.1), 5, seed + 5);
    // Flatten a valley in the middle where the hero assets sit.
    let r = (xz.length() / (size * 0.5)).clamp(0.0, 1.0);
    let valley = smoothstep(0.08, 0.55, r);
    height * (0.12 * hills + valley * mountains) - height * 0.06
}
/// A fluted (p, q) torus knot sculpture — closed and welded.
pub fn torus_knot(p: u32, q: u32, segments: u32, sides: u32, tube_radius: f32) -> Mesh {
    let curve = |t: f32| {
        let (p, q) = (p as f32, q as f32);
        let r = 2.0 + (q * t).cos();
        Vec3::new(r * (p * t).cos(), (q * t).sin() * 1.2, r * (p * t).sin()) * 0.5
    };
    let tau = std::f32::consts::TAU;
    let mut positions = Vec::with_capacity((segments * sides) as usize);
    for i in 0..segments {
        let t = i as f32 / segments as f32 * tau;
        let c = curve(t);
        let tangent = (curve(t + 1e-3) - curve(t - 1e-3)).normalize();
        let normal = tangent.cross(Vec3::Y).normalize_or(Vec3::X);
        let binormal = tangent.cross(normal);
        for j in 0..sides {
            let a = j as f32 / sides as f32 * tau;
            let flutes = 1.0 + 0.07 * (a * 14.0 + t * 3.0).cos() + 0.03 * (t * 160.0).sin();
            let offset = (normal * a.cos() + binormal * a.sin()) * tube_radius * flutes;
            positions.push(c + offset);
        }
    }
    let mut indices = Vec::with_capacity((segments * sides * 6) as usize);
    for i in 0..segments {
        for j in 0..sides {
            let a = i * sides + j;
            let b = ((i + 1) % segments) * sides + j;
            let c = ((i + 1) % segments) * sides + (j + 1) % sides;
            let d = i * sides + (j + 1) % sides;
            indices.extend_from_slice(&[a, d, b, b, d, c]);
        }
    }
    Mesh::from_positions(&positions, indices)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// UV-mapped cylinder around +Y (from y = 0 to `height`), with an explicit UV seam at
/// u = 0/1 and optional flat caps (which form hard edges).
pub fn cylinder(segments: u32, rings: u32, radius: f32, height: f32, caps: bool) -> Mesh {
    let tau = std::f32::consts::TAU;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let columns = segments + 1; // the last column duplicates the first with u = 1
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        for s in 0..columns {
            let u = s as f32 / segments as f32;
            let a = (s % segments) as f32 / segments as f32 * tau;
            let n = Vec3::new(a.cos(), 0.0, a.sin());
            vertices.push(Vertex::new(n * radius + Vec3::Y * (v * height), n, Vec2::new(u, v * height / (tau * radius))));
        }
    }
    for r in 0..rings {
        for s in 0..segments {
            let a = r * columns + s;
            let b = a + columns;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    if caps {
        for (y, up) in [(0.0, -1.0f32), (height, 1.0)] {
            let center = vertices.len() as u32;
            let n = Vec3::Y * up;
            vertices.push(Vertex::new(Vec3::Y * y, n, Vec2::splat(0.5)));
            for s in 0..segments {
                let a = s as f32 / segments as f32 * tau;
                let p = Vec3::new(a.cos() * radius, y, a.sin() * radius);
                vertices.push(Vertex::new(p, n, Vec2::new(0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin())));
            }
            for s in 0..segments {
                let (i, j) = (center + 1 + s, center + 1 + (s + 1) % segments);
                if up > 0.0 {
                    indices.extend_from_slice(&[center, j, i]);
                } else {
                    indices.extend_from_slice(&[center, i, j]);
                }
            }
        }
    }
    let mut mesh = Mesh::new(vertices, indices);
    mesh.weld();
    mesh
}

/// `count` small, disconnected diamond-shaped leaves scattered in a unit ball — aggregate
/// geometry for testing component pruning.
pub fn leaf_cloud(count: u32, seed: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut rng = seed.wrapping_mul(0x9e37_79b9) | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng >> 8) as f32 / (1u32 << 24) as f32
    };
    for _ in 0..count {
        let center = Vec3::new(next(), next(), next()) * 2.0 - 1.0;
        let size = 0.04 + 0.04 * next();
        let dir = Vec3::new(next() - 0.5, next() * 0.3, next() - 0.5).normalize_or(Vec3::X);
        let side = dir.cross(Vec3::Y).normalize_or(Vec3::Z);
        let normal = side.cross(dir).normalize();
        let base = vertices.len() as u32;
        for (p, uv) in [
            (center, Vec2::new(0.5, 0.0)),
            (center + dir * size + side * size * 0.4, Vec2::new(1.0, 0.5)),
            (center + dir * size * 2.0, Vec2::new(0.5, 1.0)),
            (center + dir * size - side * size * 0.4, Vec2::new(0.0, 0.5)),
            (center + dir * size + normal * size * 0.15, Vec2::new(0.5, 0.5)),
        ] {
            vertices.push(Vertex::new(p, normal, uv));
        }
        indices.extend_from_slice(&[base, base + 1, base + 4, base + 1, base + 2, base + 4, base + 2, base + 3, base + 4, base + 3, base, base + 4]);
    }
    let mut mesh = Mesh::new(vertices, indices);
    mesh.compute_normals();
    mesh
}
