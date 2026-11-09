//! Surfaces points are sampled on (UE's Surface Sampler): a landscape heightfield or any
//! triangle mesh.

use glam::Vec3;

use crate::point::{Point, Volume};
use crate::rng::{hash_all, unit};
use crate::MAX_POINTS;

pub trait Surface {
    /// Points on the surface inside `volume`, about `points_per_m2` per square metre of
    /// surface; `jitter` 0 places them on a regular grid, 1 anywhere in their cell. Fails when
    /// it would create more than [`MAX_POINTS`].
    fn sample(&self, volume: &Volume, points_per_m2: f32, jitter: f32, seed: u32) -> Result<Vec<Point>, String>;
}

fn too_many(count: usize) -> String {
    format!("{count} points exceed the sampler's limit of {MAX_POINTS}: lower the density or shrink the volume")
}

/// A landscape given by its height at (x, z).
pub struct Heightfield<F: Fn(f32, f32) -> f32> {
    pub height: F,
}

impl<F: Fn(f32, f32) -> f32> Heightfield<F> {
    pub fn normal(&self, x: f32, z: f32) -> Vec3 {
        let e = 0.25;
        let dx = (self.height)(x + e, z) - (self.height)(x - e, z);
        let dz = (self.height)(x, z + e) - (self.height)(x, z - e);
        Vec3::new(-dx, 2.0 * e, -dz).normalize()
    }
}

impl<F: Fn(f32, f32) -> f32> Surface for Heightfield<F> {
    /// One candidate per cell of a square grid fixed in the world (cell area 1 / density), so
    /// a point's seed and position depend only on its cell: moving or resizing the volume keeps
    /// every point in the area it still covers.
    fn sample(&self, volume: &Volume, points_per_m2: f32, jitter: f32, seed: u32) -> Result<Vec<Point>, String> {
        if points_per_m2 <= 0.0 {
            return Ok(Vec::new());
        }
        let cell = 1.0 / points_per_m2.sqrt();
        let (lo, hi) = volume.world_bounds();
        let (i0, i1) = ((lo.x / cell).floor() as i64, (hi.x / cell).ceil() as i64);
        let (k0, k1) = ((lo.z / cell).floor() as i64, (hi.z / cell).ceil() as i64);
        let cells = ((i1 - i0).max(0) * (k1 - k0).max(0)) as usize;
        if cells > MAX_POINTS * 4 {
            return Err(too_many(cells));
        }
        let jitter = jitter.clamp(0.0, 1.0);
        let mut points = Vec::new();
        for k in k0..k1 {
            for i in i0..i1 {
                let h = hash_all(&[i as u32, k as u32, seed]);
                let offset = |stream: u32| 0.5 + jitter * (unit(hash_all(&[h, stream])) - 0.5);
                let (x, z) = ((i as f32 + offset(1)) * cell, (k as f32 + offset(2)) * cell);
                let p = Vec3::new(x, (self.height)(x, z), z);
                if !volume.contains(p) {
                    continue;
                }
                if points.len() == MAX_POINTS {
                    return Err(too_many(cells));
                }
                points.push(Point::new(p, self.normal(x, z), h));
            }
        }
        Ok(points)
    }
}

/// World-space triangles (e.g. a mesh instance's finest level of detail).
pub struct TriangleSurface {
    pub triangles: Vec<[Vec3; 3]>,
}

impl Surface for TriangleSurface {
    /// Each triangle gets `area × density` points (the fraction rounded stochastically by its
    /// own seed) at uniformly random barycentric positions; seeds come from the triangle index.
    fn sample(&self, volume: &Volume, points_per_m2: f32, _jitter: f32, seed: u32) -> Result<Vec<Point>, String> {
        let mut points = Vec::new();
        if points_per_m2 <= 0.0 {
            return Ok(points);
        }
        let (lo, hi) = volume.world_bounds();
        for (t, [a, b, c]) in self.triangles.iter().enumerate() {
            let (tlo, thi) = (a.min(*b).min(*c), a.max(*b).max(*c));
            if tlo.cmpgt(hi).any() || thi.cmplt(lo).any() {
                continue;
            }
            let cross = (*b - *a).cross(*c - *a);
            let area = cross.length() * 0.5;
            if area <= 0.0 {
                continue;
            }
            let normal = cross.normalize();
            let h = hash_all(&[t as u32, seed]);
            let expected = area * points_per_m2;
            let count = expected.floor() as u32 + u32::from(unit(h) < expected.fract());
            for m in 0..count {
                let ps = hash_all(&[h, m]);
                let (mut u, mut v) = (unit(hash_all(&[ps, 1])), unit(hash_all(&[ps, 2])));
                if u + v > 1.0 {
                    (u, v) = (1.0 - u, 1.0 - v);
                }
                let p = *a + (*b - *a) * u + (*c - *a) * v;
                if !volume.contains(p) {
                    continue;
                }
                if points.len() == MAX_POINTS {
                    return Err(too_many(points.len() + 1));
                }
                points.push(Point::new(p, normal, ps));
            }
        }
        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heightfield_density_matches_the_area() {
        let surface = Heightfield { height: |_x: f32, _z: f32| 0.0 };
        let volume = Volume { center: [3.0, 0.0, -7.0], half_extents: [20.0, 5.0, 10.0], yaw_deg: 30.0 };
        let points = surface.sample(&volume, 2.0, 1.0, 1).unwrap();
        let expected = 40.0 * 20.0 * 2.0;
        assert!((points.len() as f32 - expected).abs() < expected * 0.03, "{} vs {expected}", points.len());
        assert!(points.iter().all(|p| volume.contains(p.position)));
    }

    #[test]
    fn triangle_density_matches_the_area() {
        // A 10 m × 10 m quad standing upright.
        let q = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 10.0, 0.0)];
        let surface = TriangleSurface { triangles: vec![[q[0], q[1], q[2]], [q[0], q[2], q[3]]] };
        let volume = Volume { center: [5.0, 5.0, 0.0], half_extents: [6.0, 6.0, 1.0], yaw_deg: 0.0 };
        let points = surface.sample(&volume, 3.0, 1.0, 9).unwrap();
        assert!((points.len() as f32 - 300.0).abs() < 4.0, "{}", points.len());
        assert!(points.iter().all(|p| p.position.z == 0.0 && p.normal.z.abs() > 0.99));
    }

    #[test]
    fn oversized_requests_fail_instead_of_freezing() {
        let surface = Heightfield { height: |_x: f32, _z: f32| 0.0 };
        let volume = Volume { center: [0.0; 3], half_extents: [2000.0, 10.0, 2000.0], yaw_deg: 0.0 };
        assert!(surface.sample(&volume, 50.0, 1.0, 1).is_err());
    }
}
