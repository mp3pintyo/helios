//! Point filters: drop points, or scale their density for later density filters and pruning.

use glam::{Vec2, Vec3};

use crate::noise::fbm;
use crate::point::{Point, Volume};
use crate::rng::random;

/// Keeps points whose surface slope (angle of the normal from +Y) lies in [min, max] degrees.
pub fn slope(points: &mut Vec<Point>, min_deg: f32, max_deg: f32) {
    points.retain(|p| {
        let slope = p.normal.y.clamp(-1.0, 1.0).acos().to_degrees();
        (min_deg..=max_deg).contains(&slope)
    });
}

/// Keeps points whose world height lies in [min, max].
pub fn height(points: &mut Vec<Point>, min: f32, max: f32) {
    points.retain(|p| (min..=max).contains(&p.position.y));
}

/// Multiplies density by fractal noise fixed in the world: `scale` metres per feature,
/// `threshold` the noise level where density is halved, `softness` the width of the transition
/// (0: a hard edge between patches and gaps).
pub fn noise_density(points: &mut [Point], scale: f32, threshold: f32, softness: f32, invert: bool, salt: u32) {
    for p in points {
        let n = fbm(p.position.x, p.position.z, scale, salt);
        let n = if invert { 1.0 - n } else { n };
        let s = softness.max(1e-4);
        let t = ((n - threshold + s) / (2.0 * s)).clamp(0.0, 1.0);
        p.density *= t * t * (3.0 - 2.0 * t);
    }
}

/// Keeps points with density in [min, max]; `random`: instead keeps each point with
/// probability equal to its density (stochastic thinning, e.g. forest edges).
pub fn density(points: &mut Vec<Point>, min: f32, max: f32, random_keep: bool, salt: u32) {
    if random_keep {
        points.retain(|p| random(p.seed, salt, 0) < p.density);
    } else {
        points.retain(|p| (min..=max).contains(&p.density));
    }
}

/// Distance in the XZ plane from `p` to the polyline `line`.
fn distance_to_polyline(p: Vec2, line: &[Vec2]) -> f32 {
    match line {
        [] => f32::MAX,
        [single] => p.distance(*single),
        _ => line
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                p.distance(a + ab * t)
            })
            .fold(f32::MAX, f32::min),
    }
}

/// Distance filter against points and splines (polylines) given in volume-local metres
/// (so they move with the volume), measured in the ground plane. Keeps points between `min`
/// and `max` metres from the nearest one (`invert`: outside that band); `falloff` metres at
/// either edge of the band fade the density in smoothly instead of cutting.
pub struct DistanceTargets {
    pub points: Vec<Vec2>,
    pub splines: Vec<Vec<Vec2>>,
}

impl DistanceTargets {
    pub fn new(volume: &Volume, points: &[[f32; 3]], splines: &[Vec<[f32; 3]>]) -> Self {
        let to_world = |p: &[f32; 3]| {
            let w = volume.to_world(Vec3::from(*p));
            Vec2::new(w.x, w.z)
        };
        Self { points: points.iter().map(to_world).collect(), splines: splines.iter().map(|s| s.iter().map(to_world).collect()).collect() }
    }

    pub fn distance(&self, p: Vec3) -> f32 {
        let q = Vec2::new(p.x, p.z);
        let to_points = self.points.iter().map(|t| q.distance(*t)).fold(f32::MAX, f32::min);
        self.splines.iter().map(|s| distance_to_polyline(q, s)).fold(to_points, f32::min)
    }
}

pub fn distance(points: &mut Vec<Point>, targets: &DistanceTargets, min: f32, max: f32, falloff: f32, invert: bool) {
    points.retain_mut(|p| {
        let d = targets.distance(p.position);
        let inside = (min..=max).contains(&d);
        if inside == invert {
            return false;
        }
        if falloff > 0.0 {
            let ramp = if invert {
                // Outside the band: fade in moving away from it.
                (if d < min { (min - d) / falloff } else { (d - max) / falloff }).clamp(0.0, 1.0)
            } else {
                ((d - min) / falloff).min((max - d) / falloff).clamp(0.0, 1.0)
            };
            p.density *= ramp * ramp * (3.0 - 2.0 * ramp);
        }
        true
    });
}

/// Keeps points inside the box `[min, max]` in the volume's normalised coordinates ([-1, 1]
/// spans the volume); `invert` keeps those outside.
pub fn bounds(points: &mut Vec<Point>, volume: &Volume, min: [f32; 3], max: [f32; 3], invert: bool) {
    let (min, max) = (Vec3::from(min), Vec3::from(max));
    points.retain(|p| {
        let n = volume.normalized(p.position);
        (n.cmpge(min).all() && n.cmple(max).all()) != invert
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f32, z: f32, normal: Vec3) -> Point {
        Point::new(Vec3::new(x, 0.0, z), normal.normalize(), (x * 100.0 + z) as u32)
    }

    #[test]
    fn slope_keeps_the_requested_angles() {
        let mut points = vec![at(0.0, 0.0, Vec3::Y), at(1.0, 0.0, Vec3::new(1.0, 1.0, 0.0)), at(2.0, 0.0, Vec3::X)];
        slope(&mut points, 30.0, 60.0);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].position.x, 1.0);
    }

    #[test]
    fn distance_band_with_falloff_fades_density() {
        let volume = Volume { center: [0.0; 3], half_extents: [50.0; 3], yaw_deg: 0.0 };
        let targets = DistanceTargets::new(&volume, &[], &[vec![[-50.0, 0.0, 0.0], [50.0, 0.0, 0.0]]]);
        let mut points: Vec<Point> = (0..40).map(|i| at(0.0, i as f32, Vec3::Y)).collect();
        distance(&mut points, &targets, 0.0, 20.0, 5.0, false);
        assert_eq!(points.len(), 21);
        assert!(points[10].density == 1.0 && points[18].density < 0.5 && points[18].density > 0.0);
    }

    #[test]
    fn bounds_select_a_sub_box() {
        let volume = Volume { center: [0.0; 3], half_extents: [10.0; 3], yaw_deg: 0.0 };
        let mut points: Vec<Point> = (-9..10).map(|i| at(i as f32, 0.0, Vec3::Y)).collect();
        bounds(&mut points, &volume, [0.0, -1.0, -1.0], [1.0, 1.0, 1.0], false);
        assert!(points.iter().all(|p| p.position.x >= 0.0) && points.len() == 10);
    }
}
