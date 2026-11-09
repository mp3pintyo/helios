//! Cinematic camera tours: a looping Catmull–Rom path of (eye, target) keyframes per scene.
//! Heights are relative to the ground below each point, so paths follow the terrain.

use glam::Vec3;

use crate::scenes::SceneKind;

/// Seconds spent between two keyframes.
const SEGMENT_SECONDS: f32 = 7.0;

pub struct Tour {
    /// (eye, look-at target) with y relative to the ground.
    keys: Vec<(Vec3, Vec3)>,
}

impl Tour {
    pub fn for_scene(kind: SceneKind) -> Option<Tour> {
        let k = |e: [f32; 3], t: [f32; 3]| (Vec3::from(e), Vec3::from(t));
        let keys = match kind {
            SceneKind::Valley => vec![
                k([-22.0, 6.0, 30.0], [0.0, 6.0, 0.0]),
                k([22.0, 3.0, 24.0], [0.0, 6.0, 0.0]),
                k([30.0, 7.0, -18.0], [0.0, 5.0, 0.0]),
                k([-12.0, 22.0, -38.0], [0.0, 4.0, 0.0]),
                k([-90.0, 55.0, 30.0], [120.0, 25.0, -110.0]),
                k([-45.0, 9.0, 45.0], [0.0, 6.0, 0.0]),
            ],
            SceneKind::Cathedral => vec![
                // Down the central aisle, up into the vaults, back through the south aisle.
                k([2.0, 1.8, 0.0], [40.0, 6.0, 0.0]),
                k([20.0, 3.0, 0.0], [40.0, 12.0, 3.0]),
                k([38.0, 9.0, 2.0], [60.0, 20.0, -2.0]),
                k([56.0, 14.0, 0.0], [80.0, 16.0, 0.0]),
                k([44.0, 20.0, 0.0], [72.0, 2.0, 0.0]),
                k([24.0, 9.0, -10.0], [52.0, 4.0, -10.0]),
                k([6.0, 3.0, -4.0], [30.0, 3.0, 0.0]),
            ],
            SceneKind::Forest => vec![
                k([0.0, 1.7, 0.0], [30.0, 4.0, -8.0]),
                k([20.0, 1.7, -6.0], [50.0, 3.0, -15.0]),
                k([45.0, 2.2, -14.0], [70.0, 6.0, 10.0]),
                k([60.0, 9.0, 10.0], [30.0, 2.0, 30.0]),
                k([30.0, 1.7, 25.0], [0.0, 2.0, 0.0]),
                k([6.0, 1.7, 8.0], [30.0, 4.0, -8.0]),
            ],
            // Along the row of plinths, close up, then around the back into the low sun.
            SceneKind::Gallery => vec![
                k([-12.0, 2.2, 5.0], [-11.2, 1.4, 0.0]),
                k([-7.0, 1.9, 3.2], [-5.2, 1.3, 0.0]),
                k([-1.0, 1.9, 3.0], [1.6, 1.4, 0.0]),
                k([4.5, 1.9, 3.2], [7.0, 1.3, 0.0]),
                k([11.0, 2.4, 6.0], [9.6, 0.6, 3.0]),
                k([6.0, 2.2, -5.0], [1.6, 1.4, 0.0]),
                k([0.0, 3.0, 12.5], [0.0, 1.2, 0.0]),
            ],
            _ => return None,
        };
        Some(Tour { keys })
    }

    /// Length of one loop in seconds.
    pub fn duration(&self) -> f32 {
        self.keys.len() as f32 * SEGMENT_SECONDS
    }

    /// Eye and target at `time` seconds (looping), in world space.
    pub fn sample(&self, time: f32, ground: impl Fn(f32, f32) -> f32) -> (Vec3, Vec3) {
        let n = self.keys.len();
        let t = time.rem_euclid(self.duration()) / SEGMENT_SECONDS;
        let i = t.floor() as usize;
        let f = t - i as f32;
        let world = |p: Vec3| Vec3::new(p.x, p.y + ground(p.x, p.z), p.z);
        let point = |j: usize, pick: fn(&(Vec3, Vec3)) -> Vec3| world(pick(&self.keys[(i + j + n - 1) % n]));
        let spline = |pick: fn(&(Vec3, Vec3)) -> Vec3| catmull_rom(point(0, pick), point(1, pick), point(2, pick), point(3, pick), f);
        (spline(|k| k.0), spline(|k| k.1))
    }
}

/// Uniform Catmull–Rom between `p1` and `p2`.
fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1) + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tour_passes_through_keyframes_and_loops() {
        let tour = Tour::for_scene(SceneKind::Cathedral).unwrap();
        let flat = |_: f32, _: f32| 0.0;
        for (i, key) in tour.keys.iter().enumerate() {
            let (eye, target) = tour.sample(i as f32 * SEGMENT_SECONDS, flat);
            assert!(eye.distance(key.0) < 1e-3 && target.distance(key.1) < 1e-3);
        }
        let (a, _) = tour.sample(0.0, flat);
        let (b, _) = tour.sample(tour.duration(), flat);
        assert!(a.distance(b) < 1e-3);
    }
}
