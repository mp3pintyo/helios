//! Demo motion: props that move every frame (swinging chandeliers, a drifting raft, a turning
//! sculpture, a lantern carried down the nave), to show the renderer's dynamic world — moved instances update only what they
//! touch (see `helios_render::Mobility`). `Digit0` pauses and resumes it.

use glam::{Mat4, Quat, Vec3};
use helios_render::{InstanceHandle, LightHandle, Mobility, Scene};

/// A rigid motion in world space as a function of time.
#[derive(Clone, Copy, Debug)]
pub enum Motion {
    /// Swings about `pivot` around the horizontal `axis` by up to `amplitude` radians, with the
    /// period of a simple pendulum of length `length` (metres).
    Pendulum { pivot: Vec3, axis: Vec3, amplitude: f32, length: f32, phase: f32 },
    /// Turns about the vertical axis through `center`, one turn per `period` seconds.
    Spin { center: Vec3, period: f32 },
    /// Floats on water around `center`: drifts on a slow loop of radius `drift`, turns with it,
    /// bobs and rolls on the waves.
    Float { center: Vec3, drift: f32, period: f32 },
    /// Carried at walking pace from `from` to `to` and back (`period` seconds for the round
    /// trip, easing in and out at the ends), starting `phase` of the way into the trip, and
    /// swinging about the hand holding it (`from` in the rest pose) by up to `sway` radians with
    /// the steps.
    Carry { from: Vec3, to: Vec3, period: f32, phase: f32, sway: f32 },
}

impl Motion {
    /// The world transform taking the rest pose to the pose at time `t`.
    pub fn at(&self, t: f32) -> Mat4 {
        use std::f32::consts::TAU;
        let about = |center: Vec3, rotation: Quat| Mat4::from_translation(center) * Mat4::from_quat(rotation) * Mat4::from_translation(-center);
        match *self {
            Motion::Pendulum { pivot, axis, amplitude, length, phase } => {
                let period = TAU * (length / 9.81).sqrt();
                about(pivot, Quat::from_axis_angle(axis.normalize(), amplitude * (TAU * t / period + phase).sin()))
            }
            Motion::Spin { center, period } => about(center, Quat::from_rotation_y(TAU * t / period)),
            Motion::Float { center, drift, period } => {
                let a = TAU * t / period;
                let offset = Vec3::new(drift * a.cos(), 0.04 * (1.9 * t).sin(), drift * a.sin());
                let rotation = Quat::from_rotation_y(0.35 * a.sin()) * Quat::from_rotation_x(0.025 * (1.3 * t).sin()) * Quat::from_rotation_z(0.03 * (1.7 * t + 1.0).sin());
                Mat4::from_translation(offset) * about(center, rotation)
            }
            Motion::Carry { from, to, period, phase, sway } => {
                let a = TAU * (t / period + phase);
                // Two steps a second: the hand bobs, the lantern swings across the walk.
                let step = TAU * t;
                let along = (to - from) * (0.5 - 0.5 * a.cos()) + Vec3::Y * (0.012 * (2.0 * step).sin());
                let across = (to - from).cross(Vec3::Y).normalize_or_zero();
                let swing = Quat::from_axis_angle((to - from).normalize_or_zero(), sway * step.sin()) * Quat::from_axis_angle(across, 0.5 * sway * (step + 1.0).sin());
                Mat4::from_translation(along) * about(from, swing)
            }
        }
    }
}

/// Instances and lights moved together by one [`Motion`].
struct Group {
    motion: Motion,
    /// Each instance with its rest transform.
    instances: Vec<(InstanceHandle, Mat4)>,
    /// Each light with its rest position.
    lights: Vec<(LightHandle, Vec3)>,
}

/// A scene's moving props.
pub struct DemoMotion {
    groups: Vec<Group>,
    time: f32,
    pub enabled: bool,
}

impl Default for DemoMotion {
    fn default() -> Self {
        Self { groups: Vec::new(), time: 0.0, enabled: true }
    }
}

impl DemoMotion {
    /// Moves `instances` (made [`Mobility::Movable`]) and `lights` by `motion` from their current
    /// placement. Returns the group's index (for [`DemoMotion::attach_lights`]).
    pub fn add(&mut self, scene: &mut Scene, motion: Motion, instances: &[InstanceHandle], lights: &[LightHandle]) -> usize {
        for &h in instances {
            scene.set_mobility(h, Mobility::Movable);
        }
        self.groups.push(Group { motion, instances: instances.iter().map(|&h| (h, scene.transform(h))).collect(), lights: Vec::new() });
        let group = self.groups.len() - 1;
        self.attach_lights(scene, group, lights);
        group
    }

    /// Adds `lights` to group `group`: they move with it from their current positions (lights
    /// created after the group's instances, e.g. one per candle of a swinging chandelier).
    pub fn attach_lights(&mut self, scene: &mut Scene, group: usize, lights: &[LightHandle]) {
        let group = &mut self.groups[group];
        // Rest positions: undo the motion so far (none before the first update).
        let to_rest = group.motion.at(self.time).inverse();
        group.lights.extend(lights.iter().map(|&l| (l, to_rest.transform_point3(scene.light_mut(l).position))));
    }

    /// Makes the group added last the first (for [`DemoMotion::limit`]); call once no group
    /// index from [`DemoMotion::add`] is used any more.
    pub fn move_last_first(&mut self) {
        self.groups.rotate_right(1);
    }

    /// Keeps the first `count` groups moving; the others stay where they are, static again
    /// (`--movers`, for measurements).
    pub fn limit(&mut self, scene: &mut Scene, count: usize) {
        for group in self.groups.drain(count.min(self.groups.len())..) {
            for (h, _) in group.instances {
                scene.set_mobility(h, Mobility::Static);
            }
        }
    }

    /// Gives every moving instance `mobility` (`--static-motion` moves them as `Static` ones, the
    /// path of one-off edits, for measurements).
    pub fn set_mobility(&self, scene: &mut Scene, mobility: Mobility) {
        for group in &self.groups {
            for &(h, _) in &group.instances {
                scene.set_mobility(h, mobility);
            }
        }
    }

    /// Instances it moves.
    pub fn instance_count(&self) -> usize {
        self.groups.iter().map(|g| g.instances.len()).sum()
    }

    /// Advances the clock by `dt` (while enabled) and places everything.
    pub fn update(&mut self, scene: &mut Scene, dt: f32) {
        if !self.enabled {
            return;
        }
        self.time += dt;
        for group in &self.groups {
            let m = group.motion.at(self.time);
            for &(h, rest) in &group.instances {
                scene.set_transform(h, m * rest);
            }
            for &(l, rest) in &group.lights {
                scene.light_mut(l).position = m.transform_point3(rest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motions_start_at_rest_and_are_rigid() {
        let motions = [
            Motion::Pendulum { pivot: Vec3::new(1.0, 20.0, 0.0), axis: Vec3::Z, amplitude: 0.05, length: 12.0, phase: 0.0 },
            Motion::Spin { center: Vec3::new(0.0, 5.0, 0.0), period: 40.0 },
            Motion::Float { center: Vec3::new(-6.0, -2.2, 23.0), drift: 1.5, period: 60.0 },
            Motion::Carry { from: Vec3::new(8.0, 0.9, 0.5), to: Vec3::new(30.0, 0.9, 0.5), period: 44.0, phase: 0.0, sway: 0.2 },
        ];
        for motion in motions {
            // Rigid: lengths are kept.
            let m = motion.at(3.7);
            let (a, b) = (Vec3::new(0.3, 1.0, -2.0), Vec3::new(4.0, -1.0, 0.5));
            assert!((m.transform_point3(a).distance(m.transform_point3(b)) - a.distance(b)).abs() < 1e-3);
        }
        // A pendulum's pivot stays put and it swings no further than its amplitude.
        let pendulum = motions[0];
        for t in [0.5, 1.7, 4.2] {
            assert!(pendulum.at(t).transform_point3(Vec3::new(1.0, 20.0, 0.0)).distance(Vec3::new(1.0, 20.0, 0.0)) < 1e-4);
            assert!(pendulum.at(t).transform_point3(Vec3::new(1.0, 8.0, 0.0)).distance(Vec3::new(1.0, 8.0, 0.0)) <= 12.0 * 0.05 + 1e-3);
        }
        assert!(motions[1].at(0.0).abs_diff_eq(Mat4::IDENTITY, 1e-5) && motions[0].at(0.0).abs_diff_eq(Mat4::IDENTITY, 1e-5));
        // A carried lantern: the hand goes to the far end and back, within a few centimetres
        // of the path, and the lantern hangs no further than its sway from the hand.
        let carry = motions[3];
        let hand = Vec3::new(8.0, 0.9, 0.5);
        assert!(carry.at(22.0).transform_point3(hand).distance(Vec3::new(30.0, 0.9, 0.5)) < 0.05);
        assert!(carry.at(44.0).transform_point3(hand).distance(hand) < 0.05);
        let below = hand - Vec3::Y * 0.3;
        for t in [3.3, 11.1, 27.9] {
            let m = carry.at(t);
            assert!(m.transform_point3(below).distance(m.transform_point3(hand)) - 0.3 < 1e-4);
            assert!(m.transform_point3(below).y < m.transform_point3(hand).y - 0.3 * 0.95);
        }
    }

    #[test]
    fn lights_attached_later_move_with_their_group() {
        let mut scene = Scene::new();
        let light = |scene: &mut Scene| scene.add_light(helios_render::Light::point(Vec3::new(1.0, 8.0, 0.0), Vec3::ONE, 1.0, 5.0));
        let first = light(&mut scene);
        let mut motion = DemoMotion::default();
        let pendulum = Motion::Pendulum { pivot: Vec3::new(1.0, 20.0, 0.0), axis: Vec3::Z, amplitude: 0.05, length: 12.0, phase: 0.3 };
        let group = motion.add(&mut scene, pendulum, &[], &[first]);
        motion.update(&mut scene, 1.1);
        // Attached mid-swing at the first light's current position: they move as one.
        let second = light(&mut scene);
        let swung = scene.light_mut(first).position;
        scene.light_mut(second).position = swung;
        motion.attach_lights(&mut scene, group, &[second]);
        motion.update(&mut scene, 0.7);
        assert!(scene.light_mut(first).position.distance(scene.light_mut(second).position) < 1e-4);
        assert!(scene.light_mut(first).position.distance(swung) > 1e-3, "still swinging");
    }
}
