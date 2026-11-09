//! Skeletal animation: glTF node hierarchies, skins and animation clips, sampled into joint
//! matrices for the renderer's skinning.
//!
//! An [`AnimationClip`] animates node transforms (translation, rotation, scale) through keyframe
//! channels with glTF's three interpolations: `STEP`, `LINEAR` (spherical for rotations) and
//! `CUBICSPLINE` (Hermite with per-key tangents). An [`AnimationPlayer`] plays one clip — speed,
//! looping, pause — and crossfades to the next. Evaluating a frame is: rest pose → sampled (and
//! blended) local transforms ([`AnimationPlayer::pose`]) → global matrices down the hierarchy
//! ([`global_matrices`]) → per skin `global(joint) · inverseBindMatrix(joint)`
//! ([`Skin::joint_matrices`]), the palette that linear blend skinning applies to rest vertices.

use glam::{Mat4, Quat, Vec3, Vec4};

/// A node's local transform as translation, rotation and scale (glTF's TRS; matrices are
/// decomposed on import).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for NodeTransform {
    fn default() -> Self {
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl NodeTransform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// `a` blended towards `b` by `t`: linear for translation and scale, shortest-path spherical
    /// for rotation.
    pub fn blend(a: &Self, b: &Self, t: f32) -> Self {
        Self { translation: a.translation.lerp(b.translation, t), rotation: slerp(a.rotation, b.rotation, t), scale: a.scale.lerp(b.scale, t) }
    }
}

/// A node of the scene graph with its rest (unanimated) transform.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub transform: NodeTransform,
}

/// A glTF skin: the joints (node indices) that deform the meshes bound to it, and the inverse
/// bind matrices that take mesh-space rest positions into each joint's space.
#[derive(Clone, Debug, PartialEq)]
pub struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
    /// The skeleton's root node, if the file names one (informational).
    pub skeleton: Option<usize>,
}

impl Skin {
    /// The skinning palette for `globals` ([`global_matrices`]): per joint
    /// `inverse(bind) · global(joint) · inverseBindMatrix(joint)`, which maps mesh-space rest
    /// positions to posed positions in the space of `bind` (glTF skins pose into the scene's world
    /// space and ignore the mesh node's transform; expressing the palette relative to the mesh
    /// node's world transform keeps it near the identity, so bounds and LOD computed from the rest
    /// pose stay meaningful, while `bind · palette` is exactly the glTF pose).
    pub fn joint_matrices(&self, globals: &[Mat4], bind: Mat4) -> Vec<Mat4> {
        let to_bind = bind.inverse();
        self.joints.iter().zip(&self.inverse_bind_matrices).map(|(&joint, ibm)| to_bind * globals[joint] * *ibm).collect()
    }
}

/// Which node property a channel animates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Property {
    Translation,
    Rotation,
    Scale,
}

/// Keyframe interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// The previous key's value until the next key.
    Step,
    /// Linear; spherical linear for rotations.
    Linear,
    /// Cubic Hermite spline: each key stores an in-tangent, a value and an out-tangent.
    CubicSpline,
}

/// One animated property of one node.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub node: usize,
    pub property: Property,
    pub interpolation: Interpolation,
    /// Key times in seconds, increasing.
    pub times: Vec<f32>,
    /// One value per key (xyz for translation and scale, a quaternion xyzw for rotation); three
    /// per key for [`Interpolation::CubicSpline`]: in-tangent, value, out-tangent.
    pub values: Vec<Vec4>,
}

impl Channel {
    fn key_value(&self, key: usize) -> Vec4 {
        match self.interpolation {
            Interpolation::CubicSpline => self.values[key * 3 + 1],
            _ => self.values[key],
        }
    }

    /// The channel's value at `time` (clamped to its keys).
    pub fn sample(&self, time: f32) -> Vec4 {
        let n = self.times.len();
        if n == 0 {
            return Vec4::ZERO;
        }
        if time <= self.times[0] || n == 1 {
            return self.key_value(0);
        }
        if time >= self.times[n - 1] {
            return self.key_value(n - 1);
        }
        // The key interval containing `time`.
        let next = self.times.partition_point(|&t| t <= time).clamp(1, n - 1);
        let key = next - 1;
        let dt = self.times[next] - self.times[key];
        let t = if dt > 0.0 { (time - self.times[key]) / dt } else { 0.0 };
        let rotation = self.property == Property::Rotation;
        match self.interpolation {
            Interpolation::Step => self.values[key],
            Interpolation::Linear if rotation => {
                let q = slerp(Quat::from_vec4(self.values[key]), Quat::from_vec4(self.values[next]), t);
                Vec4::from(q)
            }
            Interpolation::Linear => self.values[key].lerp(self.values[next], t),
            Interpolation::CubicSpline => {
                // glTF 2.0 appendix C: p(t) = h00·v0 + h10·dt·b0 + h01·v1 + h11·dt·a1.
                let (v0, b0) = (self.values[key * 3 + 1], self.values[key * 3 + 2]);
                let (a1, v1) = (self.values[next * 3], self.values[next * 3 + 1]);
                let (t2, t3) = (t * t, t * t * t);
                let p = v0 * (2.0 * t3 - 3.0 * t2 + 1.0) + b0 * (dt * (t3 - 2.0 * t2 + t)) + v1 * (-2.0 * t3 + 3.0 * t2) + a1 * (dt * (t3 - t2));
                if rotation {
                    p.normalize_or(Vec4::W)
                } else {
                    p
                }
            }
        }
    }
}

/// A named set of channels (glTF animation).
#[derive(Clone, Debug, PartialEq)]
pub struct AnimationClip {
    pub name: String,
    pub channels: Vec<Channel>,
    /// The last key time of any channel.
    pub duration: f32,
}

impl AnimationClip {
    /// Overwrites the animated properties of `pose` (local transforms, one per node) with their
    /// values at `time`.
    pub fn apply(&self, time: f32, pose: &mut [NodeTransform]) {
        for channel in &self.channels {
            let Some(node) = pose.get_mut(channel.node) else { continue };
            let v = channel.sample(time);
            match channel.property {
                Property::Translation => node.translation = v.truncate(),
                Property::Rotation => node.rotation = Quat::from_vec4(v).normalize(),
                Property::Scale => node.scale = v.truncate(),
            }
        }
    }
}

/// Shortest-path spherical interpolation (glam's slerp, after flipping `b` into `a`'s
/// hemisphere, which glTF requires for keyframes).
fn slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let b = if a.dot(b) < 0.0 { -b } else { b };
    a.slerp(b, t).normalize()
}

/// The rest pose: every node's own transform.
pub fn rest_pose(nodes: &[Node]) -> Vec<NodeTransform> {
    nodes.iter().map(|n| n.transform).collect()
}

/// Node indices ordered so that every parent comes before its children.
pub fn hierarchy_order(nodes: &[Node]) -> Vec<usize> {
    let mut depth = vec![usize::MAX; nodes.len()];
    fn depth_of(nodes: &[Node], depth: &mut [usize], i: usize, guard: usize) -> usize {
        if depth[i] == usize::MAX {
            // A malformed cycle would recurse forever: cut it at the node count.
            depth[i] = match nodes[i].parent {
                Some(p) if guard < nodes.len() => depth_of(nodes, depth, p, guard + 1) + 1,
                _ => 0,
            };
        }
        depth[i]
    }
    for i in 0..nodes.len() {
        depth_of(nodes, &mut depth, i, 0);
    }
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by_key(|&i| depth[i]);
    order
}

/// Global (scene) matrices of every node for the local transforms `pose`; `order` from
/// [`hierarchy_order`].
pub fn global_matrices(nodes: &[Node], order: &[usize], pose: &[NodeTransform]) -> Vec<Mat4> {
    let mut globals = vec![Mat4::IDENTITY; nodes.len()];
    for &i in order {
        let local = pose[i].matrix();
        globals[i] = match nodes[i].parent {
            Some(p) => globals[p] * local,
            None => local,
        };
    }
    globals
}

/// Plays animation clips on a node hierarchy: one clip at a time with speed, looping and pause,
/// crossfading from the previous clip when switching.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimationPlayer {
    /// Playback rate (1 = real time).
    pub speed: f32,
    /// Wrap around at the clip's end (otherwise hold its last frame).
    pub looping: bool,
    pub paused: bool,
    clip: usize,
    time: f32,
    /// The clip being faded out: (clip, its time, fade elapsed, fade duration).
    fading: Option<(usize, f32, f32, f32)>,
}

impl AnimationPlayer {
    /// Plays `clip` from its start, looping at normal speed.
    pub fn new(clip: usize) -> Self {
        Self { speed: 1.0, looping: true, paused: false, clip, time: 0.0, fading: None }
    }

    pub fn clip(&self) -> usize {
        self.clip
    }

    /// Time within the current clip, in seconds.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Weight of the current clip (below 1 during a crossfade).
    pub fn blend_weight(&self) -> f32 {
        self.fading.map_or(1.0, |(_, _, elapsed, duration)| smoothstep(elapsed / duration))
    }

    /// Switches to `clip` (from its start), crossfading from the current pose over `crossfade`
    /// seconds (0: cut).
    pub fn play(&mut self, clip: usize, crossfade: f32) {
        self.fading = (crossfade > 0.0).then_some((self.clip, self.time, 0.0, crossfade));
        self.clip = clip;
        self.time = 0.0;
    }

    /// Advances the clock by `dt` seconds (scaled by `speed`, nothing while paused).
    pub fn advance(&mut self, clips: &[AnimationClip], dt: f32) {
        if self.paused {
            return;
        }
        let step = dt * self.speed;
        let wrap = |clip: usize, time: f32| {
            let duration = clips.get(clip).map_or(0.0, |c| c.duration);
            if duration <= 0.0 {
                0.0
            } else if self.looping {
                time.rem_euclid(duration)
            } else {
                time.clamp(0.0, duration)
            }
        };
        self.time = wrap(self.clip, self.time + step);
        if let Some((clip, time, elapsed, duration)) = self.fading {
            let elapsed = elapsed + dt.abs();
            self.fading = (elapsed < duration).then_some((clip, wrap(clip, time + step), elapsed, duration));
        }
    }

    /// The local transform of every node at the current time: `rest` with the current clip
    /// applied, blended from the fading clip's pose during a crossfade.
    pub fn pose(&self, clips: &[AnimationClip], rest: &[NodeTransform]) -> Vec<NodeTransform> {
        let mut pose = rest.to_vec();
        if let Some(clip) = clips.get(self.clip) {
            clip.apply(self.time, &mut pose);
        }
        if let Some((from, time, _, _)) = self.fading {
            let mut previous = rest.to_vec();
            if let Some(clip) = clips.get(from) {
                clip.apply(time, &mut previous);
            }
            let w = self.blend_weight();
            for (p, q) in pose.iter_mut().zip(&previous) {
                *p = NodeTransform::blend(q, p, w);
            }
        }
        pose
    }
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(property: Property, interpolation: Interpolation, times: &[f32], values: &[Vec4]) -> Channel {
        Channel { node: 0, property, interpolation, times: times.to_vec(), values: values.to_vec() }
    }

    #[test]
    fn step_and_linear_channels_interpolate_and_clamp() {
        let values = [Vec4::new(0.0, 0.0, 0.0, 0.0), Vec4::new(2.0, 4.0, 0.0, 0.0)];
        let linear = channel(Property::Translation, Interpolation::Linear, &[1.0, 3.0], &values);
        assert_eq!(linear.sample(2.0), Vec4::new(1.0, 2.0, 0.0, 0.0));
        assert_eq!(linear.sample(0.0), values[0]);
        assert_eq!(linear.sample(9.0), values[1]);
        let step = Channel { interpolation: Interpolation::Step, ..linear };
        assert_eq!(step.sample(2.9), values[0]);
        assert_eq!(step.sample(3.0), values[1]);
    }

    #[test]
    fn rotations_take_the_shortest_arc() {
        let a = Quat::from_rotation_y(0.0);
        // The same 90° rotation stored with the opposite sign: still the short way round.
        let b = -Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let c = channel(Property::Rotation, Interpolation::Linear, &[0.0, 1.0], &[Vec4::from(a), Vec4::from(b)]);
        let mid = Quat::from_vec4(c.sample(0.5));
        assert!(mid.angle_between(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)) < 1e-4, "{mid}");
    }

    #[test]
    fn cubic_splines_pass_through_keys_with_their_tangents() {
        // Keys at 0 and 2 with value 0 and 1; flat tangents give a smoothstep.
        let keys = [Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ONE, Vec4::ZERO];
        let c = channel(Property::Scale, Interpolation::CubicSpline, &[0.0, 2.0], &keys);
        assert_eq!(c.sample(0.0), Vec4::ZERO);
        assert_eq!(c.sample(2.0), Vec4::ONE);
        assert!((c.sample(1.0).x - 0.5).abs() < 1e-6 && (c.sample(0.5).x - 0.15625).abs() < 1e-6);
        // An out-tangent of 1/s over a 2 s interval adds dt · (t³ - 2t² + t).
        let mut sloped = keys;
        sloped[2] = Vec4::ONE;
        let c = channel(Property::Scale, Interpolation::CubicSpline, &[0.0, 2.0], &sloped);
        assert!((c.sample(1.0).x - 0.75).abs() < 1e-6);
    }

    #[test]
    fn hierarchies_compose_parent_first_and_skins_bind_relative_to_their_mesh() {
        let nodes = vec![
            Node { name: "child".into(), parent: Some(1), transform: NodeTransform { translation: Vec3::X, ..Default::default() } },
            Node { name: "root".into(), parent: None, transform: NodeTransform { translation: Vec3::Y, ..Default::default() } },
        ];
        let order = hierarchy_order(&nodes);
        assert_eq!(order, [1, 0]);
        let globals = global_matrices(&nodes, &order, &rest_pose(&nodes));
        assert_eq!(globals[0].w_axis.truncate(), Vec3::new(1.0, 1.0, 0.0));
        // Bind pose: the inverse bind matrix undoes the joint's rest transform.
        let skin = Skin { name: "s".into(), joints: vec![0], inverse_bind_matrices: vec![globals[0].inverse()], skeleton: None };
        let bind = Mat4::from_translation(Vec3::Z);
        let palette = skin.joint_matrices(&globals, bind);
        assert!(palette[0].abs_diff_eq(bind.inverse(), 1e-6));
    }

    #[test]
    fn players_loop_pause_and_crossfade() {
        let clip = |to: f32| AnimationClip {
            name: String::new(),
            channels: vec![channel(Property::Translation, Interpolation::Linear, &[0.0, 2.0], &[Vec4::ZERO, Vec4::new(to, 0.0, 0.0, 0.0)])],
            duration: 2.0,
        };
        let clips = [clip(2.0), clip(-2.0)];
        let rest = vec![NodeTransform::default()];
        let mut player = AnimationPlayer::new(0);
        player.advance(&clips, 2.5);
        assert!((player.time() - 0.5).abs() < 1e-6, "loops");
        player.paused = true;
        player.advance(&clips, 1.0);
        assert!((player.pose(&clips, &rest)[0].translation.x - 0.5).abs() < 1e-6);
        player.paused = false;
        player.play(1, 1.0);
        assert_eq!(player.pose(&clips, &rest)[0].translation.x, 0.5, "a crossfade starts from the old pose");
        player.advance(&clips, 0.5);
        let x = player.pose(&clips, &rest)[0].translation.x;
        // Halfway: old clip at 1.0 s (x = 1), new clip at 0.5 s (x = -0.5), equal weights.
        assert!((x - 0.25).abs() < 1e-5, "{x}");
        player.advance(&clips, 0.6);
        assert_eq!(player.blend_weight(), 1.0);
        assert!((player.pose(&clips, &rest)[0].translation.x + 1.1).abs() < 1e-5);
        player.looping = false;
        player.advance(&clips, 5.0);
        assert_eq!(player.time(), 2.0, "holds the last frame");
    }
}
