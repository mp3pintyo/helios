//! Shadow page invalidation for moving instances (UE5's per-primitive VSM cache invalidation).
//!
//! A shadow caster that moves changes the cached pages under its old and its new bounds — and
//! nothing else. The CPU knows every move ([`crate::motion`]), so it writes their footprints
//! straight into the invalidation's footprint list (`vsm_invalidate.wgsl`), ahead of the ones the
//! GPU collects for streaming and wind. Pages they overlap are then handled like any other
//! invalidation, in the sun's clipmap and in every local light's faces and mips:
//!
//! * a [`crate::Mobility::Movable`] caster is a *dynamic* caster: the pages under it re-render
//!   their dynamic casters over their cached static layer;
//! * a static caster that moved (a one-off edit) is part of the static layers: the pages under it
//!   are freed and rendered again in full.
//!
//! In the sun's clipmap a footprint is the light-space rectangle around the instance's oriented
//! box in both poses, much tighter than its bounding sphere for long or flat casters; local
//! lights test bounding spheres. A long box is cut into slices along its longest axis, each with
//! a footprint of its own: a chandelier on its 12 m chain is five footprints of 2.4 m, not one
//! sphere of 6.3 m radius that covers most of the cube faces of every candle light near it.
//!
//! A movable caster's footprints also bound how fast each slice moves: its largest corner
//! displacement per frame (a rigid move's displacement is affine over the box, so the corners
//! bound it), held with a slow decay as a recent speed bound. As for the wind, a page is then
//! refreshed once the slice may have moved `WIND_REFRESH_TEXELS` of its texels since the page
//! was rendered — every frame where a swinging chandelier crosses fine pages, every few frames at
//! coarse mips, and rarely for the slices next to the chain's pivot, which hardly move. A
//! skipped page may show the caster up to that many texels off, so the GPU pads move footprints
//! by as much; when the caster stops, its pages are refreshed once more at rest. A jump or a
//! static caster's move refreshes every page it overlaps at once; at rest, a movable instance
//! costs nothing.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use helios_geometry::Aabb;

/// A shadow caster's move this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowMove {
    /// The instance's index.
    pub instance: u32,
    /// The mesh's box and the instance's transforms before and after.
    pub aabb: Aabb,
    pub old_world: Mat4,
    pub new_world: Mat4,
    /// World bounding spheres before and after.
    pub old: Vec4,
    pub new: Vec4,
    /// A dynamic caster (movable), otherwise a static one.
    pub movable: bool,
}

/// `struct Footprint` in vsm_invalidate.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct Footprint {
    /// World-space centre, w: kind (`FOOTPRINT_*`).
    world: Vec4,
    /// Sun light-space centre (x, -y).
    center: Vec2,
    /// World sphere radius (local lights).
    radius: f32,
    /// Bounds of the change overall and per frame: unbounded for a jump or a static caster.
    change: f32,
    per_frame: f32,
    /// 1 for the wind's own footprints (collected on the GPU); a move's is 0.
    wind: u32,
    /// Sun light-space half extent of the rectangle around `center`.
    rect: Vec2,
}

const _: () = assert!(std::mem::size_of::<Footprint>() == 48);

/// `FOOTPRINT_DYNAMIC`, `FOOTPRINT_STATIC` and `ALWAYS` in vsm_invalidate.wgsl.
const DYNAMIC: f32 = 0.0;
const STATIC: f32 = 2.0;
const ALWAYS: f32 = 3.0e38;
/// Most slices a box is cut into (along its longest axis, into roughly cubic pieces).
const MAX_SLICES: usize = 8;
/// Per frame, a slice's speed bound decays by this factor towards its current speed: a caster
/// slowing down (a pendulum before it turns) keeps refreshing pages as fast as it moved a few
/// dozen frames ago.
const SPEED_DECAY: f32 = 0.95;

/// Light-space (x, -y) bounds of `aabb` placed by `world`.
fn light_rect(aabb: &Aabb, world: Mat4, light_view: &Mat4) -> (Vec2, Vec2) {
    let to_light = *light_view * world;
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for c in corners(aabb) {
        let p = to_light.transform_point3(c);
        let v = Vec2::new(p.x, -p.y);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

fn corners(aabb: &Aabb) -> impl Iterator<Item = Vec3> + '_ {
    (0..8).map(|k| {
        Vec3::new(
            if k & 1 == 0 { aabb.min.x } else { aabb.max.x },
            if k & 2 == 0 { aabb.min.y } else { aabb.max.y },
            if k & 4 == 0 { aabb.min.z } else { aabb.max.z },
        )
    })
}

/// `aabb` cut along its longest axis (in world units under `world`) into up to [`MAX_SLICES`]
/// roughly cubic slices.
fn slices(aabb: &Aabb, world: Mat4) -> Vec<Aabb> {
    let scale = Vec3::new(world.x_axis.truncate().length(), world.y_axis.truncate().length(), world.z_axis.truncate().length());
    let size = (aabb.max - aabb.min) * scale;
    let axis = size.max_position();
    let mut rest = size.to_array();
    rest[axis] = 0.0;
    let across = rest.into_iter().fold(0.0f32, f32::max);
    let count = if across > 0.0 { ((size[axis] / across).ceil() as usize).clamp(1, MAX_SLICES) } else { MAX_SLICES };
    let step = (aabb.max[axis] - aabb.min[axis]) / count as f32;
    (0..count)
        .map(|i| {
            let mut slice = *aabb;
            slice.min[axis] = aabb.min[axis] + step * i as f32;
            slice.max[axis] = if i + 1 == count { aabb.max[axis] } else { aabb.min[axis] + step * (i + 1) as f32 };
            slice
        })
        .collect()
}

/// World bounding sphere of `aabb` placed by `world` (the box's image is a parallelepiped: a
/// corner is farthest from its centre).
fn sphere(aabb: &Aabb, world: Mat4) -> Vec4 {
    let center = world.transform_point3((aabb.min + aabb.max) * 0.5);
    center.extend(corners(aabb).map(|c| world.transform_point3(c).distance(center)).fold(0.0, f32::max))
}

/// Largest displacement of a point of `aabb` from pose `old` to pose `new`.
fn displacement(aabb: &Aabb, old: Mat4, new: Mat4) -> f32 {
    corners(aabb).map(|c| new.transform_point3(c).distance(old.transform_point3(c))).fold(0.0, f32::max)
}

fn footprint(sphere: Vec4, (lo, hi): (Vec2, Vec2), kind: f32, per_frame: f32) -> Footprint {
    Footprint {
        world: sphere.truncate().extend(kind),
        center: (lo + hi) * 0.5,
        radius: sphere.w,
        change: ALWAYS,
        per_frame,
        wind: 0,
        rect: ((hi - lo) * 0.5).max(Vec2::splat(1e-4)),
    }
}

/// The footprints of one box moved from `old_world` (sphere `old`) to `new_world` (`new`): one
/// around both poses (the swept volume) for a small move, one per pose for a jump, so the pages
/// between them stay cached. `per_frame` bounds the speed of a small move.
fn push_move(list: &mut Vec<Footprint>, aabb: &Aabb, worlds: (Mat4, Mat4), spheres: (Vec4, Vec4), light_view: &Mat4, kind: f32, per_frame: f32) {
    let (old, new) = (light_rect(aabb, worlds.0, light_view), light_rect(aabb, worlds.1, light_view));
    let (a, b) = (spheres.0.truncate(), spheres.1.truncate());
    let gap = a.distance(b);
    let radius = spheres.0.w.max(spheres.1.w);
    if gap <= radius {
        let swept = ((a + b) * 0.5).extend(gap * 0.5 + radius);
        list.push(footprint(swept, (old.0.min(new.0), old.1.max(new.1)), kind, per_frame));
    } else {
        list.push(footprint(spheres.0, old, kind, ALWAYS));
        list.push(footprint(spheres.1, new, kind, ALWAYS));
    }
}

/// A movable caster that moved recently.
#[derive(Debug)]
struct Moving {
    aabb: Aabb,
    /// Its latest pose.
    world: Mat4,
    /// Speed bound per slice (world units per frame).
    speed: Vec<f32>,
    /// Moved this frame.
    moved: bool,
}

/// Turns each frame's moves into footprints, remembering movable casters' recent speeds and
/// their last pose (see the module documentation).
#[derive(Debug, Default)]
pub(crate) struct MoveFootprints {
    moving: HashMap<u32, Moving>,
}

impl MoveFootprints {
    /// Forgets every caster (the scene changed: every cached page is invalidated anyway).
    pub fn reset(&mut self) {
        self.moving.clear();
    }

    /// The footprints of this frame's `moves` and of the movable casters that stopped since the
    /// last frame, for the sun's light view `light_view`.
    pub fn update(&mut self, moves: &[ShadowMove], light_view: &Mat4) -> Vec<Footprint> {
        let mut list = Vec::with_capacity(moves.len() * 2);
        for moving in self.moving.values_mut() {
            moving.moved = false;
        }
        for m in moves {
            let slices = slices(&m.aabb, m.new_world);
            if !m.movable {
                self.moving.remove(&m.instance);
                for s in &slices {
                    let spheres = if slices.len() == 1 { (m.old, m.new) } else { (sphere(s, m.old_world), sphere(s, m.new_world)) };
                    push_move(&mut list, s, (m.old_world, m.new_world), spheres, light_view, STATIC, ALWAYS);
                }
                continue;
            }
            let moving = self.moving.entry(m.instance).or_insert_with(|| Moving { aabb: m.aabb, world: m.new_world, speed: Vec::new(), moved: true });
            if moving.speed.len() != slices.len() || moving.aabb != m.aabb {
                moving.speed = vec![0.0; slices.len()];
            }
            moving.aabb = m.aabb;
            moving.world = m.new_world;
            moving.moved = true;
            for (s, speed) in slices.iter().zip(&mut moving.speed) {
                *speed = displacement(s, m.old_world, m.new_world).max(*speed * SPEED_DECAY);
                let spheres = if slices.len() == 1 { (m.old, m.new) } else { (sphere(s, m.old_world), sphere(s, m.new_world)) };
                push_move(&mut list, s, (m.old_world, m.new_world), spheres, light_view, DYNAMIC, *speed);
            }
        }
        // Casters that stopped: their pages may show them a little behind; refresh them at rest.
        self.moving.retain(|_, moving| {
            if !moving.moved {
                for s in slices(&moving.aabb, moving.world) {
                    let rest = sphere(&s, moving.world);
                    list.push(footprint(rest, light_rect(&s, moving.world, light_view), DYNAMIC, ALWAYS));
                }
            }
            moving.moved
        });
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> Aabb {
        Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) }
    }

    fn moved(old_world: Mat4, new_world: Mat4, movable: bool) -> ShadowMove {
        let sphere = |w: Mat4| w.transform_point3(Vec3::ZERO).extend(3f32.sqrt());
        ShadowMove { instance: 0, aabb: unit_box(), old_world, new_world, old: sphere(old_world), new: sphere(new_world), movable }
    }

    #[test]
    fn small_moves_sweep_and_jumps_split() {
        let view = Mat4::IDENTITY;
        let step = moved(Mat4::IDENTITY, Mat4::from_translation(Vec3::X * 0.5), true);
        let f = MoveFootprints::default().update(&[step], &view);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].world, Vec4::new(0.25, 0.0, 0.0, DYNAMIC));
        assert!((f[0].radius - (0.25 + 3f32.sqrt())).abs() < 1e-5);
        // The rectangle spans both poses' boxes; the speed is the move's.
        assert_eq!((f[0].center, f[0].rect), (Vec2::new(0.25, 0.0), Vec2::new(1.25, 1.0)));
        assert!((f[0].per_frame - 0.5).abs() < 1e-6 && f[0].change == ALWAYS);
        let jump = moved(Mat4::IDENTITY, Mat4::from_translation(Vec3::X * 10.0), false);
        let f = MoveFootprints::default().update(&[jump], &view);
        assert_eq!(f.len(), 2);
        assert_eq!(f.iter().map(|f| f.world.w).collect::<Vec<_>>(), vec![STATIC, STATIC]);
        assert_eq!((f[1].center, f[1].rect), (Vec2::new(10.0, 0.0), Vec2::ONE));
        assert!(f.iter().all(|f| f.per_frame == ALWAYS));
    }

    #[test]
    fn rectangles_are_in_light_space() {
        // A tall, thin caster seen by a light looking down the x axis: its light-space
        // rectangles are thin across and, stacked, as long as the caster along y.
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, Vec3::X, Vec3::Y);
        let pole = Mat4::from_scale(Vec3::new(0.1, 6.0, 0.1));
        let m = ShadowMove { old_world: pole, new_world: Mat4::from_translation(Vec3::Z * 0.01) * pole, ..moved(pole, pole, true) };
        let f = MoveFootprints::default().update(&[m], &view);
        assert_eq!(f.len(), MAX_SLICES);
        for s in &f {
            assert!((s.rect.x - 0.1).abs() < 0.006, "{}", s.rect);
            assert!(s.radius < 1.0, "a slice's sphere, not the pole's: {}", s.radius);
        }
        let (lo, hi) = f.iter().fold((f32::MAX, f32::MIN), |(lo, hi), s| (lo.min(s.center.y - s.rect.y), hi.max(s.center.y + s.rect.y)));
        assert!((lo + 6.0).abs() < 1e-4 && (hi - 6.0).abs() < 1e-4, "{lo} {hi}");
    }

    #[test]
    fn slices_near_a_pivot_move_slowly() {
        // A pendulum swinging about its top end: the slice at the pivot barely moves.
        let aabb = Aabb { min: Vec3::new(-0.5, 0.0, -0.5), max: Vec3::new(0.5, 12.0, 0.5) };
        let pivot = Vec3::Y * 12.0;
        let swing = Mat4::from_translation(pivot) * Mat4::from_rotation_z(0.001) * Mat4::from_translation(-pivot);
        let m = ShadowMove { instance: 3, aabb, old_world: Mat4::IDENTITY, new_world: swing, old: Vec4::new(0.0, 6.0, 0.0, 6.1), new: Vec4::new(0.0, 6.0, 0.0, 6.1), movable: true };
        let f = MoveFootprints::default().update(&[m], &Mat4::IDENTITY);
        assert_eq!(f.len(), MAX_SLICES);
        let (bottom, top) = (f[0].per_frame, f[MAX_SLICES - 1].per_frame);
        assert!((bottom - 0.012).abs() < 1e-3 && top < bottom / 6.0, "{bottom} {top}");
    }

    #[test]
    fn speed_bounds_decay_and_stopped_casters_settle() {
        let view = Mat4::IDENTITY;
        let mut moves = MoveFootprints::default();
        let fast = moved(Mat4::IDENTITY, Mat4::from_translation(Vec3::X * 0.1), true);
        moves.update(&[fast], &view);
        // Slower now: the bound keeps most of the earlier speed.
        let slow = moved(fast.new_world, Mat4::from_translation(Vec3::X * 0.11), true);
        let f = moves.update(&[slow], &view);
        assert!((f[0].per_frame - 0.1 * SPEED_DECAY).abs() < 1e-5);
        // Stopped: one footprint at rest that refreshes every page under it, then nothing.
        let f = moves.update(&[], &view);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].world, f[0].per_frame), (Vec4::new(0.11, 0.0, 0.0, DYNAMIC), ALWAYS));
        assert!(moves.update(&[], &view).is_empty());
    }
}
