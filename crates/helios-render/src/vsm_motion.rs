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
//! box in both poses, much tighter than its bounding sphere for long or flat casters (a
//! chandelier on its 12 m chain: a third of the area); local lights test the bounding
//! spheres. Unlike the wind, a rigid move has no bounded displacement or speed, so its
//! footprints refresh every level they overlap at once; at rest, a movable instance costs
//! nothing.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use helios_geometry::Aabb;

/// A shadow caster's move this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowMove {
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
    /// Bounds of the change overall and per frame: unbounded for a move.
    change: f32,
    per_frame: f32,
    _pad: f32,
    /// Sun light-space half extent of the rectangle around `center`.
    rect: Vec2,
}

const _: () = assert!(std::mem::size_of::<Footprint>() == 48);

/// `FOOTPRINT_DYNAMIC`, `FOOTPRINT_STATIC` and `ALWAYS` in vsm_invalidate.wgsl.
const DYNAMIC: f32 = 0.0;
const STATIC: f32 = 2.0;
const ALWAYS: f32 = 3.0e38;

/// Light-space (x, -y) bounds of `aabb` placed by `world`.
fn light_rect(aabb: &Aabb, world: Mat4, light_view: &Mat4) -> (Vec2, Vec2) {
    let to_light = *light_view * world;
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for k in 0..8 {
        let c = Vec3::new(
            if k & 1 == 0 { aabb.min.x } else { aabb.max.x },
            if k & 2 == 0 { aabb.min.y } else { aabb.max.y },
            if k & 4 == 0 { aabb.min.z } else { aabb.max.z },
        );
        let p = to_light.transform_point3(c);
        let v = Vec2::new(p.x, -p.y);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

/// The footprints of `moves` for the sun's light view `light_view`. A small move is covered by
/// one footprint around both poses (the swept volume), a jump by one per pose, so the pages
/// between them stay cached.
pub(crate) fn footprints(moves: &[ShadowMove], light_view: &Mat4) -> Vec<Footprint> {
    let footprint = |sphere: Vec4, (lo, hi): (Vec2, Vec2), kind: f32| Footprint {
        world: sphere.truncate().extend(kind),
        center: (lo + hi) * 0.5,
        radius: sphere.w,
        change: ALWAYS,
        per_frame: ALWAYS,
        _pad: 0.0,
        rect: ((hi - lo) * 0.5).max(Vec2::splat(1e-4)),
    };
    let mut list = Vec::with_capacity(moves.len() * 2);
    for m in moves {
        let kind = if m.movable { DYNAMIC } else { STATIC };
        let (old, new) = (light_rect(&m.aabb, m.old_world, light_view), light_rect(&m.aabb, m.new_world, light_view));
        let (a, b) = (m.old.truncate(), m.new.truncate());
        let gap = a.distance(b);
        let radius = m.old.w.max(m.new.w);
        if gap <= radius {
            let swept = ((a + b) * 0.5).extend(gap * 0.5 + radius);
            list.push(footprint(swept, (old.0.min(new.0), old.1.max(new.1)), kind));
        } else {
            list.push(footprint(m.old, old, kind));
            list.push(footprint(m.new, new, kind));
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> Aabb {
        Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) }
    }

    fn moved(old_world: Mat4, new_world: Mat4, movable: bool) -> ShadowMove {
        let sphere = |w: Mat4| w.transform_point3(Vec3::ZERO).extend(3f32.sqrt());
        ShadowMove { aabb: unit_box(), old_world, new_world, old: sphere(old_world), new: sphere(new_world), movable }
    }

    #[test]
    fn small_moves_sweep_and_jumps_split() {
        let view = Mat4::IDENTITY;
        let step = moved(Mat4::IDENTITY, Mat4::from_translation(Vec3::X * 0.5), true);
        let f = footprints(&[step], &view);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].world, Vec4::new(0.25, 0.0, 0.0, DYNAMIC));
        assert!((f[0].radius - (0.25 + 3f32.sqrt())).abs() < 1e-5);
        // The rectangle spans both poses' boxes.
        assert_eq!((f[0].center, f[0].rect), (Vec2::new(0.25, 0.0), Vec2::new(1.25, 1.0)));
        let jump = moved(Mat4::IDENTITY, Mat4::from_translation(Vec3::X * 10.0), false);
        let f = footprints(&[jump], &view);
        assert_eq!(f.len(), 2);
        assert_eq!(f.iter().map(|f| f.world.w).collect::<Vec<_>>(), vec![STATIC, STATIC]);
        assert_eq!((f[1].center, f[1].rect), (Vec2::new(10.0, 0.0), Vec2::ONE));
    }

    #[test]
    fn rectangles_are_in_light_space() {
        // A tall, thin caster seen by a light looking down the x axis: its light-space
        // rectangle is thin across and long along y.
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, Vec3::X, Vec3::Y);
        let pole = Mat4::from_scale(Vec3::new(0.1, 6.0, 0.1));
        let m = ShadowMove { aabb: unit_box(), old_world: pole, new_world: pole, old: Vec4::new(0.0, 0.0, 0.0, 6.0), new: Vec4::new(0.0, 0.0, 0.0, 6.0), movable: true };
        let f = footprints(&[m], &view)[0];
        assert!((f.rect - Vec2::new(0.1, 6.0)).abs().max_element() < 1e-5, "{}", f.rect);
        assert_eq!((f.change, f.per_frame), (ALWAYS, ALWAYS));
    }
}
