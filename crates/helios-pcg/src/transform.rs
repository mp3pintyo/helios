//! Transform randomisation (UE's Transform Points node).

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::point::Point;
use crate::rng::random;

/// Random scale, yaw, tilt, alignment to the surface and vertical offset per point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TransformParams {
    /// Uniform scale range.
    pub scale_min: f32,
    pub scale_max: f32,
    /// Extra per-axis scale variation (0: uniform).
    pub scale_jitter: f32,
    /// Random rotation about the up axis.
    pub random_yaw: bool,
    /// Random tilt away from the up axis, up to this many degrees.
    pub tilt_deg: f32,
    /// 0: stand upright, 1: follow the surface normal.
    pub align_to_normal: f32,
    /// Vertical offset range in metres (negative sinks into the ground), times the scale.
    pub offset_min: f32,
    pub offset_max: f32,
}

impl Default for TransformParams {
    fn default() -> Self {
        Self { scale_min: 1.0, scale_max: 1.0, scale_jitter: 0.0, random_yaw: true, tilt_deg: 0.0, align_to_normal: 0.0, offset_min: 0.0, offset_max: 0.0 }
    }
}

pub fn apply(points: &mut [Point], params: &TransformParams, salt: u32) {
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    for p in points {
        let r = |stream| random(p.seed, salt, stream);
        let s = lerp(params.scale_min, params.scale_max, r(0));
        let jitter = Vec3::new(r(1), r(2), r(3)) * 2.0 - 1.0;
        let scale = Vec3::splat(s) * (Vec3::ONE + jitter * params.scale_jitter).max(Vec3::splat(0.05));
        let yaw = if params.random_yaw { Quat::from_rotation_y(r(4) * std::f32::consts::TAU) } else { Quat::IDENTITY };
        let tilt_axis = Quat::from_rotation_y(r(5) * std::f32::consts::TAU) * Vec3::X;
        let tilt = Quat::from_axis_angle(tilt_axis, (r(6) * params.tilt_deg).to_radians());
        let align = Quat::IDENTITY.slerp(Quat::from_rotation_arc(Vec3::Y, p.normal), params.align_to_normal.clamp(0.0, 1.0));
        p.rotation = (align * tilt * yaw * p.rotation).normalize();
        p.scale *= scale;
        p.position.y += lerp(params.offset_min, params.offset_max, r(7)) * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_follows_the_normal() {
        let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
        let mut points = vec![Point::new(Vec3::ZERO, normal, 5)];
        let params = TransformParams { align_to_normal: 1.0, scale_min: 2.0, scale_max: 3.0, ..Default::default() };
        apply(&mut points, &params, 1);
        assert!((points[0].rotation * Vec3::Y).distance(normal) < 1e-4);
        assert!((2.0..=3.0).contains(&points[0].scale.x));
    }
}
