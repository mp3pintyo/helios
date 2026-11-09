//! Points and the volume a graph runs in.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A PCG point: where something may be spawned, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub position: Vec3,
    /// Surface normal at the point.
    pub normal: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// 0–1: how strongly the point wants to exist (filters and pruning read it).
    pub density: f32,
    /// The point's own random seed (from the world cell or triangle it was sampled in).
    pub seed: u32,
}

impl Point {
    pub fn new(position: Vec3, normal: Vec3, seed: u32) -> Self {
        Self { position, normal, rotation: Quat::IDENTITY, scale: Vec3::ONE, density: 1.0, seed }
    }
}

/// An oriented box (rotated about +Y) that bounds a graph's points, as UE's PCG volume.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Volume {
    pub center: [f32; 3],
    pub half_extents: [f32; 3],
    /// Rotation about +Y in degrees.
    #[serde(default)]
    pub yaw_deg: f32,
}

impl Volume {
    fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw_deg.to_radians())
    }

    /// World position -> volume-local metres (relative to the centre, unrotated).
    pub fn to_local(&self, p: Vec3) -> Vec3 {
        self.rotation().inverse() * (p - Vec3::from(self.center))
    }

    /// Volume-local metres -> world position.
    pub fn to_world(&self, local: Vec3) -> Vec3 {
        Vec3::from(self.center) + self.rotation() * local
    }

    /// Volume-local position scaled to [-1, 1] on every axis inside the volume.
    pub fn normalized(&self, p: Vec3) -> Vec3 {
        self.to_local(p) / Vec3::from(self.half_extents).max(Vec3::splat(1e-6))
    }

    pub fn contains(&self, p: Vec3) -> bool {
        self.normalized(p).abs().max_element() <= 1.0
    }

    /// Inside the volume's footprint (ignoring height).
    pub fn contains_xz(&self, p: Vec3) -> bool {
        let n = self.normalized(p);
        n.x.abs() <= 1.0 && n.z.abs() <= 1.0
    }

    /// World-space axis-aligned bounds.
    pub fn world_bounds(&self) -> (Vec3, Vec3) {
        let h = Vec3::from(self.half_extents);
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for k in 0..8 {
            let corner = Vec3::new(if k & 1 == 0 { -h.x } else { h.x }, if k & 2 == 0 { -h.y } else { h.y }, if k & 4 == 0 { -h.z } else { h.z });
            let p = self.to_world(corner);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_volumes_contain_their_local_box() {
        let v = Volume { center: [10.0, 0.0, 5.0], half_extents: [4.0, 2.0, 1.0], yaw_deg: 90.0 };
        // +X local is -Z world after a 90° turn about +Y.
        assert!(v.contains(Vec3::new(10.0, 0.0, 5.0 - 3.9)));
        assert!(!v.contains(Vec3::new(13.9, 0.0, 5.0)));
        let p = Vec3::new(11.0, 1.0, 3.0);
        assert!(v.to_world(v.to_local(p)).distance(p) < 1e-5);
    }
}
