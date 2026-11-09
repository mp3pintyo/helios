//! Camera, projection conventions and frustum utilities.
//!
//! Conventions: right-handed, +Y up, the camera looks down -Z in view space, and every
//! projection uses **reverse-Z** (near = 1, far = 0) for depth precision. The main camera
//! uses an infinite far plane.

use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    /// Rotation around +Y in radians (0 looks down -Z).
    pub yaw: f32,
    /// Rotation above the horizon in radians.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { position: Vec3::new(0.0, 2.0, 10.0), yaw: 0.0, pitch: 0.0, fov_y: 60f32.to_radians(), near: 0.05 }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or(Vec3::X)
    }

    pub fn view_matrix(&self) -> Mat4 {
        look_to_mat4(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        directx::perspective_infinite_reverse(self.fov_y, aspect, self.near)
    }

    /// Pixels covered by one world unit at distance 1 — the LOD scale for culling.
    pub fn lod_scale(&self, viewport_height: f32) -> f32 {
        viewport_height / (2.0 * (self.fov_y * 0.5).tan())
    }

    /// Orients the camera to look at `target`.
    pub fn look_at(&mut self, target: Vec3) {
        let d = (target - self.position).normalize_or(Vec3::NEG_Z);
        self.pitch = d.y.clamp(-1.0, 1.0).asin();
        self.yaw = (-d.x).atan2(-d.z);
    }
}

/// World-space frustum planes (xyz normal pointing inside, w distance) from a view-projection
/// matrix with a [0, 1] depth range. Degenerate planes (an infinite far plane) become
/// "always inside".
pub fn frustum_planes(view_proj: &Mat4) -> [Vec4; 6] {
    let row = |i: usize| Vec4::new(view_proj.x_axis[i], view_proj.y_axis[i], view_proj.z_axis[i], view_proj.w_axis[i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|p| {
        let len = p.truncate().length();
        if len < 1e-6 {
            Vec4::new(0.0, 0.0, 0.0, 1.0)
        } else {
            p / len
        }
    })
}

/// Halton low-discrepancy sequence, used for the sub-pixel jitter of TSR.
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}
