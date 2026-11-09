//! Water bodies and the water medium shared by the passes that draw water or what lies under it
//! (`shaders/water.wgsl`).
//!
//! A *water body* is the flat surface of a [`MaterialKind::Water`] instance: its world rectangle
//! and level, derived from the instance's bounds (UE's water bodies, without a spline editor).
//! Knowing where the water is lets the passes treat everything relative to it:
//!
//! * the landscape material turns into a wet lake bed below the level and a wet shore band
//!   just above it (`shaders/shore.wgsl`, G-buffer resolve), per pixel at the exact surface
//!   height, so the bed lines up with the water line whichever path (runtime virtual texture or
//!   direct layering) shaded the landscape;
//! * the forward water shader refracts, absorbs and adds caustics over the water column;
//! * when the camera is below a body's surface, translucency draws the underwater view (fog by
//!   absorption and scattering, caustics, the surface from below with Snell's window).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use helios_geometry::Aabb;

use crate::scene::{MaterialKind, Scene, Water};

/// Water bodies the shaders know about (the uniform's array size).
pub const MAX_WATER_BODIES: usize = 4;

/// Binding of the [`WaterUniform`] in the G-buffer resolve (lake beds and shores).
pub const BINDING_WATER: u32 = 90;

/// The flat surface of a water instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterBody {
    /// World rectangle covered by the surface (x, z).
    pub min: Vec2,
    pub max: Vec2,
    /// Height of the surface.
    pub level: f32,
}

impl WaterBody {
    /// The body of a water instance: the world bounds of its mesh (a flat sheet), its top as the
    /// level.
    pub fn from_bounds(aabb: &Aabb, transform: &Mat4) -> Self {
        let corners = (0..8).map(|i| {
            let pick = |bit: usize, axis: usize| if i & (1 << bit) != 0 { aabb.max[axis] } else { aabb.min[axis] };
            transform.transform_point3(Vec3::new(pick(0, 0), pick(1, 1), pick(2, 2)))
        });
        let world = Aabb::from_points(corners);
        Self { min: Vec2::new(world.min.x, world.min.z), max: Vec2::new(world.max.x, world.max.z), level: world.max.y }
    }

    /// Whether `p` is under this body's surface (within its rectangle and below its level).
    pub fn contains(&self, p: Vec3) -> bool {
        p.y < self.level && p.x >= self.min.x && p.x <= self.max.x && p.z >= self.min.y && p.z <= self.max.y
    }
}

/// The water bodies of the scene's water instances (at most [`MAX_WATER_BODIES`], the first ones).
pub(crate) fn water_bodies(scene: &Scene) -> Vec<WaterBody> {
    let bodies: Vec<WaterBody> = scene
        .instances
        .iter()
        .filter(|i| i.visible && scene.materials.get(i.material.0 as usize).is_some_and(|m| m.kind == MaterialKind::Water))
        .map(|i| WaterBody::from_bounds(&scene.meshes[i.mesh.0 as usize].aabb, &i.transform))
        .collect();
    if bodies.len() > MAX_WATER_BODIES {
        log::warn!("{} water bodies; the first {MAX_WATER_BODIES} get lake beds and underwater views", bodies.len());
    }
    bodies.into_iter().take(MAX_WATER_BODIES).collect()
}

/// Mirrors `Water` in water.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct WaterUniform {
    absorption: Vec4,
    scattering: Vec4,
    waves: Vec4,
    misc: Vec4,
    bodies: [Vec4; MAX_WATER_BODIES],
    levels: Vec4,
}

impl WaterUniform {
    /// `camera_body`: the body whose water the camera is in.
    pub fn new(water: &Water, bodies: &[WaterBody], camera_body: Option<usize>) -> Self {
        let direction = water.wind_direction.normalize_or(Vec2::X);
        let mut rects = [Vec4::ZERO; MAX_WATER_BODIES];
        let mut levels = [0.0; MAX_WATER_BODIES];
        for (i, b) in bodies.iter().take(MAX_WATER_BODIES).enumerate() {
            rects[i] = Vec4::new(b.min.x, b.min.y, b.max.x, b.max.y);
            levels[i] = b.level;
        }
        Self {
            absorption: water.absorption.extend(0.0),
            scattering: water.scattering.extend(0.0),
            waves: Vec4::new(direction.x, direction.y, water.wavelength.max(0.01), water.steepness),
            misc: Vec4::new(
                water.foam,
                water.caustics,
                bodies.len().min(MAX_WATER_BODIES) as f32,
                camera_body.map_or(-1.0, |b| b as f32),
            ),
            bodies: rects,
            levels: Vec4::from(levels),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_water_sheet_becomes_a_body_at_its_level() {
        let sheet = Aabb { min: Vec3::new(-10.0, 0.0, -10.0), max: Vec3::new(10.0, 0.0, 10.0) };
        let transform = Mat4::from_scale_rotation_translation(Vec3::new(2.0, 1.0, 1.0), glam::Quat::IDENTITY, Vec3::new(5.0, -2.2, 1.0));
        let body = WaterBody::from_bounds(&sheet, &transform);
        assert_eq!(body, WaterBody { min: Vec2::new(-15.0, -9.0), max: Vec2::new(25.0, 11.0), level: -2.2 });
        assert!(body.contains(Vec3::new(0.0, -3.0, 0.0)));
        assert!(!body.contains(Vec3::new(0.0, -2.0, 0.0)), "above the surface");
        assert!(!body.contains(Vec3::new(30.0, -3.0, 0.0)), "beside the surface");
    }

    #[test]
    fn the_default_water_is_a_clear_lake() {
        let w = Water::default();
        let extinction = w.absorption + w.scattering;
        // Green light travels about 10 m (1/e) - the bed stays visible for several metres - while
        // red is gone within a few: deep water turns blue-green.
        assert!((8.0..15.0).contains(&(1.0 / extinction.y)));
        assert!(extinction.x > 3.0 * extinction.y && extinction.x > 3.0 * extinction.z);
        // Scattering stays well below absorption in red and below half the green extinction
        // (a clear lake, not a milky glacial one).
        assert!(w.scattering.max_element() < 0.5 * extinction.y);
    }

    #[test]
    fn only_water_instances_are_bodies() {
        let mut scene = Scene::new();
        let sheet = Aabb { min: Vec3::splat(-1.0), max: Vec3::new(1.0, -1.0, 1.0) };
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh { aabb: sheet, ..Default::default() });
        let water = scene.add_material(crate::Material { kind: MaterialKind::Water, ..Default::default() });
        let rock = scene.add_material(crate::Material::default());
        scene.add_instance(mesh, rock, Mat4::IDENTITY);
        scene.add_instance(mesh, water, Mat4::from_translation(Vec3::Y * 4.0));
        let bodies = water_bodies(&scene);
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].level, 3.0);
        let uniform = WaterUniform::new(&scene.water, &bodies, Some(0));
        assert_eq!(uniform.misc.z, 1.0);
        assert_eq!(uniform.misc.w, 0.0);
        assert_eq!(uniform.bodies[0], Vec4::new(-1.0, -1.0, 1.0, 1.0));
    }
}
