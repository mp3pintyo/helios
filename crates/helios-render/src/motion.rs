//! Per-frame instance changes: which instances moved since the previous frame.
//!
//! UE5 sends a primitive's transform update to the renderer, which updates the GPU scene record
//! and invalidates cached data (virtual shadow map pages, distance-field bricks, surface cache
//! cards) under the primitive's old and new bounds only. Helios does the same: [`Scene`] stamps
//! every moved instance with a version, and the [`MotionTracker`] turns the stamps into this
//! frame's list of moves, which each renderer subsystem applies incrementally. Structural changes
//! (new geometry, instances added, shadow casting toggled) still rebuild everything.

use glam::Mat4;

use crate::scene::Scene;

/// An instance whose transform or mobility changed this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Moved {
    pub index: u32,
    /// Its transform in the previous frame (equal to the current one if only its mobility
    /// changed).
    pub previous: Mat4,
}

/// What changed in the scene's instances this frame.
#[derive(Default)]
pub(crate) struct SceneChanges {
    /// Instances were added or changed in a way that rebuilds every consumer's instance data
    /// (with [`MotionTracker::incremental`] off, every move too).
    pub structural: bool,
    /// In index order.
    pub moved: Vec<Moved>,
    /// Instances that moved in the previous frame but not in this one: their GPU records still
    /// carry last frame's motion (previous-frame transform) and must be brought to rest.
    pub settled: Vec<u32>,
}

impl SceneChanges {
    /// The moves that actually changed the transform (not just the mobility).
    pub fn motions<'a>(&'a self, scene: &'a Scene) -> impl Iterator<Item = &'a Moved> + 'a {
        self.moved.iter().filter(|m| scene.instances[m.index as usize].transform != m.previous)
    }
}

/// Turns the scene's per-instance versions into each frame's [`SceneChanges`].
pub(crate) struct MotionTracker {
    /// Geometry, instance and instance layout versions last seen (structural changes).
    versions: (u64, u64, u64),
    /// Transform version last seen.
    seen: u64,
    /// Every instance's transform as of the previous frame.
    previous: Vec<Mat4>,
    /// Instances that moved in the previous frame (index order).
    moving: Vec<u32>,
    /// Apply moves incrementally (off: every move is a structural change, the behaviour before
    /// per-instance invalidation; for A/B comparisons).
    pub incremental: bool,
    pub changes: SceneChanges,
}

impl Default for MotionTracker {
    fn default() -> Self {
        Self {
            versions: (u64::MAX, u64::MAX, u64::MAX),
            seen: 0,
            previous: Vec::new(),
            moving: Vec::new(),
            incremental: true,
            changes: SceneChanges::default(),
        }
    }
}

impl MotionTracker {
    /// Collects this frame's changes of `scene` into [`MotionTracker::changes`].
    pub fn update(&mut self, scene: &Scene) {
        let versions = (scene.geometry_version, scene.instance_version, scene.layout_version());
        let count = scene.instances.len();
        let mut changes = SceneChanges { structural: versions != self.versions || self.previous.len() != count, ..Default::default() };
        // Compacted: indices shifted, so last frame's transforms belong to other instances.
        let relaid = versions.2 != self.versions.2;
        self.versions = versions;
        if self.previous.len() != count || relaid {
            // A new instance layout: nothing has a previous frame to move from.
            self.previous = scene.instances.iter().map(|i| i.transform).collect();
            self.moving.clear();
        } else {
            changes.moved = scene.changed_instances(self.seen).map(|i| Moved { index: i as u32, previous: self.previous[i] }).collect();
        }
        self.seen = scene.transform_version;
        let moving: Vec<u32> = changes.motions(scene).map(|m| m.index).collect();
        changes.settled = self.moving.iter().copied().filter(|i| moving.binary_search(i).is_err()).collect();
        for m in &changes.moved {
            self.previous[m.index as usize] = scene.instances[m.index as usize].transform;
        }
        self.moving = moving;
        if !self.incremental && !changes.moved.is_empty() {
            changes.structural = true;
        }
        self.changes = changes;
    }

    /// Every instance's transform in the previous frame.
    pub fn previous(&self) -> &[Mat4] {
        &self.previous
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Material, Mobility};
    use glam::Vec3;

    fn scene_with(count: usize) -> (Scene, Vec<crate::InstanceHandle>) {
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let material = scene.add_material(Material::default());
        let handles = (0..count).map(|i| scene.add_instance(mesh, material, Mat4::from_translation(Vec3::X * i as f32))).collect();
        (scene, handles)
    }

    #[test]
    fn moves_are_reported_once_then_settle() {
        let (mut scene, h) = scene_with(4);
        let mut tracker = MotionTracker::default();
        tracker.update(&scene);
        assert!(tracker.changes.structural && tracker.changes.moved.is_empty());
        tracker.update(&scene);
        assert!(!tracker.changes.structural && tracker.changes.moved.is_empty());

        let before = scene.transform(h[2]);
        scene.set_transform(h[2], Mat4::from_translation(Vec3::Y));
        tracker.update(&scene);
        let c = &tracker.changes;
        assert!(!c.structural);
        assert_eq!(c.moved.iter().map(|m| (m.index, m.previous)).collect::<Vec<_>>(), vec![(2, before)]);
        assert_eq!(tracker.previous()[2], Mat4::from_translation(Vec3::Y));
        // The next frame: at rest again, its GPU record must lose last frame's motion.
        tracker.update(&scene);
        assert!(tracker.changes.moved.is_empty());
        assert_eq!(tracker.changes.settled, vec![2]);
        tracker.update(&scene);
        assert!(tracker.changes.settled.is_empty());
    }

    #[test]
    fn mobility_changes_are_reported_without_motion() {
        let (mut scene, h) = scene_with(2);
        let mut tracker = MotionTracker::default();
        tracker.update(&scene);
        scene.set_mobility(h[1], Mobility::Movable);
        assert_eq!(scene.mobility(h[1]), Mobility::Movable);
        // Setting the same transform again is no move.
        scene.set_transform(h[0], scene.transform(h[0]));
        tracker.update(&scene);
        assert_eq!(tracker.changes.moved.len(), 1);
        assert_eq!(tracker.changes.motions(&scene).count(), 0);
        tracker.update(&scene);
        assert!(tracker.changes.settled.is_empty());
    }

    #[test]
    fn structural_changes_and_full_invalidation() {
        let (mut scene, h) = scene_with(2);
        let mut tracker = MotionTracker::default();
        tracker.update(&scene);
        scene.set_cast_shadow(h[0], false);
        tracker.update(&scene);
        assert!(tracker.changes.structural);
        tracker.update(&scene);
        assert!(!tracker.changes.structural);
        tracker.incremental = false;
        scene.set_transform(h[1], Mat4::IDENTITY);
        tracker.update(&scene);
        assert!(tracker.changes.structural);
    }
}
