//! Editing a built [`Scene`]: what an editor needs beyond building one — hiding instances
//! (deletion without shifting handles), swapping materials, reading instances back, world
//! bounds for picking and gizmos, and stable names for meshes and materials, so a scene file
//! can refer to them independently of build order.
//!
//! Instances are never removed: a removal would shift every later handle, and handles index
//! the GPU instance buffer, the culling visibility bits, Lumen's card and distance-field
//! instances and the picking ids. A hidden instance keeps its slot; every pass skips it (the
//! GPU culling tests `INSTANCE_HIDDEN` for all views, the CPU-side passes — translucency,
//! water bodies, Lumen, the landscape's virtual texture — skip it when they gather instances).

use glam::{Mat4, Vec3};
use helios_geometry::Aabb;

use crate::scene::{InstanceHandle, Material, MaterialHandle, MeshHandle, Scene};

/// Optional names of meshes and materials (empty: unnamed).
#[derive(Clone, Debug, Default)]
pub(crate) struct SceneNames {
    meshes: Vec<String>,
    materials: Vec<String>,
}

fn set_name(names: &mut Vec<String>, index: usize, name: &str) {
    if names.len() <= index {
        names.resize(index + 1, String::new());
    }
    names[index] = name.to_string();
}

fn find_name(names: &[String], name: &str) -> Option<u32> {
    names.iter().position(|n| n == name).map(|i| i as u32)
}

impl InstanceHandle {
    /// The instance's index (stable: instances are never removed).
    pub fn index(self) -> u32 {
        self.0
    }
}

impl MeshHandle {
    pub fn index(self) -> u32 {
        self.0
    }
}

impl MaterialHandle {
    pub fn index(self) -> u32 {
        self.0
    }
}

impl Scene {
    /// Shows or hides an instance (hidden instances are skipped by every pass).
    pub fn set_visible(&mut self, handle: InstanceHandle, visible: bool) {
        let instance = &mut self.instances[handle.0 as usize];
        if instance.visible != visible {
            instance.visible = visible;
            self.instance_version = crate::scene::next_version();
        }
    }

    pub fn is_visible(&self, handle: InstanceHandle) -> bool {
        self.instances[handle.0 as usize].visible
    }

    /// Whether the instance casts shadows ([`Scene::set_cast_shadow`]).
    pub fn casts_shadow(&self, handle: InstanceHandle) -> bool {
        self.instances[handle.0 as usize].cast_shadow
    }

    pub fn instance_mesh(&self, handle: InstanceHandle) -> MeshHandle {
        self.instances[handle.0 as usize].mesh
    }

    pub fn instance_material(&self, handle: InstanceHandle) -> MaterialHandle {
        self.instances[handle.0 as usize].material
    }

    /// Assigns another material to an instance (the mesh stays: it fixes the instance's slot
    /// layout in the culling buffers).
    pub fn set_instance_material(&mut self, handle: InstanceHandle, material: MaterialHandle) {
        let instance = &mut self.instances[handle.0 as usize];
        if instance.material != material {
            instance.material = material;
            self.instance_version = crate::scene::next_version();
        }
    }

    /// Every instance handle, in order.
    pub fn instance_handles(&self) -> impl Iterator<Item = InstanceHandle> {
        (0..self.instances.len() as u32).map(InstanceHandle)
    }

    pub fn instance_handle(&self, index: u32) -> Option<InstanceHandle> {
        ((index as usize) < self.instances.len()).then_some(InstanceHandle(index))
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn mesh_handle(&self, index: u32) -> Option<MeshHandle> {
        ((index as usize) < self.meshes.len()).then_some(MeshHandle(index))
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    pub fn material_handle(&self, index: u32) -> Option<MaterialHandle> {
        ((index as usize) < self.materials.len()).then_some(MaterialHandle(index))
    }

    pub fn material(&self, handle: MaterialHandle) -> &Material {
        &self.materials[handle.0 as usize]
    }

    /// Object-space bounds of a mesh.
    pub fn mesh_bounds(&self, mesh: MeshHandle) -> Aabb {
        self.meshes[mesh.0 as usize].aabb
    }

    /// World-space axis-aligned bounds of an instance (its mesh bounds, transformed).
    pub fn instance_bounds(&self, handle: InstanceHandle) -> (Vec3, Vec3) {
        let instance = &self.instances[handle.0 as usize];
        transformed_bounds(&self.meshes[instance.mesh.0 as usize].aabb, &instance.transform)
    }

    /// The finest in-memory level of a mesh as world-space triangles of `transform` (for
    /// scattering on it; pages that are streamed and not in memory are left out).
    pub fn mesh_triangles(&self, mesh: MeshHandle, transform: &Mat4) -> Vec<[Vec3; 3]> {
        let (vertices, indices) = crate::translucency::decode_finest(&self.meshes[mesh.0 as usize]);
        let at = |i: u32| transform.transform_point3(Vec3::from(vertices[i as usize].position));
        indices.chunks_exact(3).map(|t| [at(t[0]), at(t[1]), at(t[2])]).collect()
    }

    pub fn set_mesh_name(&mut self, mesh: MeshHandle, name: &str) {
        set_name(&mut self.names.meshes, mesh.0 as usize, name);
    }

    /// The mesh's name (empty if unnamed).
    pub fn mesh_name(&self, mesh: MeshHandle) -> &str {
        self.names.meshes.get(mesh.0 as usize).map_or("", String::as_str)
    }

    pub fn find_mesh(&self, name: &str) -> Option<MeshHandle> {
        find_name(&self.names.meshes, name).map(MeshHandle)
    }

    pub fn set_material_name(&mut self, material: MaterialHandle, name: &str) {
        set_name(&mut self.names.materials, material.0 as usize, name);
    }

    /// The material's name (empty if unnamed).
    pub fn material_name(&self, material: MaterialHandle) -> &str {
        self.names.materials.get(material.0 as usize).map_or("", String::as_str)
    }

    pub fn find_material(&self, name: &str) -> Option<MaterialHandle> {
        find_name(&self.names.materials, name).map(MaterialHandle)
    }
}

/// World bounds of `aabb` under `transform` (the box of its eight transformed corners).
pub fn transformed_bounds(aabb: &Aabb, transform: &Mat4) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for k in 0..8 {
        let corner = Vec3::new(
            if k & 1 == 0 { aabb.min.x } else { aabb.max.x },
            if k & 2 == 0 { aabb.min.y } else { aabb.max.y },
            if k & 4 == 0 { aabb.min.z } else { aabb.max.z },
        );
        let p = transform.transform_point3(corner);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use helios_geometry::VirtualMesh;

    #[test]
    fn hiding_keeps_handles_and_bumps_the_instance_version() {
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(VirtualMesh::default());
        let material = scene.add_material(Material::default());
        let a = scene.add_instance(mesh, material, Mat4::IDENTITY);
        let b = scene.add_instance(mesh, material, Mat4::from_translation(Vec3::X));
        let version = scene.instance_version;
        scene.set_visible(a, false);
        assert!(!scene.is_visible(a) && scene.is_visible(b));
        assert_ne!(scene.instance_version, version);
        // Unchanged state does not invalidate anything.
        let version = scene.instance_version;
        scene.set_visible(a, false);
        assert_eq!(scene.instance_version, version);
        assert_eq!(scene.transform(b), Mat4::from_translation(Vec3::X));
        assert_eq!(scene.instance_count(), 2);
    }

    #[test]
    fn names_are_found_and_default_to_empty() {
        let mut scene = Scene::new();
        let first = scene.add_material(Material::default());
        let gold = scene.add_material(Material::default());
        scene.set_material_name(gold, "gold");
        assert_eq!(scene.material_name(first), "");
        assert_eq!(scene.material_name(gold), "gold");
        assert_eq!(scene.find_material("gold"), Some(gold));
        assert_eq!(scene.find_material("silver"), None);
    }

    #[test]
    fn bounds_follow_the_transform() {
        let aabb = Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) };
        let (lo, hi) = transformed_bounds(&aabb, &Mat4::from_scale_rotation_translation(Vec3::new(2.0, 1.0, 1.0), glam::Quat::IDENTITY, Vec3::Y));
        assert_eq!((lo, hi), (Vec3::new(-2.0, 0.0, -1.0), Vec3::new(2.0, 2.0, 1.0)));
    }
}
