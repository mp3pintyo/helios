//! Assemblies (Nanite assemblies, UE 5.7): a mesh made of instanced *parts* — branches, leaf
//! sprays — placed by transforms, instead of one mesh that bakes every copy.
//!
//! Up close the renderer draws the parts themselves, each with its own LOD DAG, transformed
//! during culling; at a distance it draws the assembly's *base*: one ordinary virtual mesh
//! built from the parts' coarsest representations (their DAG root clusters) merged in place, so
//! the parts' copies collapse into a single simplified (or voxelised) whole. The switch is a
//! LOD transition like any other: the base's finest level *is* the parts at their roots —
//! identical geometry — so its error is the worst root error of the parts (`transition_error`).
//! The base's own errors are measured from that level, so every base error is offset by it
//! (the true error of a simplification of an approximation is at most the sum): the base's
//! errors stay monotonic, start where the parts end, and the renderer draws the base exactly
//! when its finest level is detailed enough for the whole assembly, the parts otherwise.
//!
//! Memory and baking scale with the distinct parts and the base (built from the parts' roots),
//! not with the number of copies.

use glam::{Mat3, Mat4, Vec3};

use crate::dag::{build_virtual_mesh, BuildConfig, VirtualMesh, ROOT_ERROR};
use crate::plant_bones::{BoneBounds, PlantBone, PlantBones};
use crate::{Mesh, Sphere, Vertex};

/// The largest scale factor of an affine transform's axes.
pub fn max_scale(transform: &Mat4) -> f32 {
    let m = Mat3::from_mat4(*transform);
    m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length())
}

/// The coarsest representation of `mesh` (its DAG root clusters) as a mesh: what it looks like
/// at its root error.
pub fn root_mesh(mesh: &VirtualMesh) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for c in (0..mesh.clusters.len()).filter(|&c| mesh.clusters[c].parent_error >= ROOT_ERROR && mesh.clusters[c].in_mode(false)) {
        let base = vertices.len() as u32;
        for i in 0..mesh.clusters[c].vertex_count() {
            let (p, n, uv) = mesh.cluster_vertex(c, i);
            vertices.push(Vertex::new(p, n, uv));
        }
        indices.extend(mesh.cluster_triangles(c).flatten().map(|i| base + i));
    }
    let mut out = Mesh::new(vertices, indices);
    out.weld();
    out
}

/// Error of a part's root clusters in assembly units.
fn root_error(mesh: &VirtualMesh, transform: &Mat4) -> f32 {
    let roots = mesh.clusters.iter().filter(|c| c.parent_error >= ROOT_ERROR && c.in_mode(false));
    roots.map(|c| c.lod_error).fold(0.0, f32::max) * max_scale(transform)
}

/// An assembly's base: the parts (`(mesh, transform)`, transforms in assembly space) at their
/// roots, merged and built into a virtual mesh with `config`, its errors offset by the
/// parts' worst root error (see the module documentation). Also returns that error.
pub fn build_assembly_base(parts: &[(&VirtualMesh, Mat4)], config: &BuildConfig) -> (VirtualMesh, f32) {
    let mut merged = crate::modeling::MeshBuilder::default();
    let mut roots = std::collections::HashMap::new();
    let mut transition = 0.0f32;
    for (mesh, transform) in parts {
        // Parts usually share a few meshes: decode each one's roots once.
        let root = roots.entry(*mesh as *const VirtualMesh).or_insert_with(|| root_mesh(mesh));
        merged.append(&root.transformed(*transform));
        transition = transition.max(root_error(mesh, transform));
    }
    let mut base = build_virtual_mesh(&merged.build(), config);
    offset_errors(&mut base, transition);
    (base, transition)
}

/// [`build_assembly_base`] for a plant whose parts hang from its bones (`plant`, the wood's):
/// `parts` are `(part mesh, its source mesh, transform, plant bone)`. The base's vertices follow
/// their part's bone; its bounds ([`crate::BoneBounds`]) hold every part at full detail, whose
/// own bones (a spray's leaves) hang from that bone too ([`crate::plant_bones::part_bone_table`]),
/// since the parts are culled with the base's wind bounds.
pub fn build_plant_assembly_base(parts: &[(&VirtualMesh, &Mesh, Mat4, u16)], plant: &[PlantBone], config: &BuildConfig) -> (VirtualMesh, f32) {
    let mut merged = Mesh::default();
    let mut roots = std::collections::HashMap::new();
    let mut transition = 0.0f32;
    let mut points = Vec::new();
    for &(mesh, source, transform, bone) in parts {
        let root = roots.entry(mesh as *const VirtualMesh).or_insert_with(|| root_mesh(mesh));
        let mut copy = root.transformed(transform);
        crate::plant_bones::bind_all(&mut copy, bone);
        merged.append(&copy, Mat4::IDENTITY);
        transition = transition.max(root_error(mesh, &transform));
        let table = crate::plant_bones::part_bone_table(&source.bones, plant, Some(bone), &transform);
        let scale = max_scale(&transform);
        let own = |i: usize| if source.bones.is_empty() { source.bones.len() as u16 } else { source.skin_of(i).dominant_joint() };
        for (i, v) in source.vertices.iter().enumerate() {
            let reach = crate::plant_bones::vertex_reach(&table, own(i), v.position()) * scale;
            points.push((transform.transform_point3(v.position()), reach));
        }
    }
    merged.bones = plant.to_vec();
    for (v, s) in merged.vertices.iter().zip(&merged.skin) {
        points.push((v.position(), crate::plant_bones::vertex_reach(plant, s.dominant_joint(), v.position())));
    }
    let mut base = build_virtual_mesh(&merged, config);
    base.bones = Some(PlantBones { bones: plant.to_vec(), bounds: BoneBounds::fit(&points) });
    offset_errors(&mut base, transition);
    (base, transition)
}

/// Adds `offset` to every error of `mesh` (its LOD values become relative to a representation
/// whose own error is `offset`).
fn offset_errors(mesh: &mut VirtualMesh, offset: f32) {
    let add = |e: f32| if e >= ROOT_ERROR { e } else { e + offset };
    for c in &mut mesh.clusters {
        c.lod_error = add(c.lod_error);
        c.parent_error = add(c.parent_error);
    }
    for p in &mut mesh.pages {
        p.max_parent_error = add(p.max_parent_error);
    }
}

/// The error of an assembly base's finest level: below it, the renderer draws the parts.
pub fn transition_error(base: &VirtualMesh) -> f32 {
    base.clusters.iter().filter(|c| c.lod_level() == 0).map(|c| c.lod_error).fold(0.0, f32::max)
}

/// Bounding sphere of the parts' full detail in assembly space.
pub fn parts_bounds(parts: &[(&VirtualMesh, Mat4)]) -> Sphere {
    Sphere::merge_all(parts.iter().map(|(mesh, transform)| {
        Sphere::new(transform.transform_point3(mesh.bounds.center), mesh.bounds.radius * max_scale(transform))
    }))
}

/// Centre of an assembly part's bounds (assembly space).
pub fn part_center(mesh: &VirtualMesh, transform: &Mat4) -> Vec3 {
    transform.transform_point3(mesh.bounds.center)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{lod_cut, LodView};
    use crate::sdf::SdfMode;
    use crate::{primitives, VoxelConfig};

    fn view(distance: f32) -> LodView {
        LodView { position: Vec3::new(0.0, 0.5, distance), lod_scale: 1000.0, threshold: 1.0, orthographic: false }
    }

    /// The base starts where the parts end: its finest level is the parts' roots (the same
    /// triangles), its errors are monotonic and at least the transition error, and it is
    /// drawn exactly from the distance at which the parts would be at their roots.
    #[test]
    fn base_continues_the_parts_lod() {
        let config = BuildConfig { sdf: SdfMode::None, ..Default::default() };
        let part = build_virtual_mesh(&primitives::leaf_cloud(300, 4).transformed(Mat4::from_scale(Vec3::splat(0.3))), &config);
        let transforms: Vec<Mat4> = (0..24)
            .map(|i| Mat4::from_scale_rotation_translation(Vec3::splat(0.8 + (i % 3) as f32 * 0.2), glam::Quat::from_rotation_y(i as f32), Vec3::new((i % 6) as f32 * 0.7, (i / 6) as f32 * 0.6, 0.0)))
            .collect();
        let parts: Vec<(&VirtualMesh, Mat4)> = transforms.iter().map(|&t| (&part, t)).collect();
        let (base, transition) = build_assembly_base(&parts, &BuildConfig { voxels: Some(VoxelConfig::default()), ..config });
        assert!(transition > 0.0);
        assert_eq!(transition_error(&base), transition);
        let root_triangles: usize = parts.iter().map(|(m, _)| root_mesh(m).triangle_count()).sum();
        assert_eq!(base.triangle_count(), root_triangles, "the base's finest level is the parts' roots");
        for c in &base.clusters {
            assert!(c.lod_error >= transition && c.parent_error >= c.lod_error);
        }
        // The farther, the coarser the base; the parts take over exactly where the base's finest
        // level is too coarse.
        let mut previous = usize::MAX;
        for distance in [5.0f32, 20.0, 80.0, 320.0] {
            let cut = lod_cut(&base, &view(distance));
            let triangles: usize = cut.iter().map(|&c| base.clusters[c as usize].triangle_count() as usize).sum();
            assert!(triangles <= previous);
            previous = triangles;
        }
        assert!(previous < root_triangles / 4, "{previous} of {root_triangles}");
        let bounds = parts_bounds(&parts);
        let distance = transition * 1000.0 + bounds.radius + 0.5;
        let near = LodView { position: bounds.center + Vec3::Z * (distance * 0.9), ..view(0.0) };
        assert!(near.projected_error(transition, bounds.to_vec4().into()) > 1.0, "parts drawn");
        let far = LodView { position: bounds.center + Vec3::Z * (distance * 1.1), ..view(0.0) };
        assert!(far.projected_error(transition, bounds.to_vec4().into()) <= 1.0, "base drawn");
    }
}
