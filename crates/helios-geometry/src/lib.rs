//! # helios-geometry
//!
//! CPU-side virtual geometry pipeline, modelled on the ideas behind Unreal Engine's Nanite:
//!
//! 1. **Clustering** – the source mesh is split into small triangle clusters
//!    (≤ 128 triangles / ≤ 128 vertices), the unit of GPU culling and LOD.
//! 2. **Grouping** – a multilevel (METIS-style) graph partition merges neighbouring clusters
//!    into groups of ~8 with minimal shared border.
//! 3. **Simplification** – each group is simplified to half its triangle count with
//!    a quadric error metric while *locking the group border*, then re-split into clusters.
//! 4. **DAG** – repeating 2–3 builds a directed acyclic graph of LODs. Every cluster stores
//!    its own error/bounds and its parent's error/bounds; both are monotonic, so the GPU can
//!    pick a crack-free LOD cut by testing every cluster *independently*.
//!
//! 5. **Pages** – member sets of groups are packed into streaming pages with explicit
//!    dependencies, so the GPU can render whatever subset is resident without cracks.
//! 6. **Compression** – cluster vertices are quantised to per-mesh grids and bit-packed
//!    ([`encoding`]), on the GPU as on disk.
//! 7. **Distance fields** – a narrow-band SDF (or heightfield) per mesh for software ray tracing.
//! 8. **Cards** – orthographic depth/normal captures of a mesh's surfaces ([`cards`]) for a
//!    card-based surface cache.
//!
//! Aggregate geometry (foliage) gets **voxel LODs** where voxels beat its simplified triangles
//! ([`voxel`]), preserving coverage at a distance, and meshes made of many copies of a few parts
//! can be **assemblies** ([`assembly`]).
//!
//! The output, [`VirtualMesh`], is a set of flat, GPU-ready arrays.

pub mod assembly;
mod bounds;
pub mod cards;
mod cluster;
mod dag;
pub mod encoding;
mod hash;
mod mesh;
pub mod mesh_terrain;
pub mod modeling;
mod morton;
pub mod noise;
mod partition;
pub mod plant_bones;
pub mod primitives;
pub mod progress;
mod sdf;
mod simplify;
pub mod skin;
pub mod vegetation;
pub mod voxel;

pub use bounds::{Aabb, Sphere};
pub use cards::{Card, CardConfig, MeshCards};
pub use cluster::{build_clusters, ClusterLimits};
pub use dag::{
    build_virtual_mesh, lod_cut, lod_cut_in, lod_cut_resident, BuildConfig, GpuCluster, GpuPage, LodView, VirtualMesh, VirtualMeshStats,
    CLUSTER_VOXEL_ALTERNATE, INVALID_PAGE, ROOT_ERROR,
};
pub use voxel::{Voxel, VoxelConfig};
pub use encoding::{decode_octahedral, encode_octahedral, Quantization, VertexAttributes};
pub use mesh::{canonical_ids, Mesh, Vertex};
pub use partition::{group_clusters, partition_graph, Graph};
pub use sdf::{MeshSdf, SdfKind, SdfMode};
pub use simplify::{simplify, NewVertex, SimplifyMesh, SimplifyOptions, SimplifyResult};
pub use skin::SkinWeights;
pub use plant_bones::{BoneBounds, BoneClass, PlantBone, PlantBones, PlantRig};
