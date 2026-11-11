//! Cluster LOD DAG construction and the streamable, GPU-ready [`VirtualMesh`] format.
//!
//! ## LOD selection rule
//!
//! Every cluster `c` produced by simplifying group `G` stores `G`'s error and bounds as its
//! *own* LOD values; every cluster that was *consumed* by `G` stores them as its *parent*
//! values. A cluster is drawn iff
//!
//! ```text
//! projected(c.lod_error, c.lod_bounds) <= threshold < projected(c.parent_error, c.parent_lod_bounds)
//! ```
//!
//! Because errors and bounds grow monotonically up the DAG, exactly one side of every group
//! boundary passes this test, so the selected clusters always form a watertight surface.
//!
//! ## Streaming pages
//!
//! Geometry is split into **pages**. A page holds the *member clusters* of one or more groups
//! (all clusters a group consumed), never part of a group, so a group's children are either all
//! resident or all absent. Root clusters live in always-resident root pages. Each cluster
//! records `child_page`: the page holding the clusters it was simplified from. With residency
//! the rule becomes
//!
//! ```text
//! resident(page) && (own_ok || !resident(child_page)) && !parent_ok
//! ```
//!
//! — a cluster whose children are missing simply stays the finest available level, and the
//! cut is still watertight as long as a page is only resident when its parent pages are.
//!
//! ## Voxel LODs
//!
//! For aggregate geometry ([`BuildConfig::voxels`]) every group also gets a voxel candidate
//! ([`crate::voxel`]): the full-detail triangles under it, voxelised as finely as its budget
//! allows. Where that error is below the simplified triangles' (typically where component
//! pruning starts deleting leaves), the group switches to voxels — at the triangles' error, or
//! the voxels' if the budget forces coarser ones — and each output cluster gets *voxel
//! alternates* standing for the same source triangles, with identical LOD values; every group
//! above it is then voxelised too. Both representations stay in the DAG: with voxel LODs on, a
//! triangle cluster that has alternates is replaced by them ([`GpuCluster::in_mode`]); off, the
//! voxel clusters are ignored and the cut is the triangle DAG's. Either way every group keeps one
//! error and one sphere, so the selection rule — and its invariants — hold in both modes. Member
//! sets of voxel groups go to pages of their own per mode, so a mode never streams the other's
//! data (except at the groups where voxels take over, whose shared members both modes need).

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

use crate::encoding::{decode_triangle, encode_cluster, ClusterEncoding, Quantization, QuantizedVertex, VertexAttributes};
use crate::hash::{FxHashMap, FxHashSet};
use crate::plant_bones::PlantBones;
use crate::sdf::{MeshSdf, SdfMode};
use crate::simplify::{simplify, SimplifyMesh, SimplifyOptions};
use crate::voxel::{
    build_voxel_clusters, decode_voxel_cluster, fit_grid, fit_grid_at, Cell, Grid, Pieces, SourceTriangle, Voxel, VoxelCluster, VoxelConfig, VoxelSource,
    VOXEL_ERROR_PER_SIZE,
};
use crate::{build_clusters, canonical_ids, group_clusters, Aabb, ClusterLimits, Mesh, SkinWeights, Sphere};

/// Error value used for "no parent" (DAG roots). Finite to stay well-behaved in shaders.
pub const ROOT_ERROR: f32 = 3.0e38;
/// Marker for "no page" / "no child page".
pub const INVALID_PAGE: u32 = u32::MAX;

/// GPU cluster record (80 bytes, std430 compatible). Always resident.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuCluster {
    /// Culling sphere enclosing this cluster's triangles (object space).
    pub bounds: [f32; 4],
    /// LOD sphere shared by all clusters produced from the same group.
    pub lod_bounds: [f32; 4],
    /// LOD sphere of the group that consumed this cluster.
    pub parent_lod_bounds: [f32; 4],
    pub lod_error: f32,
    pub parent_error: f32,
    /// Mesh-local page holding this cluster's geometry.
    pub page: u32,
    /// Mesh-local page holding the clusters this one was simplified from ([`INVALID_PAGE`] at level 0).
    pub child_page: u32,
    /// Word offset of the cluster's encoded vertices (header first, see [`crate::encoding`])
    /// inside the page payload.
    pub vertex_offset: u32,
    /// Word offset of the triangle bit stream inside the page payload (voxel clusters: of their
    /// voxel attributes).
    pub triangle_offset: u32,
    /// `vertex_count | triangle_count << 8 | lod_level << 16` (voxel clusters: no vertices or
    /// triangles).
    pub counts: u32,
    /// Voxel clusters ([`crate::voxel`]): `bricks | voxels << 8`. Triangle clusters: 0, or
    /// [`CLUSTER_VOXEL_ALTERNATE`] if voxel clusters replace them when voxel LODs are on.
    pub voxel: u32,
}

/// [`GpuCluster::voxel`] bit of triangle clusters that have voxel alternates.
pub const CLUSTER_VOXEL_ALTERNATE: u32 = 1 << 31;

impl GpuCluster {
    pub fn vertex_count(&self) -> u32 {
        self.counts & 0xff
    }
    pub fn triangle_count(&self) -> u32 {
        (self.counts >> 8) & 0xff
    }
    pub fn lod_level(&self) -> u32 {
        self.counts >> 16
    }
    pub fn is_voxel(&self) -> bool {
        self.voxel_bricks() != 0
    }
    pub fn voxel_bricks(&self) -> u32 {
        self.voxel & 0xff
    }
    pub fn voxel_count(&self) -> u32 {
        (self.voxel >> 8) & 0x3fff
    }
    /// A triangle cluster that voxel clusters replace when voxel LODs are on.
    pub fn has_voxel_alternate(&self) -> bool {
        self.voxel & CLUSTER_VOXEL_ALTERNATE != 0
    }
    /// Whether the cluster takes part in the LOD cut with voxel LODs on (`voxels`) or off.
    pub fn in_mode(&self, voxels: bool) -> bool {
        if voxels {
            !self.has_voxel_alternate()
        } else {
            !self.is_voxel()
        }
    }
}

/// GPU page record (64 bytes). Always resident; drives hierarchical culling — when even the
/// coarsest parent in the page is detailed enough, the whole page is skipped.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuPage {
    pub bounds: [f32; 4],
    pub parent_lod_bounds: [f32; 4],
    pub max_parent_error: f32,
    /// First cluster of the page (mesh-local; clusters are stored page by page).
    pub cluster_offset: u32,
    pub cluster_count: u32,
    pub lod_level: u32,
    /// Payload size in words.
    pub data_words: u32,
    /// 1 for root pages (always resident).
    pub root: u32,
    pub _pad: [u32; 2],
}

/// Build parameters.
#[derive(Clone, Copy, Debug)]
pub struct BuildConfig {
    pub limits: ClusterLimits,
    /// Number of clusters merged into a group before simplification.
    pub group_size: usize,
    /// Maximum clusters per streaming page.
    pub page_clusters: usize,
    /// A group whose simplification keeps more than this fraction of its triangles is
    /// considered stuck and its clusters become DAG roots.
    pub min_reduction: f32,
    pub simplify: SimplifyOptions,
    pub sdf: SdfMode,
    /// Position grid step `2^-precision` (mesh units) for the compressed vertex format; `None`
    /// picks it from the triangle size (see [`Quantization::new`]).
    pub position_precision: Option<i32>,
    /// Aggregate geometry (foliage, grass): offer voxel clusters where they beat the simplified
    /// triangles (see the module documentation).
    pub voxels: Option<VoxelConfig>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            limits: ClusterLimits::default(),
            group_size: 8,
            page_clusters: 64,
            min_reduction: 0.85,
            simplify: SimplifyOptions::default(),
            sdf: SdfMode::default(),
            position_precision: None,
            voxels: None,
        }
    }
}

/// Per-level summary of a built mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VirtualMeshStats {
    /// Clusters of both representations (triangles and voxels).
    pub clusters_per_level: Vec<u32>,
    pub triangles_per_level: Vec<u32>,
    pub voxels_per_level: Vec<u32>,
}

impl VirtualMeshStats {
    pub fn level_count(&self) -> usize {
        self.clusters_per_level.len()
    }
}

/// Streamable virtual geometry: always-resident cluster and page records plus per-page
/// GPU-format payloads.
#[derive(Clone, Debug, Default)]
pub struct VirtualMesh {
    /// All clusters, stored page by page.
    pub clusters: Vec<GpuCluster>,
    pub pages: Vec<GpuPage>,
    /// Word offset of every page payload in the full payload section (`pages.len() + 1` entries).
    pub page_offsets: Vec<u32>,
    /// Payloads of pages `payload_first_page..`, i.e. of every page for a baked mesh. A mesh
    /// loaded with [`VirtualMesh::from_header_bytes`] keeps only its trailing root pages here
    /// (see [`VirtualMesh::set_tail_payload`]); the rest streams from [`VirtualMesh::page_byte_range`].
    pub page_data: Vec<u32>,
    /// First page whose payload is held in `page_data`.
    pub payload_first_page: u32,
    /// Pages that must be resident before page `i` may be (`dependency_offsets[i]..[i+1]`).
    pub dependency_offsets: Vec<u32>,
    pub dependencies: Vec<u32>,
    pub bounds: Sphere,
    pub aabb: Aabb,
    pub stats: VirtualMeshStats,
    pub sdf: Option<MeshSdf>,
    /// The bones of a procedural plant ([`Mesh::bones`]) and the bounds of their motion: the
    /// clusters' skin attribute indexes them (`.hvm` version 6).
    pub bones: Option<PlantBones>,
}

/// Which LOD cuts a cluster takes part in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ClusterKind {
    /// Both (triangles without voxel alternates).
    #[default]
    Shared,
    /// Triangles, replaced by voxel clusters when voxel LODs are on.
    Alternated,
    /// Voxels, standing in for an alternated triangle cluster.
    Voxel,
}

struct WorkCluster {
    triangles: Vec<[u32; 3]>,
    bounds: Sphere,
    lod_bounds: Sphere,
    lod_error: f32,
    parent_lod_bounds: Sphere,
    parent_error: f32,
    level: u32,
    /// Group that produced this cluster (`None` at level 0).
    source_group: Option<u32>,
    /// Group that consumed it (`None` for roots).
    parent_group: Option<u32>,
    kind: ClusterKind,
    /// Voxel meshes only: the full-detail triangles this triangle cluster stands for (taken by
    /// the group that consumes it).
    region: Vec<SourceTriangle>,
    /// Its voxel alternates (indices into the work list).
    alternates: Vec<usize>,
    /// The payload of a voxel cluster.
    voxels: Option<VoxelCluster>,
}

impl WorkCluster {
    fn new(positions: &[Vec3], triangles: Vec<[u32; 3]>, level: u32) -> Self {
        let points: Vec<Vec3> = triangles.iter().flatten().map(|&v| positions[v as usize]).collect();
        let bounds = Sphere::from_points(&points);
        Self {
            triangles,
            bounds,
            lod_bounds: bounds,
            lod_error: 0.0,
            parent_lod_bounds: bounds,
            parent_error: ROOT_ERROR,
            level,
            source_group: None,
            parent_group: None,
            kind: ClusterKind::Shared,
            region: Vec::new(),
            alternates: Vec::new(),
            voxels: None,
        }
    }
}

/// A group's voxel representation: its grid, the voxelisation of all its source triangles on it
/// and the resulting error.
struct VoxelCandidate {
    grid: Grid,
    cells: FxHashMap<u64, Cell>,
    error: f32,
}

/// Splits a group's source triangles among its output clusters: each goes to the cluster whose
/// bounding sphere is nearest (aggregate geometry has no connectivity to follow).
fn partition_region(region: Vec<SourceTriangle>, positions: &[Vec3], outputs: &[Sphere]) -> Vec<Vec<SourceTriangle>> {
    let mut parts = vec![Vec::new(); outputs.len()];
    for t in region {
        let centroid = t.vertices.iter().map(|&v| positions[v as usize]).sum::<Vec3>() / 3.0;
        let distance = |s: &Sphere| centroid.distance(s.center) - s.radius;
        let nearest = (0..outputs.len()).min_by(|&a, &b| distance(&outputs[a]).total_cmp(&distance(&outputs[b]))).unwrap_or(0);
        parts[nearest].push(t);
    }
    parts
}

/// Builds the cluster LOD DAG, streaming pages and (optionally) the distance field of `mesh`.
pub fn build_virtual_mesh(mesh: &Mesh, config: &BuildConfig) -> VirtualMesh {
    let mut positions = mesh.positions();
    let mut normals: Vec<Vec3> = mesh.vertices.iter().map(|v| v.normal()).collect();
    let mut uvs: Vec<Vec2> = mesh.vertices.iter().map(|v| v.uv()).collect();
    // Optional attributes (empty when absent) follow their vertices like the UVs.
    let mut uv1 = mesh.uv1.clone();
    let mut colors = mesh.colors.clone();
    let mut skin = mesh.skin.clone();
    let mut plant = mesh.plant.clone();
    let mut canonical = canonical_ids(mesh.vertices.iter().map(|v| v.position));
    let all_triangles: Vec<[u32; 3]> = mesh.triangles().collect();

    // Vertices of stuck groups (which become roots) must never move again.
    let mut locked_forever = vec![false; positions.len()];

    // Voxel meshes: the full-detail geometry every voxel cluster is built from. Skinned meshes
    // never get voxels: voxels carry no skin weights, so they could not follow the pose. Plants
    // do: each brick of voxels follows one bone (`voxel_bone`: the branch its leaves hang from).
    let plant_bones = !mesh.bones.is_empty();
    let voxel_lods = config.voxels.filter(|_| mesh.skin.is_empty() || plant_bones);
    let voxel_bones: Vec<u16> = if plant_bones && voxel_lods.is_some() {
        (0..positions.len()).map(|i| crate::plant_bones::voxel_bone(&mesh.bones, mesh.skin_of(i).dominant_joint().min(mesh.bones.len() as u16 - 1))).collect()
    } else {
        Vec::new()
    };
    let source_attributes = voxel_lods.map(|_| (positions.clone(), normals.clone(), uvs.clone()));
    let voxel_source = source_attributes.as_ref().map(|(positions, normals, uvs)| VoxelSource { positions, normals, uvs, bones: &voxel_bones });

    let mut work: Vec<WorkCluster> = {
        let pieces = voxel_lods.map(|_| Pieces::new(&all_triangles, &canonical));
        build_clusters(&positions, &canonical, &all_triangles, config.limits)
            .into_iter()
            .map(|tris| {
                let mut cluster = WorkCluster::new(&positions, tris, 0);
                if let Some(pieces) = &pieces {
                    cluster.region = cluster.triangles.iter().map(|&t| pieces.source(t)).collect();
                }
                cluster
            })
            .collect()
    };
    let mut current: Vec<usize> = (0..work.len()).collect();
    // Groups whose outputs have voxel alternates.
    let mut voxel_groups: Vec<bool> = Vec::new();

    let mut class_group: FxHashMap<u32, u32> = FxHashMap::default();
    let mut shared: FxHashSet<u32> = FxHashSet::default();
    let mut level = 0u32;
    let mut group_count = 0u32;

    while current.len() > 1 {
        level += 1;
        let cluster_tris: Vec<&[[u32; 3]]> = current.iter().map(|&c| work[c].triangles.as_slice()).collect();
        let groups = group_clusters(&positions, &canonical, &cluster_tris, config.group_size);

        // A position referenced by more than one group lies on a group border and is locked.
        class_group.clear();
        shared.clear();
        for (gi, group) in groups.iter().enumerate() {
            for &local in group {
                for &v in work[current[local]].triangles.iter().flatten() {
                    let class = canonical[v as usize];
                    match class_group.get(&class) {
                        None => {
                            class_group.insert(class, gi as u32);
                        }
                        Some(&g) if g != gi as u32 => {
                            shared.insert(class);
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut next: Vec<usize> = Vec::new();
        for group in &groups {
            let members: Vec<usize> = group.iter().map(|&local| current[local]).collect();
            let triangles: Vec<[u32; 3]> = members.iter().flat_map(|&c| work[c].triangles.iter().copied()).collect();
            let result = {
                let is_locked = |v: u32| locked_forever[v as usize] || shared.contains(&canonical[v as usize]);
                simplify(
                    SimplifyMesh { positions: &positions, normals: &normals, uvs: &uvs, canonical: &canonical, skin: &skin },
                    &triangles,
                    &is_locked,
                    triangles.len() / 2,
                    &config.simplify,
                )
            };

            let stuck = result.triangles.is_empty() || result.triangles.len() as f32 > triangles.len() as f32 * config.min_reduction;
            if stuck {
                if std::env::var("HELIOS_DEBUG_DAG").is_ok() {
                    let locked = triangles.iter().flatten().filter(|&&v| locked_forever[v as usize] || shared.contains(&canonical[v as usize])).count();
                    eprintln!("level {level}: stuck group of {} clusters, {} -> {} tris, locked corners {}/{}", members.len(), triangles.len(), result.triangles.len(), locked, triangles.len()*3);
                }
                // These clusters become roots; their vertices must never move again.
                for &v in triangles.iter().flatten() {
                    locked_forever[v as usize] = true;
                }
                continue;
            }

            // Vertices created by pruning (scaled copies) join the shared vertex arrays.
            let base = positions.len() as u32;
            for nv in &result.new_vertices {
                positions.push(nv.position);
                normals.push(normals[nv.source as usize]);
                uvs.push(uvs[nv.source as usize]);
                if !uv1.is_empty() {
                    uv1.push(uv1[nv.source as usize]);
                }
                if !colors.is_empty() {
                    colors.push(colors[nv.source as usize]);
                }
                if !skin.is_empty() {
                    skin.push(skin[nv.source as usize]);
                }
                if !plant.is_empty() {
                    plant.push(plant[nv.source as usize]);
                }
                canonical.push(base + nv.class);
                locked_forever.push(false);
            }

            let group_id = group_count;
            group_count += 1;
            let children_error = members.iter().map(|&c| work[c].lod_error).fold(0.0f32, f32::max);
            let region: Vec<SourceTriangle> = members.iter().flat_map(|&c| std::mem::take(&mut work[c].region)).collect();
            // Voxels compete with the simplified triangles' error at the same budget; once a
            // member has voxel alternates, the group must offer them too (the voxel cut continues
            // upwards).
            let voxels = voxel_lods.zip(voxel_source.as_ref()).and_then(|(voxel_config, source)| {
                let budget = (voxel_config.voxels_per_triangle * (triangles.len() / 2) as f32) as usize;
                let forced = members.iter().any(|&c| work[c].kind == ClusterKind::Alternated);
                let (finest, cells) = fit_grid(source, &region, budget, children_error / VOXEL_ERROR_PER_SIZE)?;
                if !forced && finest.size * VOXEL_ERROR_PER_SIZE >= result.error {
                    return None;
                }
                // No finer than the triangles' error: the group switches where the triangle cut
                // would, with no more voxels than the budget (fewer where the voxels could be
                // finer than that).
                let size = finest.size.max(result.error / VOXEL_ERROR_PER_SIZE);
                let (grid, cells) = if size > finest.size { fit_grid_at(source, &region, size) } else { (finest, cells) };
                Some(VoxelCandidate { grid, cells, error: grid.size * VOXEL_ERROR_PER_SIZE })
            });
            let group_error = voxels.as_ref().map_or(result.error, |v| v.error).max(children_error);
            let group_bounds = Sphere::merge_all(members.iter().map(|&c| work[c].lod_bounds));
            for &c in &members {
                for i in std::iter::once(c).chain(work[c].alternates.clone()) {
                    work[i].parent_error = group_error;
                    work[i].parent_lod_bounds = group_bounds;
                    work[i].parent_group = Some(group_id);
                }
            }
            voxel_groups.push(voxels.is_some());

            let outputs = build_clusters(&positions, &canonical, &result.triangles, config.limits);
            let first_output = work.len();
            for tris in outputs {
                let mut cluster = WorkCluster::new(&positions, tris, level);
                cluster.lod_error = group_error;
                cluster.lod_bounds = group_bounds;
                cluster.source_group = Some(group_id);
                if voxels.is_some() {
                    cluster.kind = ClusterKind::Alternated;
                }
                next.push(work.len());
                work.push(cluster);
            }
            if let Some(source) = &voxel_source {
                let spheres: Vec<Sphere> = work[first_output..].iter().map(|w| w.bounds).collect();
                for (o, part) in partition_region(region, &positions, &spheres).into_iter().enumerate() {
                    let output = first_output + o;
                    if let Some(candidate) = &voxels {
                        for payload in build_voxel_clusters(source, &part, candidate.grid, &candidate.cells) {
                            let index = work.len();
                            work[output].alternates.push(index);
                            work.push(WorkCluster {
                                bounds: payload.bounds,
                                lod_error: group_error,
                                lod_bounds: group_bounds,
                                source_group: Some(group_id),
                                kind: ClusterKind::Voxel,
                                voxels: Some(payload),
                                ..WorkCluster::new(&positions, Vec::new(), level)
                            });
                        }
                    }
                    work[output].region = part;
                }
            }
        }

        if next.is_empty() {
            break;
        }
        current = next;
    }

    let vertices = Vertices { positions: &positions, normals: &normals, uvs: &uvs, uv1: &uv1, colors: &colors, skin: &skin, plant: &plant };
    let mut vm = pack(&vertices, &work, &voxel_groups, config);
    vm.sdf = bake_sdf(&vm, config.sdf);
    if plant_bones {
        vm.bones = Some(PlantBones::of_mesh(mesh, mesh.bones.clone()));
    }
    vm
}

struct Vertices<'a> {
    positions: &'a [Vec3],
    normals: &'a [Vec3],
    uvs: &'a [Vec2],
    /// Optional attributes: empty or one per vertex.
    uv1: &'a [Vec2],
    colors: &'a [Vec4],
    skin: &'a [SkinWeights],
    plant: &'a [Vec2],
}

impl Vertices<'_> {
    /// The mesh's quantisation grids: the position step follows the mean shortest edge of the
    /// full-detail triangles, so its rounding error stays far below the finest LOD error.
    fn quantization(&self, work: &[WorkCluster], precision: Option<i32>) -> Quantization {
        let (mut length, mut triangles) = (0.0f64, 0usize);
        for t in work.iter().filter(|w| w.level == 0).flat_map(|w| &w.triangles) {
            let p = t.map(|v| self.positions[v as usize]);
            length += p[0].distance(p[1]).min(p[1].distance(p[2])).min(p[2].distance(p[0])) as f64;
            triangles += 1;
        }
        let max_abs = |values: &mut dyn Iterator<Item = f32>| values.fold(0.0f32, |m, v| m.max(v.abs()));
        Quantization::new(
            precision,
            (length / triangles.max(1) as f64) as f32,
            max_abs(&mut self.positions.iter().flat_map(|p| p.to_array())),
            max_abs(&mut self.uvs.iter().chain(self.uv1).flat_map(|uv| uv.to_array())),
        )
        .with_attributes(VertexAttributes { uv1: !self.uv1.is_empty(), color: !self.colors.is_empty(), skin: !self.skin.is_empty(), plant: !self.plant.is_empty() })
    }

    /// Vertex `v` on the grids of `quantization`.
    fn quantized(&self, quantization: &Quantization, v: usize) -> QuantizedVertex {
        let mut q = quantization.vertex(self.positions[v], self.normals[v], self.uvs[v]);
        if let Some(&uv1) = self.uv1.get(v) {
            q = q.with_uv1(quantization, uv1);
        }
        if let Some(&color) = self.colors.get(v) {
            q = q.with_color(color);
        }
        if let Some(&skin) = self.skin.get(v) {
            q = q.with_skin(skin);
        }
        if let Some(&plant) = self.plant.get(v) {
            q = q.with_plant(plant);
        }
        q
    }
}

/// The LOD cuts a page's clusters serve: pages only mix member sets of the same kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageMode {
    Both = 0,
    Triangles = 1,
    Voxels = 2,
}

fn pack(vertices: &Vertices, work: &[WorkCluster], voxel_groups: &[bool], config: &BuildConfig) -> VirtualMesh {
    let mut out = VirtualMesh {
        bounds: Sphere::from_points(vertices.positions),
        aabb: Aabb::from_points(vertices.positions.iter().copied()),
        ..Default::default()
    };

    // ---- Page assignment: whole member sets, one LOD level per page ----------------------------
    // A group's members in each mode: shared clusters serve both.
    let group_count = voxel_groups.len();
    let mut triangle_members: Vec<Vec<usize>> = vec![Vec::new(); group_count];
    let mut voxel_members: Vec<Vec<usize>> = vec![Vec::new(); group_count];
    let mut shared_members = vec![false; group_count];
    let mut roots: Vec<usize> = Vec::new();
    for (i, w) in work.iter().enumerate() {
        let Some(g) = w.parent_group else {
            roots.push(i);
            continue;
        };
        let g = g as usize;
        match w.kind {
            ClusterKind::Shared => {
                triangle_members[g].push(i);
                shared_members[g] = true;
            }
            ClusterKind::Alternated => triangle_members[g].push(i),
            ClusterKind::Voxel => voxel_members[g].push(i),
        }
    }
    let mut pages: Vec<Vec<usize>> = Vec::new();
    let mut page_is_root: Vec<bool> = Vec::new();
    // Per group: the page of its members in the triangle and in the voxel cut.
    let mut page_of_group = vec![[INVALID_PAGE; 2]; group_count];
    let mut open: [Option<(usize, u32)>; 3] = [None; 3]; // (page, level) per mode
    for g in 0..group_count {
        // A voxel group's outputs point at the mode's own member page; any other group's
        // (shared) outputs at one page holding both.
        let separate = voxel_groups[g] && !shared_members[g] && !voxel_members[g].is_empty();
        let sets = if separate {
            vec![(triangle_members[g].clone(), PageMode::Triangles), (voxel_members[g].clone(), PageMode::Voxels)]
        } else {
            vec![([triangle_members[g].as_slice(), &voxel_members[g]].concat(), PageMode::Both)]
        };
        for (set, mode) in sets {
            if set.is_empty() {
                continue;
            }
            let level = work[set[0]].level;
            let slot = &mut open[mode as usize];
            if !slot.is_some_and(|(p, l)| l == level && pages[p].len() + set.len() <= config.page_clusters) {
                pages.push(Vec::new());
                page_is_root.push(false);
                *slot = Some((pages.len() - 1, level));
            }
            let page = slot.unwrap().0;
            pages[page].extend_from_slice(&set);
            match mode {
                PageMode::Triangles => page_of_group[g][0] = page as u32,
                PageMode::Voxels => page_of_group[g][1] = page as u32,
                PageMode::Both => page_of_group[g] = [page as u32; 2],
            }
        }
    }
    roots.sort_by_key(|&i| work[i].level);
    for chunk in roots.chunks(config.page_clusters.max(1)) {
        pages.push(chunk.to_vec());
        page_is_root.push(true);
    }
    // The page holding the clusters a cluster was built from, in its own cut.
    let child_page = |w: &WorkCluster| w.source_group.map_or(INVALID_PAGE, |g| page_of_group[g as usize][(w.kind == ClusterKind::Voxel) as usize]);

    // ---- Dependencies: a page needs the pages holding the clusters its groups produced -----------
    let mut page_of_cluster = vec![0u32; work.len()];
    for (p, set) in pages.iter().enumerate() {
        for &c in set {
            page_of_cluster[c] = p as u32;
        }
    }
    let mut parents: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); pages.len()];
    for (c, w) in work.iter().enumerate() {
        let child = child_page(w);
        if child != INVALID_PAGE {
            parents[child as usize].insert(page_of_cluster[c]);
        }
    }

    // ---- Payloads ------------------------------------------------------------------------------
    // Every vertex is quantised once, on mesh-wide grids, so all clusters (and LOD levels) that
    // share it encode the same grid point — see `crate::encoding`.
    let quantization = vertices.quantization(work, config.position_precision);
    let quantized: Vec<QuantizedVertex> = (0..vertices.positions.len()).map(|v| vertices.quantized(&quantization, v)).collect();
    let mut local: Vec<u32> = Vec::with_capacity(256);
    let mut local_vertices: Vec<QuantizedVertex> = Vec::with_capacity(256);
    out.page_offsets.push(0);
    out.dependency_offsets.push(0);
    for (p, set) in pages.iter().enumerate() {
        let page_start = out.page_data.len();
        let cluster_offset = out.clusters.len() as u32;
        for &c in set {
            let w = &work[c];
            let vertex_offset = (out.page_data.len() - page_start) as u32;
            let (triangle_offset, counts, voxel) = if let Some(voxels) = &w.voxels {
                out.page_data.extend_from_slice(&voxels.words);
                let attributes = crate::voxel::VOXEL_HEADER_WORDS + voxels.bricks * crate::voxel::BRICK_WORDS;
                (vertex_offset + attributes, w.level << 16, voxels.bricks | voxels.voxels << 8)
            } else {
                local.clear();
                let local_triangles: Vec<[u32; 3]> = w
                    .triangles
                    .iter()
                    .map(|t| {
                        t.map(|v| match local.iter().position(|&x| x == v) {
                            Some(i) => i as u32,
                            None => {
                                local.push(v);
                                local.len() as u32 - 1
                            }
                        })
                    })
                    .collect();
                local_vertices.clear();
                local_vertices.extend(local.iter().map(|&v| quantized[v as usize]));
                let triangle_offset = vertex_offset + encode_cluster(&quantization, &local_vertices, &local_triangles, &mut out.page_data);
                let alternate = if w.kind == ClusterKind::Alternated { CLUSTER_VOXEL_ALTERNATE } else { 0 };
                (triangle_offset, local.len() as u32 | (w.triangles.len() as u32) << 8 | w.level << 16, alternate)
            };

            out.clusters.push(GpuCluster {
                bounds: w.bounds.to_vec4().into(),
                lod_bounds: w.lod_bounds.to_vec4().into(),
                parent_lod_bounds: w.parent_lod_bounds.to_vec4().into(),
                lod_error: w.lod_error,
                parent_error: w.parent_error,
                page: p as u32,
                child_page: child_page(w),
                vertex_offset,
                triangle_offset,
                counts,
                voxel,
            });

            let level = w.level as usize;
            if out.stats.clusters_per_level.len() <= level {
                out.stats.clusters_per_level.resize(level + 1, 0);
                out.stats.triangles_per_level.resize(level + 1, 0);
                out.stats.voxels_per_level.resize(level + 1, 0);
            }
            out.stats.clusters_per_level[level] += 1;
            out.stats.triangles_per_level[level] += w.triangles.len() as u32;
            out.stats.voxels_per_level[level] += w.voxels.as_ref().map_or(0, |v| v.voxels);
        }
        let range = set.iter().map(|&c| &work[c]);
        out.pages.push(GpuPage {
            bounds: Sphere::merge_all(range.clone().map(|w| w.bounds)).to_vec4().into(),
            parent_lod_bounds: Sphere::merge_all(range.clone().map(|w| w.parent_lod_bounds)).to_vec4().into(),
            max_parent_error: range.clone().map(|w| w.parent_error).fold(0.0, f32::max),
            cluster_offset,
            cluster_count: set.len() as u32,
            lod_level: range.map(|w| w.level).max().unwrap_or(0),
            data_words: (out.page_data.len() - page_start) as u32,
            root: page_is_root[p] as u32,
            _pad: [0; 2],
        });
        out.page_offsets.push(out.page_data.len() as u32);
        let mut deps: Vec<u32> = parents[p].iter().copied().collect();
        deps.sort_unstable();
        out.dependencies.extend(deps);
        out.dependency_offsets.push(out.dependencies.len() as u32);
    }
    out
}

/// Bakes the mesh's distance field from a coarse LOD whose error is below half a voxel.
fn bake_sdf(vm: &VirtualMesh, mode: SdfMode) -> Option<MeshSdf> {
    let resolution = match mode {
        SdfMode::None => return None,
        SdfMode::Volume { resolution } | SdfMode::Heightfield { resolution } => resolution.max(8),
    };
    let voxel = vm.aabb.extent().max_element() / resolution as f32;
    let view = LodView { position: Vec3::ZERO, lod_scale: 1.0, threshold: voxel * 0.5, orthographic: true };
    let triangles: Vec<[Vec3; 3]> = lod_cut(vm, &view).iter().flat_map(|&c| vm.cluster_triangle_positions(c as usize)).collect();
    match mode {
        SdfMode::Heightfield { .. } => MeshSdf::bake_heightfield(&triangles, resolution),
        _ => MeshSdf::bake_volume(&triangles, resolution),
    }
}

/// Camera description for CPU-side LOD selection (mirrors the GPU culling shader).
#[derive(Clone, Copy, Debug)]
pub struct LodView {
    pub position: Vec3,
    /// Pixels covered by one world unit at distance 1 (perspective) or anywhere (orthographic).
    pub lod_scale: f32,
    /// Maximum allowed projected error in pixels.
    pub threshold: f32,
    pub orthographic: bool,
}

impl LodView {
    /// Projected error in pixels of `error` with bounds `sphere`.
    pub fn projected_error(&self, error: f32, sphere: [f32; 4]) -> f32 {
        if error <= 0.0 {
            return 0.0;
        }
        if error >= ROOT_ERROR {
            return f32::INFINITY;
        }
        if self.orthographic {
            return error * self.lod_scale;
        }
        let distance = Vec3::new(sphere[0], sphere[1], sphere[2]).distance(self.position) - sphere[3];
        if distance <= 0.0 {
            return f32::INFINITY;
        }
        error * self.lod_scale / distance
    }

    /// The GPU selection rule for a single cluster of the triangle cut, given page residency.
    pub fn selects(&self, cluster: &GpuCluster, resident: &dyn Fn(u32) -> bool) -> bool {
        self.selects_in(cluster, resident, false)
    }

    /// The GPU selection rule with voxel LODs on (`voxels`) or off.
    pub fn selects_in(&self, cluster: &GpuCluster, resident: &dyn Fn(u32) -> bool, voxels: bool) -> bool {
        let own_ok = self.projected_error(cluster.lod_error, cluster.lod_bounds) <= self.threshold;
        let children_missing = cluster.child_page == INVALID_PAGE || !resident(cluster.child_page);
        cluster.in_mode(voxels)
            && resident(cluster.page)
            && (own_ok || children_missing)
            && self.projected_error(cluster.parent_error, cluster.parent_lod_bounds) > self.threshold
    }
}

/// Indices of the clusters of the triangle cut for `view` with every page resident.
pub fn lod_cut(mesh: &VirtualMesh, view: &LodView) -> Vec<u32> {
    lod_cut_resident(mesh, view, &|_| true)
}

/// Indices of the clusters of the triangle cut for `view` given page residency.
pub fn lod_cut_resident(mesh: &VirtualMesh, view: &LodView, resident: &dyn Fn(u32) -> bool) -> Vec<u32> {
    lod_cut_in(mesh, view, resident, false)
}

/// Indices of the clusters selected for `view` given page residency, with voxel LODs on
/// (`voxels`) or off.
pub fn lod_cut_in(mesh: &VirtualMesh, view: &LodView, resident: &dyn Fn(u32) -> bool, voxels: bool) -> Vec<u32> {
    (0..mesh.clusters.len() as u32).filter(|&i| view.selects_in(&mesh.clusters[i as usize], resident, voxels)).collect()
}

impl VirtualMesh {
    /// Full-detail triangle count.
    pub fn triangle_count(&self) -> usize {
        self.stats.triangles_per_level.first().copied().unwrap_or(0) as usize
    }

    /// Whether the payload of `page` is held in memory (as opposed to only in the `.hvm` file).
    pub fn has_payload(&self, page: usize) -> bool {
        page >= self.payload_first_page as usize
    }

    /// Payload words of `page`. Panics if the page is not held in memory ([`VirtualMesh::has_payload`]).
    pub fn page_payload(&self, page: usize) -> &[u32] {
        assert!(self.has_payload(page), "payload of page {page} is not in memory");
        let base = self.page_offsets[self.payload_first_page as usize];
        &self.page_data[(self.page_offsets[page] - base) as usize..(self.page_offsets[page + 1] - base) as usize]
    }

    pub fn page_dependencies(&self, page: usize) -> &[u32] {
        &self.dependencies[self.dependency_offsets[page] as usize..self.dependency_offsets[page + 1] as usize]
    }

    /// Bytes of streamable geometry (all page payloads, whether in memory or not).
    pub fn payload_bytes(&self) -> usize {
        self.page_offsets.last().copied().unwrap_or(0) as usize * 4
    }

    /// Bytes of CPU memory held by this mesh (records, payloads in memory, distance field).
    pub fn cpu_bytes(&self) -> usize {
        use std::mem::size_of_val;
        size_of_val(self.clusters.as_slice())
            + size_of_val(self.pages.as_slice())
            + size_of_val(self.page_offsets.as_slice())
            + size_of_val(self.page_data.as_slice())
            + size_of_val(self.dependency_offsets.as_slice())
            + size_of_val(self.dependencies.as_slice())
            + self.sdf.as_ref().map_or(0, |s| size_of_val(s.words.as_slice()))
    }

    /// The encoding header of `cluster` and its vertex bit stream.
    fn cluster_encoding(&self, cluster: usize) -> (ClusterEncoding, &[u32]) {
        let c = &self.clusters[cluster];
        let data = &self.page_payload(c.page as usize)[c.vertex_offset as usize..];
        let encoding = ClusterEncoding::from_words(data);
        (encoding, &data[encoding.header_words() as usize..])
    }

    /// Decodes vertex `i` of `cluster`: (position, normal, uv), exactly as the GPU does
    /// (positions bit-identically).
    pub fn cluster_vertex(&self, cluster: usize, i: u32) -> (Vec3, Vec3, Vec2) {
        let (encoding, stream) = self.cluster_encoding(cluster);
        let count = self.clusters[cluster].vertex_count();
        (encoding.position(stream, i), encoding.normal(stream, count, i), encoding.uv(stream, count, i))
    }

    /// Optional attributes of vertex `i` of `cluster`: (second UV, linear RGBA colour), with
    /// the defaults of meshes without them (the first UV, white).
    pub fn cluster_vertex_extra(&self, cluster: usize, i: u32) -> (Vec2, Vec4) {
        let (encoding, stream) = self.cluster_encoding(cluster);
        let count = self.clusters[cluster].vertex_count();
        (encoding.uv1(stream, count, i), encoding.color(stream, count, i))
    }

    /// Plant data (wind weight, ambient occlusion) of vertex `i` of `cluster`; (1, 1) for meshes
    /// without it.
    pub fn cluster_vertex_plant(&self, cluster: usize, i: u32) -> Vec2 {
        let (encoding, stream) = self.cluster_encoding(cluster);
        encoding.plant(stream, self.clusters[cluster].vertex_count(), i)
    }

    /// Skin weights of vertex `i` of `cluster` (`None` for meshes without skin).
    pub fn cluster_vertex_skin(&self, cluster: usize, i: u32) -> Option<SkinWeights> {
        let (encoding, stream) = self.cluster_encoding(cluster);
        encoding.skin(stream, self.clusters[cluster].vertex_count(), i)
    }

    /// Whether the mesh has skin weights (a skeletal mesh).
    pub fn is_skinned(&self) -> bool {
        (0..self.clusters.len()).find(|&c| !self.clusters[c].is_voxel() && self.has_payload(self.clusters[c].page as usize)).is_some_and(|c| self.cluster_encoding(c).0.attributes.skin)
    }

    /// Triangles of `cluster` as cluster-local vertex indices.
    pub fn cluster_triangles(&self, cluster: usize) -> impl Iterator<Item = [u32; 3]> + '_ {
        let c = &self.clusters[cluster];
        let stream = &self.page_payload(c.page as usize)[c.triangle_offset as usize..];
        (0..c.triangle_count()).map(move |t| decode_triangle(stream, c.vertex_count(), t))
    }

    /// Triangle corner positions of `cluster`.
    pub fn cluster_triangle_positions(&self, cluster: usize) -> Vec<[Vec3; 3]> {
        let (encoding, stream) = self.cluster_encoding(cluster);
        self.cluster_triangles(cluster).map(|t| t.map(|i| encoding.position(stream, i))).collect()
    }

    /// The voxels of a voxel cluster (none for triangle clusters).
    pub fn cluster_voxels(&self, cluster: usize) -> Vec<Voxel> {
        let c = &self.clusters[cluster];
        if !c.is_voxel() {
            return Vec::new();
        }
        decode_voxel_cluster(&self.page_payload(c.page as usize)[c.vertex_offset as usize..])
    }

    /// Whether the mesh has voxel LODs.
    pub fn has_voxels(&self) -> bool {
        self.stats.voxels_per_level.iter().any(|&v| v > 0)
    }

    /// Version 6: a seventh section count, the plant bones ([`VirtualMesh::bones`]) after the
    /// statistics, and voxel clusters may carry a bone per brick ([`crate::voxel`]). Version 5:
    /// clusters may carry plant data (extension bit 29, [`crate::encoding`]); versions 5 and 4
    /// (the same format without it) load as they are. Version 4 added voxel
    /// clusters ([`crate::voxel`]) and per-level voxel counts; version 3 files (triangles only, no
    /// voxel statistics) and version 2 files (float vertices) must be baked again.
    const MAGIC: [u8; 4] = *b"HVM6";
    /// Bytes at the start of a `.hvm` that determine its header length ([`VirtualMesh::header_len`])
    /// (a version 4 or 5 prefix is 28 bytes, then header data: reading 32 is harmless).
    pub const PREFIX_BYTES: usize = 32;
    const SDF_HEADER_WORDS: usize = 14;

    /// Header length for the section counts stored in the prefix
    /// (`clusters, pages, dependencies, levels, payload words, sdf words`, from version 6 `bones`).
    fn header_len_for(counts: &[usize], version: u8) -> usize {
        let (clusters, pages, dependencies, levels, sdf_words) = (counts[0], counts[1], counts[2], counts[3], counts[5]);
        let sdf = if sdf_words > 0 { (Self::SDF_HEADER_WORDS + sdf_words) * 4 } else { 0 };
        let (prefix, bones) = if version >= 6 {
            let bones = counts[6];
            (32, if bones > 0 { (crate::plant_bones::BOUND_WORDS + bones * crate::plant_bones::BONE_WORDS) * 4 } else { 0 })
        } else {
            (28, 0)
        };
        prefix
            + bones
            + 10 * 4
            + sdf
            + clusters * size_of::<GpuCluster>()
            + pages * size_of::<GpuPage>()
            + 2 * (pages + 1) * 4
            + dependencies * 4
            + 3 * levels * 4
    }

    /// Byte length of the serialised header (everything but the page payloads), i.e. the file
    /// offset at which the payload section starts.
    pub fn header_bytes(&self) -> usize {
        Self::header_len_for(
            &[
                self.clusters.len(),
                self.pages.len(),
                self.dependencies.len(),
                self.stats.clusters_per_level.len(),
                self.payload_bytes() / 4,
                self.sdf.as_ref().map_or(0, |s| s.words.len()),
                self.bones.as_ref().map_or(0, |b| b.bones.len()),
            ],
            6,
        )
    }

    /// The format version of a `.hvm` from its magic.
    fn check_magic(magic: &[u8]) -> Result<u8, String> {
        match magic {
            m if m == Self::MAGIC => Ok(6),
            b"HVM5" => Ok(5),
            b"HVM4" => Ok(4),
            b"HVM3" => Err("virtual mesh version 3 (no voxel LODs) is no longer supported: bake it again".into()),
            b"HVM2" => Err("virtual mesh version 2 (uncompressed vertices) is no longer supported: bake it again".into()),
            _ => Err("not a helios virtual mesh (HVM6)".into()),
        }
    }

    /// Header length of a `.hvm` from its first [`VirtualMesh::PREFIX_BYTES`] bytes, so a
    /// streaming reader knows how much to fetch before calling [`VirtualMesh::from_header_bytes`].
    pub fn header_len(prefix: &[u8]) -> Result<usize, String> {
        if prefix.len() < Self::PREFIX_BYTES {
            return Err("truncated virtual mesh header".into());
        }
        let version = Self::check_magic(&prefix[..4])?;
        let count = |i: usize| u32::from_le_bytes(prefix[4 + i * 4..8 + i * 4].try_into().unwrap()) as usize;
        let counts: Vec<usize> = (0..if version >= 6 { 7 } else { 6 }).map(count).collect();
        Ok(Self::header_len_for(&counts, version))
    }

    /// Byte range of `page`'s payload inside the serialised `.hvm`. Payloads are stored in page
    /// order, so adjacent pages are adjacent on disk.
    pub fn page_byte_range(&self, page: usize) -> std::ops::Range<u64> {
        let base = self.header_bytes() as u64;
        base + self.page_offsets[page] as u64 * 4..base + self.page_offsets[page + 1] as u64 * 4
    }

    /// First page of the trailing run of root pages. Roots must always be resident, so a
    /// streaming reader loads their payloads (one contiguous range at the end of the file)
    /// together with the header.
    pub fn root_tail_start(&self) -> usize {
        self.pages.iter().rposition(|p| p.root == 0).map_or(0, |p| p + 1)
    }

    /// File byte range of the payloads of pages `first_page..` (up to the end of the file).
    pub fn tail_byte_range(&self, first_page: usize) -> std::ops::Range<u64> {
        let base = self.header_bytes() as u64;
        base + self.page_offsets[first_page] as u64 * 4..base + self.payload_bytes() as u64
    }

    /// Supplies the payloads of pages `first_page..` (the bytes of [`VirtualMesh::tail_byte_range`]),
    /// e.g. the root tail of a mesh loaded with [`VirtualMesh::from_header_bytes`].
    pub fn set_tail_payload(&mut self, first_page: usize, bytes: &[u8]) -> Result<(), String> {
        let range = self.tail_byte_range(first_page);
        if bytes.len() as u64 != range.end - range.start {
            return Err(format!("tail payload has {} bytes, expected {}", bytes.len(), range.end - range.start));
        }
        self.page_data = bytemuck::pod_collect_to_vec(bytes);
        self.payload_first_page = first_page as u32;
        Ok(())
    }

    /// Serialises the mesh into a little-endian binary blob (`.hvm`). The page payloads come
    /// last, so a reader can load the header and stream page byte ranges on demand.
    pub fn to_bytes(&self) -> Vec<u8> {
        assert_eq!(self.payload_first_page, 0, "serialising a virtual mesh needs every page payload in memory");
        let mut out = Vec::new();
        out.extend_from_slice(&Self::MAGIC);
        let sdf_words = self.sdf.as_ref().map_or(0, |s| s.words.len());
        for len in [
            self.clusters.len(),
            self.pages.len(),
            self.dependencies.len(),
            self.stats.clusters_per_level.len(),
            self.page_data.len(),
            sdf_words,
            self.bones.as_ref().map_or(0, |b| b.bones.len()),
        ] {
            out.extend_from_slice(&(len as u32).to_le_bytes());
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.bounds.to_vec4().to_array()));
        out.extend_from_slice(bytemuck::cast_slice(&self.aabb.min.to_array()));
        out.extend_from_slice(bytemuck::cast_slice(&self.aabb.max.to_array()));
        if let Some(sdf) = &self.sdf {
            let header: [u32; 14] = [
                sdf.kind as u32,
                sdf.origin.x.to_bits(),
                sdf.origin.y.to_bits(),
                sdf.origin.z.to_bits(),
                sdf.voxel_size.to_bits(),
                sdf.resolution[0],
                sdf.resolution[1],
                sdf.resolution[2],
                sdf.band.to_bits(),
                sdf.height_range[0].to_bits(),
                sdf.height_range[1].to_bits(),
                sdf.two_sided as u32,
                0,
                0,
            ];
            out.extend_from_slice(bytemuck::cast_slice(&header));
            out.extend_from_slice(bytemuck::cast_slice(&sdf.words));
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.clusters));
        out.extend_from_slice(bytemuck::cast_slice(&self.pages));
        out.extend_from_slice(bytemuck::cast_slice(&self.page_offsets));
        out.extend_from_slice(bytemuck::cast_slice(&self.dependency_offsets));
        out.extend_from_slice(bytemuck::cast_slice(&self.dependencies));
        out.extend_from_slice(bytemuck::cast_slice(&self.stats.clusters_per_level));
        out.extend_from_slice(bytemuck::cast_slice(&self.stats.triangles_per_level));
        out.extend_from_slice(bytemuck::cast_slice(&self.stats.voxels_per_level));
        if let Some(bones) = self.bones.as_ref().filter(|b| !b.bones.is_empty()) {
            out.extend_from_slice(bytemuck::cast_slice(&bones.to_words()));
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.page_data));
        out
    }

    /// Inverse of [`VirtualMesh::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::parse(bytes, true)
    }

    /// Parses only the header of a `.hvm` (the first [`VirtualMesh::header_len`] bytes; more
    /// are ignored): every record, the dependencies and the distance field, but no page
    /// payloads. The result is ready for streaming: supply the root pages with
    /// [`VirtualMesh::set_tail_payload`] and fetch the others by [`VirtualMesh::page_byte_range`].
    pub fn from_header_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::parse(bytes, false)
    }

    fn parse(bytes: &[u8], with_payload: bool) -> Result<Self, String> {
        struct Reader<'a>(&'a [u8]);
        impl<'a> Reader<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
                if self.0.len() < n {
                    return Err("unexpected end of virtual mesh data".into());
                }
                let (head, tail) = self.0.split_at(n);
                self.0 = tail;
                Ok(head)
            }
            fn u32(&mut self) -> Result<usize, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
            }
            fn vec<T: Pod>(&mut self, count: usize) -> Result<Vec<T>, String> {
                Ok(bytemuck::pod_collect_to_vec(self.take(count * std::mem::size_of::<T>())?))
            }
        }

        let mut r = Reader(bytes);
        let version = Self::check_magic(r.take(4)?)?;
        let counts: Vec<usize> = (0..if version >= 6 { 7 } else { 6 }).map(|_| r.u32()).collect::<Result<_, _>>()?;
        let header: Vec<f32> = r.vec(10)?;
        let sdf = if counts[5] > 0 {
            let h: Vec<u32> = r.vec(14)?;
            Some(MeshSdf {
                kind: if h[0] == 1 { crate::sdf::SdfKind::Heightfield } else { crate::sdf::SdfKind::Volume },
                origin: Vec3::new(f32::from_bits(h[1]), f32::from_bits(h[2]), f32::from_bits(h[3])),
                voxel_size: f32::from_bits(h[4]),
                resolution: [h[5], h[6], h[7]],
                band: f32::from_bits(h[8]),
                height_range: [f32::from_bits(h[9]), f32::from_bits(h[10])],
                two_sided: h[11] != 0,
                words: r.vec(counts[5])?,
            })
        } else {
            None
        };
        let mut mesh = Self {
            bounds: Sphere::new(Vec3::new(header[0], header[1], header[2]), header[3]),
            aabb: Aabb { min: Vec3::new(header[4], header[5], header[6]), max: Vec3::new(header[7], header[8], header[9]) },
            sdf,
            clusters: r.vec(counts[0])?,
            pages: r.vec(counts[1])?,
            page_offsets: r.vec(counts[1] + 1)?,
            dependency_offsets: r.vec(counts[1] + 1)?,
            dependencies: r.vec(counts[2])?,
            stats: VirtualMeshStats {
                clusters_per_level: r.vec(counts[3])?,
                triangles_per_level: r.vec(counts[3])?,
                voxels_per_level: r.vec(counts[3])?,
            },
            bones: None,
            page_data: Vec::new(),
            payload_first_page: counts[1] as u32,
        };
        let bone_count = counts.get(6).copied().unwrap_or(0);
        if bone_count > 0 {
            let words: Vec<u32> = r.vec(crate::plant_bones::BOUND_WORDS + bone_count * crate::plant_bones::BONE_WORDS)?;
            mesh.bones = Some(PlantBones::from_words(&words, bone_count)?);
        }
        if with_payload {
            mesh.page_data = r.vec(counts[4])?;
            mesh.payload_first_page = 0;
        }
        mesh.validate(counts[4])?;
        Ok(mesh)
    }

    /// Structural checks on deserialised data, so a corrupt or truncated download fails here
    /// instead of indexing out of bounds in the streamer.
    fn validate(&self, payload_words: usize) -> Result<(), String> {
        let monotonic = |v: &[u32], last: usize| v.first() == Some(&0) && v.windows(2).all(|w| w[0] <= w[1]) && v.last() == Some(&(last as u32));
        let pages = self.pages.len() as u32;
        let ok = monotonic(&self.page_offsets, payload_words)
            && monotonic(&self.dependency_offsets, self.dependencies.len())
            && self.dependencies.iter().all(|&d| d < pages)
            && self.clusters.iter().all(|c| c.page < pages && (c.child_page == INVALID_PAGE || c.child_page < pages));
        if ok {
            Ok(())
        } else {
            Err("corrupt virtual mesh: inconsistent page tables".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::edge_key;
    use crate::primitives;

    fn assert_watertight(mesh: &VirtualMesh, selected: &[u32]) {
        assert_watertight_with(selected, &|c| mesh.cluster_triangle_positions(c));
    }

    /// `triangles(cluster)`: the triangle corner positions of a cluster as some consumer sees
    /// them (decoded, or decoded and skinned).
    fn assert_watertight_with(selected: &[u32], triangles: &dyn Fn(usize) -> Vec<[Vec3; 3]>) {
        // Vertex identity = position bits (wedges at seams share positions).
        let mut ids: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        let mut edges: FxHashMap<u64, u32> = FxHashMap::default();
        for &c in selected {
            for t in triangles(c as usize) {
                let id = t.map(|p| {
                    let next = ids.len() as u32;
                    *ids.entry(p.to_array().map(f32::to_bits)).or_insert(next)
                });
                for (a, b) in [(id[0], id[1]), (id[1], id[2]), (id[2], id[0])] {
                    *edges.entry(edge_key(a, b)).or_insert(0) += 1;
                }
            }
        }
        // An odd use count means an open border, i.e. a visible crack. (Even counts > 2 are
        // harmless pinches where two locked group borders touch.)
        let open = edges.values().filter(|&&n| n % 2 != 0).count();
        assert_eq!(open, 0, "{open} open edges (cracks) in LOD cut of {} clusters", selected.len());
    }

    fn build_rock() -> VirtualMesh {
        let mesh = primitives::rock(7, 5);
        build_virtual_mesh(&mesh, &BuildConfig { sdf: SdfMode::None, ..Default::default() })
    }

    fn view(distance: f32) -> LodView {
        LodView { position: Vec3::new(0.3, 0.2, distance), lod_scale: 1000.0, threshold: 1.0, orthographic: false }
    }

    #[test]
    fn dag_reduces_to_few_roots() {
        let vm = build_rock();
        let levels = vm.stats.level_count();
        assert!(levels >= 4, "expected a deep DAG, got {levels} levels: {:?}", vm.stats);
        let roots = vm.clusters.iter().filter(|c| c.parent_error >= ROOT_ERROR).count();
        let level0 = vm.stats.clusters_per_level[0] as usize;
        assert!(roots * 8 < level0, "too many roots: {roots} of {level0} level-0 clusters");
    }

    #[test]
    fn errors_and_bounds_are_monotonic() {
        let vm = build_rock();
        for c in &vm.clusters {
            assert!(c.parent_error >= c.lod_error);
            let own = Sphere::new(Vec3::from_slice(&c.lod_bounds[..3]), c.lod_bounds[3]);
            let parent = Sphere::new(Vec3::from_slice(&c.parent_lod_bounds[..3]), c.parent_lod_bounds[3]);
            if c.parent_error < ROOT_ERROR {
                assert!(parent.contains_sphere(&own, 1e-4), "parent LOD sphere must contain child");
            }
        }
    }

    #[test]
    fn every_lod_cut_is_watertight() {
        let vm = build_rock();
        let full = vm.stats.triangles_per_level[0] as usize;
        let mut previous = usize::MAX;
        for distance in [1.5f32, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 400.0] {
            let selected = lod_cut(&vm, &view(distance));
            assert!(!selected.is_empty());
            assert_watertight(&vm, &selected);
            let tris: usize = selected.iter().map(|&c| vm.clusters[c as usize].triangle_count() as usize).sum();
            assert!(tris <= full);
            assert!(tris <= previous, "triangle count must not grow with distance");
            previous = tris;
        }
        assert!(previous < full / 20, "far LOD should be much coarser ({previous} of {full})");
    }

    #[test]
    fn partial_residency_is_watertight() {
        let vm = build_rock();
        // Make a random subset resident, closed under dependencies (parents first).
        let mut resident = vec![false; vm.pages.len()];
        let mut rng = 12345u32;
        for _ in 0..4 {
            for p in 0..vm.pages.len() {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let deps_ok = vm.page_dependencies(p).iter().all(|&d| resident[d as usize]);
                if vm.pages[p].root != 0 || (deps_ok && !rng.is_multiple_of(3)) {
                    resident[p] = true;
                }
            }
        }
        let partial = resident.iter().filter(|&&r| r).count();
        assert!(partial < vm.pages.len(), "test needs some non-resident pages");
        for distance in [1.5f32, 6.0, 25.0, 100.0] {
            let selected = lod_cut_resident(&vm, &view(distance), &|p| resident[p as usize]);
            assert!(!selected.is_empty());
            assert_watertight(&vm, &selected);
        }
    }

    #[test]
    fn pages_hold_whole_groups_and_roots_have_no_dependencies() {
        let vm = build_rock();
        for (p, page) in vm.pages.iter().enumerate() {
            assert!(page.cluster_count as usize <= BuildConfig::default().page_clusters);
            if page.root != 0 {
                assert!(vm.page_dependencies(p).is_empty());
            } else {
                assert!(!vm.page_dependencies(p).is_empty(), "page {p} has no parents");
            }
            let clusters = &vm.clusters[page.cluster_offset as usize..(page.cluster_offset + page.cluster_count) as usize];
            assert!(clusters.iter().all(|c| c.page == p as u32));
        }
    }

    #[test]
    fn aggregate_geometry_simplifies() {
        let mesh = primitives::leaf_cloud(3000, 9);
        let vm = build_virtual_mesh(&mesh, &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let roots: u32 = vm.clusters.iter().filter(|c| c.parent_error >= ROOT_ERROR).map(|c| c.triangle_count()).sum();
        assert!((roots as usize) < mesh.triangle_count() / 8, "leaf cloud roots keep {roots} of {} triangles", mesh.triangle_count());
    }

    #[test]
    fn serialization_round_trips() {
        let vm = build_virtual_mesh(&primitives::icosphere(3), &BuildConfig::default());
        assert!(vm.sdf.is_some());
        let back = VirtualMesh::from_bytes(&vm.to_bytes()).unwrap();
        assert_eq!(back.clusters, vm.clusters);
        assert_eq!(back.pages, vm.pages);
        assert_eq!(back.page_data, vm.page_data);
        assert_eq!(back.dependencies, vm.dependencies);
        assert_eq!(back.sdf, vm.sdf);
    }

    /// A plant's bones survive `.hvm` version 6 (with the header length a streaming reader
    /// computes), its voxel bricks carry bones, and version 5 files (six section counts, no
    /// bones) still load.
    #[test]
    fn plant_bones_round_trip_and_version_5_loads() {
        let grass = crate::vegetation::grass_clump(3, 40, 0.45);
        let vm = build_virtual_mesh(&grass, &BuildConfig { voxels: Some(crate::VoxelConfig::default()), sdf: SdfMode::None, ..Default::default() });
        let bones = vm.bones.clone().expect("a plant keeps its bones");
        assert_eq!(bones.bones.len(), 40);
        assert!(bones.bounds.bend[0] + bones.bounds.bend[1] + bones.bounds.bend[2] > 0.0);
        let bytes = vm.to_bytes();
        assert_eq!(&bytes[..4], b"HVM6");
        assert_eq!(VirtualMesh::header_len(&bytes[..VirtualMesh::PREFIX_BYTES]).unwrap(), vm.header_bytes());
        let back = VirtualMesh::from_bytes(&bytes).unwrap();
        assert_eq!(back.bones, vm.bones);
        assert_eq!(back.page_data, vm.page_data);
        // Every vertex's bone is a blade; voxel bricks follow blades too.
        let voxel_clusters: Vec<usize> = (0..vm.clusters.len()).filter(|&c| vm.clusters[c].is_voxel()).collect();
        assert!(!voxel_clusters.is_empty());
        for c in voxel_clusters {
            let words = &vm.page_payload(vm.clusters[c].page as usize)[vm.clusters[c].vertex_offset as usize..];
            let (bricks, _) = crate::voxel::voxel_counts(words);
            for b in 0..bricks {
                assert!((crate::voxel::voxel_brick_bone(words, b).expect("plant bricks carry bones") as usize) < 40);
            }
        }
        for c in (0..vm.clusters.len()).filter(|&c| !vm.clusters[c].is_voxel()) {
            for i in 0..vm.clusters[c].vertex_count() {
                assert!((vm.cluster_vertex_skin(c, i).unwrap().dominant_joint() as usize) < 40);
            }
        }

        // Version 5: the same layout without the seventh count and the bone section.
        let plain = build_virtual_mesh(&primitives::icosphere(2), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let v6 = plain.to_bytes();
        let mut v5 = b"HVM5".to_vec();
        v5.extend_from_slice(&v6[4..28]);
        v5.extend_from_slice(&v6[32..]);
        assert_eq!(VirtualMesh::header_len(&v5[..VirtualMesh::PREFIX_BYTES]).unwrap(), plain.header_bytes() - 4);
        let old = VirtualMesh::from_bytes(&v5).unwrap();
        assert_eq!(old.clusters, plain.clusters);
        assert_eq!(old.page_data, plain.page_data);
        assert!(old.bones.is_none());
    }

    #[test]
    fn header_and_page_ranges_stream_the_same_mesh() {
        let vm = build_rock();
        let bytes = vm.to_bytes();
        let header_len = VirtualMesh::header_len(&bytes[..VirtualMesh::PREFIX_BYTES]).unwrap();
        assert_eq!(header_len, vm.header_bytes());
        assert_eq!(bytes.len(), header_len + vm.payload_bytes());

        let mut streamed = VirtualMesh::from_header_bytes(&bytes[..header_len]).unwrap();
        assert_eq!(streamed.clusters, vm.clusters);
        assert_eq!(streamed.payload_bytes(), vm.payload_bytes());
        assert!(streamed.page_data.is_empty() && !streamed.has_payload(0));
        for p in 0..vm.pages.len() {
            let range = streamed.page_byte_range(p);
            let words: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes[range.start as usize..range.end as usize]);
            assert_eq!(words, vm.page_payload(p));
        }

        // Roots sit in one contiguous run at the end of the file.
        let tail = streamed.root_tail_start();
        assert!(tail > 0 && tail < vm.pages.len());
        assert!(vm.pages.iter().enumerate().all(|(p, page)| (page.root != 0) == (p >= tail)));
        let start = streamed.page_byte_range(tail).start as usize;
        streamed.set_tail_payload(tail, &bytes[start..]).unwrap();
        assert!(!streamed.has_payload(tail - 1));
        for p in tail..vm.pages.len() {
            assert_eq!(streamed.page_payload(p), vm.page_payload(p));
        }
        assert!(streamed.cpu_bytes() < vm.cpu_bytes());
        assert!(VirtualMesh::from_header_bytes(&bytes[..header_len - 1]).is_err());
    }

    /// Watertight LOD cuts (full and partial residency) on a deliberately coarse position grid,
    /// where a vertex shared by two clusters or two LOD levels would visibly split if it were
    /// rounded differently on each side.
    #[test]
    fn coarse_quantization_stays_watertight_across_cluster_and_lod_borders() {
        let mesh = primitives::rock(7, 5);
        let vm = build_virtual_mesh(&mesh, &BuildConfig { sdf: SdfMode::None, position_precision: Some(7), ..Default::default() });
        let step = 2f32.powi(-7);
        let mut shared_borders = 0;
        for distance in [1.5f32, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0, 100.0, 400.0] {
            let selected = lod_cut(&vm, &view(distance));
            assert_watertight(&vm, &selected);
            // The cut mixes LOD levels whose clusters have different grid minima, so shared
            // vertices really are encoded relative to different origins.
            let levels: FxHashSet<u32> = selected.iter().map(|&c| vm.clusters[c as usize].lod_level()).collect();
            shared_borders += (levels.len() > 1) as u32;
        }
        assert!(shared_borders >= 6, "test needs cuts that mix LOD levels ({shared_borders})");
        // Every decoded position lies on the mesh grid.
        for c in 0..vm.clusters.len() {
            for t in vm.cluster_triangle_positions(c) {
                for p in t {
                    assert_eq!(p / step, (p / step).round(), "position {p} is off the grid");
                }
            }
        }
    }

    /// Full-detail vertices decode to the source vertex rounded to the grid (≤ half a step
    /// away), with normals within a fraction of a degree and UVs well under a texel.
    #[test]
    fn full_detail_vertices_match_the_source() {
        let mesh = primitives::rock(7, 5);
        let vm = build_virtual_mesh(&mesh, &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let (encoding, _) = vm.cluster_encoding(0);
        let q = Quantization {
            position_exponent: encoding.position_exponent,
            uv_exponent: encoding.uv_exponent,
            normal_bits: encoding.normal_bits,
            attributes: VertexAttributes::default(),
        };
        let mut source: FxHashMap<[i32; 3], Vec<&crate::Vertex>> = FxHashMap::default();
        for v in &mesh.vertices {
            source.entry(q.quantize_position(v.position()).to_array()).or_default().push(v);
        }
        let mut checked = 0;
        for c in (0..vm.clusters.len()).filter(|&c| vm.clusters[c].lod_level() == 0) {
            for i in 0..vm.clusters[c].vertex_count() {
                let (p, n, uv) = vm.cluster_vertex(c, i);
                let grid = q.quantize_position(p);
                assert_eq!(grid.as_vec3() * q.position_step(), p);
                let candidates = &source[&grid.to_array()];
                assert!(candidates.iter().any(|v| (v.position() - p).abs().max_element() <= q.position_step() * 0.5));
                assert!(candidates.iter().any(|v| v.normal().normalize().dot(n) > 0.99999));
                assert!(candidates.iter().any(|v| (v.uv() - uv).abs().max_element() * 2048.0 <= 1.0 / 32.0));
                checked += 1;
            }
        }
        assert!(checked > 1000);
        assert!(q.position_step() < vm.aabb.extent().max_element() / 1000.0);
    }

    /// A second UV set and vertex colours follow their vertices through every LOD level, and
    /// cost only their own bits: the same mesh without them bakes to the plain payload.
    #[test]
    fn optional_attributes_survive_simplification() {
        let plain = primitives::rock(3, 4);
        let uv1_of = |p: Vec3| Vec2::new(p.x * 0.5 + 2.0, p.z * 0.25 - 1.0);
        let color_of = |p: Vec3| Vec4::new(p.x.abs().min(1.0), 0.25, (p.y * 0.5 + 0.5).clamp(0.0, 1.0), 1.0);
        let positions = plain.positions();
        let with = Mesh::with_attributes(
            plain.vertices.clone(),
            plain.indices.clone(),
            positions.iter().map(|&p| uv1_of(p)).collect(),
            positions.iter().map(|&p| color_of(p)).collect(),
        );
        let config = BuildConfig { sdf: SdfMode::None, ..Default::default() };
        let (vm_plain, vm) = (build_virtual_mesh(&plain, &config), build_virtual_mesh(&with, &config));
        assert_eq!(vm.clusters.len(), vm_plain.clusters.len(), "attributes do not change the DAG");
        let (encoding, _) = vm.cluster_encoding(0);
        let uv_step = f32::from_bits(((encoding.uv_exponent + 127) as u32) << 23);
        let mut levels = FxHashSet::default();
        for c in 0..vm.clusters.len() {
            levels.insert(vm.clusters[c].lod_level());
            for i in 0..vm.clusters[c].vertex_count() {
                // Simplification keeps source vertices, so attributes stay those of the position.
                let (p, _, _) = vm.cluster_vertex(c, i);
                let (uv1, color) = vm.cluster_vertex_extra(c, i);
                assert!((uv1 - uv1_of(p)).abs().max_element() <= uv_step + 1e-3, "{uv1} vs {}", uv1_of(p));
                assert!((color - color_of(p)).abs().max_element() < 0.01, "{color} vs {}", color_of(p));
            }
        }
        assert!(levels.len() >= 3);
        // Plain clusters carry no extension; the extended ones a 4-word header extension plus the
        // per-vertex bits (uv1 and three varying colour channels; alpha is constant: 0 bits).
        assert!((0..vm_plain.clusters.len()).all(|c| vm_plain.cluster_encoding(c).0.header_words() == 7));
        let vertices: usize = vm.clusters.iter().map(|c| c.vertex_count() as usize).sum();
        let extra = vm.payload_bytes() - vm_plain.payload_bytes();
        assert!(extra > vm.clusters.len() * 16 && extra < vm.clusters.len() * 20 + vertices * 8, "{extra} bytes for {vertices} vertices");
        println!("optional attributes: +{extra} bytes ({:.1} per vertex) on {} bytes", extra as f32 / vertices as f32, vm_plain.payload_bytes());
    }

    /// Plant data (wind weight, occlusion) follows its vertices through every LOD level and the
    /// voxel LODs of foliage, at 10 bits per vertex.
    #[test]
    fn plant_data_survives_simplification() {
        let mut plain = primitives::rock(5, 4);
        let plant_of = |p: Vec3| Vec2::new((p.y * 0.5 + 0.5).clamp(0.0, 1.0), (p.x * 0.5 + 0.5).clamp(0.0, 1.0));
        let config = BuildConfig { sdf: SdfMode::None, ..Default::default() };
        let vm_plain = build_virtual_mesh(&plain, &config);
        plain.plant = plain.positions().iter().map(|&p| plant_of(p)).collect();
        let vm = build_virtual_mesh(&plain, &config);
        assert_eq!(vm.clusters.len(), vm_plain.clusters.len(), "plant data does not change the DAG");
        for c in 0..vm.clusters.len() {
            for i in 0..vm.clusters[c].vertex_count() {
                let (p, _, _) = vm.cluster_vertex(c, i);
                let plant = vm.cluster_vertex_plant(c, i);
                assert!((plant - plant_of(p)).abs().max_element() <= 0.5 / 31.0 + 1e-3, "{plant} vs {}", plant_of(p));
            }
        }
        let vertices: usize = vm.clusters.iter().map(|c| c.vertex_count() as usize).sum();
        let extra = vm.payload_bytes() - vm_plain.payload_bytes();
        assert!(extra <= vm.clusters.len() * 20 + vertices * 10 / 8 + vm.clusters.len() * 4, "{extra} bytes for {vertices} vertices");
        // Foliage with voxel LODs keeps its plant data on the triangle clusters.
        let grass = crate::vegetation::grass_clump(3, 40, 0.45);
        let vm = build_virtual_mesh(&grass, &BuildConfig { voxels: Some(VoxelConfig::default()), ..config });
        let held = (0..vm.clusters.len()).filter(|&c| !vm.clusters[c].is_voxel()).any(|c| (0..vm.clusters[c].vertex_count()).any(|i| vm.cluster_vertex_plant(c, i).x == 0.0));
        assert!(held, "the blades' roots keep weight 0");
    }

    /// A limb: a capped cylinder (y from 0 to 2) following three joints — blended smoothly
    /// around y = 0.7, with a hard (rigid) seam at y = 1.4 — where a few vertices carry all four
    /// influences. Skins are functions of the position, so seam wedges agree.
    fn skinned_limb() -> Mesh {
        let mesh = primitives::cylinder(40, 48, 0.2, 2.0, true);
        let skin = mesh
            .vertices
            .iter()
            .map(|v| {
                let y = v.position[1];
                let t = ((y - 0.6) / 0.2).clamp(0.0, 1.0);
                if y >= 1.4 {
                    SkinWeights::rigid(2)
                } else if (v.position[0].to_bits() ^ v.position[2].to_bits()) % 7 == 0 {
                    SkinWeights::new([0, 1, 2, 3], [1.0 - t, t, 0.1, 0.05])
                } else {
                    SkinWeights::new([0, 1, 0, 0], [1.0 - t, t, 0.0, 0.0])
                }
            })
            .collect();
        mesh.with_skin(skin)
    }

    /// The limb bent at both joints, and a small extra joint.
    fn limb_pose(bend: f32) -> [glam::Mat4; 4] {
        use glam::Mat4;
        let about = |pivot: Vec3, angle: f32| Mat4::from_translation(pivot) * Mat4::from_rotation_x(angle) * Mat4::from_translation(-pivot);
        let elbow = about(Vec3::new(0.0, 0.7, 0.0), bend);
        [Mat4::IDENTITY, elbow, elbow * about(Vec3::new(0.0, 1.4, 0.0), -0.8 * bend), Mat4::from_translation(Vec3::new(0.03, 0.0, -0.02))]
    }

    fn posed_triangles(vm: &VirtualMesh, cluster: usize, palette: &[glam::Mat4]) -> Vec<[Vec3; 3]> {
        let corner = |i: u32| crate::skin::skin_point(&vm.cluster_vertex_skin(cluster, i).unwrap(), palette, vm.cluster_vertex(cluster, i).0);
        vm.cluster_triangles(cluster).map(|t| t.map(corner)).collect()
    }

    /// Skinning is a function of the decoded rest position and the decoded weights, both
    /// identical wherever a vertex is shared, so posed cuts are as watertight as rest ones —
    /// with full and partial residency.
    #[test]
    fn skinned_lod_cuts_stay_watertight_when_posed() {
        let vm = build_virtual_mesh(&skinned_limb(), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        assert!(vm.is_skinned() && vm.stats.level_count() >= 3, "{:?}", vm.stats);
        let resident: Vec<bool> = (0..vm.pages.len()).map(|p| vm.pages[p].root != 0 || p % 3 != 1).collect();
        let resident = |p: u32| resident[p as usize] && vm.page_dependencies(p as usize).iter().all(|&d| d as usize % 3 != 1 || vm.pages[d as usize].root != 0);
        for bend in [0.0f32, 0.6, 1.4] {
            let palette = limb_pose(bend);
            for distance in [0.8f32, 1.5, 3.0, 6.0, 12.0, 25.0, 100.0] {
                let selected = lod_cut(&vm, &view(distance));
                assert_watertight_with(&selected, &|c| posed_triangles(&vm, c, &palette));
                let partial = lod_cut_resident(&vm, &view(distance), &resident);
                assert_watertight_with(&partial, &|c| posed_triangles(&vm, c, &palette));
            }
        }
    }

    /// Simplification keeps source vertices, so every LOD level carries each vertex's own
    /// weights.
    #[test]
    fn skin_weights_follow_their_vertices_through_every_level() {
        let mesh = skinned_limb();
        let vm = build_virtual_mesh(&mesh, &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let source: FxHashMap<[u32; 3], SkinWeights> = mesh.vertices.iter().zip(&mesh.skin).map(|(v, &s)| (v.position.map(f32::to_bits), s)).collect();
        let (encoding, _) = vm.cluster_encoding(0);
        let step = f32::from_bits(((encoding.position_exponent + 127) as u32) << 23);
        let mut four = 0;
        for c in 0..vm.clusters.len() {
            for i in 0..vm.clusters[c].vertex_count() {
                let skin = vm.cluster_vertex_skin(c, i).unwrap();
                let p = vm.cluster_vertex(c, i).0;
                // Decoded positions are on the grid: find the source vertex within half a step.
                let matches = mesh.vertices.iter().zip(&mesh.skin).any(|(v, s)| (v.position() - p).abs().max_element() <= step * 0.5 + 1e-6 && *s == skin);
                assert!(matches, "cluster {c} vertex {i} at {p}: {skin:?}");
                four += (skin.influences() == 4) as usize;
            }
        }
        assert!(four > 0 && !source.is_empty());
        // The same mesh without skin bakes to the plain format.
        let plain = build_virtual_mesh(&Mesh { skin: Vec::new(), ..mesh }, &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        assert!(!plain.is_skinned());
        assert!((0..plain.clusters.len()).all(|c| plain.cluster_encoding(c).0.header_words() == 7));
        println!("skin: +{} bytes on {} ({} clusters)", vm.payload_bytes() as i64 - plain.payload_bytes() as i64, plain.payload_bytes(), vm.clusters.len());
    }

    fn point_triangle_distance(p: Vec3, [a, b, c]: [Vec3; 3]) -> f32 {
        // Ericson, Real-Time Collision Detection 5.1.5.
        let (ab, ac, ap) = (b - a, c - a, p - a);
        let (d1, d2) = (ab.dot(ap), ac.dot(ap));
        if d1 <= 0.0 && d2 <= 0.0 {
            return p.distance(a);
        }
        let bp = p - b;
        let (d3, d4) = (ab.dot(bp), ac.dot(bp));
        if d3 >= 0.0 && d4 <= d3 {
            return p.distance(b);
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return p.distance(a + ab * (d1 / (d1 - d3)));
        }
        let cp = p - c;
        let (d5, d6) = (ab.dot(cp), ac.dot(cp));
        if d6 >= 0.0 && d5 <= d6 {
            return p.distance(c);
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return p.distance(a + ac * (d2 / (d2 - d6)));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            return p.distance(b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
        }
        let denom = 1.0 / (va + vb + vc);
        p.distance(a + ab * (vb * denom) + ac * (vc * denom))
    }

    /// Voxels carry no skin weights: a skinned mesh baked with voxel LODs keeps triangles only.
    #[test]
    fn skinned_meshes_never_get_voxels() {
        let config = BuildConfig { sdf: SdfMode::None, voxels: Some(crate::voxel::VoxelConfig::default()), ..Default::default() };
        let vm = build_virtual_mesh(&skinned_limb(), &config);
        assert!(vm.is_skinned() && !vm.has_voxels());
        assert!(vm.clusters.iter().all(|c| c.voxel == 0));
    }

    /// The weight-deviation cost keeps coarse levels from collapsing across the weight seams: a
    /// vertex that follows one joint merging into one that follows another bends the coarse
    /// limb where the source doesn't. Measured as the mean distance of the posed source
    /// vertices from the posed LOD cut, relative to the same distance at rest (which is all the
    /// LOD error accounts for).
    #[test]
    fn weight_seams_collapse_last() {
        let mesh = skinned_limb();
        let palette = limb_pose(1.2);
        let posed_source: Vec<Vec3> = mesh.vertices.iter().zip(&mesh.skin).map(|(v, s)| crate::skin::skin_point(s, &palette, v.position())).collect();
        let excess = |skin_weight: f32| {
            let config = BuildConfig { sdf: SdfMode::None, simplify: SimplifyOptions { skin_weight, ..Default::default() }, ..Default::default() };
            let vm = build_virtual_mesh(&mesh, &config);
            let mut total = 0.0;
            for distance in [3.0f32, 6.0, 12.0] {
                let selected = lod_cut(&vm, &view(distance));
                let mean = |points: &[Vec3], triangles: &[[Vec3; 3]]| {
                    points.iter().map(|&p| triangles.iter().map(|&t| point_triangle_distance(p, t)).fold(f32::MAX, f32::min)).sum::<f32>() / points.len() as f32
                };
                let rest: Vec<[Vec3; 3]> = selected.iter().flat_map(|&c| vm.cluster_triangle_positions(c as usize)).collect();
                let posed: Vec<[Vec3; 3]> = selected.iter().flat_map(|&c| posed_triangles(&vm, c as usize, &palette)).collect();
                total += mean(&posed_source, &posed) - mean(&mesh.positions(), &rest);
            }
            total / 3.0
        };
        let (aware, blind) = (excess(SimplifyOptions::default().skin_weight), excess(0.0));
        println!("posed LOD deviation beyond the rest one: {aware:.5} (weights considered) vs {blind:.5} (ignored)");
        assert!(aware < blind * 0.75, "{aware} vs {blind}");
    }

    /// The compressed payload is at least twice smaller than 24-byte float vertices plus one
    /// word per triangle.
    #[test]
    fn payload_is_compressed() {
        let vm = build_rock();
        let uncompressed: usize = vm.clusters.iter().map(|c| c.vertex_count() as usize * 24 + c.triangle_count() as usize * 4).sum();
        let ratio = uncompressed as f32 / vm.payload_bytes() as f32;
        assert!(ratio >= 2.0, "compression ratio {ratio:.2}");
    }

    #[test]
    fn version_2_and_3_files_are_rejected_with_a_clear_message() {
        for magic in [b"HVM2", b"HVM3"] {
            let mut bytes = build_virtual_mesh(&primitives::icosphere(1), &BuildConfig { sdf: SdfMode::None, ..Default::default() }).to_bytes();
            bytes[..4].copy_from_slice(magic);
            assert!(VirtualMesh::from_bytes(&bytes).unwrap_err().contains("bake it again"));
            assert!(VirtualMesh::header_len(&bytes).unwrap_err().contains("bake it again"));
        }
    }

    /// A leaf cloud with a voxel budget generous enough for voxels to win from the first levels.
    fn build_foliage() -> (Mesh, VirtualMesh) {
        let mesh = primitives::leaf_cloud(3000, 9);
        let voxels = Some(VoxelConfig { voxels_per_triangle: 16.0 });
        let vm = build_virtual_mesh(&mesh, &BuildConfig { sdf: SdfMode::None, voxels, ..Default::default() });
        (mesh, vm)
    }

    /// The LOD key of a group's outputs: what both representations of a group share (roots
    /// have no parent sphere).
    fn group_key(c: &GpuCluster) -> [u32; 10] {
        let mut key = [0; 10];
        key[..4].copy_from_slice(&c.lod_bounds.map(f32::to_bits));
        if c.parent_error < ROOT_ERROR {
            key[4..8].copy_from_slice(&c.parent_lod_bounds.map(f32::to_bits));
        }
        key[8] = c.lod_error.to_bits();
        key[9] = c.parent_error.to_bits();
        key
    }

    /// Voxel alternates carry their triangles' LOD values: both cuts select the outputs of the
    /// same groups, errors stay monotonic, and coarse levels really switch to voxels.
    #[test]
    fn voxel_cuts_mirror_the_triangle_cut() {
        let (_, vm) = build_foliage();
        assert!(vm.has_voxels(), "{:?}", vm.stats);
        for c in &vm.clusters {
            assert!(c.parent_error >= c.lod_error);
            assert!(!(c.is_voxel() && c.has_voxel_alternate()));
            if c.is_voxel() {
                assert!(c.triangle_count() == 0 && c.voxel_count() > 0 && c.voxel_bricks() as usize <= crate::voxel::MAX_CLUSTER_BRICKS);
                assert!(vm.cluster_voxels(vm.clusters.iter().position(|x| x == c).unwrap()).len() == c.voxel_count() as usize);
            }
        }
        let mut voxel_cuts = 0;
        for distance in [2.0f32, 8.0, 30.0, 60.0, 120.0, 250.0, 500.0, 2000.0] {
            let view = view(distance);
            let triangles = lod_cut(&vm, &view);
            let voxels = lod_cut_in(&vm, &view, &|_| true, true);
            let keys = |cut: &[u32]| cut.iter().map(|&c| group_key(&vm.clusters[c as usize])).collect::<FxHashSet<_>>();
            // (A triangle cluster whose share of its group's source triangles came out empty
            // has no voxels: the group's other alternates cover them.)
            assert!(keys(&voxels).is_subset(&keys(&triangles)), "both cuts take the same groups at {distance}");
            assert!(keys(&triangles).len() <= keys(&voxels).len() + keys(&triangles).len() / 8);
            assert!(triangles.iter().all(|&c| !vm.clusters[c as usize].is_voxel()));
            voxel_cuts += voxels.iter().any(|&c| vm.clusters[c as usize].is_voxel()) as u32;
        }
        assert!(voxel_cuts >= 3, "distant views should draw voxels ({voxel_cuts} of 8)");
    }

    /// Pages of voxel groups serve one mode: the triangle cut never needs a page holding only
    /// voxel clusters, and dependencies hold in each cut.
    #[test]
    fn voxel_pages_stream_per_mode() {
        let (_, vm) = build_foliage();
        let voxel_only = |p: usize| {
            let page = &vm.pages[p];
            vm.clusters[page.cluster_offset as usize..(page.cluster_offset + page.cluster_count) as usize].iter().all(GpuCluster::is_voxel)
        };
        assert!((0..vm.pages.len()).any(voxel_only));
        for c in vm.clusters.iter().filter(|c| c.in_mode(false) && c.child_page != INVALID_PAGE) {
            assert!(!voxel_only(c.child_page as usize));
        }
        // Children pages depend on their parents' pages in both cuts.
        for c in &vm.clusters {
            if c.child_page != INVALID_PAGE {
                assert!(vm.page_dependencies(c.child_page as usize).contains(&c.page));
            }
        }
    }

    /// Coverage of a mesh projected along -z onto square pixels of `pixel` world units over
    /// [-1.2, 1.2]²: 1 - transmittance per pixel.
    struct Coverage {
        n: usize,
        pixel: f32,
        transmittance: Vec<f32>,
    }

    impl Coverage {
        const ORIGIN: f32 = -1.2;
        const SUB: usize = 4;

        fn new(pixel: f32) -> Self {
            let n = (2.4 / pixel).ceil() as usize;
            Self { n, pixel, transmittance: vec![1.0; n * n] }
        }

        /// Opaque triangles, 4×4 samples per pixel.
        fn triangles(&mut self, triangles: &[[Vec3; 3]]) {
            let samples = self.n * Self::SUB;
            let step = self.pixel / Self::SUB as f32;
            let mut covered = vec![false; samples * samples];
            for t in triangles {
                let p = t.map(|v| (Vec2::new(v.x, v.y) - Self::ORIGIN) / step - 0.5);
                let lo = p[0].min(p[1]).min(p[2]).ceil().max(Vec2::ZERO);
                let hi = p[0].max(p[1]).max(p[2]).floor().min(Vec2::splat(samples as f32 - 1.0));
                let edge = |a: Vec2, b: Vec2, q: Vec2| (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
                let area = edge(p[0], p[1], p[2]);
                if area == 0.0 {
                    continue;
                }
                for y in lo.y as usize..=hi.y.max(lo.y) as usize {
                    for x in lo.x as usize..=hi.x.max(lo.x) as usize {
                        let q = Vec2::new(x as f32, y as f32);
                        let w = [edge(p[1], p[2], q), edge(p[2], p[0], q), edge(p[0], p[1], q)].map(|e| e * area.signum());
                        if w.iter().all(|&e| e >= 0.0) && x < samples && y < samples {
                            covered[y * samples + x] = true;
                        }
                    }
                }
            }
            for y in 0..self.n {
                for x in 0..self.n {
                    let mut hits = 0;
                    for sy in 0..Self::SUB {
                        for sx in 0..Self::SUB {
                            hits += covered[(y * Self::SUB + sy) * samples + x * Self::SUB + sx] as u32;
                        }
                    }
                    self.transmittance[y * self.n + x] *= 1.0 - hits as f32 / (Self::SUB * Self::SUB) as f32;
                }
            }
        }

        /// Voxels seen along z after `rotation` about y: rectangles (the cube's footprint) of
        /// their coverage along that direction (as the renderer dithers them).
        fn voxels(&mut self, voxels: &[Voxel], rotation: f32) {
            let (sin, cos) = rotation.sin_cos();
            let view = Vec3::new(-sin, 0.0, cos);
            for v in voxels {
                let alpha = crate::voxel::directional_coverage(v.coverage, view);
                let center = Vec2::new(cos * v.center.x + sin * v.center.z, v.center.y);
                let extent = Vec2::new(v.size * (cos.abs() + sin.abs()), v.size);
                let lo = (center - extent * 0.5 - Self::ORIGIN) / self.pixel;
                let hi = lo + extent / self.pixel;
                for y in lo.y.floor().max(0.0) as usize..(hi.y.ceil() as usize).min(self.n) {
                    for x in lo.x.floor().max(0.0) as usize..(hi.x.ceil() as usize).min(self.n) {
                        let overlap = (hi.x.min(x as f32 + 1.0) - lo.x.max(x as f32)).max(0.0) * (hi.y.min(y as f32 + 1.0) - lo.y.max(y as f32)).max(0.0);
                        self.transmittance[y * self.n + x] *= 1.0 - alpha * overlap;
                    }
                }
            }
        }

        fn mean(&self) -> f32 {
            self.transmittance.iter().map(|t| 1.0 - t).sum::<f32>() / self.transmittance.len() as f32
        }
    }

    /// Distant foliage keeps its coverage with voxels: projected along each axis, the voxel cut
    /// covers what the full-detail mesh covers within a few percent at every distance. (The mean
    /// coverage does not depend on the pixel size, so all images share a fine one.)
    #[test]
    fn voxel_cuts_preserve_coverage() {
        let (mesh, vm) = build_foliage();
        const PIXEL: f32 = 0.004;
        let full: Vec<[Vec3; 3]> = mesh.triangles().map(|t| t.map(|v| mesh.vertices[v as usize].position())).collect();
        // Views along x, y and z, and diagonally between x and z: the coverage image projects
        // along z after a swizzle and a rotation about y.
        let quarter = std::f32::consts::FRAC_PI_4;
        let views: [(fn(Vec3) -> Vec3, f32); 4] = [(|p| p, 0.0), (|p| Vec3::new(p.y, p.z, p.x), 0.0), (|p| Vec3::new(p.z, p.x, p.y), 0.0), (|p| p, quarter)];
        for (a, &(swizzle, rotation)) in views.iter().enumerate() {
            let place = |p: Vec3| glam::Quat::from_rotation_y(rotation) * swizzle(p);
            let mut reference = Coverage::new(PIXEL);
            reference.triangles(&full.iter().map(|t| t.map(place)).collect::<Vec<_>>());
            let reference = reference.mean();
            for distance in [30.0f32, 120.0, 500.0, 2000.0] {
                let render = |voxels: bool| {
                    let mut image = Coverage::new(PIXEL);
                    let cut = lod_cut_in(&vm, &view(distance), &|_| true, voxels);
                    let triangles: Vec<[Vec3; 3]> = cut.iter().flat_map(|&c| vm.cluster_triangle_positions(c as usize)).map(|t| t.map(place)).collect();
                    image.triangles(&triangles);
                    let voxels: Vec<Voxel> = cut.iter().flat_map(|&c| vm.cluster_voxels(c as usize)).map(|v| Voxel { center: swizzle(v.center), coverage: swizzle(v.coverage), ..v }).collect();
                    image.voxels(&voxels, rotation);
                    image.mean()
                };
                let (with_voxels, triangles_only) = (render(true), render(false));
                let relative = |c: f32| (c - reference) / reference * 100.0;
                println!(
                    "axis {a}, {distance:5} m: full {reference:.3}, voxel cut {with_voxels:.3} ({:+.1} %), triangle cut {triangles_only:.3} ({:+.1} %)",
                    relative(with_voxels),
                    relative(triangles_only)
                );
                assert!(relative(with_voxels).abs() < 10.0, "voxel cut coverage {with_voxels} vs {reference} at {distance} m along axis {a}");
            }
        }
    }

    /// Voxel alternates leave the triangle DAG's cut watertight (a solid mesh where voxels win
    /// near the root only).
    #[test]
    fn triangle_cut_of_a_voxel_mesh_stays_watertight() {
        let vm = build_virtual_mesh(&primitives::rock(7, 5), &BuildConfig { sdf: SdfMode::None, voxels: Some(VoxelConfig::default()), ..Default::default() });
        for distance in [1.5f32, 6.0, 25.0, 100.0, 400.0] {
            assert_watertight(&vm, &lod_cut(&vm, &view(distance)));
        }
        let back = VirtualMesh::from_bytes(&vm.to_bytes()).unwrap();
        assert_eq!(back.clusters, vm.clusters);
        assert_eq!(back.stats, vm.stats);
    }
}
