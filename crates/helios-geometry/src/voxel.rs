//! Voxel LODs for aggregate geometry (Nanite Foliage, UE 5.7).
//!
//! Canopies, needles and grass are made of many small, disconnected pieces. Simplifying them as
//! triangles must delete pieces (component pruning, `simplify.rs`), which thins the aggregate:
//! distant forests lose coverage and colour. A voxel cluster stands for the same region as a
//! sparse grid of voxels, each carrying the *statistics* of the geometry inside it — how much of
//! the voxel it covers seen along each axis, its dominant orientation and its texture coordinates
//! — so coverage is preserved at any resolution. The DAG builder (`dag.rs`) offers voxel clusters
//! as an alternative to a group's simplified triangles wherever their error is lower.
//!
//! ## Coverage
//!
//! A *piece* (a connected component: a leaf, a blade, a frond) inside a voxel of edge `s`
//! blocks rays along axis `i` over its projected area `a_i`. Where its triangles facing `+i` and
//! those facing `−i` project onto the same place — the two sides of a folded or two-sided leaf,
//! the bump of a curved one — that area is `max(Σ₊ a·n_i, Σ₋ a·|n_i|)` rather than the sum. A
//! piece covers the fraction `min(a_i / s², 1)` of the voxel's face; pieces lie independently of
//! each other, so the voxel's coverage along that axis is `α_i = 1 − Π (1 − min(a_i / s², 1))`
//! (Beer–Lambert's `1 − e^(−Σ a_i / s²)` for many small leaves, exact for one leaf filling a
//! voxel). Seen from direction `d`, a cube's projected area is `s²·Σ|d_i|`, face `i`
//! contributing `s²·|d_i|`, so its coverage is `Σ|d_i|·α_i / Σ|d_i|`
//! ([`directional_coverage`]). The renderer draws each voxel as an opaque cube dithered with
//! that coverage (stochastic transparency, averaged by temporal upsampling).
//!
//! The geometry is sampled with points spaced at most half a voxel apart on every triangle, so
//! triangles spanning several voxels spread their area over all of them.
//!
//! ## Orientation and colour
//!
//! A voxel's normal blends the principal axis of its geometry's normals (leaves facing one way)
//! with the outward direction of the aggregate (the negative gradient of the coverage over the
//! group's voxels), by how anisotropic the normals are: an isotropic tangle of leaves shades
//! like the canopy surface it belongs to. Bricks keep the mean texture coordinate of their
//! geometry and its spread, so the material is sampled where (and as blurred as) the leaves are.
//!
//! ## Layout of one voxel cluster (words, at `GpuCluster::vertex_offset` of its page)
//!
//! ```text
//! [0..3) grid origin (object space, f32)     [3] voxel edge length (f32)
//! [4]    bricks | voxels << 8                [5..7) uv minimum (f32)   [7] uv range (f32)
//! [8 + 4 b ..)  brick b: occupancy bits 0..32, 32..64 (bit x + 4y + 16z of its 4³ voxels),
//!               position x | y << 6 | z << 12 (bricks from the origin) | first voxel << 18,
//!               mean uv u | v << 12 (12-bit fractions of the uv range) | uv spread << 24
//! [8 + 4 bricks ..) one word per occupied voxel, brick by brick in bit order:
//!               octahedral normal x | y << 8 | coverage x << 16 | y << 21 | z << 26
//!               (5 bits each, square-root code: α = (c / 31)²)
//! plants only (word 4 bit 31, `.hvm` version 6): then the plant bone of every brick, 16 bits
//!               each, two to a word (brick 2k in the low half)
//! ```
//!
//! A plant's bricks follow the bone most of their geometry belongs to — the branch its leaves
//! hang from ([`crate::plant_bones::voxel_bone`]) — so distant voxels move with the branches as
//! the triangles they stand in for do.
//!
//! A cluster holds at most [`MAX_CLUSTER_BRICKS`] bricks within 64 bricks along each axis.

use glam::{IVec3, Vec2, Vec3};

use crate::encoding::{decode_octahedral, encode_octahedral};
use crate::hash::FxHashMap;
use crate::Sphere;

/// Voxels along each axis of a brick.
pub const BRICK_SIZE: i32 = 4;
/// Bricks per voxel cluster (one per thread of a 128-wide rasteriser workgroup).
pub const MAX_CLUSTER_BRICKS: usize = 128;
/// Words in front of a voxel cluster's bricks.
pub const VOXEL_HEADER_WORDS: u32 = 8;
/// Words per brick record.
pub const BRICK_WORDS: u32 = 4;
/// A voxel cluster spans at most this many bricks along an axis (6-bit brick coordinates).
const MAX_BRICK_SPAN: i32 = 63;
/// Bits per octahedral normal component.
const NORMAL_BITS: u32 = 8;
/// Largest coverage code (5 bits).
const COVERAGE_MAX: f32 = 31.0;
/// Geometric error of a voxel representation per unit of voxel edge length: at the LOD threshold
/// a voxel covers about a pixel.
pub const VOXEL_ERROR_PER_SIZE: f32 = 1.0;
/// Triangles are sampled at most this fraction of a voxel apart.
const SAMPLE_SPACING: f32 = 0.5;

/// Voxel LOD parameters of an aggregate mesh ([`crate::BuildConfig::voxels`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelConfig {
    /// Voxels a group may spend for every triangle its triangle simplification keeps. The finest
    /// voxels within this budget compete with the simplified triangles' error: more voxels win
    /// at more levels (finer voxels, more memory and raster work). A winning group never uses
    /// voxels finer than the triangles' error, so its LOD switches where the triangles' would.
    pub voxels_per_triangle: f32,
}

impl Default for VoxelConfig {
    fn default() -> Self {
        Self { voxels_per_triangle: 4.0 }
    }
}

/// A full-detail source triangle and the piece (connected component) it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SourceTriangle {
    pub vertices: [u32; 3],
    pub piece: u32,
}

/// The connected components of a mesh's triangles, over vertices identified by position
/// (`canonical`, see [`crate::canonical_ids`]): two-sided sheets made of coincident, opposite
/// triangles are one piece.
pub(crate) struct Pieces<'a> {
    canonical: &'a [u32],
    root: FxHashMap<u32, u32>,
}

impl<'a> Pieces<'a> {
    pub fn new(triangles: &[[u32; 3]], canonical: &'a [u32]) -> Self {
        let mut parent: FxHashMap<u32, u32> = FxHashMap::default();
        fn find(parent: &mut FxHashMap<u32, u32>, x: u32) -> u32 {
            let mut root = x;
            while let Some(&p) = parent.get(&root).filter(|&&p| p != root) {
                root = p;
            }
            let mut y = x;
            while y != root {
                let next = parent.insert(y, root).unwrap_or(root);
                y = next;
            }
            root
        }
        for t in triangles {
            let c = t.map(|v| canonical[v as usize]);
            for &x in &c {
                parent.entry(x).or_insert(x);
            }
            for k in 1..3 {
                let (a, b) = (find(&mut parent, c[0]), find(&mut parent, c[k]));
                if a != b {
                    parent.insert(a, b);
                }
            }
        }
        let classes: Vec<u32> = parent.keys().copied().collect();
        let root = classes.into_iter().map(|x| (x, find(&mut parent, x))).collect();
        Self { canonical, root }
    }

    pub fn source(&self, t: [u32; 3]) -> SourceTriangle {
        SourceTriangle { vertices: t, piece: self.root[&self.canonical[t[0] as usize]] }
    }
}

/// Vertex attributes the voxeliser reads.
#[derive(Clone, Copy)]
pub(crate) struct VoxelSource<'a> {
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2],
    /// Per vertex, the plant bone its voxels follow (empty: not a plant).
    pub bones: &'a [u16],
}

/// Header word 4 flag of a voxel cluster whose bricks carry plant bones.
pub const VOXEL_BONES_FLAG: u32 = 1 << 31;

/// A voxel grid: cell `c` spans `origin + c·size .. origin + (c + 1)·size`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Grid {
    pub origin: Vec3,
    pub size: f32,
}

impl Grid {
    fn cell(&self, p: Vec3) -> IVec3 {
        ((p - self.origin) / self.size).floor().as_ivec3()
    }

    fn center(&self, cell: IVec3) -> Vec3 {
        self.origin + (cell.as_vec3() + 0.5) * self.size
    }
}

/// Statistics of the geometry inside one voxel.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cell {
    /// Per axis, ln of the fraction of the voxel's face its pieces leave uncovered.
    log_transmittance: Vec3,
    /// Σ a·n nᵀ (xx, yy, zz, xy, xz, yz).
    covariance: [f32; 6],
    /// Σ a, and the area-weighted sums of uv and |uv|².
    area: f32,
    uv: Vec2,
    uv_square: f32,
    /// Plant bones of the geometry and their areas (the four largest).
    bones: [(u16, f32); 4],
}

impl Cell {
    fn add_bone(&mut self, bone: u16, area: f32) {
        if let Some(slot) = self.bones.iter_mut().find(|b| b.0 == bone && b.1 > 0.0) {
            slot.1 += area;
        } else if let Some(slot) = self.bones.iter_mut().min_by(|a, b| a.1.total_cmp(&b.1)).filter(|b| b.1 < area) {
            *slot = (bone, area);
        }
    }

    fn add(&mut self, area: f32, n: Vec3, uv: Vec2) {
        let c = &mut self.covariance;
        c[0] += n.x * n.x * area;
        c[1] += n.y * n.y * area;
        c[2] += n.z * n.z * area;
        c[3] += n.x * n.y * area;
        c[4] += n.x * n.z * area;
        c[5] += n.y * n.z * area;
        self.area += area;
        self.uv += uv * area;
        self.uv_square += uv.length_squared() * area;
    }

    /// Coverage seen along each axis.
    fn coverage(&self) -> Vec3 {
        Vec3::ONE - self.log_transmittance.exp()
    }

    /// Principal axis of the normals and how dominant it is (0: isotropic, 1: one plane).
    fn principal_axis(&self, start: Vec3) -> (Vec3, f32) {
        let c = &self.covariance;
        let apply = |v: Vec3| {
            Vec3::new(c[0] * v.x + c[3] * v.y + c[4] * v.z, c[3] * v.x + c[1] * v.y + c[5] * v.z, c[4] * v.x + c[5] * v.y + c[2] * v.z)
        };
        let mut v = (start + Vec3::new(0.11, 0.07, 0.05)).normalize_or(Vec3::Y);
        for _ in 0..12 {
            v = apply(v).normalize_or(v);
        }
        let trace = c[0] + c[1] + c[2];
        let ratio = if trace > 0.0 { v.dot(apply(v)) / trace } else { 1.0 / 3.0 };
        (v, ((ratio - 1.0 / 3.0) * 1.5).clamp(0.0, 1.0))
    }
}

/// 21 bits per axis around the origin.
fn cell_key(c: IVec3) -> u64 {
    let field = |v: i32| ((v + (1 << 20)) as u64) & 0x1f_ffff;
    field(c.x) | (field(c.y) << 21) | (field(c.z) << 42)
}

fn key_cell(key: u64) -> IVec3 {
    let field = |shift: u32| ((key >> shift) & 0x1f_ffff) as i32 - (1 << 20);
    IVec3::new(field(0), field(21), field(42))
}

/// Calls `f(point, area)` for points spread over the triangle at most `spacing` apart: the
/// centroids of its regular subdivision into n² equal triangles.
fn sample_triangle(p: [Vec3; 3], spacing: f32, mut f: impl FnMut(Vec3, f32)) {
    let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
    let area = e1.cross(e2).length() * 0.5;
    if area <= 0.0 {
        return;
    }
    let longest = e1.length().max(e2.length()).max((p[2] - p[1]).length());
    let n = (longest / spacing).ceil().clamp(1.0, 64.0) as u32;
    let (inv, piece) = (1.0 / n as f32, area / (n * n) as f32);
    for i in 0..n {
        for j in 0..n - i {
            let (u, v) = (i as f32, j as f32);
            f(p[0] + e1 * ((u + 1.0 / 3.0) * inv) + e2 * ((v + 1.0 / 3.0) * inv), piece);
            if i + j + 1 < n {
                f(p[0] + e1 * ((u + 2.0 / 3.0) * inv) + e2 * ((v + 2.0 / 3.0) * inv), piece);
            }
        }
    }
}

/// Accumulates the geometry of `triangles` into the cells of `grid`.
pub(crate) fn voxelize(source: &VoxelSource, triangles: &[SourceTriangle], grid: Grid) -> FxHashMap<u64, Cell> {
    let mut cells: FxHashMap<u64, Cell> = FxHashMap::default();
    // Per (cell, piece): the area facing +axis and -axis.
    let mut sides: FxHashMap<(u64, u32), (Vec3, Vec3)> = FxHashMap::default();
    for t in triangles {
        let p = t.vertices.map(|v| source.positions[v as usize]);
        let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let uv = t.vertices.iter().map(|&v| source.uvs.get(v as usize).copied().unwrap_or_default()).sum::<Vec2>() / 3.0;
        // Shading normals of a flat leaf all agree with its face; the face normal also covers
        // meshes whose normals were smoothed across pieces.
        let n = if face == Vec3::ZERO {
            t.vertices.iter().map(|&v| source.normals.get(v as usize).copied().unwrap_or(Vec3::Y)).sum::<Vec3>().normalize_or(Vec3::Y)
        } else {
            face
        };
        let (positive, negative) = (n.max(Vec3::ZERO), (-n).max(Vec3::ZERO));
        let bone = source.bones.get(t.vertices[0] as usize).copied();
        sample_triangle(p, grid.size * SAMPLE_SPACING, |point, area| {
            let key = cell_key(grid.cell(point));
            let cell = cells.entry(key).or_default();
            cell.add(area, n, uv);
            if let Some(bone) = bone {
                cell.add_bone(bone, area);
            }
            let side = sides.entry((key, t.piece)).or_default();
            side.0 += positive * area;
            side.1 += negative * area;
        });
    }
    // Each piece covers its projected share of the face; independent pieces multiply the
    // fractions they leave open (a piece filling the face still leaves a sliver: no log of 0).
    let face = grid.size * grid.size;
    for ((key, _), (positive, negative)) in sides {
        let covered = (positive.max(negative) / face).min(Vec3::splat(0.999));
        cells.get_mut(&key).unwrap().log_transmittance += (Vec3::ONE - covered).map(f32::ln);
    }
    cells
}

/// Total area and bounds of `triangles`.
fn extent_of(source: &VoxelSource, triangles: &[SourceTriangle]) -> (f32, Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    let mut area = 0.0f32;
    for t in triangles {
        let p = t.vertices.map(|v| source.positions[v as usize]);
        area += (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5;
        for q in p {
            lo = lo.min(q);
            hi = hi.max(q);
        }
    }
    (area, lo, hi)
}

/// The voxelisation of `triangles` on the grid of voxel size `size` aligned to their bounds.
pub(crate) fn fit_grid_at(source: &VoxelSource, triangles: &[SourceTriangle], size: f32) -> (Grid, FxHashMap<u64, Cell>) {
    let (_, lo, _) = extent_of(source, triangles);
    let grid = Grid { origin: (lo / size).floor() * size, size };
    (grid, voxelize(source, triangles, grid))
}

/// The finest voxelisation of a group (no finer than `min_size`) whose occupied voxels stay
/// within `budget`. `None` if the triangles have no area.
pub(crate) fn fit_grid(source: &VoxelSource, triangles: &[SourceTriangle], budget: usize, min_size: f32) -> Option<(Grid, FxHashMap<u64, Cell>)> {
    let (area, lo, hi) = extent_of(source, triangles);
    if area <= 0.0 || budget == 0 {
        return None;
    }
    // A sheet puts `size²` of area in every voxel it crosses: a first guess, refined by counting.
    // Counts fall roughly with the square of the size (a surface-like aggregate).
    let extent = (hi - lo).max_element().max(1e-6);
    let mut size = (area / budget as f32).sqrt().max(min_size).max(extent / 1024.0);
    let grid_at = |size: f32| Grid { origin: (lo / size).floor() * size, size };
    let mut best: Option<(Grid, FxHashMap<u64, Cell>)> = None;
    for _ in 0..8 {
        let grid = grid_at(size);
        let cells = voxelize(source, triangles, grid);
        let count = cells.len();
        if count <= budget {
            let done = count * 2 > budget || size <= min_size;
            best = Some((grid, cells));
            if done {
                break;
            }
            size = (size * (count as f32 / budget as f32).sqrt().max(0.5)).max(min_size);
        } else {
            if best.is_some() {
                break;
            }
            size *= (count as f32 / budget as f32).sqrt().max(1.05) * 1.02;
        }
    }
    best.or_else(|| {
        // Never more than a few voxels per budget unit: the coarsest single-brick grid.
        let grid = grid_at(extent / BRICK_SIZE as f32);
        let cells = voxelize(source, triangles, grid);
        Some((grid, cells))
    })
}

/// Coverage of a voxel seen along `direction`, from its per-axis coverage.
///
/// Seen along a unit direction `d`, the cube's footprint is `s²·Σ|d_i|`, while its leaves cover
/// about `sqrt(Σ (d_i·A_i)²)` of it, `A_i` their projection along axis `i` (exact for leaves
/// facing an axis and for an isotropic tangle, whose projection is the same from everywhere;
/// adding the axes' projections linearly would count a leaf seen diagonally up to √3 times). In
/// optical depth, so a dense voxel stays opaque from every side:
/// `α(d) = 1 − exp(−sqrt(Σ (d_i·τ_i)²) / Σ|d_i|)`, `τ_i = −ln(1 − α_i)`.
pub fn directional_coverage(coverage: Vec3, direction: Vec3) -> f32 {
    let d = direction.normalize_or(Vec3::Y);
    let tau = -(Vec3::ONE - coverage.min(Vec3::splat(0.999))).map(f32::ln);
    1.0 - (-(d * tau).length() / d.abs().element_sum()).exp()
}

fn encode_coverage(alpha: f32) -> u32 {
    (alpha.clamp(0.0, 1.0).sqrt() * COVERAGE_MAX).round() as u32
}

fn decode_coverage(code: u32) -> f32 {
    let c = code as f32 / COVERAGE_MAX;
    c * c
}

/// uv spread code: log2 in eighths from 2^-16.
fn encode_spread(spread: f32) -> u32 {
    ((spread.max(1e-6).log2() + 16.0) * 8.0).round().clamp(0.0, 255.0) as u32
}

fn decode_spread(code: u32) -> f32 {
    (code as f32 / 8.0 - 16.0).exp2()
}

/// An encoded voxel cluster.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VoxelCluster {
    pub words: Vec<u32>,
    pub bounds: Sphere,
    pub bricks: u32,
    pub voxels: u32,
}

/// Voxel clusters for `triangles` on `grid`. `group` is the voxelisation of the whole group the
/// triangles belong to (on the same grid): the outward direction of the aggregate comes from it.
pub(crate) fn build_voxel_clusters(source: &VoxelSource, triangles: &[SourceTriangle], grid: Grid, group: &FxHashMap<u64, Cell>) -> Vec<VoxelCluster> {
    let cells = voxelize(source, triangles, grid);
    if cells.is_empty() {
        return Vec::new();
    }
    let density_at = |c: IVec3| group.get(&cell_key(c)).map_or(0.0, |cell| cell.coverage().element_sum());
    let centroid = {
        let (sum, weight) = group.iter().fold((Vec3::ZERO, 0.0), |(s, w), (&k, c)| (s + grid.center(key_cell(k)) * c.area, w + c.area));
        sum / weight.max(1e-30)
    };

    // Bricks of the occupied voxels, in Z order.
    let mut bricks: FxHashMap<IVec3, Vec<(u32, IVec3)>> = FxHashMap::default();
    for &key in cells.keys() {
        let c = key_cell(key);
        let brick = c.div_euclid(IVec3::splat(BRICK_SIZE));
        let local = c - brick * BRICK_SIZE;
        bricks.entry(brick).or_default().push(((local.x + 4 * local.y + 16 * local.z) as u32, c));
    }
    let lo = bricks.keys().copied().reduce(IVec3::min).unwrap();
    let mut order: Vec<(u64, IVec3)> = bricks.keys().map(|&b| (morton(b - lo), b)).collect();
    order.sort_unstable_by_key(|&(code, _)| code);

    // Clusters of up to MAX_CLUSTER_BRICKS consecutive bricks within MAX_BRICK_SPAN.
    let mut out = Vec::new();
    let mut chunk: Vec<IVec3> = Vec::new();
    let (mut chunk_lo, mut chunk_hi) = (IVec3::ZERO, IVec3::ZERO);
    for (_, brick) in order {
        let (new_lo, new_hi) = if chunk.is_empty() { (brick, brick) } else { (chunk_lo.min(brick), chunk_hi.max(brick)) };
        if !chunk.is_empty() && (chunk.len() == MAX_CLUSTER_BRICKS || (new_hi - new_lo).max_element() > MAX_BRICK_SPAN) {
            out.push(encode_cluster(&chunk, chunk_lo, &bricks, &cells, grid, &density_at, centroid));
            chunk.clear();
            (chunk_lo, chunk_hi) = (brick, brick);
        } else {
            (chunk_lo, chunk_hi) = (new_lo, new_hi);
        }
        chunk.push(brick);
    }
    out.push(encode_cluster(&chunk, chunk_lo, &bricks, &cells, grid, &density_at, centroid));
    out
}

fn morton(c: IVec3) -> u64 {
    let spread = |v: i32| {
        let mut x = (v.max(0) as u64) & 0x1f_ffff;
        x = (x | (x << 32)) & 0x1f00000000ffff;
        x = (x | (x << 16)) & 0x1f0000ff0000ff;
        x = (x | (x << 8)) & 0x100f00f00f00f00f;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3;
        (x | (x << 2)) & 0x1249249249249249
    };
    spread(c.x) | (spread(c.y) << 1) | (spread(c.z) << 2)
}

fn encode_cluster(
    chunk: &[IVec3],
    lo: IVec3,
    bricks: &FxHashMap<IVec3, Vec<(u32, IVec3)>>,
    cells: &FxHashMap<u64, Cell>,
    grid: Grid,
    density_at: &dyn Fn(IVec3) -> f32,
    centroid: Vec3,
) -> VoxelCluster {
    let size = grid.size;
    // Per brick: its voxels in bit order, mean uv and uv spread.
    struct Brick {
        mask: u64,
        position: IVec3,
        voxels: Vec<IVec3>,
        uv: Vec2,
        spread: f32,
        /// The plant bone with the most area in the brick.
        bone: Option<u16>,
    }
    let records: Vec<Brick> = chunk
        .iter()
        .map(|&b| {
            let mut voxels = bricks[&b].clone();
            voxels.sort_unstable_by_key(|&(bit, _)| bit);
            let mask = voxels.iter().fold(0u64, |m, &(bit, _)| m | (1 << bit));
            let (mut area, mut uv, mut uv_square) = (0.0, Vec2::ZERO, 0.0);
            let mut bones: FxHashMap<u16, f32> = FxHashMap::default();
            for &(_, c) in &voxels {
                let cell = &cells[&cell_key(c)];
                area += cell.area;
                uv += cell.uv;
                uv_square += cell.uv_square;
                for &(bone, a) in cell.bones.iter().filter(|b| b.1 > 0.0) {
                    *bones.entry(bone).or_default() += a;
                }
            }
            let mean = uv / area.max(1e-30);
            let variance = (uv_square / area.max(1e-30) - mean.length_squared()).max(0.0);
            let bone = bones.into_iter().max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0))).map(|b| b.0);
            Brick { mask, position: b - lo, voxels: voxels.into_iter().map(|(_, c)| c).collect(), uv: mean, spread: 2.0 * variance.sqrt(), bone }
        })
        .collect();

    let uv_min = records.iter().map(|b| b.uv).reduce(Vec2::min).unwrap();
    let uv_max = records.iter().map(|b| b.uv).reduce(Vec2::max).unwrap();
    let uv_range = (uv_max - uv_min).max_element().max(1e-6);
    let voxel_count: usize = records.iter().map(|b| b.voxels.len()).sum();
    let origin = grid.origin + (lo * BRICK_SIZE).as_vec3() * size;

    let mut words = Vec::with_capacity(VOXEL_HEADER_WORDS as usize + records.len() * BRICK_WORDS as usize + voxel_count);
    words.extend([origin.x.to_bits(), origin.y.to_bits(), origin.z.to_bits(), size.to_bits()]);
    let has_bones = records.iter().any(|b| b.bone.is_some());
    let flags = if has_bones { VOXEL_BONES_FLAG } else { 0 };
    words.extend([records.len() as u32 | (voxel_count as u32) << 8 | flags, uv_min.x.to_bits(), uv_min.y.to_bits(), uv_range.to_bits()]);
    let mut first = 0u32;
    for b in &records {
        let p = b.position.as_uvec3();
        let uv = ((b.uv - uv_min) / uv_range * 4095.0).round().as_uvec2();
        words.extend([b.mask as u32, (b.mask >> 32) as u32, p.x | (p.y << 6) | (p.z << 12) | (first << 18), uv.x | (uv.y << 12) | (encode_spread(b.spread) << 24)]);
        first += b.voxels.len() as u32;
    }
    let mut points = Vec::with_capacity(voxel_count);
    for b in &records {
        for &c in &b.voxels {
            let cell = &cells[&cell_key(c)];
            let center = grid.center(c);
            points.push(center);
            let coverage = cell.coverage();
            // Outward: down the coverage gradient of the group's aggregate.
            let gradient = Vec3::new(
                density_at(c + IVec3::X) - density_at(c - IVec3::X),
                density_at(c + IVec3::Y) - density_at(c - IVec3::Y),
                density_at(c + IVec3::Z) - density_at(c - IVec3::Z),
            );
            let outward = (-gradient).try_normalize().unwrap_or_else(|| (center - centroid).normalize_or(Vec3::Y));
            let (axis, anisotropy) = cell.principal_axis(outward);
            let axis = if axis.dot(outward) < 0.0 { -axis } else { axis };
            let normal = (axis * anisotropy + outward * (1.0 - anisotropy)).normalize_or(outward);
            let [x, y, z] = coverage.to_array().map(encode_coverage);
            words.push(encode_octahedral(normal, NORMAL_BITS) | (x << 16) | (y << 21) | (z << 26));
        }
    }
    if has_bones {
        for pair in records.chunks(2) {
            let bone = |b: &Brick| b.bone.unwrap_or(0) as u32;
            words.push(bone(&pair[0]) | pair.get(1).map_or(0, bone) << 16);
        }
    }
    let sphere = Sphere::from_points(&points);
    VoxelCluster {
        words,
        bounds: Sphere::new(sphere.center, sphere.radius + size * 0.8660254),
        bricks: records.len() as u32,
        voxels: voxel_count as u32,
    }
}

/// A decoded voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voxel {
    pub center: Vec3,
    pub size: f32,
    /// Coverage seen along x, y and z ([`directional_coverage`]).
    pub coverage: Vec3,
    pub normal: Vec3,
    /// Mean texture coordinate of its brick and the spread around it.
    pub uv: Vec2,
    pub uv_spread: f32,
    pub brick: u32,
}

/// (bricks, voxels) of the voxel cluster starting at `words`.
pub fn voxel_counts(words: &[u32]) -> (u32, u32) {
    (words[4] & 0xff, (words[4] >> 8) & 0x7fff)
}

/// The plant bone of brick `brick` of the voxel cluster starting at `words` (`None` if its
/// bricks carry none).
pub fn voxel_brick_bone(words: &[u32], brick: u32) -> Option<u16> {
    if words[4] & VOXEL_BONES_FLAG == 0 {
        return None;
    }
    let (bricks, voxels) = voxel_counts(words);
    let word = words[(VOXEL_HEADER_WORDS + bricks * BRICK_WORDS + voxels + brick / 2) as usize];
    Some((word >> (16 * (brick & 1))) as u16)
}

/// Decodes the voxel cluster starting at `words` (exactly as the renderer does).
pub fn decode_voxel_cluster(words: &[u32]) -> Vec<Voxel> {
    let f = |i: usize| f32::from_bits(words[i]);
    let (origin, size) = (Vec3::new(f(0), f(1), f(2)), f(3));
    let (bricks, _) = voxel_counts(words);
    let (uv_min, uv_range) = (Vec2::new(f(5), f(6)), f(7));
    let attributes = (VOXEL_HEADER_WORDS + bricks * BRICK_WORDS) as usize;
    let mut out = Vec::new();
    for b in 0..bricks {
        let r = &words[(VOXEL_HEADER_WORDS + b * BRICK_WORDS) as usize..][..BRICK_WORDS as usize];
        let mask = r[0] as u64 | (r[1] as u64) << 32;
        let position = IVec3::new((r[2] & 63) as i32, ((r[2] >> 6) & 63) as i32, ((r[2] >> 12) & 63) as i32);
        let first = r[2] >> 18;
        let uv = uv_min + Vec2::new((r[3] & 4095) as f32, ((r[3] >> 12) & 4095) as f32) / 4095.0 * uv_range;
        let spread = decode_spread(r[3] >> 24);
        let mut index = first as usize;
        for bit in (0..64).filter(|&bit| mask & (1 << bit) != 0) {
            let local = IVec3::new(bit & 3, (bit >> 2) & 3, bit >> 4);
            let cell = position * BRICK_SIZE + local;
            let a = words[attributes + index];
            index += 1;
            out.push(Voxel {
                center: origin + (cell.as_vec3() + 0.5) * size,
                size,
                coverage: Vec3::new(decode_coverage((a >> 16) & 31), decode_coverage((a >> 21) & 31), decode_coverage((a >> 26) & 31)),
                normal: decode_octahedral(a & 0xffff, NORMAL_BITS),
                uv,
                uv_spread: spread,
                brick: b,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn source_of(mesh: &crate::Mesh) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec2>, Vec<SourceTriangle>) {
        let positions = mesh.positions();
        let normals = mesh.vertices.iter().map(|v| v.normal()).collect();
        let uvs = mesh.vertices.iter().map(|v| v.uv()).collect();
        let canonical = crate::canonical_ids(mesh.vertices.iter().map(|v| v.position));
        let all: Vec<[u32; 3]> = mesh.triangles().collect();
        let pieces = Pieces::new(&all, &canonical);
        let triangles = all.iter().map(|&t| pieces.source(t)).collect();
        (positions, normals, uvs, triangles)
    }

    #[test]
    fn sampling_preserves_area() {
        let p = [Vec3::ZERO, Vec3::new(1.3, 0.0, 0.2), Vec3::new(0.1, 0.9, 0.0)];
        let exact = (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5;
        for spacing in [2.0, 0.3, 0.05] {
            let (mut area, mut count) = (0.0, 0);
            sample_triangle(p, spacing, |_, a| {
                area += a;
                count += 1;
            });
            assert!((area - exact).abs() < exact * 1e-4, "{area} vs {exact}");
            assert!(count as f32 >= (1.3f32 / spacing).floor().powi(2));
        }
    }

    #[test]
    fn two_sided_sheets_count_once() {
        let mut b = crate::modeling::MeshBuilder::default();
        crate::vegetation::add_leaf(&mut b, Vec3::ZERO, Vec3::X, Vec3::Y, crate::vegetation::LeafShape { length: 1.0, width: 0.5, fold: 0.0 });
        let mesh = b.build();
        let (positions, normals, uvs, triangles) = source_of(&mesh);
        assert!(triangles.iter().all(|t| t.piece == triangles[0].piece));
        // The flat leaf (normal ±y) seen from above covers its own area once: 0.25 m² of a
        // 2 m voxel's face.
        let grid = Grid { origin: Vec3::splat(-1.0), size: 2.0 };
        let cells = voxelize(&VoxelSource { positions: &positions, normals: &normals, uvs: &uvs, bones: &[] }, &triangles, grid);
        assert_eq!(cells.len(), 1);
        let coverage = cells.values().next().unwrap().coverage();
        assert!((coverage.y - 0.25 / 4.0).abs() < 1e-4, "{coverage}");
        assert!(coverage.x.abs() < 1e-4, "seen edge-on: {coverage}");
    }

    /// One leaf filling a voxel's face covers it; many small leaves saturate as Beer–Lambert's
    /// law has it.
    #[test]
    fn pieces_combine_independently() {
        let mut b = crate::modeling::MeshBuilder::default();
        b.quad(Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.0, 0.5, 1.0), Vec3::new(1.0, 0.5, 1.0), Vec3::new(1.0, 0.5, 0.0));
        let (positions, normals, uvs, triangles) = source_of(&b.build());
        let source = VoxelSource { positions: &positions, normals: &normals, uvs: &uvs, bones: &[] };
        let cells = voxelize(&source, &triangles, Grid { origin: Vec3::ZERO, size: 1.0 });
        assert!(cells.values().next().unwrap().coverage().y > 0.99);

        let mesh = primitives::leaf_cloud(400, 7).transformed(glam::Mat4::from_scale(Vec3::splat(0.5)));
        let (positions, normals, uvs, triangles) = source_of(&mesh);
        let source = VoxelSource { positions: &positions, normals: &normals, uvs: &uvs, bones: &[] };
        let cells = voxelize(&source, &triangles, Grid { origin: Vec3::splat(-1.0), size: 2.0 });
        let coverage = cells.values().next().unwrap().coverage();
        let projected: f32 = mesh.triangles().map(|t| {
            let p = t.map(|v| mesh.vertices[v as usize].position());
            (p[1] - p[0]).cross(p[2] - p[0]).y.abs() * 0.5
        }).sum();
        let poisson = 1.0 - (-projected / 4.0).exp();
        assert!((coverage.y - poisson).abs() < 0.05, "{} vs {poisson}", coverage.y);
    }

    /// A densely leaved block reads as nearly opaque, a sparse one as mostly transparent, and
    /// the encoding round-trips the voxels' positions and coverage.
    #[test]
    fn coverage_follows_leaf_density_and_round_trips() {
        let mesh = primitives::leaf_cloud(4000, 3);
        let (positions, normals, uvs, triangles) = source_of(&mesh);
        let source = VoxelSource { positions: &positions, normals: &normals, uvs: &uvs, bones: &[] };
        let (grid, group) = fit_grid(&source, &triangles, 400, 0.0).unwrap();
        assert!(group.len() <= 400 && group.len() > 100, "{} voxels", group.len());
        let clusters = build_voxel_clusters(&source, &triangles, grid, &group);
        let voxels: Vec<Voxel> = clusters.iter().flat_map(|c| decode_voxel_cluster(&c.words)).collect();
        assert_eq!(voxels.len(), group.len());
        for c in &clusters {
            assert!(c.bricks as usize <= MAX_CLUSTER_BRICKS);
            assert_eq!(voxel_counts(&c.words), (c.bricks, c.voxels));
        }
        let mean = voxels.iter().map(|v| directional_coverage(v.coverage, Vec3::new(0.3, 0.8, 0.5))).sum::<f32>() / voxels.len() as f32;
        // 4000 leaves of ~0.012 m² in 8 m³ with ~0.15 m voxels: about one leaf per voxel.
        assert!(mean > 0.1 && mean < 0.9, "mean coverage {mean}");
        for v in &voxels {
            let cell = grid.cell(v.center);
            assert!((grid.center(cell) - v.center).abs().max_element() < grid.size * 1e-3);
            let exact = group[&cell_key(cell)].coverage();
            assert!((v.coverage - exact).abs().max_element() < 0.07, "{} vs {exact}", v.coverage);
            assert!((v.normal.length() - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn fitted_grids_respect_the_budget() {
        let mesh = primitives::leaf_cloud(2000, 5);
        let (positions, normals, uvs, triangles) = source_of(&mesh);
        let source = VoxelSource { positions: &positions, normals: &normals, uvs: &uvs, bones: &[] };
        let mut previous = 0.0;
        for budget in [4000usize, 1000, 250, 60] {
            let (grid, cells) = fit_grid(&source, &triangles, budget, 0.0).unwrap();
            assert!(cells.len() <= budget, "{} voxels for a budget of {budget}", cells.len());
            assert!(grid.size > previous, "smaller budgets need larger voxels");
            previous = grid.size;
        }
        // A minimum size wins over the budget.
        let (grid, _) = fit_grid(&source, &triangles, 100000, 0.5).unwrap();
        assert!(grid.size >= 0.5);
    }
}
