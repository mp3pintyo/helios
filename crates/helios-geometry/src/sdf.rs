//! Mesh signed distance fields for software ray tracing (as used by Lumen's global SDF).
//!
//! * **Volume** fields are narrow-band: every triangle splats exact distances into the voxels
//!   within `band` of it; voxels within a voxel of the surface take their sign from
//!   angle-weighted pseudonormals (Bærentzen & Aanæs), all others are classified by a flood
//!   fill from the grid boundary. Open surfaces (leaves, cloth) become *two-sided*: a thin
//!   shell; closed meshes with hairline cracks still bake as solids.
//! * **Heightfield** fields store the maximum surface height per XZ cell — the landscape
//!   representation Lumen also uses — and are far more accurate than a volume for terrain.
//!
//! Distances are quantised to 8 bits relative to the band (the GPU global SDF is truncated
//! to a few voxels anyway), heights to 16 bits.

use glam::Vec3;

use crate::hash::{edge_key, FxHashMap};
use crate::Aabb;

/// How a mesh's distance field is built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfMode {
    None,
    /// Voxel grid with at most `resolution` voxels along the longest axis. Open surfaces become
    /// a shell half a voxel thick: bake large open sheets (terrain, ground planes) as heightfields.
    Volume { resolution: u32 },
    /// Height map with at most `resolution` cells along the longest horizontal axis.
    Heightfield { resolution: u32 },
}

impl Default for SdfMode {
    fn default() -> Self {
        SdfMode::Volume { resolution: 48 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfKind {
    Volume = 0,
    Heightfield = 1,
}

/// A baked distance field in mesh (object) space.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshSdf {
    pub kind: SdfKind,
    /// Centre of voxel/cell (0, 0, 0).
    pub origin: Vec3,
    pub voxel_size: f32,
    /// Volume: voxels per axis. Heightfield: `[cells_x, 1, cells_z]`.
    pub resolution: [u32; 3],
    /// Distances are stored as `d / band` in `[-1, 1]`.
    pub band: f32,
    /// Heightfield: height range `[min_y, max_y]` of the 16-bit heights.
    pub height_range: [f32; 2],
    pub two_sided: bool,
    /// Volume: snorm8 per voxel (x fastest), 4 per word. Heightfield: unorm16, 2 per word.
    pub words: Vec<u32>,
}

impl MeshSdf {
    pub fn voxel_count(&self) -> usize {
        self.resolution.iter().map(|&r| r as usize).product()
    }

    /// Distance at voxel `(x, y, z)` (volume fields).
    pub fn voxel(&self, x: u32, y: u32, z: u32) -> f32 {
        let [rx, ry, _] = self.resolution;
        let i = (x + rx * (y + ry * z)) as usize;
        let byte = (self.words[i / 4] >> ((i % 4) * 8)) as u8 as i8;
        byte as f32 / 127.0 * self.band
    }

    /// Height of cell `(x, z)` (heightfields).
    pub fn height(&self, x: u32, z: u32) -> f32 {
        let i = (x + self.resolution[0] * z) as usize;
        let h = (self.words[i / 2] >> ((i % 2) * 16)) & 0xffff;
        self.height_range[0] + (self.height_range[1] - self.height_range[0]) * h as f32 / 65535.0
    }

    /// Trilinearly interpolated distance at object-space `p` (clamped to the grid).
    pub fn sample(&self, p: Vec3) -> f32 {
        match self.kind {
            SdfKind::Volume => {
                let g = ((p - self.origin) / self.voxel_size).max(Vec3::ZERO);
                let max = Vec3::new(self.resolution[0] as f32, self.resolution[1] as f32, self.resolution[2] as f32) - 1.0;
                let g = g.min(max);
                let i = g.floor().min(max - 1.0).max(Vec3::ZERO);
                let f = g - i;
                let (x, y, z) = (i.x as u32, i.y as u32, i.z as u32);
                let mut v = [0.0f32; 8];
                for (k, value) in v.iter_mut().enumerate() {
                    let (dx, dy, dz) = ((k & 1) as u32, ((k >> 1) & 1) as u32, ((k >> 2) & 1) as u32);
                    *value = self.voxel((x + dx).min(max.x as u32), (y + dy).min(max.y as u32), (z + dz).min(max.z as u32));
                }
                let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                let x0 = lerp(lerp(v[0], v[1], f.x), lerp(v[2], v[3], f.x), f.y);
                let x1 = lerp(lerp(v[4], v[5], f.x), lerp(v[6], v[7], f.x), f.y);
                lerp(x0, x1, f.z)
            }
            SdfKind::Heightfield => {
                let gx = ((p.x - self.origin.x) / self.voxel_size).clamp(0.0, self.resolution[0] as f32 - 1.0);
                let gz = ((p.z - self.origin.z) / self.voxel_size).clamp(0.0, self.resolution[2] as f32 - 1.0);
                p.y - self.height(gx.round() as u32, gz.round() as u32)
            }
        }
    }

    /// Bakes a narrow-band volume SDF from triangle soup (positions only).
    pub fn bake_volume(triangles: &[[Vec3; 3]], resolution: u32) -> Option<Self> {
        if triangles.is_empty() {
            return None;
        }
        let aabb = Aabb::from_points(triangles.iter().flatten().copied());
        let extent = aabb.extent().max_element().max(1e-4);
        let resolution = resolution.clamp(8, 256);
        const MARGIN: f32 = 2.0;
        let voxel_size = extent / (resolution as f32 - 2.0 * MARGIN - 1.0);
        let band = voxel_size * 4.0;
        let dims = aabb.extent() / voxel_size + 2.0 * MARGIN + 1.0;
        let res = [dims.x.ceil() as u32, dims.y.ceil() as u32, dims.z.ceil() as u32].map(|r| r.clamp(4, 256));
        let origin = aabb.min - Vec3::splat(MARGIN * voxel_size);

        let topology = Topology::new(triangles);
        let count = (res[0] * res[1] * res[2]) as usize;
        let mut distance = vec![f32::INFINITY; count];
        let mut inside = vec![false; count];
        let mut nearest = vec![0u32; count];
        let index = |x: u32, y: u32, z: u32| (x + res[0] * (y + res[1] * z)) as usize;

        for (ti, tri) in triangles.iter().enumerate() {
            if ti % 1024 == 0 {
                crate::progress::tick();
            }
            let lo = ((Aabb::from_points(tri.iter().copied()).min - Vec3::splat(band) - origin) / voxel_size).floor().max(Vec3::ZERO);
            let hi = ((Aabb::from_points(tri.iter().copied()).max + Vec3::splat(band) - origin) / voxel_size).ceil();
            let hi = hi.min(Vec3::new(res[0] as f32 - 1.0, res[1] as f32 - 1.0, res[2] as f32 - 1.0));
            for z in lo.z as u32..=hi.z as u32 {
                for y in lo.y as u32..=hi.y as u32 {
                    for x in lo.x as u32..=hi.x as u32 {
                        let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;
                        let (closest, feature) = closest_point_on_triangle(p, tri);
                        let d = p.distance(closest);
                        let i = index(x, y, z);
                        if d < distance[i] {
                            distance[i] = d;
                            nearest[i] = ti as u32;
                            let n = topology.pseudonormal(ti, feature);
                            inside[i] = (p - closest).dot(n) < 0.0;
                        }
                    }
                }
            }
        }

        // A mesh with boundary edges may still enclose a volume: generated meshes often carry
        // hairline cracks (T-junctions where openings were cut, displacement seams) narrower than
        // a voxel, which the flood fill cannot leak through. Such meshes bake as solids whenever
        // the resulting sign field is consistent; only genuinely open surfaces (leaves, cloth, a
        // half dome) become a thin two-sided shell, which inflates them by half a voxel and, on
        // large meshes, blocks the light around them.
        let (solid, conflicts) = solid_field(&distance, &inside, res, voxel_size, band);
        let two_sided = topology.open && conflicts > MAX_SIGN_CONFLICTS;
        let shell = voxel_size * 0.5;
        let value: Vec<f32> = if two_sided {
            distance.iter().map(|&d| if d.is_finite() { (d - shell).min(band) } else { band }).collect()
        } else {
            // Parts thinner than a voxel (rods, rings, thin walls of coarse meshes) fall between
            // the voxel centres of a solid and vanish: where the nearest triangle is not
            // represented (no sign change in its cell), add the shell around it.
            let grid = Grid { origin, voxel_size, res };
            let represented: Vec<bool> = triangles.iter().map(|t| grid.cell_changes_sign(&solid, (t[0] + t[1] + t[2]) / 3.0)).collect();
            (0..count)
                .map(|i| {
                    let d = distance[i];
                    if d.is_finite() && !represented[nearest[i] as usize] { solid[i].min(d - shell) } else { solid[i] }
                })
                .collect()
        };

        let mut words = vec![0u32; count.div_ceil(4)];
        for (i, v) in value.iter().enumerate() {
            let q = (v / band).clamp(-1.0, 1.0) * 127.0;
            words[i / 4] |= ((q.round() as i8) as u8 as u32) << ((i % 4) * 8);
        }
        Some(Self {
            kind: SdfKind::Volume,
            origin,
            voxel_size,
            resolution: res,
            band,
            height_range: [0.0; 2],
            two_sided,
            words,
        })
    }

    /// Bakes a heightfield (maximum surface height per XZ cell).
    pub fn bake_heightfield(triangles: &[[Vec3; 3]], resolution: u32) -> Option<Self> {
        if triangles.is_empty() {
            return None;
        }
        let aabb = Aabb::from_points(triangles.iter().flatten().copied());
        let extent = aabb.extent();
        let longest = extent.x.max(extent.z).max(1e-4);
        let cell = longest / (resolution.clamp(16, 4096) as f32 - 1.0);
        let res = [((extent.x / cell).ceil() as u32 + 1).max(2), 1, ((extent.z / cell).ceil() as u32 + 1).max(2)];
        let origin = aabb.min;
        let mut height = vec![f32::NEG_INFINITY; (res[0] * res[2]) as usize];
        for tri in triangles {
            let lo = ((Aabb::from_points(tri.iter().copied()).min - origin) / cell).floor().max(Vec3::ZERO);
            let hi = ((Aabb::from_points(tri.iter().copied()).max - origin) / cell).ceil();
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            let det = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
            if det.abs() < 1e-12 {
                continue;
            }
            for z in lo.z as u32..=(hi.z as u32).min(res[2] - 1) {
                for x in lo.x as u32..=(hi.x as u32).min(res[0] - 1) {
                    let p = origin + Vec3::new(x as f32, 0.0, z as f32) * cell;
                    let l1 = ((p.x - a.x) * (c.z - a.z) - (c.x - a.x) * (p.z - a.z)) / det;
                    let l2 = ((b.x - a.x) * (p.z - a.z) - (p.x - a.x) * (b.z - a.z)) / det;
                    let l0 = 1.0 - l1 - l2;
                    let eps = -1e-4;
                    if l0 >= eps && l1 >= eps && l2 >= eps {
                        let y = a.y * l0 + b.y * l1 + c.y * l2;
                        let h = &mut height[(x + res[0] * z) as usize];
                        *h = h.max(y);
                    }
                }
            }
        }
        let (min_y, max_y) = (aabb.min.y, aabb.max.y.max(aabb.min.y + 1e-3));
        let mut words = vec![0u32; height.len().div_ceil(2)];
        for (i, h) in height.iter().enumerate() {
            let h = if h.is_finite() { *h } else { min_y };
            let q = (((h - min_y) / (max_y - min_y)).clamp(0.0, 1.0) * 65535.0).round() as u32;
            words[i / 2] |= q << ((i % 2) * 16);
        }
        Some(Self {
            kind: SdfKind::Heightfield,
            origin,
            voxel_size: cell,
            resolution: res,
            band: cell * 4.0,
            height_range: [min_y, max_y],
            two_sided: false,
            words,
        })
    }
}

/// Which feature of a triangle the closest point lies on.
#[derive(Clone, Copy, Debug)]
enum Feature {
    Face,
    Edge(usize),
    Vertex(usize),
}

/// Ericson, "Real-Time Collision Detection", 5.1.5 — with feature classification.
fn closest_point_on_triangle(p: Vec3, t: &[Vec3; 3]) -> (Vec3, Feature) {
    let (a, b, c) = (t[0], t[1], t[2]);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Feature::Edge(0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Feature::Edge(1));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face)
}

/// Largest fraction of surface/flooded neighbour pairs whose signs may disagree for a mesh with
/// boundary edges to still bake as a solid. Cracked but closed meshes (the generated cathedral
/// walls, piers, tower) measure 0–2 %; open surfaces, whose outside wraps around to their
/// "inside" face, measure from about 8 % (an open ring of tubes) to tens of percent.
const MAX_SIGN_CONFLICTS: f32 = 0.05;

/// A voxel grid's placement, for inspecting baked values before quantisation.
struct Grid {
    origin: Vec3,
    voxel_size: f32,
    res: [u32; 3],
}

impl Grid {
    /// Whether per-voxel `values` change sign across the corners of the cell containing `p`
    /// (where a trilinearly interpolated field has its surface).
    fn cell_changes_sign(&self, values: &[f32], p: Vec3) -> bool {
        let max = Vec3::new(self.res[0] as f32 - 2.0, self.res[1] as f32 - 2.0, self.res[2] as f32 - 2.0);
        let base = ((p - self.origin) / self.voxel_size).floor().clamp(Vec3::ZERO, max);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for k in 0..8u32 {
            let (x, y, z) = (base.x as u32 + (k & 1), base.y as u32 + ((k >> 1) & 1), base.z as u32 + (k >> 2));
            let v = values[(x + self.res[0] * (y + self.res[1] * z)) as usize];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        lo <= 0.0 && hi >= 0.0
    }
}

/// Signed values of a solid. The sign comes from a flood fill from the grid boundary that
/// stops only at voxels within one voxel of the surface — any path crossing a closed surface
/// passes within half a voxel of it, and cracks narrower than a voxel stay sealed — so
/// pseudonormals (unreliable at T-junctions and cracks) decide only the voxels right at the
/// surface. Also returns the fraction of surface voxels next to flooded ones whose
/// pseudonormal sign disagrees (≈0 for closed meshes; 1 when nothing could be compared).
fn solid_field(distance: &[f32], inside: &[bool], res: [u32; 3], voxel_size: f32, band: f32) -> (Vec<f32>, f32) {
    let index = |x: u32, y: u32, z: u32| (x + res[0] * (y + res[1] * z)) as usize;
    let count = distance.len();
    let surface = |i: usize| distance[i] <= voxel_size;
    let neighbours = |x: u32, y: u32, z: u32| {
        [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)].into_iter().filter_map(move |(dx, dy, dz)| {
            let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
            let valid = nx >= 0 && ny >= 0 && nz >= 0 && nx < res[0] as i32 && ny < res[1] as i32 && nz < res[2] as i32;
            valid.then_some([nx as u32, ny as u32, nz as u32])
        })
    };
    // Flood fill from the boundary: everything reached is outside.
    let mut outside = vec![false; count];
    let mut stack: Vec<[u32; 3]> = Vec::new();
    for z in 0..res[2] {
        for y in 0..res[1] {
            for x in 0..res[0] {
                let boundary = x == 0 || y == 0 || z == 0 || x == res[0] - 1 || y == res[1] - 1 || z == res[2] - 1;
                let i = index(x, y, z);
                if boundary && (!surface(i) || !inside[i]) {
                    stack.push([x, y, z]);
                }
            }
        }
    }
    while let Some([x, y, z]) = stack.pop() {
        let i = index(x, y, z);
        if outside[i] {
            continue;
        }
        outside[i] = true;
        if surface(i) {
            continue; // Surface voxels are classified by their own sign; don't leak through.
        }
        for [nx, ny, nz] in neighbours(x, y, z) {
            if !outside[index(nx, ny, nz)] {
                stack.push([nx, ny, nz]);
            }
        }
    }
    // Consistency: a surface voxel's own sign agrees with the flooded voxels next to it.
    let (mut pairs, mut conflicts) = (0u32, 0u32);
    for z in 0..res[2] {
        for y in 0..res[1] {
            for x in 0..res[0] {
                let i = index(x, y, z);
                if surface(i) {
                    continue;
                }
                for [nx, ny, nz] in neighbours(x, y, z) {
                    let j = index(nx, ny, nz);
                    if surface(j) {
                        pairs += 1;
                        conflicts += (inside[j] == outside[i]) as u32;
                    }
                }
            }
        }
    }
    let value = (0..count)
        .map(|i| {
            let d = distance[i].min(band);
            let is_inside = if surface(i) { inside[i] } else { !outside[i] };
            if is_inside { -d } else { d }
        })
        .collect();
    // Without any voxel off the surface (a thin sheet), the sign cannot be verified.
    let ratio = if pairs == 0 { 1.0 } else { conflicts as f32 / pairs as f32 };
    (value, ratio)
}

/// Position-welded adjacency and pseudonormals of a triangle soup.
struct Topology {
    ids: Vec<[u32; 3]>,
    face_normals: Vec<Vec3>,
    vertex_normals: Vec<Vec3>,
    edge_normals: FxHashMap<u64, Vec3>,
    open: bool,
}

impl Topology {
    fn new(triangles: &[[Vec3; 3]]) -> Self {
        let mut index: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        let ids: Vec<[u32; 3]> = triangles
            .iter()
            .map(|t| {
                t.map(|p| {
                    let next = index.len() as u32;
                    *index.entry(p.to_array().map(f32::to_bits)).or_insert(next)
                })
            })
            .collect();
        let mut vertex_normals = vec![Vec3::ZERO; index.len()];
        let mut edge_normals: FxHashMap<u64, Vec3> = FxHashMap::default();
        let mut edge_uses: FxHashMap<u64, u32> = FxHashMap::default();
        let mut face_normals = Vec::with_capacity(triangles.len());
        for (t, id) in triangles.iter().zip(&ids) {
            let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            face_normals.push(n);
            for k in 0..3 {
                let e0 = (t[(k + 1) % 3] - t[k]).normalize_or_zero();
                let e1 = (t[(k + 2) % 3] - t[k]).normalize_or_zero();
                let angle = e0.dot(e1).clamp(-1.0, 1.0).acos();
                vertex_normals[id[k] as usize] += n * angle;
                let key = edge_key(id[k], id[(k + 1) % 3]);
                *edge_normals.entry(key).or_insert(Vec3::ZERO) += n;
                *edge_uses.entry(key).or_insert(0) += 1;
            }
        }
        let open = edge_uses.values().any(|&u| u != 2);
        Self { ids, face_normals, vertex_normals, edge_normals, open }
    }

    fn pseudonormal(&self, triangle: usize, feature: Feature) -> Vec3 {
        match feature {
            Feature::Face => self.face_normals[triangle],
            Feature::Vertex(k) => self.vertex_normals[self.ids[triangle][k] as usize],
            Feature::Edge(k) => {
                let id = self.ids[triangle];
                self.edge_normals.get(&edge_key(id[k], id[(k + 1) % 3])).copied().unwrap_or(self.face_normals[triangle])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn soup(mesh: &crate::Mesh) -> Vec<[Vec3; 3]> {
        mesh.triangles().map(|t| t.map(|i| mesh.vertices[i as usize].position())).collect()
    }

    #[test]
    fn sphere_distances_are_accurate() {
        let mesh = primitives::icosphere(4);
        let sdf = MeshSdf::bake_volume(&soup(&mesh), 32).unwrap();
        assert!(!sdf.two_sided);
        for p in [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.0, 0.95, 0.0), Vec3::new(1.05, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.1)] {
            let expected = (p.length() - 1.0).clamp(-sdf.band, sdf.band);
            let got = sdf.sample(p);
            assert!((got - expected).abs() < sdf.voxel_size * 0.6, "at {p}: {got} vs {expected}");
        }
        // Deep inside saturates to -band, far outside to +band.
        assert!(sdf.sample(Vec3::ZERO) < -0.9 * sdf.band);
    }

    #[test]
    fn heightfield_matches_terrain() {
        let mesh = primitives::terrain(100.0, 64, 20.0, 1);
        let sdf = MeshSdf::bake_heightfield(&soup(&mesh), 128).unwrap();
        let p = mesh.vertices[64 * 65 / 2].position();
        assert!(sdf.sample(p).abs() < 0.2, "{}", sdf.sample(p));
        assert!(sdf.sample(p + Vec3::Y * 3.0) > 2.5);
    }

    #[test]
    fn cracked_closed_mesh_bakes_solid() {
        // A sphere with a few triangles removed: hairline openings, still a solid.
        let mesh = primitives::icosphere(4);
        let mut triangles = soup(&mesh);
        triangles.truncate(triangles.len() - 3);
        let sdf = MeshSdf::bake_volume(&triangles, 32).unwrap();
        assert!(!sdf.two_sided);
        assert!(sdf.sample(Vec3::ZERO) < -0.9 * sdf.band);
    }

    #[test]
    fn thin_parts_keep_their_surface() {
        // A closed disc far thinner than a voxel: a pure solid field would lose it.
        let mut slab = primitives::icosphere(3);
        slab.deform(|p| p * Vec3::new(4.0, 0.02, 4.0));
        let sdf = MeshSdf::bake_volume(&soup(&slab), 32).unwrap();
        assert!(sdf.voxel_size > 0.04 * 4.0);
        assert!(sdf.sample(Vec3::ZERO) < 0.0, "{}", sdf.sample(Vec3::ZERO));
        assert!(sdf.sample(Vec3::new(0.0, 1.0, 0.0)) > 0.0);
    }

    #[test]
    fn open_surfaces_bake_two_sided() {
        // The upper half of a sphere (a dome) and a single quad: genuinely open.
        let mesh = primitives::icosphere(4);
        let dome: Vec<[Vec3; 3]> = soup(&mesh).into_iter().filter(|t| t.iter().all(|p| p.y >= -1e-3)).collect();
        let quad = vec![
            [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
            [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        ];
        for (name, triangles) in [("dome", dome), ("quad", quad)] {
            let sdf = MeshSdf::bake_volume(&triangles, 32).unwrap();
            assert!(sdf.two_sided, "{name}");
        }
    }
}
