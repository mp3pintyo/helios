//! Source mesh representation and a small builder for procedural modelling.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::hash::FxHashMap;
use crate::{Aabb, SkinWeights, Sphere};

/// Source vertex: two `vec4<f32>` — `(position, u)` and `(normal, v)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub u: f32,
    pub normal: [f32; 3],
    pub v: f32,
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self { position: position.into(), u: uv.x, normal: normal.into(), v: uv.y }
    }

    pub fn position(&self) -> Vec3 {
        Vec3::from(self.position)
    }

    pub fn normal(&self) -> Vec3 {
        Vec3::from(self.normal)
    }

    pub fn uv(&self) -> Vec2 {
        Vec2::new(self.u, self.v)
    }
}

/// An indexed triangle mesh.
///
/// Vertices that share a position but differ in normal or UV (hard edges, UV seams) are kept
/// as separate *wedges*; the simplifier understands them through [`Mesh::canonical_ids`].
/// Call [`Mesh::weld`] after procedural construction so identical vertices are merged and
/// simplification sees the true topology.
///
/// Imported meshes may carry optional attributes beyond [`Vertex`]: a second UV set, vertex
/// colours and skin weights. Each is either empty (absent: no memory, and the compressed
/// clusters of the mesh stay exactly as without the attribute) or holds one value per vertex.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Second texture coordinate set (glTF `TEXCOORD_1`, e.g. for ambient occlusion or
    /// lightmap-style unwraps): empty, or one per vertex.
    pub uv1: Vec<Vec2>,
    /// Linear RGBA vertex colours (glTF `COLOR_0`, multiplying the base colour): empty, or one
    /// per vertex.
    pub colors: Vec<Vec4>,
    /// Joint influences of a skeletal mesh (glTF `JOINTS_0` / `WEIGHTS_0`, quantised): empty, or
    /// one per vertex. Joints index the skin's joint list.
    pub skin: Vec<SkinWeights>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        assert!(indices.len().is_multiple_of(3), "index count must be a multiple of 3");
        Self { vertices, indices, uv1: Vec::new(), colors: Vec::new(), skin: Vec::new() }
    }

    /// A mesh with optional attributes; `uv1` and `colors` are empty or one per vertex.
    pub fn with_attributes(vertices: Vec<Vertex>, indices: Vec<u32>, uv1: Vec<Vec2>, colors: Vec<Vec4>) -> Self {
        for (name, len) in [("uv1", uv1.len()), ("colors", colors.len())] {
            assert!(len == 0 || len == vertices.len(), "{name}: {len} values for {} vertices", vertices.len());
        }
        Self { uv1, colors, ..Self::new(vertices, indices) }
    }

    /// Second UV of vertex `i` (its first UV set if the mesh has no second one).
    pub fn uv1_of(&self, i: usize) -> Vec2 {
        self.uv1.get(i).copied().unwrap_or_else(|| self.vertices[i].uv())
    }

    /// Colour of vertex `i` (white if the mesh has no vertex colours).
    pub fn color_of(&self, i: usize) -> Vec4 {
        self.colors.get(i).copied().unwrap_or(Vec4::ONE)
    }

    /// Skin weights of vertex `i` (rigidly bound to joint 0 if the mesh has none).
    pub fn skin_of(&self, i: usize) -> SkinWeights {
        self.skin.get(i).copied().unwrap_or_default()
    }

    /// The mesh with skin weights (`skin`: one per vertex).
    pub fn with_skin(self, skin: Vec<SkinWeights>) -> Self {
        assert_eq!(skin.len(), self.vertices.len(), "skin: one per vertex");
        Self { skin, ..self }
    }

    /// Builds a mesh from positions and indices, computing smooth normals.
    pub fn from_positions(positions: &[Vec3], indices: Vec<u32>) -> Self {
        let vertices = positions.iter().map(|&p| Vertex::new(p, Vec3::Y, Vec2::ZERO)).collect();
        let mut mesh = Self::new(vertices, indices);
        mesh.compute_normals();
        mesh
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn triangles(&self) -> impl Iterator<Item = [u32; 3]> + '_ {
        self.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]])
    }

    pub fn positions(&self) -> Vec<Vec3> {
        self.vertices.iter().map(Vertex::position).collect()
    }

    pub fn aabb(&self) -> Aabb {
        Aabb::from_points(self.vertices.iter().map(Vertex::position))
    }

    pub fn bounding_sphere(&self) -> Sphere {
        Sphere::from_points(&self.positions())
    }

    /// Area-weighted smooth vertex normals, accumulated per *position* so that UV seams do
    /// not show up as shading seams.
    pub fn compute_normals(&mut self) {
        let canonical = self.canonical_ids();
        let mut normals = vec![Vec3::ZERO; self.vertices.len()];
        for [a, b, c] in self.triangles() {
            let (pa, pb, pc) = (
                self.vertices[a as usize].position(),
                self.vertices[b as usize].position(),
                self.vertices[c as usize].position(),
            );
            let n = (pb - pa).cross(pc - pa);
            for i in [a, b, c] {
                normals[canonical[i as usize] as usize] += n;
            }
        }
        for (i, v) in self.vertices.iter_mut().enumerate() {
            v.normal = normals[canonical[i] as usize].normalize_or(Vec3::Y).into();
        }
    }

    /// Like [`Mesh::compute_normals`], but faces whose normals differ by more than
    /// `crease_degrees` from a wedge's own faces do not contribute, keeping hard edges hard.
    pub fn compute_normals_creased(&mut self, crease_degrees: f32) {
        let canonical = self.canonical_ids();
        let cos_crease = crease_degrees.to_radians().cos();
        let face_normals: Vec<Vec3> = self
            .triangles()
            .map(|[a, b, c]| {
                let (pa, pb, pc) =
                    (self.vertices[a as usize].position(), self.vertices[b as usize].position(), self.vertices[c as usize].position());
                (pb - pa).cross(pc - pa)
            })
            .collect();
        // Faces around every position class, and each wedge's own average face direction.
        let mut class_faces: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        let mut own = vec![Vec3::ZERO; self.vertices.len()];
        for (f, t) in self.triangles().enumerate() {
            for v in t {
                class_faces.entry(canonical[v as usize]).or_default().push(f as u32);
                own[v as usize] += face_normals[f].normalize_or_zero();
            }
        }
        for (i, v) in self.vertices.iter_mut().enumerate() {
            let reference = own[i].normalize_or(Vec3::Y);
            let mut n = Vec3::ZERO;
            if let Some(faces) = class_faces.get(&canonical[i]) {
                for &f in faces {
                    let fnorm = face_normals[f as usize];
                    if fnorm.normalize_or_zero().dot(reference) >= cos_crease {
                        n += fnorm;
                    }
                }
            }
            v.normal = n.normalize_or(reference).into();
        }
    }

    /// For every vertex, the index of the first vertex with a bit-identical position.
    pub fn canonical_ids(&self) -> Vec<u32> {
        canonical_ids(self.vertices.iter().map(|v| v.position))
    }

    /// Merges bit-identical vertices (position, normal, UVs, colour and skin weights) and drops
    /// degenerate triangles.
    pub fn weld(&mut self) {
        let mut map: FxHashMap<[u32; 17], u32> = FxHashMap::default();
        let mut remap = Vec::with_capacity(self.vertices.len());
        let mut vertices = Vec::new();
        let (mut uv1, mut colors, mut skin) = (Vec::new(), Vec::new(), Vec::new());
        for (i, v) in self.vertices.iter().enumerate() {
            let mut key = [0u32; 17];
            key[..8].copy_from_slice(&bytemuck::cast::<Vertex, [u32; 8]>(*v));
            if let Some(uv) = self.uv1.get(i) {
                key[8..10].copy_from_slice(&uv.to_array().map(f32::to_bits));
            }
            if let Some(color) = self.colors.get(i) {
                key[10..14].copy_from_slice(&color.to_array().map(f32::to_bits));
            }
            if let Some(s) = self.skin.get(i) {
                let j = s.joints.map(u32::from);
                key[14..].copy_from_slice(&[j[0] | (j[1] << 16), j[2] | (j[3] << 16), u32::from_le_bytes(s.weights)]);
            }
            let id = *map.entry(key).or_insert_with(|| {
                vertices.push(*v);
                uv1.extend(self.uv1.get(i));
                colors.extend(self.colors.get(i));
                skin.extend(self.skin.get(i));
                (vertices.len() - 1) as u32
            });
            remap.push(id);
        }
        self.uv1 = uv1;
        self.colors = colors;
        self.skin = skin;
        let positions: Vec<[u32; 3]> = vertices.iter().map(|v: &Vertex| v.position.map(f32::to_bits)).collect();
        let mut indices = Vec::with_capacity(self.indices.len());
        for [a, b, c] in self.triangles() {
            let t = [remap[a as usize], remap[b as usize], remap[c as usize]];
            let p = t.map(|i| positions[i as usize]);
            if p[0] != p[1] && p[1] != p[2] && p[0] != p[2] {
                indices.extend_from_slice(&t);
            }
        }
        self.vertices = vertices;
        self.indices = indices;
    }

    /// Welds by position only, discarding hard edges and seams (one wedge per position) and
    /// the optional attributes.
    pub fn weld_positions(&mut self) {
        for v in &mut self.vertices {
            v.normal = [0.0; 3];
            v.u = 0.0;
            v.v = 0.0;
        }
        self.uv1.clear();
        self.colors.clear();
        self.skin.clear();
        self.weld();
        self.compute_normals();
    }

    /// Appends `other`, transformed by `transform`. An optional attribute present in either
    /// mesh is kept, with the defaults of [`Mesh::uv1_of`] / [`Mesh::color_of`] /
    /// [`Mesh::skin_of`] where missing.
    pub fn append(&mut self, other: &Mesh, transform: Mat4) {
        let base = self.vertices.len() as u32;
        if !self.uv1.is_empty() || !other.uv1.is_empty() {
            self.uv1 = (0..self.vertices.len()).map(|i| self.uv1_of(i)).chain((0..other.vertices.len()).map(|i| other.uv1_of(i))).collect();
        }
        if !self.colors.is_empty() || !other.colors.is_empty() {
            self.colors = (0..self.vertices.len()).map(|i| self.color_of(i)).chain((0..other.vertices.len()).map(|i| other.color_of(i))).collect();
        }
        if !self.skin.is_empty() || !other.skin.is_empty() {
            self.skin = (0..self.vertices.len()).map(|i| self.skin_of(i)).chain((0..other.vertices.len()).map(|i| other.skin_of(i))).collect();
        }
        let normal_matrix = Mat3::from_mat4(transform).inverse().transpose();
        let flip = Mat3::from_mat4(transform).determinant() < 0.0;
        self.vertices.extend(other.vertices.iter().map(|v| {
            Vertex::new(
                transform.transform_point3(v.position()),
                (normal_matrix * v.normal()).normalize_or(Vec3::Y),
                v.uv(),
            )
        }));
        for t in other.indices.chunks_exact(3) {
            if flip {
                self.indices.extend_from_slice(&[t[0] + base, t[2] + base, t[1] + base]);
            } else {
                self.indices.extend_from_slice(&[t[0] + base, t[1] + base, t[2] + base]);
            }
        }
    }

    /// Applies `f` to every vertex position, then recomputes normals keeping creases > 50°.
    pub fn deform(&mut self, f: impl Fn(Vec3) -> Vec3) {
        for v in &mut self.vertices {
            v.position = f(v.position()).into();
        }
        self.compute_normals_creased(50.0);
    }

    /// Mesh transformed by `transform`.
    pub fn transformed(&self, transform: Mat4) -> Mesh {
        let mut out = Mesh::default();
        out.append(self, transform);
        out
    }
}

/// For every position, the index of the first occurrence of a bit-identical position.
pub fn canonical_ids(positions: impl Iterator<Item = [f32; 3]>) -> Vec<u32> {
    let mut first: FxHashMap<[u32; 3], u32> = FxHashMap::default();
    positions
        .enumerate()
        .map(|(i, p)| *first.entry(p.map(f32::to_bits)).or_insert(i as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::primitives;

    #[test]
    fn weld_keeps_seams_and_canonical_ids_join_them() {
        let mesh = primitives::cylinder(16, 4, 1.0, 2.0, false);
        let canonical = mesh.canonical_ids();
        let classes: crate::hash::FxHashSet<u32> = canonical.iter().copied().collect();
        assert!(classes.len() < mesh.vertices.len(), "seam wedges should share positions");
    }

    #[test]
    fn weld_and_append_keep_optional_attributes() {
        use glam::{Mat4, Vec2, Vec3, Vec4};
        let v = |x: f32| super::Vertex::new(Vec3::new(x, 0.0, 0.0), Vec3::Y, Vec2::ZERO);
        // Vertices 0 and 3 are identical; 1 and 4 differ only in their second UV (a seam).
        let vertices = vec![v(0.0), v(1.0), v(2.0), v(0.0), v(1.0)];
        let uv1 = vec![Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::ZERO, Vec2::ONE];
        let mut mesh = super::Mesh::with_attributes(vertices, vec![0, 1, 2, 3, 4, 2], uv1, Vec::new());
        mesh.weld();
        assert_eq!((mesh.vertices.len(), mesh.uv1.len()), (4, 4));
        assert_eq!(mesh.indices, [0, 1, 2, 0, 3, 2]);
        assert_eq!(mesh.uv1[3], Vec2::ONE);

        // Appending a mesh without the attribute fills in the defaults; colours appear with white.
        let plain = super::Mesh::new(vec![v(5.0), v(6.0), v(7.0)], vec![0, 1, 2]);
        let mut colored = plain.clone();
        colored.colors = vec![Vec4::new(1.0, 0.0, 0.0, 1.0); 3];
        mesh.append(&colored, Mat4::IDENTITY);
        assert_eq!((mesh.uv1.len(), mesh.colors.len()), (7, 7));
        assert_eq!((mesh.colors[0], mesh.colors[6]), (Vec4::ONE, Vec4::new(1.0, 0.0, 0.0, 1.0)));
        assert_eq!(mesh.uv1[5], plain.vertices[1].uv());
    }
}
