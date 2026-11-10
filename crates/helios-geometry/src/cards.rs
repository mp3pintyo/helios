//! Mesh cards: orthographic captures of a mesh's surfaces for a card-based surface cache (UE5
//! Lumen's *mesh cards*).
//!
//! A card looks at the mesh from outside one face of its bounding box, along one of the six axis
//! directions, and records per texel the depth and normal of the surface it sees. Rays that hit
//! the mesh later find the hit point in the cards facing its normal and read the lighting stored
//! for that texel: lighting lives *on* the surface, so it cannot leak through a wall to its other
//! side the way a volume of lighting voxels does.
//!
//! Only surfaces facing the card's direction are captured, and a direction can hold several
//! **layers** (depth peeling): layer 0 is the first such surface along each texel's line of
//! sight, layer 1 the next one behind it, and so on. A courtyard ringed by one mesh (Sponza's
//! arcades) is captured from the outside by layer 0 and its inner walls by layer 1. Each layer
//! is cropped to the texels it covers.
//!
//! The capture is a CPU rasteriser over the mesh's triangles with square texels of one size per
//! mesh, so all of a mesh's cards share their texel density. On request it also records which
//! triangle each texel sees and where ([`TexelSource`]), so a renderer can interpolate the
//! attributes a material needs (texture coordinates, ...) and capture the surface's albedo.

use glam::{Vec2, Vec3};

use crate::bounds::Aabb;
use crate::encoding::{decode_octahedral, encode_octahedral};

/// Card directions: +X, -X, +Y, -Y, +Z, -Z (`axis = 2 · coordinate + (negative as usize)`).
pub const CARD_AXES: usize = 6;
/// Depth layers per direction at most.
pub const MAX_CARD_LAYERS: usize = 3;
/// A texel that sees no surface.
pub const INVALID_TEXEL: u32 = u32::MAX;
/// Largest stored depth value (16 bits; 0xffff marks invalid texels).
const DEPTH_MAX: f32 = 65534.0;
/// Surfaces are captured by the cards whose direction is within ~73° of their normal. Every
/// normal has a component of at least 1/√3 along some axis, so each surface has a card.
const MIN_FACING: f32 = 0.3;
/// Bits per component of the octahedral texel normals.
const NORMAL_BITS: u32 = 8;

/// How a mesh is captured.
#[derive(Clone, Copy, Debug)]
pub struct CardConfig {
    /// Texel size in mesh units ...
    pub texel_size: f32,
    /// ... unless a card side would get more texels than this; then texels grow to fit.
    pub max_resolution: u32,
    /// A later layer is kept only if it covers at least this many texels.
    pub min_layer_texels: u32,
    /// Capture back faces too, into the opposite cards (open meshes rendered from both sides).
    pub two_sided: bool,
    /// Record each texel's [`TexelSource`] ([`Card::sources`]).
    pub sources: bool,
}

impl Default for CardConfig {
    fn default() -> Self {
        Self { texel_size: 0.1, max_resolution: 256, min_layer_texels: 16, two_sided: false, sources: false }
    }
}

/// A captured vertex: position and normal in mesh space.
pub type CardVertex = (Vec3, Vec3);

/// The surface point a texel sees: the triangle (its index in the captured slice) and the
/// barycentric weights of its corners 1 and 2 (corner 0's is the rest) at the texel's centre.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TexelSource {
    pub triangle: u32,
    pub barycentrics: [f32; 2],
}

/// One card: an orthographic depth + normal image of the surfaces facing `axis`.
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    /// Direction the captured surfaces face (see [`CARD_AXES`]).
    pub axis: u32,
    pub layer: u32,
    /// Mesh-space (u, v) coordinates of texel (0, 0)'s corner; `u`, `v` are the coordinates
    /// `(c + 1) % 3` and `(c + 2) % 3` after the axis coordinate `c` ([`Card::coordinates`]).
    pub origin: Vec2,
    /// Texels along u and v.
    pub resolution: [u32; 2],
    /// Position of the card plane along the axis coordinate: the bounding-box face the card
    /// looks in from. Depths are measured from it into the mesh.
    pub plane: f32,
    /// Depth range of the texels: stored depth `k` (16 bits) is `min + k / 65534 · (max - min)`.
    pub depth_range: [f32; 2],
    /// Row-major texels: depth (16 bits) | octahedral normal (2 × 8 bits) << 16, or
    /// [`INVALID_TEXEL`].
    pub texels: Vec<u32>,
    /// Per texel, the triangle point it sees (default where it sees none); empty unless
    /// [`CardConfig::sources`] asked for them.
    pub sources: Vec<TexelSource>,
}

impl Card {
    /// (axis coordinate, u coordinate, v coordinate) indices of a card direction.
    pub fn coordinates(axis: u32) -> (usize, usize, usize) {
        let c = (axis / 2) as usize;
        (c, (c + 1) % 3, (c + 2) % 3)
    }

    /// +1 or -1: the sign of the direction along its coordinate.
    pub fn sign(axis: u32) -> f32 {
        if axis.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }

    /// Unit vector of the direction the captured surfaces face.
    pub fn direction(axis: u32) -> Vec3 {
        let mut d = Vec3::ZERO;
        d[Self::coordinates(axis).0] = Self::sign(axis);
        d
    }

    pub fn valid_texels(&self) -> usize {
        self.texels.iter().filter(|&&t| t != INVALID_TEXEL).count()
    }

    /// Mesh-space position and normal of the surface texel `(x, y)` sees, if any.
    pub fn texel_surface(&self, x: u32, y: u32, texel_size: f32) -> Option<(Vec3, Vec3)> {
        let t = self.texels[(x + y * self.resolution[0]) as usize];
        if t == INVALID_TEXEL {
            return None;
        }
        let [lo, hi] = self.depth_range;
        let depth = lo + (t & 0xffff) as f32 / DEPTH_MAX * (hi - lo);
        let (c, u, v) = Self::coordinates(self.axis);
        let mut p = Vec3::ZERO;
        p[u] = self.origin.x + (x as f32 + 0.5) * texel_size;
        p[v] = self.origin.y + (y as f32 + 0.5) * texel_size;
        p[c] = self.plane - Self::sign(self.axis) * depth;
        Some((p, decode_octahedral(t >> 16, NORMAL_BITS)))
    }
}

/// The cards of one mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshCards {
    /// Texel size in mesh units, shared by all cards.
    pub texel_size: f32,
    /// Sorted by (axis, layer); layers of a direction are consecutive from 0.
    pub cards: Vec<Card>,
}

impl MeshCards {
    /// Captures the cards of a triangle soup with mesh-space `bounds`.
    pub fn capture(triangles: &[[CardVertex; 3]], bounds: Aabb, config: &CardConfig) -> Self {
        let extent = bounds.extent().max(Vec3::splat(1e-4));
        let texel_size = config.texel_size.max(extent.max_element() / config.max_resolution.max(1) as f32);
        let mut cards = Vec::new();
        for axis in 0..CARD_AXES as u32 {
            let mut capture = AxisCapture::new(axis, bounds, texel_size);
            for (index, triangle) in triangles.iter().enumerate() {
                capture.add(triangle, index as u32, false);
                if config.two_sided {
                    capture.add(triangle, index as u32, true);
                }
            }
            for layer in 0..MAX_CARD_LAYERS as u32 {
                let Some(card) = capture.peel(layer, config.min_layer_texels.max(1), config.sources) else { break };
                cards.push(card);
            }
        }
        Self { texel_size, cards }
    }

    /// Number of layers captured for direction `axis`.
    pub fn layer_count(&self, axis: u32) -> usize {
        self.cards.iter().filter(|c| c.axis == axis).count()
    }

    pub fn texel_count(&self) -> usize {
        self.cards.iter().map(|c| c.texels.len()).sum()
    }
}

/// A triangle projected onto the card grid: texel-space (x, y), depth and normal per vertex.
struct Projected {
    /// Index of the source triangle.
    index: u32,
    xy: [Vec2; 3],
    depth: [f32; 3],
    normal: [Vec3; 3],
}

/// The capture of one direction: the projected facing triangles and the per-texel depth of the
/// previous layer.
struct AxisCapture {
    axis: u32,
    origin: Vec2,
    plane: f32,
    texel_size: f32,
    resolution: [u32; 2],
    triangles: Vec<Projected>,
    /// Depth of the last peeled layer per texel (f32::INFINITY where it saw nothing).
    previous: Vec<f32>,
}

impl AxisCapture {
    fn new(axis: u32, bounds: Aabb, texel_size: f32) -> Self {
        let (c, u, v) = Card::coordinates(axis);
        let extent = bounds.extent();
        let resolution = [u, v].map(|k| ((extent[k] / texel_size).ceil() as u32).max(1));
        let plane = if Card::sign(axis) > 0.0 { bounds.max[c] } else { bounds.min[c] };
        Self {
            axis,
            origin: Vec2::new(bounds.min[u], bounds.min[v]),
            plane,
            texel_size,
            resolution,
            triangles: Vec::new(),
            previous: vec![f32::NEG_INFINITY; (resolution[0] * resolution[1]) as usize],
        }
    }

    /// Projects a triangle if it faces the card (its back face, flipped, if `back`).
    fn add(&mut self, triangle: &[CardVertex; 3], index: u32, back: bool) {
        let [a, b, c] = triangle.map(|v| v.0);
        let face = (b - a).cross(c - a);
        let flip = if back { -1.0 } else { 1.0 };
        let direction = Card::direction(self.axis);
        if face.dot(direction) * flip <= MIN_FACING * face.length() {
            return;
        }
        let (k, u, v) = Card::coordinates(self.axis);
        let sign = Card::sign(self.axis);
        self.triangles.push(Projected {
            index,
            xy: triangle.map(|(p, _)| (Vec2::new(p[u], p[v]) - self.origin) / self.texel_size),
            depth: triangle.map(|(p, _)| sign * (self.plane - p[k])),
            normal: triangle.map(|(_, n)| n * flip),
        });
    }

    /// The next layer: per texel, the nearest facing surface beyond the previous layer (by more
    /// than a texel, so the same sheet is not captured twice). `None` once it covers fewer than
    /// `min_texels` texels (layer 0 is kept if it covers any). With `sources`, records the
    /// triangle point each texel sees.
    fn peel(&mut self, layer: u32, min_texels: u32, sources: bool) -> Option<Card> {
        let [w, h] = self.resolution;
        let mut depth = vec![f32::INFINITY; (w * h) as usize];
        let mut normal = vec![Vec3::ZERO; (w * h) as usize];
        let mut source = vec![TexelSource::default(); if sources { (w * h) as usize } else { 0 }];
        for t in &self.triangles {
            let [p0, p1, p2] = t.xy;
            let area = (p1 - p0).perp_dot(p2 - p0);
            if area.abs() < 1e-12 {
                continue;
            }
            let lo = p0.min(p1).min(p2).floor().max(Vec2::ZERO);
            let hi = p0.max(p1).max(p2).ceil().min(Vec2::new(w as f32, h as f32));
            for y in lo.y as u32..hi.y as u32 {
                for x in lo.x as u32..hi.x as u32 {
                    let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    // Barycentric coordinates; inclusive edges (a texel on a shared edge may be
                    // claimed by both triangles, which agree on its depth).
                    let b0 = (p1 - p).perp_dot(p2 - p) / area;
                    let b1 = (p2 - p).perp_dot(p0 - p) / area;
                    let b2 = 1.0 - b0 - b1;
                    if b0 < -1e-5 || b1 < -1e-5 || b2 < -1e-5 {
                        continue;
                    }
                    let i = (x + y * w) as usize;
                    let d = b0 * t.depth[0] + b1 * t.depth[1] + b2 * t.depth[2];
                    if d < depth[i] && d > self.previous[i] + self.texel_size {
                        depth[i] = d;
                        normal[i] = b0 * t.normal[0] + b1 * t.normal[1] + b2 * t.normal[2];
                        if sources {
                            source[i] = TexelSource { triangle: t.index, barycentrics: [b1, b2] };
                        }
                    }
                }
            }
        }

        // Crop to the covered texels.
        let covered: Vec<usize> = (0..depth.len()).filter(|&i| depth[i].is_finite()).collect();
        if covered.is_empty() || (layer > 0 && covered.len() < min_texels as usize) {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        let (mut d0, mut d1) = (f32::INFINITY, f32::NEG_INFINITY);
        for &i in &covered {
            let (x, y) = (i as u32 % w, i as u32 / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            (d0, d1) = (d0.min(depth[i]), d1.max(depth[i]));
        }
        let resolution = [x1 - x0 + 1, y1 - y0 + 1];
        let direction = Card::direction(self.axis);
        let mut texels = vec![INVALID_TEXEL; (resolution[0] * resolution[1]) as usize];
        let mut card_sources = vec![TexelSource::default(); if sources { texels.len() } else { 0 }];
        for &i in &covered {
            let (x, y) = (i as u32 % w, i as u32 / w);
            let quantised = if d1 > d0 { ((depth[i] - d0) / (d1 - d0) * DEPTH_MAX).round() as u32 } else { 0 };
            // Interpolated vertex normals can lean away from the card at creases; the card only
            // holds surfaces facing it.
            let mut n = normal[i].normalize_or(direction);
            if n.dot(direction) < 0.05 {
                n = (n + direction * (0.05 - n.dot(direction))).normalize();
            }
            let k = ((x - x0) + (y - y0) * resolution[0]) as usize;
            texels[k] = quantised.min(DEPTH_MAX as u32) | (encode_octahedral(n, NORMAL_BITS) << 16);
            if sources {
                card_sources[k] = source[i];
            }
        }
        // Texels without a surface in this layer have none in later ones either.
        self.previous = depth;
        Some(Card {
            axis: self.axis,
            layer,
            origin: self.origin + Vec2::new(x0 as f32, y0 as f32) * self.texel_size,
            resolution,
            plane: self.plane,
            depth_range: [d0, d1],
            texels,
            sources: card_sources,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::primitives;

    fn soup(mesh: &Mesh) -> Vec<[CardVertex; 3]> {
        mesh.triangles().map(|t| t.map(|i| (mesh.vertices[i as usize].position(), mesh.vertices[i as usize].normal()))).collect()
    }

    /// An axis-aligned quad at height `y` facing `+Y` (or `-Y` if `down`).
    fn quad(lo: Vec2, hi: Vec2, y: f32, down: bool) -> Vec<[CardVertex; 3]> {
        let n = if down { -Vec3::Y } else { Vec3::Y };
        let p = |x: f32, z: f32| (Vec3::new(x, y, z), n);
        let (a, b, c, d) = (p(lo.x, lo.y), p(hi.x, lo.y), p(hi.x, hi.y), p(lo.x, hi.y));
        // Counter-clockwise seen from +Y: (x, z) -> cross((b - a), (c - a)) points to -Y for
        // a -> b -> c along +x then +z, so +Y faces use a, c, b.
        if down {
            vec![[a, b, c], [a, c, d]]
        } else {
            vec![[a, c, b], [a, d, c]]
        }
    }

    /// Triangles with their face normal at every corner (hard edges).
    fn flat_soup(mesh: &Mesh) -> Vec<[CardVertex; 3]> {
        mesh.triangles()
            .map(|t| {
                let p = t.map(|i| mesh.vertices[i as usize].position());
                let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
                p.map(|p| (p, n))
            })
            .collect()
    }

    fn cube() -> Mesh {
        Mesh::from_positions(
            &[
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, 1.0, -1.0),
                Vec3::new(-1.0, -1.0, 1.0),
                Vec3::new(1.0, -1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(-1.0, 1.0, 1.0),
            ],
            vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        )
    }

    #[test]
    fn cube_has_one_full_card_per_face() {
        let mesh = cube();
        let config = CardConfig { texel_size: 0.25, ..Default::default() };
        let cards = MeshCards::capture(&flat_soup(&mesh), mesh.aabb(), &config);
        assert_eq!(cards.cards.len(), 6);
        for card in &cards.cards {
            assert_eq!(card.resolution, [8, 8], "axis {}", card.axis);
            assert_eq!(card.valid_texels(), 64);
            for y in 0..8 {
                for x in 0..8 {
                    let (p, n) = card.texel_surface(x, y, cards.texel_size).unwrap();
                    let d = Card::direction(card.axis);
                    assert!((p.dot(d) - 1.0).abs() < 1e-3, "on the face: {p}");
                    assert!(n.dot(d) > 0.99, "normal {n} for axis {}", card.axis);
                }
            }
        }
    }

    #[test]
    fn stacked_floors_are_peeled_into_layers() {
        // Two slabs facing +Y (tops at 0 and 2) whose bottoms face -Y; the upper one is smaller.
        let mut triangles = quad(Vec2::splat(-2.0), Vec2::splat(2.0), 0.0, false);
        triangles.extend(quad(Vec2::splat(-2.0), Vec2::splat(2.0), -0.5, true));
        triangles.extend(quad(Vec2::splat(-1.0), Vec2::splat(1.0), 2.0, false));
        triangles.extend(quad(Vec2::splat(-1.0), Vec2::splat(1.0), 1.5, true));
        let bounds = Aabb { min: Vec3::new(-2.0, -0.5, -2.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let cards = MeshCards::capture(&triangles, bounds, &CardConfig { texel_size: 0.25, ..Default::default() });
        assert_eq!(cards.layer_count(2), 2, "+Y: the upper slab, then the floor under it");
        assert_eq!(cards.layer_count(3), 2, "-Y: the lower slab's bottom, then the upper one's");
        assert_eq!(cards.layer_count(0), 0, "no surface faces +X");
        let top = &cards.cards.iter().find(|c| c.axis == 2 && c.layer == 0).unwrap();
        assert_eq!(top.resolution, [16, 16]);
        let under = &cards.cards.iter().find(|c| c.axis == 2 && c.layer == 1).unwrap();
        // Layer 1 only exists below the upper slab: cropped to its 2x2 m footprint.
        assert_eq!(under.resolution, [8, 8]);
        assert_eq!(under.valid_texels(), 64);
        let (p, n) = under.texel_surface(3, 4, cards.texel_size).unwrap();
        assert!(p.y.abs() < 1e-3 && n.y > 0.99, "{p} {n}");
        // Layer 0 sees the upper slab in the middle and the floor around it.
        let (inner, _) = top.texel_surface(8, 8, cards.texel_size).unwrap();
        let (outer, _) = top.texel_surface(1, 1, cards.texel_size).unwrap();
        assert!((inner.y - 2.0).abs() < 1e-3 && outer.y.abs() < 1e-3, "{inner} {outer}");
    }

    #[test]
    fn two_sided_sheets_are_captured_from_both_sides() {
        let triangles = quad(Vec2::splat(-1.0), Vec2::splat(1.0), 0.0, false);
        let bounds = Aabb { min: Vec3::new(-1.0, 0.0, -1.0), max: Vec3::new(1.0, 0.0, 1.0) };
        let one = MeshCards::capture(&triangles, bounds, &CardConfig { texel_size: 0.5, ..Default::default() });
        assert_eq!((one.layer_count(2), one.layer_count(3)), (1, 0));
        let both = MeshCards::capture(&triangles, bounds, &CardConfig { texel_size: 0.5, two_sided: true, ..Default::default() });
        assert_eq!((both.layer_count(2), both.layer_count(3)), (1, 1));
        let (_, n) = both.cards.iter().find(|c| c.axis == 3).unwrap().texel_surface(1, 1, 0.5).unwrap();
        assert!(n.y < -0.99);
    }

    #[test]
    fn sphere_texels_lie_on_the_surface() {
        let mesh = primitives::icosphere(4);
        let cards = MeshCards::capture(&soup(&mesh), mesh.aabb(), &CardConfig { texel_size: 0.05, ..Default::default() });
        assert_eq!(cards.cards.len(), 6, "a convex mesh needs one layer per direction");
        for card in &cards.cards {
            let d = Card::direction(card.axis);
            for y in 0..card.resolution[1] {
                for x in 0..card.resolution[0] {
                    if let Some((p, n)) = card.texel_surface(x, y, cards.texel_size) {
                        assert!((p.length() - 1.0).abs() < 0.02, "{p}");
                        assert!(n.dot(p.normalize()) > 0.95 && n.dot(d) > 0.0, "{n} at {p}");
                    }
                }
            }
        }
    }

    #[test]
    fn sources_locate_the_texel_on_its_triangle() {
        // Two stacked slabs: every texel's source interpolates its triangle's corners to the
        // point the texel sees (layer 1 under the upper slab included).
        let mut triangles = quad(Vec2::splat(-2.0), Vec2::splat(2.0), 0.0, false);
        triangles.extend(quad(Vec2::splat(-1.0), Vec2::splat(1.0), 2.0, false));
        let bounds = Aabb { min: Vec3::new(-2.0, 0.0, -2.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let config = CardConfig { texel_size: 0.25, sources: true, ..Default::default() };
        let cards = MeshCards::capture(&triangles, bounds, &config);
        assert_eq!(cards.layer_count(2), 2);
        for card in &cards.cards {
            assert_eq!(card.sources.len(), card.texels.len());
            for y in 0..card.resolution[1] {
                for x in 0..card.resolution[0] {
                    let Some((p, _)) = card.texel_surface(x, y, cards.texel_size) else { continue };
                    let s = card.sources[(x + y * card.resolution[0]) as usize];
                    let t = &triangles[s.triangle as usize];
                    let [b1, b2] = s.barycentrics;
                    let q = t[0].0 * (1.0 - b1 - b2) + t[1].0 * b1 + t[2].0 * b2;
                    assert!(q.distance(p) < 1e-3, "{q} vs {p}");
                }
            }
        }
        // Without the flag, nothing is recorded.
        let plain = MeshCards::capture(&triangles, bounds, &CardConfig { texel_size: 0.25, ..Default::default() });
        assert!(plain.cards.iter().all(|c| c.sources.is_empty()));
    }

    #[test]
    fn large_meshes_get_coarser_texels() {
        let mesh = cube().transformed(glam::Mat4::from_scale(Vec3::splat(50.0)));
        let config = CardConfig { texel_size: 0.1, max_resolution: 64, ..Default::default() };
        let cards = MeshCards::capture(&flat_soup(&mesh), mesh.aabb(), &config);
        assert!((cards.texel_size - 100.0 / 64.0).abs() < 1e-4);
        assert!(cards.cards.iter().all(|c| c.resolution == [64, 64]));
    }
}
