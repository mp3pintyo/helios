//! Procedural vegetation: branching trees, conifers, ferns, grass and bushes.
//!
//! Wood and foliage are returned as separate meshes (different materials). Leaves are real
//! geometry — small double-sided blades — in the spirit of UE's Nanite foliage: the cluster
//! DAG's component pruning thins and enlarges them with distance, preserving canopy density.

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::modeling::{tube, MeshBuilder};
use crate::Mesh;

/// Deterministic xorshift generator.
#[derive(Clone)]
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9) | 1)
    }

    pub fn float(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.float()
    }

    pub fn unit(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            let l = v.length_squared();
            if l > 1e-3 && l <= 1.0 {
                return v / l.sqrt();
            }
        }
    }
}

/// A tree split by material.
pub struct Plant {
    pub wood: Mesh,
    pub foliage: Mesh,
}

/// A plant whose foliage is an assembly (see [`crate::assembly`]): a few spray meshes (leaves
/// on a twig, needles on a branch) placed many times.
pub struct PlantAssembly {
    pub wood: Mesh,
    pub sprays: Vec<Mesh>,
    /// (spray, transform in plant space) of every copy.
    pub placements: Vec<(usize, Mat4)>,
}

/// Lengths of the spray variants, a geometric series from `shortest` by `ratio`: a copy uses
/// the variant nearest its length, scaled by at most `sqrt(ratio)`.
fn spray_lengths(shortest: f32, ratio: f32, count: usize) -> Vec<f32> {
    (0..count).map(|i| shortest * ratio.powi(i as i32)).collect()
}

/// The variant (index into `lengths`) nearest to `length` in scale, and the scale to it.
fn nearest_spray(lengths: &[f32], length: f32) -> (usize, f32) {
    let ratio = |l: f32| (length / l).ln().abs();
    let best = (0..lengths.len()).min_by(|&a, &b| ratio(lengths[a]).total_cmp(&ratio(lengths[b]))).unwrap_or(0);
    (best, length / lengths[best])
}

/// Leaf blade shape.
#[derive(Clone, Copy, Debug)]
pub struct LeafShape {
    pub length: f32,
    pub width: f32,
    /// Upward fold along the midrib (0 = flat).
    pub fold: f32,
}

/// Adds a double-sided leaf at `base` growing along `dir`, with `up` as its rough normal.
/// UV: v along the midrib 0..1, u across 0..1 (matches the `Leaf` texture pattern).
pub fn add_leaf(b: &mut MeshBuilder, base: Vec3, dir: Vec3, up: Vec3, shape: LeafShape) {
    let dir = dir.normalize_or(Vec3::Y);
    let side = dir.cross(up).normalize_or(dir.any_orthonormal_vector());
    let normal = side.cross(dir).normalize();
    let l = shape.length;
    let w = shape.width * 0.5;
    let fold = normal * shape.fold * w;
    let points = [
        (base, Vec2::new(0.5, 0.0)),
        (base + dir * l * 0.45 + side * w + fold, Vec2::new(1.0, 0.45)),
        (base + dir * l, Vec2::new(0.5, 1.0)),
        (base + dir * l * 0.45 - side * w + fold, Vec2::new(0.0, 0.45)),
        (base + dir * l * 0.5 - fold * 0.3, Vec2::new(0.5, 0.5)),
    ];
    for flip in [false, true] {
        let n = if flip { -normal } else { normal };
        let i: Vec<u32> = points.iter().map(|&(p, uv)| b.vertex(p, n, uv)).collect();
        let tris = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        for t in tris {
            if flip {
                b.triangle(i[t[0]], i[t[2]], i[t[1]]);
            } else {
                b.triangle(i[t[0]], i[t[1]], i[t[2]]);
            }
        }
    }
}

/// Broadleaf tree parameters.
#[derive(Clone, Copy, Debug)]
pub struct TreeParams {
    pub seed: u32,
    pub height: f32,
    pub trunk_radius: f32,
    /// Branching depth below the trunk.
    pub depth: u32,
    /// Children per branch.
    pub children: u32,
    /// Leaves per terminal twig.
    pub leaves_per_twig: u32,
    pub leaf: LeafShape,
    /// Sides of the trunk tube (branches use fewer).
    pub sides: usize,
    /// Upward tendency of branches (0 = straight, 1 = strongly upward).
    pub gravitropism: f32,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            seed: 1,
            height: 14.0,
            trunk_radius: 0.35,
            depth: 3,
            children: 4,
            leaves_per_twig: 14,
            leaf: LeafShape { length: 0.16, width: 0.09, fold: 0.3 },
            sides: 12,
            gravitropism: 0.35,
        }
    }
}

struct Grower<'a> {
    p: &'a TreeParams,
    rng: Rng,
    wood: Vec<Mesh>,
    leaves: MeshBuilder,
    /// The leaf-bearing outer part of every terminal twig: (start, end).
    twigs: Vec<(Vec3, Vec3)>,
}

impl Grower<'_> {
    fn branch(&mut self, start: Vec3, dir: Vec3, length: f32, radius: f32, depth: u32) {
        let segments = (6 + (length * 1.5) as usize).min(14);
        let mut points = vec![start];
        let mut radii = vec![radius];
        let mut d = dir.normalize();
        let mut p = start;
        let step = length / segments as f32;
        for i in 1..=segments {
            let wobble = self.rng.unit() * 0.18;
            d = (d + wobble + Vec3::Y * self.p.gravitropism * 0.08 * (depth as f32 + 1.0)).normalize();
            p += d * step;
            points.push(p);
            radii.push(radius * (1.0 - 0.75 * i as f32 / segments as f32).max(0.08));
        }
        let sides = if depth == 0 { self.p.sides } else { (self.p.sides / (depth as usize + 1)).max(4) };
        if radius > 0.006 {
            self.wood.push(tube(&points, &radii, sides, true));
        }

        if depth < self.p.depth {
            let count = if depth == 0 { self.p.children + 2 } else { self.p.children };
            for k in 0..count {
                let t = if depth == 0 { self.rng.range(0.35, 0.95) } else { self.rng.range(0.25, 1.0) };
                let index = ((t * segments as f32) as usize).min(segments);
                let origin = points[index];
                let along = (points[index.min(segments - 1) + 1] - points[index.min(segments - 1)]).normalize_or(d);
                let around = Quat::from_axis_angle(along, k as f32 / count as f32 * std::f32::consts::TAU + self.rng.range(-0.4, 0.4));
                let perpendicular = along.any_orthonormal_vector();
                let angle = self.rng.range(0.55, 1.05);
                let child = (around * (Quat::from_axis_angle(perpendicular, angle) * along)).normalize();
                let child_length = length * self.rng.range(0.45, 0.65) * (1.15 - t * 0.4);
                let child_radius = radii[index] * self.rng.range(0.55, 0.75);
                self.branch(origin, child, child_length, child_radius, depth + 1);
            }
        }
        if depth + 1 >= self.p.depth {
            // Leaves along the outer half of terminal twigs.
            self.twigs.push((points[((0.35 * segments as f32) as usize).min(segments)], points[segments]));
            for _ in 0..self.p.leaves_per_twig {
                let t = self.rng.range(0.35, 1.0);
                let index = ((t * segments as f32) as usize).min(segments);
                let base = points[index] + self.rng.unit() * 0.03;
                let out = (self.rng.unit() + d * 0.6 + Vec3::Y * 0.25).normalize();
                let up = (Vec3::Y + self.rng.unit() * 0.5).normalize();
                let scale = self.rng.range(0.75, 1.25);
                let shape = LeafShape { length: self.p.leaf.length * scale, width: self.p.leaf.width * scale, fold: self.p.leaf.fold };
                add_leaf(&mut self.leaves, base, out, up, shape);
            }
        }
    }
}

/// A broadleaf tree (oak / beech like) rooted at the origin.
pub fn broadleaf_tree(params: &TreeParams) -> Plant {
    let g = grow_broadleaf(params);
    let mut wood = MeshBuilder::default();
    for m in &g.wood {
        wood.append(m);
    }
    Plant { wood: wood.build(), foliage: g.leaves.build() }
}

/// The same tree with its foliage as an assembly: every terminal twig carries a copy of one of
/// `2 × 5` leaf sprays (two seeds, five lengths), turned so its leaves face up like the
/// originals and scaled to the twig.
pub fn broadleaf_assembly(params: &TreeParams) -> PlantAssembly {
    let g = grow_broadleaf(params);
    let mut wood = MeshBuilder::default();
    for m in &g.wood {
        wood.append(m);
    }
    let lengths = spray_lengths(0.3, 1.6, 5);
    let sprays = (0..2u32)
        .flat_map(|seed| lengths.iter().map(move |&length| (seed, length)))
        .map(|(seed, length)| leaf_spray(params.seed * 31 + seed, params.leaves_per_twig, length, params.leaf))
        .collect();
    let mut rng = Rng::new(params.seed ^ 0x5eed);
    let placements = g
        .twigs
        .iter()
        .map(|&(start, end)| {
            let (variant, scale) = nearest_spray(&lengths, start.distance(end));
            // Spray space: the twig along +y, up (+z) towards the sky, a little rolled.
            let y = (end - start).normalize_or(Vec3::Y);
            let z = (Vec3::Y - y * y.y).normalize_or(y.any_orthonormal_vector());
            let frame = Quat::from_mat3(&glam::Mat3::from_cols(y.cross(z), y, z)) * Quat::from_rotation_y(rng.range(-0.5, 0.5));
            let seed = (rng.float() * 2.0) as usize % 2;
            (seed * lengths.len() + variant, Mat4::from_scale_rotation_translation(Vec3::splat(scale), frame, start))
        })
        .collect();
    PlantAssembly { wood: wood.build(), sprays, placements }
}

/// Leaves along a twig of `length` on +y (as the trees place them along a twig's outer half),
/// facing +z.
fn leaf_spray(seed: u32, leaves: u32, length: f32, leaf: LeafShape) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    for _ in 0..leaves {
        let base = Vec3::Y * (rng.float() * length) + rng.unit() * 0.03;
        let out = (rng.unit() + Vec3::Y * 0.6 + Vec3::Z * 0.25).normalize();
        let up = (Vec3::Z + rng.unit() * 0.5).normalize();
        let scale = rng.range(0.75, 1.25);
        add_leaf(&mut b, base, out, up, LeafShape { length: leaf.length * scale, width: leaf.width * scale, fold: leaf.fold });
    }
    b.build()
}

fn grow_broadleaf(params: &TreeParams) -> Grower<'_> {
    let mut g = Grower { p: params, rng: Rng::new(params.seed), wood: Vec::new(), leaves: MeshBuilder::default(), twigs: Vec::new() };
    let lean = Vec3::new(g.rng.range(-0.08, 0.08), 1.0, g.rng.range(-0.08, 0.08));
    // Root flare under the trunk.
    let flare: Vec<Mesh> = (0..5)
        .map(|k| {
            let a = k as f32 / 5.0 * std::f32::consts::TAU + g.rng.range(-0.3, 0.3);
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let path = [out * 0.05 + Vec3::Y * 0.6, out * params.trunk_radius * 1.6 + Vec3::Y * 0.15, out * params.trunk_radius * 3.0 - Vec3::Y * 0.1];
            tube(&path, &[params.trunk_radius * 0.5, params.trunk_radius * 0.35, params.trunk_radius * 0.12], 6, true)
        })
        .collect();
    g.wood.extend(flare);
    g.branch(Vec3::new(0.0, -0.2, 0.0), lean, params.height * 0.62, params.trunk_radius, 0);
    g
}

/// A conifer (spruce / fir like): straight trunk with whorls of drooping branches carrying
/// needle sprays.
pub fn conifer(seed: u32, height: f32, radius: f32, needles_per_branch: u32, detail_sides: usize) -> Plant {
    let mut rng = Rng::new(seed);
    let mut needle_rng = Rng::new(seed ^ 0x6e65_6564);
    let mut wood = MeshBuilder::default();
    let mut foliage = MeshBuilder::default();
    let trunk: Vec<Vec3> = (0..=12).map(|i| Vec3::new(rng.range(-0.02, 0.02), height * i as f32 / 12.0, rng.range(-0.02, 0.02))).collect();
    let radii: Vec<f32> = (0..=12).map(|i| radius * (1.0 - i as f32 / 12.0 * 0.95)).collect();
    wood.append(&tube(&trunk, &radii, detail_sides, true));
    let whorls = (height * 1.6) as u32;
    // Needle sprays: flat blades carrying the needle texture.
    let needle = LeafShape { length: 0.42, width: 0.16, fold: 0.15 };
    for w in 0..whorls {
        let t = 0.18 + 0.8 * w as f32 / whorls as f32;
        let y = height * t;
        let span = (1.0 - t).powf(0.9) * height * 0.28 + 0.25;
        let count = 5 + (rng.float() * 3.0) as u32;
        for k in 0..count {
            let a = k as f32 / count as f32 * std::f32::consts::TAU + w as f32 * 0.7 + rng.range(-0.2, 0.2);
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let path: Vec<Vec3> = (0..=5)
                .map(|i| {
                    let s = i as f32 / 5.0;
                    out * span * s + Vec3::Y * (y + span * (0.15 * s - 0.35 * s * s))
                })
                .collect();
            let r0 = radius * 0.22 * (1.0 - t) + 0.01;
            wood.append(&tube(&path, &[r0, r0 * 0.8, r0 * 0.6, r0 * 0.45, r0 * 0.3, r0 * 0.15], 4, true));
            // Needle sprays along the branch, mostly sideways and drooping (their own random
            // stream: `conifer_assembly` grows the same branches).
            for _ in 0..needles_per_branch {
                let s = needle_rng.range(0.15, 1.0);
                let i = (s * 5.0) as usize;
                let base = path[i.min(5)] + (path[(i + 1).min(5)] - path[i.min(5)]) * (s * 5.0).fract();
                let side = out.cross(Vec3::Y);
                let dir = (side * needle_rng.range(-1.0, 1.0) + out * needle_rng.range(0.2, 0.8) - Vec3::Y * needle_rng.range(0.0, 0.6)).normalize();
                let s = needle_rng.range(0.7, 1.2) * (0.6 + 0.4 * (1.0 - t));
                add_leaf(&mut foliage, base, dir, Vec3::Y, LeafShape { length: needle.length * s, width: needle.width * s, fold: needle.fold });
            }
        }
    }
    // Leader at the top.
    for _ in 0..needles_per_branch {
        let base = Vec3::new(0.0, height * needle_rng.range(0.9, 1.0), 0.0);
        add_leaf(&mut foliage, base, (needle_rng.unit() + Vec3::Y).normalize(), Vec3::Y, needle);
    }
    Plant { wood: wood.build(), foliage: foliage.build() }
}

/// The same conifer with its needles as an assembly: each branch carries a copy of one of six
/// needle sprays shaped like a branch (span variants from the top to the bottom whorls), turned
/// to the branch's direction and scaled to its span; the leader's needles are one more part.
pub fn conifer_assembly(seed: u32, height: f32, radius: f32, needles_per_branch: u32, detail_sides: usize) -> PlantAssembly {
    let mut rng = Rng::new(seed);
    let mut wood = MeshBuilder::default();
    let trunk: Vec<Vec3> = (0..=12).map(|i| Vec3::new(rng.range(-0.02, 0.02), height * i as f32 / 12.0, rng.range(-0.02, 0.02))).collect();
    let radii: Vec<f32> = (0..=12).map(|i| radius * (1.0 - i as f32 / 12.0 * 0.95)).collect();
    wood.append(&tube(&trunk, &radii, detail_sides, true));
    let whorls = (height * 1.6) as u32;
    // Spans from the top whorl (t = 0.98) to the lowest (t = 0.18), as `conifer` grows them.
    let span_at = |t: f32| (1.0 - t).powf(0.9) * height * 0.28 + 0.25;
    let lengths = spray_lengths(span_at(0.98), (span_at(0.18) / span_at(0.98)).powf(1.0 / 5.0), 6);
    // Each variant's needles are sized for the height at which branches of its span grow.
    let height_at = |span: f32| 1.0 - (((span - 0.25).max(0.0) / (height * 0.28)).powf(1.0 / 0.9)).min(1.0);
    let mut sprays: Vec<Mesh> = lengths.iter().enumerate().map(|(i, &span)| needle_spray(seed * 17 + i as u32, needles_per_branch, span, height_at(span))).collect();
    let mut placements = Vec::new();
    for w in 0..whorls {
        let t = 0.18 + 0.8 * w as f32 / whorls as f32;
        let y = height * t;
        let span = span_at(t);
        let count = 5 + (rng.float() * 3.0) as u32;
        for k in 0..count {
            let a = k as f32 / count as f32 * std::f32::consts::TAU + w as f32 * 0.7 + rng.range(-0.2, 0.2);
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let path: Vec<Vec3> = (0..=5).map(|i| branch_point(out, span, i as f32 / 5.0) + Vec3::Y * y).collect();
            let r0 = radius * 0.22 * (1.0 - t) + 0.01;
            wood.append(&tube(&path, &[r0, r0 * 0.8, r0 * 0.6, r0 * 0.45, r0 * 0.3, r0 * 0.15], 4, true));
            let (variant, scale) = nearest_spray(&lengths, span);
            // Spray space has the branch along +x: turn it to `out` about the trunk.
            placements.push((variant, Mat4::from_scale_rotation_translation(Vec3::splat(scale), Quat::from_rotation_y(-a), Vec3::Y * y)));
        }
    }
    // The leader's needles at the top.
    let needle = LeafShape { length: 0.42, width: 0.16, fold: 0.15 };
    let mut leader = MeshBuilder::default();
    let mut needle_rng = Rng::new(seed ^ 0x6c65_6164);
    for _ in 0..needles_per_branch {
        let base = Vec3::new(0.0, height * needle_rng.range(0.9, 1.0), 0.0);
        add_leaf(&mut leader, base, (needle_rng.unit() + Vec3::Y).normalize(), Vec3::Y, needle);
    }
    sprays.push(leader.build());
    placements.push((sprays.len() - 1, Mat4::IDENTITY));
    PlantAssembly { wood: wood.build(), sprays, placements }
}

/// Point `s` (0..1) along a conifer branch of `span` growing towards `out` from the trunk: out
/// and a little up, then drooping.
fn branch_point(out: Vec3, span: f32, s: f32) -> Vec3 {
    out * span * s + Vec3::Y * (span * (0.15 * s - 0.35 * s * s))
}

/// The needle sprays of one conifer branch of `span` along +x, needles sized for relative height
/// `t` on the tree.
fn needle_spray(seed: u32, needles: u32, span: f32, t: f32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    let needle = LeafShape { length: 0.42, width: 0.16, fold: 0.15 };
    let path: Vec<Vec3> = (0..=5).map(|i| branch_point(Vec3::X, span, i as f32 / 5.0)).collect();
    for _ in 0..needles {
        let s = rng.range(0.15, 1.0);
        let i = (s * 5.0) as usize;
        let base = path[i.min(5)] + (path[(i + 1).min(5)] - path[i.min(5)]) * (s * 5.0).fract();
        let dir = (Vec3::Z * rng.range(-1.0, 1.0) + Vec3::X * rng.range(0.2, 0.8) - Vec3::Y * rng.range(0.0, 0.6)).normalize();
        let size = rng.range(0.7, 1.2) * (0.6 + 0.4 * (1.0 - t));
        add_leaf(&mut b, base, dir, Vec3::Y, LeafShape { length: needle.length * size, width: needle.width * size, fold: needle.fold });
    }
    b.build()
}

/// A fern: fronds arching out from the centre, each with paired leaflets.
pub fn fern(seed: u32, fronds: u32, length: f32, leaflets: u32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    for f in 0..fronds {
        let a = f as f32 / fronds as f32 * std::f32::consts::TAU + rng.range(-0.3, 0.3);
        let out = Vec3::new(a.cos(), 0.0, a.sin());
        let lift = rng.range(0.7, 1.2);
        let len = length * rng.range(0.75, 1.1);
        let point = |s: f32| out * len * s + Vec3::Y * len * (lift * s - 0.9 * s * s);
        for k in 1..leaflets {
            let s = k as f32 / leaflets as f32;
            let p = point(s);
            let tangent = (point(s + 0.01) - p).normalize();
            let side = tangent.cross(Vec3::Y).normalize_or(Vec3::X);
            let size = len * 0.22 * (1.0 - s * 0.8);
            for sign in [-1.0f32, 1.0] {
                let dir = (side * sign + tangent * 0.35 - Vec3::Y * 0.15).normalize();
                add_leaf(&mut b, p, dir, Vec3::Y, LeafShape { length: size, width: size * 0.32, fold: 0.2 });
            }
        }
        // Rachis as thin blades along the frond.
        let p0 = point(0.0);
        add_leaf(&mut b, p0, (point(0.5) - p0).normalize(), Vec3::Y, LeafShape { length: len * 0.5, width: 0.012, fold: 0.0 });
    }
    b.build()
}

/// A clump of grass blades.
pub fn grass_clump(seed: u32, blades: u32, height: f32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    for _ in 0..blades {
        let base = Vec3::new(rng.range(-0.15, 0.15), 0.0, rng.range(-0.15, 0.15));
        let h = height * rng.range(0.6, 1.2);
        let bend = Vec3::new(rng.range(-0.4, 0.4), 0.0, rng.range(-0.4, 0.4));
        let dir = (Vec3::Y + bend * 0.5).normalize();
        add_leaf(&mut b, base, dir, bend.normalize_or(Vec3::X).cross(Vec3::Y), LeafShape { length: h, width: 0.018, fold: 0.1 });
    }
    b.build()
}

/// A low bush: a few short branches carrying dense leaves.
pub fn bush(seed: u32, size: f32, leaves: u32) -> Plant {
    let params = TreeParams {
        seed,
        height: size * 1.6,
        trunk_radius: size * 0.04,
        depth: 2,
        children: 5,
        leaves_per_twig: leaves,
        leaf: LeafShape { length: 0.1, width: 0.06, fold: 0.3 },
        sides: 5,
        gravitropism: 0.6,
    };
    broadleaf_tree(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An assembly has the tree's wood, a copy of a spray per twig or branch, and about the
    /// foliage of the baked tree from far fewer distinct triangles.
    #[test]
    fn assemblies_place_sprays_on_every_twig() {
        let params = TreeParams { depth: 3, leaves_per_twig: 10, ..Default::default() };
        let (tree, assembly) = (broadleaf_tree(&params), broadleaf_assembly(&params));
        assert_eq!(assembly.wood.triangle_count(), tree.wood.triangle_count());
        let copies: usize = assembly.placements.iter().map(|&(s, _)| assembly.sprays[s].triangle_count()).sum();
        assert_eq!(copies, tree.foliage.triangle_count(), "one spray per twig, as many leaves");
        let distinct: usize = assembly.sprays.iter().map(Mesh::triangle_count).sum();
        assert!(distinct * 5 < copies, "{distinct} distinct of {copies} placed triangles");
        for &(_, transform) in &assembly.placements {
            let scale = crate::assembly::max_scale(&transform);
            assert!((0.75..=1.3).contains(&scale), "spray scale {scale}");
        }

        let conifer_plain = conifer(3, 12.0, 0.25, 6, 8);
        let conifer = conifer_assembly(3, 12.0, 0.25, 6, 8);
        let copies: usize = conifer.placements.iter().map(|&(s, _)| conifer.sprays[s].triangle_count()).sum();
        assert_eq!(copies, conifer_plain.foliage.triangle_count());
        assert_eq!(conifer.wood.triangle_count(), conifer_plain.wood.triangle_count());
    }

    #[test]
    fn trees_have_wood_and_leaves() {
        let tree = broadleaf_tree(&TreeParams { depth: 2, leaves_per_twig: 4, ..Default::default() });
        assert!(tree.wood.triangle_count() > 100);
        assert!(tree.foliage.triangle_count() > 100);
        let conifer = conifer(3, 12.0, 0.25, 6, 8);
        assert!(conifer.foliage.triangle_count() > 500);
    }
}
