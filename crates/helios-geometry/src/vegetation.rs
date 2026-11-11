//! Procedural vegetation: branching trees, conifers, ferns, grass and bushes.
//!
//! Wood and foliage are returned as separate meshes (different materials). Leaves are real
//! geometry — small double-sided blades — in the spirit of UE's Nanite foliage: the cluster
//! DAG's component pruning thins and enlarges them with distance, preserving canopy density.

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::modeling::{tube, MeshBuilder};
use crate::plant_bones::{bind_all, BoneClass, PlantRig};
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

/// A tree split by material. Both meshes carry the plant's bones ([`Mesh::bones`]): the
/// branches first (all the wood's bones), then the leaves, so one plant's wood and foliage move
/// as one.
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
    /// The bone of `wood.bones` each copy hangs from (its sprays' leaves are bones without a
    /// parent: [`crate::plant_bones::part_bone_table`]).
    pub part_bones: Vec<u16>,
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
    leaf(b, base, dir, up, shape, None);
}

/// [`add_leaf`] with plant data ([`Mesh::plant`]): `plant.0` at the leaf's base, `plant.1` at its
/// tip, interpolated along the midrib.
pub fn add_leaf_with_plant(b: &mut MeshBuilder, base: Vec3, dir: Vec3, up: Vec3, shape: LeafShape, plant: (Vec2, Vec2)) {
    leaf(b, base, dir, up, shape, Some(plant));
}

fn leaf(b: &mut MeshBuilder, base: Vec3, dir: Vec3, up: Vec3, shape: LeafShape, plant: Option<(Vec2, Vec2)>) {
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
        let i: Vec<u32> = points
            .iter()
            .map(|&(p, uv)| match plant {
                Some((at_base, at_tip)) => b.plant_vertex(p, n, uv, at_base.lerp(at_tip, uv.y)),
                None => b.vertex(p, n, uv),
            })
            .collect();
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

/// Rotation per unit flutter (metres of the material's response) of a leaf of `length`: its tip
/// moves 2.5 × the response.
fn leaf_amplitude(length: f32) -> f32 {
    2.5 / length.max(0.02)
}

/// Flutter frequency fraction of a leaf of `length`: larger leaves flutter slower.
fn leaf_frequency(length: f32) -> f32 {
    (0.12 / length.max(0.02)).sqrt().clamp(0.6, 1.0)
}

/// A leaf's bone: from its base along its blade, hanging from `parent`.
fn leaf_bone(rig: &mut PlantRig, parent: Option<u16>, base: Vec3, dir: Vec3, length: f32, phase: f32) -> u16 {
    rig.add(parent, base, base + dir.normalize_or(Vec3::Y) * length, BoneClass::Flutter, leaf_amplitude(length), leaf_frequency(length), phase)
}

/// Mechanics of a branch for its wind response: base radius, length and the leaf area it holds
/// (its own and its sub-branches').
#[derive(Clone, Copy, Default)]
struct Limb {
    radius: f32,
    length: f32,
    leaf_area: f32,
}

/// Gives the [`BoneClass::Sway`] bones of a rig their amplitude and frequency from their
/// mechanics, as a cantilever loaded by the wind: stiffness `∝ r⁴/L³`, mass `∝ r²L` plus the
/// leaves', drag area `∝ 2rL` plus the leaves'. The static deflection angle (drag / (stiffness ·
/// L)) relative to the plant's median sets the amplitude (compressed, 0.5–2.5 rad per unit
/// sway), the natural frequency `√(k/m)` relative to the fastest branch the frequency (0.4–1):
/// thin twigs whip quickly, heavy leafy limbs swing slowly and far.
fn tune_branches(rig: &mut PlantRig, limbs: &[Limb]) {
    let sway: Vec<usize> = (0..rig.bones.len().min(limbs.len())).filter(|&i| rig.bones[i].class == BoneClass::Sway).collect();
    if sway.is_empty() {
        return;
    }
    let mechanics = |l: &Limb| {
        let (r, len) = (l.radius.max(0.003), l.length.max(0.05));
        let stiffness = r.powi(4) / len.powi(3);
        let mass = std::f32::consts::PI * r * r * len + 3e-4 * l.leaf_area;
        let area = 2.0 * r * len + l.leaf_area;
        (area / (stiffness * len), (stiffness / mass).sqrt())
    };
    let values: Vec<(f32, f32)> = sway.iter().map(|&i| mechanics(&limbs[i])).collect();
    let mut deflections: Vec<f32> = values.iter().map(|v| v.0).collect();
    deflections.sort_by(f32::total_cmp);
    let median = deflections[deflections.len() / 2].max(1e-12);
    let fastest = values.iter().map(|v| v.1).fold(0.0f32, f32::max).max(1e-12);
    for (&i, &(deflection, frequency)) in sway.iter().zip(&values) {
        rig.bones[i].amplitude = (deflection / median).powf(0.3).clamp(0.5, 2.5);
        rig.bones[i].frequency = (frequency / fastest).sqrt().clamp(0.4, 1.0);
    }
}

/// Rotation per unit bend of a stem of `length` that bends like the procedural wind's
/// cantilever (`bend · h²` at its tip): an arc of angle θ moves its tip by θL/2.
fn stem_amplitude(length: f32) -> f32 {
    2.0 * length
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
    /// Leaves as (base, direction, up, shape, branch bone), built into the foliage once every
    /// branch has its bone (leaf bones follow the branches').
    leaves: Vec<(Vec3, Vec3, Vec3, LeafShape, u16)>,
    /// The leaf-bearing outer part of every terminal twig: (start, end, bone).
    twigs: Vec<(Vec3, Vec3, u16)>,
    rig: PlantRig,
    limbs: Vec<Limb>,
}

/// Branching depth up to which branches get bones of their own; deeper twigs ride rigidly on
/// their ancestor's (a bone chain is at most trunk → two branch levels → leaf).
const BONE_DEPTH: u32 = 2;

impl Grower<'_> {
    fn branch(&mut self, start: Vec3, dir: Vec3, length: f32, radius: f32, depth: u32, parent: Option<u16>) {
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
        let bone = match parent {
            Some(parent) if depth > BONE_DEPTH => parent,
            _ => {
                let class = if depth == 0 { BoneClass::Bend } else { BoneClass::Sway };
                let chord = start.distance(points[segments]);
                let amplitude = if depth == 0 { stem_amplitude(chord) } else { 1.0 };
                // A deterministic phase (the rng's stream shapes the tree).
                let phase = (self.limbs.len() as f32 * 0.618_034).fract();
                let b = self.rig.add(parent, start, points[segments], class, amplitude, 1.0, phase);
                self.limbs.push(Limb { radius, length: chord, leaf_area: 0.0 });
                b
            }
        };
        let sides = if depth == 0 { self.p.sides } else { (self.p.sides / (depth as usize + 1)).max(4) };
        if radius > 0.006 {
            let mut tube = tube(&points, &radii, sides, true);
            bind_all(&mut tube, bone);
            self.wood.push(tube);
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
                self.branch(origin, child, child_length, child_radius, depth + 1, Some(bone));
            }
        }
        if depth + 1 >= self.p.depth {
            // Leaves along the outer half of terminal twigs.
            self.twigs.push((points[((0.35 * segments as f32) as usize).min(segments)], points[segments], bone));
            for _ in 0..self.p.leaves_per_twig {
                let t = self.rng.range(0.35, 1.0);
                let index = ((t * segments as f32) as usize).min(segments);
                let base = points[index] + self.rng.unit() * 0.03;
                let out = (self.rng.unit() + d * 0.6 + Vec3::Y * 0.25).normalize();
                let up = (Vec3::Y + self.rng.unit() * 0.5).normalize();
                let scale = self.rng.range(0.75, 1.25);
                let shape = LeafShape { length: self.p.leaf.length * scale, width: self.p.leaf.width * scale, fold: self.p.leaf.fold };
                self.leaves.push((base, out, up, shape, bone));
            }
        }
    }

    /// Sums every bone's leaf area up its chain and tunes the branches ([`tune_branches`]).
    fn tune(&mut self) {
        for &(_, _, _, shape, bone) in &self.leaves {
            let mut b = Some(bone);
            while let Some(i) = b {
                self.limbs[i as usize].leaf_area += 0.5 * shape.length * shape.width;
                b = self.rig.bones[i as usize].parent;
            }
        }
        tune_branches(&mut self.rig, &self.limbs);
    }

    /// The wood with the branches' bones (call before [`Grower::foliage`]).
    fn wood(&self) -> Mesh {
        let mut wood = MeshBuilder::default();
        for m in &self.wood {
            wood.append(m);
        }
        wood.skin = self.wood.iter().flat_map(|m| m.skin.iter().copied()).collect();
        let mut wood = wood.build();
        wood.bones = self.rig.bones.clone();
        wood
    }

    /// The foliage: one bone per leaf after the branches' (`rig` then holds every bone).
    fn foliage(&mut self) -> Mesh {
        let mut b = MeshBuilder::default();
        for (k, &(base, out, up, shape, branch)) in self.leaves.iter().enumerate() {
            let bone = leaf_bone(&mut self.rig, Some(branch), base, out, shape.length, (k as f32 * 0.618_034).fract());
            b.set_bone(bone);
            add_leaf(&mut b, base, out, up, shape);
        }
        let mut mesh = b.build();
        mesh.bones = self.rig.bones.clone();
        mesh
    }
}

/// Smooth step from 0 at `edge0` to 1 at `edge1`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Gives a woody plant rooted at the origin its wind weights ([`Mesh::plant`]): held where it
/// meets the ground (y = 0, root flares below it), free from `free_height` up. For trunks and
/// bushes the height above the ground *is* the distance along the plant from its root; the
/// wood and the leaves of one plant get the same function of height, so a leaf moves with the
/// twig it sits on. No occlusion is baked (1).
pub fn root_weights(mesh: &mut Mesh, free_height: f32) {
    mesh.plant = mesh.vertices.iter().map(|v| Vec2::new(smoothstep(0.0, free_height, v.position[1]), 1.0)).collect();
}

/// Height over which a tree's trunk stiffens into its free sway (wind weight 0 → 1).
const TREE_FREE_HEIGHT: f32 = 0.8;

/// A broadleaf tree (oak / beech like) rooted at the origin.
pub fn broadleaf_tree(params: &TreeParams) -> Plant {
    let mut g = grow_broadleaf(params);
    g.tune();
    let mut wood = g.wood();
    root_weights(&mut wood, TREE_FREE_HEIGHT.min(0.25 * params.height));
    let foliage = g.foliage();
    Plant { wood, foliage }
}

/// The same tree with its foliage as an assembly: every terminal twig carries a copy of one of
/// `2 × 5` leaf sprays (two seeds, five lengths), turned so its leaves face up like the
/// originals and scaled to the twig.
pub fn broadleaf_assembly(params: &TreeParams) -> PlantAssembly {
    let mut g = grow_broadleaf(params);
    g.tune();
    let mut wood = g.wood();
    let lengths = spray_lengths(0.3, 1.6, 5);
    let sprays = (0..2u32)
        .flat_map(|seed| lengths.iter().map(move |&length| (seed, length)))
        .map(|(seed, length)| leaf_spray(params.seed * 31 + seed, params.leaves_per_twig, length, params.leaf))
        .collect();
    let mut rng = Rng::new(params.seed ^ 0x5eed);
    let (placements, part_bones) = g
        .twigs
        .iter()
        .map(|&(start, end, bone)| {
            let (variant, scale) = nearest_spray(&lengths, start.distance(end));
            // Spray space: the twig along +y, up (+z) towards the sky, a little rolled.
            let y = (end - start).normalize_or(Vec3::Y);
            let z = (Vec3::Y - y * y.y).normalize_or(y.any_orthonormal_vector());
            let frame = Quat::from_mat3(&glam::Mat3::from_cols(y.cross(z), y, z)) * Quat::from_rotation_y(rng.range(-0.5, 0.5));
            let seed = (rng.float() * 2.0) as usize % 2;
            ((seed * lengths.len() + variant, Mat4::from_scale_rotation_translation(Vec3::splat(scale), frame, start)), bone)
        })
        .unzip();
    root_weights(&mut wood, TREE_FREE_HEIGHT.min(0.25 * params.height));
    PlantAssembly { wood, sprays, placements, part_bones }
}

/// Leaves along a twig of `length` on +y (as the trees place them along a twig's outer half),
/// facing +z.
fn leaf_spray(seed: u32, leaves: u32, length: f32, leaf: LeafShape) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    let mut rig = PlantRig::default();
    for k in 0..leaves {
        let base = Vec3::Y * (rng.float() * length) + rng.unit() * 0.03;
        let out = (rng.unit() + Vec3::Y * 0.6 + Vec3::Z * 0.25).normalize();
        let up = (Vec3::Z + rng.unit() * 0.5).normalize();
        let scale = rng.range(0.75, 1.25);
        let shape = LeafShape { length: leaf.length * scale, width: leaf.width * scale, fold: leaf.fold };
        b.set_bone(leaf_bone(&mut rig, None, base, out, shape.length, (k as f32 * 0.618_034).fract()));
        add_leaf(&mut b, base, out, up, shape);
    }
    let mut mesh = b.build();
    mesh.bones = rig.bones;
    mesh
}

fn grow_broadleaf(params: &TreeParams) -> Grower<'_> {
    let mut g = Grower { p: params, rng: Rng::new(params.seed), wood: Vec::new(), leaves: Vec::new(), twigs: Vec::new(), rig: PlantRig::default(), limbs: Vec::new() };
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
    // The trunk is bone 0; the root flare rides on it (held where it meets the ground).
    g.wood.extend(flare.into_iter().map(|mut m| {
        bind_all(&mut m, 0);
        m
    }));
    g.branch(Vec3::new(0.0, -0.2, 0.0), lean, params.height * 0.62, params.trunk_radius, 0, None);
    g
}

/// A conifer (spruce / fir like): straight trunk with whorls of drooping branches carrying
/// needle sprays.
pub fn conifer(seed: u32, height: f32, radius: f32, needles_per_branch: u32, detail_sides: usize) -> Plant {
    let mut rng = Rng::new(seed);
    let mut needle_rng = Rng::new(seed ^ 0x6e65_6564);
    let mut wood = MeshBuilder::default();
    let mut foliage = MeshBuilder::default();
    let mut rig = PlantRig::default();
    let mut limbs = Vec::new();
    let trunk: Vec<Vec3> = (0..=12).map(|i| Vec3::new(rng.range(-0.02, 0.02), height * i as f32 / 12.0, rng.range(-0.02, 0.02))).collect();
    let radii: Vec<f32> = (0..=12).map(|i| radius * (1.0 - i as f32 / 12.0 * 0.95)).collect();
    let trunk_bone = conifer_trunk(&mut rig, &mut limbs, height, radius);
    wood.set_bone(trunk_bone);
    wood.append(&tube(&trunk, &radii, detail_sides, true));
    let whorls = (height * 1.6) as u32;
    // Needle sprays: flat blades carrying the needle texture.
    let needle = LeafShape { length: 0.42, width: 0.16, fold: 0.15 };
    let mut needles = Vec::new();
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
            let branch = conifer_branch(&mut rig, &mut limbs, trunk_bone, &path, r0, w);
            wood.set_bone(branch);
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
                let shape = LeafShape { length: needle.length * s, width: needle.width * s, fold: needle.fold };
                limbs[branch as usize].leaf_area += 0.5 * shape.length * shape.width;
                needles.push((base, dir, shape, branch));
            }
        }
    }
    // Leader at the top.
    for _ in 0..needles_per_branch {
        let base = Vec3::new(0.0, height * needle_rng.range(0.9, 1.0), 0.0);
        needles.push((base, (needle_rng.unit() + Vec3::Y).normalize(), needle, trunk_bone));
    }
    tune_branches(&mut rig, &limbs);
    let mut wood = wood.build();
    wood.bones = rig.bones.clone();
    root_weights(&mut wood, TREE_FREE_HEIGHT);
    for (k, &(base, dir, shape, branch)) in needles.iter().enumerate() {
        foliage.set_bone(leaf_bone(&mut rig, Some(branch), base, dir, shape.length, (k as f32 * 0.618_034).fract()));
        add_leaf(&mut foliage, base, dir, Vec3::Y, shape);
    }
    let mut foliage = foliage.build();
    foliage.bones = rig.bones;
    Plant { wood, foliage }
}

/// A conifer's trunk bone (bone 0): a straight stem from the ground.
fn conifer_trunk(rig: &mut PlantRig, limbs: &mut Vec<Limb>, height: f32, radius: f32) -> u16 {
    limbs.push(Limb { radius, length: height, leaf_area: 0.0 });
    rig.add(None, Vec3::ZERO, Vec3::Y * height, BoneClass::Bend, stem_amplitude(height), 1.0, 0.0)
}

/// The bone of a conifer branch along `path` (from the trunk), base radius `r0`, whorl `w`.
fn conifer_branch(rig: &mut PlantRig, limbs: &mut Vec<Limb>, trunk: u16, path: &[Vec3], r0: f32, w: u32) -> u16 {
    let (start, end) = (path[0], path[path.len() - 1]);
    limbs.push(Limb { radius: r0, length: start.distance(end), leaf_area: 0.0 });
    rig.add(Some(trunk), start, end, BoneClass::Sway, 1.0, 1.0, (limbs.len() as f32 * 0.618_034 + w as f32 * 0.29).fract())
}

/// The same conifer with its needles as an assembly: each branch carries a copy of one of six
/// needle sprays shaped like a branch (span variants from the top to the bottom whorls), turned
/// to the branch's direction and scaled to its span; the leader's needles are one more part.
pub fn conifer_assembly(seed: u32, height: f32, radius: f32, needles_per_branch: u32, detail_sides: usize) -> PlantAssembly {
    let mut rng = Rng::new(seed);
    let mut wood = MeshBuilder::default();
    let mut rig = PlantRig::default();
    let mut limbs = Vec::new();
    let trunk: Vec<Vec3> = (0..=12).map(|i| Vec3::new(rng.range(-0.02, 0.02), height * i as f32 / 12.0, rng.range(-0.02, 0.02))).collect();
    let radii: Vec<f32> = (0..=12).map(|i| radius * (1.0 - i as f32 / 12.0 * 0.95)).collect();
    let trunk_bone = conifer_trunk(&mut rig, &mut limbs, height, radius);
    wood.set_bone(trunk_bone);
    wood.append(&tube(&trunk, &radii, detail_sides, true));
    let whorls = (height * 1.6) as u32;
    let mut part_bones = Vec::new();
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
            let branch = conifer_branch(&mut rig, &mut limbs, trunk_bone, &path, r0, w);
            wood.set_bone(branch);
            wood.append(&tube(&path, &[r0, r0 * 0.8, r0 * 0.6, r0 * 0.45, r0 * 0.3, r0 * 0.15], 4, true));
            let (variant, scale) = nearest_spray(&lengths, span);
            // Spray space has the branch along +x: turn it to `out` about the trunk.
            placements.push((variant, Mat4::from_scale_rotation_translation(Vec3::splat(scale), Quat::from_rotation_y(-a), Vec3::Y * y)));
            part_bones.push(branch);
            limbs[branch as usize].leaf_area += needles_per_branch as f32 * 0.5 * 0.42 * 0.16 * (0.6 + 0.4 * (1.0 - t)).powi(2);
        }
    }
    tune_branches(&mut rig, &limbs);
    // The leader's needles at the top.
    let needle = LeafShape { length: 0.42, width: 0.16, fold: 0.15 };
    let mut leader = MeshBuilder::default();
    let mut leader_rig = PlantRig::default();
    let mut needle_rng = Rng::new(seed ^ 0x6c65_6164);
    for k in 0..needles_per_branch {
        let base = Vec3::new(0.0, height * needle_rng.range(0.9, 1.0), 0.0);
        let dir = (needle_rng.unit() + Vec3::Y).normalize();
        leader.set_bone(leaf_bone(&mut leader_rig, None, base, dir, needle.length, (k as f32 * 0.618_034).fract()));
        add_leaf(&mut leader, base, dir, Vec3::Y, needle);
    }
    let mut leader = leader.build();
    leader.bones = leader_rig.bones;
    sprays.push(leader);
    placements.push((sprays.len() - 1, Mat4::IDENTITY));
    part_bones.push(trunk_bone);
    let mut wood = wood.build();
    wood.bones = rig.bones;
    root_weights(&mut wood, TREE_FREE_HEIGHT);
    PlantAssembly { wood, sprays, placements, part_bones }
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
    let mut rig = PlantRig::default();
    let needle = LeafShape { length: 0.42, width: 0.16, fold: 0.15 };
    let path: Vec<Vec3> = (0..=5).map(|i| branch_point(Vec3::X, span, i as f32 / 5.0)).collect();
    for k in 0..needles {
        let s = rng.range(0.15, 1.0);
        let i = (s * 5.0) as usize;
        let base = path[i.min(5)] + (path[(i + 1).min(5)] - path[i.min(5)]) * (s * 5.0).fract();
        let dir = (Vec3::Z * rng.range(-1.0, 1.0) + Vec3::X * rng.range(0.2, 0.8) - Vec3::Y * rng.range(0.0, 0.6)).normalize();
        let size = rng.range(0.7, 1.2) * (0.6 + 0.4 * (1.0 - t));
        let shape = LeafShape { length: needle.length * size, width: needle.width * size, fold: needle.fold };
        b.set_bone(leaf_bone(&mut rig, None, base, dir, shape.length, (k as f32 * 0.618_034).fract()));
        add_leaf(&mut b, base, dir, Vec3::Y, shape);
    }
    let mut mesh = b.build();
    mesh.bones = rig.bones;
    mesh
}

/// Depth below the ground (the plant's origin) at which grass blades and fern stalks start:
/// they grow out of the soil instead of standing on it, so a slope or a terrain a little above
/// the analytic ground never shows a floating base.
pub const ROOT_DEPTH: f32 = 0.04;

/// Adds a double-sided strip along `points` (at least two; the last is the tip) with the given
/// half widths across `side`, facing `normal` (its edges' normals turned outwards by `round`,
/// shading it like a rounded or folded blade). Every point carries its plant data. UV: v from 0
/// at the first point to 1 at the tip, u across 0..1.
fn add_strip(b: &mut MeshBuilder, points: &[(Vec3, f32, Vec2)], side: Vec3, normal: Vec3, round: f32) {
    let last = points.len() - 1;
    for flip in [false, true] {
        let n = if flip { -normal } else { normal };
        let mut rows: Vec<(u32, u32)> = Vec::new();
        for (k, &(p, half, plant)) in points.iter().enumerate() {
            let v = k as f32 / last as f32;
            if k == last || half <= 0.0 {
                let tip = b.plant_vertex(p, n, Vec2::new(0.5, v), plant);
                rows.push((tip, tip));
            } else {
                let left = b.plant_vertex(p - side * half, (n - side * round).normalize(), Vec2::new(0.0, v), plant);
                let right = b.plant_vertex(p + side * half, (n + side * round).normalize(), Vec2::new(1.0, v), plant);
                rows.push((left, right));
            }
        }
        for k in 0..last {
            let ((l0, r0), (l1, r1)) = (rows[k], rows[k + 1]);
            let mut tris = vec![[l0, r0, r1]];
            if l1 != r1 {
                tris.push([l0, r1, l1]);
            }
            for t in tris {
                if flip {
                    b.triangle(t[0], t[2], t[1]);
                } else {
                    b.triangle(t[0], t[1], t[2]);
                }
            }
        }
    }
}

/// A fern: fronds arching out of a crown, each a tapering stalk (rachis) with paired leaflets.
/// The stalks start below the ground; a frond's wind weight grows with the distance along it
/// (the crown is held, a tip resting near the ground still moves), and the crown, under the
/// fronds, is darker (baked occlusion).
pub fn fern(seed: u32, fronds: u32, length: f32, leaflets: u32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    let mut rig = PlantRig::default();
    // Wind weight and occlusion at fraction `s` along a frond.
    let plant = |s: f32| Vec2::new(smoothstep(0.03, 0.35, s), 0.35 + 0.65 * smoothstep(0.0, 0.5, s));
    for f in 0..fronds {
        let a = f as f32 / fronds as f32 * std::f32::consts::TAU + rng.range(-0.3, 0.3);
        let out = Vec3::new(a.cos(), 0.0, a.sin());
        let lift = rng.range(0.7, 1.2);
        let len = length * rng.range(0.75, 1.1);
        let point = |s: f32| out * len * s + Vec3::Y * len * (lift * s - 0.9 * s * s);
        // The frond: a stem from its root below the ground to its tip, bending as an arc; the
        // fronds of a fern swing a little out of step.
        let root = point(0.0) - Vec3::Y * ROOT_DEPTH;
        let frond = rig.add(None, root, point(1.0), BoneClass::Bend, 0.6 * stem_amplitude(root.distance(point(1.0))), 1.0, 0.15 * (f as f32 * 0.618_034).fract());
        for k in 1..leaflets {
            let s = k as f32 / leaflets as f32;
            let p = point(s);
            let tangent = (point(s + 0.01) - p).normalize();
            let side = tangent.cross(Vec3::Y).normalize_or(Vec3::X);
            let size = len * 0.22 * (1.0 - s * 0.8);
            // A leaflet's tip is a leaflet's length further along: its own weight is the stalk's.
            let along = (plant(s), plant((s + size / len).min(1.0)));
            for sign in [-1.0f32, 1.0] {
                let dir = (side * sign + tangent * 0.35 - Vec3::Y * 0.15).normalize();
                let phase = ((f * leaflets + k) as f32 * 0.618_034 + sign * 0.25).rem_euclid(1.0);
                b.set_bone(leaf_bone(&mut rig, Some(frond), p, dir, size, phase));
                add_leaf_with_plant(&mut b, p, dir, Vec3::Y, LeafShape { length: size, width: size * 0.32, fold: 0.2 }, along);
            }
        }
        // The rachis: a stalk following the frond from below the ground to its tip, tapering.
        let side = out.cross(Vec3::Y);
        let mut stalk = vec![(point(0.0) - Vec3::Y * ROOT_DEPTH, 0.008, Vec2::new(0.0, 0.35))];
        for k in 0..=3 {
            let s = k as f32 / 3.0;
            stalk.push((point(s), 0.007 * (1.0 - 0.75 * s), plant(s)));
        }
        // Facing up (the strip's winding: side × along).
        let normal = side.cross(point(0.5) - point(0.4)).normalize_or(Vec3::Y);
        b.set_bone(frond);
        add_strip(&mut b, &stalk, side, normal, 0.6);
    }
    let mut mesh = b.build();
    mesh.bones = rig.bones;
    mesh
}

/// A clump of grass: blades grow out of the soil from a tuft a few centimetres wide and fan out,
/// widest at the base and tapering to the tip, arching away from the centre. Wind weights hold
/// a blade where it leaves the ground and free it within its lower third; the baked occlusion
/// darkens the tuft's base, inner blades most.
pub fn grass_clump(seed: u32, blades: u32, height: f32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut b = MeshBuilder::default();
    let mut rig = PlantRig::default();
    let tuft = 0.06 * height / 0.45;
    for _ in 0..blades {
        let r = tuft * rng.float().sqrt();
        let a = rng.range(0.0, std::f32::consts::TAU);
        let out = Vec3::new(a.cos(), 0.0, a.sin());
        // Outer blades lean further out; each turns a little about its root.
        let tilt = 0.12 + 0.5 * (r / tuft) + rng.range(0.0, 0.2);
        let turn = rng.range(-0.5, 0.5);
        let lean = (out * turn.cos() + out.cross(Vec3::Y) * turn.sin()).normalize();
        let dir = Vec3::Y * tilt.cos() + lean * tilt.sin();
        let visible = height * rng.range(0.6, 1.2);
        let arch = rng.range(0.05, 0.3);
        let width = rng.range(0.012, 0.02) * 0.5;
        let root = out * r - Vec3::Y * ROOT_DEPTH;
        let total = visible + ROOT_DEPTH;
        let darkness = 0.35 + 0.25 * (r / tuft);
        // Rows from the root below the ground to the tip: two segments, 6 triangles a blade (the
        // attributes interpolate between them: at the ground the weight is a fifth of its value
        // at the middle row).
        let rows = [0.0, 0.45, 1.0];
        let points: Vec<(Vec3, f32, Vec2)> = rows
            .iter()
            .map(|&s| {
                let p = root + (dir * s + (lean - Vec3::Y * 0.4) * arch * s * s) * total;
                let above = ((s * total - ROOT_DEPTH) / visible).max(0.0);
                let half = width * (1.0 - 0.85 * s.powf(1.5));
                (p, half, Vec2::new(smoothstep(0.0, 0.35, above), darkness + (1.0 - darkness) * smoothstep(0.0, 0.7, above)))
            })
            .collect();
        let side = lean.cross(Vec3::Y).normalize_or(Vec3::X);
        // Each blade is a bone from its root to its tip, bending as an arc, out of step with
        // the others; longer blades swing slower.
        let tip = points[points.len() - 1].0;
        let blade = rig.add(None, root, tip, BoneClass::Bend, 0.9 * stem_amplitude(root.distance(tip)), (0.5 / total).sqrt().clamp(0.75, 1.0), (rig.bones.len() as f32 * 0.618_034).fract());
        b.set_bone(blade);
        add_strip(&mut b, &points, side, side.cross(dir).normalize_or(lean), 0.35);
    }
    let mut mesh = b.build();
    mesh.bones = rig.bones;
    mesh
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
    let mut plant = broadleaf_tree(&params);
    // Low leaves move with the low twigs: the wood's weights.
    root_weights(&mut plant.foliage, TREE_FREE_HEIGHT.min(0.25 * params.height));
    plant
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

    /// Grass and ferns grow out of the soil: their lowest vertices are below the ground and held
    /// (wind weight 0), what stands above it moves (a fern's drooping frond tips near the ground
    /// too), and their bases are darker than their tips.
    #[test]
    fn plants_are_rooted_in_the_ground() {
        for mesh in [grass_clump(91, 60, 0.45), fern(81, 9, 0.9, 14)] {
            assert_eq!(mesh.plant.len(), mesh.vertices.len());
            let lowest = mesh.vertices.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
            assert!(lowest <= -ROOT_DEPTH + 1e-4, "roots start below the ground: {lowest}");
            for (v, plant) in mesh.vertices.iter().zip(&mesh.plant) {
                if v.position[1] < -0.01 && Vec2::new(v.position[0], v.position[2]).length() < 0.1 {
                    assert!(plant.x < 0.2, "a root at {:?} moves with weight {}", v.position, plant.x);
                }
            }
            let free = mesh.plant.iter().filter(|p| p.x > 0.99).count();
            assert!(free * 3 > mesh.vertices.len(), "most of the plant moves freely: {free}/{}", mesh.vertices.len());
            let (base, tip) = (mesh.plant.iter().map(|p| p.y).fold(1.0f32, f32::min), mesh.plant.iter().map(|p| p.y).fold(0.0f32, f32::max));
            assert!(base < 0.5 && tip > 0.99, "occlusion from {base} at the base to {tip}");
        }
        // A fern frond tip resting near the ground still moves.
        let fern = fern(81, 9, 0.9, 14);
        let mut low_and_far = fern.vertices.iter().zip(&fern.plant).filter(|(v, _)| v.position[1] < 0.15 && Vec2::new(v.position[0], v.position[2]).length() > 0.5);
        assert!(low_and_far.clone().count() > 0 && low_and_far.all(|(_, p)| p.x > 0.99));
        // Trees and bushes are held where they meet the ground.
        let bush = bush(61, 1.4, 10);
        for mesh in [&bush.wood, &bush.foliage] {
            for (v, plant) in mesh.vertices.iter().zip(&mesh.plant) {
                assert!(v.position[1] > 0.0 || plant.x == 0.0);
            }
        }
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
