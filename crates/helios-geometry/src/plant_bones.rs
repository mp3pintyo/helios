//! Plant bones: the skeleton of a procedural plant for the renderer's dynamic wind (UE 5.9's
//! Dynamic Wind for Nanite foliage; the motion itself is `helios-render`'s `plant_wind.wgsl`).
//!
//! The vegetation generators know a plant's structure — trunk, branches, twigs, leaves, grass
//! blades, fern fronds and leaflets — and emit one [`PlantBone`] per element plus, per vertex,
//! the bone it belongs to (the mesh's skin attribute, [`crate::Mesh::skin`], rigidly bound:
//! the cluster encoding, simplification and LOD already carry it crack-free). Each bone is a
//! segment from its *pivot* along its *axis*: the wind bends it about the pivot as a circular
//! arc (its frame at distance `s` along it turns by `θ·s/L`, so a blade or a branch curves
//! rather than tilting like a stick, and nothing stretches), and everything attached to it
//! rides on its frame at the attachment point. A leaf turns about its stem, a frond's
//! leaflets about the rachis, and normals turn with them.
//!
//! Every bone belongs to one of three *classes* whose motion over time is the same as the
//! three layers of the procedural wind (`wind.wgsl`): [`BoneClass::Bend`] (stems and trunks
//! lean with the gusts and sway at the plant's frequency), [`BoneClass::Sway`] (branches bob,
//! at most 2.3× faster) and [`BoneClass::Flutter`] (leaves flutter and twist). That keeps the
//! renderer's culling and shadow-invalidation bounds, which only know the three layers, valid:
//! [`BoneBounds`] fits, per mesh, coefficients of the procedural layers that bound the bones'
//! displacement ([`BoneBounds::equivalent`]).

use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::{Mesh, SkinWeights};

/// The motion layer of a bone (its time function; see the module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BoneClass {
    /// Trunks, stems, grass blades and fern fronds: lean with the gusts, sway at the plant's
    /// frequency (the material's `bend`).
    #[default]
    Bend,
    /// Branches: bob at up to 2.3× the plant's frequency (the material's `sway`).
    Sway,
    /// Leaves and leaflets: flutter and twist about their stem (the material's `flutter`).
    Flutter,
}

impl BoneClass {
    pub fn index(self) -> u32 {
        self as u32
    }

    pub fn from_index(i: u32) -> Self {
        match i {
            1 => Self::Sway,
            2 => Self::Flutter,
            _ => Self::Bend,
        }
    }

    /// Upper bound of the magnitude of the class's force vector (its time function): the
    /// rotation of a bone is at most `amplitude × response × strength × OMEGA_BOUND`.
    pub fn omega_bound(self) -> f32 {
        match self {
            // |lean| ≤ 1, |lateral| ≤ 0.3: sqrt(1 + 0.09)
            Self::Bend => 1.0441,
            Self::Sway => 1.0,
            // three components in [-1, 1]
            Self::Flutter => 1.7321,
        }
    }
}

/// One bone of a plant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantBone {
    /// The bone it is attached to (`None`: the plant's root, held by the ground).
    pub parent: Option<u16>,
    /// Where it starts (object space): it bends about this point.
    pub pivot: Vec3,
    /// Unit direction from the pivot along the bone.
    pub axis: Vec3,
    pub length: f32,
    /// Distance along the parent's axis at which it is attached (its pivot's projection,
    /// clamped to the parent's length): it rides on the parent's frame there.
    pub attach: f32,
    pub class: BoneClass,
    /// Rotation (radians) of the whole bone per unit of the material's response of its class
    /// and unit wind strength: how compliant it is (from its length, thickness and the area it
    /// catches the wind with).
    pub amplitude: f32,
    /// Its frequency as a fraction (≤ 1) of its class's fastest: heavier bones swing slower.
    pub frequency: f32,
    /// Phase offset in [0, 1) of a period (blades and fronds of one plant swing out of step).
    pub phase: f32,
}

impl PlantBone {
    /// Distance along the bone of `p`'s projection, clamped to the bone.
    pub fn project(&self, p: Vec3) -> f32 {
        (p - self.pivot).dot(self.axis).clamp(0.0, self.length)
    }

    /// Upper bound of how far the bone's frame at `s` moves a point at `p` per radian of the
    /// bone's rotation: the arc's offset `s²/2L` plus the frame's rotation `s/L` times the
    /// distance from the axis point at `s` (exact in the limit of small angles, an upper bound
    /// for any: chords are shorter than arcs).
    pub fn reach(&self, s: f32, p: Vec3) -> f32 {
        let l = self.length.max(1e-6);
        s * s / (2.0 * l) + s / l * (p - (self.pivot + self.axis * s)).length()
    }

    /// The bone transformed by an affine `transform` with uniform scale (assembly parts).
    pub fn transformed(&self, transform: &Mat4, scale: f32) -> Self {
        Self {
            pivot: transform.transform_point3(self.pivot),
            axis: (Mat3::from_mat4(*transform) * self.axis).normalize_or(self.axis),
            length: self.length * scale,
            attach: self.attach * scale,
            ..*self
        }
    }
}

/// Builds a plant's bone table.
#[derive(Clone, Debug, Default)]
pub struct PlantRig {
    pub bones: Vec<PlantBone>,
}

impl PlantRig {
    /// Adds a bone from `pivot` to `end` and returns its index.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&mut self, parent: Option<u16>, pivot: Vec3, end: Vec3, class: BoneClass, amplitude: f32, frequency: f32, phase: f32) -> u16 {
        let length = pivot.distance(end).max(1e-4);
        let axis = (end - pivot).normalize_or(Vec3::Y);
        let attach = parent.map_or(0.0, |p| self.bones[p as usize].project(pivot));
        self.bones.push(PlantBone { parent, pivot, axis, length, attach, class, amplitude, frequency: frequency.clamp(0.05, 1.0), phase: phase.rem_euclid(1.0) });
        assert!(self.bones.len() <= u16::MAX as usize, "too many plant bones");
        (self.bones.len() - 1) as u16
    }
}

/// The bones of a mesh and the bounds of their motion ([`crate::VirtualMesh::bones`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlantBones {
    pub bones: Vec<PlantBone>,
    pub bounds: BoneBounds,
}

/// Words per bone in a `.hvm` and in [`PlantBones::to_words`].
pub const BONE_WORDS: usize = 13;
/// Words of [`BoneBounds`].
pub const BOUND_WORDS: usize = 6;

impl PlantBones {
    /// Bones and bounds fitted to every vertex of `mesh` (its skin: one bone per vertex).
    pub fn of_mesh(mesh: &Mesh, bones: Vec<PlantBone>) -> Self {
        let points = mesh.vertices.iter().enumerate().map(|(i, v)| (v.position(), mesh.skin_of(i).dominant_joint()));
        let bounds = BoneBounds::fit(&class_reaches(&bones, points));
        Self { bones, bounds }
    }

    /// Serialised: the bounds, then [`BONE_WORDS`] per bone.
    pub fn to_words(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self.bounds.to_array().iter().map(|f| f.to_bits()).collect();
        for b in &self.bones {
            let f = [
                b.pivot.x, b.pivot.y, b.pivot.z, b.axis.x, b.axis.y, b.axis.z, b.length, b.attach, b.amplitude, b.frequency, b.phase,
            ];
            out.extend(f.iter().map(|f| f.to_bits()));
            out.push(b.parent.map_or(u32::MAX, u32::from));
            out.push(b.class.index());
        }
        out
    }

    /// Inverse of [`PlantBones::to_words`] for `count` bones.
    pub fn from_words(words: &[u32], count: usize) -> Result<Self, String> {
        if words.len() != BOUND_WORDS + count * BONE_WORDS {
            return Err("plant bone section has the wrong size".into());
        }
        let f = |i: usize| f32::from_bits(words[i]);
        let bounds = BoneBounds::from_array(std::array::from_fn(f));
        let bones = (0..count)
            .map(|i| {
                let o = BOUND_WORDS + i * BONE_WORDS;
                let w = &words[o..o + BONE_WORDS];
                let g = |k: usize| f32::from_bits(w[k]);
                PlantBone {
                    pivot: Vec3::new(g(0), g(1), g(2)),
                    axis: Vec3::new(g(3), g(4), g(5)),
                    length: g(6),
                    attach: g(7),
                    amplitude: g(8),
                    frequency: g(9),
                    phase: g(10),
                    parent: (w[11] != u32::MAX).then_some(w[11] as u16),
                    class: BoneClass::from_index(w[12]),
                }
            })
            .collect();
        Ok(Self { bones, bounds })
    }
}

/// Per vertex, the bound of its displacement per unit response of each class (bend, sway,
/// flutter) at unit wind strength: the sum over its bone chain of each bone's
/// `amplitude × class bound × reach`. Rigid frames are isometries, so the displacements of the
/// chain's levels simply add up (evaluated at the rest position).
pub fn vertex_reach(bones: &[PlantBone], bone: u16, p: Vec3) -> Vec3 {
    let mut out = Vec3::ZERO;
    let mut b = bone as usize;
    let mut s = bones[b].project(p);
    for _ in 0..64 {
        let bone = &bones[b];
        let r = bone.amplitude * bone.class.omega_bound() * bone.reach(s, p);
        out[bone.class.index() as usize] += r;
        match bone.parent {
            Some(parent) => {
                s = bone.attach;
                b = parent as usize;
            }
            None => break,
        }
    }
    out
}

/// `(position, per-class reach)` of points bound to bones.
pub fn class_reaches(bones: &[PlantBone], points: impl Iterator<Item = (Vec3, u16)>) -> Vec<(Vec3, Vec3)> {
    if bones.is_empty() {
        return Vec::new();
    }
    points.map(|(p, b)| (p, vertex_reach(bones, b.min(bones.len() as u16 - 1), p))).collect()
}

/// The voxel bone of `bone`: its nearest ancestor that is not a leaf. Voxels (far LODs) move
/// with their branches; leaf flutter is far below their size.
pub fn voxel_bone(bones: &[PlantBone], bone: u16) -> u16 {
    let mut b = bone;
    while bones[b as usize].class == BoneClass::Flutter {
        match bones[b as usize].parent {
            Some(p) => b = p,
            None => break,
        }
    }
    b
}

/// Bounds of the bones' displacement in the shape of the procedural wind's three layers
/// (`wind.wgsl` `wind_reach`): `bend·h² + sway·ρ + flutter` with `h` the height above the
/// plant's origin and `ρ` the distance from its vertical axis (object space, per unit response
/// and strength). A class may only spill into layers whose speed bound is at least its own
/// (bend → sway → flutter), so the layers' speed bounds stay valid too.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoneBounds {
    /// Bend class: (h² coefficient, ρ coefficient, constant).
    pub bend: [f32; 3],
    /// Sway class: (ρ coefficient, constant).
    pub sway: [f32; 2],
    /// Flutter class: constant.
    pub flutter: f32,
}

/// Safety margin of the fitted bounds: second-order terms of the speed bound (a parent's
/// rotation acting on what its children already moved) and float rounding.
const BOUND_MARGIN: f32 = 1.06;

impl BoneBounds {
    pub fn to_array(&self) -> [f32; BOUND_WORDS] {
        [self.bend[0], self.bend[1], self.bend[2], self.sway[0], self.sway[1], self.flutter]
    }

    pub fn from_array(a: [f32; BOUND_WORDS]) -> Self {
        Self { bend: [a[0], a[1], a[2]], sway: [a[3], a[4]], flutter: a[5] }
    }

    /// Fits the bounds to `(position, per-class reach)` points.
    pub fn fit(points: &[(Vec3, Vec3)]) -> Self {
        let basis: Vec<(f32, f32)> = points.iter().map(|(p, _)| (p.y.max(0.0), (p.x * p.x + p.z * p.z).sqrt())).collect();
        let class = |c: usize| -> Vec<f32> { points.iter().map(|(_, r)| r[c] * BOUND_MARGIN).collect() };
        let flutter = class(2).into_iter().fold(0.0f32, f32::max);
        let (sway_rho, sway_const) = fit_linear(&basis, &class(1));
        let bend = fit_quadratic(&basis, &class(0));
        Self { bend, sway: [sway_rho, sway_const], flutter }
    }

    /// The procedural wind's (bend, sway, flutter, frequency) whose bounds (`wind_reach`,
    /// `wind_speed`) hold the bones' motion for a material `response` (bend, sway, flutter,
    /// frequency), at an instance scale `scale` (world-space heights and distances).
    pub fn equivalent(&self, response: Vec4, scale: f32) -> Vec4 {
        // wind_reach: strength · (1.05·bend·h² + sway·ρ + 1.7321·flutter), world units.
        const BEND_BOUND: f32 = 1.05;
        const FLUTTER_BOUND: f32 = 1.7321;
        let s = scale.max(1e-6);
        let bend = response.x * self.bend[0] / (BEND_BOUND * s);
        let sway = response.y * self.sway[0] + response.x * self.bend[1];
        let flutter = (response.z * self.flutter + response.y * self.sway[1] + response.x * self.bend[2]) * s / FLUTTER_BOUND;
        let any = response.x != 0.0 || response.y != 0.0 || response.z != 0.0;
        // Never all zero for a moving material (zero marks static instances).
        Vec4::new(bend, sway, if any { flutter.max(1e-7) } else { flutter }, response.w)
    }

    /// The bound itself (object space, unit strength) at height `h` and axis distance `rho`.
    pub fn at(&self, response: Vec4, h: f32, rho: f32) -> f32 {
        let e = self.equivalent(response, 1.0);
        1.05 * e.x * h * h + e.y * rho + 1.7321 * e.z
    }
}

/// `(a, c)` minimising the mean of `a·ρ + c` subject to `a·ρᵢ + c ≥ rᵢ`.
fn fit_linear(basis: &[(f32, f32)], r: &[f32]) -> (f32, f32) {
    let max = r.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return (0.0, 0.0);
    }
    let mean_rho = basis.iter().map(|b| b.1).sum::<f32>() / basis.len().max(1) as f32;
    let mut candidates: Vec<f32> = vec![0.0];
    let mut ratios: Vec<f32> = basis.iter().zip(r).filter(|(b, &r)| b.1 > 1e-4 && r > 0.0).map(|(b, &r)| r / b.1).collect();
    ratios.sort_by(f32::total_cmp);
    for q in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0] {
        if let Some(&v) = ratios.get(((ratios.len() as f32 - 1.0) * q) as usize) {
            candidates.push(v);
        }
    }
    let mut best = (0.0, max, f32::MAX);
    for a in candidates {
        let c = basis.iter().zip(r).map(|(b, &r)| r - a * b.1).fold(0.0f32, f32::max);
        let cost = a * mean_rho + c;
        if cost < best.2 {
            best = (a, c, cost);
        }
    }
    (best.0, best.1)
}

/// `[a, b, c]` keeping `a·h² + b·ρ + c ≥ rᵢ` with a small mean bound: a search over `a`, the
/// rest fitted linearly.
fn fit_quadratic(basis: &[(f32, f32)], r: &[f32]) -> [f32; 3] {
    let max = r.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return [0.0; 3];
    }
    let mean_h2 = basis.iter().map(|b| b.0 * b.0).sum::<f32>() / basis.len().max(1) as f32;
    let mut ratios: Vec<f32> = basis.iter().zip(r).filter(|(b, &r)| b.0 > 1e-3 && r > 0.0).map(|(b, &r)| r / (b.0 * b.0)).collect();
    ratios.sort_by(f32::total_cmp);
    let mut candidates: Vec<f32> = vec![0.0];
    for q in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0] {
        if let Some(&v) = ratios.get(((ratios.len() as f32 - 1.0) * q) as usize) {
            candidates.push(v);
        }
    }
    let mut best = ([0.0, 0.0, max], f32::MAX);
    for a in candidates {
        let rest: Vec<f32> = basis.iter().zip(r).map(|(b, &r)| (r - a * b.0 * b.0).max(0.0)).collect();
        let (b, c) = fit_linear(basis, &rest);
        let mean_rho = basis.iter().map(|b| b.1).sum::<f32>() / basis.len().max(1) as f32;
        let cost = a * mean_h2 + b * mean_rho + c;
        if cost < best.1 {
            best = ([a, b, c], cost);
        }
    }
    best.0
}

/// The bone table of an assembly part placed at `transform` (uniform scale) hanging from
/// `bone` of its plant's bones: the part mesh's own bones (`part`, e.g. the needles of a spray,
/// whose roots have no parent) in the part's space first — the part's vertices keep their
/// indices — then the chain from `bone` to the plant's root transformed into the part's space.
/// Without `bone` (or plant bones) the part's own bones are its roots.
pub fn part_bone_table(part: &[PlantBone], plant: &[PlantBone], bone: Option<u16>, transform: &Mat4) -> Vec<PlantBone> {
    let mut out: Vec<PlantBone> = part.to_vec();
    let Some(bone) = bone.filter(|&b| (b as usize) < plant.len()) else {
        return out;
    };
    let scale = crate::assembly::max_scale(transform);
    let inverse = transform.inverse();
    let first = out.len() as u16;
    // The chain, re-indexed: plant bone b → first + its position in the chain.
    let mut chain = vec![bone];
    while let Some(p) = plant[*chain.last().unwrap() as usize].parent {
        chain.push(p);
    }
    for (k, &b) in chain.iter().enumerate() {
        let mut t = plant[b as usize].transformed(&inverse, 1.0 / scale);
        t.parent = (k + 1 < chain.len()).then_some(first + k as u16 + 1);
        out.push(t);
    }
    // The part's roots ride on the chain's first bone at their pivots.
    let hang = out[first as usize];
    for b in out.iter_mut().take(first as usize) {
        if b.parent.is_none() {
            b.parent = Some(first);
            b.attach = hang.project(b.pivot);
        }
    }
    out
}

/// Binds every vertex of `mesh` rigidly to `bone` (its skin).
pub fn bind_all(mesh: &mut Mesh, bone: u16) {
    mesh.skin = vec![SkinWeights::rigid(bone); mesh.vertices.len()];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> PlantRig {
        let mut rig = PlantRig::default();
        let trunk = rig.add(None, Vec3::ZERO, Vec3::Y * 10.0, BoneClass::Bend, 20.0, 1.0, 0.0);
        let branch = rig.add(Some(trunk), Vec3::Y * 6.0, Vec3::new(4.0, 8.0, 0.0), BoneClass::Sway, 1.0, 0.7, 0.3);
        rig.add(Some(branch), Vec3::new(3.0, 7.5, 0.0), Vec3::new(3.0, 7.5, 0.3), BoneClass::Flutter, 8.0, 0.9, 0.6);
        rig
    }

    #[test]
    fn bones_round_trip_and_chain_reaches_add_up() {
        let rig = rig();
        let bones = PlantBones { bones: rig.bones.clone(), bounds: BoneBounds { bend: [1.0, 2.0, 3.0], sway: [4.0, 5.0], flutter: 6.0 } };
        let words = bones.to_words();
        assert_eq!(PlantBones::from_words(&words, 3).unwrap(), bones);
        // The leaf's tip: its own flutter plus the branch's and the trunk's frames.
        let tip = Vec3::new(3.0, 7.5, 0.3);
        let r = vertex_reach(&rig.bones, 2, tip);
        assert!(r.x > 0.0 && r.y > 0.0 && r.z > 0.0, "{r}");
        // At the trunk's pivot nothing moves.
        assert_eq!(vertex_reach(&rig.bones, 0, Vec3::ZERO), Vec3::ZERO);
        assert_eq!(voxel_bone(&rig.bones, 2), 1);
    }

    #[test]
    fn fitted_bounds_hold_every_point() {
        let rig = rig();
        let mut points = Vec::new();
        for i in 0..200 {
            let t = i as f32 / 199.0;
            points.push((Vec3::new(0.1, 10.0 * t, 0.0), 0u16));
            points.push((Vec3::new(4.0 * t, 6.0 + 2.0 * t, 0.05), 1u16));
            points.push((Vec3::new(3.0, 7.5, 0.3 * t), 2u16));
        }
        let reaches = class_reaches(&rig.bones, points.into_iter());
        let bounds = BoneBounds::fit(&reaches);
        for response in [Vec4::new(0.0009, 0.012, 0.01, 0.3), Vec4::new(0.2, 0.0, 0.003, 0.9), Vec4::new(0.0, 0.0, 0.01, 1.0)] {
            for (p, r) in &reaches {
                let need = response.x * r.x + response.y * r.y + response.z * r.z;
                let have = bounds.at(response, p.y.max(0.0), Vec3::new(p.x, 0.0, p.z).length());
                assert!(need <= have * 1.0001 + 1e-7, "{response} at {p}: {need} > {have}");
            }
        }
    }

    #[test]
    fn part_tables_hang_the_part_on_its_bone() {
        let rig = rig();
        let needle = PlantBone { parent: None, pivot: Vec3::new(0.5, 0.0, 0.0), axis: Vec3::X, length: 0.2, attach: 0.0, class: BoneClass::Flutter, amplitude: 5.0, frequency: 1.0, phase: 0.0 };
        let transform = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), glam::Quat::from_rotation_y(0.7), Vec3::new(0.0, 6.0, 0.0));
        let table = part_bone_table(&[needle], &rig.bones, Some(1), &transform);
        assert_eq!(table.len(), 3);
        assert_eq!(table[0].parent, Some(1));
        assert_eq!(table[1].parent, Some(2));
        assert_eq!(table[2].parent, None);
        // The branch in part space, mapped back, is the plant's branch.
        let back = table[1].transformed(&transform, 2.0);
        assert!(back.pivot.distance(rig.bones[1].pivot) < 1e-4 && back.axis.distance(rig.bones[1].axis) < 1e-4);
        assert!((back.length - rig.bones[1].length).abs() < 1e-4);
    }
}
