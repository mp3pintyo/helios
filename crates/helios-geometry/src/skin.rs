//! Skin weights of skeletal meshes (glTF `JOINTS_0` / `WEIGHTS_0`) and the linear blend skinning
//! they drive.
//!
//! Weights are quantised **per vertex, once, before anything else** (welding, clustering,
//! simplification, compression): every cluster and every LOD level that shares a vertex then
//! carries the very same integers for it, and skinning — a pure function of the rest position,
//! these integers and the joint palette ([`skin_point`], mirrored by `skinning.wgsl`) — moves it
//! to the same bits in all of them. That is what keeps skinned LOD cuts crack-free, as the shared
//! position grid does for static ones (see [`crate::encoding`]).

use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::{Mesh, Sphere};

/// Influences per vertex (glTF's first joint set; more sets are rare and dropped).
pub const MAX_INFLUENCES: usize = 4;
/// Sum of a vertex's quantised weights: 8-bit weights, as GPU skinning commonly uses.
pub const WEIGHT_TOTAL: u32 = 255;

/// Up to [`MAX_INFLUENCES`] (joint, weight) pairs of a vertex.
///
/// Invariants (established by [`SkinWeights::new`]): weights sum to exactly [`WEIGHT_TOTAL`] and
/// are sorted by decreasing weight (ties: lower joint first), so the non-zero ones come first;
/// unused slots hold joint 0 and weight 0. The first weight is therefore the largest, and every
/// later one is at most half the total — the encoder stores only those and derives the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkinWeights {
    pub joints: [u16; MAX_INFLUENCES],
    pub weights: [u8; MAX_INFLUENCES],
}

impl Default for SkinWeights {
    /// Rigidly bound to joint 0.
    fn default() -> Self {
        Self::rigid(0)
    }
}

impl SkinWeights {
    /// A vertex that follows one joint.
    pub const fn rigid(joint: u16) -> Self {
        Self { joints: [joint, 0, 0, 0], weights: [WEIGHT_TOTAL as u8, 0, 0, 0] }
    }

    /// Quantises float weights (any scale; glTF's should sum to 1): non-positive and non-finite
    /// weights are dropped, repeated joints merged, the rest renormalised and rounded to integers
    /// summing to [`WEIGHT_TOTAL`] (largest remainders round up). A vertex without any positive
    /// weight follows its first joint.
    pub fn new(joints: [u16; MAX_INFLUENCES], weights: [f32; MAX_INFLUENCES]) -> Self {
        let mut pairs: Vec<(u16, f32)> = Vec::with_capacity(MAX_INFLUENCES);
        for (&joint, &weight) in joints.iter().zip(&weights) {
            if !(weight > 0.0 && weight.is_finite()) {
                continue;
            }
            match pairs.iter_mut().find(|(j, _)| *j == joint) {
                Some(pair) => pair.1 += weight,
                None => pairs.push((joint, weight)),
            }
        }
        let total: f32 = pairs.iter().map(|p| p.1).sum();
        if pairs.is_empty() || !total.is_finite() {
            return Self::rigid(joints[0]);
        }
        // Largest remainder rounding: the integers sum to WEIGHT_TOTAL exactly. Ties go to the
        // lower joint, so the result does not depend on the order of the input.
        let scaled: Vec<f32> = pairs.iter().map(|p| p.1 / total * WEIGHT_TOTAL as f32).collect();
        let mut quantised: Vec<u32> = scaled.iter().map(|s| s.floor() as u32).collect();
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by(|&a, &b| (scaled[b] - scaled[b].floor()).total_cmp(&(scaled[a] - scaled[a].floor())).then(pairs[a].0.cmp(&pairs[b].0)));
        let missing = WEIGHT_TOTAL - quantised.iter().sum::<u32>().min(WEIGHT_TOTAL);
        for &i in order.iter().cycle().take(missing as usize) {
            quantised[i] += 1;
        }
        let mut out: Vec<(u16, u32)> = pairs.iter().zip(&quantised).map(|(p, &q)| (p.0, q)).filter(|p| p.1 > 0).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut skin = Self { joints: [0; MAX_INFLUENCES], weights: [0; MAX_INFLUENCES] };
        for (slot, (joint, weight)) in out.into_iter().enumerate() {
            skin.joints[slot] = joint;
            skin.weights[slot] = weight as u8;
        }
        skin
    }

    /// Number of joints with a non-zero weight (1..=4).
    pub fn influences(&self) -> usize {
        self.weights.iter().take_while(|&&w| w > 0).count()
    }

    /// The (joint, weight in 0..=1) pairs with non-zero weight, heaviest first.
    pub fn iter(&self) -> impl Iterator<Item = (u16, f32)> + '_ {
        self.joints.iter().zip(&self.weights).take_while(|(_, &w)| w > 0).map(|(&j, &w)| (j, w as f32 / WEIGHT_TOTAL as f32))
    }

    /// The joint with the largest weight.
    pub fn dominant_joint(&self) -> u16 {
        self.joints[0]
    }

    /// Fraction of the weight bound to different joints: half the L1 distance between the two
    /// weight distributions, 0 for identical skins, 1 for disjoint joint sets.
    pub fn distance(&self, other: &SkinWeights) -> f32 {
        let weight_of = |s: &SkinWeights, joint: u16| s.iter().filter(|p| p.0 == joint).map(|p| p.1).sum::<f32>();
        let mut l1 = 0.0;
        for (joint, w) in self.iter() {
            l1 += (w - weight_of(other, joint)).abs();
        }
        for (joint, w) in other.iter() {
            if weight_of(self, joint) == 0.0 {
                l1 += w;
            }
        }
        (l1 * 0.5).min(1.0)
    }
}

/// The blended skinning matrix of `skin`: `Σ wᵢ · palette[jointᵢ]` over the non-zero weights in
/// slot order (zero weights are skipped rather than added, so a vertex blends exactly the same
/// terms wherever it is decoded). Mirrors `skin_matrix` in `skinning.wgsl`.
pub fn skin_matrix(skin: &SkinWeights, palette: &[Mat4]) -> Mat4 {
    let mut m = Mat4::ZERO;
    for (joint, &w) in skin.joints.iter().zip(&skin.weights) {
        if w > 0 {
            m += palette[*joint as usize] * (w as f32 / WEIGHT_TOTAL as f32);
        }
    }
    m
}

/// Linear blend skinning of a rest position.
pub fn skin_point(skin: &SkinWeights, palette: &[Mat4], rest: Vec3) -> Vec3 {
    skin_matrix(skin, palette).transform_point3(rest)
}

/// Skinned normal: the blended matrix's cofactor (inverse transpose up to scale), so non-uniform
/// joint scale keeps normals perpendicular.
pub fn skin_normal(skin: &SkinWeights, palette: &[Mat4], normal: Vec3) -> Vec3 {
    let m = Mat3::from_mat4(skin_matrix(skin, palette));
    let cofactor = Mat3::from_cols(m.y_axis.cross(m.z_axis), m.z_axis.cross(m.x_axis), m.x_axis.cross(m.y_axis));
    (cofactor * normal).normalize_or(normal)
}

/// Per joint, a bounding sphere of the rest positions of every vertex it influences (`None` for
/// joints without vertices). Linear blend skinning moves a vertex to a convex combination of its
/// joints' transforms of it, so the posed mesh lies within the joints' posed spheres — what the
/// renderer's conservative culling bounds and motion bounds build on.
pub fn joint_bounds(mesh: &Mesh, joint_count: usize) -> Vec<Option<Sphere>> {
    let mut points: Vec<Vec<Vec3>> = vec![Vec::new(); joint_count];
    for (v, skin) in mesh.vertices.iter().zip(&mesh.skin) {
        for (joint, _) in skin.iter() {
            if let Some(list) = points.get_mut(joint as usize) {
                list.push(v.position());
            }
        }
    }
    points.iter().map(|p| (!p.is_empty()).then(|| Sphere::from_points(p))).collect()
}

/// `joint_bounds` as `(centre, radius)` vectors, radius -1 for joints without vertices.
pub fn joint_bounds_vec4(bounds: &[Option<Sphere>]) -> Vec<Vec4> {
    bounds.iter().map(|b| b.map_or(Vec4::new(0.0, 0.0, 0.0, -1.0), |s| s.center.extend(s.radius))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_quantise_to_the_exact_total_heaviest_first() {
        let s = SkinWeights::new([3, 7, 1, 9], [0.2, 0.5, 0.0, 0.3]);
        assert_eq!(s.weights.iter().map(|&w| w as u32).sum::<u32>(), WEIGHT_TOTAL);
        assert_eq!((s.joints[..3].to_vec(), s.influences()), (vec![7, 9, 3], 3));
        assert_eq!((s.joints[3], s.weights[3]), (0, 0));
        // Unnormalised input, repeated joints and junk weights.
        let s = SkinWeights::new([4, 4, 2, 5], [2.0, 2.0, 4.0, f32::NAN]);
        assert_eq!((s.joints[..2].to_vec(), s.weights[..2].to_vec()), (vec![2, 4], vec![128, 127]));
        assert_eq!(SkinWeights::new([6, 1, 2, 3], [0.0; 4]), SkinWeights::rigid(6));
        // Every later weight is at most half the total (the encoder relies on it).
        for i in 0..200u32 {
            let w = [i % 7, i % 5 + 1, i % 3, i % 11].map(|x| x as f32);
            let s = SkinWeights::new([0, 1, 2, 3], w);
            assert!(s.weights[1..].iter().all(|&w| w as u32 <= WEIGHT_TOTAL / 2), "{s:?}");
            assert!(s.weights.windows(2).all(|p| p[0] >= p[1]));
        }
    }

    #[test]
    fn distance_measures_moved_weight() {
        let a = SkinWeights::new([1, 2, 0, 0], [0.5, 0.5, 0.0, 0.0]);
        let b = SkinWeights::new([2, 1, 0, 0], [0.5, 0.5, 0.0, 0.0]);
        assert_eq!(a.distance(&b), 0.0);
        assert_eq!(a.distance(&SkinWeights::rigid(3)), 1.0);
        assert!((a.distance(&SkinWeights::rigid(1)) - 0.5).abs() < 0.01);
    }

    #[test]
    fn skinning_blends_joint_transforms() {
        let palette = [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0))];
        let half = SkinWeights::new([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0]);
        let p = skin_point(&half, &palette, Vec3::X);
        assert!((p - Vec3::new(1.0, 1.0, 0.0)).length() < 0.01, "{p}");
        // A uniformly scaled joint keeps the normal's direction.
        let palette = [Mat4::from_scale(Vec3::splat(2.0))];
        assert!((skin_normal(&SkinWeights::rigid(0), &palette, Vec3::Y) - Vec3::Y).length() < 1e-6);
    }
}
