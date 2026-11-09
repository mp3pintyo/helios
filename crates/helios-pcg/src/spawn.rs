//! Weighted mesh spawner (UE's Static Mesh Spawner): each point becomes one entry, picked by
//! weight from the point's seed. An entry is a small prefab of mesh/material parts sharing the
//! point's transform (a tree's wood and foliage, a flower's stem and petals).

use glam::Mat4;
use serde::{Deserialize, Serialize};

use crate::point::Point;
use crate::rng::random;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpawnPart {
    /// Mesh and material names, resolved by whoever spawns them.
    pub mesh: String,
    pub material: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpawnEntry {
    pub weight: f32,
    pub parts: Vec<SpawnPart>,
}

/// One spawned instance: a part (index into [`crate::Output::parts`]) and its transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spawned {
    pub part: u32,
    pub transform: Mat4,
}

/// Picks an entry per point by weight; returns the chosen entry index per point (`None`
/// without entries or weights).
pub fn choose(points: &[Point], entries: &[SpawnEntry], salt: u32) -> Vec<Option<usize>> {
    let total: f32 = entries.iter().map(|e| e.weight.max(0.0)).sum();
    points
        .iter()
        .map(|p| {
            if total <= 0.0 {
                return None;
            }
            let mut pick = random(p.seed, salt, 0) * total;
            for (i, e) in entries.iter().enumerate() {
                pick -= e.weight.max(0.0);
                if pick < 0.0 {
                    return Some(i);
                }
            }
            entries.iter().rposition(|e| e.weight > 0.0)
        })
        .collect()
}

pub fn transform_of(p: &Point) -> Mat4 {
    Mat4::from_scale_rotation_translation(p.scale, p.rotation, p.position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn weights_set_the_mix() {
        let part = |m: &str| SpawnPart { mesh: m.into(), material: "m".into() };
        let entries = vec![SpawnEntry { weight: 3.0, parts: vec![part("a")] }, SpawnEntry { weight: 1.0, parts: vec![part("b")] }];
        let points: Vec<Point> = (0..4000).map(|i| Point::new(Vec3::ZERO, Vec3::Y, i * 7919)).collect();
        let chosen = choose(&points, &entries, 3);
        let a = chosen.iter().filter(|c| **c == Some(0)).count();
        assert!((2850..3150).contains(&a), "{a}");
        assert!(chosen.iter().all(Option::is_some));
    }
}
