//! Self-pruning and differences: Poisson-disk thinning to a minimum distance.
//!
//! Points are visited by priority — highest density first, ties broken by seed, so the order
//! is total and independent of how the points were produced — and accepted unless an accepted
//! point lies within the radius. A hash grid with cells of the radius makes it linear.

use std::collections::HashMap;

use glam::Vec3;

use crate::point::Point;

type Cell = (i32, i32, i32);

struct Grid {
    cell: f32,
    cells: HashMap<Cell, Vec<Vec3>>,
}

impl Grid {
    fn new(radius: f32) -> Self {
        Self { cell: radius.max(1e-4), cells: HashMap::new() }
    }

    fn key(&self, p: Vec3) -> Cell {
        let c = (p / self.cell).floor();
        (c.x as i32, c.y as i32, c.z as i32)
    }

    fn insert(&mut self, p: Vec3) {
        let key = self.key(p);
        self.cells.entry(key).or_default().push(p);
    }

    /// Whether a stored point lies closer than `radius` (= the cell size) to `p`.
    fn any_within(&self, p: Vec3, radius: f32) -> bool {
        let (x, y, z) = self.key(p);
        let r2 = radius * radius;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) = self.cells.get(&(x + dx, y + dy, z + dz)) {
                        if list.iter().any(|q| q.distance_squared(p) < r2) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// Removes points closer than `radius` to a point of higher priority.
pub fn self_prune(points: &mut Vec<Point>, radius: f32) {
    if radius <= 0.0 {
        return;
    }
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| points[b].density.total_cmp(&points[a].density).then(points[a].seed.cmp(&points[b].seed)));
    let mut grid = Grid::new(radius);
    let mut keep = vec![false; points.len()];
    for i in order {
        if !grid.any_within(points[i].position, radius) {
            grid.insert(points[i].position);
            keep[i] = true;
        }
    }
    let mut index = 0;
    points.retain(|_| {
        index += 1;
        keep[index - 1]
    });
}

/// Removes the points of `points` that lie within `radius` of any point of `others`.
pub fn difference(points: &mut Vec<Point>, others: &[Point], radius: f32) {
    if radius <= 0.0 || others.is_empty() {
        return;
    }
    let mut grid = Grid::new(radius);
    for p in others {
        grid.insert(p.position);
    }
    points.retain(|p| !grid.any_within(p.position, radius));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{hash, unit};

    fn scattered(n: u32) -> Vec<Point> {
        (0..n).map(|i| Point::new(Vec3::new(unit(hash(i)) * 30.0, 0.0, unit(hash(i + 99_991)) * 30.0), Vec3::Y, hash(i))).collect()
    }

    #[test]
    fn pruned_points_keep_their_distance() {
        let mut points = scattered(4000);
        self_prune(&mut points, 1.5);
        assert!(points.len() > 150 && points.len() < 600, "{}", points.len());
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                assert!(a.position.distance(b.position) >= 1.5);
            }
        }
    }

    #[test]
    fn pruning_ignores_input_order() {
        let mut a = scattered(2000);
        let mut b = a.clone();
        b.reverse();
        self_prune(&mut a, 1.0);
        self_prune(&mut b, 1.0);
        let mut seeds_a: Vec<u32> = a.iter().map(|p| p.seed).collect();
        let mut seeds_b: Vec<u32> = b.iter().map(|p| p.seed).collect();
        seeds_a.sort_unstable();
        seeds_b.sort_unstable();
        assert_eq!(seeds_a, seeds_b);
    }

    #[test]
    fn difference_clears_around_the_other_points() {
        let mut points = scattered(1000);
        let others = vec![Point::new(Vec3::new(15.0, 0.0, 15.0), Vec3::Y, 1)];
        difference(&mut points, &others, 5.0);
        assert!(points.iter().all(|p| p.position.distance(Vec3::new(15.0, 0.0, 15.0)) >= 5.0));
    }
}
