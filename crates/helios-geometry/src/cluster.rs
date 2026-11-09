//! Greedy, connectivity-driven cluster ("meshlet") builder.
//!
//! Triangles are visited in Morton order. Each cluster is grown from a seed by repeatedly
//! adding the frontier triangle that introduces the fewest new vertices, breaking ties by
//! distance to the cluster centroid. This produces compact, well-filled clusters, which
//! matters both for culling efficiency and for the quality of group borders later on.
//!
//! When the connected frontier runs dry before the cluster is full (disconnected geometry
//! such as leaves, needles or grass blades), growth continues with the nearest unassigned
//! triangle among the Morton-order neighbours of the seed. Without this, foliage ends up
//! with a handful of triangles per cluster: millions of tiny clusters that overflow the
//! culling lists and waste most of every software-raster workgroup.

use glam::Vec3;

use crate::hash::FxHashMap;
use crate::morton::morton_order;

/// Upper bounds for a single cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterLimits {
    /// Must be ≤ 256 so local vertex indices fit in a byte.
    pub max_vertices: usize,
    /// Must be ≤ 128 so triangle indices fit in 7 bits of the visibility buffer.
    pub max_triangles: usize,
}

impl Default for ClusterLimits {
    fn default() -> Self {
        Self { max_vertices: 128, max_triangles: 128 }
    }
}

/// Splits `triangles` (indices into `positions`) into clusters.
/// `canonical` maps every vertex to its position class (see [`crate::Mesh::canonical_ids`]),
/// so clusters grow across UV seams; the vertex limit counts real vertices (wedges).
/// Returned triangles keep their original (global) vertex indices.
pub fn build_clusters(
    positions: &[Vec3],
    canonical: &[u32],
    triangles: &[[u32; 3]],
    limits: ClusterLimits,
) -> Vec<Vec<[u32; 3]>> {
    assert!(limits.max_vertices >= 3 && limits.max_vertices <= 256);
    assert!(limits.max_triangles >= 1 && limits.max_triangles <= 128);
    if triangles.is_empty() {
        return Vec::new();
    }

    // Compact the vertex range used by this triangle set.
    let mut remap: FxHashMap<u32, u32> = FxHashMap::default();
    let local_tris: Vec<[u32; 3]> = triangles
        .iter()
        .map(|t| {
            t.map(|v| {
                let next = remap.len() as u32;
                *remap.entry(v).or_insert(next)
            })
        })
        .collect();
    let vertex_count = remap.len();

    // Position-class ids (compact) for adjacency, so growth crosses seams.
    let mut class_remap: FxHashMap<u32, u32> = FxHashMap::default();
    let class_tris: Vec<[u32; 3]> = triangles
        .iter()
        .map(|t| {
            t.map(|v| {
                let next = class_remap.len() as u32;
                *class_remap.entry(canonical[v as usize]).or_insert(next)
            })
        })
        .collect();
    let class_count = class_remap.len();

    // Class -> triangle adjacency in CSR form.
    let mut offsets = vec![0u32; class_count + 1];
    for t in &class_tris {
        for &v in t {
            offsets[v as usize + 1] += 1;
        }
    }
    for i in 0..class_count {
        offsets[i + 1] += offsets[i];
    }
    let mut fill = offsets.clone();
    let mut vertex_tris = vec![0u32; class_tris.len() * 3];
    for (ti, t) in class_tris.iter().enumerate() {
        for &v in t {
            vertex_tris[fill[v as usize] as usize] = ti as u32;
            fill[v as usize] += 1;
        }
    }
    let tris_of = |v: u32| &vertex_tris[offsets[v as usize] as usize..offsets[v as usize + 1] as usize];

    let centroids: Vec<Vec3> = triangles
        .iter()
        .map(|t| (positions[t[0] as usize] + positions[t[1] as usize] + positions[t[2] as usize]) / 3.0)
        .collect();
    let order = morton_order(&centroids);
    let mut rank = vec![0u32; order.len()];
    for (r, &t) in order.iter().enumerate() {
        rank[t as usize] = r as u32;
    }
    let (lo, hi) = centroids.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &c| (lo.min(c), hi.max(c)));
    // Typical spacing of clusters if the set were split evenly into full clusters.
    let clusters_estimate = (triangles.len() as f32 / limits.max_triangles as f32).max(1.0);
    let neighbour_radius = (hi - lo).length() / clusters_estimate.cbrt() * 0.75;

    let tri_count = local_tris.len();
    let mut assigned = vec![false; tri_count];
    let mut vertex_stamp = vec![u32::MAX; vertex_count];
    let mut frontier_stamp = vec![u32::MAX; tri_count];
    let mut frontier: Vec<u32> = Vec::new();
    let mut clusters: Vec<Vec<[u32; 3]>> = Vec::new();

    for &seed in &order {
        if assigned[seed as usize] {
            continue;
        }
        let id = clusters.len() as u32;
        let mut members: Vec<u32> = Vec::with_capacity(limits.max_triangles);
        let mut used_vertices = 0usize;
        let mut centroid_sum = Vec3::ZERO;

        frontier.clear();
        frontier.push(seed);
        frontier_stamp[seed as usize] = id;
        let mut radius_limit = f32::MAX;

        while members.len() < limits.max_triangles {
            let center = if members.is_empty() {
                centroids[seed as usize]
            } else {
                centroid_sum / members.len() as f32
            };

            let mut best: Option<(usize, u32, f32)> = None;
            let mut i = 0;
            while i < frontier.len() {
                let t = frontier[i] as usize;
                if assigned[t] {
                    frontier.swap_remove(i);
                    continue;
                }
                let new_vertices =
                    local_tris[t].iter().filter(|&&v| vertex_stamp[v as usize] != id).count() as u32;
                if used_vertices + new_vertices as usize > limits.max_vertices {
                    // The vertex count only grows, so this triangle can never fit again.
                    frontier.swap_remove(i);
                    continue;
                }
                let dist = centroids[t].distance_squared(center);
                let better = match best {
                    None => true,
                    Some((_, bn, bd)) => new_vertices < bn || (new_vertices == bn && dist < bd),
                };
                if better {
                    best = Some((i, new_vertices, dist));
                }
                i += 1;
            }

            let Some((index, _, _)) = best else {
                // Connected growth is exhausted: continue with the nearest spatial neighbour.
                let center = centroid_sum / members.len() as f32;
                if radius_limit == f32::MAX {
                    // Let the cluster reach a few times the extent of what it already holds (at
                    // least the typical cluster spacing), never across the whole mesh.
                    let extent = members.iter().map(|&m| centroids[m as usize].distance(center)).fold(0.0f32, f32::max);
                    radius_limit = (extent * 6.0).max(neighbour_radius);
                }
                match nearest_unassigned(&order, rank[seed as usize] as usize, &assigned, &centroids, center, radius_limit) {
                    Some(t) if used_vertices + 3 <= limits.max_vertices => {
                        frontier_stamp[t as usize] = id;
                        frontier.push(t);
                        continue;
                    }
                    _ => break,
                }
            };
            let t = frontier.swap_remove(index);
            assigned[t as usize] = true;
            members.push(t);
            centroid_sum += centroids[t as usize];

            for &v in &local_tris[t as usize] {
                if vertex_stamp[v as usize] != id {
                    vertex_stamp[v as usize] = id;
                    used_vertices += 1;
                }
            }
            for &v in &class_tris[t as usize] {
                for &n in tris_of(v) {
                    if !assigned[n as usize] && frontier_stamp[n as usize] != id {
                        frontier_stamp[n as usize] = id;
                        frontier.push(n);
                    }
                }
            }
        }

        clusters.push(members.into_iter().map(|t| triangles[t as usize]).collect());
    }

    merge_small_clusters(clusters, canonical, limits)
}

/// How far around the seed (in Morton order) to look for a spatial neighbour.
const NEIGHBOUR_WINDOW: usize = 384;

/// The unassigned triangle nearest to `center` within a Morton-order window around `seed_rank`
/// and within `max_distance`.
fn nearest_unassigned(order: &[u32], seed_rank: usize, assigned: &[bool], centroids: &[Vec3], center: Vec3, max_distance: f32) -> Option<u32> {
    let lo = seed_rank.saturating_sub(NEIGHBOUR_WINDOW);
    let hi = (seed_rank + NEIGHBOUR_WINDOW).min(order.len());
    let limit = max_distance * max_distance;
    let mut best: Option<(u32, f32)> = None;
    for &t in &order[lo..hi] {
        if assigned[t as usize] {
            continue;
        }
        let d = centroids[t as usize].distance_squared(center);
        if d <= limit && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((t, d));
        }
    }
    best.map(|(t, _)| t)
}

/// Folds fragments left over by greedy growth into a neighbouring cluster that still has room.
fn merge_small_clusters(mut clusters: Vec<Vec<[u32; 3]>>, canonical: &[u32], limits: ClusterLimits) -> Vec<Vec<[u32; 3]>> {
    let small = limits.max_triangles / 4;
    if clusters.len() < 2 {
        return clusters;
    }
    let mut owners: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (ci, c) in clusters.iter().enumerate() {
        for &v in c.iter().flatten() {
            let list = owners.entry(canonical[v as usize]).or_default();
            if list.last() != Some(&(ci as u32)) {
                list.push(ci as u32);
            }
        }
    }
    let unique_vertices = |a: &[[u32; 3]], b: &[[u32; 3]]| {
        let mut v: Vec<u32> = a.iter().chain(b).flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    let mut order: Vec<usize> = (0..clusters.len()).filter(|&c| clusters[c].len() < small).collect();
    order.sort_by_key(|&c| clusters[c].len());
    let mut alias: Vec<u32> = (0..clusters.len() as u32).collect();
    let resolve = |alias: &[u32], mut c: u32| {
        while alias[c as usize] != c {
            c = alias[c as usize];
        }
        c
    };
    for c in order {
        if clusters[c].is_empty() || clusters[c].len() >= small {
            continue;
        }
        let mut shared: FxHashMap<u32, u32> = FxHashMap::default();
        for &v in clusters[c].iter().flatten() {
            for &o in &owners[&canonical[v as usize]] {
                let o = resolve(&alias, o);
                if o as usize != c {
                    *shared.entry(o).or_insert(0) += 1;
                }
            }
        }
        let mut candidates: Vec<(u32, u32)> = shared.into_iter().collect();
        candidates.sort_by_key(|&(o, n)| (std::cmp::Reverse(n), o));
        for (o, _) in candidates {
            let o = o as usize;
            if clusters[o].len() + clusters[c].len() <= limits.max_triangles
                && unique_vertices(&clusters[o], &clusters[c]) <= limits.max_vertices
            {
                let moved = std::mem::take(&mut clusters[c]);
                clusters[o].extend(moved);
                alias[c] = o as u32;
                break;
            }
        }
    }
    clusters.retain(|c| !c.is_empty());
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn clusters_respect_limits_and_cover_all_triangles() {
        let mesh = primitives::icosphere(4);
        let positions = mesh.positions();
        let tris: Vec<[u32; 3]> = mesh.triangles().collect();
        let limits = ClusterLimits::default();
        let clusters = build_clusters(&positions, &mesh.canonical_ids(), &tris, limits);

        let total: usize = clusters.iter().map(Vec::len).sum();
        assert_eq!(total, tris.len());
        for c in &clusters {
            assert!(c.len() <= limits.max_triangles);
            let mut verts: Vec<u32> = c.iter().flatten().copied().collect();
            verts.sort_unstable();
            verts.dedup();
            assert!(verts.len() <= limits.max_vertices);
        }
        // Clusters should be well filled on a closed, regular mesh.
        let average = total as f32 / clusters.len() as f32;
        assert!(average > 80.0, "average cluster fill too low: {average}");
    }

    #[test]
    fn disconnected_leaves_fill_clusters_spatially() {
        // Thousands of separate small components: connectivity alone yields a cluster per leaf.
        let mesh = primitives::leaf_cloud(3000, 5);
        let positions = mesh.positions();
        let tris: Vec<[u32; 3]> = mesh.triangles().collect();
        let clusters = build_clusters(&positions, &mesh.canonical_ids(), &tris, ClusterLimits::default());
        let total: usize = clusters.iter().map(Vec::len).sum();
        assert_eq!(total, tris.len());
        let average = total as f32 / clusters.len() as f32;
        assert!(average > 48.0, "foliage clusters too sparse: {average} triangles on average");
        // Spatial filling must stay local: no cluster may span a large part of the cloud.
        let extent = |c: &Vec<[u32; 3]>| {
            let (lo, hi) = c.iter().flatten().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &v| {
                (lo.min(positions[v as usize]), hi.max(positions[v as usize]))
            });
            (hi - lo).length()
        };
        let all = extent(&tris);
        let widest = clusters.iter().map(extent).fold(0.0, f32::max);
        assert!(widest < all * 0.6, "cluster spans {widest} of {all}");
    }
}
