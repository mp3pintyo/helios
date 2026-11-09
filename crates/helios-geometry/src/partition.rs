//! Cluster grouping by multilevel graph partitioning (the METIS scheme Nanite relies on).
//!
//! Clusters become graph nodes; two clusters are connected with a weight equal to the number
//! of (position-level) edges they share. Clusters that touch nothing — leaves, twigs, debris —
//! get weak edges to their nearest spatial neighbours so they still group by locality.
//!
//! The partition is computed by **recursive bisection**, each bisection being multilevel:
//!
//! 1. *Coarsen* by heavy-edge matching until the graph is tiny,
//! 2. *bisect* the coarsest graph by greedy graph growing (best of several seeds),
//! 3. *uncoarsen*, refining the cut at every level with Fiduccia–Mattheyses moves under a
//!    balance constraint.
//!
//! Minimising the cut minimises the locked border of every group, which is what lets the
//! simplifier halve each group without getting stuck.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::Vec3;

use crate::hash::{edge_key, FxHashMap};
use crate::morton::morton_order;

/// Undirected weighted graph in CSR form.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub xadj: Vec<u32>,
    pub adjncy: Vec<u32>,
    pub adjwgt: Vec<u32>,
    pub vwgt: Vec<u32>,
}

impl Graph {
    /// Builds a graph from an undirected edge list. Duplicate edges are merged (weights add).
    pub fn from_edges(node_count: usize, node_weights: Vec<u32>, edges: &[(u32, u32, u32)]) -> Self {
        let mut merged: FxHashMap<u64, u32> = FxHashMap::default();
        for &(a, b, w) in edges {
            if a != b {
                *merged.entry(edge_key(a, b)).or_insert(0) += w;
            }
        }
        let mut degree = vec![0u32; node_count + 1];
        for &key in merged.keys() {
            degree[(key & 0xffff_ffff) as usize + 1] += 1;
            degree[(key >> 32) as usize + 1] += 1;
        }
        for i in 0..node_count {
            degree[i + 1] += degree[i];
        }
        let mut fill = degree.clone();
        let total = degree[node_count] as usize;
        let mut adjncy = vec![0u32; total];
        let mut adjwgt = vec![0u32; total];
        let mut sorted: Vec<(u64, u32)> = merged.into_iter().collect();
        sorted.sort_unstable();
        for (key, w) in sorted {
            let (a, b) = ((key & 0xffff_ffff) as u32, (key >> 32) as u32);
            for (from, to) in [(a, b), (b, a)] {
                let slot = fill[from as usize] as usize;
                adjncy[slot] = to;
                adjwgt[slot] = w;
                fill[from as usize] += 1;
            }
        }
        Self { xadj: degree, adjncy, adjwgt, vwgt: node_weights }
    }

    pub fn node_count(&self) -> usize {
        self.vwgt.len()
    }

    fn neighbors(&self, v: usize) -> impl Iterator<Item = (usize, u32)> + '_ {
        let range = self.xadj[v] as usize..self.xadj[v + 1] as usize;
        self.adjncy[range.clone()].iter().zip(&self.adjwgt[range]).map(|(&n, &w)| (n as usize, w))
    }

    fn total_weight(&self) -> u64 {
        self.vwgt.iter().map(|&w| w as u64).sum()
    }

    /// Sum of the weights of edges crossing between different parts.
    pub fn edge_cut(&self, parts: &[u32]) -> u64 {
        let mut cut = 0u64;
        for v in 0..self.node_count() {
            for (n, w) in self.neighbors(v) {
                if n > v && parts[n] != parts[v] {
                    cut += w as u64;
                }
            }
        }
        cut
    }

    /// Induced subgraph over `nodes` (local index = position in `nodes`).
    fn subgraph(&self, nodes: &[u32], local_of: &mut [u32]) -> Graph {
        for (i, &v) in nodes.iter().enumerate() {
            local_of[v as usize] = i as u32;
        }
        let mut xadj = Vec::with_capacity(nodes.len() + 1);
        let mut adjncy = Vec::new();
        let mut adjwgt = Vec::new();
        xadj.push(0);
        for &v in nodes {
            for (n, w) in self.neighbors(v as usize) {
                let local = local_of[n];
                if local != u32::MAX {
                    adjncy.push(local);
                    adjwgt.push(w);
                }
            }
            xadj.push(adjncy.len() as u32);
        }
        let vwgt = nodes.iter().map(|&v| self.vwgt[v as usize]).collect();
        for &v in nodes {
            local_of[v as usize] = u32::MAX;
        }
        Graph { xadj, adjncy, adjwgt, vwgt }
    }
}

/// Partitions `graph` into parts of at most about `part_weight` total node weight.
/// Returns the part id of every node; ids are dense, `0..part_count`.
pub fn partition_graph(graph: &Graph, part_weight: u32) -> Vec<u32> {
    let n = graph.node_count();
    let mut parts = vec![0u32; n];
    if n == 0 {
        return parts;
    }
    let total = graph.total_weight();
    let k = total.div_ceil(part_weight.max(1) as u64).max(1) as u32;
    let nodes: Vec<u32> = (0..n as u32).collect();
    let mut local_of = vec![u32::MAX; n];
    let mut next_part = 0u32;
    recursive_bisect(graph, &nodes, k, &mut parts, &mut next_part, &mut local_of);
    parts
}

fn recursive_bisect(graph: &Graph, nodes: &[u32], k: u32, parts: &mut [u32], next_part: &mut u32, local_of: &mut [u32]) {
    if k <= 1 || nodes.len() <= 1 {
        for &v in nodes {
            parts[v as usize] = *next_part;
        }
        *next_part += 1;
        return;
    }
    let sub = graph.subgraph(nodes, local_of);
    let k0 = k / 2;
    let side = bisect(&sub, k0 as f64 / k as f64);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for (i, &v) in nodes.iter().enumerate() {
        if side[i] == 0 {
            left.push(v);
        } else {
            right.push(v);
        }
    }
    if left.is_empty() || right.is_empty() {
        // Degenerate split (e.g. a single heavy node): fall back to an ordered split.
        let mid = (nodes.len() as f64 * k0 as f64 / k as f64).round().clamp(1.0, nodes.len() as f64 - 1.0) as usize;
        left = nodes[..mid].to_vec();
        right = nodes[mid..].to_vec();
    }
    recursive_bisect(graph, &left, k0, parts, next_part, local_of);
    recursive_bisect(graph, &right, k - k0, parts, next_part, local_of);
}

/// Multilevel bisection. `fraction` is the target share of node weight on side 0.
fn bisect(graph: &Graph, fraction: f64) -> Vec<u8> {
    const COARSEST: usize = 48;
    let mut levels: Vec<(Graph, Vec<u32>)> = Vec::new(); // (finer graph, fine -> coarse map)
    let mut current = graph.clone();
    while current.node_count() > COARSEST {
        let max_weight = (current.total_weight() as f64 * fraction.min(1.0 - fraction) * 0.5).max(1.0) as u32;
        let (coarse, map) = coarsen(&current, max_weight);
        if coarse.node_count() as f64 > current.node_count() as f64 * 0.95 {
            break; // Matching no longer shrinks the graph.
        }
        levels.push((current, map));
        current = coarse;
    }

    let mut side = initial_bisection(&current, fraction);
    refine(&current, &mut side, fraction);
    while let Some((finer, map)) = levels.pop() {
        side = map.iter().map(|&c| side[c as usize]).collect();
        refine(&finer, &mut side, fraction);
    }
    side
}

/// Heavy-edge matching: every node is merged with its unmatched neighbour of heaviest edge.
fn coarsen(graph: &Graph, max_node_weight: u32) -> (Graph, Vec<u32>) {
    let n = graph.node_count();
    let mut order: Vec<u32> = (0..n as u32).collect();
    // Deterministic shuffle so matching is not biased by input order.
    order.sort_by_key(|&v| v.wrapping_mul(0x9e37_79b9).rotate_left(7));
    let mut mate = vec![u32::MAX; n];
    for &v in &order {
        let v = v as usize;
        if mate[v] != u32::MAX {
            continue;
        }
        let mut best: Option<(usize, u32)> = None;
        for (u, w) in graph.neighbors(v) {
            if mate[u] == u32::MAX && u != v && graph.vwgt[u] + graph.vwgt[v] <= max_node_weight && best.is_none_or(|(_, bw)| w > bw) {
                best = Some((u, w));
            }
        }
        match best {
            Some((u, _)) => {
                mate[v] = u as u32;
                mate[u] = v as u32;
            }
            None => mate[v] = v as u32,
        }
    }

    let mut map = vec![u32::MAX; n];
    let mut vwgt = Vec::new();
    for v in 0..n {
        if map[v] != u32::MAX {
            continue;
        }
        let c = vwgt.len() as u32;
        map[v] = c;
        let u = mate[v] as usize;
        let mut w = graph.vwgt[v];
        if u != v {
            map[u] = c;
            w += graph.vwgt[u];
        }
        vwgt.push(w);
    }

    let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(graph.adjncy.len() / 2);
    for v in 0..n {
        for (u, w) in graph.neighbors(v) {
            if u > v && map[u] != map[v] {
                edges.push((map[v], map[u], w));
            }
        }
    }
    let count = vwgt.len();
    (Graph::from_edges(count, vwgt, &edges), map)
}

/// Greedy graph growing from several seeds; keeps the smallest cut.
fn initial_bisection(graph: &Graph, fraction: f64) -> Vec<u8> {
    let n = graph.node_count();
    let target = (graph.total_weight() as f64 * fraction).round() as u64;
    let mut best: Option<(u64, Vec<u8>)> = None;
    let seeds = n.min(6);
    for s in 0..seeds {
        let seed = (s * n) / seeds;
        let mut side = vec![1u8; n];
        let mut weight = 0u64;
        // Max-heap on "connection to the growing region".
        let mut gain = vec![0i64; n];
        let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
        heap.push((0, Reverse(seed)));
        while weight < target {
            let v = match heap.pop() {
                Some((g, Reverse(v))) => {
                    if side[v] == 0 || g != gain[v] {
                        continue;
                    }
                    v
                }
                // Disconnected remainder: take any node still outside.
                None => match side.iter().position(|&x| x == 1) {
                    Some(v) => v,
                    None => break,
                },
            };
            side[v] = 0;
            weight += graph.vwgt[v] as u64;
            for (u, w) in graph.neighbors(v) {
                if side[u] == 1 {
                    gain[u] += w as i64;
                    heap.push((gain[u], Reverse(u)));
                }
            }
        }
        let parts: Vec<u32> = side.iter().map(|&x| x as u32).collect();
        let cut = graph.edge_cut(&parts);
        if best.as_ref().is_none_or(|(c, _)| cut < *c) {
            best = Some((cut, side));
        }
    }
    best.map(|(_, s)| s).unwrap_or_else(|| vec![0; n])
}

/// Fiduccia–Mattheyses refinement of a bisection with a balance constraint.
fn refine(graph: &Graph, side: &mut [u8], fraction: f64) {
    let n = graph.node_count();
    if n < 2 {
        return;
    }
    let total = graph.total_weight() as f64;
    let max_node = *graph.vwgt.iter().max().unwrap_or(&1) as f64;
    let tolerance = (total * 0.03).max(max_node);
    let targets = [total * fraction, total * (1.0 - fraction)];

    for _pass in 0..6 {
        let mut weights = [0.0f64; 2];
        for v in 0..n {
            weights[side[v] as usize] += graph.vwgt[v] as f64;
        }
        // gain = (edges to other side) - (edges to own side)
        let mut gain = vec![0i64; n];
        for v in 0..n {
            for (u, w) in graph.neighbors(v) {
                gain[v] += if side[u] != side[v] { w as i64 } else { -(w as i64) };
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
        for v in 0..n {
            let boundary = graph.neighbors(v).any(|(u, _)| side[u] != side[v]);
            if boundary {
                heap.push((gain[v], Reverse(v)));
            }
        }
        let mut locked = vec![false; n];
        let mut moves: Vec<usize> = Vec::new();
        let mut cumulative = 0i64;
        let mut best_gain = 0i64;
        let mut best_len = 0usize;
        let imbalance = |w: &[f64; 2]| (w[0] - targets[0]).abs().max((w[1] - targets[1]).abs());
        let mut best_balance = imbalance(&weights);
        let mut stall = 0;

        while let Some((g, Reverse(v))) = heap.pop() {
            if locked[v] || g != gain[v] {
                continue;
            }
            let from = side[v] as usize;
            let to = 1 - from;
            let w = graph.vwgt[v] as f64;
            // Allow the move if it keeps the destination within tolerance, or improves balance.
            let fits = weights[to] + w <= targets[to] + tolerance;
            let improves_balance = weights[from] > targets[from] + tolerance;
            if !fits && !improves_balance {
                continue;
            }
            locked[v] = true;
            side[v] = to as u8;
            weights[from] -= w;
            weights[to] += w;
            cumulative += g;
            moves.push(v);
            for (u, ew) in graph.neighbors(v) {
                if locked[u] {
                    continue;
                }
                // v changed side: edges to u flip between internal and external.
                let delta = 2 * ew as i64;
                gain[u] += if side[u] as usize == to { -delta } else { delta };
                heap.push((gain[u], Reverse(u)));
            }
            let balance = imbalance(&weights);
            let balanced = balance <= tolerance;
            if (cumulative > best_gain && balanced) || (cumulative == best_gain && balance < best_balance) {
                best_gain = cumulative;
                best_len = moves.len();
                best_balance = balance;
                stall = 0;
            } else {
                stall += 1;
                if stall > 64 {
                    break;
                }
            }
        }
        // Roll back everything after the best prefix.
        for &v in &moves[best_len..] {
            side[v] = 1 - side[v];
        }
        if best_len == 0 {
            break;
        }
    }
}

/// Per-cluster constant added to its triangle count when balancing groups. Bounds the number
/// of clusters per group to `group_size * (128 + bias) / (1 + bias)`.
const CLUSTER_WEIGHT_BIAS: u32 = 24;

/// Groups `clusters` (lists of triangles over `canonical` position ids) into groups of
/// about `group_size` clusters. Returns the member cluster indices of every group.
pub fn group_clusters(
    positions: &[Vec3],
    canonical: &[u32],
    clusters: &[&[[u32; 3]]],
    group_size: usize,
) -> Vec<Vec<usize>> {
    let count = clusters.len();
    let total: u32 = clusters.iter().map(|c| c.len() as u32 + CLUSTER_WEIGHT_BIAS).sum();
    if count <= 1 || total <= group_size as u32 * (128 + CLUSTER_WEIGHT_BIAS) {
        return vec![(0..count).collect()];
    }

    // Shared position-level edges between clusters.
    let mut edge_owner: FxHashMap<u64, u32> = FxHashMap::default();
    let mut edges: Vec<(u32, u32, u32)> = Vec::new();
    for (ci, cluster) in clusters.iter().enumerate() {
        for t in cluster.iter() {
            let t = t.map(|v| canonical[v as usize]);
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                match edge_owner.entry(edge_key(a, b)) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(ci as u32);
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let other = *e.get();
                        if other != ci as u32 {
                            edges.push((ci as u32, other, 4));
                        }
                    }
                }
            }
        }
    }

    // Weak locality edges to the nearest clusters (along a Morton window), so that
    // disconnected geometry still groups spatially.
    let centers: Vec<Vec3> = clusters
        .iter()
        .map(|c| {
            let sum: Vec3 = c.iter().flatten().map(|&v| positions[v as usize]).sum();
            sum / (c.len() * 3).max(1) as f32
        })
        .collect();
    let order = morton_order(&centers);
    const WINDOW: usize = 12;
    const NEAREST: usize = 3;
    for (rank, &ci) in order.iter().enumerate() {
        let lo = rank.saturating_sub(WINDOW);
        let hi = (rank + WINDOW + 1).min(count);
        let mut candidates: Vec<(f32, u32)> = order[lo..hi]
            .iter()
            .filter(|&&o| o != ci)
            .map(|&o| (centers[o as usize].distance_squared(centers[ci as usize]), o))
            .collect();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, o) in candidates.iter().take(NEAREST) {
            edges.push((ci, o, 1));
        }
    }

    // Weight clusters by triangle count (plus a constant per cluster), so a group of
    // under-filled clusters still holds a meaningful amount of geometry.
    let weights: Vec<u32> = clusters.iter().map(|c| c.len() as u32 + CLUSTER_WEIGHT_BIAS).collect();
    let graph = Graph::from_edges(count, weights, &edges);
    let parts = partition_graph(&graph, group_size as u32 * (128 + CLUSTER_WEIGHT_BIAS));
    let part_count = parts.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut groups = vec![Vec::new(); part_count];
    for (cluster, &p) in parts.iter().enumerate() {
        groups[p as usize].push(cluster);
    }
    groups.retain(|g| !g.is_empty());
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w × h` grid graph: the optimal 2-way cut of a 16×16 grid is 16 edges.
    fn grid(w: u32, h: u32) -> Graph {
        let mut edges = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    edges.push((i, i + 1, 1));
                }
                if y + 1 < h {
                    edges.push((i, i + w, 1));
                }
            }
        }
        Graph::from_edges((w * h) as usize, vec![1; (w * h) as usize], &edges)
    }

    #[test]
    fn bisection_of_a_grid_is_near_optimal() {
        let g = grid(16, 16);
        let parts = partition_graph(&g, 128);
        let sizes = [0, 1].map(|p| parts.iter().filter(|&&x| x == p).count());
        assert!(sizes.iter().all(|&s| (110..=146).contains(&s)), "unbalanced: {sizes:?}");
        let cut = g.edge_cut(&parts);
        assert!(cut <= 22, "cut {cut} is far from the optimum of 16");
    }

    #[test]
    fn k_way_parts_are_compact() {
        let g = grid(32, 32);
        let parts = partition_graph(&g, 8);
        let count = *parts.iter().max().unwrap() + 1;
        assert_eq!(count, 128);
        let cut = g.edge_cut(&parts);
        // Perfect 2×4 tiles would cut 32*31*2 / ... ≈ 1.5 edges per node; allow some slack.
        assert!(cut < 1024 * 2, "cut {cut}");
        for p in 0..count {
            let size = parts.iter().filter(|&&x| x == p).count();
            assert!((4..=12).contains(&size), "part {p} has {size} nodes");
        }
    }
}
