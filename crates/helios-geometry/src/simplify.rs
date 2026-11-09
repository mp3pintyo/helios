//! Quadric error metric (Garland–Heckbert) simplification for cluster-group LOD.
//!
//! * **Half-edge collapses** (`U -> V`) reuse the original vertices, so every LOD level shares
//!   the source vertex data and locked group-border vertices stay bit-identical — the basis of
//!   crack-free cluster LOD.
//! * **Wedges and seams.** Topology is evaluated on *position classes* (all vertices with the
//!   same position). A class can own several *wedges* (vertices that differ in normal or UV,
//!   i.e. lie on a hard edge or UV seam). A collapse is only legal if every wedge of `U` shares
//!   a triangle with exactly one wedge of `V` and no two wedges of `U` map to the same wedge —
//!   which allows collapses *along* a seam and rejects ones that would tear or smear it.
//! * **Open borders** may only collapse along the border, guarded by boundary quadrics.
//! * **Attributes** (normal, UV) add an area-weighted deviation cost that steers the collapse
//!   order; the reported error stays purely geometric (an object-space length).
//! * **Skin weights** add a cost for the weight a collapse moves to other joints: a vertex that
//!   follows the forearm merging into one that follows the upper arm would make the coarse
//!   level bend where the source doesn't. Strong weight seams therefore collapse last, like
//!   attribute seams, and coarse LODs still deform sensibly.
//! * **Aggregate geometry** (leaves, grass, twigs): when collapses run out, whole small
//!   disconnected components are removed — smallest first — and the survivors are scaled up to
//!   preserve total area (stochastic pruning). Their error is the size of what vanished.
//! * Fold-overs and link-condition violations (non-manifold results) are rejected.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::{DVec3, Vec2, Vec3};

use crate::hash::{edge_key, FxHashMap};
use crate::SkinWeights;

/// Symmetric 4×4 quadric plus the accumulated area weight.
#[derive(Clone, Copy, Debug, Default)]
struct Quadric {
    a2: f64,
    ab: f64,
    ac: f64,
    ad: f64,
    b2: f64,
    bc: f64,
    bd: f64,
    c2: f64,
    cd: f64,
    d2: f64,
    weight: f64,
}

impl Quadric {
    fn from_plane(n: DVec3, d: f64, weight: f64) -> Self {
        Self {
            a2: n.x * n.x * weight,
            ab: n.x * n.y * weight,
            ac: n.x * n.z * weight,
            ad: n.x * d * weight,
            b2: n.y * n.y * weight,
            bc: n.y * n.z * weight,
            bd: n.y * d * weight,
            c2: n.z * n.z * weight,
            cd: n.z * d * weight,
            d2: d * d * weight,
            weight,
        }
    }

    fn add(&mut self, o: &Self) {
        self.a2 += o.a2;
        self.ab += o.ab;
        self.ac += o.ac;
        self.ad += o.ad;
        self.b2 += o.b2;
        self.bc += o.bc;
        self.bd += o.bd;
        self.c2 += o.c2;
        self.cd += o.cd;
        self.d2 += o.d2;
        self.weight += o.weight;
    }

    /// Weighted mean squared distance from `p` to the accumulated planes.
    fn error(&self, p: DVec3) -> f64 {
        if self.weight <= 0.0 {
            return 0.0;
        }
        let (x, y, z) = (p.x, p.y, p.z);
        let e = self.a2 * x * x
            + 2.0 * self.ab * x * y
            + 2.0 * self.ac * x * z
            + 2.0 * self.ad * x
            + self.b2 * y * y
            + 2.0 * self.bc * y * z
            + 2.0 * self.bd * y
            + self.c2 * z * z
            + 2.0 * self.cd * z
            + self.d2;
        (e / self.weight).max(0.0)
    }
}

/// Area-weighted attribute statistics of a wedge: `Σw`, `Σw·a`, `Σw·|a|²`.
#[derive(Clone, Copy, Debug, Default)]
struct AttributeMoments {
    weight: f64,
    sum: [f64; 5],
    squares: f64,
}

impl AttributeMoments {
    fn new(a: [f64; 5], weight: f64) -> Self {
        let squares = a.iter().map(|x| x * x).sum::<f64>() * weight;
        Self { weight, sum: a.map(|x| x * weight), squares }
    }

    fn add(&mut self, o: &Self) {
        self.weight += o.weight;
        for i in 0..5 {
            self.sum[i] += o.sum[i];
        }
        self.squares += o.squares;
    }

    /// `Σw·|x - a|²`
    fn error(&self, x: [f64; 5]) -> f64 {
        let xx: f64 = x.iter().map(|v| v * v).sum();
        let xs: f64 = (0..5).map(|i| x[i] * self.sum[i]).sum();
        (self.weight * xx - 2.0 * xs + self.squares).max(0.0)
    }
}

/// Vertex data seen by the simplifier. `canonical[v]` is the representative vertex of `v`'s
/// position class (vertices with bit-identical positions share it).
#[derive(Clone, Copy)]
pub struct SimplifyMesh<'a> {
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2],
    pub canonical: &'a [u32],
    /// Skin weights (skeletal meshes), empty or one per vertex.
    pub skin: &'a [SkinWeights],
}

#[derive(Clone, Copy, Debug)]
pub struct SimplifyOptions {
    /// Never move vertices on open mesh borders.
    pub lock_borders: bool,
    /// Remove small disconnected components when collapses alone cannot reach the target.
    pub prune_components: bool,
    /// Attribute deviation cost, in multiples of the mean edge length per attribute unit.
    pub attribute_weight: f32,
    /// Reported error of component pruning relative to the size of the pruned pieces.
    ///
    /// Removing a few leaves from a canopy whose survivors grow to preserve area changes
    /// coverage stochastically, not the silhouette, so its perceived error is well below the
    /// leaf radius (Cook et al. 2007, "Stochastic Simplification of Aggregate Detail"). At 1.0
    /// a tree keeps every leaf until each one is about a pixel: dense forests then draw
    /// several triangles per pixel. Lower values coarsen aggregates sooner.
    pub prune_error_scale: f32,
    /// Skin weight deviation cost, in multiples of the mean edge length per unit of weight moved
    /// to other joints ([`SkinWeights::distance`]): a collapse across a seam between two joints
    /// costs like moving the surface this many edges.
    pub skin_weight: f32,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self { lock_borders: false, prune_components: true, attribute_weight: 0.5, prune_error_scale: 0.35, skin_weight: 2.0 }
    }
}

/// A vertex created by component pruning: a scaled copy of `source`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewVertex {
    pub source: u32,
    pub position: Vec3,
    /// Index (into [`SimplifyResult::new_vertices`]) of the first new vertex at this position.
    pub class: u32,
}

/// Result of [`simplify`].
#[derive(Clone, Debug, Default)]
pub struct SimplifyResult {
    /// Remaining triangles. Indices `>= positions.len()` refer to `new_vertices`.
    pub triangles: Vec<[u32; 3]>,
    /// Object-space error (RMS distance to the source surface) of the worst operation.
    pub error: f32,
    pub new_vertices: Vec<NewVertex>,
}

#[derive(Clone, Copy, Debug)]
struct Collapse {
    cost: f64,
    from: u32,
    to: u32,
    from_version: u32,
    to_version: u32,
}

impl PartialEq for Collapse {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Collapse {}
impl PartialOrd for Collapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Collapse {
    // Reversed: BinaryHeap is a max-heap, we want the cheapest collapse first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

/// Minimum cosine between a triangle's normal before and after a collapse.
const MAX_NORMAL_DEVIATION_COS: f64 = 0.2;
/// Weight of the planes that keep open borders in place.
const BORDER_WEIGHT: f64 = 8.0;
/// Pruned survivors are scaled up by at most this factor per step.
const MAX_PRUNE_SCALE: f64 = 1.5;

struct State {
    pos: Vec<DVec3>,
    locked: Vec<bool>,
    /// Locked by the caller (group borders): such components must never be pruned. `locked`
    /// additionally contains structural locks (non-manifold edges, border corners).
    externally_locked: Vec<bool>,
    border_only: Vec<bool>,
    quadrics: Vec<Quadric>,
    /// Wedges (local vertices) per class.
    class_wedges: Vec<Vec<u32>>,
    wedge_class: Vec<u32>,
    attributes: Vec<[f64; 5]>,
    moments: Vec<AttributeMoments>,
    tris: Vec<[u32; 3]>,
    tri_alive: Vec<bool>,
    class_tris: Vec<Vec<u32>>,
    class_alive: Vec<bool>,
    version: Vec<u32>,
    border_edges: crate::hash::FxHashSet<u64>,
    attribute_scale: f64,
    /// Skin weights per wedge (empty without skin) and the cost scale of their deviation.
    skin: Vec<SkinWeights>,
    skin_scale: f64,
}

impl State {
    fn tri_classes(&self, t: usize) -> [u32; 3] {
        self.tris[t].map(|w| self.wedge_class[w as usize])
    }

    fn neighbors(&self, c: u32, out: &mut Vec<u32>) {
        out.clear();
        for &t in &self.class_tris[c as usize] {
            if !self.tri_alive[t as usize] {
                continue;
            }
            for x in self.tri_classes(t as usize) {
                if x != c && !out.contains(&x) {
                    out.push(x);
                }
            }
        }
    }

    /// For each wedge of `u`, the wedge of `v` it merges into. `None` if the collapse would
    /// tear or smear a seam.
    fn wedge_mapping(&self, u: u32, v: u32) -> Option<Vec<(u32, u32)>> {
        let mut mapping: Vec<(u32, u32)> = Vec::with_capacity(2);
        for &uw in &self.class_wedges[u as usize] {
            let mut target: Option<u32> = None;
            let mut used = false;
            for &t in &self.class_tris[u as usize] {
                if !self.tri_alive[t as usize] {
                    continue;
                }
                let tri = self.tris[t as usize];
                if !tri.contains(&uw) {
                    continue;
                }
                used = true;
                for &w in &tri {
                    if self.wedge_class[w as usize] == v {
                        match target {
                            None => target = Some(w),
                            Some(x) if x != w => return None,
                            _ => {}
                        }
                    }
                }
            }
            if !used {
                continue; // Dead wedge.
            }
            let target = target?;
            if mapping.iter().any(|&(_, t)| t == target) {
                return None; // Two wedges collapsing into one would erase a seam.
            }
            mapping.push((uw, target));
        }
        if mapping.is_empty() {
            None
        } else {
            Some(mapping)
        }
    }

    fn cost(&self, u: u32, v: u32, mapping: &[(u32, u32)]) -> (f64, f64) {
        let mut q = self.quadrics[u as usize];
        q.add(&self.quadrics[v as usize]);
        let geometric = q.error(self.pos[v as usize]);
        let mut attribute = 0.0;
        let mut weight = 0.0;
        for &(uw, vw) in mapping {
            let mut m = self.moments[uw as usize];
            m.add(&self.moments[vw as usize]);
            attribute += m.error(self.attributes[vw as usize]);
            weight += m.weight;
        }
        let attribute = if weight > 0.0 { attribute / weight } else { 0.0 };
        // The weight the moving wedges hand to other joints (V's skin replaces U's).
        let skin = if self.skin.is_empty() {
            0.0
        } else {
            mapping.iter().map(|&(uw, vw)| self.skin[uw as usize].distance(&self.skin[vw as usize]) as f64).fold(0.0, f64::max)
        };
        (geometric, geometric + attribute * self.attribute_scale + skin * skin * self.skin_scale)
    }

    fn push(&self, heap: &mut BinaryHeap<Collapse>, a: u32, b: u32) {
        for (from, to) in [(a, b), (b, a)] {
            if self.locked[from as usize] {
                continue;
            }
            if self.border_only[from as usize] && !self.border_edges.contains(&edge_key(from, to)) {
                continue;
            }
            let Some(mapping) = self.wedge_mapping(from, to) else { continue };
            let (_, cost) = self.cost(from, to, &mapping);
            heap.push(Collapse {
                cost,
                from,
                to,
                from_version: self.version[from as usize],
                to_version: self.version[to as usize],
            });
        }
    }
}

/// Simplifies `triangles` down to roughly `target_triangles`.
/// Vertices whose position class contains a vertex for which `is_locked` returns true are
/// never moved or removed.
pub fn simplify(
    mesh: SimplifyMesh,
    triangles: &[[u32; 3]],
    is_locked: &dyn Fn(u32) -> bool,
    target_triangles: usize,
    options: &SimplifyOptions,
) -> SimplifyResult {
    // ---- Local indexing: wedges and position classes --------------------------------------------
    let mut wedge_of: FxHashMap<u32, u32> = FxHashMap::default();
    let mut class_of: FxHashMap<u32, u32> = FxHashMap::default();
    let mut global: Vec<u32> = Vec::new();
    let mut wedge_class: Vec<u32> = Vec::new();
    let mut class_canonical: Vec<u32> = Vec::new();
    let tris: Vec<[u32; 3]> = triangles
        .iter()
        .map(|t| {
            t.map(|v| {
                *wedge_of.entry(v).or_insert_with(|| {
                    let canonical = mesh.canonical[v as usize];
                    let next_class = class_canonical.len() as u32;
                    let class = *class_of.entry(canonical).or_insert(next_class);
                    if class == next_class {
                        class_canonical.push(canonical);
                    }
                    global.push(v);
                    wedge_class.push(class);
                    (global.len() - 1) as u32
                })
            })
        })
        .collect();
    let class_count = class_canonical.len();
    let wedge_count = global.len();

    let mut class_wedges = vec![Vec::new(); class_count];
    let mut locked = vec![false; class_count];
    for (w, &c) in wedge_class.iter().enumerate() {
        class_wedges[c as usize].push(w as u32);
        if is_locked(global[w]) {
            locked[c as usize] = true;
        }
    }
    let pos: Vec<DVec3> = class_canonical.iter().map(|&g| mesh.positions[g as usize].as_dvec3()).collect();
    let externally_locked = locked.clone();

    // ---- Topology: class edges, borders and non-manifold edges --------------------------------
    let mut edge_uses: FxHashMap<u64, u32> = FxHashMap::default();
    for t in &tris {
        let c = t.map(|w| wedge_class[w as usize]);
        for (a, b) in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])] {
            *edge_uses.entry(edge_key(a, b)).or_insert(0) += 1;
        }
    }
    let mut border_only = vec![false; class_count];
    let mut border_degree = vec![0u32; class_count];
    let mut border_edges = crate::hash::FxHashSet::default();
    for (&key, &uses) in &edge_uses {
        let (a, b) = ((key & 0xffff_ffff) as usize, (key >> 32) as usize);
        if uses == 1 {
            border_edges.insert(key);
            border_degree[a] += 1;
            border_degree[b] += 1;
        } else if uses > 2 {
            locked[a] = true;
            locked[b] = true;
        }
    }
    for c in 0..class_count {
        if border_degree[c] > 0 {
            if options.lock_borders || border_degree[c] != 2 {
                locked[c] = true;
            } else {
                border_only[c] = true;
            }
        }
    }

    // ---- Quadrics and attribute moments --------------------------------------------------------
    let mut quadrics = vec![Quadric::default(); class_count];
    let mut moments = vec![AttributeMoments::default(); wedge_count];
    let mut class_tris: Vec<Vec<u32>> = vec![Vec::new(); class_count];
    let mut edge_length_sum = 0.0f64;
    for (ti, t) in tris.iter().enumerate() {
        let c = t.map(|w| wedge_class[w as usize]);
        let p = c.map(|i| pos[i as usize]);
        let cross = (p[1] - p[0]).cross(p[2] - p[0]);
        let len = cross.length();
        let area = len * 0.5;
        edge_length_sum += (p[1] - p[0]).length() + (p[2] - p[1]).length() + (p[0] - p[2]).length();
        if len > 1e-30 {
            let normal = cross / len;
            let q = Quadric::from_plane(normal, -normal.dot(p[0]), area);
            for &i in &c {
                quadrics[i as usize].add(&q);
            }
            // Planes through each border edge, perpendicular to the face.
            for k in 0..3 {
                let (a, b) = (c[k], c[(k + 1) % 3]);
                if border_edges.contains(&edge_key(a, b)) {
                    let edge = p[(k + 1) % 3] - p[k];
                    let n = edge.cross(normal).normalize_or_zero();
                    if n != DVec3::ZERO {
                        let bq = Quadric::from_plane(n, -n.dot(p[k]), edge.length_squared() * BORDER_WEIGHT);
                        quadrics[a as usize].add(&bq);
                        quadrics[b as usize].add(&bq);
                    }
                }
            }
        }
        for &w in t {
            let g = global[w as usize] as usize;
            let n = mesh.normals.get(g).copied().unwrap_or(Vec3::Y).as_dvec3();
            let uv = mesh.uvs.get(g).copied().unwrap_or(Vec2::ZERO).as_dvec2();
            moments[w as usize].add(&AttributeMoments::new([n.x, n.y, n.z, uv.x, uv.y], area.max(1e-30)));
        }
        for &i in &c {
            if !class_tris[i as usize].contains(&(ti as u32)) {
                class_tris[i as usize].push(ti as u32);
            }
        }
    }
    let attributes: Vec<[f64; 5]> = global
        .iter()
        .map(|&g| {
            let g = g as usize;
            let n = mesh.normals.get(g).copied().unwrap_or(Vec3::Y).as_dvec3();
            let uv = mesh.uvs.get(g).copied().unwrap_or(Vec2::ZERO).as_dvec2();
            [n.x, n.y, n.z, uv.x, uv.y]
        })
        .collect();
    let mean_edge = edge_length_sum / (tris.len() * 3).max(1) as f64;
    let attribute_scale = (options.attribute_weight as f64 * mean_edge).powi(2);
    let skin: Vec<SkinWeights> = if mesh.skin.is_empty() { Vec::new() } else { global.iter().map(|&g| mesh.skin[g as usize]).collect() };
    let skin_scale = (options.skin_weight as f64 * mean_edge).powi(2);

    let mut s = State {
        pos,
        locked,
        externally_locked,
        border_only,
        quadrics,
        class_wedges,
        wedge_class,
        attributes,
        moments,
        tri_alive: vec![true; tris.len()],
        tris,
        class_tris,
        class_alive: vec![true; class_count],
        version: vec![0; class_count],
        border_edges,
        attribute_scale,
        skin,
        skin_scale,
    };

    let mut heap: BinaryHeap<Collapse> = BinaryHeap::new();
    for &key in edge_uses.keys() {
        s.push(&mut heap, (key & 0xffff_ffff) as u32, (key >> 32) as u32);
    }

    // ---- Aggregate geometry (many tiny components): prune before collapsing --------------------
    let mut alive_count = s.tris.len();
    let mut new_vertices = Vec::new();
    let mut prune_error = 0.0f64;
    if options.prune_components && component_count(&s) * 12 > s.tris.len() {
        prune_error = prune(&mut s, &global, &mut alive_count, target_triangles, &mut new_vertices) * options.prune_error_scale as f64;
    }

    // ---- Collapse loop ---------------------------------------------------------------------------
    let mut max_error = 0.0f64;
    let mut neighbors_u: Vec<u32> = Vec::new();
    let mut neighbors_v: Vec<u32> = Vec::new();

    // Scaled survivors use new wedges the collapse state does not track; they are done.
    let collapse = new_vertices.is_empty();
    while collapse && alive_count > target_triangles {
        let Some(c) = heap.pop() else { break };
        let (u, v) = (c.from as usize, c.to as usize);
        if !s.class_alive[u] || !s.class_alive[v] || s.version[u] != c.from_version || s.version[v] != c.to_version {
            continue;
        }

        // Link condition on position classes.
        s.neighbors(c.from, &mut neighbors_u);
        s.neighbors(c.to, &mut neighbors_v);
        let shared_tris = s.class_tris[u]
            .iter()
            .filter(|&&t| s.tri_alive[t as usize] && s.tri_classes(t as usize).contains(&c.to))
            .count();
        if shared_tris == 0 {
            continue;
        }
        let common = neighbors_u.iter().filter(|x| neighbors_v.contains(x)).count();
        if common != shared_tris {
            continue;
        }
        // Never collapse a component away entirely; pruning removes whole components with
        // proper error and area accounting.
        let survives = |class: usize, other: u32| {
            s.class_tris[class].iter().any(|&t| s.tri_alive[t as usize] && !s.tri_classes(t as usize).contains(&other))
        };
        if !survives(u, c.to) && !survives(v, c.from) {
            continue;
        }
        let Some(mapping) = s.wedge_mapping(c.from, c.to) else { continue };

        // Fold-over check for every triangle that moves.
        let target = s.pos[v];
        let folds = s.class_tris[u].iter().any(|&t| {
            if !s.tri_alive[t as usize] {
                return false;
            }
            let tc = s.tri_classes(t as usize);
            if tc.contains(&c.to) {
                return false;
            }
            let p = tc.map(|i| s.pos[i as usize]);
            let q = tc.map(|i| if i == c.from { target } else { s.pos[i as usize] });
            let before = (p[1] - p[0]).cross(p[2] - p[0]);
            let after = (q[1] - q[0]).cross(q[2] - q[0]);
            let (lb, la) = (before.length(), after.length());
            if la <= 1e-12 * lb.max(1e-30) {
                return true;
            }
            before.dot(after) < MAX_NORMAL_DEVIATION_COS * lb * la
        });
        if folds {
            continue;
        }

        // Apply. Border edges of U now start at V.
        let (geometric, _) = s.cost(c.from, c.to, &mapping);
        if s.border_only[u] {
            for &x in &neighbors_u {
                if x != c.to && s.border_edges.remove(&edge_key(c.from, x)) {
                    s.border_edges.insert(edge_key(c.to, x));
                }
            }
            s.border_edges.remove(&edge_key(c.from, c.to));
        }
        let moved = std::mem::take(&mut s.class_tris[u]);
        for &t in &moved {
            if !s.tri_alive[t as usize] {
                continue;
            }
            if s.tri_classes(t as usize).contains(&c.to) {
                s.tri_alive[t as usize] = false;
                alive_count -= 1;
            } else {
                for w in s.tris[t as usize].iter_mut() {
                    if let Some(&(_, to)) = mapping.iter().find(|&&(from, _)| from == *w) {
                        *w = to;
                    }
                }
                s.class_tris[v].push(t);
            }
        }
        let alive = &s.tri_alive;
        s.class_tris[v].retain(|&t| alive[t as usize]);
        s.class_alive[u] = false;
        let qu = s.quadrics[u];
        s.quadrics[v].add(&qu);
        for &(from, to) in &mapping {
            let m = s.moments[from as usize];
            s.moments[to as usize].add(&m);
        }
        s.version[v] += 1;
        max_error = max_error.max(geometric);

        s.neighbors(c.to, &mut neighbors_v);
        for &w in &neighbors_v {
            s.push(&mut heap, c.to, w);
        }
    }

    let mut error = max_error.sqrt().max(prune_error);
    if alive_count > target_triangles && options.prune_components && new_vertices.is_empty() {
        error = error.max(prune(&mut s, &global, &mut alive_count, target_triangles, &mut new_vertices) * options.prune_error_scale as f64);
    }

    let triangles = s
        .tris
        .iter()
        .zip(&s.tri_alive)
        .filter(|(_, &alive)| alive)
        .map(|(t, _)| t.map(|w| if w >= wedge_count as u32 { w - wedge_count as u32 + mesh.positions.len() as u32 } else { global[w as usize] }))
        .collect();

    SimplifyResult { triangles, error: error as f32, new_vertices }
}

/// Number of connected components (over position classes) of the alive triangles.
fn component_count(s: &State) -> usize {
    let mut parent: Vec<u32> = (0..s.pos.len() as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    let mut components = s.pos.len();
    for t in 0..s.tris.len() {
        if !s.tri_alive[t] {
            continue;
        }
        let c = s.tri_classes(t);
        for k in 1..3 {
            let (a, b) = (find(&mut parent, c[0]), find(&mut parent, c[k]));
            if a != b {
                parent[a as usize] = b;
                components -= 1;
            }
        }
    }
    components
}

/// Removes the smallest unlocked connected components until `target` triangles remain and
/// scales the surviving unlocked components to preserve area. Returns the error introduced.
///
/// New (scaled) vertices are appended to `new_vertices`; triangles are rewritten to refer to
/// them as wedge indices `wedge_count + i` (translated by the caller).
fn prune(
    s: &mut State,
    global: &[u32],
    alive_count: &mut usize,
    target: usize,
    new_vertices: &mut Vec<NewVertex>,
) -> f64 {
    let class_count = s.pos.len();
    // Union-find over classes of alive triangles.
    let mut parent: Vec<u32> = (0..class_count as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for t in 0..s.tris.len() {
        if !s.tri_alive[t] {
            continue;
        }
        let c = s.tri_classes(t);
        for k in 1..3 {
            let (a, b) = (find(&mut parent, c[0]), find(&mut parent, c[k]));
            if a != b {
                parent[a as usize] = b;
            }
        }
    }

    #[derive(Default, Clone)]
    struct Component {
        tris: Vec<u32>,
        locked: bool,
        area: f64,
        centroid: DVec3,
        radius: f64,
    }
    let mut index_of: FxHashMap<u32, usize> = FxHashMap::default();
    let mut components: Vec<Component> = Vec::new();
    for t in 0..s.tris.len() {
        if !s.tri_alive[t] {
            continue;
        }
        let c = s.tri_classes(t);
        let root = find(&mut parent, c[0]);
        let next = components.len();
        let i = *index_of.entry(root).or_insert(next);
        if i == components.len() {
            components.push(Component::default());
        }
        let p = c.map(|x| s.pos[x as usize]);
        let area = (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5;
        let comp = &mut components[i];
        comp.tris.push(t as u32);
        comp.locked |= c.iter().any(|&x| s.externally_locked[x as usize]);
        comp.area += area;
        comp.centroid += (p[0] + p[1] + p[2]) / 3.0 * area.max(1e-30);
    }
    for comp in &mut components {
        comp.centroid /= comp.area.max(1e-30 * comp.tris.len() as f64);
        for &t in &comp.tris {
            for x in s.tri_classes(t as usize) {
                comp.radius = comp.radius.max(s.pos[x as usize].distance(comp.centroid));
            }
        }
    }

    let mut candidates: Vec<usize> = (0..components.len()).filter(|&i| !components[i].locked).collect();
    if candidates.len() < 2 {
        return 0.0;
    }
    candidates.sort_by(|&a, &b| components[a].radius.total_cmp(&components[b].radius).then(a.cmp(&b)));
    let area_before: f64 = candidates.iter().map(|&i| components[i].area).sum();
    let mut removed_area = 0.0;
    let mut error = 0.0f64;
    let mut removed = vec![false; components.len()];
    for &i in &candidates[..candidates.len() - 1] {
        if *alive_count <= target {
            break;
        }
        // Never shrink the area by more than the scale cap can compensate.
        if area_before / (area_before - removed_area - components[i].area).max(1e-30) > MAX_PRUNE_SCALE * MAX_PRUNE_SCALE {
            break;
        }
        removed[i] = true;
        removed_area += components[i].area;
        error = error.max(components[i].radius);
        for &t in &components[i].tris {
            s.tri_alive[t as usize] = false;
        }
        *alive_count -= components[i].tris.len();
    }
    if removed_area <= 0.0 {
        return 0.0;
    }

    // Scale the survivors so the total area is preserved.
    let scale = (area_before / (area_before - removed_area)).sqrt();
    if scale < 1.005 {
        return error;
    }
    let wedge_count = s.wedge_class.len() as u32;
    let mut new_wedge: FxHashMap<u32, u32> = FxHashMap::default();
    let mut class_first: FxHashMap<u32, u32> = FxHashMap::default();
    for &i in &candidates {
        if removed[i] {
            continue;
        }
        let comp = &components[i];
        error = error.max(comp.radius * (scale - 1.0));
        for &t in &comp.tris {
            for k in 0..3 {
                let w = s.tris[t as usize][k];
                let next = new_vertices.len() as u32;
                let nw = *new_wedge.entry(w).or_insert_with(|| {
                    let class = s.wedge_class[w as usize];
                    let p = s.pos[class as usize];
                    let first = *class_first.entry(class).or_insert(next);
                    new_vertices.push(NewVertex {
                        source: global[w as usize],
                        position: (comp.centroid + (p - comp.centroid) * scale).as_vec3(),
                        class: first,
                    });
                    next
                });
                s.tris[t as usize][k] = wedge_count + nw;
            }
        }
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;
    use crate::Mesh;

    fn identity_canonical(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    fn run(mesh: &Mesh, canonical: &[u32], locked: &dyn Fn(u32) -> bool, target: usize) -> SimplifyResult {
        let positions = mesh.positions();
        let normals: Vec<Vec3> = mesh.vertices.iter().map(|v| v.normal()).collect();
        let uvs: Vec<Vec2> = mesh.vertices.iter().map(|v| Vec2::new(v.u, v.v)).collect();
        let tris: Vec<[u32; 3]> = mesh.triangles().collect();
        simplify(
            SimplifyMesh { positions: &positions, normals: &normals, uvs: &uvs, canonical, skin: &[] },
            &tris,
            locked,
            target,
            &SimplifyOptions::default(),
        )
    }

    #[test]
    fn halves_a_sphere_with_small_error() {
        let mesh = primitives::icosphere(4);
        let result = run(&mesh, &identity_canonical(mesh.vertices.len()), &|_| false, mesh.triangle_count() / 2);
        assert!(result.triangles.len() <= mesh.triangle_count() / 2 + 2);
        assert!(result.error > 0.0 && result.error < 0.05, "error {}", result.error);
    }

    #[test]
    fn locked_vertices_are_preserved() {
        let mesh = primitives::icosphere(3);
        let positions = mesh.positions();
        let locked = |v: u32| positions[v as usize].y > 0.0;
        let result = run(&mesh, &identity_canonical(mesh.vertices.len()), &locked, mesh.triangle_count() / 4);
        let used: crate::hash::FxHashSet<u32> = result.triangles.iter().flatten().copied().collect();
        for (i, p) in positions.iter().enumerate() {
            if p.y > 0.0 {
                assert!(used.contains(&(i as u32)), "locked vertex {i} was removed");
            }
        }
    }

    #[test]
    fn uv_seams_stay_closed() {
        // A UV-mapped cylinder: the seam column has two wedges per position.
        let mesh = primitives::cylinder(64, 32, 1.0, 4.0, true);
        let canonical = mesh.canonical_ids();
        let result = run(&mesh, &canonical, &|_| false, mesh.triangle_count() / 4);
        assert!(result.triangles.len() < mesh.triangle_count() / 2, "{} triangles left", result.triangles.len());
        // Every position-level edge must still be shared by exactly two triangles (closed surface).
        let mut uses: FxHashMap<u64, u32> = FxHashMap::default();
        for t in &result.triangles {
            let c = t.map(|v| canonical[v as usize]);
            for (a, b) in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])] {
                *uses.entry(edge_key(a, b)).or_insert(0) += 1;
            }
        }
        assert!(uses.values().all(|&n| n == 2), "seam opened up");
        // Each wedge must keep its own side's UVs: no triangle may mix u≈0 and u≈1 at the seam.
        for t in &result.triangles {
            if t.iter().any(|&v| mesh.vertices[v as usize].normal[1].abs() > 0.5) {
                continue; // cap triangles have their own planar UVs
            }
            let us: Vec<f32> = t.iter().map(|&v| mesh.vertices[v as usize].u).collect();
            let span = us.iter().copied().fold(f32::MIN, f32::max) - us.iter().copied().fold(f32::MAX, f32::min);
            assert!(span < 0.5, "triangle spans the UV seam: {us:?}");
        }
    }

    #[test]
    fn disconnected_leaves_are_pruned_and_area_preserved() {
        let mesh = primitives::leaf_cloud(400, 3);
        let canonical = mesh.canonical_ids();
        let tris = mesh.triangle_count();
        let result = run(&mesh, &canonical, &|_| false, tris / 2);
        assert!(result.triangles.len() <= tris / 2 + 8, "pruning did not reach the target: {}", result.triangles.len());
        assert!(!result.new_vertices.is_empty(), "survivors should be scaled up");
        let area = |t: &[u32; 3]| {
            let p = t.map(|v| {
                if (v as usize) < mesh.vertices.len() {
                    mesh.vertices[v as usize].position()
                } else {
                    result.new_vertices[v as usize - mesh.vertices.len()].position
                }
            });
            (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5
        };
        let before: f32 = mesh.triangles().map(|t| area(&t)).sum();
        let after: f32 = result.triangles.iter().map(area).sum();
        assert!((after / before - 1.0).abs() < 0.15, "area changed from {before} to {after}");
        assert!(result.error > 0.0);
    }
}
