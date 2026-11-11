//! Mesh terrain: ground that is a mesh rather than a heightfield (UE 5.8's experimental Mesh
//! Terrain), so it can hold cliffs, overhangs, arches and tunnels.
//!
//! The terrain is a *density field* — negative inside the ground, positive in the air, close to a
//! signed distance — built from a base heightfield and a non-destructive stack of
//! [`Modifier`]s:
//!
//! * **height modifiers** reshape the heightfield itself (a carved river [`Modifier::Channel`],
//!   a [`Modifier::Flatten`]ed pad), so whatever reads the ground as a heightfield (a water
//!   simulation's bed, the landscape's distance field) sees them;
//! * **volume modifiers** add rock ([`Modifier::Rock`]: cliffs, fins, shelves that overhang) or
//!   remove it ([`Modifier::Tunnel`]: tunnels, the opening of an arch) with smooth CSG, and roughen
//!   their own surfaces with 3D noise and horizontal strata.
//!
//! [`MeshTerrain::build_chunks`] evaluates the field into meshes with **surface nets** (one
//! vertex per cell the surface crosses, at the mean of its edge crossings; one quad per crossed
//! lattice edge), chunk by chunk. Chunks are the unit of culling, streaming and LOD (each one is
//! an ordinary virtual mesh); neighbouring chunks of the same resolution share their border cells
//! and are watertight by construction. Resolution is **adaptive**: chunks near points of interest
//! (the volume modifiers) use the fine cell size, the rest a coarser one; where the two meet, the
//! chunks stop half a cell short and each hangs a *skirt* from its border across the gap,
//! sinking a little (the crack filler of chunked voxel terrain; the two strips overlap under each
//! other), which works because fine zones reach a margin beyond every volume modifier, so
//! transitions only cross heightfield-like ground.
//!
//! The terrain can replace a rectangle of a heightfield *landscape* ([`LandscapeSeam`]): the
//! landscape mesh leaves out its cells in the rectangle, and the mesh terrain follows the
//! landscape's own piecewise-linear surface exactly along the rectangle's edge, then dips a few
//! centimetres below it over a band outside, hidden by the landscape — no crack, no z-fighting,
//! and the landscape keeps its own resolution and material path.
//!
//! [`TerrainCollider`] answers ray casts and "ground below this point" queries against the
//! generated triangles, so placement and the camera respect overhangs (the highest surface at
//! (x, z) is the top of an arch, not the ground under it).

use std::sync::Arc;

use glam::{Vec2, Vec3, Vec3Swizzles};

use crate::{noise, Aabb, Mesh, Vertex};

/// Height of the base landscape at (x, z).
pub type HeightFn = Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>;

/// One operation of the modifier stack (applied in order).
#[derive(Clone, Debug, PartialEq)]
pub enum Modifier {
    /// Sets the ground to `height` within `radius`, blending back over the outer `falloff`.
    Flatten { center: Vec2, radius: f32, falloff: f32, height: f32 },
    /// Carves a channel along a polyline whose points are (x, bed height, z): a flat bed
    /// `width / 2` either side of the line, rising to the original ground over `bank` metres.
    /// It only lowers the ground.
    Channel { path: Vec<Vec3>, width: f32, bank: f32 },
    /// A mass of rock: a rounded box turned by `yaw` about +Y, merged into the ground over
    /// `blend` metres. `taper` narrows its top (positive) or widens it (negative: the faces
    /// overhang); `roughness` is the amplitude (metres) of the 3D noise and strata on its faces.
    Rock { center: Vec3, half_extents: Vec3, yaw: f32, taper: f32, rounding: f32, blend: f32, roughness: f32, seed: u32 },
    /// Removes a capsule of radius `radius` from `a` to `b` (a tunnel, an arch's opening), its
    /// rim rounded over `blend` metres, its walls roughened by `roughness`.
    Tunnel { a: Vec3, b: Vec3, radius: f32, blend: f32, roughness: f32 },
}

impl Modifier {
    fn is_volume(&self) -> bool {
        matches!(self, Modifier::Rock { .. } | Modifier::Tunnel { .. })
    }

    /// World bounds of everything a volume modifier can change (`None` for height modifiers).
    pub fn bounds(&self) -> Option<Aabb> {
        match self {
            Modifier::Rock { center, half_extents, yaw, taper, blend, roughness, .. } => {
                let widen = 1.0 + (-taper).max(0.0);
                let (s, c) = yaw.sin_cos();
                let rx = (c.abs() * half_extents.x + s.abs() * half_extents.z) * widen;
                let rz = (s.abs() * half_extents.x + c.abs() * half_extents.z) * widen;
                let pad = blend + roughness * 1.3;
                Some(Aabb { min: *center - Vec3::new(rx, half_extents.y, rz) - pad, max: *center + Vec3::new(rx, half_extents.y, rz) + pad })
            }
            Modifier::Tunnel { a, b, radius, blend, roughness } => {
                let pad = radius + blend + roughness * 1.6;
                Some(Aabb { min: a.min(*b) - pad, max: a.max(*b) + pad })
            }
            _ => None,
        }
    }
}

/// How a mesh terrain replaces a rectangle of a regular-grid heightfield landscape
/// (`primitives::terrain`'s layout: `resolution` cells of `size / resolution` metres centred on
/// the origin, each split along the diagonal from (x + 1, z) to (x, z + 1)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandscapeSeam {
    /// The landscape's size and cells per side.
    pub size: f32,
    pub resolution: u32,
    /// The rectangle the landscape leaves out (on its grid lines).
    pub hole_min: Vec2,
    pub hole_max: Vec2,
    /// Outside the hole the mesh terrain sinks this far below the landscape over `band` metres.
    pub dip: f32,
    pub band: f32,
}

impl LandscapeSeam {
    pub fn cell(&self) -> f32 {
        self.size / self.resolution as f32
    }

    /// Grid line index range of the hole (cells `x0..x1`, `z0..z1`).
    pub fn hole_cells(&self) -> (u32, u32, u32, u32) {
        let index = |v: f32| ((v + self.size * 0.5) / self.cell()).round().max(0.0) as u32;
        (index(self.hole_min.x), index(self.hole_max.x), index(self.hole_min.y), index(self.hole_max.y))
    }

    /// The landscape mesh's own height at (x, z): its vertices' heights, interpolated on its
    /// triangles. `vertex_height` gives the height at a grid vertex position.
    fn linear_height(&self, x: f32, z: f32, vertex_height: &dyn Fn(f32, f32) -> f32) -> f32 {
        let res = self.resolution as f32;
        let gx = ((x / self.size + 0.5) * res).clamp(0.0, res - 1e-3);
        let gz = ((z / self.size + 0.5) * res).clamp(0.0, res - 1e-3);
        let (ix, iz) = (gx.floor(), gz.floor());
        let (fx, fz) = (gx - ix, gz - iz);
        // Vertex positions exactly as `primitives::terrain` computes them.
        let at = |i: f32, k: f32| {
            let uv = Vec2::new(i, k) / res;
            let p = (uv - 0.5) * self.size;
            vertex_height(p.x, p.y)
        };
        if fx + fz <= 1.0 {
            let (h00, h10, h01) = (at(ix, iz), at(ix + 1.0, iz), at(ix, iz + 1.0));
            h00 + fx * (h10 - h00) + fz * (h01 - h00)
        } else {
            let (h11, h10, h01) = (at(ix + 1.0, iz + 1.0), at(ix + 1.0, iz), at(ix, iz + 1.0));
            h11 + (1.0 - fx) * (h01 - h11) + (1.0 - fz) * (h10 - h11)
        }
    }
}

/// The terrain description: base heightfield, modifier stack and the region to mesh.
#[derive(Clone)]
pub struct MeshTerrain {
    pub base: HeightFn,
    pub modifiers: Vec<Modifier>,
    /// World rectangle (x, z) meshed by [`MeshTerrain::build_chunks`].
    pub region_min: Vec2,
    pub region_max: Vec2,
    pub seam: Option<LandscapeSeam>,
}

/// Meshing parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkConfig {
    /// Cell size near points of interest and elsewhere (`coarse` a multiple of `fine`).
    pub fine: f32,
    pub coarse: f32,
    /// Chunk side in metres (a multiple of `coarse`).
    pub chunk_size: f32,
    /// Fine chunks reach this far beyond every volume modifier's bounds.
    pub margin: f32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self { fine: 0.25, coarse: 0.5, chunk_size: 16.0, margin: 4.0 }
    }
}

/// One meshed chunk.
#[derive(Clone, Debug)]
pub struct TerrainChunk {
    pub mesh: Mesh,
    /// Its (x, z) chunk coordinates and cell size.
    pub coords: (u32, u32),
    pub cell: f32,
    /// Triangles of the skirts hung at resolution changes (counted in `mesh` too).
    pub skirt_triangles: usize,
}

/// Polynomial smooth minimum (Quilez): `min` with the corner rounded over `k`.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    -smin(-a, -b, k)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Distance in (x, z) from `p` to a polyline of (x, y, z) points, and the interpolated y there.
fn polyline_distance(path: &[Vec3], p: Vec2) -> (f32, f32) {
    let mut best = (f32::MAX, path.first().map_or(0.0, |q| q.y));
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (a2, b2) = (a.xz(), b.xz());
        let ab = b2 - a2;
        let t = ((p - a2).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
        let d = (p - (a2 + ab * t)).length();
        if d < best.0 {
            best = (d, a.y + (b.y - a.y) * t);
        }
    }
    best
}

/// Angular frequency (per metre of warped height) of a rock's horizontal strata.
const STRATA_FREQUENCY: f32 = 2.2;

/// A rock's domain-warped position: its relief and strata wander with the warp.
fn rock_warp(p: Vec3, seed: u32) -> Vec3 {
    let w = p * 0.05;
    let warp = Vec3::new(noise::fbm(w, 2, seed + 3), noise::fbm(w + Vec3::splat(17.0), 2, seed + 4), noise::fbm(w + Vec3::splat(31.0), 2, seed + 5));
    p + warp * 6.0
}

/// Rough rock surface: domain-warped 3D noise (large bulges and buttresses, then smaller lumps)
/// plus horizontal strata (ledges) that wander with the warp.
fn rock_relief(p: Vec3, seed: u32) -> f32 {
    let q = rock_warp(p, seed);
    let n = 0.8 * noise::fbm(q * 0.1, 3, seed) + 0.35 * noise::fbm(q * 0.45, 3, seed + 1);
    let strata = (q.y * STRATA_FREQUENCY).sin();
    n + 0.25 * strata * strata * strata
}

/// Signed distance to a rock modifier's rounded, tapered box (before its relief).
fn rock_shape(p: Vec3, center: Vec3, half_extents: Vec3, yaw: f32, taper: f32, rounding: f32) -> f32 {
    let (s, c) = yaw.sin_cos();
    let q = p - center;
    let mut q = Vec3::new(c * q.x - s * q.z, q.y, s * q.x + c * q.z);
    // Taper: the footprint shrinks (or grows) towards the top.
    let t = (q.y / half_extents.y).clamp(-1.0, 1.0) * 0.5 + 0.5;
    let scale = 1.0 - taper * t;
    q.x /= scale;
    q.z /= scale;
    let r = rounding.min(half_extents.min_element());
    let e = q.abs() - (half_extents - r);
    let rock = e.max(Vec3::ZERO).length() + e.x.max(e.y).max(e.z).min(0.0) - r;
    rock * scale.min(1.0)
}

impl MeshTerrain {
    /// Ground height of the base heightfield after the seam and the height modifiers.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let mut h = (self.base)(x, z);
        if let Some(seam) = &self.seam {
            let inside = (x - seam.hole_min.x).min(seam.hole_max.x - x).min(z - seam.hole_min.y).min(seam.hole_max.y - z);
            if inside < seam.cell() {
                let linear = seam.linear_height(x, z, &*self.base);
                h = if inside >= 0.0 {
                    // Follow the landscape's own triangles at the hole's edge, the true surface
                    // a cell inside.
                    linear + (h - linear) * smoothstep(0.0, seam.cell(), inside)
                } else {
                    linear - seam.dip * smoothstep(0.0, seam.band, -inside)
                };
            }
        }
        for m in &self.modifiers {
            match m {
                Modifier::Flatten { center, radius, falloff, height } => {
                    let d = (Vec2::new(x, z) - *center).length();
                    let w = 1.0 - smoothstep(radius - falloff, *radius, d);
                    h += (height - h) * w;
                }
                Modifier::Channel { path, width, bank } => {
                    let (d, bed) = polyline_distance(path, Vec2::new(x, z));
                    if bed < h {
                        let half = width * 0.5;
                        let s = smoothstep(half * 0.5, half + bank, d);
                        h = bed + (h - bed) * s;
                    }
                }
                _ => {}
            }
        }
        h
    }

    /// Density at `p` given the column's height and slope factor (`1 / sqrt(1 + |grad h|^2)`,
    /// which turns the height difference into a distance).
    fn density_with(&self, p: Vec3, h: f32, slope: f32) -> f32 {
        let mut d = (p.y - h) * slope;
        for m in self.modifiers.iter().filter(|m| m.is_volume()) {
            let b = m.bounds().expect("volume modifiers have bounds");
            if p.cmplt(b.min).any() || p.cmpgt(b.max).any() {
                continue;
            }
            match *m {
                Modifier::Rock { center, half_extents, yaw, taper, rounding, blend, roughness, seed } => {
                    let mut rock = rock_shape(p, center, half_extents, yaw, taper, rounding);
                    if roughness > 0.0 {
                        rock += roughness * rock_relief(p, seed);
                    }
                    d = smin(d, rock, blend);
                }
                Modifier::Tunnel { a, b, radius, blend, roughness } => {
                    let ab = b - a;
                    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                    let mut hole = (p - (a + ab * t)).length() - radius;
                    if roughness > 0.0 {
                        hole += roughness * noise::fbm(p * 0.3, 3, 41);
                    }
                    d = smax(d, -hole, blend);
                }
                _ => {}
            }
        }
        d
    }

    /// Density at `p`: negative inside the ground, about the distance to the surface nearby.
    pub fn density(&self, p: Vec3) -> f32 {
        let e = 0.1;
        let h = self.height(p.x, p.z);
        let gx = (self.height(p.x + e, p.z) - self.height(p.x - e, p.z)) / (2.0 * e);
        let gz = (self.height(p.x, p.z + e) - self.height(p.x, p.z - e)) / (2.0 * e);
        self.density_with(p, h, 1.0 / (1.0 + gx * gx + gz * gz).sqrt())
    }

    /// Shading attributes of a surface point at `p` with normal `n`, which the mesh terrain's
    /// material reads from its vertices' UVs:
    ///
    /// * x, the **strata coordinate**: the warped height the rock modifiers cut their strata at,
    ///   in half periods (`q.y · STRATA_FREQUENCY / π`), so the ledges `rock_relief` carves lie
    ///   at its integers and its parity tells the beds apart: from an even integer to the next
    ///   one up the bed is recessed (soft), from an odd one it protrudes (hard). Blended between
    ///   rocks by their distance; plain height (in the same unit) away from them.
    /// * y, the **cavity**: how much the surface is enclosed around `p` (0 open or convex, up to
    ///   1 deep in a crevice), from the density probed along the normal at two distances.
    ///   Only computed near volume modifiers (the heightfield ground is open).
    pub fn shading_attributes(&self, p: Vec3, n: Vec3) -> Vec2 {
        let unit = STRATA_FREQUENCY / std::f32::consts::PI;
        // The ground's own weight: what a rock at 4 m would weigh.
        let mut weight = 1.0 / 18.0;
        let mut strata = p.y * weight;
        let mut near = false;
        for m in &self.modifiers {
            let Some(b) = m.bounds() else { continue };
            if p.cmplt(b.min).any() || p.cmpgt(b.max).any() {
                continue;
            }
            near = true;
            if let Modifier::Rock { center, half_extents, yaw, taper, rounding, blend, roughness, seed } = *m {
                // Fades out towards the bounds, so the coordinate is continuous across them.
                let pad = blend + roughness * 1.3;
                let d = rock_shape(p, center, half_extents, yaw, taper, rounding).max(0.0);
                let w = (1.0 - smoothstep(0.5 * pad, pad, d)) / ((d + 0.25) * (d + 0.25));
                if w > 0.0 {
                    let q = if roughness > 0.0 { rock_warp(p, seed) } else { p };
                    strata += q.y * w;
                    weight += w;
                }
            }
        }
        let cavity = if near {
            let probe = |r: f32| ((r - self.density(p + n * r)) / r).clamp(0.0, 1.0);
            0.5 * (probe(0.5) + probe(1.5))
        } else {
            0.0
        };
        Vec2::new(strata / weight * unit, cavity)
    }

    /// The highest point at or below `p` where the ground's surface lies (the floor under an
    /// arch, the arch's top above it), searching `depth` metres down in `step`s.
    pub fn surface_below(&self, p: Vec3, depth: f32, step: f32) -> Option<f32> {
        let mut prev = self.density(p);
        if prev < 0.0 {
            return None;
        }
        let mut y = p.y;
        while y > p.y - depth {
            let next_y = y - step;
            let d = self.density(Vec3::new(p.x, next_y, p.z));
            if d <= 0.0 {
                let t = prev / (prev - d).max(1e-6);
                return Some(y - step * t);
            }
            prev = d;
            y = next_y;
        }
        None
    }

    /// Bounds of the volume modifiers (the points of interest that get the fine resolution).
    pub fn volume_bounds(&self) -> Vec<Aabb> {
        self.modifiers.iter().filter_map(Modifier::bounds).collect()
    }

    /// Meshes the region chunk by chunk (empty chunks are left out).
    pub fn build_chunks(&self, config: &ChunkConfig) -> Vec<TerrainChunk> {
        let size = self.region_max - self.region_min;
        let chunks_x = (size.x / config.chunk_size).ceil().max(1.0) as u32;
        let chunks_z = (size.y / config.chunk_size).ceil().max(1.0) as u32;
        let pois = self.volume_bounds();
        let fine_chunk = |cx: u32, cz: u32| {
            let lo = self.region_min + Vec2::new(cx as f32, cz as f32) * config.chunk_size - config.margin;
            let hi = lo + config.chunk_size + 2.0 * config.margin;
            pois.iter().any(|b| b.min.x <= hi.x && b.max.x >= lo.x && b.min.z <= hi.y && b.max.z >= lo.y)
        };
        let fine: Vec<bool> = (0..chunks_z).flat_map(|cz| (0..chunks_x).map(move |cx| (cx, cz))).map(|(cx, cz)| fine_chunk(cx, cz)).collect();
        let is_fine = |cx: i64, cz: i64| -> Option<bool> {
            (cx >= 0 && cz >= 0 && cx < chunks_x as i64 && cz < chunks_z as i64).then(|| fine[(cz * chunks_x as i64 + cx) as usize])
        };
        let mut out = Vec::new();
        for cz in 0..chunks_z {
            for cx in 0..chunks_x {
                let me = is_fine(cx as i64, cz as i64).unwrap();
                // A side meets a chunk of the other resolution: stop short and hang a skirt.
                let differs = |dx: i64, dz: i64| is_fine(cx as i64 + dx, cz as i64 + dz).is_some_and(|f| f != me);
                let sides = [differs(-1, 0), differs(1, 0), differs(0, -1), differs(0, 1)];
                let cell = if me { config.fine } else { config.coarse };
                if let Some(chunk) = self.mesh_chunk(cx, cz, cell, config, sides) {
                    out.push(chunk);
                }
            }
        }
        out
    }

    /// Surface nets over one chunk. `short`: sides (-x, +x, -z, +z) that end half a cell inside
    /// the chunk with a skirt.
    fn mesh_chunk(&self, cx: u32, cz: u32, cell: f32, config: &ChunkConfig, short: [bool; 4]) -> Option<TerrainChunk> {
        let per_chunk = (config.chunk_size / cell).round() as i64;
        let total_x = ((self.region_max.x - self.region_min.x) / cell).round() as i64;
        let total_z = ((self.region_max.y - self.region_min.y) / cell).round() as i64;
        let (gx0, gz0) = (cx as i64 * per_chunk, cz as i64 * per_chunk);
        let nx = per_chunk.min(total_x - gx0);
        let nz = per_chunk.min(total_z - gz0);
        if nx <= 0 || nz <= 0 {
            return None;
        }
        // Lattice 0..=n+1 per axis (cells 0..=n), positions from global indices so that
        // neighbouring chunks compute bit-identical samples.
        let (lx, lz) = ((nx + 2) as usize, (nz + 2) as usize);
        let world_x = |i: i64| self.region_min.x + (gx0 + i) as f32 * cell;
        let world_z = |k: i64| self.region_min.y + (gz0 + k) as f32 * cell;
        // Column heights and slope factors (one more ring for the gradient).
        let mut heights = vec![0.0f32; (lx + 2) * (lz + 2)];
        for k in 0..lz + 2 {
            for i in 0..lx + 2 {
                heights[k * (lx + 2) + i] = self.height(world_x(i as i64 - 1), world_z(k as i64 - 1));
            }
        }
        let column = |i: usize, k: usize| heights[(k + 1) * (lx + 2) + i + 1];
        let mut slopes = vec![0.0f32; lx * lz];
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for k in 0..lz {
            for i in 0..lx {
                let gx = (heights[(k + 1) * (lx + 2) + i + 2] - heights[(k + 1) * (lx + 2) + i]) / (2.0 * cell);
                let gz = (heights[(k + 2) * (lx + 2) + i + 1] - heights[k * (lx + 2) + i + 1]) / (2.0 * cell);
                slopes[k * lx + i] = 1.0 / (1.0 + gx * gx + gz * gz).sqrt();
                let h = column(i, k);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        // Volume modifiers over this chunk extend the vertical range.
        let (x_lo, x_hi, z_lo, z_hi) = (world_x(0), world_x(nx + 1), world_z(0), world_z(nz + 1));
        for b in self.volume_bounds() {
            if b.min.x <= x_hi && b.max.x >= x_lo && b.min.z <= z_hi && b.max.z >= z_lo {
                lo = lo.min(b.min.y);
                hi = hi.max(b.max.y);
            }
        }
        let j0 = ((lo / cell).floor() as i64) - 2;
        let j1 = ((hi / cell).ceil() as i64) + 2;
        let ly = (j1 - j0 + 1) as usize;
        let index = |i: usize, j: usize, k: usize| (k * ly + j) * lx + i;
        let mut density = vec![0.0f32; lx * ly * lz];
        for k in 0..lz {
            for j in 0..ly {
                let y = (j0 + j as i64) as f32 * cell;
                for i in 0..lx {
                    let p = Vec3::new(world_x(i as i64), y, world_z(k as i64));
                    density[index(i, j, k)] = self.density_with(p, column(i, k), slopes[k * lx + i]);
                }
            }
        }
        let position = |i: usize, j: usize, k: usize| Vec3::new(world_x(i as i64), (j0 + j as i64) as f32 * cell, world_z(k as i64));

        // Cells (0..=n per horizontal axis) the surface crosses get a vertex.
        let (cxn, cyn, czn) = (lx - 1, ly - 1, lz - 1);
        let mut cell_vertex = vec![u32::MAX; cxn * cyn * czn];
        let cell_index = |i: usize, j: usize, k: usize| (k * cyn + j) * cxn + i;
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut vertex_cells: Vec<(usize, usize)> = Vec::new();
        const CORNERS: [(usize, usize, usize); 8] = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)];
        const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
        for k in 0..czn {
            for j in 0..cyn {
                for i in 0..cxn {
                    let d: [f32; 8] = CORNERS.map(|(a, b, c)| density[index(i + a, j + b, k + c)]);
                    let inside = d.iter().filter(|v| **v < 0.0).count();
                    if inside == 0 || inside == 8 {
                        continue;
                    }
                    let mut sum = Vec3::ZERO;
                    let mut count = 0.0;
                    for (a, b) in EDGES {
                        if (d[a] < 0.0) != (d[b] < 0.0) {
                            let t = d[a] / (d[a] - d[b]);
                            let (pa, pb) = (CORNERS[a], CORNERS[b]);
                            let fa = Vec3::new(pa.0 as f32, pa.1 as f32, pa.2 as f32);
                            let fb = Vec3::new(pb.0 as f32, pb.1 as f32, pb.2 as f32);
                            sum += fa + (fb - fa) * t;
                            count += 1.0;
                        }
                    }
                    let local = sum / count;
                    // Gradient of the cell's trilinear interpolant at the vertex.
                    let (u, v, w) = (local.x, local.y, local.z);
                    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                    let gx = lerp(lerp(d[1] - d[0], d[3] - d[2], v), lerp(d[5] - d[4], d[7] - d[6], v), w);
                    let gy = lerp(lerp(d[2] - d[0], d[3] - d[1], u), lerp(d[6] - d[4], d[7] - d[5], u), w);
                    let gz = lerp(lerp(d[4] - d[0], d[5] - d[1], u), lerp(d[6] - d[2], d[7] - d[3], u), v);
                    let normal = Vec3::new(gx, gy, gz).normalize_or(Vec3::Y);
                    let p = position(i, j, k) + local * cell;
                    cell_vertex[cell_index(i, j, k)] = vertices.len() as u32;
                    vertices.push(Vertex::new(p, normal, Vec2::new(p.x, p.z) / 16.0));
                    vertex_cells.push((i, k));
                }
            }
        }
        if vertices.is_empty() {
            return None;
        }
        // Edge ownership: along an axis cells 0..n-1, across it lattice lines 1..=n (whose quads
        // reach cell n, the neighbour's first cell, which it shares). A short side keeps to the
        // chunk's own cells: its high side gives up lattice line n.
        let n = [nx as usize, nz as usize];
        let along = |axis: usize| 0..n[axis];
        let across = |axis: usize, high: bool| 1..=(n[axis] - high as usize);
        let mut indices = Vec::new();
        let mut quad = |a: u32, b: u32, c: u32, d: u32, flip: bool| {
            if [a, b, c, d].contains(&u32::MAX) {
                return;
            }
            let (b, d) = if flip { (d, b) } else { (b, d) };
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        };
        // Edges along x: lattice (i, j, k) - (i + 1, j, k), cells (i, j-1..j, k-1..k).
        // Unflipped, a quad faces +x, +z and -y for the three edge directions; it must face the
        // air (positive density).
        for k in across(1, short[3]) {
            for j in 1..cyn {
                for i in along(0) {
                    let (d0, d1) = (density[index(i, j, k)], density[index(i + 1, j, k)]);
                    if (d0 < 0.0) == (d1 < 0.0) {
                        continue;
                    }
                    let c = |jj: usize, kk: usize| cell_vertex[cell_index(i, jj, kk)];
                    quad(c(j - 1, k - 1), c(j, k - 1), c(j, k), c(j - 1, k), d0 >= 0.0);
                }
            }
        }
        // Edges along z: lattice (i, j, k) - (i, j, k + 1), cells (i-1..i, j-1..j, k).
        for k in along(1) {
            for j in 1..cyn {
                for i in across(0, short[1]) {
                    let (d0, d1) = (density[index(i, j, k)], density[index(i, j, k + 1)]);
                    if (d0 < 0.0) == (d1 < 0.0) {
                        continue;
                    }
                    let c = |ii: usize, jj: usize| cell_vertex[cell_index(ii, jj, k)];
                    quad(c(i - 1, j - 1), c(i, j - 1), c(i, j), c(i - 1, j), d0 >= 0.0);
                }
            }
        }
        // Edges along y: lattice (i, j, k) - (i, j + 1, k), cells (i-1..i, j, k-1..k).
        for k in across(1, short[3]) {
            for j in 0..cyn {
                for i in across(0, short[1]) {
                    let (d0, d1) = (density[index(i, j, k)], density[index(i, j + 1, k)]);
                    if (d0 < 0.0) == (d1 < 0.0) {
                        continue;
                    }
                    let c = |ii: usize, kk: usize| cell_vertex[cell_index(ii, j, kk)];
                    quad(c(i - 1, k - 1), c(i, k - 1), c(i, k), c(i - 1, k), d0 < 0.0);
                }
            }
        }
        if indices.is_empty() {
            return None;
        }
        // Drop vertices no quad uses (cells of a short side's last ring).
        let mut remap = vec![u32::MAX; vertices.len()];
        let mut kept = Vec::new();
        let mut kept_cells = Vec::new();
        for idx in indices.iter_mut() {
            let v = *idx as usize;
            if remap[v] == u32::MAX {
                remap[v] = kept.len() as u32;
                kept.push(vertices[v]);
                kept_cells.push(vertex_cells[v]);
            }
            *idx = remap[v];
        }
        // Shading attributes in the UVs (the triplanar material needs no texture coordinates).
        for v in &mut kept {
            let a = self.shading_attributes(v.position(), v.normal());
            v.u = a.x;
            v.v = a.y;
        }
        // The gap to a neighbour of the other resolution is half a cell of each.
        let reach = 0.5 * (config.fine + config.coarse) + 0.1;
        let skirt = if short.iter().any(|&s| s) { hang_skirts(&mut kept, &mut indices, &kept_cells, n, short, reach, 0.12) } else { 0 };
        Some(TerrainChunk { mesh: Mesh::new(kept, indices), coords: (cx, cz), cell, skirt_triangles: skirt })
    }
}

/// Hangs a skirt from the open edges along the short sides: a strip reaching `reach` metres
/// outwards (across the gap to the neighbouring chunk of the other resolution, which hangs its
/// own towards this one) while sinking `drop` metres, so the two strips overlap under each other
/// and close the gap from above and from the side. Returns the triangles added.
fn hang_skirts(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, cells: &[(usize, usize)], n: [usize; 2], short: [bool; 4], reach: f32, drop: f32) -> usize {
    use std::collections::{HashMap, HashSet};
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for t in indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert((a, b));
        }
    }
    // A short side's outermost ring of cells (the chunk's own first or last).
    let on_side = |v: u32, side: usize| {
        let (i, k) = cells[v as usize];
        match side {
            0 => i == 0,
            1 => i + 1 == n[0],
            2 => k == 0,
            _ => k + 1 == n[1],
        }
    };
    const OUTWARD: [Vec3; 4] = [Vec3::NEG_X, Vec3::X, Vec3::NEG_Z, Vec3::Z];
    let mut open: Vec<(u32, u32)> = edges.iter().copied().filter(|&(a, b)| !edges.contains(&(b, a))).collect();
    open.sort_unstable();
    let mut lowered: HashMap<(u32, usize), u32> = HashMap::new();
    let mut added = 0;
    for (a, b) in open {
        let Some(side) = (0..4).find(|&s| short[s] && on_side(a, s) && on_side(b, s)) else { continue };
        let mut low = |v: u32| {
            *lowered.entry((v, side)).or_insert_with(|| {
                let mut copy = vertices[v as usize];
                let p = copy.position() + OUTWARD[side] * reach - Vec3::Y * drop;
                copy.position = p.into();
                vertices.push(copy);
                vertices.len() as u32 - 1
            })
        };
        let (la, lb) = (low(a), low(b));
        // The open edge runs a -> b in its triangle; the skirt continues the surface with b -> a.
        indices.extend_from_slice(&[b, a, la, b, la, lb]);
        added += 2;
    }
    added
}

/// Ray casts and ground queries against a terrain's triangles (a uniform grid of triangle
/// buckets over x, z).
pub struct TerrainCollider {
    triangles: Vec<[Vec3; 3]>,
    origin: Vec2,
    cell: f32,
    dims: (usize, usize),
    buckets: Vec<Vec<u32>>,
}

/// A ray cast's hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub distance: f32,
    pub position: Vec3,
    /// Geometric normal of the triangle hit (facing the ray's origin side as built).
    pub normal: Vec3,
}

impl TerrainCollider {
    /// Buckets the triangles of `meshes` into cells of `cell` metres.
    pub fn new<'a>(meshes: impl IntoIterator<Item = &'a Mesh>, cell: f32) -> Self {
        let triangles: Vec<[Vec3; 3]> = meshes
            .into_iter()
            .flat_map(|m| m.triangles().map(move |[a, b, c]| [a, b, c].map(|i| m.vertices[i as usize].position())))
            .collect();
        let bounds = Aabb::from_points(triangles.iter().flatten().copied());
        let origin = bounds.min.xz();
        let extent = (bounds.max.xz() - origin).max(Vec2::splat(cell));
        let dims = ((extent.x / cell).ceil() as usize + 1, (extent.y / cell).ceil() as usize + 1);
        let mut buckets = vec![Vec::new(); dims.0 * dims.1];
        for (t, tri) in triangles.iter().enumerate() {
            let lo = ((tri[0].xz().min(tri[1].xz()).min(tri[2].xz()) - origin) / cell).floor();
            let hi = ((tri[0].xz().max(tri[1].xz()).max(tri[2].xz()) - origin) / cell).floor();
            for z in lo.y as usize..=(hi.y as usize).min(dims.1 - 1) {
                for x in lo.x as usize..=(hi.x as usize).min(dims.0 - 1) {
                    buckets[z * dims.0 + x].push(t as u32);
                }
            }
        }
        Self { triangles, origin, cell, dims, buckets }
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// The world triangles (for surface samplers).
    pub fn triangles(&self) -> &[[Vec3; 3]] {
        &self.triangles
    }

    fn bucket(&self, x: i64, z: i64) -> Option<&[u32]> {
        (x >= 0 && z >= 0 && (x as usize) < self.dims.0 && (z as usize) < self.dims.1).then(|| self.buckets[z as usize * self.dims.0 + x as usize].as_slice())
    }

    /// Möller–Trumbore, both faces.
    fn intersect(&self, t: u32, origin: Vec3, dir: Vec3) -> Option<f32> {
        let [a, b, c] = self.triangles[t as usize];
        let (e1, e2) = (b - a, c - a);
        let pv = dir.cross(e2);
        let det = e1.dot(pv);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let tv = origin - a;
        let u = tv.dot(pv) * inv;
        if !(-1e-6..=1.0 + 1e-6).contains(&u) {
            return None;
        }
        let qv = tv.cross(e1);
        let v = dir.dot(qv) * inv;
        if v < -1e-6 || u + v > 1.0 + 1e-6 {
            return None;
        }
        Some(e2.dot(qv) * inv)
    }

    /// Nearest hit along `dir` (unit) within `max_distance`, walking the grid cells the ray's
    /// (x, z) projection crosses.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_distance: f32) -> Option<RayHit> {
        let mut best: Option<(f32, u32)> = None;
        let test_cell = |x: i64, z: i64, best: &mut Option<(f32, u32)>| {
            if let Some(bucket) = self.bucket(x, z) {
                for &t in bucket {
                    if let Some(d) = self.intersect(t, origin, dir) {
                        if d >= 0.0 && d <= max_distance && best.is_none_or(|(b, _)| d < b) {
                            *best = Some((d, t));
                        }
                    }
                }
            }
        };
        let start = (origin.xz() - self.origin) / self.cell;
        let d2 = dir.xz();
        let (mut x, mut z) = (start.x.floor() as i64, start.y.floor() as i64);
        if d2.length_squared() < 1e-10 {
            test_cell(x, z, &mut best);
        } else {
            // 2D DDA over the cells, until past the best hit or the ray's end.
            let step = (if d2.x > 0.0 { 1 } else { -1 }, if d2.y > 0.0 { 1 } else { -1 });
            let inv = Vec2::new(1.0 / d2.x.abs().max(1e-12), 1.0 / d2.y.abs().max(1e-12)) * self.cell;
            let f = start - start.floor();
            let frac = Vec2::new(if d2.x > 0.0 { 1.0 - f.x } else { f.x }, if d2.y > 0.0 { 1.0 - f.y } else { f.y });
            let mut next = frac * inv;
            let mut travelled = 0.0;
            loop {
                test_cell(x, z, &mut best);
                if travelled > max_distance || best.is_some_and(|(b, _)| b <= travelled) {
                    break;
                }
                if next.x < next.y {
                    travelled = next.x;
                    next.x += inv.x;
                    x += step.0;
                } else {
                    travelled = next.y;
                    next.y += inv.y;
                    z += step.1;
                }
                if x < -1 || z < -1 || x > self.dims.0 as i64 || z > self.dims.1 as i64 {
                    break;
                }
            }
        }
        best.map(|(distance, t)| {
            let [a, b, c] = self.triangles[t as usize];
            RayHit { distance, position: origin + dir * distance, normal: (b - a).cross(c - a).normalize_or(Vec3::Y) }
        })
    }

    /// Height of the highest surface at (x, z) at or below `p.y` (the floor under an arch when
    /// standing beneath it), `None` beyond the terrain.
    pub fn ground_below(&self, p: Vec3) -> Option<f32> {
        let x = ((p.x - self.origin.x) / self.cell).floor() as i64;
        let z = ((p.z - self.origin.y) / self.cell).floor() as i64;
        let bucket = self.bucket(x, z)?;
        let top = p + Vec3::Y * 1e-3;
        bucket.iter().filter_map(|&t| self.intersect(t, top, -Vec3::Y)).filter(|d| *d >= 0.0).fold(None, |best: Option<f32>, d| Some(best.map_or(d, |b| b.min(d)))).map(|d| top.y - d)
    }

    /// Height of the highest surface at (x, z).
    pub fn top(&self, x: f32, z: f32) -> Option<f32> {
        self.ground_below(Vec3::new(x, 1.0e6, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(height: f32) -> HeightFn {
        Arc::new(move |_, _| height)
    }

    fn terrain(modifiers: Vec<Modifier>) -> MeshTerrain {
        MeshTerrain { base: flat(0.0), modifiers, region_min: Vec2::splat(-16.0), region_max: Vec2::splat(16.0), seam: None }
    }

    /// Edges used by one triangle only (the open border).
    fn open_edges(mesh: &Mesh) -> Vec<(Vec3, Vec3)> {
        use std::collections::HashMap;
        let key = |v: Vec3| (v * 1024.0).round().as_ivec3().to_array();
        let mut count: HashMap<([i32; 3], [i32; 3]), i32> = HashMap::new();
        for [a, b, c] in mesh.triangles() {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let (pu, pv) = (key(mesh.vertices[u as usize].position()), key(mesh.vertices[v as usize].position()));
                *count.entry(if pu < pv { (pu, pv) } else { (pv, pu) }).or_default() += 1;
            }
        }
        count.into_iter().filter(|(_, c)| *c == 1).map(|((a, b), _)| (Vec3::from(a.map(|v| v as f32)) / 1024.0, Vec3::from(b.map(|v| v as f32)) / 1024.0)).collect()
    }

    #[test]
    fn flat_ground_meshes_into_a_watertight_plane_across_chunks() {
        let t = MeshTerrain { base: flat(1.3), ..terrain(Vec::new()) };
        let config = ChunkConfig { fine: 0.5, coarse: 1.0, chunk_size: 8.0, margin: 0.0 };
        let chunks = t.build_chunks(&config);
        assert_eq!(chunks.len(), 16);
        let mut all = Mesh::new(Vec::new(), Vec::new());
        for c in &chunks {
            assert!(c.mesh.vertices.iter().all(|v| (v.position[1] - 1.3).abs() < 1e-4 && v.normal[1] > 0.999));
            all.append(&c.mesh, glam::Mat4::IDENTITY);
        }
        // Every border edge lies on the region's outline (half a cell inside): none between chunks.
        for (a, b) in open_edges(&all) {
            let outer = |v: Vec3| v.x.abs() > 15.0 || v.z.abs() > 15.0;
            assert!(outer(a) && outer(b), "inner crack between {a} and {b}");
        }
        // Triangles face up (counter-clockwise seen from above).
        for c in &chunks {
            for [a, b, cc] in c.mesh.triangles() {
                let p = [a, b, cc].map(|i| c.mesh.vertices[i as usize].position());
                assert!((p[1] - p[0]).cross(p[2] - p[0]).y > 0.0);
            }
        }
    }

    #[test]
    fn a_rock_with_a_tunnel_has_an_overhang_and_a_passage() {
        let rock = Modifier::Rock { center: Vec3::new(0.0, 3.0, 0.0), half_extents: Vec3::new(6.0, 3.0, 2.0), yaw: 0.0, taper: 0.0, rounding: 0.5, blend: 0.5, roughness: 0.0, seed: 1 };
        let tunnel = Modifier::Tunnel { a: Vec3::new(0.0, 1.0, -5.0), b: Vec3::new(0.0, 1.0, 5.0), radius: 1.6, blend: 0.3, roughness: 0.0 };
        let t = MeshTerrain { region_min: Vec2::splat(-24.0), region_max: Vec2::splat(24.0), ..terrain(vec![rock, tunnel]) };
        // Solid rock above the passage, air inside it, ground below.
        assert!(t.density(Vec3::new(0.0, 4.5, 0.0)) < 0.0);
        assert!(t.density(Vec3::new(0.0, 1.0, 0.0)) > 0.0);
        assert!(t.density(Vec3::new(0.0, -1.0, 0.0)) < 0.0);
        let chunks = t.build_chunks(&ChunkConfig { fine: 0.25, coarse: 0.5, chunk_size: 8.0, margin: 2.0 });
        let collider = TerrainCollider::new(chunks.iter().map(|c| &c.mesh), 1.0);
        // From above, the top of the rock; from inside the passage, its floor (the ground,
        // carved a little by the tunnel's round bottom).
        let top = collider.top(0.0, 0.0).unwrap();
        assert!((top - 6.0).abs() < 0.3, "top {top}");
        let floor = collider.ground_below(Vec3::new(0.0, 1.5, 0.0)).unwrap();
        assert!(floor < 0.1 && floor > -0.8, "floor {floor}");
        // A horizontal ray through the passage gets through; one beside it hits the rock.
        assert!(collider.raycast(Vec3::new(0.0, 1.2, -12.0), Vec3::Z, 24.0).is_none());
        let hit = collider.raycast(Vec3::new(3.0, 1.2, -12.0), Vec3::Z, 24.0).unwrap();
        assert!((hit.position.z + 2.0).abs() < 0.4, "{hit:?}");
        // The rock's faces are fine chunks, the far ground coarse ones with skirts between.
        assert!(chunks.iter().any(|c| c.cell == 0.25) && chunks.iter().any(|c| c.cell == 0.5));
        assert!(chunks.iter().any(|c| c.skirt_triangles > 0));
        assert!((t.surface_below(Vec3::new(0.0, 1.5, 0.0), 4.0, 0.05).unwrap() - floor).abs() < 0.2);
    }

    #[test]
    fn shading_attributes_follow_the_strata_and_find_crevices() {
        let rock = Modifier::Rock { center: Vec3::new(0.0, 3.0, 0.0), half_extents: Vec3::new(6.0, 3.0, 2.0), yaw: 0.3, taper: -0.1, rounding: 0.5, blend: 0.8, roughness: 0.6, seed: 5 };
        let t = MeshTerrain { region_min: Vec2::splat(-24.0), region_max: Vec2::splat(24.0), ..terrain(vec![rock]) };
        let unit = STRATA_FREQUENCY / std::f32::consts::PI;
        // On the rock's face: the warped height the strata are cut at (the rock dominates).
        let face = Vec3::new(0.0, 3.0, 2.0);
        let a = t.shading_attributes(face, Vec3::Z);
        assert!((a.x - rock_warp(face, 5).y * unit).abs() < 0.1, "{a}");
        // Far from any rock: the plain height in the same unit, and no cavity.
        let open = Vec3::new(20.0, 0.0, 20.0);
        assert_eq!(t.shading_attributes(open, Vec3::Y), Vec2::new(open.y * unit, 0.0));
        // The corner between the face and the ground is enclosed; the top of the rock is not
        // (points moved onto the surface along their normals).
        let on_surface = |mut p: Vec3, n: Vec3| {
            for _ in 0..16 {
                p -= n * t.density(p);
            }
            p
        };
        let diagonal = (Vec3::Y + Vec3::Z).normalize();
        let corner = t.shading_attributes(on_surface(Vec3::new(0.0, 0.3, 3.0), diagonal), diagonal);
        let top = t.shading_attributes(on_surface(Vec3::new(0.0, 6.0, 0.0), Vec3::Y), Vec3::Y);
        assert!(corner.y > 0.2 && top.y < 0.05, "corner {corner}, top {top}");
        // Chunks write them into their vertices' UVs, identically on both sides of a border.
        let chunks = t.build_chunks(&ChunkConfig { fine: 0.25, coarse: 0.5, chunk_size: 8.0, margin: 2.0 });
        use std::collections::HashMap;
        let mut seen: HashMap<[u32; 3], (usize, [f32; 2])> = HashMap::new();
        let mut shared = 0;
        for (i, c) in chunks.iter().enumerate() {
            for v in &c.mesh.vertices {
                assert!((0.0..=1.0).contains(&v.v));
                // The same lattice cell meshed by two chunks gives a bit-identical vertex.
                if let Some((j, uv)) = seen.insert(v.position.map(f32::to_bits), (i, [v.u, v.v])) {
                    if j != i {
                        assert_eq!(uv, [v.u, v.v], "attributes differ across a chunk border at {:?}", v.position);
                        shared += 1;
                    }
                }
            }
        }
        assert!(shared > 0);
    }

    #[test]
    fn a_channel_carves_the_bed_and_keeps_the_banks() {
        let path = vec![Vec3::new(-10.0, -1.0, 0.0), Vec3::new(10.0, -2.0, 0.0)];
        let t = terrain(vec![Modifier::Channel { path, width: 3.0, bank: 2.0 }]);
        assert!((t.height(0.0, 0.0) + 1.5).abs() < 1e-4);
        assert!((t.height(0.0, 0.6) + 1.5).abs() < 1e-4, "flat bed");
        assert_eq!(t.height(0.0, 5.0), 0.0, "untouched beyond the bank");
        let mid = t.height(0.0, 2.5);
        assert!(mid > -1.5 && mid < 0.0);
    }

    #[test]
    fn the_seam_follows_the_landscape_triangles_and_dips_outside() {
        let base: HeightFn = Arc::new(|x, z| 0.3 * x + 0.1 * z * z * 0.01);
        let seam = LandscapeSeam { size: 64.0, resolution: 32, hole_min: Vec2::new(-8.0, -8.0), hole_max: Vec2::new(8.0, 8.0), dip: 0.1, band: 2.0 };
        let t = MeshTerrain { base: base.clone(), modifiers: Vec::new(), region_min: Vec2::splat(-10.0), region_max: Vec2::splat(10.0), seam: Some(seam) };
        // On the hole's edge: the landscape mesh's interpolated height.
        let lin = seam.linear_height(-8.0, 1.3, &*base);
        assert!((t.height(-8.0, 1.3) - lin).abs() < 1e-5);
        // Outside: below it; a cell inside: the true surface.
        assert!(t.height(-9.9, 1.3) < seam.linear_height(-9.9, 1.3, &*base) - 0.09);
        assert!((t.height(-5.0, 1.3) - base(-5.0, 1.3)).abs() < 1e-5);
        assert_eq!(seam.hole_cells(), (12, 20, 12, 20));
    }
}
