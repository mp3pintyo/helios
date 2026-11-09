//! # helios-pcg
//!
//! Procedural content generation in the spirit of Unreal Engine's PCG framework: a small node
//! graph that turns a volume and a surface into scattered, transformed mesh instances.
//!
//! ```text
//! SurfaceSampler ─ SlopeFilter ─ NoiseDensity ─ DensityFilter ─ SelfPruning ─ Transform ─ Spawner
//!                                                                  └─ Difference(trees) ─ ... ─ Spawner
//! ```
//!
//! * **Points** ([`Point`]) carry a position, normal, rotation, scale, density and a seed, as UE's
//!   PCG points do. Samplers create them, filters drop them or scale their density, pruning
//!   thins them to a minimum distance, transform nodes randomise them and spawners turn them
//!   into instances of weighted mesh/material prefabs.
//! * **Graphs** ([`Graph`]) are DAGs: every node reads the points of its inputs (the previous
//!   node by default), so layers can branch, be merged or subtracted from one another. All
//!   spawners contribute to the output.
//! * **Deterministic.** Every random decision is a hash of the graph seed, the node and the
//!   point's own seed — never of evaluation order — and the sampler's seeds come from the world
//!   grid cell a point was born in. The same graph, volume and surface give bit-identical
//!   output, and moving a volume keeps the points in the region it still covers (as UE's
//!   world-space sampling grid does). Nothing depends on hash-map iteration order or threads.
//!
//! Everything is plain data (`serde`), so graphs and their output can be stored in scene files.

mod filter;
mod graph;
mod noise;
mod point;
mod prune;
mod rng;
mod spawn;
mod surface;
mod transform;

pub use graph::{evaluate, Graph, Node, NodeStats, Op, Output};
pub use point::{Point, Volume};
pub use spawn::{SpawnEntry, SpawnPart, Spawned};
pub use surface::{Heightfield, Surface, TriangleSurface};
pub use transform::TransformParams;

/// Upper bound on the points a sampler may create, so a huge volume at a high density fails
/// with a message instead of freezing an editor.
pub const MAX_POINTS: usize = 400_000;
