//! The node graph and its evaluation.

use serde::{Deserialize, Serialize};

use crate::filter::{self, DistanceTargets};
use crate::point::{Point, Volume};
use crate::prune;
use crate::rng::hash_all;
use crate::spawn::{self, SpawnEntry, SpawnPart, Spawned};
use crate::surface::Surface;
use crate::transform::{self, TransformParams};

fn one() -> f32 {
    1.0
}

fn half() -> f32 {
    0.5
}

fn tenth() -> f32 {
    0.1
}

fn yes() -> bool {
    true
}

/// What a node does. Every node but the sampler reads the points of its inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Op {
    /// Points on the surface inside the volume (a source: ignores inputs).
    SurfaceSampler {
        points_per_m2: f32,
        #[serde(default = "one")]
        jitter: f32,
    },
    /// Keeps slopes between the angles (degrees from horizontal ground).
    SlopeFilter { min_deg: f32, max_deg: f32 },
    /// Keeps world heights between `min` and `max` metres.
    HeightFilter { min: f32, max: f32 },
    /// Multiplies density by world-space fractal noise (patches `scale` metres across).
    NoiseDensity {
        scale: f32,
        #[serde(default = "half")]
        threshold: f32,
        #[serde(default = "tenth")]
        softness: f32,
        #[serde(default)]
        invert: bool,
    },
    /// Keeps densities in [min, max], or (`random`) each point with probability = density.
    DensityFilter {
        #[serde(default)]
        min: f32,
        #[serde(default = "one")]
        max: f32,
        #[serde(default)]
        random: bool,
    },
    /// Keeps points within [min, max] metres (in the ground plane) of the given points and
    /// splines, in volume-local metres; `falloff` fades density at the band's edges.
    DistanceFilter {
        #[serde(default)]
        points: Vec<[f32; 3]>,
        #[serde(default)]
        splines: Vec<Vec<[f32; 3]>>,
        min: f32,
        max: f32,
        #[serde(default)]
        falloff: f32,
        #[serde(default)]
        invert: bool,
    },
    /// Keeps points inside a sub-box of the volume (normalised coordinates, [-1, 1]).
    BoundsFilter {
        min: [f32; 3],
        max: [f32; 3],
        #[serde(default)]
        invert: bool,
    },
    /// Poisson-disk thinning: no two points closer than `radius` metres.
    SelfPruning { radius: f32 },
    /// The first input minus every point within `radius` of the other inputs' points.
    Difference { radius: f32 },
    /// The union of the inputs, in input order.
    Merge,
    /// Random scale, rotation and offset.
    Transform(TransformParams),
    /// Spawns a weighted prefab per point (points pass through).
    Spawner { entries: Vec<SpawnEntry> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Display name (optional).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Indices of earlier nodes this one reads; empty: the previous node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<usize>,
    /// A disabled node passes its (first) input through unchanged.
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(flatten)]
    pub op: Op,
}

impl Node {
    pub fn new(op: Op) -> Self {
        Self { name: String::new(), inputs: Vec::new(), enabled: true, op }
    }

    pub fn named(name: &str, op: Op) -> Self {
        Self { name: name.into(), ..Self::new(op) }
    }

    pub fn with_inputs(mut self, inputs: &[usize]) -> Self {
        self.inputs = inputs.to_vec();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub seed: u32,
    pub nodes: Vec<Node>,
}

/// Points leaving each node (for an editor's graph view).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeStats {
    pub points: usize,
}

/// A graph's result: the distinct mesh/material parts and the instances of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Output {
    pub parts: Vec<SpawnPart>,
    pub instances: Vec<Spawned>,
    pub stats: Vec<NodeStats>,
}

impl Output {
    fn part_index(&mut self, part: &SpawnPart) -> u32 {
        match self.parts.iter().position(|p| p == part) {
            Some(i) => i as u32,
            None => {
                self.parts.push(part.clone());
                self.parts.len() as u32 - 1
            }
        }
    }
}

/// Runs `graph` in `volume` on `surface`.
pub fn evaluate(graph: &Graph, volume: &Volume, surface: &dyn Surface) -> Result<Output, String> {
    let mut results: Vec<Vec<Point>> = Vec::with_capacity(graph.nodes.len());
    let mut output = Output::default();
    for (index, node) in graph.nodes.iter().enumerate() {
        let inputs: Vec<usize> = if node.inputs.is_empty() { index.checked_sub(1).into_iter().collect() } else { node.inputs.clone() };
        if let Some(&bad) = inputs.iter().find(|&&i| i >= index) {
            return Err(format!("node {index} reads node {bad}, which does not come before it"));
        }
        let input = |k: usize| inputs.get(k).map(|&i| results[i].clone()).unwrap_or_default();
        let salt = hash_all(&[graph.seed, index as u32, 0x5eed]);
        let points = if !node.enabled {
            input(0)
        } else {
            run(&node.op, input, inputs.len(), volume, surface, graph.seed, salt, &mut output)?
        };
        output.stats.push(NodeStats { points: points.len() });
        results.push(points);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn run(
    op: &Op,
    input: impl Fn(usize) -> Vec<Point>,
    input_count: usize,
    volume: &Volume,
    surface: &dyn Surface,
    seed: u32,
    salt: u32,
    output: &mut Output,
) -> Result<Vec<Point>, String> {
    Ok(match op {
        Op::SurfaceSampler { points_per_m2, jitter } => surface.sample(volume, *points_per_m2, *jitter, hash_all(&[seed]))?,
        Op::SlopeFilter { min_deg, max_deg } => {
            let mut p = input(0);
            filter::slope(&mut p, *min_deg, *max_deg);
            p
        }
        Op::HeightFilter { min, max } => {
            let mut p = input(0);
            filter::height(&mut p, *min, *max);
            p
        }
        Op::NoiseDensity { scale, threshold, softness, invert } => {
            let mut p = input(0);
            filter::noise_density(&mut p, *scale, *threshold, *softness, *invert, salt);
            p
        }
        Op::DensityFilter { min, max, random } => {
            let mut p = input(0);
            filter::density(&mut p, *min, *max, *random, salt);
            p
        }
        Op::DistanceFilter { points, splines, min, max, falloff, invert } => {
            let mut p = input(0);
            filter::distance(&mut p, &DistanceTargets::new(volume, points, splines), *min, *max, *falloff, *invert);
            p
        }
        Op::BoundsFilter { min, max, invert } => {
            let mut p = input(0);
            filter::bounds(&mut p, volume, *min, *max, *invert);
            p
        }
        Op::SelfPruning { radius } => {
            let mut p = input(0);
            prune::self_prune(&mut p, *radius);
            p
        }
        Op::Difference { radius } => {
            let mut p = input(0);
            let others: Vec<Point> = (1..input_count).flat_map(&input).collect();
            prune::difference(&mut p, &others, *radius);
            p
        }
        Op::Merge => (0..input_count).flat_map(&input).collect(),
        Op::Transform(params) => {
            let mut p = input(0);
            transform::apply(&mut p, params, salt);
            p
        }
        Op::Spawner { entries } => {
            let p = input(0);
            for (point, entry) in p.iter().zip(spawn::choose(&p, entries, salt)) {
                let Some(entry) = entry else { continue };
                let transform = spawn::transform_of(point);
                for part in &entries[entry].parts {
                    let part = output.part_index(part);
                    output.instances.push(Spawned { part, transform });
                }
            }
            p
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::Heightfield;
    use glam::Vec3;

    fn hills(x: f32, z: f32) -> f32 {
        (x * 0.05).sin() * 8.0 + (z * 0.07).cos() * 6.0
    }

    fn rocks_on_slopes(seed: u32) -> Graph {
        let part = |m: &str| SpawnPart { mesh: m.into(), material: "rock".into() };
        Graph {
            seed,
            nodes: vec![
                Node::new(Op::SurfaceSampler { points_per_m2: 0.5, jitter: 1.0 }),
                Node::new(Op::SlopeFilter { min_deg: 10.0, max_deg: 60.0 }),
                Node::new(Op::NoiseDensity { scale: 15.0, threshold: 0.45, softness: 0.1, invert: false }),
                Node::new(Op::DensityFilter { min: 0.0, max: 1.0, random: true }),
                Node::new(Op::SelfPruning { radius: 2.0 }),
                Node::new(Op::Transform(TransformParams { scale_min: 0.5, scale_max: 1.5, align_to_normal: 0.6, offset_min: -0.2, offset_max: 0.0, ..Default::default() })),
                Node::new(Op::Spawner {
                    entries: vec![SpawnEntry { weight: 2.0, parts: vec![part("rock a")] }, SpawnEntry { weight: 1.0, parts: vec![part("rock b")] }],
                }),
            ],
        }
    }

    const VOLUME: Volume = Volume { center: [0.0, 0.0, 0.0], half_extents: [60.0, 30.0, 40.0], yaw_deg: 20.0 };

    #[test]
    fn evaluation_is_deterministic() {
        let surface = Heightfield { height: hills };
        let a = evaluate(&rocks_on_slopes(7), &VOLUME, &surface).unwrap();
        let b = evaluate(&rocks_on_slopes(7), &VOLUME, &surface).unwrap();
        assert!(a.instances.len() > 100, "{}", a.instances.len());
        // Bit-identical, not merely close.
        assert_eq!(a, b);
        let bits = |o: &Output| o.instances.iter().flat_map(|s| s.transform.to_cols_array().map(f32::to_bits)).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b));
        let c = evaluate(&rocks_on_slopes(8), &VOLUME, &surface).unwrap();
        assert_ne!(a.instances, c.instances);
    }

    #[test]
    fn moving_the_volume_keeps_the_points_it_still_covers() {
        let surface = Heightfield { height: hills };
        let graph = Graph { seed: 1, nodes: vec![Node::new(Op::SurfaceSampler { points_per_m2: 0.3, jitter: 1.0 })] };
        let a = Volume { yaw_deg: 0.0, ..VOLUME };
        let b = Volume { center: [17.0, 0.0, 9.0], ..a };
        let positions = |v: &Volume| surface.sample(v, 0.3, 1.0, hash_all(&[graph.seed])).unwrap().into_iter().map(|p| (p.seed, p.position.to_array().map(f32::to_bits))).collect::<Vec<_>>();
        let (pa, pb) = (positions(&a), positions(&b));
        let shared: Vec<_> = pa.iter().filter(|p| pb.contains(p)).collect();
        let overlap = pa.iter().filter(|(_, pos)| b.contains(glam::Vec3::from(pos.map(f32::from_bits)))).count();
        assert_eq!(shared.len(), overlap);
        assert!(overlap > 200);
    }

    #[test]
    fn graphs_round_trip_through_json() {
        let graph = rocks_on_slopes(3);
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"type\":\"SlopeFilter\""));
        assert_eq!(serde_json::from_str::<Graph>(&json).unwrap(), graph);
        // Defaults fill in missing fields.
        let short: Graph = serde_json::from_str(r#"{"seed":1,"nodes":[{"type":"SurfaceSampler","points_per_m2":1},{"type":"DensityFilter"}]}"#).unwrap();
        assert_eq!(short.nodes[0].op, Op::SurfaceSampler { points_per_m2: 1.0, jitter: 1.0 });
        assert!(short.nodes[1].enabled);
    }

    #[test]
    fn branches_merge_and_subtract() {
        let surface = Heightfield { height: |_x: f32, _z: f32| 0.0 };
        let volume = Volume { center: [0.0; 3], half_extents: [30.0, 5.0, 30.0], yaw_deg: 0.0 };
        let part = |m: &str| SpawnPart { mesh: m.into(), material: "m".into() };
        let graph = Graph {
            seed: 4,
            nodes: vec![
                Node::new(Op::SurfaceSampler { points_per_m2: 0.5, jitter: 1.0 }),
                Node::new(Op::SelfPruning { radius: 6.0 }),
                Node::new(Op::Spawner { entries: vec![SpawnEntry { weight: 1.0, parts: vec![part("tree"), part("leaves")] }] }),
                Node::new(Op::SelfPruning { radius: 1.0 }).with_inputs(&[0]),
                Node::new(Op::Difference { radius: 2.5 }).with_inputs(&[3, 2]),
                Node::new(Op::Spawner { entries: vec![SpawnEntry { weight: 1.0, parts: vec![part("bush")] }] }),
            ],
        };
        let out = evaluate(&graph, &volume, &surface).unwrap();
        assert_eq!(out.parts.len(), 3);
        let trees: Vec<Vec3> = out.instances.iter().filter(|s| s.part == 0).map(|s| s.transform.w_axis.truncate()).collect();
        let bushes: Vec<Vec3> = out.instances.iter().filter(|s| s.part == 2).map(|s| s.transform.w_axis.truncate()).collect();
        assert_eq!(out.instances.iter().filter(|s| s.part == 1).count(), trees.len());
        assert!(!bushes.is_empty() && !trees.is_empty());
        assert!(bushes.iter().all(|b| trees.iter().all(|t| t.distance(*b) >= 2.5)));
        assert_eq!(out.stats.len(), 6);
    }

    #[test]
    fn disabled_nodes_pass_points_through_and_bad_inputs_fail() {
        let surface = Heightfield { height: hills };
        let mut graph = rocks_on_slopes(2);
        graph.nodes[1].enabled = false;
        let out = evaluate(&graph, &VOLUME, &surface).unwrap();
        assert_eq!(out.stats[0].points, out.stats[1].points);
        graph.nodes[2].inputs = vec![5];
        assert!(evaluate(&graph, &VOLUME, &surface).is_err());
    }
}
