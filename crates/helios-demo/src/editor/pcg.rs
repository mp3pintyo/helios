//! PCG volumes in the editor: preset graphs, running a volume's graph on its surface, and the
//! generated output in its saved form.

use glam::{Mat4, Vec3};
use helios_pcg::{evaluate, Graph, Heightfield, Node, Op, SpawnEntry, SpawnPart, TransformParams, TriangleSurface, Volume};
use helios_render::{InstanceHandle, Scene};

use super::doc::{PcgOutputDef, PcgVolumeDef, SurfaceDef, Trs};

/// Preset graphs: (id, label).
pub const PRESETS: &[(&str, &str)] = &[("rocks-on-slopes", "Rocks on slopes"), ("flowers-in-meadows", "Flowers in meadows"), ("forest-edge", "Forest edge")];

fn entry(weight: f32, parts: &[(&str, &str)]) -> SpawnEntry {
    SpawnEntry { weight, parts: parts.iter().map(|(mesh, material)| SpawnPart { mesh: (*mesh).into(), material: (*material).into() }).collect() }
}

fn flower(colour: &str, weight: f32) -> SpawnEntry {
    let petals = format!("engine/petals-{colour}");
    entry(weight, &[("engine/flower-stem", "engine/stem"), ("engine/flower-head", petals.as_str())])
}

fn transform(scale: (f32, f32), jitter: f32, tilt_deg: f32, align: f32, offset: (f32, f32)) -> Op {
    Op::Transform(TransformParams {
        scale_min: scale.0,
        scale_max: scale.1,
        scale_jitter: jitter,
        random_yaw: true,
        tilt_deg,
        align_to_normal: align,
        offset_min: offset.0,
        offset_max: offset.1,
    })
}

/// A preset's graph and the size of the volume it is meant for (half extents in metres).
pub fn preset(id: &str, seed: u32) -> Option<(Graph, Vec3)> {
    let nodes = match id {
        // Boulders gather on steeper ground in noisy clusters, half sunk and leaning with the slope.
        "rocks-on-slopes" => vec![
            Node::named("Sample landscape", Op::SurfaceSampler { points_per_m2: 0.25, jitter: 1.0 }),
            Node::named("Slopes 12–55°", Op::SlopeFilter { min_deg: 12.0, max_deg: 55.0 }),
            Node::named("Clusters", Op::NoiseDensity { scale: 18.0, threshold: 0.42, softness: 0.15, invert: false }),
            Node::named("Thin by density", Op::DensityFilter { min: 0.0, max: 1.0, random: true }),
            Node::named("Keep 2.2 m apart", Op::SelfPruning { radius: 2.2 }),
            Node::named("Randomise", transform((0.35, 1.4), 0.25, 15.0, 0.5, (-0.25, -0.05))),
            Node::named("Spawn rocks", Op::Spawner { entries: vec![entry(2.0, &[("engine/rock", "engine/rock")]), entry(1.0, &[("engine/boulder", "engine/rock")])] }),
        ],
        // Patches of wild flowers in grass on flat ground.
        "flowers-in-meadows" => vec![
            Node::named("Sample landscape", Op::SurfaceSampler { points_per_m2: 5.0, jitter: 1.0 }),
            Node::named("Flat ground", Op::SlopeFilter { min_deg: 0.0, max_deg: 18.0 }),
            Node::named("Meadow patches", Op::NoiseDensity { scale: 7.0, threshold: 0.5, softness: 0.12, invert: false }),
            Node::named("Thin by density", Op::DensityFilter { min: 0.0, max: 1.0, random: true }),
            Node::named("Keep 0.2 m apart", Op::SelfPruning { radius: 0.2 }),
            Node::named("Randomise", transform((0.7, 1.3), 0.1, 12.0, 0.3, (0.0, 0.0))),
            Node::named(
                "Spawn flowers and grass",
                Op::Spawner {
                    entries: vec![flower("red", 3.0), flower("yellow", 3.0), flower("blue", 2.0), flower("white", 2.0), entry(5.0, &[("engine/grass", "engine/grass")])],
                },
            ),
        ],
        // Trees thicken away from an edge line across the volume; bushes and ferns line the edge,
        // kept clear of the trunks (a second branch subtracted from the first).
        "forest-edge" => {
            let edge = vec![vec![[-40.0, 0.0, -2.0], [-12.0, 0.0, 3.0], [12.0, 0.0, -3.0], [40.0, 0.0, 2.0]]];
            vec![
                Node::named("Sample landscape", Op::SurfaceSampler { points_per_m2: 0.08, jitter: 1.0 }),
                Node::named("Forest side", Op::BoundsFilter { min: [-1.0, -1.0, -1.0], max: [1.0, 1.0, 0.0], invert: false }),
                Node::named("Thicken inwards", Op::DistanceFilter { points: vec![], splines: edge.clone(), min: 0.0, max: 1000.0, falloff: 18.0, invert: false }),
                Node::named("Thin by density", Op::DensityFilter { min: 0.0, max: 1.0, random: true }),
                Node::named("Keep 5 m apart", Op::SelfPruning { radius: 5.0 }),
                Node::named("Randomise", transform((0.75, 1.2), 0.05, 4.0, 0.0, (-0.2, -0.2))),
                Node::named("Spawn trees", Op::Spawner { entries: vec![entry(3.0, &[("engine/tree-wood", "engine/bark"), ("engine/tree-leaves", "engine/leaves")]), entry(2.0, &[("engine/conifer-wood", "engine/pine-bark"), ("engine/conifer-needles", "engine/needles")])] }),
                Node::named("Sample edge", Op::SurfaceSampler { points_per_m2: 0.5, jitter: 1.0 }),
                Node::named("Along the edge", Op::DistanceFilter { points: vec![], splines: edge, min: 0.0, max: 9.0, falloff: 4.0, invert: false }),
                Node::named("Thin by density", Op::DensityFilter { min: 0.0, max: 1.0, random: true }),
                Node::named("Clear the trunks", Op::Difference { radius: 2.5 }).with_inputs(&[9, 4]),
                Node::named("Keep 1.4 m apart", Op::SelfPruning { radius: 1.4 }),
                Node::named("Randomise", transform((0.6, 1.25), 0.1, 8.0, 0.2, (-0.05, 0.0))),
                Node::named("Spawn undergrowth", Op::Spawner { entries: vec![entry(3.0, &[("engine/bush-wood", "engine/bark"), ("engine/bush-leaves", "engine/leaves")]), entry(2.0, &[("engine/fern", "engine/fern")])] }),
            ]
        }
        _ => return None,
    };
    let half = match id {
        "rocks-on-slopes" => Vec3::new(35.0, 40.0, 35.0),
        "flowers-in-meadows" => Vec3::new(14.0, 20.0, 14.0),
        _ => Vec3::new(40.0, 30.0, 25.0),
    };
    Some((Graph { seed, nodes }, half))
}

/// FNV-1a over the definition's JSON: identifies the output a definition generates.
pub fn definition_hash(def: &PcgVolumeDef) -> String {
    let json = serde_json::to_string(def).expect("PCG definitions serialise");
    let hash = json.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
    format!("{hash:016x}")
}

/// Runs a volume's graph on its surface (`surface`: the instance a mesh surface resolved to).
pub fn generate(def: &PcgVolumeDef, scene: &Scene, floor: fn(f32, f32) -> f32, surface: Option<InstanceHandle>) -> Result<PcgOutputDef, String> {
    let output = match &def.surface {
        SurfaceDef::Landscape => evaluate(&def.graph, &def.volume, &Heightfield { height: floor })?,
        SurfaceDef::Instance { .. } | SurfaceDef::Component { .. } => {
            let handle = surface.ok_or_else(|| format!("the surface {:?} does not exist", def.surface))?;
            let triangles = scene.mesh_triangles(scene.instance_mesh(handle), &scene.transform(handle));
            evaluate(&def.graph, &def.volume, &TriangleSurface { triangles })?
        }
    };
    Ok(PcgOutputDef {
        id: def.id,
        hash: definition_hash(def),
        parts: output.parts.iter().map(|p| [p.mesh.clone(), p.material.clone()]).collect(),
        instances: output.instances.iter().map(|s| (s.part, pack(&s.transform))).collect(),
    })
}

fn pack(m: &Mat4) -> [f32; 10] {
    let trs = Trs::from_mat4(m);
    [trs.t[0], trs.t[1], trs.t[2], trs.r[0], trs.r[1], trs.r[2], trs.r[3], trs.s[0], trs.s[1], trs.s[2]]
}

pub fn unpack(v: &[f32; 10]) -> Mat4 {
    Trs { t: [v[0], v[1], v[2]], r: [v[3], v[4], v[5], v[6]], s: [v[7], v[8], v[9]] }.to_mat4()
}

/// A volume for `preset` centred on `center`, its height fitted around the ground there.
pub fn new_volume(id: u32, preset_id: &str, center: Vec3, floor: fn(f32, f32) -> f32) -> Option<PcgVolumeDef> {
    let (graph, half) = preset(preset_id, id.wrapping_mul(7919))?;
    let label = PRESETS.iter().find(|(p, _)| *p == preset_id).map_or(preset_id, |(_, l)| l);
    let ground = floor(center.x, center.z);
    Some(PcgVolumeDef {
        id,
        name: label.into(),
        preset: preset_id.into(),
        volume: Volume { center: [center.x, ground, center.z], half_extents: half.into(), yaw_deg: 0.0 },
        surface: SurfaceDef::Landscape,
        graph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hills(x: f32, z: f32) -> f32 {
        (x * 0.05).sin() * 9.0 + (z * 0.04).cos() * 7.0
    }

    #[test]
    fn presets_generate_deterministically() {
        let scene = Scene::new();
        for (id, _) in PRESETS {
            let def = new_volume(3, id, Vec3::new(10.0, 0.0, -20.0), hills).unwrap();
            let a = generate(&def, &scene, hills, None).unwrap();
            let b = generate(&def, &scene, hills, None).unwrap();
            assert!(!a.instances.is_empty(), "{id}");
            assert_eq!(a, b, "{id}");
            assert_eq!(a.hash, definition_hash(&def));
        }
    }

    #[test]
    fn packed_transforms_round_trip() {
        let m = Mat4::from_scale_rotation_translation(Vec3::new(1.2, 0.8, 1.1), glam::Quat::from_rotation_y(2.0), Vec3::new(4.0, 5.0, 6.0));
        assert!(unpack(&pack(&m)).abs_diff_eq(m, 1e-5));
    }
}
