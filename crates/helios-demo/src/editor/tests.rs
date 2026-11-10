use glam::{Mat4, Vec3};
use helios_geometry::{build_virtual_mesh, primitives, BuildConfig, SdfMode};
use helios_pcg::{Graph, Node, Op, SpawnEntry, SpawnPart, Volume};
use helios_render::{Light, Material, RenderSettings, Scene};
use serde_json::json;

use super::ops::EditOp;
use super::*;

fn flat(_x: f32, _z: f32) -> f32 {
    0.0
}

struct Fixture {
    scene: Scene,
    settings: RenderSettings,
    sun: [f32; 3],
}

impl Fixture {
    fn new() -> Self {
        let mut scene = Scene::new();
        let config = BuildConfig { sdf: SdfMode::None, ..Default::default() };
        let rock = scene.add_mesh(build_virtual_mesh(&primitives::rock(1, 1), &config));
        scene.set_mesh_name(rock, "rock");
        let stone = scene.add_material(Material::standard(Vec3::splat(0.4), 0.8, 0.0));
        scene.set_material_name(stone, "stone");
        let gold = scene.add_material(Material::standard(Vec3::new(1.0, 0.8, 0.3), 0.2, 1.0));
        scene.set_material_name(gold, "gold");
        for i in 0..3 {
            scene.add_instance(rock, stone, Mat4::from_translation(Vec3::X * i as f32));
        }
        scene.add_light(Light::point(Vec3::Y * 3.0, Vec3::ONE, 2.0, 10.0));
        Self { scene, settings: RenderSettings::default(), sun: [0.5, 0.4, 10.0] }
    }

    fn world(&mut self) -> World<'_> {
        let [a, e, i] = &mut self.sun;
        World { scene: &mut self.scene, settings: &mut self.settings, sun: [a, e, i], floor: flat }
    }

    fn editor(&mut self) -> Editor {
        Editor::new(&self.scene, BaseScene::Procedural { scene: "test".into(), detail: "medium".into() })
    }
}

fn op(editor: &mut Editor, fixture: &mut Fixture, value: serde_json::Value) -> ops::Created {
    let op: EditOp = serde_json::from_value(value).unwrap();
    editor.apply(op, &mut fixture.world()).unwrap()
}

fn handle(f: &Fixture, i: u32) -> helios_render::InstanceHandle {
    f.scene.instance_handle(i).unwrap()
}

#[test]
fn graphs_materials_and_instance_parameters() {
    use helios_render::material_graph::presets;
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let stone = f.scene.find_material("stone").unwrap();
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Lava", "graph": presets::lava() }));
    let lava = f.scene.find_material_graph("Lava").expect("the graph is added to the scene");
    op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": {
        "graph": "Lava", "parameters": { "Glow Intensity": 30.0 }, "slab": { "clearcoat": 1.0 } } }));
    let m = *f.scene.material(stone);
    assert_eq!(m.graph, Some(lava));
    assert_eq!(m.slab.clearcoat, 1.0);
    assert_eq!(f.scene.material_parameter(stone, "Glow Intensity"), Some(helios_render::ParameterValue::Scalar(30.0)));
    // Editing the graph replaces it in place: the material keeps it.
    let mut edited = presets::lava();
    edited.nodes.retain(|n| n.op != helios_render::material_graph::Op::Noise { scale: 0.9, octaves: 5 });
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Lava", "graph": edited }));
    assert!(!f.scene.material_graph_errors(lava).is_empty(), "the noise's links now dangle");
    assert_eq!(f.scene.material(stone).graph, Some(lava));
    // Undo twice: the graph compiles again, then the material is plain again.
    assert!(editor.undo(&mut f.world()));
    assert!(f.scene.material_graph_errors(lava).is_empty());
    assert!(editor.undo(&mut f.world()));
    assert_eq!(f.scene.material(stone).graph, None);
    assert_eq!(f.scene.material_parameter(stone, "Glow Intensity"), None);
    assert_eq!(f.scene.material(stone).slab.clearcoat, 0.0);
    // Scene files keep graphs and material instances.
    assert!(editor.redo(&mut f.world()));
    let saved = editor.save(CameraDef { position: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0, fov_deg: 60.0 });
    let loaded = SceneDoc::parse(&saved.to_json()).unwrap();
    assert_eq!(loaded.graphs["Lava"], presets::lava());
    assert_eq!(loaded.materials["stone"].graph.as_deref(), Some("Lava"));
}

#[test]
fn materials_get_and_lose_displacement() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let stone = f.scene.find_material("stone").unwrap();
    assert!(f.scene.material(stone).displacement.is_none());
    op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": { "displacement": { "magnitude": 0.1, "center": 0.25 } } }));
    let d = f.scene.material(stone).displacement.expect("displaced by its own texture's height");
    assert_eq!((d.magnitude, d.center, d.channel), (0.1, 0.25, 3));
    op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": { "displacement": { "magnitude": 0.0 } } }));
    assert!(f.scene.material(stone).displacement.is_none());
}

#[test]
fn graphs_drive_displacement_and_are_saved() {
    use helios_render::material_graph::{presets, MaterialGraph};
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let stone = f.scene.find_material("stone").unwrap();
    let bricks = MaterialGraph { name: "Bricks".into(), ..presets::brick_pavement() };
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Bricks", "graph": bricks }));
    op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": { "graph": "Bricks", "displacement": { "magnitude": 0.03, "center": 0.9 } } }));
    let graph = f.scene.material(stone).graph.expect("assigned");
    assert!(f.scene.material_graph_displaces(graph));
    let state = describe::state(&editor, &f.world());
    let entry = state["materials"].as_array().unwrap().iter().find(|m| m["name"] == "stone").unwrap().clone();
    assert_eq!(entry["displacement_graph"], true, "the Details panel shows the height as the graph's");
    assert_eq!((&entry["displacement_mode"], &entry["displacement_animated"]), (&json!("height"), &json!(false)));
    // Unlinking the Displacement pin hands the height back to the texture channel.
    let mut unlinked = bricks.clone();
    unlinked.links.retain(|l| l.to.pin != "Displacement");
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Bricks", "graph": unlinked }));
    assert!(!f.scene.material_graph_displaces(graph));
    // Saved with the scene: the graph's Displacement link and the material's magnitude.
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Bricks", "graph": bricks }));
    let saved = editor.save(CameraDef { position: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0, fov_deg: 60.0 });
    let loaded = SceneDoc::parse(&saved.to_json()).unwrap();
    assert!(loaded.graphs["Bricks"].links.iter().any(|l| l.to.pin == "Displacement"));
    assert_eq!(loaded.materials["stone"].displacement.as_ref().and_then(|d| d.magnitude), Some(0.03));
}

#[test]
fn vector_and_animated_displacement_graphs_round_trip() {
    use helios_render::material_graph::{presets, MaterialGraph};
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let stone = f.scene.find_material("stone").unwrap();
    for (preset, mode, animated, pin) in [(presets::mushrooms(), "vector", false, "VectorDisplacement"), (presets::rippling_cloth(), "height", true, "Displacement")] {
        let graph = MaterialGraph { name: "Displacing".into(), ..preset };
        op(&mut editor, &mut f, json!({ "op": "graph", "name": "Displacing", "graph": graph }));
        op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": { "graph": "Displacing", "displacement": { "magnitude": 0.1, "center": 0.5 } } }));
        assert!(f.scene.material_graph_displaces(f.scene.material(stone).graph.expect("assigned")));
        // The Details panel and the graph editor's status line name the mode.
        let state = describe::state(&editor, &f.world());
        let entry = state["materials"].as_array().unwrap().iter().find(|m| m["name"] == "stone").unwrap();
        assert_eq!((&entry["displacement_mode"], &entry["displacement_animated"]), (&json!(mode), &json!(animated)));
        let listed = state["graphs"].as_array().unwrap().iter().find(|g| g["name"] == "Displacing").unwrap();
        assert_eq!((&listed["displacement"], &listed["displacement_animated"]), (&json!(mode), &json!(animated)));
        // Scene files keep the pin's link and reload to the same graph.
        let saved = editor.save(CameraDef { position: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0, fov_deg: 60.0 });
        let loaded = SceneDoc::parse(&saved.to_json()).unwrap();
        assert_eq!(loaded.graphs["Displacing"], graph);
        assert!(loaded.graphs["Displacing"].links.iter().any(|l| l.to.pin == pin));
    }
}

#[test]
fn speed_bounds_and_vector_spaces_reach_the_editor_and_scene_files() {
    use helios_render::material_graph::{presets, MaterialGraph, Op, VectorSpace};
    let mut f = Fixture::new();
    let mut editor = f.editor();
    // The cloth's speed bound: per material (m/s, its magnitude times the graph's bound) and per
    // graph (units of the magnitude per second).
    let cloth = MaterialGraph { name: "Cloth".into(), ..presets::rippling_cloth() };
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Cloth", "graph": cloth }));
    op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": { "graph": "Cloth", "displacement": { "magnitude": 0.2, "center": 0.5 } } }));
    let state = describe::state(&editor, &f.world());
    let entry = state["materials"].as_array().unwrap().iter().find(|m| m["name"] == "stone").unwrap();
    let graph = state["graphs"].as_array().unwrap().iter().find(|g| g["name"] == "Cloth").unwrap();
    let (per_material, per_graph) = (entry["displacement_speed"].as_f64().unwrap(), graph["displacement_speed"].as_f64().unwrap());
    assert!(per_graph > 0.0 && (per_material - per_graph * 0.2).abs() < 1e-5, "{per_material} {per_graph}");
    // A tangent-space vector graph: its TransformVector node and spaces survive a scene file.
    let tufts = MaterialGraph { name: "Tufts".into(), ..presets::combed_tufts() };
    op(&mut editor, &mut f, json!({ "op": "graph", "name": "Tufts", "graph": tufts }));
    op(&mut editor, &mut f, json!({ "op": "material", "name": "stone", "params": { "graph": "Tufts", "displacement": { "magnitude": 0.1, "center": 0.0 } } }));
    let state = describe::state(&editor, &f.world());
    let entry = state["materials"].as_array().unwrap().iter().find(|m| m["name"] == "stone").unwrap();
    assert_eq!((&entry["displacement_mode"], &entry["displacement_speed"]), (&json!("vector"), &json!(0.0)));
    let saved = editor.save(CameraDef { position: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0, fov_deg: 60.0 });
    let text = saved.to_json();
    assert!(text.contains("\"TransformVector\"") && text.contains("\"Tangent\""), "spaces are saved by name");
    let loaded = SceneDoc::parse(&text).unwrap();
    assert_eq!(loaded.graphs["Tufts"], tufts);
    assert!(loaded.graphs["Tufts"].nodes.iter().any(|n| n.op == Op::TransformVector { source: VectorSpace::Tangent, destination: VectorSpace::World }));
    // The node palette (heliosMaterialNodes) offers it, Object → World by default.
    let catalog = serde_json::to_value(helios_render::material_graph::catalog()).unwrap();
    let entry = catalog.as_array().unwrap().iter().find(|n| n["type"] == "TransformVector").expect("in the palette");
    assert_eq!((&entry["template"]["source"], &entry["template"]["destination"]), (&json!("Object"), &json!("World")));
}

#[test]
fn transforms_undo_and_redo() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let moved = Trs::from_mat4(&Mat4::from_translation(Vec3::new(5.0, 1.0, 0.0)));
    op(&mut editor, &mut f, json!({ "op": "transform", "instances": [1], "transforms": [moved] }));
    assert_eq!(f.scene.transform(handle(&f, 1)), moved.to_mat4());
    assert!(editor.undo(&mut f.world()));
    assert_eq!(f.scene.transform(handle(&f, 1)), Mat4::from_translation(Vec3::X));
    assert!(editor.redo(&mut f.world()));
    assert_eq!(f.scene.transform(handle(&f, 1)), moved.to_mat4());
    assert_eq!(editor.doc.instances[&1].mesh, "rock");
}

#[test]
fn deleted_instances_hide_and_their_slots_are_reused() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    op(&mut editor, &mut f, json!({ "op": "delete", "instances": [0] }));
    assert!(!f.scene.is_visible(handle(&f, 0)));
    let created = op(&mut editor, &mut f, json!({ "op": "duplicate", "instances": [2], "offset": [0.0, 0.0, 4.0] }));
    assert_eq!(created.instances, vec![3]);
    assert_eq!(f.scene.transform(handle(&f, 3)), Mat4::from_translation(Vec3::new(2.0, 0.0, 4.0)));
    assert_eq!(editor.owner(3), Owner::Added(editor.doc.added[0].id));
    // Deleting the copy hides it; the next copy of the same mesh reuses its slot.
    op(&mut editor, &mut f, json!({ "op": "delete", "instances": [3] }));
    assert!(!f.scene.is_visible(handle(&f, 3)));
    let again = op(&mut editor, &mut f, json!({ "op": "duplicate", "instances": [1], "offset": [0.0, 1.0, 0.0] }));
    assert_eq!(again.instances, vec![3]);
    assert_eq!(f.scene.instance_count(), 4);
    // Undo everything: back to the base scene.
    while editor.undo(&mut f.world()) {}
    assert!((0..3).all(|i| f.scene.is_visible(handle(&f, i))));
    assert!(!f.scene.is_visible(handle(&f, 3)));
}

#[test]
fn material_overrides_revert_to_the_base_values() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let gold = f.scene.find_material("gold").unwrap();
    op(&mut editor, &mut f, json!({ "op": "material", "name": "gold", "params": { "roughness": 0.7, "base_color": [0.1, 0.2, 0.3] } }));
    assert_eq!(f.scene.material(gold).roughness, 0.7);
    assert_eq!(f.scene.material(gold).metallic, 1.0);
    op(&mut editor, &mut f, json!({ "op": "assign_material", "instances": [2], "material": "gold" }));
    assert_eq!(f.scene.instance_material(handle(&f, 2)), gold);
    editor.undo(&mut f.world());
    editor.undo(&mut f.world());
    assert_eq!(f.scene.material(gold).roughness, 0.2);
    assert_ne!(f.scene.instance_material(handle(&f, 2)), gold);
}

#[test]
fn lights_environment_and_settings_round_trip() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let mut def = light_def(&f.scene.lights()[0]);
    def.intensity = 7.0;
    op(&mut editor, &mut f, json!({ "op": "light", "light": 0, "def": def }));
    assert_eq!(f.scene.lights()[0].intensity, 7.0);
    let created = op(&mut editor, &mut f, json!({ "op": "add_light", "def": def }));
    assert_eq!(created.lights, vec![1]);
    op(&mut editor, &mut f, json!({ "op": "delete", "lights": [1] }));
    assert_eq!(f.scene.lights()[1].intensity, 0.0);
    let mut env = Editor::current_environment(&f.world());
    env.sun_elevation_deg = 10.0;
    env.fog.density = 0.02;
    op(&mut editor, &mut f, json!({ "op": "environment", "env": env }));
    assert!((f.sun[1] - 10f32.to_radians()).abs() < 1e-6 && f.scene.fog.density == 0.02);
    op(&mut editor, &mut f, json!({ "op": "render", "key": "global_illumination", "value": false }));
    assert!(!f.settings.global_illumination);
    while editor.undo(&mut f.world()) {}
    assert!(f.settings.global_illumination);
    assert_eq!(f.scene.lights()[0].intensity, 2.0);
    assert_eq!(f.sun[1], 0.4);
}

fn scatter_volume(id: u32) -> PcgVolumeDef {
    let part = |material: &str| SpawnPart { mesh: "rock".into(), material: material.into() };
    PcgVolumeDef {
        id,
        name: "scatter".into(),
        preset: String::new(),
        volume: Volume { center: [0.0, 0.0, 0.0], half_extents: [10.0, 5.0, 10.0], yaw_deg: 0.0 },
        surface: SurfaceDef::Landscape,
        graph: Graph {
            seed: 5,
            nodes: vec![
                Node::new(Op::SurfaceSampler { points_per_m2: 0.5, jitter: 1.0 }),
                Node::new(Op::SelfPruning { radius: 1.5 }),
                Node::new(Op::Spawner { entries: vec![SpawnEntry { weight: 1.0, parts: vec![part("stone")] }, SpawnEntry { weight: 1.0, parts: vec![part("gold")] }] }),
            ],
        },
    }
}

#[test]
fn pcg_volumes_spawn_regenerate_and_reload() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    let mut next = editor.doc.clone();
    let id = next.take_id();
    next.pcg.push(scatter_volume(id));
    editor.commit(next, &mut f.world());
    let spawned = editor.pcg_instances(id);
    assert!(spawned > 20, "{spawned}");
    let count = f.scene.instance_count();
    // A smaller pruning radius regenerates into the same slots plus new ones; a larger one
    // retires the surplus instead of deleting.
    let mut def = scatter_volume(id);
    def.graph.nodes[1].op = Op::SelfPruning { radius: 3.0 };
    op(&mut editor, &mut f, json!({ "op": "pcg", "def": def }));
    assert!(editor.pcg_instances(id) < spawned);
    assert_eq!(f.scene.instance_count(), count);
    let visible = |f: &Fixture| f.scene.instance_handles().filter(|&h| f.scene.is_visible(h)).count();
    assert_eq!(visible(&f), 3 + editor.pcg_instances(id));

    // Save, then load into a fresh base scene: same visible instances, from the saved output.
    let saved = editor.save(CameraDef { position: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0, fov_deg: 60.0 }).to_json();
    let doc = SceneDoc::parse(&saved).unwrap();
    assert_eq!(doc.pcg_output.len(), 1);
    let mut g = Fixture::new();
    let mut loaded = g.editor();
    loaded.load(doc, &mut g.world());
    assert_eq!(loaded.pcg_instances(id), editor.pcg_instances(id));
    let transforms = |f: &Fixture| f.scene.instance_handles().filter(|&h| f.scene.is_visible(h)).map(|h| f.scene.transform(h)).collect::<Vec<_>>();
    assert_eq!(transforms(&f), transforms(&g));

    op(&mut editor, &mut f, json!({ "op": "delete", "pcg": [id] }));
    assert_eq!(visible(&f), 3);
}

#[test]
fn edits_with_one_merge_key_are_one_undo_step() {
    let mut f = Fixture::new();
    let mut editor = f.editor();
    for roughness in [0.3, 0.4, 0.5] {
        let op: EditOp = serde_json::from_value(json!({ "op": "material", "name": "gold", "params": { "roughness": roughness } })).unwrap();
        editor.apply_merged(op, &mut f.world(), Some("slider".into())).unwrap();
    }
    let gold = f.scene.find_material("gold").unwrap();
    assert_eq!(f.scene.material(gold).roughness, 0.5);
    assert!(editor.undo(&mut f.world()));
    assert_eq!(f.scene.material(gold).roughness, 0.2);
    assert!(!editor.can_undo());
}
