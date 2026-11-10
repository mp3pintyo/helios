//! JSON views of the edited scene for the editor UI (outliner, details, content browser).

use helios_render::{MaterialKind, Scene};
use serde_json::{json, Value};

use super::content::{self, MATERIALS};
use super::doc::Trs;
use super::{light_def, materials, pcg, settings, Editor, Owner, World};

fn kind_name(kind: MaterialKind) -> &'static str {
    match kind {
        MaterialKind::Standard => "standard",
        MaterialKind::Rock => "rock",
        MaterialKind::Terrain => "terrain",
        MaterialKind::Metal => "metal",
        MaterialKind::Foliage => "foliage",
        MaterialKind::Glass => "glass",
        MaterialKind::Water => "water",
        MaterialKind::Translucent => "translucent",
    }
}

pub fn owner_value(owner: Owner) -> Value {
    match owner {
        Owner::Base(i) => json!({ "kind": "base", "id": i }),
        Owner::Added(id) => json!({ "kind": "added", "id": id }),
        Owner::Pcg(id) => json!({ "kind": "pcg", "id": id }),
        Owner::Spare => json!({ "kind": "deleted" }),
    }
}

/// Everything the UI lists: meshes, materials, instances (as parallel arrays: thousands of
/// them), lights, PCG volumes, environment, render settings and the content browser.
pub fn state(editor: &Editor, world: &World) -> Value {
    let scene = &*world.scene;
    let materials: Vec<Value> = (0..scene.material_count() as u32)
        .filter_map(|i| scene.material_handle(i))
        .map(|h| {
            let m = scene.material(h);
            let name = scene.material_name(h);
            let (graph, parameters) = materials::graph_value(scene, h);
            json!({
                "name": name,
                "kind": kind_name(m.kind),
                "base_color": m.base_color.to_array(),
                "roughness": m.roughness,
                "metallic": m.metallic,
                "opacity": m.opacity,
                "emissive": m.emissive,
                "emissive_color": m.emissive_color.to_array(),
                "edited": editor.doc.materials.contains_key(name),
                "graph": graph,
                "parameters": parameters,
                "slab": materials::slab_value(m),
                "displacement": materials::displacement_value(m),
                "displacement_graph": m.graph.is_some_and(|g| scene.material_graph_displaces(g)),
            })
        })
        .collect();
    let count = scene.instance_count();
    let (mut mesh, mut material, mut visible, mut owner, mut owner_id) =
        (Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count));
    for h in scene.instance_handles() {
        mesh.push(scene.instance_mesh(h).index());
        material.push(scene.instance_material(h).index());
        // 0 hidden, 1 visible, 2 deleted.
        let deleted = match editor.owner(h.index()) {
            Owner::Base(i) => editor.doc.instances.get(&i).is_some_and(|e| e.deleted),
            Owner::Spare => true,
            _ => false,
        };
        visible.push(if deleted { 2 } else { scene.is_visible(h) as u8 });
        let (kind, id) = match editor.owner(h.index()) {
            Owner::Base(i) => (0, i),
            Owner::Added(id) => (1, id),
            Owner::Pcg(id) => (2, id),
            Owner::Spare => (3, 0),
        };
        owner.push(kind);
        owner_id.push(id);
    }
    let mut lights: Vec<Value> = (0..editor.base_light_count().min(scene.lights().len()))
        .map(|i| {
            let def = editor.doc.lights.get(&(i as u32)).cloned().unwrap_or_else(|| light_def(&scene.lights()[i]));
            json!({ "index": i, "added": false, "def": def })
        })
        .collect();
    for added in &editor.doc.added_lights {
        if let Some(index) = editor.added_light_index(added.id) {
            lights.push(json!({ "index": index, "added": true, "def": added.light }));
        }
    }
    let volumes: Vec<Value> = editor
        .doc
        .pcg
        .iter()
        .map(|v| json!({ "def": v, "instances": editor.pcg_instances(v.id), "error": editor.pcg_error(v.id) }))
        .collect();
    let mut items = content::items();
    for (name, parts) in &editor.prefabs {
        items.push(content::Item { id: name.clone(), label: name.clone(), category: "Imported".into(), parts: parts.clone() });
    }
    let meshes: Vec<Value> = (0..scene.mesh_count() as u32)
        .filter_map(|i| scene.mesh_handle(i))
        .map(|h| json!({ "name": scene.mesh_name(h), "triangles": scene.mesh(h).triangle_count() }))
        .collect();
    json!({
        "revision": editor.revision,
        "can_undo": editor.can_undo(),
        "can_redo": editor.can_redo(),
        "base": editor.doc.base,
        "base_instances": editor.base_instance_count(),
        "meshes": meshes,
        "materials": materials,
        "graphs": materials::graphs_value(scene),
        "instances": { "mesh": mesh, "material": material, "visible": visible, "owner": owner, "owner_id": owner_id },
        "lights": lights,
        "pcg": volumes,
        "environment": Editor::current_environment(world),
        "render": { "schema": settings::schema(), "values": settings::values(world.settings) },
        "content": {
            "items": items,
            "materials": MATERIALS,
            "presets": pcg::PRESETS.iter().map(|(id, label)| json!({ "id": id, "label": label })).collect::<Vec<_>>(),
        },
        "warnings": editor.warnings,
    })
}

/// One instance for the Details panel.
pub fn instance(editor: &Editor, scene: &Scene, index: u32) -> Value {
    let Some(h) = scene.instance_handle(index) else { return Value::Null };
    let (lo, hi) = scene.instance_bounds(h);
    let transform = editor.effective_transform(scene, index).unwrap_or_else(|| scene.transform(h));
    let mesh = scene.instance_mesh(h);
    json!({
        "index": index,
        "mesh": scene.mesh_name(mesh),
        "material": scene.material_name(scene.instance_material(h)),
        "transform": Trs::from_mat4(&transform),
        "visible": scene.is_visible(h),
        "cast_shadow": scene.casts_shadow(h),
        "bounds": [lo.to_array(), hi.to_array()],
        "local_bounds": [scene.mesh_bounds(mesh).min.to_array(), scene.mesh_bounds(mesh).max.to_array()],
        "triangles": scene.mesh(mesh).triangle_count(),
        "owner": owner_value(editor.owner(index)),
    })
}
