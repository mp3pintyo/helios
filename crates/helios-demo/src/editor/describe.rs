//! JSON views of the edited scene for the editor UI (outliner, details, content browser).

use helios_render::{MaterialKind, Scene};
use serde_json::{json, Value};

use super::content::{self, MATERIALS};
use super::doc::Trs;
use super::actors::{world_light, Kind};
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

pub fn owner_value(editor: &Editor, owner: Owner) -> Value {
    match owner {
        Owner::Actor { actor, component } => {
            let name = editor.actor(actor).and_then(|a| a.components.get(component as usize)).map(|c| c.name.clone());
            json!({ "kind": "actor", "actor": actor, "component": name })
        }
        Owner::Pcg(id) => json!({ "kind": "pcg", "id": id }),
        Owner::Spare => json!({ "kind": "deleted" }),
    }
}

/// Everything the UI lists: meshes, materials, actors (as parallel arrays: thousands of them),
/// lights, PCG volumes, environment, render settings and the content browser.
pub fn state(editor: &Editor, world: &World) -> Value {
    let scene = &*world.scene;
    let materials: Vec<Value> = (0..scene.material_count() as u32)
        .filter_map(|i| scene.material_handle(i))
        .map(|h| {
            let m = scene.material(h);
            let name = scene.material_name(h);
            let (graph, parameters) = materials::graph_value(scene, h);
            let (displacement_mode, displacement_animated) = m.graph.map_or((Value::Null, false), |g| materials::graph_displacement(scene, g));
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
                "displacement_mode": displacement_mode,
                "displacement_animated": displacement_animated,
                // m/s at unit scale; null: animated without a bound (refreshed every frame).
                "displacement_speed": scene.displacement_speed(h),
            })
        })
        .collect();
    // Actors: id, name, parent id (-1: root), flags.
    let count = editor.actors().count();
    let (mut ids, mut names, mut parents, mut flags) = (Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count));
    for (id, a) in editor.actors() {
        ids.push(id);
        names.push(a.name.as_str());
        parents.push(a.parent.map_or(-1, |p| p as i64));
        let mut f = (a.visible as u32) | (a.shown as u32) << 1 | (a.added as u32) << 2 | (a.folder as u32) << 3;
        for c in &a.components {
            f |= match c.kind {
                Kind::Mesh { .. } => 32,
                Kind::Light { .. } => 16,
                Kind::Volume { .. } | Kind::Particles { .. } => 64,
            };
        }
        if a.components.is_empty() {
            f |= 128;
        }
        flags.push(f);
    }
    // Light components with their world-space light (overlay icons).
    let mut lights: Vec<Value> = Vec::new();
    for (id, a) in editor.actors() {
        for c in &a.components {
            if let Kind::Light { slot: Some(i), .. } = c.kind {
                if let Some(light) = scene.lights().get(i).filter(|_| a.shown && c.visible) {
                    lights.push(json!({ "index": i, "actor": id, "component": c.name, "def": light_def(light) }));
                }
            }
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
        "meshes": meshes,
        "materials": materials,
        "graphs": materials::graphs_value(scene),
        "actors": { "id": ids, "name": names, "parent": parents, "flags": flags },
        "instance_count": scene.instance_count(),
        "free_slots": editor.free_slots(),
        "compactions": editor.compactions,
        "moving": editor.moving_count(),
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
    let mesh = scene.instance_mesh(h);
    json!({
        "index": index,
        "mesh": scene.mesh_name(mesh),
        "material": scene.material_name(scene.instance_material(h)),
        "transform": Trs::from_mat4(&scene.transform(h)),
        "visible": scene.is_visible(h),
        "cast_shadow": scene.casts_shadow(h),
        "bounds": [lo.to_array(), hi.to_array()],
        "local_bounds": [scene.mesh_bounds(mesh).min.to_array(), scene.mesh_bounds(mesh).max.to_array()],
        "triangles": scene.mesh(mesh).triangle_count(),
        "owner": owner_value(editor, editor.owner(index)),
    })
}

/// One actor for the Details panel and the gizmo: its transforms, bounds and components.
pub fn actor(editor: &Editor, scene: &Scene, id: u32) -> Value {
    let Some(a) = editor.actor(id) else { return Value::Null };
    let bounds = editor.actor_bounds(scene, id).map(|(lo, hi)| [lo.to_array(), hi.to_array()]);
    let components: Vec<Value> = a
        .components
        .iter()
        .map(|c| {
            let world = a.world * c.local;
            let mut v = json!({ "name": c.name, "kind": c.kind.name(), "visible": c.visible, "local": Trs::from_mat4(&c.local), "world": Trs::from_mat4(&world) });
            match &c.kind {
                Kind::Mesh { mesh, material, cast_shadow, slot } => {
                    let b = scene.mesh_bounds(*mesh);
                    v["mesh"] = json!(scene.mesh_name(*mesh));
                    v["material"] = json!(scene.material_name(*material));
                    v["cast_shadow"] = json!(cast_shadow);
                    v["instance"] = json!(slot.map(|h| h.index()));
                    v["local_bounds"] = json!([b.min.to_array(), b.max.to_array()]);
                    v["triangles"] = json!(scene.mesh(*mesh).triangle_count());
                }
                Kind::Light { def, slot } => {
                    v["light"] = json!(world_light(a.world, def, true));
                    v["index"] = json!(slot);
                }
                Kind::Volume { index } | Kind::Particles { index } => v["index"] = json!(index),
            }
            v
        })
        .collect();
    json!({
        "id": id,
        "key": a.key,
        "name": a.name,
        "parent": a.parent,
        "parent_name": a.parent.and_then(|p| editor.actor(p)).map(|p| p.name.clone()),
        "added": a.added,
        "folder": a.folder,
        "visible": a.visible,
        "shown": a.shown,
        "local": Trs::from_mat4(&a.local),
        "world": Trs::from_mat4(&a.world),
        "bounds": bounds,
        "children": a.children.len(),
        "descendants": editor.subtree(id).len() - 1,
        "components": components,
    })
}
