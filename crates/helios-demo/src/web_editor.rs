//! Page API of the browser editor (`web/editor/`): scene state as JSON, edit operations with
//! undo / redo, gizmo previews, picking, camera matrices for the overlay, selection outline,
//! scene files and glTF imports. Everything is addressed by scene instance / light indices and
//! PCG volume ids (see `editor`). JSON strings cross the boundary: the UI is plain JavaScript.

use std::cell::RefCell;

use glam::Vec3;
use wasm_bindgen::prelude::*;

use crate::app::DemoApp;
use crate::editor::doc::{SceneDoc, Trs};
use crate::editor::{describe, ops::EditOp};
use crate::scenes::model::Model;
use crate::scenes::SceneKind;
use crate::web::{fetch_bytes, load_model, with_state, State};

thread_local! {
    /// Outstanding picks: id and the world ray through the picked pixel.
    static PICKS: RefCell<(u32, Vec<(u32, (Vec3, Vec3))>)> = const { RefCell::new((0, Vec::new())) };
}

fn error_json(message: impl std::fmt::Display) -> String {
    serde_json::json!({ "error": message.to_string() }).to_string()
}

/// CSS pixels of the canvas -> output pixels.
fn output_pixel(st: &State, x: f32, y: f32) -> (f32, f32) {
    let (w, h) = (st.canvas.client_width().max(1) as f32, st.canvas.client_height().max(1) as f32);
    (x * st.config.width as f32 / w, y * st.config.height as f32 / h)
}

/// Switches the editor tools on (selection outline, picking; the editor UI owns the keyboard
/// and the left mouse button, the camera flies while the right button is held) or off.
#[wasm_bindgen(js_name = heliosSetEditorMode)]
pub fn helios_set_editor_mode(on: bool) {
    with_state(|st| {
        st.app.editing = on;
        st.input.held.clear();
        if !on {
            st.app.renderer.set_selection(&[]);
        }
    });
}

/// The whole editable scene (see `editor::describe::state`).
#[wasm_bindgen(js_name = heliosEditorState)]
pub fn helios_editor_state() -> String {
    with_state(|st| st.app.edit(|editor, world| describe::state(editor, world).to_string())).unwrap_or_default()
}

/// The material graph editor's palette: every node type (pins, defaults, properties) and the
/// preset graphs (id, label, graph).
#[wasm_bindgen(js_name = heliosMaterialNodes)]
pub fn helios_material_nodes() -> String {
    use helios_render::material_graph::{catalog, presets};
    let presets: Vec<serde_json::Value> =
        presets::PRESETS.iter().map(|(id, label)| serde_json::json!({ "id": id, "label": label, "graph": presets::preset(id) })).collect();
    serde_json::json!({ "nodes": catalog(), "presets": presets }).to_string()
}

/// Increments whenever an edit changed the scene (cheap: poll it to refresh panels).
#[wasm_bindgen(js_name = heliosEditorRevision)]
pub fn helios_editor_revision() -> f64 {
    with_state(|st| st.app.editor.revision as f64).unwrap_or(0.0)
}

/// One instance for the Details panel.
#[wasm_bindgen(js_name = heliosEditorInstance)]
pub fn helios_editor_instance(index: u32) -> String {
    with_state(|st| describe::instance(&st.app.editor, &st.app.scene, index).to_string()).unwrap_or_default()
}

/// Applies an edit operation (JSON, see `editor::ops::EditOp`) as one undo step; an op with a
/// `"merge"` key joins the previous step if that had the same key (sliders, light drags).
/// Returns `{"created": {...}}` or `{"error": "..."}`.
#[wasm_bindgen(js_name = heliosEditorApply)]
pub fn helios_editor_apply(op: &str) -> String {
    let value: serde_json::Value = match serde_json::from_str(op) {
        Ok(value) => value,
        Err(e) => return error_json(format!("bad edit: {e}")),
    };
    let merge = value.get("merge").and_then(|m| m.as_str()).map(String::from);
    let op: EditOp = match serde_json::from_value(value) {
        Ok(op) => op,
        Err(e) => return error_json(format!("bad edit: {e}")),
    };
    with_state(|st| match st.app.edit(|editor, world| editor.apply_merged(op, world, merge)) {
        Ok(created) => serde_json::json!({ "created": created }).to_string(),
        Err(e) => error_json(e),
    })
    .unwrap_or_else(|| error_json("the engine is not running"))
}

fn transforms(values: &[f32]) -> Vec<Trs> {
    values.chunks_exact(10).map(|v| Trs { t: [v[0], v[1], v[2]], r: [v[3], v[4], v[5], v[6]], s: [v[7], v[8], v[9]] }).collect()
}

/// Moves instances' components during a gizmo drag without an undo step: 10 numbers per
/// instance (translation, rotation quaternion, scale).
#[wasm_bindgen(js_name = heliosEditorPreview)]
pub fn helios_editor_preview(instances: Vec<u32>, values: Vec<f32>) {
    with_state(|st| {
        st.app.edit(|editor, world| {
            let items: Vec<(u32, String, glam::Mat4)> = instances
                .iter()
                .zip(transforms(&values))
                .filter_map(|(&i, t)| {
                    let (actor, c) = editor.instance_component(i)?;
                    Some((actor, editor.actor(actor)?.components[c].name.clone(), t.to_mat4()))
                })
                .collect();
            editor.preview_components(world, &items);
        })
    });
}

/// Moves actors (with their children and components) during a gizmo drag without an undo
/// step: 10 numbers of world transform per actor. Their instances move as movable ones until
/// the drag settles.
#[wasm_bindgen(js_name = heliosEditorPreviewActors)]
pub fn helios_editor_preview_actors(actors: Vec<u32>, values: Vec<f32>) {
    with_state(|st| {
        st.app.edit(|editor, world| {
            let items: Vec<(u32, glam::Mat4)> = actors.iter().zip(transforms(&values)).map(|(&a, t)| (a, t.to_mat4())).collect();
            editor.preview_actors(world, &items);
        })
    });
}

/// Moves components during a drag: JSON `[{actor, component, t, r, s}]` (world transforms).
#[wasm_bindgen(js_name = heliosEditorPreviewComponents)]
pub fn helios_editor_preview_components(json: &str) {
    #[derive(serde::Deserialize)]
    struct Item {
        actor: u32,
        component: String,
        #[serde(flatten)]
        transform: Trs,
    }
    let Ok(items) = serde_json::from_str::<Vec<Item>>(json) else { return };
    with_state(|st| {
        st.app.edit(|editor, world| {
            let items: Vec<(u32, String, glam::Mat4)> = items.into_iter().map(|i| (i.actor, i.component, i.transform.to_mat4())).collect();
            editor.preview_components(world, &items);
        })
    });
}

/// Puts previewed actors and components back where the document has them (a cancelled drag).
#[wasm_bindgen(js_name = heliosEditorRevertPreview)]
pub fn helios_editor_revert_preview(_instances: Vec<u32>) {
    with_state(|st| st.app.edit(|editor, world| editor.revert_preview(world)));
}

/// One actor for the Details panel and the gizmo (see `editor::describe::actor`).
#[wasm_bindgen(js_name = heliosEditorActor)]
pub fn helios_editor_actor(id: u32) -> String {
    with_state(|st| describe::actor(&st.app.editor, &st.app.scene, id).to_string()).unwrap_or_default()
}

/// The selection: JSON `{actors: [id…], components: [[actor, name]…]}`; the viewport outlines
/// their instances (an actor's children too).
#[wasm_bindgen(js_name = heliosEditorSelect)]
pub fn helios_editor_select(json: &str) {
    #[derive(serde::Deserialize, Default)]
    struct Selection {
        #[serde(default)]
        actors: Vec<u32>,
        #[serde(default)]
        components: Vec<(u32, String)>,
    }
    let selection: Selection = serde_json::from_str(json).unwrap_or_default();
    with_state(|st| st.app.select(selection.actors, selection.components));
}

/// Removes the free instance and light slots of deleted things now (otherwise done once the
/// editor has been idle for a while); returns [instances, lights] removed.
#[wasm_bindgen(js_name = heliosEditorCompact)]
pub fn helios_editor_compact() -> Vec<u32> {
    with_state(|st| {
        let (i, l) = st.app.compact_editor(true);
        vec![i as u32, l as u32]
    })
    .unwrap_or_default()
}

#[wasm_bindgen(js_name = heliosEditorUndo)]
pub fn helios_editor_undo() -> bool {
    with_state(|st| st.app.edit(|editor, world| editor.undo(world))).unwrap_or(false)
}

#[wasm_bindgen(js_name = heliosEditorRedo)]
pub fn helios_editor_redo() -> bool {
    with_state(|st| st.app.edit(|editor, world| editor.redo(world))).unwrap_or(false)
}

/// Instances outlined in the viewport (their components; see [`helios_editor_select`]).
#[wasm_bindgen(js_name = heliosEditorSetSelection)]
pub fn helios_editor_set_selection(instances: Vec<u32>) {
    with_state(|st| {
        let components: Vec<(u32, String)> = instances
            .iter()
            .filter_map(|&i| {
                let (actor, c) = st.app.editor.instance_component(i)?;
                Some((actor, st.app.editor.actor(actor)?.components[c].name.clone()))
            })
            .collect();
        st.app.select(Vec::new(), components);
    });
}

/// Asks what lies under canvas point (x, y) in CSS pixels; returns the pick's id. The answer
/// arrives a frame or two later from [`helios_editor_pick_result`].
#[wasm_bindgen(js_name = heliosEditorPick)]
pub fn helios_editor_pick(x: f32, y: f32) -> u32 {
    with_state(|st| {
        let (px, py) = output_pixel(st, x, y);
        let ray = st.app.pixel_ray(px, py, st.config.width as f32, st.config.height as f32);
        let id = PICKS.with(|p| {
            let mut p = p.borrow_mut();
            p.0 += 1;
            let id = p.0;
            p.1.push((id, ray));
            p.1.retain(|(i, _)| i + 8 > id);
            id
        });
        st.app.renderer.request_pick(id, (px.max(0.0) as u32, py.max(0.0) as u32));
        id
    })
    .unwrap_or(0)
}

/// The latest finished pick as JSON (instance, owner, world position and normal, ground hit),
/// or an empty string.
#[wasm_bindgen(js_name = heliosEditorPickResult)]
pub fn helios_editor_pick_result() -> String {
    with_state(|st| {
        let Some(pick) = st.app.renderer.take_pick() else { return String::new() };
        let Some(ray) = PICKS.with(|p| p.borrow().1.iter().find(|(id, _)| *id == pick.id).map(|(_, r)| *r)) else { return String::new() };
        st.app.resolve_pick(&pick, ray).to_string()
    })
    .unwrap_or_default()
}

/// Where the ray through canvas point (x, y) meets the ground (empty if it doesn't): placement
/// for drag and drop without waiting for a pick.
#[wasm_bindgen(js_name = heliosEditorGroundAt)]
pub fn helios_editor_ground_at(x: f32, y: f32) -> Vec<f32> {
    with_state(|st| {
        let (px, py) = output_pixel(st, x, y);
        let ray = st.app.pixel_ray(px, py, st.config.width as f32, st.config.height as f32);
        st.app.ground_hit(ray).map(|p| p.to_array().to_vec()).unwrap_or_default()
    })
    .unwrap_or_default()
}

/// Camera for the overlay: unjittered view-projection (16, column-major), eye (3), forward
/// (3), vertical field of view (radians), canvas CSS width and height.
#[wasm_bindgen(js_name = heliosEditorCamera)]
pub fn helios_editor_camera() -> Vec<f32> {
    with_state(|st| {
        let c = &st.app.camera;
        let aspect = st.config.width as f32 / st.config.height.max(1) as f32;
        let mut out = (c.projection(aspect) * c.view_matrix()).to_cols_array().to_vec();
        out.extend(c.position.to_array());
        out.extend(c.forward().to_array());
        out.extend([c.fov_y, st.canvas.client_width() as f32, st.canvas.client_height() as f32]);
        out
    })
    .unwrap_or_default()
}

/// Frames the box `lo`–`hi` (F).
#[wasm_bindgen(js_name = heliosEditorFocus)]
pub fn helios_editor_focus(lo: Vec<f32>, hi: Vec<f32>) {
    if let ([a, b, c], [d, e, f]) = (lo.as_slice(), hi.as_slice()) {
        with_state(|st| st.app.focus(Vec3::new(*a, *b, *c), Vec3::new(*d, *e, *f)));
    }
}

/// The scene file (`.helios.json`) of the current edits.
#[wasm_bindgen(js_name = heliosEditorSave)]
pub fn helios_editor_save() -> String {
    with_state(|st| st.app.save_scene_file()).unwrap_or_default()
}

/// Loads a scene file: rebuilds its base scene if needed, imports its models, applies its
/// edits. Resolves when done; rejects with a message.
#[wasm_bindgen(js_name = heliosEditorLoad)]
pub fn helios_editor_load(json: String) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        let doc = SceneDoc::parse(&json).map_err(|e| JsValue::from_str(&e))?;
        load_document(doc).await.map(|_| JsValue::TRUE).map_err(|e| JsValue::from_str(&e))
    })
}

/// Imports a glTF model as a content item: fetched from `url`, or parsed from `bytes` (a
/// dropped `.glb`; `url` is then its file name). Resolves with the item id.
#[wasm_bindgen(js_name = heliosEditorImport)]
pub fn helios_editor_import(url: String, bytes: Option<js_sys::Uint8Array>) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        let model = match bytes {
            Some(bytes) => model_from_bytes(&url, &bytes.to_vec()),
            None => load_model(&url).await,
        }
        .map_err(|e| JsValue::from_str(&e))?;
        with_state(|st| st.app.import_model(&model)).map(|id| JsValue::from_str(&id)).ok_or_else(|| JsValue::from_str("the engine is not running"))
    })
}

fn model_from_bytes(name: &str, bytes: &[u8]) -> Result<Model, String> {
    let document = helios_asset::GltfDocument::parse(bytes).map_err(|e| e.to_string())?;
    if !document.external_uris().is_empty() {
        return Err(format!("{name} references other files: drop a self-contained .glb, or import by URL"));
    }
    let asset = document.import(&Default::default()).map_err(|e| e.to_string())?;
    let stem = name.rsplit(['/', '\\']).next().and_then(|f| f.split('.').next()).filter(|n| !n.is_empty()).unwrap_or("model");
    let mut model = Model::new(stem, asset);
    model.source = name.to_string();
    Ok(model)
}

/// Rebuilds the document's base if needed, imports its models and applies it.
pub(crate) async fn load_document(doc: SceneDoc) -> Result<(), String> {
    let has_base = with_state(|st| st.app.has_base(&doc)).ok_or("the engine is not running")?;
    if !has_base {
        match &doc.base {
            crate::editor::doc::BaseScene::Gltf { url, ground } => {
                let mut model = load_model(url).await?;
                model.ground = *ground;
                with_state(|st| {
                    st.app.model = Some(model);
                    st.app.load_scene(SceneKind::Model);
                });
            }
            _ => with_state(|st| st.app.build_base(&doc)).unwrap_or(Ok(()))?,
        }
    }
    for import in &doc.imports {
        let model = load_model(&import.url).await.map_err(|e| format!("importing {}: {e}", import.name))?;
        with_state(|st| st.app.import_model(&model));
    }
    with_state(|st| st.app.load_scene_file(doc));
    Ok(())
}

/// Fetches and parses a scene file (`?scene-file=URL`).
pub(crate) async fn fetch_document(url: &str) -> Result<SceneDoc, String> {
    let bytes = fetch_bytes(url).await?;
    SceneDoc::parse(&String::from_utf8_lossy(&bytes))
}

impl DemoApp {
    /// Dolly along the view direction (the editor's mouse wheel without the right button).
    pub(crate) fn dolly(&mut self, steps: f32) {
        let forward = self.camera.forward();
        self.camera.position += forward * steps * 2.0;
    }
}
