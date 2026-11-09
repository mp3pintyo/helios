//! Editor tools in the native window (Tab): click to select an instance (picked through the
//! visibility buffer, outlined in the viewport), Delete / Ctrl+D, F to focus, Ctrl+Z / Ctrl+Y,
//! Ctrl+S to save the scene file. The full editor UI is the browser's (`web/editor.html`); this
//! is the same editor core without panels, for checking scene files and picking natively.

use glam::Vec3;
use helios_render::InstanceHandle;
use serde_json::json;

use crate::app::DemoApp;
use crate::editor::doc::SceneDoc;
use crate::editor::ops::EditOp;

#[derive(Default)]
pub struct NativeEditor {
    /// The selected instance.
    pub selected: Option<u32>,
    pick: Option<(u32, (Vec3, Vec3))>,
    next_pick: u32,
    /// Where Ctrl+S saves.
    pub path: Option<String>,
}

impl NativeEditor {
    /// Editor tools saving to `path` (Ctrl+S; default: `SCENE.helios.json`).
    pub fn new(path: Option<String>) -> Self {
        Self { path, ..Default::default() }
    }

    /// Tab: editor tools on / off.
    pub fn toggle(&mut self, app: &mut DemoApp) -> String {
        app.editing = !app.editing;
        if !app.editing {
            self.select(app, None);
        }
        format!("editor tools {} (click selects · Del deletes · Ctrl+D duplicates · F focuses · Ctrl+Z/Y undo/redo · Ctrl+S saves)", if app.editing { "on" } else { "off" })
    }

    fn select(&mut self, app: &mut DemoApp, instance: Option<u32>) {
        self.selected = instance;
        let handles: Vec<InstanceHandle> = instance.and_then(|i| app.scene.instance_handle(i)).into_iter().collect();
        app.renderer.set_selection(&handles);
    }

    /// A left click at output pixel (x, y) of a `size` output.
    pub fn click(&mut self, app: &mut DemoApp, (x, y): (f32, f32), size: (u32, u32)) {
        self.next_pick += 1;
        let ray = app.pixel_ray(x, y, size.0 as f32, size.1 as f32);
        self.pick = Some((self.next_pick, ray));
        app.renderer.request_pick(self.next_pick, (x.max(0.0) as u32, y.max(0.0) as u32));
    }

    /// Takes a finished pick: selects what it hit; returns a description.
    pub fn poll(&mut self, app: &mut DemoApp) -> Option<String> {
        let pick = app.renderer.take_pick()?;
        let (id, ray) = self.pick?;
        if pick.id != id {
            return None;
        }
        self.pick = None;
        let result = app.resolve_pick(&pick, ray);
        let instance = result["instance"].as_u64().map(|i| i as u32);
        self.select(app, instance);
        Some(match instance {
            Some(i) => format!("selected {}", crate::editor::describe::instance(&app.editor, &app.scene, i)),
            None => "nothing selected".into(),
        })
    }

    /// Editor keys (with Ctrl held or not); returns a message if the key was the editor's.
    pub fn key(&mut self, app: &mut DemoApp, code: &str, ctrl: bool) -> Option<String> {
        let selected: Vec<u32> = self.selected.into_iter().collect();
        let op = |value: serde_json::Value| serde_json::from_value::<EditOp>(value).expect("well-formed edit");
        let message = match (code, ctrl) {
            ("KeyZ", true) => format!("undo: {}", app.edit(|e, w| e.undo(w))),
            ("KeyY", true) => format!("redo: {}", app.edit(|e, w| e.redo(w))),
            ("KeyS", true) => {
                let path = self.path.clone().unwrap_or_else(|| format!("{}.helios.json", app.scene_name()));
                match std::fs::write(&path, app.save_scene_file()) {
                    Ok(()) => format!("saved {path}"),
                    Err(e) => format!("could not save {path}: {e}"),
                }
            }
            ("KeyD", true) if !selected.is_empty() => {
                let created = app.edit(|e, w| e.apply(op(json!({ "op": "duplicate", "instances": selected, "offset": [1.0, 0.0, 0.0] })), w));
                match created {
                    Ok(c) => {
                        self.select(app, c.instances.first().copied());
                        format!("duplicated into {:?}", c.instances)
                    }
                    Err(e) => e,
                }
            }
            ("Delete", false) if !selected.is_empty() => {
                let result = app.edit(|e, w| e.apply(op(json!({ "op": "delete", "instances": selected })), w));
                self.select(app, None);
                result.map_or_else(|e| e, |_| "deleted".into())
            }
            ("KeyF", false) => {
                let handle = self.selected.and_then(|i| app.scene.instance_handle(i))?;
                let (lo, hi) = app.scene.instance_bounds(handle);
                app.focus(lo, hi);
                "focused".into()
            }
            _ => return None,
        };
        Some(message)
    }
}

/// Reads a scene file; relative glTF and import URLs resolve against its directory.
pub fn read_scene_file(path: &str) -> Result<SceneDoc, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let mut doc = SceneDoc::parse(&text)?;
    let dir = std::path::Path::new(path).parent().map(std::path::Path::to_path_buf).unwrap_or_default();
    let resolve = |url: &mut String| {
        if std::path::Path::new(url.as_str()).is_relative() && !std::path::Path::new(url.as_str()).exists() {
            *url = dir.join(&*url).to_string_lossy().into_owned();
        }
    };
    if let crate::editor::doc::BaseScene::Gltf { url, .. } = &mut doc.base {
        resolve(url);
    }
    for import in &mut doc.imports {
        resolve(&mut import.url);
    }
    Ok(doc)
}
