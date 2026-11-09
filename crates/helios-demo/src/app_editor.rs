//! The demo app's side of the editor: scene documents in and out, imports, camera framing and
//! picking (shared by the browser editor and the native `--scene-file` / Tab tools).

use glam::{Mat4, Vec3};
use helios_render::{InstanceHandle, MaterialKind, PickResult, Scene};
use serde_json::{json, Value};

use crate::app::DemoApp;
use crate::editor::content::PrefabPart;
use crate::editor::doc::{BaseScene, CameraDef, ImportDef, SceneDoc};
use crate::editor::{Editor, World};
use crate::scenes::model::{add_model, Model};
use crate::scenes::{Detail, SceneKind};

/// The document base of a built scene.
pub fn base_scene(kind: SceneKind, detail: Detail, model: Option<&Model>) -> BaseScene {
    match (kind, model) {
        (SceneKind::Model, Some(model)) => BaseScene::Gltf { url: model.source.clone(), ground: model.ground },
        _ => BaseScene::Procedural { scene: kind.name().into(), detail: detail.pick("medium", "high").into() },
    }
}

/// The procedural scene and detail a document's base asks for (`None`: a glTF base).
pub fn procedural_base(base: &BaseScene) -> Option<(SceneKind, Detail)> {
    match base {
        BaseScene::Procedural { scene, detail } => {
            let kind = SceneKind::parse(scene).filter(|&k| k != SceneKind::Model)?;
            Some((kind, if detail == "high" { Detail::High } else { Detail::Medium }))
        }
        BaseScene::Gltf { .. } => None,
    }
}

impl DemoApp {
    /// Runs `f` with the editor and what it edits.
    pub fn edit<R>(&mut self, f: impl FnOnce(&mut Editor, &mut World) -> R) -> R {
        let mut world = World {
            scene: &mut self.scene,
            settings: &mut self.renderer.settings,
            sun: [&mut self.sun_azimuth, &mut self.sun_elevation, &mut self.sun_intensity],
            floor: self.floor,
        };
        f(&mut self.editor, &mut world)
    }

    pub fn camera_def(&self) -> CameraDef {
        let c = &self.camera;
        CameraDef { position: c.position.into(), yaw_deg: c.yaw.to_degrees(), pitch_deg: c.pitch.to_degrees(), fov_deg: c.fov_y.to_degrees() }
    }

    pub fn set_camera_def(&mut self, def: &CameraDef) {
        self.camera.position = def.position.into();
        self.camera.yaw = def.yaw_deg.to_radians();
        self.camera.pitch = def.pitch_deg.to_radians();
        self.camera.fov_y = def.fov_deg.clamp(5.0, 120.0).to_radians();
        self.renderer.reset_history();
    }

    /// The scene file of the current edits (with the camera and generated PCG output).
    pub fn save_scene_file(&self) -> String {
        self.editor.save(self.camera_def()).to_json()
    }

    /// Whether `doc` applies to the scene as built now (else the base must be rebuilt first).
    pub fn has_base(&self, doc: &SceneDoc) -> bool {
        base_scene(self.kind, self.detail, self.model.as_ref()) == doc.base
    }

    /// Rebuilds the document's procedural base scene if it is not the current one (bakes: takes
    /// a few seconds). glTF bases are loaded by the platform layer.
    pub fn build_base(&mut self, doc: &SceneDoc) -> Result<(), String> {
        if self.has_base(doc) {
            return Ok(());
        }
        let (kind, detail) = procedural_base(&doc.base).ok_or("the scene file's base is a glTF model; load it first")?;
        self.detail = detail;
        self.load_scene(kind);
        Ok(())
    }

    /// Applies a scene file to the current base (call [`DemoApp::build_base`] and import its
    /// models first).
    pub fn load_scene_file(&mut self, doc: SceneDoc) {
        if let Some(camera) = doc.camera {
            self.set_camera_def(&camera);
        }
        self.edit(|editor, world| editor.load(doc, world));
    }

    /// Adds an imported model's meshes and materials (named `NAME/…`) and makes it a content
    /// item; returns the item id.
    pub fn import_model(&mut self, model: &Model) -> String {
        // Placed copies are static: the editor's prefabs carry no animation.
        let parts = add_model(&mut self.scene, model)
            .parts
            .into_iter()
            .map(|p| PrefabPart {
                mesh: self.scene.mesh_name(p.mesh).to_string(),
                material: self.scene.material_name(p.material).to_string(),
                transform: p.transform,
                cast_shadow: p.cast_shadow,
            })
            .collect();
        self.editor.imports.push(ImportDef { name: model.name.clone(), url: model.source.clone() });
        self.editor.prefabs.insert(model.name.clone(), parts);
        model.name.clone()
    }

    /// Moves the camera back along its view direction until the box fits the view.
    pub fn focus(&mut self, lo: Vec3, hi: Vec3) {
        let center = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(0.1);
        let distance = radius / (self.camera.fov_y * 0.5).tan().max(0.05) * 1.1;
        self.camera.position = center - self.camera.forward() * distance;
    }

    /// The world-space ray through output pixel `(x, y)` of a `width` × `height` output.
    pub fn pixel_ray(&self, x: f32, y: f32, width: f32, height: f32) -> (Vec3, Vec3) {
        let projection = self.camera.projection(width / height.max(1.0));
        let inverse = (projection * self.camera.view_matrix()).inverse();
        let ndc = glam::Vec2::new(x / width * 2.0 - 1.0, 1.0 - y / height * 2.0);
        let near = inverse.project_point3(ndc.extend(1.0));
        let far = inverse.project_point3(ndc.extend(0.5));
        (self.camera.position, (far - near).normalize_or(self.camera.forward()))
    }

    /// A pick result as JSON: the opaque instance under the pixel, or a translucent one (glass,
    /// water: not in the visibility buffer) whose box the ray enters first; with the owner, so
    /// the UI can select a PCG volume for its spawned instances.
    pub fn resolve_pick(&self, pick: &PickResult, ray: (Vec3, Vec3)) -> Value {
        let opaque_distance = pick.instance.map_or(f32::MAX, |_| (pick.position - ray.0).length());
        let translucent = nearest_translucent(&self.scene, ray, opaque_distance);
        let (instance, position, normal) = match (translucent, pick.instance) {
            (Some((h, t)), _) => (Some(h), ray.0 + ray.1 * t, Vec3::Y),
            (None, Some(h)) => (Some(h), pick.position, pick.normal),
            (None, None) => (None, ray.0 + ray.1 * 50.0, Vec3::Y),
        };
        let ground = self.ground_hit(ray);
        json!({
            "id": pick.id,
            "instance": instance.map(InstanceHandle::index),
            "owner": instance.map(|h| crate::editor::describe::owner_value(self.editor.owner(h.index()))),
            "position": position.to_array(),
            "normal": normal.to_array(),
            "ground": ground.map(|p| p.to_array()),
        })
    }

    /// Where the ray meets the ground height field (marching, then bisecting).
    pub fn ground_hit(&self, (origin, dir): (Vec3, Vec3)) -> Option<Vec3> {
        let above = |t: f32| {
            let p = origin + dir * t;
            p.y - (self.floor)(p.x, p.z)
        };
        let (mut a, mut t) = (0.0, 0.5);
        while t < 2000.0 {
            if above(t) < 0.0 {
                for _ in 0..24 {
                    let mid = (a + t) * 0.5;
                    if above(mid) < 0.0 {
                        t = mid;
                    } else {
                        a = mid;
                    }
                }
                return Some(origin + dir * t);
            }
            a = t;
            t *= 1.08;
        }
        None
    }
}

/// The visible translucent instance whose oriented bounds the ray enters first, nearer than
/// `max_distance`.
fn nearest_translucent(scene: &Scene, (origin, dir): (Vec3, Vec3), max_distance: f32) -> Option<(InstanceHandle, f32)> {
    scene
        .instance_handles()
        .filter(|&h| scene.is_visible(h) && scene.material(scene.instance_material(h)).kind.is_translucent())
        .filter(|&h| scene.material(scene.instance_material(h)).kind != MaterialKind::Water || dir.y < 0.0)
        .filter_map(|h| {
            let inverse: Mat4 = scene.transform(h).inverse();
            let (o, d) = (inverse.transform_point3(origin), inverse.transform_vector3(dir));
            let aabb = scene.mesh_bounds(scene.instance_mesh(h));
            let pad = Vec3::splat(0.01);
            let (t0, t1) = ((aabb.min - pad - o) / d, (aabb.max + pad - o) / d);
            let (near, far) = (t0.min(t1).max_element(), t0.max(t1).min_element());
            // `t` along the local ray equals `t` along the world ray (same parameterisation).
            (near <= far && far > 0.0 && near < max_distance).then_some((h, near.max(0.0)))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
}
