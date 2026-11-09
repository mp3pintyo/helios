//! The scene file (`.helios.json`): a base scene plus every edit made to it.
//!
//! A document never stores the base scene itself — that is rebuilt from its recipe (a
//! procedural scene and detail, or a glTF URL) — only the differences: per-instance overrides
//! keyed by the base instance's index (with its mesh name, to detect a builder that changed
//! since), instances added from the content browser and imported models, material parameter
//! overrides and lights keyed by stable names / indices, the environment, render settings and
//! PCG volumes with their graphs. Saved files also carry each PCG volume's output, so a file
//! loads with exactly the instances it was saved with.

use std::collections::BTreeMap;

use glam::{Mat4, Quat, Vec3};
use helios_pcg::{Graph, Volume};
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "helios-scene";
pub const VERSION: u32 = 1;

fn yes() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

/// Translation, rotation (quaternion x, y, z, w) and scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trs {
    pub t: [f32; 3],
    pub r: [f32; 4],
    pub s: [f32; 3],
}

impl Trs {
    pub fn from_mat4(m: &Mat4) -> Self {
        let (s, r, t) = m.to_scale_rotation_translation();
        Self { t: t.into(), r: r.normalize().into(), s: s.into() }
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(Vec3::from(self.s), Quat::from_array(self.r).normalize(), Vec3::from(self.t))
    }
}

/// What the edits apply to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BaseScene {
    /// A procedural demo scene (`valley`, `cathedral`, `forest`) at a detail (`medium`, `high`):
    /// both change the instance list, so both are part of the identity.
    Procedural { scene: String, detail: String },
    /// A glTF model (URL relative to the page, or a file path relative to the scene file).
    Gltf {
        url: String,
        #[serde(default = "yes")]
        ground: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraDef {
    pub position: [f32; 3],
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub fov_deg: f32,
}

/// A glTF model imported into the scene; its meshes and materials are named `name/…`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportDef {
    pub name: String,
    pub url: String,
}

/// Overrides of a base instance (absent fields keep the base value).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstanceEdit {
    /// The base instance's mesh name, checked on load.
    pub mesh: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Trs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    /// Deleted (hidden and left out of the outliner; visibility is a separate switch).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deleted: bool,
}

/// An instance added in the editor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AddedInstance {
    pub id: u32,
    pub mesh: String,
    pub material: String,
    pub transform: Trs,
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub cast_shadow: bool,
}

/// Material parameter overrides (absent fields keep the base value).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaterialParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_color: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f32>,
    /// Emission as a multiple of the base colour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive_color: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_strength: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_scale: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightShape {
    Point,
    Spot,
}

/// A local light, complete (lights are small, so edits store all of it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightDef {
    pub shape: LightShape,
    pub position: [f32; 3],
    /// Spot direction.
    #[serde(default)]
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub source_radius: f32,
    pub cast_shadows: bool,
    /// Spot cone half-angles in degrees.
    #[serde(default)]
    pub inner_deg: f32,
    #[serde(default)]
    pub outer_deg: f32,
    /// Deleted lights stay in the list (handles don't shift) but are switched off.
    #[serde(default = "yes")]
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AddedLight {
    pub id: u32,
    pub light: LightDef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FogDef {
    pub density: f32,
    pub height_falloff: f32,
    pub base_height: f32,
    pub albedo: [f32; 3],
    pub anisotropy: f32,
    pub ambient: f32,
    pub noise: f32,
    pub distance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CloudsDef {
    pub coverage: f32,
    pub density: f32,
    pub base_altitude: f32,
    pub thickness: f32,
    pub wind: [f32; 2],
    pub shadow_strength: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindDef {
    pub direction: [f32; 2],
    pub strength: f32,
    pub gust_frequency: f32,
    pub gust_wavelength: f32,
    pub turbulence: f32,
}

/// Sun, sky and air (UE's directional light, sky atmosphere, height fog, clouds and wind).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub fog: FogDef,
    pub clouds: CloudsDef,
    pub wind: WindDef,
}

/// What a PCG volume samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SurfaceDef {
    /// The scene's ground (landscape heightfield).
    Landscape,
    /// The triangles of a mesh instance (its finest in-memory level of detail).
    Instance { instance: u32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcgVolumeDef {
    pub id: u32,
    pub name: String,
    /// The preset the graph started from (informative).
    #[serde(default)]
    pub preset: String,
    pub volume: Volume,
    pub surface: SurfaceDef,
    pub graph: Graph,
}

/// A PCG volume's generated instances, as saved: the parts (mesh, material) and per instance
/// the part index and its transform (translation, rotation quaternion, scale: 10 numbers).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcgOutputDef {
    pub id: u32,
    /// Hash of the volume definition it was generated from.
    pub hash: String,
    pub parts: Vec<[String; 2]>,
    pub instances: Vec<(u32, [f32; 10])>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneDoc {
    pub format: String,
    pub version: u32,
    pub base: BaseScene,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraDef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<ImportDef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub instances: BTreeMap<u32, InstanceEdit>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added: Vec<AddedInstance>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub materials: BTreeMap<String, MaterialParams>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lights: BTreeMap<u32, LightDef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added_lights: Vec<AddedLight>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<Environment>,
    /// Render setting overrides by key (see `settings.rs`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub render: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pcg: Vec<PcgVolumeDef>,
    /// Generated PCG instances; only in saved files (the live document keeps them in a cache, so
    /// undo snapshots stay small).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pcg_output: Vec<PcgOutputDef>,
    /// Next id for added instances, lights and volumes.
    #[serde(default)]
    pub next_id: u32,
}

impl SceneDoc {
    pub fn new(base: BaseScene) -> Self {
        Self {
            format: FORMAT.into(),
            version: VERSION,
            base,
            camera: None,
            imports: Vec::new(),
            instances: BTreeMap::new(),
            added: Vec::new(),
            materials: BTreeMap::new(),
            lights: BTreeMap::new(),
            added_lights: Vec::new(),
            environment: None,
            render: BTreeMap::new(),
            pcg: Vec::new(),
            pcg_output: Vec::new(),
            next_id: 1,
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let doc: SceneDoc = serde_json::from_str(text).map_err(|e| format!("not a scene file: {e}"))?;
        if doc.format != FORMAT {
            return Err(format!("not a scene file (format '{}', expected '{FORMAT}')", doc.format));
        }
        if doc.version > VERSION {
            return Err(format!("scene file version {} is newer than this build reads ({VERSION})", doc.version));
        }
        Ok(doc)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("scene documents serialise")
    }

    pub fn take_id(&mut self) -> u32 {
        let id = self.next_id.max(1);
        self.next_id = id + 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documents_round_trip() {
        let mut doc = SceneDoc::new(BaseScene::Procedural { scene: "valley".into(), detail: "medium".into() });
        let id = doc.take_id();
        doc.instances.insert(12, InstanceEdit { mesh: "rock".into(), visible: Some(false), ..Default::default() });
        doc.added.push(AddedInstance {
            id,
            mesh: "engine/cube".into(),
            material: "engine/default".into(),
            transform: Trs::from_mat4(&Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))),
            cast_shadow: true,
        });
        doc.materials.insert("gold".into(), MaterialParams { roughness: Some(0.6), ..Default::default() });
        doc.render.insert("global_illumination".into(), serde_json::Value::Bool(false));
        let json = doc.to_json();
        assert!(json.contains("\"12\"") && !json.contains("cast_shadow"));
        assert_eq!(SceneDoc::parse(&json).unwrap(), doc);
        assert!(SceneDoc::parse("{\"format\":\"other\"}").is_err());
    }

    #[test]
    fn trs_round_trips_through_matrices() {
        let m = Mat4::from_scale_rotation_translation(Vec3::new(1.0, 2.0, 0.5), Quat::from_rotation_y(0.7), Vec3::new(3.0, -1.0, 2.0));
        assert!(Trs::from_mat4(&m).to_mat4().abs_diff_eq(m, 1e-5));
    }
}
