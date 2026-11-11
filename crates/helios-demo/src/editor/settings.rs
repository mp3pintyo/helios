//! Render settings exposed to the editor's Details panel, as data: each entry is a key, a
//! label, a group, a value kind and a getter / setter on [`RenderSettings`]. The panel is built
//! from [`schema`], scene files store overrides by key, and a new setting is one entry here.

use helios_render::{DebugView, DepthOfField, DynamicResolution, FogScattering, LocalExposure, RenderSettings};
use serde_json::{json, Value};

pub enum Kind {
    Bool,
    Number { min: f32, max: f32, step: f32 },
    Choice(Vec<&'static str>),
}

pub struct Setting {
    pub key: &'static str,
    pub label: &'static str,
    pub group: &'static str,
    pub kind: Kind,
    get: fn(&RenderSettings) -> Value,
    set: fn(&mut RenderSettings, &Value),
}

fn flag(key: &'static str, label: &'static str, group: &'static str, get: fn(&RenderSettings) -> Value, set: fn(&mut RenderSettings, &Value)) -> Setting {
    Setting { key, label, group, kind: Kind::Bool, get, set }
}

#[allow(clippy::too_many_arguments)]
fn number(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    (min, max, step): (f32, f32, f32),
    get: fn(&RenderSettings) -> Value,
    set: fn(&mut RenderSettings, &Value),
) -> Setting {
    Setting { key, label, group, kind: Kind::Number { min, max, step }, get, set }
}

fn b(v: &Value) -> bool {
    v.as_bool().unwrap_or(false)
}

fn f(v: &Value) -> f32 {
    v.as_f64().unwrap_or(0.0) as f32
}

/// Every editable setting, in display order.
pub fn all() -> Vec<Setting> {
    vec![
        flag("global_illumination", "Lumen global illumination", "Lighting", |s| json!(s.global_illumination), |s, v| s.global_illumination = b(v)),
        number("gi_intensity", "GI intensity", "Lighting", (0.0, 4.0, 0.05), |s| json!(s.gi_intensity), |s, v| s.gi_intensity = f(v)),
        flag("shadows", "Virtual shadow maps", "Lighting", |s| json!(s.shadows), |s, v| s.shadows = b(v)),
        flag("local_light_shadow_maps", "Local light shadow maps", "Lighting", |s| json!(s.local_light_shadow_maps), |s, v| s.local_light_shadow_maps = b(v)),
        number("shadow_refresh_budget", "Shadow page refreshes per frame (0 = all)", "Lighting", (0.0, 512.0, 8.0), |s| json!(s.shadow_refresh_budget), |s, v| {
            s.shadow_refresh_budget = f(v).max(0.0) as u32
        }),
        flag("shadow_distant_prefiltered", "Prefiltered distant shadows", "Lighting", |s| json!(s.shadow_distant_prefiltered), |s, v| {
            s.shadow_distant_prefiltered = b(v)
        }),
        flag("volume_shadows", "Volume shadows (smoke, fire, mist)", "Lighting", |s| json!(s.volume_shadows), |s, v| s.volume_shadows = b(v)),
        number("shadow_distant_level", "Distant shadow levels from (0 = none)", "Lighting", (0.0, 15.0, 1.0), |s| json!(s.shadow_distant_level), |s, v| {
            s.shadow_distant_level = f(v).max(0.0) as u32
        }),
        flag("mega_lights", "MegaLights", "Lighting", |s| json!(s.mega_lights), |s, v| s.mega_lights = b(v)),
        number("mega_lights_importance_cull", "MegaLights importance cull", "Lighting", (0.0, 0.1, 0.0025), |s| json!(s.mega_lights_importance_cull), |s, v| {
            s.mega_lights_importance_cull = f(v)
        }),
        flag("mega_lights_shared_rays", "MegaLights shared rays", "Lighting", |s| json!(s.mega_lights_shared_rays), |s, v| s.mega_lights_shared_rays = b(v)),
        flag("mega_lights_hzb_traces", "MegaLights HZB screen traces", "Lighting", |s| json!(s.mega_lights_hzb_traces), |s, v| s.mega_lights_hzb_traces = b(v)),
        flag("mega_lights_volume", "MegaLights fog + translucency", "Lighting", |s| json!(s.mega_lights_volume), |s, v| s.mega_lights_volume = b(v)),
        flag("auto_exposure", "Auto exposure", "Lighting", |s| json!(s.auto_exposure), |s, v| s.auto_exposure = b(v)),
        flag("local_exposure", "Local exposure", "Lighting", |s| json!(s.local_exposure.is_some()), |s, v| {
            s.local_exposure = b(v).then(LocalExposure::default)
        }),
        number("bloom_intensity", "Bloom", "Lighting", (0.0, 0.3, 0.005), |s| json!(s.bloom_intensity), |s, v| s.bloom_intensity = f(v)),
        flag("aerial_perspective", "Aerial perspective", "Atmosphere", |s| json!(s.aerial_perspective), |s, v| s.aerial_perspective = b(v)),
        number("aerial_perspective_scale", "Haze density", "Atmosphere", (0.0, 16.0, 0.25), |s| json!(s.aerial_perspective_scale), |s, v| {
            s.aerial_perspective_scale = f(v)
        }),
        flag("clouds", "Volumetric clouds", "Atmosphere", |s| json!(s.clouds), |s, v| s.clouds = b(v)),
        flag("volumetric_fog", "Volumetric fog", "Atmosphere", |s| json!(s.volumetric_fog), |s, v| s.volumetric_fog = b(v)),
        number("fog_scattering", "Fog screen space scattering", "Atmosphere", (0.0, 4.0, 0.25), |s| json!(s.fog_scattering.map_or(0.0, |f| f.intensity)), |s, v| {
            s.fog_scattering = (f(v) > 0.0).then(|| FogScattering { intensity: f(v), ..Default::default() })
        }),
        flag("wind", "Wind animation", "Atmosphere", |s| json!(s.wind), |s, v| s.wind = b(v)),
        number("lod_threshold", "Nanite error (px)", "Geometry", (0.125, 16.0, 0.125), |s| json!(s.lod_threshold), |s, v| s.lod_threshold = f(v)),
        flag("tessellation", "Nanite tessellation (displacement)", "Geometry", |s| json!(s.tessellation), |s, v| s.tessellation = b(v)),
        number("dice_rate", "Tessellation dice rate (px)", "Geometry", (0.5, 8.0, 0.25), |s| json!(s.dice_rate), |s, v| s.dice_rate = f(v)),
        flag("occlusion_culling", "Occlusion culling", "Geometry", |s| json!(s.occlusion_culling), |s, v| s.occlusion_culling = b(v)),
        flag("sw_raster", "Software rasteriser", "Geometry", |s| json!(s.sw_raster_threshold > 0.0), |s, v| {
            s.sw_raster_threshold = if b(v) { 2.5 } else { 0.0 }
        }),
        flag("virtual_texturing", "Runtime virtual texture", "Materials", |s| json!(s.virtual_texturing), |s, v| s.virtual_texturing = b(v)),
        flag("texture_streaming", "Texture streaming", "Materials", |s| json!(s.texture_streaming), |s, v| s.texture_streaming = b(v)),
        flag("substrate", "Substrate slabs", "Materials", |s| json!(s.substrate), |s, v| s.substrate = b(v)),
        flag("translucency", "Translucency", "Materials", |s| json!(s.translucency), |s, v| s.translucency = b(v)),
        flag("translucency_layers", "Translucency layers", "Materials", |s| json!(s.translucency_layers), |s, v| s.translucency_layers = b(v)),
        flag("glass_shadows", "Coloured light through glass", "Materials", |s| json!(s.glass_shadows), |s, v| s.glass_shadows = b(v)),
        flag("avboit", "Voxel-based OIT (AVBOIT)", "Materials", |s| json!(s.avboit), |s, v| s.avboit = b(v)),
        flag("taa", "Temporal accumulation (TSR)", "Camera", |s| json!(s.taa), |s, v| s.taa = b(v)),
        flag("tsr_depth_history", "TSR depth history (disocclusion)", "Camera", |s| json!(s.tsr_depth_history), |s, v| s.tsr_depth_history = b(v)),
        number("motion_blur", "Motion blur shutter", "Camera", (0.0, 1.0, 0.05), |s| json!(s.motion_blur), |s, v| s.motion_blur = f(v)),
        flag("depth_of_field", "Depth of field", "Camera", |s| json!(s.depth_of_field.is_some()), |s, v| {
            s.depth_of_field = b(v).then(DepthOfField::default)
        }),
        number("exposure_compensation", "Exposure compensation (EV)", "Camera", (-4.0, 4.0, 0.1), |s| json!(s.exposure_compensation), |s, v| {
            s.exposure_compensation = f(v)
        }),
        number("screen_percentage", "Screen percentage", "Resolution", (25.0, 100.0, 1.0), |s| json!(s.screen_percentage), |s, v| {
            s.screen_percentage = f(v)
        }),
        flag("dynamic_resolution", "Dynamic resolution", "Resolution", |s| json!(s.dynamic_resolution.is_some()), |s, v| {
            s.dynamic_resolution = b(v).then(|| DynamicResolution::with_budget(14.0))
        }),
        Setting {
            key: "debug_view",
            label: "View mode",
            group: "Resolution",
            kind: Kind::Choice(DebugView::ALL.iter().map(|v| v.name()).collect()),
            get: |s| json!(s.debug_view.name()),
            set: |s, v| {
                if let Some(view) = DebugView::ALL.iter().find(|d| Some(d.name()) == v.as_str()) {
                    s.debug_view = *view;
                }
            },
        },
    ]
}

/// The settings' descriptions for the Details panel.
pub fn schema() -> Value {
    Value::Array(
        all()
            .iter()
            .map(|s| {
                let kind = match &s.kind {
                    Kind::Bool => json!({ "type": "bool" }),
                    Kind::Number { min, max, step } => json!({ "type": "number", "min": min, "max": max, "step": step }),
                    Kind::Choice(options) => json!({ "type": "choice", "options": options }),
                };
                json!({ "key": s.key, "label": s.label, "group": s.group, "kind": kind })
            })
            .collect(),
    )
}

/// Current values by key.
pub fn values(settings: &RenderSettings) -> Value {
    Value::Object(all().iter().map(|s| (s.key.to_string(), (s.get)(settings))).collect())
}

pub fn get(settings: &RenderSettings, key: &str) -> Option<Value> {
    all().iter().find(|s| s.key == key).map(|s| (s.get)(settings))
}

/// Sets one setting; unknown keys are an error (a scene file from a newer build).
pub fn set(settings: &mut RenderSettings, key: &str, value: &Value) -> Result<(), String> {
    let setting = all().into_iter().find(|s| s.key == key).ok_or_else(|| format!("unknown render setting '{key}'"))?;
    (setting.set)(settings, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_setting_round_trips_its_own_value() {
        let defaults = RenderSettings::default();
        for s in all() {
            let mut settings = RenderSettings::default();
            let value = (s.get)(&defaults);
            set(&mut settings, s.key, &value).unwrap();
            assert_eq!(get(&settings, s.key).unwrap(), value, "{}", s.key);
        }
        let mut settings = RenderSettings::default();
        set(&mut settings, "global_illumination", &json!(false)).unwrap();
        assert!(!settings.global_illumination);
        assert!(set(&mut settings, "warp_drive", &json!(true)).is_err());
        assert_eq!(schema().as_array().unwrap().len(), all().len());
    }
}
