//! The `KHR_materials_*` PBR extensions, read from their JSON (the `gltf` crate keeps
//! extensions it has no feature enabled for as raw values, which covers all of them uniformly).

use glam::Vec3;
use gltf::json::Value;

use crate::{MaterialExtensions, TextureRef};

/// Extensions read here (the importer accepts them in `extensionsRequired`).
pub const EXTENSIONS: [&str; 8] = [
    "KHR_materials_clearcoat",
    "KHR_materials_sheen",
    "KHR_materials_ior",
    "KHR_materials_specular",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_unlit",
    "KHR_materials_anisotropy",
];

/// The material's extensions; `texture` resolves a JSON texture info into a reference.
pub fn read(m: &gltf::Material, texture: &dyn Fn(&Value) -> Option<TextureRef>) -> MaterialExtensions {
    let mut e = MaterialExtensions::default();
    let number = |v: &Value, key: &str, default: f32| v.get(key).and_then(Value::as_f64).map_or(default, |x| x as f32);
    let color = |v: &Value, key: &str, default: Vec3| {
        v.get(key).and_then(Value::as_array).filter(|a| a.len() == 3).map_or(default, |a| Vec3::from_array(std::array::from_fn(|i| a[i].as_f64().unwrap_or(1.0) as f32)))
    };
    let tex = |v: &Value, key: &str| v.get(key).and_then(texture);
    if let Some(v) = m.extension_value("KHR_materials_clearcoat") {
        e.clearcoat = number(v, "clearcoatFactor", 0.0);
        e.clearcoat_texture = tex(v, "clearcoatTexture");
        e.clearcoat_roughness = number(v, "clearcoatRoughnessFactor", 0.0);
        e.clearcoat_roughness_texture = tex(v, "clearcoatRoughnessTexture");
        e.clearcoat_normal_texture = tex(v, "clearcoatNormalTexture");
        e.clearcoat_normal_scale = v.get("clearcoatNormalTexture").map_or(1.0, |n| number(n, "scale", 1.0));
    }
    if let Some(v) = m.extension_value("KHR_materials_sheen") {
        e.sheen_color = color(v, "sheenColorFactor", Vec3::ZERO);
        e.sheen_color_texture = tex(v, "sheenColorTexture");
        e.sheen_roughness = number(v, "sheenRoughnessFactor", 0.0);
        e.sheen_roughness_texture = tex(v, "sheenRoughnessTexture");
    }
    if let Some(v) = m.extension_value("KHR_materials_ior") {
        e.ior = number(v, "ior", 1.5);
    }
    if let Some(v) = m.extension_value("KHR_materials_specular") {
        e.specular = number(v, "specularFactor", 1.0);
        e.specular_texture = tex(v, "specularTexture");
        e.specular_color = color(v, "specularColorFactor", Vec3::ONE);
        e.specular_color_texture = tex(v, "specularColorTexture");
    }
    if let Some(v) = m.extension_value("KHR_materials_transmission") {
        e.transmission = number(v, "transmissionFactor", 0.0);
        e.transmission_texture = tex(v, "transmissionTexture");
    }
    if let Some(v) = m.extension_value("KHR_materials_volume") {
        e.thickness = number(v, "thicknessFactor", 0.0);
        e.attenuation_distance = number(v, "attenuationDistance", f32::INFINITY);
        e.attenuation_color = color(v, "attenuationColor", Vec3::ONE);
    }
    e.unlit = m.extension_value("KHR_materials_unlit").is_some();
    if let Some(v) = m.extension_value("KHR_materials_anisotropy") {
        e.anisotropy = number(v, "anisotropyStrength", 0.0);
        e.anisotropy_rotation = number(v, "anisotropyRotation", 0.0);
        if v.get("anisotropyTexture").is_some() {
            log::warn!("material '{}': anisotropy textures are not supported; using the factors", m.name().unwrap_or(""));
        }
    }
    e
}
