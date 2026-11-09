//! Engine content: built-in meshes, materials and prefabs (UE's engine content / starter
//! content) that the content browser places and PCG presets spawn. Everything is procedural
//! and named `engine/…`; a mesh or material is baked into the scene the first time something
//! uses it, so scenes that never use the editor pay nothing.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use helios_geometry::modeling::{cuboid, MeshBuilder};
use helios_geometry::vegetation::{self, LeafShape, TreeParams};
use helios_geometry::{primitives, BuildConfig, Mesh, SdfMode};
use helios_render::{Material, MaterialHandle, MaterialKind, MeshHandle, Scene, TextureDesc, TextureHandle, TextureMapping, TexturePattern, WindResponse};
use serde::Serialize;

use crate::scenes;

/// One part of a prefab: a mesh with a material at a transform relative to the prefab.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrefabPart {
    pub mesh: String,
    pub material: String,
    #[serde(skip)]
    pub transform: Mat4,
    pub cast_shadow: bool,
}

impl PrefabPart {
    pub fn new(mesh: &str, material: &str) -> Self {
        Self { mesh: mesh.into(), material: material.into(), transform: Mat4::IDENTITY, cast_shadow: true }
    }
}

/// A placeable item of the content browser.
#[derive(Clone, Debug, Serialize)]
pub struct Item {
    pub id: String,
    pub label: String,
    pub category: String,
    pub parts: Vec<PrefabPart>,
}

const BASIC: &[(&str, &str, &str)] = &[
    ("cube", "Cube", "engine/default"),
    ("sphere", "Sphere", "engine/default"),
    ("cylinder", "Cylinder", "engine/default"),
    ("plane", "Plane", "engine/default"),
];

const PETALS: &[(&str, &str)] = &[("red", "Red flower"), ("yellow", "Yellow flower"), ("blue", "Blue flower"), ("white", "White flower")];

/// Built-in placeable items.
pub fn items() -> Vec<Item> {
    let single = |id: &str, label: &str, category: &str, mesh: &str, material: &str| Item {
        id: format!("engine/{id}"),
        label: label.into(),
        category: category.into(),
        parts: vec![PrefabPart::new(&format!("engine/{mesh}"), material)],
    };
    let mut items: Vec<Item> = BASIC.iter().map(|(id, label, material)| single(id, label, "Basic", id, material)).collect();
    items.push(single("rock", "Rock", "Nature", "rock", "engine/rock"));
    items.push(single("boulder", "Boulder", "Nature", "boulder", "engine/rock"));
    items.push(single("grass", "Grass clump", "Nature", "grass", "engine/grass"));
    items.push(single("fern", "Fern", "Nature", "fern", "engine/fern"));
    for (colour, label) in PETALS {
        items.push(Item { id: format!("engine/flower-{colour}"), label: (*label).into(), category: "Nature".into(), parts: prefab(&format!("engine/flower-{colour}")).unwrap() });
    }
    for (id, label) in [("bush", "Bush"), ("tree", "Broadleaf tree"), ("conifer", "Conifer")] {
        items.push(Item { id: format!("engine/{id}"), label: label.into(), category: "Nature".into(), parts: prefab(&format!("engine/{id}")).unwrap() });
    }
    items
}

/// The parts of a built-in prefab.
pub fn prefab(id: &str) -> Option<Vec<PrefabPart>> {
    let name = id.strip_prefix("engine/")?;
    let parts = match name {
        "bush" => vec![PrefabPart::new("engine/bush-wood", "engine/bark"), PrefabPart::new("engine/bush-leaves", "engine/leaves")],
        "tree" => vec![PrefabPart::new("engine/tree-wood", "engine/bark"), PrefabPart::new("engine/tree-leaves", "engine/leaves")],
        "conifer" => vec![PrefabPart::new("engine/conifer-wood", "engine/pine-bark"), PrefabPart::new("engine/conifer-needles", "engine/needles")],
        flower if flower.starts_with("flower-") => {
            let colour = &flower["flower-".len()..];
            PETALS.iter().find(|(c, _)| *c == colour)?;
            vec![PrefabPart::new("engine/flower-stem", "engine/stem"), PrefabPart::new("engine/flower-head", &format!("engine/petals-{colour}"))]
        }
        _ => {
            let material = items_material(name)?;
            vec![PrefabPart::new(id, material)]
        }
    };
    Some(parts)
}

fn items_material(mesh: &str) -> Option<&'static str> {
    BASIC.iter().find(|(id, ..)| *id == mesh).map(|(.., m)| *m).or(match mesh {
        "rock" | "boulder" => Some("engine/rock"),
        "grass" => Some("engine/grass"),
        "fern" => Some("engine/fern"),
        _ => None,
    })
}

/// Names of every built-in material (for the content browser).
pub const MATERIALS: &[&str] = &[
    "engine/default",
    "engine/white",
    "engine/rock",
    "engine/gold",
    "engine/glass",
    "engine/emissive",
    "engine/bark",
    "engine/pine-bark",
    "engine/leaves",
    "engine/needles",
    "engine/grass",
    "engine/fern",
    "engine/stem",
    "engine/petals-red",
    "engine/petals-yellow",
    "engine/petals-blue",
    "engine/petals-white",
];

/// Creates built-in meshes and materials on demand and remembers their textures.
#[derive(Default)]
pub struct Content {
    textures: HashMap<&'static str, TextureHandle>,
}

impl Content {
    /// The scene's mesh called `name`, baking a built-in one on first use.
    pub fn mesh(&mut self, scene: &mut Scene, name: &str) -> Option<MeshHandle> {
        if let Some(mesh) = scene.find_mesh(name) {
            return Some(mesh);
        }
        let (mesh, sdf) = builtin_mesh(name.strip_prefix("engine/")?)?;
        Some(scenes::add_baked_with(scene, name, &mesh, BuildConfig { sdf, ..Default::default() }))
    }

    /// The scene's material called `name`, creating a built-in one on first use.
    pub fn material(&mut self, scene: &mut Scene, name: &str) -> Option<MaterialHandle> {
        if let Some(material) = scene.find_material(name) {
            return Some(material);
        }
        let material = self.builtin_material(scene, name.strip_prefix("engine/")?)?;
        let handle = scene.add_material(material);
        scene.set_material_name(handle, name);
        Some(handle)
    }

    fn texture(&mut self, scene: &mut Scene, key: &'static str) -> TextureHandle {
        if let Some(&t) = self.textures.get(key) {
            return t;
        }
        let desc = match key {
            "rock" => TextureDesc::new(TexturePattern::Rock, Vec3::new(0.3, 0.28, 0.25), Vec3::new(0.14, 0.13, 0.12)).roughness(0.75, 0.9).bump(1.4).seed(213),
            "hammered" => TextureDesc::new(TexturePattern::Hammered, Vec3::ONE, Vec3::ONE).roughness(0.18, 0.3).scale(1.5).bump(0.6).seed(215),
            "bark" => TextureDesc::new(TexturePattern::Bark, Vec3::new(0.24, 0.21, 0.17), Vec3::new(0.06, 0.05, 0.04)).roughness(0.9, 0.95).param(0.6).bump(1.5).seed(275),
            "pine-bark" => TextureDesc::new(TexturePattern::Bark, Vec3::new(0.2, 0.12, 0.08), Vec3::new(0.06, 0.04, 0.03)).roughness(0.85, 0.95).param(0.2).scale(1.4).bump(1.8).seed(276),
            "leaf" => TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.13, 0.25, 0.04), Vec3::new(0.22, 0.28, 0.05)).roughness(0.55, 0.6).seed(277),
            "needle" => TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.035, 0.085, 0.03), Vec3::new(0.06, 0.12, 0.04)).scale(3.0).roughness(0.6, 0.6).seed(278),
            "blade" => TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.1, 0.18, 0.035), Vec3::new(0.16, 0.22, 0.05)).roughness(0.6, 0.6).seed(279),
            "fern" => TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.1, 0.24, 0.04), Vec3::new(0.16, 0.3, 0.05)).roughness(0.55, 0.6).seed(280),
            // Near-white veined petals, tinted by the material's base colour.
            _ => TextureDesc::new(TexturePattern::Leaf, Vec3::new(0.86, 0.84, 0.8), Vec3::new(0.98, 0.97, 0.95)).roughness(0.5, 0.55).seed(281),
        };
        let handle = scene.add_texture(desc);
        self.textures.insert(key, handle);
        handle
    }

    fn builtin_material(&mut self, scene: &mut Scene, name: &str) -> Option<Material> {
        let foliage = |texture, tint: Vec3, wind| Material { base_color: tint, roughness: 0.6, wind, ..Material::textured(MaterialKind::Foliage, texture, TextureMapping::MeshUv, 1.0) };
        Some(match name {
            "default" => Material::standard(Vec3::splat(0.5), 0.55, 0.0),
            "white" => Material::standard(Vec3::splat(0.8), 0.45, 0.0),
            "rock" => Material {
                kind: MaterialKind::Rock,
                base_color: Vec3::new(0.32, 0.29, 0.25),
                roughness: 0.8,
                detail_scale: 7.0,
                detail_strength: 0.35,
                texture: Some(self.texture(scene, "rock")),
                mapping: TextureMapping::ObjectTriplanar,
                uv_scale: 0.6,
                ..Default::default()
            },
            "gold" => Material {
                kind: MaterialKind::Metal,
                base_color: Vec3::new(1.0, 0.76, 0.42),
                roughness: 0.22,
                metallic: 1.0,
                detail_scale: 3.0,
                texture: Some(self.texture(scene, "hammered")),
                mapping: TextureMapping::ObjectTriplanar,
                uv_scale: 1.2,
                normal_strength: 0.6,
                ..Default::default()
            },
            "glass" => Material { kind: MaterialKind::Glass, base_color: Vec3::new(0.85, 0.95, 0.92), roughness: 0.05, ..Default::default() },
            "emissive" => Material { emissive: 6.0, ..Material::standard(Vec3::new(1.0, 0.7, 0.4), 0.5, 0.0) },
            "bark" => Material { wind: WindResponse::WOOD, ..Material::textured(MaterialKind::Standard, self.texture(scene, "bark"), TextureMapping::MeshUv, 1.2) },
            "pine-bark" => Material { wind: WindResponse::WOOD, ..Material::textured(MaterialKind::Standard, self.texture(scene, "pine-bark"), TextureMapping::MeshUv, 1.2) },
            "leaves" => foliage(self.texture(scene, "leaf"), Vec3::ONE, WindResponse::LEAVES),
            "needles" => foliage(self.texture(scene, "needle"), Vec3::ONE, WindResponse::LEAVES),
            "grass" => foliage(self.texture(scene, "blade"), Vec3::ONE, WindResponse::GRASS),
            "fern" => foliage(self.texture(scene, "fern"), Vec3::ONE, WindResponse::FERN),
            "stem" => foliage(self.texture(scene, "blade"), Vec3::new(0.8, 1.0, 0.7), WindResponse::GRASS),
            petals if petals.starts_with("petals-") => {
                let tint = match &petals["petals-".len()..] {
                    "red" => Vec3::new(0.75, 0.06, 0.05),
                    "yellow" => Vec3::new(0.95, 0.72, 0.08),
                    "blue" => Vec3::new(0.18, 0.28, 0.85),
                    "white" => Vec3::new(0.92, 0.92, 0.9),
                    _ => return None,
                };
                foliage(self.texture(scene, "petal"), tint, WindResponse::GRASS)
            }
            _ => return None,
        })
    }
}

/// A built-in mesh and its distance field mode (thin vegetation gets none, as in the forest).
fn builtin_mesh(name: &str) -> Option<(Mesh, SdfMode)> {
    let volume = SdfMode::Volume { resolution: 40 };
    Some(match name {
        "cube" => (cuboid(Vec3::splat(-0.5), Vec3::splat(0.5)), volume),
        "sphere" => {
            let mut sphere = primitives::icosphere(4);
            sphere.deform(|p| p * 0.5);
            (sphere, volume)
        }
        "cylinder" => {
            let mut cylinder = primitives::cylinder(48, 1, 0.5, 1.0, true);
            cylinder.deform(|p| p - Vec3::Y * 0.5);
            (cylinder, volume)
        }
        "plane" => (primitives::terrain(10.0, 8, 0.0, 1), SdfMode::Heightfield { resolution: 64 }),
        "rock" => (primitives::rock(5, 5), volume),
        "boulder" => (primitives::rock(23, 5), volume),
        "grass" => (vegetation::grass_clump(191, 40, 0.45), SdfMode::None),
        "fern" => (vegetation::fern(181, 9, 0.9, 14), SdfMode::None),
        "flower-stem" => (flower_stem(), SdfMode::None),
        "flower-head" => (flower_head(), SdfMode::None),
        "bush-wood" | "bush-leaves" => {
            let bush = vegetation::bush(161, 1.4, 10);
            (if name == "bush-wood" { bush.wood } else { bush.foliage }, SdfMode::None)
        }
        "tree-wood" | "tree-leaves" => {
            let tree = vegetation::broadleaf_tree(&TreeParams {
                seed: 118,
                height: 12.0,
                trunk_radius: 0.3,
                depth: 3,
                children: 4,
                leaves_per_twig: 24,
                leaf: LeafShape { length: 0.3, width: 0.18, fold: 0.35 },
                sides: 10,
                gravitropism: 0.35,
            });
            (if name == "tree-wood" { tree.wood } else { tree.foliage }, volume)
        }
        "conifer-wood" | "conifer-needles" => {
            let tree = vegetation::conifer(141, 14.0, 0.25, 30, 10);
            (if name == "conifer-wood" { tree.wood } else { tree.foliage }, volume)
        }
        _ => return None,
    })
}

const STEM_HEIGHT: f32 = 0.38;

/// A flower's stem with two leaves (UV along the stem for the blade texture).
fn flower_stem() -> Mesh {
    let mut b = MeshBuilder::default();
    let stem = primitives::cylinder(6, 4, 0.004, STEM_HEIGHT, false);
    b.append(&stem);
    for (angle, height) in [(0.3f32, 0.08f32), (2.6, 0.15)] {
        let dir = Vec3::new(angle.cos(), 0.9, angle.sin()).normalize();
        vegetation::add_leaf(&mut b, Vec3::Y * height, dir, Vec3::Y.cross(dir).normalize(), LeafShape { length: 0.11, width: 0.03, fold: 0.2 });
    }
    b.build()
}

/// Petals around a small centre, facing up and outwards at the top of the stem.
fn flower_head() -> Mesh {
    let mut b = MeshBuilder::default();
    let top = Vec3::Y * STEM_HEIGHT;
    for i in 0..7 {
        let angle = i as f32 / 7.0 * std::f32::consts::TAU;
        let out = Vec3::new(angle.cos(), 0.0, angle.sin());
        let dir = (out + Vec3::Y * 0.35).normalize();
        vegetation::add_leaf(&mut b, top, dir, Vec3::Y, LeafShape { length: 0.045, width: 0.03, fold: 0.25 });
    }
    let centre = primitives::icosphere(1).transformed(Mat4::from_scale_rotation_translation(Vec3::new(0.012, 0.007, 0.012), Quat::IDENTITY, top));
    b.append(&centre);
    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_and_material_resolves() {
        for item in items() {
            for part in &item.parts {
                assert!(builtin_mesh(part.mesh.strip_prefix("engine/").unwrap()).is_some(), "{}", part.mesh);
                assert!(MATERIALS.contains(&part.material.as_str()), "{}", part.material);
            }
        }
        let mut scene = Scene::new();
        let mut content = Content::default();
        for name in MATERIALS {
            let handle = content.material(&mut scene, name).unwrap();
            assert_eq!(scene.material_name(handle), *name);
        }
        // Created once, then found by name.
        let count = scene.material_count();
        content.material(&mut scene, "engine/rock").unwrap();
        assert_eq!(scene.material_count(), count);
        assert!(content.material(&mut scene, "engine/unobtainium").is_none());
    }
}
