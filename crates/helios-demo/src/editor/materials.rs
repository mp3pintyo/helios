//! Material graphs and material instances in the editor: graphs created or edited in the
//! document (by name), a material's graph assignment, its graph parameter values and its
//! Substrate slab overrides.

use helios_render::material_graph::{MaterialGraph, ParameterKind};
use helios_render::{Displacement, Material, MaterialHandle, ParameterValue, Scene};
use serde_json::{json, Value};

use super::doc::{DisplacementDef, MaterialParams, ParameterDef, SlabDef};
use super::{Editor, World};

impl Editor {
    /// Applies the document's graphs: new names are added to the scene, existing ones replaced;
    /// a graph removed from the document gets the base scene's version back.
    pub(super) fn sync_graphs(&mut self, world: &mut World) {
        let names: std::collections::BTreeSet<String> = self.doc.graphs.keys().chain(self.applied.graphs.keys()).cloned().collect();
        for name in names {
            let wanted = self.doc.graphs.get(&name);
            if wanted == self.applied.graphs.get(&name) {
                continue;
            }
            let scene = &mut *world.scene;
            match (scene.find_material_graph(&name), wanted) {
                (Some(handle), Some(graph)) => {
                    self.base_graphs.entry(name.clone()).or_insert_with(|| scene.material_graph(handle).clone());
                    scene.set_material_graph(handle, graph.clone());
                }
                (None, Some(graph)) => {
                    scene.add_material_graph(graph.clone());
                }
                (Some(handle), None) => {
                    if let Some(base) = self.base_graphs.get(&name) {
                        scene.set_material_graph(handle, base.clone());
                    }
                }
                (None, None) => {}
            }
        }
    }

    /// Sets material `name` to `base` with `params` applied: fields, slab, graph and the
    /// graph's parameter values (the base scene's values, then the document's).
    pub(super) fn apply_material(&mut self, scene: &mut Scene, name: &str, handle: MaterialHandle, base: Material, params: &MaterialParams) {
        let mut material = with_params(base, params);
        material.graph = match params.graph.as_deref() {
            None => base.graph,
            Some("") => None,
            Some(graph) => scene.find_material_graph(graph).or_else(|| {
                self.warnings.push(format!("material '{name}': unknown graph '{graph}'"));
                base.graph
            }),
        };
        scene.set_material(handle, material);
        let base_parameters = self
            .base_parameters
            .entry(name.to_string())
            .or_insert_with(|| scene.material_parameters(handle).map(|(k, v)| (k.to_string(), v)).collect())
            .clone();
        let current: Vec<String> = scene.material_parameters(handle).map(|(k, _)| k.to_string()).collect();
        for key in current {
            scene.clear_material_parameter(handle, &key);
        }
        for (key, value) in base_parameters {
            scene.set_material_parameter(handle, &key, value);
        }
        for (key, value) in &params.parameters {
            let value = match *value {
                ParameterDef::Scalar(x) => ParameterValue::Scalar(x),
                ParameterDef::Vector(v) => ParameterValue::Vector(v),
            };
            scene.set_material_parameter(handle, key, value);
        }
    }
}

/// A material with field and slab overrides (not the graph: see [`Editor::apply_material`]).
pub fn with_params(base: Material, p: &MaterialParams) -> Material {
    use glam::Vec3;
    let mut slab = base.slab;
    if let Some(s) = &p.slab {
        let f = |o: Option<f32>, v: f32| o.unwrap_or(v);
        let c = |o: Option<[f32; 3]>, v: Vec3| o.map_or(v, Vec3::from);
        slab.ior = f(s.ior, slab.ior);
        slab.specular = f(s.specular, slab.specular);
        slab.clearcoat = f(s.clearcoat, slab.clearcoat);
        slab.clearcoat_roughness = f(s.clearcoat_roughness, slab.clearcoat_roughness);
        slab.sheen_color = c(s.sheen_color, slab.sheen_color);
        slab.sheen_roughness = f(s.sheen_roughness, slab.sheen_roughness);
        slab.subsurface = f(s.subsurface, slab.subsurface);
        slab.subsurface_color = c(s.subsurface_color, slab.subsurface_color);
        slab.subsurface_radius = f(s.subsurface_radius, slab.subsurface_radius);
        slab.transmission = f(s.transmission, slab.transmission);
        slab.anisotropy = f(s.anisotropy, slab.anisotropy);
        slab.unlit = s.unlit.unwrap_or(slab.unlit);
    }
    let mut displacement = base.displacement;
    if let Some(d) = &p.displacement {
        let mut value = displacement.unwrap_or(Displacement::height(0.0, 0.5));
        value.magnitude = d.magnitude.unwrap_or(value.magnitude);
        value.center = d.center.unwrap_or(value.center).clamp(0.0, 1.0);
        value.channel = d.channel.unwrap_or(value.channel).min(3);
        displacement = (value.magnitude != 0.0).then_some(value);
    }
    Material {
        base_color: p.base_color.map_or(base.base_color, Vec3::from),
        roughness: p.roughness.unwrap_or(base.roughness),
        metallic: p.metallic.unwrap_or(base.metallic),
        opacity: p.opacity.unwrap_or(base.opacity),
        emissive: p.emissive.unwrap_or(base.emissive),
        emissive_color: p.emissive_color.map_or(base.emissive_color, Vec3::from),
        normal_strength: p.normal_strength.unwrap_or(base.normal_strength),
        uv_scale: p.uv_scale.unwrap_or(base.uv_scale),
        slab,
        displacement,
        ..base
    }
}

/// A material's displacement as the Details panel edits it (magnitude 0: none).
pub fn displacement_value(m: &Material) -> Value {
    let d = m.displacement.unwrap_or(Displacement::height(0.0, 0.5));
    json!(DisplacementDef { magnitude: Some(d.magnitude), center: Some(d.center), channel: Some(d.channel) })
}

/// A material's slab as the Details panel edits it.
pub fn slab_value(m: &Material) -> Value {
    let s = &m.slab;
    json!(SlabDef {
        ior: Some(s.ior),
        specular: Some(s.specular),
        clearcoat: Some(s.clearcoat),
        clearcoat_roughness: Some(s.clearcoat_roughness),
        sheen_color: Some(s.sheen_color.to_array()),
        sheen_roughness: Some(s.sheen_roughness),
        subsurface: Some(s.subsurface),
        subsurface_color: Some(s.subsurface_color.to_array()),
        subsurface_radius: Some(s.subsurface_radius),
        transmission: Some(s.transmission),
        anisotropy: Some(s.anisotropy),
        unlit: Some(s.unlit),
    })
}

/// A material's graph and its parameters (definition and the material's value), for the UI.
pub fn graph_value(scene: &Scene, handle: MaterialHandle) -> (Value, Value) {
    let m = scene.material(handle);
    let Some(graph) = m.graph else { return (Value::Null, json!([])) };
    let name = scene.material_graph(graph).name.clone();
    let parameters: Vec<Value> = scene
        .compiled_material_graph(graph)
        .map(|c| {
            c.parameters
                .iter()
                .map(|p| {
                    let value = match (p.kind, scene.material_parameter(handle, &p.name)) {
                        (ParameterKind::Scalar, Some(ParameterValue::Scalar(x))) => json!(x),
                        (ParameterKind::Vector, Some(ParameterValue::Vector(v))) => json!(v),
                        (ParameterKind::Scalar, _) => json!(p.default[0]),
                        (ParameterKind::Vector, _) => json!(p.default),
                        (ParameterKind::Texture, Some(ParameterValue::Texture(_))) => json!("texture set by the scene"),
                        (ParameterKind::Texture, _) => json!("the material's texture"),
                    };
                    json!({ "name": p.name, "kind": p.kind, "range": p.range, "value": value })
                })
                .collect()
        })
        .unwrap_or_default();
    (json!(name), json!(parameters))
}

/// What a graph's displacement output is, for the editor: `"height"` (the Displacement pin),
/// `"vector"` (VectorDisplacement) or null, and whether it is animated (reads the time).
pub fn graph_displacement(scene: &Scene, graph: helios_render::MaterialGraphHandle) -> (Value, bool) {
    match scene.compiled_material_graph(graph).filter(|c| c.displacement.is_some()) {
        Some(c) => (json!(c.displacement_kind), c.displacement_animated),
        None => (Value::Null, false),
    }
}

/// Every graph of the scene: its definition, compile errors, parameters and displacement.
pub fn graphs_value(scene: &Scene) -> Value {
    let graphs: Vec<Value> = (0..scene.material_graph_count() as u32)
        .filter_map(|i| scene.material_graph_handle(i))
        .map(|h| {
            let graph: &MaterialGraph = scene.material_graph(h);
            let compiled = scene.compiled_material_graph(h);
            let (displacement, animated) = graph_displacement(scene, h);
            json!({
                "name": graph.name,
                "graph": graph,
                "errors": scene.material_graph_errors(h),
                "parameters": compiled.map(|c| json!(c.parameters)).unwrap_or(json!([])),
                "hash": compiled.map(|c| format!("{:016x}", c.hash)),
                "displacement": displacement,
                "displacement_animated": animated,
            })
        })
        .collect();
    Value::Array(graphs)
}
