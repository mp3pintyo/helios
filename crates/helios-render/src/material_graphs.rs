//! Material graphs on a [`Scene`] (UE's materials and material instances).
//!
//! A graph ([`helios_material::MaterialGraph`]) is compiled to WGSL when it is added or
//! changed; materials refer to it through [`Material::graph`](crate::Material::graph), and the
//! renderer compiles one extended resolve pipeline per distinct graph (by the hash of its code,
//! so materials sharing a graph — or two identical graphs — share a pipeline). A material is
//! also an *instance* of its graph: [`Scene::set_material_parameter`] overrides the graph's
//! parameters for that material only. Parameter values live in a uniform block, so changing
//! one never recompiles anything.
//!
//! A graph whose Displacement output is connected also gives the height of its displaced
//! materials (Nanite tessellation: see `tessellation/graphs.rs`); its parameters update those
//! surfaces the same way.

use std::collections::BTreeMap;

use helios_material::{CompiledGraph, GraphError, MaterialGraph, ParameterKind};

use crate::scene::{next_version, MaterialHandle, Scene, TextureHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialGraphHandle(pub(crate) u32);

impl MaterialGraphHandle {
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A material instance's value for a graph parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParameterValue {
    Scalar(f32),
    /// A colour or vector (rgba).
    Vector([f32; 4]),
    Texture(TextureHandle),
}

impl ParameterValue {
    pub fn kind(&self) -> ParameterKind {
        match self {
            ParameterValue::Scalar(_) => ParameterKind::Scalar,
            ParameterValue::Vector(_) => ParameterKind::Vector,
            ParameterValue::Texture(_) => ParameterKind::Texture,
        }
    }
}

pub(crate) struct GraphEntry {
    pub graph: MaterialGraph,
    pub compiled: Result<CompiledGraph, Vec<GraphError>>,
}

impl GraphEntry {
    /// The graph compiles with a Displacement output.
    fn displaces(&self) -> bool {
        self.compiled.as_ref().is_ok_and(|c| c.displacement.is_some())
    }
}

/// The scene's graphs and its materials' parameter overrides.
#[derive(Default)]
pub(crate) struct GraphStore {
    pub entries: Vec<GraphEntry>,
    /// Per material index: overridden parameters by name.
    pub parameters: Vec<BTreeMap<String, ParameterValue>>,
    /// Bumped when a graph is added or changed (pipelines may need compiling).
    pub version: u64,
    /// Bumped when a parameter value changes (the parameter block is rewritten).
    pub parameter_version: u64,
    /// Bumped when a graph starts or stops driving displacement (instances' displaced flags).
    pub displacement_version: u64,
}

impl Scene {
    /// Adds a graph (compiling it; see [`Scene::material_graph_errors`]).
    pub fn add_material_graph(&mut self, graph: MaterialGraph) -> MaterialGraphHandle {
        let entry = GraphEntry { compiled: helios_material::compile(&graph), graph };
        if entry.displaces() {
            self.graphs.displacement_version = next_version();
        }
        self.graphs.entries.push(entry);
        self.graphs.version = next_version();
        MaterialGraphHandle(self.graphs.entries.len() as u32 - 1)
    }

    /// Replaces a graph. Materials using it switch to the new code once it compiles; while it
    /// does not, they keep rendering without it.
    pub fn set_material_graph(&mut self, handle: MaterialGraphHandle, graph: MaterialGraph) {
        let entry = &mut self.graphs.entries[handle.0 as usize];
        if entry.graph != graph {
            let displaced = entry.displaces();
            entry.compiled = helios_material::compile(&graph);
            entry.graph = graph;
            if entry.displaces() != displaced {
                self.graphs.displacement_version = next_version();
            }
            self.graphs.version = next_version();
        }
    }

    pub fn material_graph(&self, handle: MaterialGraphHandle) -> &MaterialGraph {
        &self.graphs.entries[handle.0 as usize].graph
    }

    /// The compiled graph, or `None` if it has errors.
    pub fn compiled_material_graph(&self, handle: MaterialGraphHandle) -> Option<&CompiledGraph> {
        self.graphs.entries.get(handle.0 as usize).and_then(|e| e.compiled.as_ref().ok())
    }

    /// Why the graph does not compile (empty if it does).
    pub fn material_graph_errors(&self, handle: MaterialGraphHandle) -> &[GraphError] {
        match &self.graphs.entries[handle.0 as usize].compiled {
            Ok(_) => &[],
            Err(errors) => errors,
        }
    }

    /// Whether the graph compiles with its Displacement output connected: it then gives the
    /// height of its materials that displace ([`crate::Material::displacement`]: magnitude and
    /// centre) instead of a texture channel.
    pub fn material_graph_displaces(&self, handle: MaterialGraphHandle) -> bool {
        self.graphs.entries.get(handle.0 as usize).is_some_and(GraphEntry::displaces)
    }

    pub fn material_graph_count(&self) -> usize {
        self.graphs.entries.len()
    }

    pub fn material_graph_handle(&self, index: u32) -> Option<MaterialGraphHandle> {
        ((index as usize) < self.graphs.entries.len()).then_some(MaterialGraphHandle(index))
    }

    /// The first graph named `name`.
    pub fn find_material_graph(&self, name: &str) -> Option<MaterialGraphHandle> {
        self.graphs.entries.iter().position(|e| e.graph.name == name).map(|i| MaterialGraphHandle(i as u32))
    }

    /// Overrides a graph parameter for `material` (a material instance). Takes effect without
    /// recompiling; values of the wrong kind are ignored by the renderer.
    pub fn set_material_parameter(&mut self, material: MaterialHandle, name: &str, value: ParameterValue) {
        let index = material.0 as usize;
        if self.graphs.parameters.len() <= index {
            self.graphs.parameters.resize(index + 1, BTreeMap::new());
        }
        if self.graphs.parameters[index].insert(name.to_string(), value) != Some(value) {
            self.graphs.parameter_version = next_version();
        }
    }

    /// Returns a parameter of `material` to the graph's default.
    pub fn clear_material_parameter(&mut self, material: MaterialHandle, name: &str) {
        if let Some(params) = self.graphs.parameters.get_mut(material.0 as usize) {
            if params.remove(name).is_some() {
                self.graphs.parameter_version = next_version();
            }
        }
    }

    /// The parameters `material` overrides.
    pub fn material_parameters(&self, material: MaterialHandle) -> impl Iterator<Item = (&str, ParameterValue)> {
        self.graphs.parameters.get(material.0 as usize).into_iter().flatten().map(|(k, v)| (k.as_str(), *v))
    }

    pub fn material_parameter(&self, material: MaterialHandle, name: &str) -> Option<ParameterValue> {
        self.graphs.parameters.get(material.0 as usize).and_then(|p| p.get(name)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Material;

    #[test]
    fn graphs_compile_when_added_and_report_errors() {
        let mut scene = Scene::new();
        let lava = scene.add_material_graph(helios_material::presets::lava());
        assert!(scene.compiled_material_graph(lava).is_some());
        assert!(scene.material_graph_errors(lava).is_empty());
        let version = scene.graphs.version;
        scene.set_material_graph(lava, MaterialGraph { name: "broken".into(), ..Default::default() });
        assert!(scene.compiled_material_graph(lava).is_none());
        assert_eq!(scene.material_graph_errors(lava)[0].message, "the graph has no Output node");
        assert_ne!(scene.graphs.version, version);
        assert_eq!(scene.find_material_graph("broken"), Some(lava));
    }

    #[test]
    fn gaining_or_losing_a_displacement_output_is_versioned() {
        let mut scene = Scene::new();
        let version = scene.graphs.displacement_version;
        let lava = scene.add_material_graph(helios_material::presets::lava());
        assert_eq!(scene.graphs.displacement_version, version, "lava does not displace");
        scene.set_material_graph(lava, helios_material::presets::brick_pavement());
        assert_ne!(scene.graphs.displacement_version, version, "bricks displace");
        let version = scene.graphs.displacement_version;
        let mut bricks = helios_material::presets::brick_pavement();
        bricks.nodes[0].pos = [1.0, 2.0];
        scene.set_material_graph(lava, bricks);
        assert_eq!(scene.graphs.displacement_version, version, "still displacing");
        scene.set_material_graph(lava, MaterialGraph { name: "broken".into(), ..Default::default() });
        assert_ne!(scene.graphs.displacement_version, version, "a graph that stops compiling stops displacing");
    }

    #[test]
    fn parameters_are_per_material_and_versioned() {
        let mut scene = Scene::new();
        let a = scene.add_material(Material::default());
        let b = scene.add_material(Material::default());
        let version = scene.graphs.parameter_version;
        scene.set_material_parameter(b, "Glow", ParameterValue::Scalar(3.0));
        assert_ne!(scene.graphs.parameter_version, version);
        let version = scene.graphs.parameter_version;
        scene.set_material_parameter(b, "Glow", ParameterValue::Scalar(3.0));
        assert_eq!(scene.graphs.parameter_version, version, "an unchanged value is not a change");
        assert_eq!(scene.material_parameter(b, "Glow"), Some(ParameterValue::Scalar(3.0)));
        assert_eq!(scene.material_parameters(a).count(), 0);
        scene.clear_material_parameter(b, "Glow");
        assert_eq!(scene.material_parameter(b, "Glow"), None);
    }
}
