//! # helios-material
//!
//! Material graphs in the spirit of Unreal Engine's Material Editor: a node graph whose output
//! node sets a surface's attributes — base colour, metallic, roughness, specular, normal,
//! emission, ambient occlusion and the Substrate slab parameters (clear coat, sheen,
//! subsurface, transmission, anisotropy) — compiled to a WGSL function the renderer's G-buffer
//! resolve calls per pixel.
//!
//! ```text
//! WorldPosition ─ ComponentMask(xz) ─ Panner ─ Append ─ Noise ─ SmoothStep ─┬─ Lerp ─ BaseColor
//!                                         Time ─ Multiply ─┘                └─ … ─── Emissive
//! ```
//!
//! * **Data** ([`MaterialGraph`]): nodes ([`Op`]) linked output pin → input pin, plain `serde`
//!   data, so graphs live in scene files. Unconnected inputs take defaults (constants a node can
//!   override, or the mesh UVs, world position, normal, time).
//! * **Validation and code generation** ([`compile`]): ids, links, pins and cycles are
//!   checked, value widths inferred (scalars broadcast), and every problem reported with the
//!   node it concerns. The generated code carries screen-space derivatives through the values
//!   that feed textures and noise (see [`compile`]'s module docs).
//! * **Parameters** ([`Parameter`]): scalar, vector and texture parameters get slots in a
//!   parameter block that material instances fill — changing one never recompiles.
//! * **Catalog** ([`catalog`]): pin layouts and default properties of every node type, which an
//!   editor builds its palette from.

mod compile;
mod graph;
mod nodes;
pub mod presets;

pub use compile::{compile, CompiledGraph, GraphError, Parameter, ParameterKind};
pub use graph::{GraphBuilder, Link, MaterialGraph, Node, Op, PinRef};
pub use nodes::{catalog, output_bit, spec, CatalogEntry, InputDefault, InputSpec, NodeSpec, OUTPUT_PINS};

#[cfg(test)]
mod tests {
    use super::*;

    fn simple() -> (GraphBuilder, u32) {
        let mut g = GraphBuilder::new("test");
        let output = g.node(Op::Output, [0.0, 0.0]);
        (g, output)
    }

    #[test]
    fn presets_compile() {
        for (id, _) in presets::PRESETS {
            let graph = presets::preset(id).unwrap();
            let compiled = compile(&graph).unwrap_or_else(|e| panic!("{id}: {e:?}"));
            assert!(compiled.wgsl.contains("fn graph_eval"));
        }
        let lava = compile(&presets::lava()).unwrap();
        assert_eq!(lava.outputs, output_bit("BaseColor") | output_bit("Roughness") | output_bit("Emissive"));
        let names: Vec<&str> = lava.parameters.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["Glow Color", "Glow Intensity"]);
    }

    #[test]
    fn graphs_round_trip_through_json() {
        for (id, _) in presets::PRESETS {
            let graph = presets::preset(id).unwrap();
            let json = serde_json::to_string(&graph).unwrap();
            let back: MaterialGraph = serde_json::from_str(&json).unwrap();
            assert_eq!(back, graph, "{id}");
        }
        let parsed: MaterialGraph = serde_json::from_str(
            r#"{"name":"t","nodes":[{"id":1,"type":"Constant","value":0.5},{"id":2,"type":"Output","pos":[10,0]}],
               "links":[{"from":{"node":1,"pin":"Out"},"to":{"node":2,"pin":"Roughness"}}]}"#,
        )
        .unwrap();
        assert!(compile(&parsed).unwrap().wgsl.contains("o.roughness = 0.5f;"));
    }

    #[test]
    fn identical_graphs_share_a_hash_and_parameters_do_not_change_it() {
        let a = compile(&presets::lava()).unwrap();
        let b = compile(&presets::lava()).unwrap();
        assert_eq!(a.hash, b.hash);
        let mut moved = presets::lava();
        moved.nodes[0].pos = [5.0, 5.0];
        for node in &mut moved.nodes {
            if let Op::ScalarParameter { default, .. } = &mut node.op {
                *default = 3.0;
            }
        }
        assert_eq!(compile(&moved).unwrap().hash, a.hash, "positions and parameter defaults are not code");
        assert_ne!(compile(&presets::wet_stone()).unwrap().hash, a.hash);
    }

    #[test]
    fn structural_errors_name_their_node() {
        let mut g = MaterialGraph { name: "bad".into(), ..Default::default() };
        assert_eq!(compile(&g).unwrap_err()[0].message, "the graph has no Output node");

        let (mut b, output) = simple();
        let add = b.node(Op::Add, [0.0, 0.0]);
        b.link(add, "Out", output, "Nope").link(99, "Out", add, "A").link(add, "Out", add, "B");
        g = b.build();
        let errors = compile(&g).unwrap_err();
        let text: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        assert!(text.iter().any(|t| t.contains("Output has no input 'Nope'")), "{text:?}");
        assert!(text.iter().any(|t| t.contains("node #99, which does not exist")), "{text:?}");
    }

    #[test]
    fn cycles_are_reported() {
        let (mut b, output) = simple();
        let a = b.node(Op::Add, [0.0, 0.0]);
        let m = b.node(Op::Multiply, [0.0, 0.0]);
        b.link(a, "Out", m, "A").link(m, "Out", a, "A").link(m, "Out", output, "Roughness");
        let errors = compile(&b.build()).unwrap_err();
        assert!(errors[0].message.contains("cycle"), "{errors:?}");
    }

    #[test]
    fn type_errors_are_explained() {
        let (mut b, output) = simple();
        let uv = b.node(Op::TexCoord { set: 0, tiling: [1.0, 1.0] }, [0.0, 0.0]);
        let color = b.node(Op::Constant3 { value: [1.0, 0.0, 0.0] }, [0.0, 0.0]);
        let add = b.node(Op::Add, [0.0, 0.0]);
        let mask = b.node(Op::ComponentMask { channels: "xz".into() }, [0.0, 0.0]);
        b.link(uv, "Out", add, "A").link(color, "Out", add, "B").link(add, "Out", output, "BaseColor");
        b.link(uv, "Out", mask, "In").link(mask, "Out", output, "Emissive");
        let errors = compile(&b.build()).unwrap_err();
        let text: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        assert!(text.iter().any(|t| t.contains(&format!("node #{add}: Add: cannot combine a vec2<f32> with a vec3<f32>"))), "{text:?}");
        assert!(text.iter().any(|t| t.contains("component 'z' does not exist in a vec2<f32>")), "{text:?}");

        let (mut b, output) = simple();
        let color = b.node(Op::Constant3 { value: [1.0, 0.0, 0.0] }, [0.0, 0.0]);
        b.link(color, "Out", output, "Roughness");
        let errors = compile(&b.build()).unwrap_err();
        assert!(errors[0].message.contains("Roughness takes a f32, not a vec3<f32>"), "{errors:?}");
    }

    #[test]
    fn parameter_kinds_must_agree() {
        let (mut b, output) = simple();
        let s = b.node(Op::ScalarParameter { name: "Tint".into(), default: 1.0, min: 0.0, max: 1.0 }, [0.0, 0.0]);
        let v = b.node(Op::VectorParameter { name: "Tint".into(), default: [1.0; 4] }, [0.0, 0.0]);
        b.link(s, "Out", output, "Roughness").link(v, "Out", output, "BaseColor");
        let errors = compile(&b.build()).unwrap_err();
        assert!(errors[0].message.contains("parameter 'Tint'"), "{errors:?}");
    }

    #[test]
    fn scalars_broadcast_and_constants_override_defaults() {
        let (mut b, output) = simple();
        let m = b.node(Op::Multiply, [0.0, 0.0]);
        let c = b.node(Op::Constant3 { value: [0.5, 0.25, -1.0] }, [0.0, 0.0]);
        b.link(c, "Out", m, "A").value(m, "B", 2.0).link(m, "Out", output, "BaseColor");
        let wgsl = compile(&b.build()).unwrap().wgsl;
        assert!(wgsl.contains("(vec3<f32>(0.5f, 0.25f, (-1.0f)) * vec3<f32>(2.0f))"), "{wgsl}");
    }

    #[test]
    fn only_texture_coordinates_carry_derivatives() {
        let (mut b, output) = simple();
        let uv = b.node(Op::TexCoord { set: 0, tiling: [4.0, 4.0] }, [0.0, 0.0]);
        let pan = b.node(Op::Panner { speed: [0.1, 0.0] }, [0.0, 0.0]);
        let tex = b.node(Op::TextureSample { parameter: "Albedo".into() }, [0.0, 0.0]);
        let rough = b.node(Op::Multiply, [0.0, 0.0]);
        b.link(uv, "Out", pan, "UV").link(pan, "Out", tex, "UV").link(tex, "RGB", output, "BaseColor");
        b.link(tex, "R", rough, "A").link(rough, "Out", output, "Roughness");
        let compiled = compile(&b.build()).unwrap();
        let wgsl = &compiled.wgsl;
        assert!(wgsl.contains(&format!("let v{pan}_0_dx")), "{wgsl}");
        assert!(wgsl.contains(&format!("let v{uv}_0_dx: vec2<f32> = (input.duv_dx * vec2<f32>(4.0f, 4.0f));")), "{wgsl}");
        assert!(!wgsl.contains(&format!("v{rough}_0_dx")), "values that feed no texture carry no derivatives");
        assert_eq!(compiled.parameters[0].kind, ParameterKind::Texture);
    }

    #[test]
    fn the_catalog_lists_every_node_type() {
        let catalog = catalog();
        assert_eq!(catalog.len(), Op::palette().len());
        let json = serde_json::to_value(&catalog).unwrap();
        let lerp = json.as_array().unwrap().iter().find(|e| e["type"] == "Lerp").unwrap();
        assert_eq!(lerp["inputs"][2]["name"], "Alpha");
        assert_eq!(lerp["template"]["type"], "Lerp");
        assert_eq!(json.as_array().unwrap().iter().find(|e| e["type"] == "Output").unwrap()["inputs"].as_array().unwrap().len(), OUTPUT_PINS.len());
    }
}
