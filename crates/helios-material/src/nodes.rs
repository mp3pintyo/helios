//! Pin layout of every operation: input pins with the value they take when unconnected, output
//! pins, and the palette category. The compiler and the editor's catalog both read it.

use serde::Serialize;

use crate::graph::Op;

/// What an unconnected input reads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum InputDefault {
    /// A constant (overridable per node through [`crate::Node::values`]).
    Scalar(f32),
    /// A constant vector.
    Vector3([f32; 3]),
    /// Mesh UV set 0.
    TexCoord,
    WorldPosition,
    VertexNormal,
    Time,
    /// The material keeps its own value (output pins).
    Material,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct InputSpec {
    pub name: &'static str,
    pub default: InputDefault,
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeSpec {
    pub category: &'static str,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<&'static str>,
}

/// Output pins of the [`Op::Output`] node: name, WGSL field of `GraphOutput`, components. The
/// last two, [`DISPLACEMENT_PIN`] and [`VECTOR_DISPLACEMENT_PIN`], are not `GraphOutput` fields:
/// the one that is connected compiles to a function of its own (see
/// [`crate::CompiledGraph::displacement`]).
pub const OUTPUT_PINS: [(&str, &str, u8); 16] = [
    ("BaseColor", "base_color", 3),
    ("Metallic", "metallic", 1),
    ("Roughness", "roughness", 1),
    ("Specular", "specular", 1),
    ("Normal", "normal", 3),
    ("Emissive", "emissive", 3),
    ("AmbientOcclusion", "occlusion", 1),
    ("Clearcoat", "clearcoat", 1),
    ("ClearcoatRoughness", "clearcoat_roughness", 1),
    ("SheenColor", "sheen_color", 3),
    ("SheenRoughness", "sheen_roughness", 1),
    ("Subsurface", "subsurface", 1),
    ("Transmission", "transmission", 1),
    ("Anisotropy", "anisotropy", 1),
    (DISPLACEMENT_PIN, "displacement", 1),
    (VECTOR_DISPLACEMENT_PIN, "vector_displacement", 3),
];

/// The output pin whose height (0..1 around the material's displacement centre, scaled by its
/// magnitude) displaces the surface along its normal through Nanite tessellation.
pub const DISPLACEMENT_PIN: &str = "Displacement";

/// The output pin whose vector displaces the surface in any direction (overhangs, undercuts,
/// sideways shifts): a world-space offset in units of the material's displacement magnitude,
/// (0, 0, 0) leaving the point in place; its length is clamped to 1, so the magnitude bounds the
/// displacement. `VertexNormal` is the world normal, so `VertexNormal * height` displaces as the
/// height pin does. A graph connects this pin or [`DISPLACEMENT_PIN`], not both.
pub const VECTOR_DISPLACEMENT_PIN: &str = "VectorDisplacement";

/// Bit of [`OUTPUT_PINS`] entry `name` in [`crate::CompiledGraph::outputs`].
pub fn output_bit(name: &str) -> u32 {
    OUTPUT_PINS.iter().position(|p| p.0 == name).map_or(0, |i| 1 << i)
}

/// "Height" and "Metallic" are the same channel: a procedural texture's height, an image layer's
/// metallic.
const TEXTURE_OUTPUTS: [&str; 10] = ["RGB", "R", "G", "B", "A", "RGBA", "Normal", "Roughness", "Metallic", "Height"];

fn input(name: &'static str, default: InputDefault) -> InputSpec {
    InputSpec { name, default }
}

fn scalar(name: &'static str, value: f32) -> InputSpec {
    input(name, InputDefault::Scalar(value))
}

/// The pins of `op`.
pub fn spec(op: &Op) -> NodeSpec {
    use InputDefault::*;
    let (category, inputs, outputs): (&str, Vec<InputSpec>, Vec<&str>) = match op {
        Op::Constant { .. } | Op::Constant2 { .. } | Op::Constant3 { .. } | Op::Constant4 { .. } => ("Constants", vec![], vec!["Out"]),
        Op::ScalarParameter { .. } | Op::VectorParameter { .. } => ("Parameters", vec![], vec!["Out"]),
        Op::TextureSample { .. } => ("Textures", vec![input("UV", TexCoord)], TEXTURE_OUTPUTS.to_vec()),
        Op::Triplanar { .. } => (
            "Textures",
            vec![input("Position", WorldPosition), input("Normal", VertexNormal)],
            vec!["RGB", "A", "Normal", "TangentNormal", "Roughness", "Height"],
        ),
        Op::TexCoord { .. }
        | Op::WorldPosition
        | Op::ObjectPosition
        | Op::VertexNormal
        | Op::VertexColor
        | Op::Time
        | Op::CameraVector => ("Inputs", vec![], vec!["Out"]),
        Op::Add | Op::Subtract => ("Math", vec![scalar("A", 0.0), scalar("B", 0.0)], vec!["Out"]),
        Op::Multiply | Op::Divide => ("Math", vec![scalar("A", 1.0), scalar("B", 1.0)], vec!["Out"]),
        Op::Min | Op::Max => ("Math", vec![scalar("A", 0.0), scalar("B", 1.0)], vec!["Out"]),
        Op::Lerp => ("Math", vec![scalar("A", 0.0), scalar("B", 1.0), scalar("Alpha", 0.5)], vec!["Out"]),
        Op::Power => ("Math", vec![scalar("Base", 1.0), scalar("Exponent", 2.0)], vec!["Out"]),
        Op::Clamp => ("Math", vec![scalar("X", 0.0), scalar("Min", 0.0), scalar("Max", 1.0)], vec!["Out"]),
        Op::Dot | Op::Cross => ("Vector", vec![input("A", Vector3([0.0, 0.0, 1.0])), input("B", Vector3([0.0, 0.0, 1.0]))], vec!["Out"]),
        Op::Normalize | Op::Length => ("Vector", vec![input("In", Vector3([0.0, 0.0, 1.0]))], vec!["Out"]),
        Op::Append => ("Vector", vec![scalar("A", 0.0), scalar("B", 0.0)], vec!["Out"]),
        Op::ComponentMask { .. } => ("Vector", vec![input("In", Vector3([0.0, 0.0, 0.0]))], vec!["Out"]),
        Op::TransformVector { .. } => ("Vector", vec![input("In", Vector3([0.0, 0.0, 1.0]))], vec!["Out"]),
        Op::OneMinus | Op::Saturate | Op::Abs | Op::Sine | Op::Cosine | Op::Frac | Op::Floor => ("Math", vec![scalar("In", 0.0)], vec!["Out"]),
        Op::Step => ("Math", vec![scalar("Edge", 0.5), scalar("X", 0.0)], vec!["Out"]),
        Op::SmoothStep => ("Math", vec![scalar("Min", 0.0), scalar("Max", 1.0), scalar("X", 0.0)], vec!["Out"]),
        Op::Noise { .. } => ("Patterns", vec![input("Position", WorldPosition)], vec!["Out"]),
        Op::Fresnel { .. } => ("Utility", vec![input("Normal", VertexNormal)], vec!["Out"]),
        Op::Panner { .. } => ("Utility", vec![input("UV", TexCoord), input("Time", Time)], vec!["Out"]),
        Op::NormalBlend => ("Utility", vec![input("Base", Vector3([0.0, 0.0, 1.0])), input("Detail", Vector3([0.0, 0.0, 1.0]))], vec!["Out"]),
        Op::If => (
            "Utility",
            vec![scalar("A", 0.0), scalar("B", 0.0), scalar("AGreaterB", 1.0), scalar("AEqualsB", 0.0), scalar("ALessB", 0.0)],
            vec!["Out"],
        ),
        Op::Output => ("Output", OUTPUT_PINS.iter().map(|p| input(p.0, Material)).collect(), vec![]),
    };
    NodeSpec { category, inputs, outputs }
}

/// One palette entry for an editor: the type, a default node of it (its properties) and its
/// pins.
#[derive(Clone, Debug, Serialize)]
pub struct CatalogEntry {
    #[serde(rename = "type")]
    pub type_name: &'static str,
    pub template: Op,
    #[serde(flatten)]
    pub spec: NodeSpec,
}

/// Every node type, for editors.
pub fn catalog() -> Vec<CatalogEntry> {
    Op::palette().into_iter().map(|op| CatalogEntry { type_name: op.type_name(), spec: spec(&op), template: op }).collect()
}
