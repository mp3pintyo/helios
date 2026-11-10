//! The graph data model: nodes with an operation, links from output pins to input pins.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

fn one() -> f32 {
    1.0
}

fn ones() -> [f32; 2] {
    [1.0, 1.0]
}

fn octaves() -> u32 {
    4
}

fn fresnel_exponent() -> f32 {
    5.0
}

fn fresnel_base() -> f32 {
    0.04
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

/// A material graph: its nodes and the links between their pins. Exactly one node is the
/// [`Op::Output`]; nodes it does not reach are kept (an editor shows them) but not compiled.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaterialGraph {
    pub name: String,
    pub nodes: Vec<Node>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<Link>,
}

/// A node: an operation, where the editor draws it, and the constants its unconnected scalar
/// inputs take instead of the operation's defaults (UE's "Const A" / "Const B").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    #[serde(flatten)]
    pub op: Op,
    #[serde(default)]
    pub pos: [f32; 2],
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub values: BTreeMap<String, f32>,
}

/// A pin of a node, by name (see [`crate::spec`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PinRef {
    pub node: u32,
    pub pin: String,
}

impl PinRef {
    pub fn new(node: u32, pin: &str) -> Self {
        Self { node, pin: pin.to_string() }
    }
}

/// A link from an output pin to an input pin (an input takes at most one link).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub from: PinRef,
    pub to: PinRef,
}

/// A coordinate space of [`Op::TransformVector`] (UE's Transform node: Tangent, Local, World).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorSpace {
    /// The world axes: what `VertexNormal`, `WorldPosition` and the VectorDisplacement pin use.
    #[default]
    World,
    /// The instance's own axes (UE's Local space), normalised by its largest scale: a unit
    /// vector stays at most unit length, and an object-space displacement scales with the
    /// instance as a height does.
    Object,
    /// The surface's frame: x the tangent, y the bitangent, z the normal (see
    /// [`Op::TransformVector`] for which frame each function uses).
    Tangent,
}

impl VectorSpace {
    /// The space's code in the generated WGSL (`*_transform_vector` helpers).
    pub fn code(self) -> u32 {
        match self {
            VectorSpace::World => 0,
            VectorSpace::Object => 1,
            VectorSpace::Tangent => 2,
        }
    }
}

fn object_space() -> VectorSpace {
    VectorSpace::Object
}

/// What a node computes. Values are floats or float vectors of 2–4 components; a scalar
/// broadcasts wherever a vector is expected, as in UE. Angles are radians.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Op {
    // ---- Constants and parameters -------------------------------------------------------------
    Constant {
        value: f32,
    },
    Constant2 {
        value: [f32; 2],
    },
    Constant3 {
        value: [f32; 3],
    },
    Constant4 {
        value: [f32; 4],
    },
    /// A float that material instances override without recompiling.
    ScalarParameter {
        name: String,
        default: f32,
        /// Slider range in an editor.
        #[serde(default)]
        min: f32,
        #[serde(default = "one")]
        max: f32,
    },
    /// A colour or vector (rgba) that material instances override.
    VectorParameter {
        name: String,
        default: [f32; 4],
    },
    /// Samples the texture layer of parameter `parameter` (base colour, alpha, tangent-space
    /// normal, roughness, metallic). Unset, the parameter is the material's own texture.
    TextureSample {
        parameter: String,
    },
    // ---- Geometry inputs ----------------------------------------------------------------------
    /// Mesh UVs (set 0 or 1) times `tiling`.
    TexCoord {
        #[serde(default, skip_serializing_if = "is_zero")]
        set: u32,
        #[serde(default = "ones")]
        tiling: [f32; 2],
    },
    WorldPosition,
    ObjectPosition,
    /// The interpolated world-space normal (before normal mapping).
    VertexNormal,
    VertexColor,
    /// Seconds since the renderer started.
    Time,
    /// Unit vector from the surface towards the camera.
    CameraVector,
    // ---- Math ---------------------------------------------------------------------------------
    Add,
    Subtract,
    Multiply,
    Divide,
    Lerp,
    Power,
    Min,
    Max,
    Clamp,
    Dot,
    Cross,
    Normalize,
    Length,
    OneMinus,
    Saturate,
    Abs,
    Sine,
    Cosine,
    Frac,
    Floor,
    Step,
    SmoothStep,
    /// Concatenates the components of A and B.
    Append,
    /// Picks components, e.g. `"rg"` or `"xz"`.
    ComponentMask {
        channels: String,
    },
    /// Re-expresses a float3 direction of space `source` in space `destination` (UE's
    /// TransformVector; directions, not positions: no translation). Object → World turns a
    /// vector with its instance, so a vector displacement built from constants follows a
    /// rotating instance. The tangent frame differs by function: the surface function uses the
    /// pixel's shading frame (the one normal maps and the Normal output use, from the UVs), the
    /// displacement a frame the tessellator can evaluate identically on both sides of every
    /// edge — the interpolated object normal with the object X axis projected into its plane (Z
    /// where the normal runs along X). The two agree where U runs along object X (planes and
    /// terrain).
    TransformVector {
        #[serde(default = "object_space")]
        source: VectorSpace,
        #[serde(default)]
        destination: VectorSpace,
    },
    // ---- Patterns and utilities ---------------------------------------------------------------
    /// Fractal gradient noise in [0, 1], filtered by the pixel footprint.
    Noise {
        #[serde(default = "one")]
        scale: f32,
        #[serde(default = "octaves")]
        octaves: u32,
    },
    /// Schlick-style facing ratio: `base + (1 - base) (1 - N·V)^exponent`.
    Fresnel {
        #[serde(default = "fresnel_exponent")]
        exponent: f32,
        #[serde(default = "fresnel_base")]
        base_reflect: f32,
    },
    /// Scrolls coordinates: `UV + speed * Time`.
    Panner {
        speed: [f32; 2],
    },
    /// Blends two tangent-space normals (reoriented normal mapping, UE's
    /// BlendAngleCorrectedNormals).
    NormalBlend,
    /// Samples the texture of `parameter` projected along the three world axes.
    Triplanar {
        parameter: String,
        #[serde(default = "one")]
        scale: f32,
    },
    /// Picks AGreaterB, AEqualsB or ALessB by comparing A and B.
    If,
    /// The material's attributes.
    Output,
}

impl Op {
    /// The serialised type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Op::Constant { .. } => "Constant",
            Op::Constant2 { .. } => "Constant2",
            Op::Constant3 { .. } => "Constant3",
            Op::Constant4 { .. } => "Constant4",
            Op::ScalarParameter { .. } => "ScalarParameter",
            Op::VectorParameter { .. } => "VectorParameter",
            Op::TextureSample { .. } => "TextureSample",
            Op::TexCoord { .. } => "TexCoord",
            Op::WorldPosition => "WorldPosition",
            Op::ObjectPosition => "ObjectPosition",
            Op::VertexNormal => "VertexNormal",
            Op::VertexColor => "VertexColor",
            Op::Time => "Time",
            Op::CameraVector => "CameraVector",
            Op::Add => "Add",
            Op::Subtract => "Subtract",
            Op::Multiply => "Multiply",
            Op::Divide => "Divide",
            Op::Lerp => "Lerp",
            Op::Power => "Power",
            Op::Min => "Min",
            Op::Max => "Max",
            Op::Clamp => "Clamp",
            Op::Dot => "Dot",
            Op::Cross => "Cross",
            Op::Normalize => "Normalize",
            Op::Length => "Length",
            Op::OneMinus => "OneMinus",
            Op::Saturate => "Saturate",
            Op::Abs => "Abs",
            Op::Sine => "Sine",
            Op::Cosine => "Cosine",
            Op::Frac => "Frac",
            Op::Floor => "Floor",
            Op::Step => "Step",
            Op::SmoothStep => "SmoothStep",
            Op::Append => "Append",
            Op::ComponentMask { .. } => "ComponentMask",
            Op::TransformVector { .. } => "TransformVector",
            Op::Noise { .. } => "Noise",
            Op::Fresnel { .. } => "Fresnel",
            Op::Panner { .. } => "Panner",
            Op::NormalBlend => "NormalBlend",
            Op::Triplanar { .. } => "Triplanar",
            Op::If => "If",
            Op::Output => "Output",
        }
    }

    /// One node of every type with default properties (the editor's palette).
    pub fn palette() -> Vec<Op> {
        let named = |n: &str| n.to_string();
        vec![
            Op::Constant { value: 1.0 },
            Op::Constant2 { value: [0.0, 0.0] },
            Op::Constant3 { value: [1.0, 1.0, 1.0] },
            Op::Constant4 { value: [1.0, 1.0, 1.0, 1.0] },
            Op::ScalarParameter { name: named("Scalar"), default: 0.5, min: 0.0, max: 1.0 },
            Op::VectorParameter { name: named("Color"), default: [1.0, 1.0, 1.0, 1.0] },
            Op::TextureSample { parameter: named("Texture") },
            Op::TexCoord { set: 0, tiling: [1.0, 1.0] },
            Op::WorldPosition,
            Op::ObjectPosition,
            Op::VertexNormal,
            Op::VertexColor,
            Op::Time,
            Op::CameraVector,
            Op::Add,
            Op::Subtract,
            Op::Multiply,
            Op::Divide,
            Op::Lerp,
            Op::Power,
            Op::Min,
            Op::Max,
            Op::Clamp,
            Op::Dot,
            Op::Cross,
            Op::Normalize,
            Op::Length,
            Op::OneMinus,
            Op::Saturate,
            Op::Abs,
            Op::Sine,
            Op::Cosine,
            Op::Frac,
            Op::Floor,
            Op::Step,
            Op::SmoothStep,
            Op::Append,
            Op::ComponentMask { channels: named("rg") },
            Op::TransformVector { source: VectorSpace::Object, destination: VectorSpace::World },
            Op::Noise { scale: 1.0, octaves: 4 },
            Op::Fresnel { exponent: 5.0, base_reflect: 0.04 },
            Op::Panner { speed: [0.1, 0.0] },
            Op::NormalBlend,
            Op::Triplanar { parameter: named("Texture"), scale: 1.0 },
            Op::If,
            Op::Output,
        ]
    }
}

impl MaterialGraph {
    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// The link into input pin `to`, if any.
    pub fn link_into(&self, to: &PinRef) -> Option<&Link> {
        self.links.iter().find(|l| &l.to == to)
    }

    /// An id no node has yet.
    pub fn next_id(&self) -> u32 {
        self.nodes.iter().map(|n| n.id + 1).max().unwrap_or(1)
    }
}

/// Builds graphs in code (presets, tests): nodes are placed left to right by `column`.
#[derive(Default)]
pub struct GraphBuilder {
    graph: MaterialGraph,
}

impl GraphBuilder {
    pub fn new(name: &str) -> Self {
        Self { graph: MaterialGraph { name: name.to_string(), ..Default::default() } }
    }

    /// Adds a node at canvas position `pos`; returns its id.
    pub fn node(&mut self, op: Op, pos: [f32; 2]) -> u32 {
        let id = self.graph.next_id();
        self.graph.nodes.push(Node { id, op, pos, values: BTreeMap::new() });
        id
    }

    /// Sets the constant of an unconnected scalar input.
    pub fn value(&mut self, node: u32, input: &str, value: f32) -> &mut Self {
        if let Some(n) = self.graph.nodes.iter_mut().find(|n| n.id == node) {
            n.values.insert(input.to_string(), value);
        }
        self
    }

    pub fn link(&mut self, from: u32, from_pin: &str, to: u32, to_pin: &str) -> &mut Self {
        self.graph.links.push(Link { from: PinRef::new(from, from_pin), to: PinRef::new(to, to_pin) });
        self
    }

    pub fn build(self) -> MaterialGraph {
        self.graph
    }
}
