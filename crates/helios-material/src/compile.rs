//! Graph → WGSL.
//!
//! The compiler validates the graph (ids, links, pins, cycles, one output), infers every
//! value's width (float, vec2–vec4; scalars broadcast) and emits one function,
//!
//! ```wgsl
//! fn graph_eval(m: Material, input: SurfaceInput, params: u32, vertex_color: vec4<f32>,
//!               defaults: GraphOutput) -> GraphOutput
//! ```
//!
//! which the renderer's G-buffer resolve calls per pixel (`material_graph.wgsl` declares
//! `GraphOutput`, the parameter block and the helpers the generated code uses). Outputs left
//! unconnected keep `defaults` — the material's own values.
//!
//! **Displacement.** The output's `Displacement` pin (a height along the normal) or its
//! `VectorDisplacement` pin (a world-space offset: overhangs) compiles to a function of its own,
//! from the nodes it depends on (sharing the node code generation and the parameter slots),
//!
//! ```wgsl
//! fn graph_displacement(m: Material, input: DisplacementSurface, params: u32) -> f32  // or vec3<f32>
//! ```
//!
//! which Nanite tessellation calls per micro-triangle vertex — in the main view and the shadow
//! pages — and the resolve where it shades the displaced point (`displacement_graph.wgsl`). The
//! value must be a pure function of the undisplaced point and the time, the same in every pass
//! and on both sides of a shared edge, so in this context the compiler rejects what would break
//! that: the camera vector and fresnel (every view must displace a point alike) and the
//! attributes the tessellator does not read (vertex colours, UV set 1, the tangent frame).
//! Time is allowed (`input.time`: the renderer evaluates the previous frame's too, for motion
//! vectors) and reported as [`CompiledGraph::displacement_animated`]: the renderer then treats
//! the graph's displaced instances as moving shadow casters, which static graphs never pay for.
//! Texture and noise filtering use the size of a dice segment instead of a pixel.
//!
//! **Derivatives.** The resolve runs in a compute shader over a visibility buffer: there are no
//! hardware screen derivatives. Texture sampling and noise filtering need them, so every value
//! that feeds a texture coordinate or a noise position carries its screen-space derivatives
//! (`_dx`, `_dy`) through the math, by the chain rule (forward-mode differentiation), starting
//! from the analytic derivatives of the mesh UVs and positions. Only those values pay for it.
//!
//! **Parameters** get one `vec4` slot each in declaration order: material instances write their
//! values into the block (see the renderer), so changing one never recompiles.
//!
//! The emitted code parenthesises every operation and uses only generated identifiers, so it
//! passes Chrome's Tint as well as naga.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use serde::Serialize;

use crate::graph::{MaterialGraph, Node, Op};
use crate::motion::DisplacementMotion;
use crate::nodes::{spec, InputDefault, DISPLACEMENT_PIN, OUTPUT_PINS, VECTOR_DISPLACEMENT_PIN};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    Scalar,
    Vector,
    /// A texture layer (`u32` in the slot's first component).
    Texture,
}

/// A parameter of a compiled graph and its slot (its index in [`CompiledGraph::parameters`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Parameter {
    pub name: String,
    pub kind: ParameterKind,
    pub default: [f32; 4],
    /// Slider range (scalars).
    pub range: [f32; 2],
}

/// What a graph's displacement function returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplacementKind {
    /// A height (`f32`, 0..1) along the normal ([`DISPLACEMENT_PIN`]).
    Height,
    /// A world-space offset (`vec3<f32>`) in units of the magnitude ([`VECTOR_DISPLACEMENT_PIN`]).
    Vector,
}

impl DisplacementKind {
    /// The WGSL type of the displacement function's result.
    pub fn wgsl_type(self) -> &'static str {
        match self {
            DisplacementKind::Height => "f32",
            DisplacementKind::Vector => "vec3<f32>",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledGraph {
    /// `graph_eval`, and `graph_displacement` (with the constants `GRAPH_DISPLACEMENT_VECTOR` and
    /// `GRAPH_DISPLACEMENT_ANIMATED`) if a displacement output is connected.
    pub wgsl: String,
    pub parameters: Vec<Parameter>,
    /// Bits of the connected [`OUTPUT_PINS`].
    pub outputs: u32,
    /// FNV-1a of `wgsl`: identical graphs share a compiled pipeline.
    pub hash: u64,
    /// The body of the displacement function, if a displacement output is connected (see
    /// [`CompiledGraph::displacement_function`]).
    pub displacement: Option<String>,
    /// What the displacement function returns ([`DisplacementKind::Height`] without one).
    pub displacement_kind: DisplacementKind,
    /// The displacement reads the time: the displaced surface moves every frame.
    pub displacement_animated: bool,
    /// The displacement function's nodes, for its speed bound
    /// ([`CompiledGraph::displacement_speed`]).
    pub displacement_motion: Option<DisplacementMotion>,
}

impl CompiledGraph {
    /// The displacement function under `name`: `fn name(m: Material, input: DisplacementSurface,
    /// params: u32) -> f32` returning the raw height, or `-> vec3<f32>` the raw vector (the
    /// renderer clamps the height to 0..1 and the vector's length to 1).
    pub fn displacement_function(&self, name: &str) -> Option<String> {
        let body = self.displacement.as_ref()?;
        let result = self.displacement_kind.wgsl_type();
        Some(format!("fn {name}(m: Material, input: DisplacementSurface, params: u32) -> {result} {{\n{body}}}\n"))
    }

    /// FNV-1a of the displacement code and its kind: graphs that displace alike share it.
    pub fn displacement_hash(&self) -> Option<u64> {
        self.displacement.as_ref().map(|body| fnv1a(body.as_bytes()) ^ self.displacement_kind as u64)
    }

    /// Bound of how fast the displacement moves a point, in units of the material's magnitude
    /// per second (a height's 0..1, a vector's length), for a material instance's parameter
    /// values (`parameters`, by slot: [`CompiledGraph::parameters`] order; missing slots take
    /// the defaults): 0 for a static displacement, `None` where the graph's time dependence
    /// cannot be bounded (see [`crate::motion`]). The renderer refreshes cached shadow pages by
    /// it.
    pub fn displacement_speed(&self, parameters: &[[f32; 4]]) -> Option<f32> {
        if !self.displacement_animated {
            return Some(0.0);
        }
        self.displacement_motion.as_ref()?.speed(parameters)
    }
}

/// A problem that keeps a graph from compiling, with the node it concerns.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GraphError {
    pub node: Option<u32>,
    pub message: String,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.node {
            Some(node) => write!(f, "node #{node}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// Compiles `graph`, or lists everything wrong with it.
pub fn compile(graph: &MaterialGraph) -> Result<CompiledGraph, Vec<GraphError>> {
    let mut c = Compiler::new(graph);
    c.check_structure();
    if !c.errors.is_empty() {
        return Err(c.errors);
    }
    let output = c.output;
    let surface = c.order(output, &[DISPLACEMENT_PIN, VECTOR_DISPLACEMENT_PIN]);
    let source = |c: &Compiler, pin: &str| c.sources.get(&(output, pin.to_string())).map(|s| s.0);
    let (pin, kind, root) = match (source(&c, DISPLACEMENT_PIN), source(&c, VECTOR_DISPLACEMENT_PIN)) {
        (Some(_), Some(_)) => {
            c.error(Some(output), format!("Output: connect {DISPLACEMENT_PIN} (a height along the normal) or {VECTOR_DISPLACEMENT_PIN} (an offset vector), not both"));
            return Err(c.errors);
        }
        (None, Some(root)) => (VECTOR_DISPLACEMENT_PIN, DisplacementKind::Vector, Some(root)),
        (root, None) => (DISPLACEMENT_PIN, DisplacementKind::Height, root),
    };
    let displacement = root.map(|root| c.order(root, &[]));
    if !c.errors.is_empty() {
        return Err(c.errors);
    }
    let mut reachable: Vec<u32> = surface.iter().chain(displacement.iter().flatten()).copied().collect();
    reachable.sort_unstable();
    reachable.dedup();
    c.assign_parameters(&reachable);
    let eval = c.function(Context::Surface, &surface);
    let motion = match (&displacement, root) {
        (Some(order), Some(_)) => Some(c.motion(order, pin, kind)),
        _ => None,
    };
    let displacement = displacement.map(|order| {
        let mut body = c.function(Context::Displacement, &order);
        let value = c.input(output, pin);
        let width = if kind == DisplacementKind::Vector { 3 } else { 1 };
        match widen(&value.expr, value.width, width) {
            Some(expr) => {
                let _ = writeln!(body, "    return {expr};");
            }
            None => c.error(Some(output), format!("Output: {pin} takes a {}, not a {}", ty(width), ty(value.width))),
        }
        c.outputs |= crate::nodes::output_bit(pin);
        body
    });
    if !c.errors.is_empty() {
        return Err(c.errors);
    }
    let mut wgsl = String::new();
    let name: String = graph.name.chars().filter(|ch| !ch.is_control()).collect();
    let _ = writeln!(wgsl, "// Material graph \"{name}\" (generated by helios-material).");
    let _ = writeln!(wgsl, "const GRAPH_OUTPUTS: u32 = {}u;", c.outputs);
    wgsl.push_str("fn graph_eval(m: Material, input: SurfaceInput, params: u32, vertex_color: vec4<f32>, defaults: GraphOutput) -> GraphOutput {\n");
    wgsl.push_str("    var o = defaults;\n");
    wgsl.push_str(&eval);
    wgsl.push_str("    return o;\n}\n");
    let mut compiled = CompiledGraph {
        wgsl,
        parameters: c.params,
        outputs: c.outputs,
        hash: 0,
        displacement,
        displacement_kind: kind,
        displacement_animated: c.animated,
        displacement_motion: motion,
    };
    if let Some(function) = compiled.displacement_function("graph_displacement") {
        let _ = writeln!(compiled.wgsl, "const GRAPH_DISPLACEMENT_VECTOR: bool = {};", kind == DisplacementKind::Vector);
        let _ = writeln!(compiled.wgsl, "const GRAPH_DISPLACEMENT_ANIMATED: bool = {};", compiled.displacement_animated);
        compiled.wgsl.push_str(&function);
    }
    compiled.hash = fnv1a(compiled.wgsl.as_bytes());
    Ok(compiled)
}

/// Where generated code runs, which decides the inputs and helpers it may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Context {
    /// `graph_eval` in the G-buffer resolve: screen-space derivatives, every input.
    Surface,
    /// `graph_displacement` in the tessellator: a function of the undisplaced point and the time.
    Displacement,
}

/// Helper functions of a context (material_graph.wgsl, displacement_graph.wgsl).
struct Helpers {
    texture: &'static str,
    sample: &'static str,
    triplanar: &'static str,
    decode_normal: &'static str,
    blend_normals: &'static str,
    transform: &'static str,
}

impl Context {
    fn helpers(self) -> Helpers {
        match self {
            Context::Surface => Helpers {
                texture: "graph_texture",
                sample: "graph_sample",
                triplanar: "graph_triplanar",
                decode_normal: "decode_tangent_normal",
                blend_normals: "graph_blend_normals",
                transform: "graph_transform_vector",
            },
            Context::Displacement => Helpers {
                texture: "displacement_graph_texture",
                sample: "displacement_graph_sample",
                triplanar: "displacement_graph_triplanar",
                decode_normal: "displacement_decode_normal",
                blend_normals: "displacement_blend_normals",
                transform: "displacement_transform_vector",
            },
        }
    }
}

const VIEW_DISPLACEMENT: &str = "depends on the camera, which cannot drive Displacement: every view and the shadow pages must displace a point alike";

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// A value in the generated code: an expression (identifier or literal) of `width`
/// components, with its screen derivatives where it carries them (`None`: constant).
#[derive(Clone, Debug)]
struct Value {
    expr: String,
    width: u8,
    deriv: Option<(String, String)>,
    /// Why the value cannot be used in this context (reported where it is linked).
    unavailable: Option<&'static str>,
}

impl Value {
    fn plain(expr: impl Into<String>, width: u8) -> Self {
        Self { expr: expr.into(), width, deriv: None, unavailable: None }
    }

    fn with_deriv(expr: &str, width: u8, dx: &str, dy: &str) -> Self {
        Self { expr: expr.into(), width, deriv: Some((dx.into(), dy.into())), unavailable: None }
    }

    fn unavailable(width: u8, why: &'static str) -> Self {
        Self { unavailable: Some(why), ..Self::plain(zero(width), width) }
    }
}

fn ty(width: u8) -> &'static str {
    ["f32", "vec2<f32>", "vec3<f32>", "vec4<f32>"][(width.clamp(1, 4) - 1) as usize]
}

fn zero(width: u8) -> String {
    if width == 1 {
        "0.0f".into()
    } else {
        format!("{}(0.0)", ty(width))
    }
}

/// `expr` (of `from` components) as `to` components: equal, or a scalar broadcast.
fn widen(expr: &str, from: u8, to: u8) -> Option<String> {
    match (from, to) {
        _ if from == to => Some(expr.to_string()),
        (1, _) => Some(format!("{}({expr})", ty(to))),
        _ => None,
    }
}

/// The common width of two operands (equal, or one a scalar).
fn unify(a: u8, b: u8) -> Result<u8, String> {
    if a == b || a == 1 || b == 1 {
        Ok(a.max(b))
    } else {
        Err(format!("cannot combine a {} with a {}", ty(a), ty(b)))
    }
}

fn swizzle(channels: &str) -> Result<Vec<usize>, String> {
    let index = |c: char| "xyzw".find(c).or_else(|| "rgba".find(c));
    let picks: Option<Vec<usize>> = channels.chars().map(index).collect();
    match picks {
        Some(p) if !p.is_empty() && p.len() <= 4 => Ok(p),
        _ => Err(format!("component mask '{channels}' must be 1-4 of x, y, z, w (or r, g, b, a)")),
    }
}

struct Compiler<'g> {
    graph: &'g MaterialGraph,
    nodes: BTreeMap<u32, &'g Node>,
    /// Input pin (node, pin) → source output pin (node, pin).
    sources: HashMap<(u32, String), (u32, String)>,
    output: u32,
    code: String,
    errors: Vec<GraphError>,
    values: HashMap<(u32, String), Value>,
    params: Vec<Parameter>,
    /// Nodes whose values carry derivatives.
    derivative: HashSet<u32>,
    outputs: u32,
    context: Context,
    /// The displacement function reads the time.
    animated: bool,
}

impl<'g> Compiler<'g> {
    fn new(graph: &'g MaterialGraph) -> Self {
        Self {
            graph,
            nodes: BTreeMap::new(),
            sources: HashMap::new(),
            output: 0,
            code: String::new(),
            errors: Vec::new(),
            values: HashMap::new(),
            params: Vec::new(),
            derivative: HashSet::new(),
            outputs: 0,
            context: Context::Surface,
            animated: false,
        }
    }

    fn error(&mut self, node: Option<u32>, message: impl Into<String>) {
        self.errors.push(GraphError { node, message: message.into() });
    }

    /// Unique ids, valid links, one output node.
    fn check_structure(&mut self) {
        for node in &self.graph.nodes {
            if self.nodes.insert(node.id, node).is_some() {
                self.error(Some(node.id), "duplicate node id");
            }
        }
        for link in &self.graph.links {
            let (from, to) = (&link.from, &link.to);
            let Some(&source) = self.nodes.get(&from.node) else {
                self.error(Some(to.node), format!("input '{}' is linked from node #{}, which does not exist", to.pin, from.node));
                continue;
            };
            let Some(&target) = self.nodes.get(&to.node) else {
                self.error(Some(from.node), format!("output '{}' is linked to node #{}, which does not exist", from.pin, to.node));
                continue;
            };
            let (source_op, target_op) = (source.op.type_name(), target.op.type_name());
            if !spec(&source.op).outputs.contains(&from.pin.as_str()) {
                self.error(Some(from.node), format!("{source_op} has no output '{}'", from.pin));
            }
            if !spec(&target.op).inputs.iter().any(|i| i.name == to.pin) {
                self.error(Some(to.node), format!("{target_op} has no input '{}'", to.pin));
            }
            if self.sources.insert((to.node, to.pin.clone()), (from.node, from.pin.clone())).is_some() {
                self.error(Some(to.node), format!("input '{}' has more than one link", to.pin));
            }
        }
        let outputs: Vec<u32> = self.nodes.values().filter(|n| n.op == Op::Output).map(|n| n.id).collect();
        match outputs.as_slice() {
            [id] => self.output = *id,
            [] => self.error(None, "the graph has no Output node"),
            _ => self.error(Some(outputs[1]), "the graph has more than one Output node"),
        }
    }

    /// Nodes `root` depends on (`root` last), each after its inputs, not following the root's
    /// inputs `skip`; reports cycles.
    fn order(&mut self, root: u32, skip: &[&str]) -> Vec<u32> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Visiting,
            Done,
        }
        let mut marks: HashMap<u32, Mark> = HashMap::new();
        let mut order = Vec::new();
        // Iterative DFS: (node, next input index).
        let mut stack = vec![(root, 0usize)];
        marks.insert(root, Mark::Visiting);
        while let Some(&mut (id, ref mut next)) = stack.last_mut() {
            let inputs = spec(&self.nodes[&id].op).inputs;
            if let Some(input) = inputs.get(*next) {
                *next += 1;
                if id == root && skip.contains(&input.name) {
                    continue;
                }
                let Some((source, _)) = self.sources.get(&(id, input.name.to_string())).cloned() else { continue };
                match marks.get(&source) {
                    Some(Mark::Visiting) => {
                        let path: Vec<String> = stack.iter().map(|(n, _)| format!("#{n}")).collect();
                        self.error(Some(source), format!("the graph has a cycle ({} -> #{source})", path.join(" -> ")));
                        return Vec::new();
                    }
                    Some(Mark::Done) => {}
                    None => {
                        marks.insert(source, Mark::Visiting);
                        stack.push((source, 0));
                    }
                }
            } else {
                marks.insert(id, Mark::Done);
                order.push(id);
                stack.pop();
            }
        }
        order
    }

    /// One slot per parameter name, in node id order (stable across edits that don't touch
    /// parameters).
    fn assign_parameters(&mut self, order: &[u32]) {
        let mut reachable: Vec<u32> = order.to_vec();
        reachable.sort_unstable();
        for id in reachable {
            let (name, kind, default, range) = match &self.nodes[&id].op {
                Op::ScalarParameter { name, default, min, max } => (name, ParameterKind::Scalar, [*default, 0.0, 0.0, 0.0], [*min, *max]),
                Op::VectorParameter { name, default } => (name, ParameterKind::Vector, *default, [0.0, 1.0]),
                Op::TextureSample { parameter } | Op::Triplanar { parameter, .. } => (parameter, ParameterKind::Texture, [0.0; 4], [0.0, 1.0]),
                _ => continue,
            };
            if name.trim().is_empty() {
                self.error(Some(id), "parameter without a name");
                continue;
            }
            match self.params.iter().find(|p| &p.name == name) {
                Some(p) if p.kind != kind => {
                    let message = format!("parameter '{name}' is used as a {kind:?} and as a {:?}", p.kind);
                    self.error(Some(id), message);
                }
                Some(_) => {}
                None => self.params.push(Parameter { name: name.clone(), kind, default, range }),
            }
        }
    }

    fn slot(&self, name: &str) -> usize {
        self.params.iter().position(|p| p.name == name).unwrap_or(0)
    }

    /// The displacement function's nodes `order` (Output pin `pin`) for its speed bound.
    fn motion(&self, order: &[u32], pin: &str, kind: DisplacementKind) -> DisplacementMotion {
        let steps: Vec<(u32, Op, usize)> = order
            .iter()
            .map(|&id| {
                let op = self.nodes[&id].op.clone();
                let slot = match &op {
                    Op::ScalarParameter { name, .. } | Op::VectorParameter { name, .. } => self.slot(name),
                    _ => 0,
                };
                (id, op, slot)
            })
            .collect();
        let source = |id: u32, pin: &str| -> Result<(u32, String), InputDefault> {
            if let Some(s) = self.sources.get(&(id, pin.to_string())) {
                return Ok(s.clone());
            }
            let node = self.nodes[&id];
            match spec(&node.op).inputs.iter().find(|i| i.name == pin).map_or(InputDefault::Scalar(0.0), |i| i.default) {
                InputDefault::Scalar(x) => Err(InputDefault::Scalar(node.values.get(pin).copied().unwrap_or(x))),
                default => Err(default),
            }
        };
        let defaults = self.params.iter().map(|p| p.default).collect();
        DisplacementMotion::new(&steps, source, source(self.output, pin), kind, defaults)
    }

    /// Values feeding texture coordinates, triplanar positions and noise positions carry
    /// derivatives, and so do the values they are computed from.
    fn mark_derivatives(&mut self, order: &[u32]) {
        for &id in order.iter().rev() {
            let wanted: &[&str] = match self.nodes[&id].op {
                Op::TextureSample { .. } => &["UV"],
                Op::Triplanar { .. } | Op::Noise { .. } => &["Position"],
                _ if self.derivative.contains(&id) => &[],
                _ => continue,
            };
            let all: Vec<String> = spec(&self.nodes[&id].op).inputs.iter().map(|i| i.name.to_string()).collect();
            let pins: Vec<String> = if self.derivative.contains(&id) { all } else { wanted.iter().map(|s| s.to_string()).collect() };
            for pin in pins {
                if let Some((source, _)) = self.sources.get(&(id, pin)) {
                    self.derivative.insert(*source);
                }
            }
        }
    }

    // ---- Emission ---------------------------------------------------------------------------

    /// The code of `order`'s nodes in `context` (each function has values and derivatives of
    /// its own).
    fn function(&mut self, context: Context, order: &[u32]) -> String {
        self.context = context;
        self.code.clear();
        self.values.clear();
        self.derivative.clear();
        self.mark_derivatives(order);
        for &id in order {
            self.emit_node(id);
        }
        std::mem::take(&mut self.code)
    }

    fn displacement(&self) -> bool {
        self.context == Context::Displacement
    }

    /// The renderer's clock in seconds. The displacement reads it from its input, so the
    /// renderer can evaluate the surface at the previous frame's time too (motion vectors).
    fn time(&mut self) -> Value {
        if self.displacement() {
            self.animated = true;
            Value::plain("input.time", 1)
        } else {
            Value::plain("frame.time", 1)
        }
    }

    fn line(&mut self, text: String) {
        self.code.push_str("    ");
        self.code.push_str(&text);
        self.code.push('\n');
    }

    /// Binds `expr` (and its derivatives, if the node carries them) to identifiers.
    fn bind(&mut self, id: u32, pin: usize, width: u8, expr: String, deriv: Option<(String, String)>) -> Value {
        let name = format!("v{id}_{pin}");
        self.line(format!("let {name}: {} = {expr};", ty(width)));
        let deriv = deriv.filter(|_| self.derivative.contains(&id)).map(|(dx, dy)| {
            self.line(format!("let {name}_dx: {} = {dx};", ty(width)));
            self.line(format!("let {name}_dy: {} = {dy};", ty(width)));
            (format!("{name}_dx"), format!("{name}_dy"))
        });
        Value { expr: name, width, deriv, unavailable: None }
    }

    fn literal(&mut self, id: u32, x: f32) -> String {
        if !x.is_finite() {
            self.error(Some(id), format!("{x} is not a finite number"));
            return "0.0f".into();
        }
        let text = format!("{x:?}f");
        if x < 0.0 {
            format!("({text})")
        } else {
            text
        }
    }

    fn vector_literal(&mut self, id: u32, v: &[f32]) -> String {
        let parts: Vec<String> = v.iter().map(|&x| self.literal(id, x)).collect();
        format!("{}({})", ty(v.len() as u8), parts.join(", "))
    }

    /// The value at input `pin` of node `id`: its link's source, or the input's default.
    fn input(&mut self, id: u32, pin: &str) -> Value {
        if let Some(source) = self.sources.get(&(id, pin.to_string())).cloned() {
            let value = self.values.get(&source).cloned().unwrap_or_else(|| Value::plain("0.0f", 1));
            if let Some(why) = value.unavailable {
                self.error(Some(source.0), format!("{} {why}", source.1));
            }
            return value;
        }
        let node = self.nodes[&id];
        let default = spec(&node.op).inputs.iter().find(|i| i.name == pin).map(|i| i.default).unwrap_or(InputDefault::Scalar(0.0));
        match default {
            InputDefault::Scalar(x) => {
                let value = node.values.get(pin).copied().unwrap_or(x);
                Value::plain(self.literal(id, value), 1)
            }
            InputDefault::Vector3(v) => Value::plain(self.vector_literal(id, &v), 3),
            InputDefault::TexCoord => Value::with_deriv("input.uv", 2, "input.duv_dx", "input.duv_dy"),
            InputDefault::WorldPosition => Value::with_deriv("input.world_position", 3, "input.dpdx", "input.dpdy"),
            InputDefault::VertexNormal => Value::plain("input.normal", 3),
            InputDefault::Time => self.time(),
            InputDefault::Material => Value::plain("0.0f", 1),
        }
    }

    /// Derivatives of `v` as `width` components (zero for constants).
    fn d(v: &Value, width: u8) -> (String, String) {
        match &v.deriv {
            Some((dx, dy)) => (widen(dx, v.width, width).unwrap_or_else(|| dx.clone()), widen(dy, v.width, width).unwrap_or_else(|| dy.clone())),
            None => (zero(width), zero(width)),
        }
    }

    fn emit_node(&mut self, id: u32) {
        let node = self.nodes[&id];
        let outputs = spec(&node.op).outputs;
        match self.emit(id, &node.op) {
            Ok(values) => {
                for (pin, value) in values {
                    self.values.insert((id, pin.to_string()), value);
                }
            }
            Err(message) => {
                self.error(Some(id), format!("{}: {message}", node.op.type_name()));
                // Placeholders keep later nodes compiling, so one pass reports every problem.
                for pin in outputs {
                    self.values.insert((id, pin.to_string()), Value::plain("0.0f", 1));
                }
            }
        }
    }

    /// Emits node `id`; returns its output values by pin.
    fn emit(&mut self, id: u32, op: &Op) -> Result<Vec<(&'static str, Value)>, String> {
        let out = |v: Value| Ok(vec![("Out", v)]);
        match op {
            Op::Constant { value } => {
                let e = self.literal(id, *value);
                out(Value::plain(e, 1))
            }
            Op::Constant2 { value } => {
                let e = self.vector_literal(id, value);
                out(Value::plain(e, 2))
            }
            Op::Constant3 { value } => {
                let e = self.vector_literal(id, value);
                out(Value::plain(e, 3))
            }
            Op::Constant4 { value } => {
                let e = self.vector_literal(id, value);
                out(Value::plain(e, 4))
            }
            Op::ScalarParameter { name, .. } => {
                let slot = self.slot(name);
                out(self.bind(id, 0, 1, format!("graph_params[params + {slot}u].x"), None))
            }
            Op::VectorParameter { name, .. } => {
                let slot = self.slot(name);
                out(self.bind(id, 0, 4, format!("graph_params[params + {slot}u]"), None))
            }
            Op::TexCoord { set, tiling } => {
                let t = self.vector_literal(id, tiling);
                let (uv, dx, dy) = match set {
                    0 => ("input.uv", "input.duv_dx", "input.duv_dy"),
                    1 if self.displacement() => return Err("UV set 1 is not available to Displacement (the tessellator reads UV set 0)".into()),
                    1 => ("input.uv1", "input.duv1_dx", "input.duv1_dy"),
                    s => return Err(format!("UV set {s} does not exist (0 or 1)")),
                };
                let d = Some((format!("({dx} * {t})"), format!("({dy} * {t})")));
                out(self.bind(id, 0, 2, format!("({uv} * {t})"), d))
            }
            Op::WorldPosition => out(Value::with_deriv("input.world_position", 3, "input.dpdx", "input.dpdy")),
            Op::ObjectPosition => out(Value::with_deriv("input.object_position", 3, "input.object_dpdx", "input.object_dpdy")),
            Op::VertexNormal => out(Value::plain("input.normal", 3)),
            Op::VertexColor if self.displacement() => {
                Err("vertex colours are not available to Displacement (the tessellator reads positions, normals and UV set 0)".into())
            }
            Op::VertexColor => out(Value::plain("vertex_color", 4)),
            Op::Time => out(self.time()),
            Op::CameraVector | Op::Fresnel { .. } if self.displacement() => Err(VIEW_DISPLACEMENT.into()),
            Op::CameraVector => out(self.bind(id, 0, 3, "normalize(frame.camera_position.xyz - input.world_position)".into(), None)),
            Op::Add | Op::Subtract | Op::Multiply | Op::Divide | Op::Min | Op::Max => {
                let (a, b) = (self.input(id, "A"), self.input(id, "B"));
                let w = unify(a.width, b.width)?;
                let (ea, eb) = (widen(&a.expr, a.width, w).unwrap(), widen(&b.expr, b.width, w).unwrap());
                let ((ax, ay), (bx, by)) = (Self::d(&a, w), Self::d(&b, w));
                let (expr, dx, dy) = match op {
                    Op::Add => (format!("({ea} + {eb})"), format!("({ax} + {bx})"), format!("({ay} + {by})")),
                    Op::Subtract => (format!("({ea} - {eb})"), format!("({ax} - {bx})"), format!("({ay} - {by})")),
                    Op::Multiply => (format!("({ea} * {eb})"), format!("(({ax} * {eb}) + ({ea} * {bx}))"), format!("(({ay} * {eb}) + ({ea} * {by}))")),
                    Op::Divide => (
                        format!("({ea} / {eb})"),
                        format!("((({ax} * {eb}) - ({ea} * {bx})) / ({eb} * {eb}))"),
                        format!("((({ay} * {eb}) - ({ea} * {by})) / ({eb} * {eb}))"),
                    ),
                    Op::Min => (format!("min({ea}, {eb})"), format!("select({bx}, {ax}, {ea} < {eb})"), format!("select({by}, {ay}, {ea} < {eb})")),
                    _ => (format!("max({ea}, {eb})"), format!("select({bx}, {ax}, {ea} > {eb})"), format!("select({by}, {ay}, {ea} > {eb})")),
                };
                out(self.bind(id, 0, w, expr, Some((dx, dy))))
            }
            Op::Lerp => {
                let (a, b, t) = (self.input(id, "A"), self.input(id, "B"), self.input(id, "Alpha"));
                let w = unify(a.width, b.width)?;
                if t.width != 1 && t.width != w {
                    return Err(format!("Alpha must be a float or a {}, not a {}", ty(w), ty(t.width)));
                }
                let (ea, eb, et) = (widen(&a.expr, a.width, w).unwrap(), widen(&b.expr, b.width, w).unwrap(), widen(&t.expr, t.width, w).unwrap());
                let ((ax, ay), (bx, by), (tx, ty_)) = (Self::d(&a, w), Self::d(&b, w), Self::d(&t, w));
                let dx = format!("(mix({ax}, {bx}, {et}) + (({eb} - {ea}) * {tx}))");
                let dy = format!("(mix({ay}, {by}, {et}) + (({eb} - {ea}) * {ty_}))");
                out(self.bind(id, 0, w, format!("mix({ea}, {eb}, {et})"), Some((dx, dy))))
            }
            Op::Power => {
                let (a, e) = (self.input(id, "Base"), self.input(id, "Exponent"));
                if e.width != 1 && e.width != a.width {
                    return Err(format!("Exponent must be a float or a {}", ty(a.width)));
                }
                let w = a.width;
                let base = format!("max({}, {})", a.expr, zero(w));
                let ee = widen(&e.expr, e.width, w).unwrap();
                let (ax, ay) = Self::d(&a, w);
                let slope = format!("({ee} * pow({base}, {ee} - {}))", widen("1.0f", 1, w).unwrap());
                out(self.bind(id, 0, w, format!("pow({base}, {ee})"), Some((format!("({slope} * {ax})"), format!("({slope} * {ay})")))))
            }
            Op::Clamp => {
                let (x, lo, hi) = (self.input(id, "X"), self.input(id, "Min"), self.input(id, "Max"));
                let w = x.width;
                let (el, eh) = (widen(&lo.expr, lo.width, w), widen(&hi.expr, hi.width, w));
                let (Some(el), Some(eh)) = (el, eh) else { return Err(format!("Min and Max must be floats or {}", ty(w))) };
                let (xx, xy) = Self::d(&x, w);
                let inside = format!("(({} > {el}) & ({} < {eh}))", x.expr, x.expr);
                let (dx, dy) = (format!("select({}, {xx}, {inside})", zero(w)), format!("select({}, {xy}, {inside})", zero(w)));
                out(self.bind(id, 0, w, format!("clamp({}, {el}, {eh})", x.expr), Some((dx, dy))))
            }
            Op::Saturate => {
                let x = self.input(id, "In");
                let w = x.width;
                let (xx, xy) = Self::d(&x, w);
                let (lo, hi) = (zero(w), widen("1.0f", 1, w).unwrap());
                let inside = format!("(({} > {lo}) & ({} < {hi}))", x.expr, x.expr);
                let (dx, dy) = (format!("select({lo}, {xx}, {inside})"), format!("select({lo}, {xy}, {inside})"));
                out(self.bind(id, 0, w, format!("saturate({})", x.expr), Some((dx, dy))))
            }
            Op::OneMinus | Op::Abs | Op::Sine | Op::Cosine | Op::Frac | Op::Floor => {
                let x = self.input(id, "In");
                let (w, e) = (x.width, x.expr.clone());
                let (xx, xy) = Self::d(&x, w);
                let one = widen("1.0f", 1, w).unwrap();
                let (expr, dx, dy) = match op {
                    Op::OneMinus => (format!("({one} - {e})"), format!("(-{xx})"), format!("(-{xy})")),
                    Op::Abs => (format!("abs({e})"), format!("(sign({e}) * {xx})"), format!("(sign({e}) * {xy})")),
                    Op::Sine => (format!("sin({e})"), format!("(cos({e}) * {xx})"), format!("(cos({e}) * {xy})")),
                    Op::Cosine => (format!("cos({e})"), format!("(-sin({e}) * {xx})"), format!("(-sin({e}) * {xy})")),
                    Op::Frac => (format!("fract({e})"), xx, xy),
                    _ => (format!("floor({e})"), zero(w), zero(w)),
                };
                out(self.bind(id, 0, w, expr, Some((dx, dy))))
            }
            Op::Step => {
                let (edge, x) = (self.input(id, "Edge"), self.input(id, "X"));
                let w = unify(edge.width, x.width)?;
                let expr = format!("step({}, {})", widen(&edge.expr, edge.width, w).unwrap(), widen(&x.expr, x.width, w).unwrap());
                out(self.bind(id, 0, w, expr, Some((zero(w), zero(w)))))
            }
            Op::SmoothStep => {
                let (lo, hi, x) = (self.input(id, "Min"), self.input(id, "Max"), self.input(id, "X"));
                let w = unify(unify(lo.width, hi.width)?, x.width)?;
                let (el, eh, ex) = (widen(&lo.expr, lo.width, w).unwrap(), widen(&hi.expr, hi.width, w).unwrap(), widen(&x.expr, x.width, w).unwrap());
                let (xx, xy) = Self::d(&x, w);
                // d/dx smoothstep = 6 t (1 - t) / (max - min) for t inside (0, 1).
                let t = format!("saturate(({ex} - {el}) / ({eh} - {el}))");
                let slope = format!("((6.0 * {t} * ({} - {t})) / ({eh} - {el}))", widen("1.0f", 1, w).unwrap());
                out(self.bind(id, 0, w, format!("smoothstep({el}, {eh}, {ex})"), Some((format!("({slope} * {xx})"), format!("({slope} * {xy})")))))
            }
            Op::Dot => {
                let (a, b) = (self.input(id, "A"), self.input(id, "B"));
                if a.width != b.width {
                    return Err(format!("A ({}) and B ({}) must have the same size", ty(a.width), ty(b.width)));
                }
                let w = a.width;
                let ((ax, ay), (bx, by)) = (Self::d(&a, w), Self::d(&b, w));
                let (ea, eb) = (&a.expr, &b.expr);
                if w == 1 {
                    let d = (format!("(({ax} * {eb}) + ({ea} * {bx}))"), format!("(({ay} * {eb}) + ({ea} * {by}))"));
                    return out(self.bind(id, 0, 1, format!("({ea} * {eb})"), Some(d)));
                }
                let d = (format!("(dot({ax}, {eb}) + dot({ea}, {bx}))"), format!("(dot({ay}, {eb}) + dot({ea}, {by}))"));
                out(self.bind(id, 0, 1, format!("dot({ea}, {eb})"), Some(d)))
            }
            Op::Cross => {
                let (a, b) = (self.input(id, "A"), self.input(id, "B"));
                if a.width != 3 || b.width != 3 {
                    return Err("A and B must be float3".into());
                }
                let ((ax, ay), (bx, by)) = (Self::d(&a, 3), Self::d(&b, 3));
                let (ea, eb) = (&a.expr, &b.expr);
                let d = (format!("(cross({ax}, {eb}) + cross({ea}, {bx}))"), format!("(cross({ay}, {eb}) + cross({ea}, {by}))"));
                out(self.bind(id, 0, 3, format!("cross({ea}, {eb})"), Some(d)))
            }
            Op::Normalize => {
                let x = self.input(id, "In");
                let (w, e) = (x.width, x.expr.clone());
                if w == 1 {
                    return out(self.bind(id, 0, 1, format!("sign({e})"), Some(("0.0f".into(), "0.0f".into()))));
                }
                let (xx, xy) = Self::d(&x, w);
                let n = format!("normalize({e})");
                let d = |dd: &str| format!("(({dd} - ({n} * dot({n}, {dd}))) / max(length({e}), 1e-6))");
                let deriv = (d(&xx), d(&xy));
                out(self.bind(id, 0, w, n.clone(), Some(deriv)))
            }
            Op::Length => {
                let x = self.input(id, "In");
                let (w, e) = (x.width, x.expr.clone());
                let (xx, xy) = Self::d(&x, w);
                if w == 1 {
                    return out(self.bind(id, 0, 1, format!("abs({e})"), Some((format!("(sign({e}) * {xx})"), format!("(sign({e}) * {xy})")))));
                }
                let d = |dd: &str| format!("(dot({e}, {dd}) / max(length({e}), 1e-6))");
                let deriv = (d(&xx), d(&xy));
                out(self.bind(id, 0, 1, format!("length({e})"), Some(deriv)))
            }
            Op::Append => {
                let (a, b) = (self.input(id, "A"), self.input(id, "B"));
                let w = a.width + b.width;
                if w > 4 {
                    return Err(format!("{} + {} has more than four components", ty(a.width), ty(b.width)));
                }
                let ((ax, ay), (bx, by)) = (Self::d(&a, a.width), Self::d(&b, b.width));
                let t = ty(w);
                let d = (format!("{t}({ax}, {bx})"), format!("{t}({ay}, {by})"));
                out(self.bind(id, 0, w, format!("{t}({}, {})", a.expr, b.expr), Some(d)))
            }
            Op::ComponentMask { channels } => {
                let x = self.input(id, "In");
                let picks = swizzle(channels)?;
                if let Some(i) = picks.iter().position(|&p| p >= x.width as usize) {
                    return Err(format!("component '{}' does not exist in a {}", channels.chars().nth(i).unwrap_or('?'), ty(x.width)));
                }
                let w = picks.len() as u8;
                let pick = |e: &str| {
                    if x.width == 1 {
                        widen(e, 1, w).unwrap()
                    } else {
                        format!("{e}.{}", picks.iter().map(|&p| ['x', 'y', 'z', 'w'][p]).collect::<String>())
                    }
                };
                let (xx, xy) = Self::d(&x, x.width);
                let d = (pick(&xx), pick(&xy));
                out(self.bind(id, 0, w, pick(&x.expr), Some(d)))
            }
            Op::TransformVector { source, destination } => {
                let v = self.input(id, "In");
                if v.width != 3 {
                    return Err(format!("In must be a float3 direction, not a {}", ty(v.width)));
                }
                if source == destination {
                    return out(v);
                }
                // A linear map: derivatives go through it too (the frame's own change across a
                // pixel is left out, as for normal maps).
                let f = self.context.helpers().transform;
                let (s, d) = (source.code(), destination.code());
                let call = |e: &str| format!("{f}(input, {e}, {s}u, {d}u)");
                let (vx, vy) = Self::d(&v, 3);
                out(self.bind(id, 0, 3, call(&v.expr), Some((call(&vx), call(&vy)))))
            }
            Op::Noise { scale, octaves } => {
                let p = self.input(id, "Position");
                let (position, (px, py)) = match p.width {
                    3 => (p.expr.clone(), Self::d(&p, 3)),
                    2 => {
                        let (dx, dy) = Self::d(&p, 2);
                        (format!("vec3<f32>({}, 0.0)", p.expr), (format!("vec3<f32>({dx}, 0.0)"), format!("vec3<f32>({dy}, 0.0)")))
                    }
                    _ => return Err(format!("Position must be a float2 or float3, not a {}", ty(p.width))),
                };
                let s = self.literal(id, *scale);
                let footprint = format!("(max(length({px}), length({py})) * {s})");
                let expr = format!("saturate(0.5 + (0.5 * fbm({position} * {s}, {}u, {footprint})))", (*octaves).clamp(1, 8));
                out(self.bind(id, 0, 1, expr, None))
            }
            Op::Fresnel { exponent, base_reflect } => {
                let n = self.input(id, "Normal");
                if n.width != 3 {
                    return Err(format!("Normal must be a float3, not a {}", ty(n.width)));
                }
                let (e, b) = (self.literal(id, *exponent), self.literal(id, *base_reflect));
                let facing = format!("saturate(dot({}, normalize(frame.camera_position.xyz - input.world_position)))", n.expr);
                out(self.bind(id, 0, 1, format!("({b} + ((1.0 - {b}) * pow(1.0 - {facing}, {e})))"), None))
            }
            Op::Panner { speed } => {
                let (uv, t) = (self.input(id, "UV"), self.input(id, "Time"));
                if uv.width != 2 || t.width != 1 {
                    return Err(format!("UV must be a float2 and Time a float (got {} and {})", ty(uv.width), ty(t.width)));
                }
                let s = self.vector_literal(id, speed);
                let (ux, uy) = Self::d(&uv, 2);
                let (tx, ty_) = Self::d(&t, 1);
                let d = (format!("({ux} + ({s} * {tx}))"), format!("({uy} + ({s} * {ty_}))"));
                out(self.bind(id, 0, 2, format!("({} + ({s} * {}))", uv.expr, t.expr), Some(d)))
            }
            Op::NormalBlend => {
                let (a, b) = (self.input(id, "Base"), self.input(id, "Detail"));
                if a.width != 3 || b.width != 3 {
                    return Err("Base and Detail must be float3 tangent-space normals".into());
                }
                let blend = self.context.helpers().blend_normals;
                out(self.bind(id, 0, 3, format!("{blend}({}, {})", a.expr, b.expr), None))
            }
            Op::TextureSample { parameter } => {
                let uv = self.input(id, "UV");
                if uv.width != 2 {
                    return Err(format!("UV must be a float2, not a {}", ty(uv.width)));
                }
                let slot = self.slot(parameter);
                let (dx, dy) = Self::d(&uv, 2);
                let t = format!("t{id}");
                let h = self.context.helpers();
                self.line(format!("let {t} = {}({}(m, params, {slot}u), {}, {dx}, {dy});", h.sample, h.texture, uv.expr));
                let field = |e: String, w: u8| Value::plain(e, w);
                Ok(vec![
                    ("RGB", field(format!("{t}.albedo.rgb"), 3)),
                    ("R", field(format!("{t}.albedo.r"), 1)),
                    ("G", field(format!("{t}.albedo.g"), 1)),
                    ("B", field(format!("{t}.albedo.b"), 1)),
                    ("A", field(format!("{t}.albedo.a"), 1)),
                    ("RGBA", field(format!("{t}.albedo"), 4)),
                    ("Normal", field(format!("{}({t}.data, 1.0)", h.decode_normal), 3)),
                    ("Roughness", field(format!("{t}.data.z"), 1)),
                    ("Metallic", field(format!("{t}.data.w"), 1)),
                    ("Height", field(format!("{t}.data.w"), 1)),
                ])
            }
            Op::Triplanar { parameter, scale } => {
                let (p, n) = (self.input(id, "Position"), self.input(id, "Normal"));
                if p.width != 3 || n.width != 3 {
                    return Err("Position and Normal must be float3".into());
                }
                let slot = self.slot(parameter);
                let s = self.literal(id, *scale);
                let (px, py) = Self::d(&p, 3);
                let t = format!("t{id}");
                let h = self.context.helpers();
                self.line(format!(
                    "let {t} = {}({}(m, params, {slot}u), {} * {s}, {px} * {s}, {py} * {s}, normalize({}));",
                    h.triplanar, h.texture, p.expr, n.expr
                ));
                let tangent = if self.displacement() {
                    Value::unavailable(3, "is not available to Displacement (the tessellator has no tangent frame)")
                } else {
                    Value::plain(format!("graph_to_tangent(input, {t}.normal)"), 3)
                };
                Ok(vec![
                    ("RGB", Value::plain(format!("{t}.albedo"), 3)),
                    ("A", Value::plain(format!("{t}.alpha"), 1)),
                    ("Normal", Value::plain(format!("{t}.normal"), 3)),
                    ("TangentNormal", tangent),
                    ("Roughness", Value::plain(format!("{t}.roughness"), 1)),
                    ("Height", Value::plain(format!("{t}.height"), 1)),
                ])
            }
            Op::If => {
                let (a, b) = (self.input(id, "A"), self.input(id, "B"));
                if a.width != 1 || b.width != 1 {
                    return Err("A and B must be floats".into());
                }
                let branches = [self.input(id, "AGreaterB"), self.input(id, "AEqualsB"), self.input(id, "ALessB")];
                let w = unify(unify(branches[0].width, branches[1].width)?, branches[2].width)?;
                let [gt, eq, lt] = branches.each_ref().map(|v| widen(&v.expr, v.width, w).unwrap());
                let expr = format!("select(select({lt}, {eq}, {} == {}), {gt}, {} > {})", a.expr, b.expr, a.expr, b.expr);
                let [(gx, gy), (qx, qy), (lx, ly)] = [Self::d(&branches[0], w), Self::d(&branches[1], w), Self::d(&branches[2], w)];
                let pick = |g: &str, q: &str, l: &str| format!("select(select({l}, {q}, {} == {}), {g}, {} > {})", a.expr, b.expr, a.expr, b.expr);
                let d = (pick(&gx, &qx, &lx), pick(&gy, &qy, &ly));
                out(self.bind(id, 0, w, expr, Some(d)))
            }
            Op::Output => {
                for (pin, field, width) in OUTPUT_PINS {
                    if pin == DISPLACEMENT_PIN || pin == VECTOR_DISPLACEMENT_PIN || !self.sources.contains_key(&(id, pin.to_string())) {
                        continue;
                    }
                    let v = self.input(id, pin);
                    let expr = match (v.width, width) {
                        (w, t) if w == t => v.expr.clone(),
                        (1, t) => widen(&v.expr, 1, t).unwrap(),
                        (4, 3) => format!("{}.xyz", v.expr),
                        (w, t) => return Err(format!("{pin} takes a {}, not a {}", ty(t), ty(w))),
                    };
                    self.line(format!("o.{field} = {expr};"));
                    self.outputs |= crate::nodes::output_bit(pin);
                }
                Ok(Vec::new())
            }
        }
    }
}
