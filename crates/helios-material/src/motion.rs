//! How fast an animated displacement can move a point: a speed bound for the renderer's shadow
//! page cache (UE has none for World Position Offset and re-renders such casters every frame).
//!
//! A cached shadow page under an animated displacement only needs rendering again once the
//! surface may have moved a couple of its texels. The wind and moving instances bound their
//! speed; a graph is arbitrary code, but most animated displacements are simple functions of the
//! time — a constant times `Time` into a sine, a Panner scrolling a noise, products with bounded
//! factors. [`DisplacementMotion`] is the displacement function's nodes, kept by the compiler,
//! and [`DisplacementMotion::speed`] an abstract interpretation of them: every value gets
//!
//! * an *interval* per component that holds it for every point of every surface at every time,
//!   and
//! * a *rate*: a bound of |∂v/∂t| per component (forward-mode differentiation in the time, by
//!   intervals: the product rule takes the factors' largest magnitudes, `sin` and `cos` keep
//!   their argument's rate, a smoothstep's slope is at most 1.5 / (max − min), noise is
//!   Lipschitz).
//!
//! The displacement's rate is then how many units of the magnitude it moves a point per second:
//! a height's (clamped to 0..1, which is 1-Lipschitz), a vector's length of the rates (clamped to
//! the unit ball, a projection: 1-Lipschitz too). Parameters are inputs of the analysis, so a
//! material instance's values give its own bound without recompiling. Where the analysis cannot
//! bound a rate — a discontinuity in the time (`Frac`, `Floor`, `Step`, `If` on a moving
//! value), a texture scrolled by the time, a division by a value that may cross zero, the time
//! raised to a power — the speed is unbounded and the renderer keeps refreshing every frame.

use crate::graph::{Op, VectorSpace};
use crate::nodes::{spec, InputDefault};
use crate::DisplacementKind;

/// The largest slope of `gradient_noise` (noise.wgsl) for any gradients in [-1, 1]³: the
/// maximum over the cell of Σ_c ‖∂(w_c (f − c))/∂d‖₁ for unit directions d, which is 4.625 at
/// the cell's centre along an axis (computed numerically; the bound is attained there). Padded.
pub const NOISE_SLOPE: f64 = 4.7;

/// Frequency step of the noise's octaves (noise.wgsl `fbm`).
const OCTAVE_FREQUENCY: f64 = 2.02;

/// Where a step's input comes from.
#[derive(Clone, Debug, PartialEq)]
enum Source {
    /// Output `pin` of an earlier step.
    Step(usize, &'static str),
    /// An unconnected input (constants already carry the node's override).
    Default(InputDefault),
}

#[derive(Clone, Debug, PartialEq)]
struct Step {
    op: Op,
    inputs: Vec<Source>,
    /// Parameter slot (parameters).
    slot: usize,
}

/// The displacement function's nodes, for its speed bound (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct DisplacementMotion {
    steps: Vec<Step>,
    result: Source,
    kind: DisplacementKind,
    /// Parameter defaults by slot (scalars in x).
    defaults: Vec<[f32; 4]>,
}

impl DisplacementMotion {
    /// The steps of `order` (node ids, each after its inputs; ends with the displacement's root)
    /// whose inputs `input(id, pin)` resolves to (step position of the source node, its output
    /// pin) or the input's default, and the Output pin's source `result`.
    pub(crate) fn new(
        order: &[(u32, Op, usize)],
        mut input: impl FnMut(u32, &str) -> Result<(u32, String), InputDefault>,
        result: Result<(u32, String), InputDefault>,
        kind: DisplacementKind,
        defaults: Vec<[f32; 4]>,
    ) -> Self {
        let position = |id: u32| order.iter().position(|o| o.0 == id).unwrap_or(0);
        let resolve = |r: Result<(u32, String), InputDefault>, order: &[(u32, Op, usize)]| match r {
            Ok((node, pin)) => {
                let step = position(node);
                let pins = spec(&order[step].1).outputs;
                Source::Step(step, pins.iter().copied().find(|p| *p == pin).unwrap_or("Out"))
            }
            Err(default) => Source::Default(default),
        };
        let steps = order
            .iter()
            .map(|(id, op, slot)| {
                let inputs = spec(op).inputs.iter().map(|i| resolve(input(*id, i.name), order)).collect();
                Step { op: op.clone(), inputs, slot: *slot }
            })
            .collect();
        Self { steps, result: resolve(result, order), kind, defaults }
    }

    /// Bound of how fast the displacement moves a point, in units of the magnitude per second
    /// (heights: of the 0..1 height), for parameter values `parameters` (by slot; missing slots
    /// take the graph's defaults). `None`: no bound could be derived.
    pub fn speed(&self, parameters: &[[f32; 4]]) -> Option<f32> {
        let mut values: Vec<Vec<(&'static str, Abs)>> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let inputs: Vec<Abs> = step.inputs.iter().map(|s| fetch(&values, s)).collect();
            let parameter = parameters.get(step.slot).or_else(|| self.defaults.get(step.slot)).copied().unwrap_or([0.0; 4]);
            values.push(evaluate(&step.op, &inputs, parameter));
        }
        let value = fetch(&values, &self.result);
        let rate = match self.kind {
            DisplacementKind::Height => value.rate[0],
            // A scalar broadcasts to (s, s, s).
            DisplacementKind::Vector if value.width == 1 => value.rate[0] * 3f64.sqrt(),
            DisplacementKind::Vector => value.vector_rate(),
        };
        (rate.is_finite()).then_some(rate as f32)
    }
}

fn fetch(values: &[Vec<(&'static str, Abs)>], source: &Source) -> Abs {
    match source {
        Source::Step(step, pin) => values.get(*step).and_then(|v| v.iter().find(|(p, _)| p == pin)).map_or(Abs::unknown(1), |v| v.1),
        Source::Default(default) => match *default {
            InputDefault::Scalar(x) => Abs::constant(&[x]),
            InputDefault::Vector3(v) => Abs::constant(&v),
            InputDefault::TexCoord => Abs::still(2, f64::NEG_INFINITY, f64::INFINITY),
            InputDefault::WorldPosition => Abs::still(3, f64::NEG_INFINITY, f64::INFINITY),
            InputDefault::VertexNormal => Abs::still(3, -1.0, 1.0),
            InputDefault::Time => time(),
            InputDefault::Material => Abs::constant(&[0.0]),
        },
    }
}

/// `Time`: any value (seconds, possibly the previous frame's), one second per second.
fn time() -> Abs {
    Abs { rate: [1.0; 4], ..Abs::still(1, f64::NEG_INFINITY, f64::INFINITY) }
}

/// Abstract value: `width` components, each within [lo, hi] and changing by at most `rate` per
/// second; as a vector, by at most `speed` (tighter than the length of the rates after a
/// rotation, which mixes components).
#[derive(Clone, Copy, Debug)]
struct Abs {
    width: usize,
    lo: [f64; 4],
    hi: [f64; 4],
    rate: [f64; 4],
    speed: f64,
}

/// `a · b` where a zero factor wins over an infinite one (a bounded rate times a constant).
fn mul0(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Rate of a function discontinuous in its inputs: unbounded unless they are still.
fn jump(rates: impl IntoIterator<Item = f64>) -> f64 {
    if rates.into_iter().any(|r| r != 0.0) {
        f64::INFINITY
    } else {
        0.0
    }
}

impl Abs {
    fn still(width: usize, lo: f64, hi: f64) -> Self {
        Self { width, lo: [lo; 4], hi: [hi; 4], rate: [0.0; 4], speed: f64::INFINITY }
    }

    /// Bound of |∂v/∂t| for the whole vector.
    fn vector_rate(&self) -> f64 {
        norm(&self.rate[..self.width]).min(self.speed)
    }

    fn with_speed(self, speed: f64) -> Self {
        Self { speed, ..self }
    }

    /// Anything, changing arbitrarily (what the analysis does not know).
    fn unknown(width: usize) -> Self {
        Self { rate: [f64::INFINITY; 4], ..Self::still(width, f64::NEG_INFINITY, f64::INFINITY) }
    }

    fn constant(v: &[f32]) -> Self {
        let mut a = Self::still(v.len(), 0.0, 0.0);
        for (i, &x) in v.iter().enumerate() {
            a.lo[i] = x as f64;
            a.hi[i] = x as f64;
        }
        a
    }

    /// Component `i`, broadcasting a scalar: (lo, hi, rate).
    fn at(&self, i: usize) -> (f64, f64, f64) {
        let i = if self.width == 1 { 0 } else { i.min(3) };
        (self.lo[i], self.hi[i], self.rate[i])
    }

    /// Component-wise map to `width` components of (lo, hi, rate).
    fn map(width: usize, mut f: impl FnMut(usize) -> (f64, f64, f64)) -> Self {
        let mut a = Self::still(width, 0.0, 0.0);
        for i in 0..width {
            let (lo, hi, rate) = f(i);
            // NaN (∞ − ∞, 0 · ∞ that slipped through) knows nothing.
            a.lo[i] = if lo.is_nan() { f64::NEG_INFINITY } else { lo };
            a.hi[i] = if hi.is_nan() { f64::INFINITY } else { hi };
            a.rate[i] = if rate.is_nan() { f64::INFINITY } else { rate.abs() };
        }
        a
    }

    fn rates(&self) -> &[f64] {
        &self.rate[..self.width]
    }
    /// The largest length of the vector and its smallest (the box's distance from the origin).
    fn lengths(&self) -> (f64, f64) {
        let mut far = 0.0;
        let mut near = 0.0;
        for i in 0..self.width {
            let (lo, hi, _) = self.at(i);
            far += lo.abs().max(hi.abs()).powi(2);
            let gap = if lo > 0.0 { lo } else if hi < 0.0 { -hi } else { 0.0 };
            near += gap * gap;
        }
        (far.sqrt(), near.sqrt())
    }
}

fn interval_mul((al, ah): (f64, f64), (bl, bh): (f64, f64)) -> (f64, f64) {
    let c = [mul0(al, bl), mul0(al, bh), mul0(ah, bl), mul0(ah, bh)];
    (c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

/// Product of components: interval and product-rule rate.
fn product(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    let (lo, hi) = interval_mul((a.0, a.1), (b.0, b.1));
    let (ma, mb) = (a.0.abs().max(a.1.abs()), b.0.abs().max(b.1.abs()));
    (lo, hi, mul0(ma, b.2) + mul0(mb, a.2))
}

fn sum(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

fn difference(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.1, a.1 - b.0, a.2 + b.2)
}

/// The abstract outputs of `op` for its abstract inputs (in pin order) and its parameter value.
fn evaluate(op: &Op, inputs: &[Abs], parameter: [f32; 4]) -> Vec<(&'static str, Abs)> {
    let arg = |i: usize| inputs.get(i).copied().unwrap_or(Abs::unknown(1));
    let out = |a: Abs| vec![("Out", a)];
    let width2 = |a: &Abs, b: &Abs| a.width.max(b.width);
    match op {
        Op::Constant { value } => out(Abs::constant(&[*value])),
        Op::Constant2 { value } => out(Abs::constant(value)),
        Op::Constant3 { value } => out(Abs::constant(value)),
        Op::Constant4 { value } => out(Abs::constant(value)),
        Op::ScalarParameter { .. } => out(Abs::constant(&parameter[..1])),
        Op::VectorParameter { .. } => out(Abs::constant(&parameter)),
        Op::TexCoord { .. } => out(Abs::still(2, f64::NEG_INFINITY, f64::INFINITY)),
        Op::WorldPosition | Op::ObjectPosition => out(Abs::still(3, f64::NEG_INFINITY, f64::INFINITY)),
        Op::VertexNormal => out(Abs::still(3, -1.0, 1.0)),
        Op::Time => out(time()),
        // Rejected in the displacement anyway.
        Op::VertexColor | Op::CameraVector | Op::Fresnel { .. } => out(Abs::unknown(3)),
        Op::Add | Op::Subtract | Op::Multiply | Op::Divide | Op::Min | Op::Max => {
            let (a, b) = (arg(0), arg(1));
            out(Abs::map(width2(&a, &b), |i| {
                let (x, y) = (a.at(i), b.at(i));
                match op {
                    Op::Add => sum(x, y),
                    Op::Subtract => difference(x, y),
                    Op::Multiply => product(x, y),
                    Op::Divide => {
                        if x.2 == 0.0 && y.2 == 0.0 && (y.0 > 0.0 || y.1 < 0.0) {
                            let (lo, hi) = interval_mul((x.0, x.1), (1.0 / y.1, 1.0 / y.0));
                            (lo, hi, 0.0)
                        } else if y.0 > 0.0 || y.1 < 0.0 {
                            // (a / b)' = a' / b − a b' / b².
                            let near = y.0.abs().min(y.1.abs());
                            let (lo, hi) = interval_mul((x.0, x.1), (1.0 / y.1, 1.0 / y.0));
                            (lo, hi, x.2 / near + mul0(x.0.abs().max(x.1.abs()), y.2) / (near * near))
                        } else {
                            // The divisor may cross zero.
                            (f64::NEG_INFINITY, f64::INFINITY, jump([x.2, y.2]))
                        }
                    }
                    Op::Min => (x.0.min(y.0), x.1.min(y.1), x.2.max(y.2)),
                    _ => (x.0.max(y.0), x.1.max(y.1), x.2.max(y.2)),
                }
            }))
        }
        Op::Lerp => {
            let (a, b, t) = (arg(0), arg(1), arg(2));
            out(Abs::map(width2(&a, &b).max(t.width), |i| {
                // a (1 − t) + b t.
                let (x, y, s) = (a.at(i), b.at(i), t.at(i));
                let one_minus = (1.0 - s.1, 1.0 - s.0, s.2);
                let (lo, hi, _) = sum(product(x, one_minus), product(y, s));
                let spread = difference(y, x);
                let rate = mul0(x.2, one_minus.0.abs().max(one_minus.1.abs())) + mul0(y.2, s.0.abs().max(s.1.abs())) + mul0(spread.0.abs().max(spread.1.abs()), s.2);
                (lo, hi, rate)
            }))
        }
        Op::Power => {
            let (base, e) = (arg(0), arg(1));
            out(Abs::map(base.width, |i| {
                let (b, x) = (base.at(i), e.at(i));
                let (lo, hi) = (b.0.max(0.0), b.1.max(0.0));
                let exponent = (x.0 == x.1).then_some(x.0);
                let range = match exponent {
                    Some(p) if p >= 0.0 && hi.is_finite() => (lo.powf(p), hi.powf(p)),
                    _ => (0.0, f64::INFINITY),
                };
                let rate = match exponent {
                    _ if b.2 == 0.0 && x.2 == 0.0 => 0.0,
                    Some(p) if p == 0.0 => 0.0,
                    Some(p) if p >= 1.0 && x.2 == 0.0 => mul0(p * hi.powf(p - 1.0), b.2),
                    _ => f64::INFINITY,
                };
                (range.0, range.1, rate)
            }))
        }
        Op::Clamp => {
            let (x, lo, hi) = (arg(0), arg(1), arg(2));
            out(Abs::map(x.width, |i| {
                // min(max(x, min), max): 1-Lipschitz in each argument.
                let (v, l, h) = (x.at(i), lo.at(i), hi.at(i));
                (v.0.max(l.0).min(h.0), v.1.max(l.1).min(h.1), v.2.max(l.2).max(h.2))
            }))
        }
        Op::Saturate => {
            let x = arg(0);
            out(Abs::map(x.width, |i| {
                let v = x.at(i);
                (v.0.clamp(0.0, 1.0), v.1.clamp(0.0, 1.0), v.2)
            }))
        }
        Op::OneMinus => {
            let x = arg(0);
            out(Abs::map(x.width, |i| {
                let v = x.at(i);
                (1.0 - v.1, 1.0 - v.0, v.2)
            }))
        }
        Op::Abs => {
            let x = arg(0);
            out(Abs::map(x.width, |i| {
                let v = x.at(i);
                let low = if v.0 <= 0.0 && v.1 >= 0.0 { 0.0 } else { v.0.abs().min(v.1.abs()) };
                (low, v.0.abs().max(v.1.abs()), v.2)
            }))
        }
        // |sin'| = |cos| <= 1.
        Op::Sine | Op::Cosine => {
            let x = arg(0);
            out(Abs::map(x.width, |i| (-1.0, 1.0, x.at(i).2)))
        }
        Op::Frac => {
            let x = arg(0);
            out(Abs::map(x.width, |i| (0.0, 1.0, jump([x.at(i).2]))))
        }
        Op::Floor => {
            let x = arg(0);
            out(Abs::map(x.width, |i| {
                let v = x.at(i);
                (v.0.floor(), v.1.floor(), jump([v.2]))
            }))
        }
        Op::Step => {
            let (edge, x) = (arg(0), arg(1));
            out(Abs::map(width2(&edge, &x), |i| (0.0, 1.0, jump([edge.at(i).2, x.at(i).2]))))
        }
        Op::SmoothStep => {
            let (lo, hi, x) = (arg(0), arg(1), arg(2));
            out(Abs::map(lo.width.max(hi.width).max(x.width), |i| {
                let (l, h, v) = (lo.at(i), hi.at(i), x.at(i));
                // Slope 6 t (1 − t) / (max − min) <= 1.5 / |max − min|.
                let span = difference(h, l);
                let rate = if l.2 != 0.0 || h.2 != 0.0 {
                    f64::INFINITY
                } else if span.0 > 0.0 || span.1 < 0.0 {
                    mul0(1.5 / span.0.abs().min(span.1.abs()), v.2)
                } else {
                    jump([v.2])
                };
                (0.0, 1.0, rate)
            }))
        }
        Op::Dot => {
            let (a, b) = (arg(0), arg(1));
            let mut total = (0.0, 0.0, 0.0);
            for i in 0..width2(&a, &b) {
                total = sum(total, product(a.at(i), b.at(i)));
            }
            out(Abs::map(1, |_| total))
        }
        Op::Cross => {
            let (a, b) = (arg(0), arg(1));
            out(Abs::map(3, |i| {
                let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                difference(product(a.at(j), b.at(k)), product(a.at(k), b.at(j)))
            }))
        }
        Op::Normalize => {
            let x = arg(0);
            if x.width == 1 {
                let v = x.at(0);
                return out(Abs::map(1, |_| (-1.0, 1.0, jump([v.2]))));
            }
            // |d(v / |v|)| <= |dv| / |v|.
            let (_, near) = x.lengths();
            let speed = x.vector_rate();
            let rate = if speed == 0.0 { 0.0 } else if near > 0.0 { speed / near } else { f64::INFINITY };
            out(Abs::map(x.width, |_| (-1.0, 1.0, rate)).with_speed(rate))
        }
        Op::Length => {
            let x = arg(0);
            let (far, near) = x.lengths();
            out(Abs::map(1, |_| (near, far, x.vector_rate())))
        }
        Op::Append => {
            let (a, b) = (arg(0), arg(1));
            let width = (a.width + b.width).min(4);
            out(Abs::map(width, |i| if i < a.width { a.at(i) } else { b.at(i - a.width) }))
        }
        Op::ComponentMask { channels } => {
            let x = arg(0);
            let picks: Vec<usize> = channels.chars().filter_map(|c| "xyzw".find(c).or_else(|| "rgba".find(c))).collect();
            out(Abs::map(picks.len().clamp(1, 4), |i| x.at(picks.get(i).copied().unwrap_or(0))))
        }
        Op::TransformVector { source, destination } => {
            let x = arg(0);
            if source == destination {
                return out(x);
            }
            // Into world space through the instance's linear part over its largest scale, or
            // between two orthonormal frames: no longer than before. Out of world space the
            // inverse stretches by the instance's anisotropy, which the graph does not know.
            let contracts = *destination == VectorSpace::World || *source != VectorSpace::World;
            let (far, _) = x.lengths();
            let speed = x.vector_rate();
            if contracts {
                out(Abs::map(3, |_| (-far, far, speed)).with_speed(speed))
            } else {
                out(Abs::map(3, |_| (f64::NEG_INFINITY, f64::INFINITY, jump([speed]))))
            }
        }
        Op::Noise { scale, octaves } => {
            // saturate(0.5 + 0.5 fbm(p · scale)): fbm's octave o has amplitude 0.5^(o+1) at
            // frequency 2.02^o.
            let p = arg(0);
            let octaves = (*octaves).clamp(1, 8) as i32;
            let slope: f64 = (0..octaves).map(|o| 0.5f64.powi(o + 1) * OCTAVE_FREQUENCY.powi(o)).sum::<f64>() * NOISE_SLOPE;
            let rate = mul0(0.5 * (*scale as f64).abs() * slope, p.vector_rate());
            out(Abs::map(1, |_| (0.0, 1.0, rate)))
        }
        Op::Panner { speed } => {
            let (uv, t) = (arg(0), arg(1));
            out(Abs::map(2, |i| {
                let s = speed[i] as f64;
                let (lo, hi) = interval_mul((s, s), (t.at(0).0, t.at(0).1));
                let u = uv.at(i);
                (u.0 + lo, u.1 + hi, u.2 + mul0(s.abs(), t.at(0).2))
            }))
        }
        Op::NormalBlend => {
            let (a, b) = (arg(0), arg(1));
            let rate = jump(a.rates().iter().chain(b.rates()).copied());
            out(Abs::map(3, |_| (-1.0, 1.0, rate)))
        }
        Op::TextureSample { .. } => {
            // A texture's slope is unknown: still coordinates only.
            let rate = jump(arg(0).rates().iter().copied());
            let unit = |width| Abs::map(width, |_| (0.0, 1.0, rate));
            vec![
                ("RGB", unit(3)),
                ("R", unit(1)),
                ("G", unit(1)),
                ("B", unit(1)),
                ("A", unit(1)),
                ("RGBA", unit(4)),
                ("Normal", Abs::map(3, |_| (-1.0, 1.0, rate))),
                ("Roughness", unit(1)),
                ("Metallic", unit(1)),
                ("Height", unit(1)),
            ]
        }
        Op::Triplanar { .. } => {
            let rate = jump(arg(0).rates().iter().chain(arg(1).rates()).copied());
            let unit = |width| Abs::map(width, |_| (0.0, 1.0, rate));
            vec![
                ("RGB", unit(3)),
                ("A", unit(1)),
                ("Normal", Abs::map(3, |_| (-1.0, 1.0, rate))),
                ("TangentNormal", Abs::unknown(3)),
                ("Roughness", unit(1)),
                ("Height", unit(1)),
            ]
        }
        Op::If => {
            let (a, b) = (arg(0), arg(1));
            let branches = [arg(2), arg(3), arg(4)];
            let choice = jump([a.rate[0], b.rate[0]]);
            let width = branches.iter().map(|b| b.width).max().unwrap_or(1);
            out(Abs::map(width, |i| {
                let c = branches.map(|b| b.at(i));
                let lo = c.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
                let hi = c.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
                (lo, hi, c.iter().map(|c| c.2).fold(choice, f64::max))
            }))
        }
        Op::Output => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile, presets, GraphBuilder, VECTOR_DISPLACEMENT_PIN};

    fn speed(build: impl FnOnce(&mut GraphBuilder, u32)) -> Option<f32> {
        let mut g = GraphBuilder::new("t");
        let output = g.node(Op::Output, [0.0, 0.0]);
        build(&mut g, output);
        let compiled = compile(&g.build()).unwrap();
        compiled.displacement_speed(&[])
    }

    #[test]
    fn sines_of_scaled_time_are_bounded_by_their_frequency() {
        // 0.5 + 0.3 sin(2 t + x): at most 0.6 per second.
        let s = speed(|g, output| {
            let time = g.node(Op::Time, [0.0, 0.0]);
            let fast = g.node(Op::Multiply, [0.0, 0.0]);
            let x = g.node(Op::WorldPosition, [0.0, 0.0]);
            let mask = g.node(Op::ComponentMask { channels: "x".into() }, [0.0, 0.0]);
            let phase = g.node(Op::Add, [0.0, 0.0]);
            let wave = g.node(Op::Sine, [0.0, 0.0]);
            let amplitude = g.node(Op::Multiply, [0.0, 0.0]);
            let lift = g.node(Op::Add, [0.0, 0.0]);
            g.link(time, "Out", fast, "A").value(fast, "B", 2.0).link(x, "Out", mask, "In").link(mask, "Out", phase, "A").link(fast, "Out", phase, "B");
            g.link(phase, "Out", wave, "In").link(wave, "Out", amplitude, "A").value(amplitude, "B", 0.3);
            g.link(amplitude, "Out", lift, "A").value(lift, "B", 0.5).link(lift, "Out", output, "Displacement");
        });
        assert!((s.unwrap() - 0.6).abs() < 1e-6, "{s:?}");
    }

    #[test]
    fn static_graphs_do_not_move_and_discontinuities_are_unbounded() {
        assert_eq!(speed(|g, output| {
            let noise = g.node(Op::Noise { scale: 3.0, octaves: 4 }, [0.0, 0.0]);
            g.link(noise, "Out", output, "Displacement");
        }), Some(0.0));
        assert_eq!(speed(|g, output| {
            let time = g.node(Op::Time, [0.0, 0.0]);
            let frac = g.node(Op::Frac, [0.0, 0.0]);
            g.link(time, "Out", frac, "In").link(frac, "Out", output, "Displacement");
        }), None);
        // Accelerating: t · t has no bound.
        assert_eq!(speed(|g, output| {
            let time = g.node(Op::Time, [0.0, 0.0]);
            let square = g.node(Op::Multiply, [0.0, 0.0]);
            let wave = g.node(Op::Sine, [0.0, 0.0]);
            g.link(time, "Out", square, "A").link(time, "Out", square, "B").link(square, "Out", wave, "In").link(wave, "Out", output, "Displacement");
        }), None);
        // A texture scrolled by the time: its slope is unknown.
        assert_eq!(speed(|g, output| {
            let pan = g.node(Op::Panner { speed: [0.1, 0.0] }, [0.0, 0.0]);
            let tex = g.node(Op::TextureSample { parameter: "H".into() }, [0.0, 0.0]);
            g.link(pan, "Out", tex, "UV").link(tex, "Height", output, "Displacement");
        }), None);
    }

    #[test]
    fn parameters_scale_the_bound_per_material() {
        let mut g = GraphBuilder::new("t");
        let output = g.node(Op::Output, [0.0, 0.0]);
        let time = g.node(Op::Time, [0.0, 0.0]);
        let rate = g.node(Op::ScalarParameter { name: "Rate".into(), default: 1.0, min: 0.0, max: 4.0 }, [0.0, 0.0]);
        let clock = g.node(Op::Multiply, [0.0, 0.0]);
        let wave = g.node(Op::Cosine, [0.0, 0.0]);
        let half = g.node(Op::Multiply, [0.0, 0.0]);
        g.link(time, "Out", clock, "A").link(rate, "Out", clock, "B").link(clock, "Out", wave, "In").link(wave, "Out", half, "A").value(half, "B", 0.5);
        g.link(half, "Out", output, "Displacement");
        let compiled = compile(&g.build()).unwrap();
        assert_eq!(compiled.displacement_speed(&[]), Some(0.5), "the default");
        assert_eq!(compiled.displacement_speed(&[[3.0, 0.0, 0.0, 0.0]]), Some(1.5), "an instance's value");
    }

    #[test]
    fn noise_scrolled_by_a_panner_is_lipschitz() {
        let s = speed(|g, output| {
            let pan = g.node(Op::Panner { speed: [0.3, 0.4] }, [0.0, 0.0]);
            let noise = g.node(Op::Noise { scale: 2.0, octaves: 1 }, [0.0, 0.0]);
            g.link(pan, "Out", noise, "Position").link(noise, "Out", output, "Displacement");
        })
        .unwrap();
        // 0.5 · scale · 0.5 (first octave) · slope · |speed|.
        assert!((s as f64 - 0.5 * 2.0 * 0.5 * NOISE_SLOPE * 0.5).abs() < 1e-5, "{s}");
    }

    #[test]
    fn vectors_turned_into_world_space_keep_their_speed() {
        let s = speed(|g, output| {
            let time = g.node(Op::Time, [0.0, 0.0]);
            let wave = g.node(Op::Sine, [0.0, 0.0]);
            let lean = g.node(Op::Constant3 { value: [0.6, 0.8, 0.0] }, [0.0, 0.0]);
            let swing = g.node(Op::Multiply, [0.0, 0.0]);
            let turn = g.node(Op::TransformVector { source: VectorSpace::Object, destination: VectorSpace::World }, [0.0, 0.0]);
            g.link(time, "Out", wave, "In").link(wave, "Out", swing, "A").link(lean, "Out", swing, "B").link(swing, "Out", turn, "In");
            g.link(turn, "Out", output, VECTOR_DISPLACEMENT_PIN);
        })
        .unwrap();
        // |(0.6, 0.8)| · 1: a rotation keeps the vector's speed (not the √3 of its components').
        assert!((s - 1.0).abs() < 1e-5, "{s}");
        // Out of world space the instance's anisotropy is unknown.
        assert_eq!(speed(|g, output| {
            let time = g.node(Op::Time, [0.0, 0.0]);
            let wave = g.node(Op::Sine, [0.0, 0.0]);
            let up = g.node(Op::Constant3 { value: [0.0, 1.0, 0.0] }, [0.0, 0.0]);
            let swing = g.node(Op::Multiply, [0.0, 0.0]);
            let turn = g.node(Op::TransformVector { source: VectorSpace::World, destination: VectorSpace::Object }, [0.0, 0.0]);
            g.link(time, "Out", wave, "In").link(wave, "Out", swing, "A").link(up, "Out", swing, "B");
            g.link(swing, "Out", turn, "In").link(turn, "Out", output, VECTOR_DISPLACEMENT_PIN);
        }), None);
    }

    #[test]
    fn the_cloth_has_a_bound() {
        // Two waves (0.27 · 1.9 + 0.1 · 3.1 per second), scrolling wrinkles, pinned edges.
        let cloth = compile(&presets::rippling_cloth()).unwrap();
        let s = cloth.displacement_speed(&[]).unwrap();
        let waves = 0.27 * 1.9 + 0.1 * 3.1;
        assert!(s > waves && s < 2.0, "{s}");
        // Static presets do not move.
        for graph in [presets::brick_pavement(), presets::mushrooms()] {
            assert_eq!(compile(&graph).unwrap().displacement_speed(&[]), Some(0.0));
        }
    }

    #[test]
    fn sampled_rates_stay_under_the_bound() {
        // Finite differences of a CPU model of a bounded graph never exceed its bound:
        // h = 0.5 + 0.2 sin(3 t) · smoothstep(0, 0.5, x) + 0.1 cos(t · p) / (2 + sin t).
        let p = 1.7f64;
        let model = |t: f64, x: f64| {
            let s = (x / 0.5).clamp(0.0, 1.0);
            0.5 + 0.2 * (3.0 * t).sin() * (s * s * (3.0 - 2.0 * s)) + 0.1 * (t * p).cos() / (2.0 + t.sin())
        };
        let mut g = GraphBuilder::new("t");
        let output = g.node(Op::Output, [0.0, 0.0]);
        let time = g.node(Op::Time, [0.0, 0.0]);
        let fast = g.node(Op::Multiply, [0.0, 0.0]);
        let wave = g.node(Op::Sine, [0.0, 0.0]);
        let x = g.node(Op::ObjectPosition, [0.0, 0.0]);
        let mask = g.node(Op::ComponentMask { channels: "x".into() }, [0.0, 0.0]);
        let ramp = g.node(Op::SmoothStep, [0.0, 0.0]);
        let shaped = g.node(Op::Multiply, [0.0, 0.0]);
        let amplitude = g.node(Op::Multiply, [0.0, 0.0]);
        let rate = g.node(Op::ScalarParameter { name: "P".into(), default: p as f32, min: 0.0, max: 4.0 }, [0.0, 0.0]);
        let clock = g.node(Op::Multiply, [0.0, 0.0]);
        let slow = g.node(Op::Cosine, [0.0, 0.0]);
        let sine = g.node(Op::Sine, [0.0, 0.0]);
        let denominator = g.node(Op::Add, [0.0, 0.0]);
        let ratio = g.node(Op::Divide, [0.0, 0.0]);
        let small = g.node(Op::Multiply, [0.0, 0.0]);
        let both = g.node(Op::Add, [0.0, 0.0]);
        let lift = g.node(Op::Add, [0.0, 0.0]);
        g.link(time, "Out", fast, "A").value(fast, "B", 3.0).link(fast, "Out", wave, "In");
        g.link(x, "Out", mask, "In").value(ramp, "Min", 0.0).value(ramp, "Max", 0.5).link(mask, "Out", ramp, "X");
        g.link(wave, "Out", shaped, "A").link(ramp, "Out", shaped, "B").link(shaped, "Out", amplitude, "A").value(amplitude, "B", 0.2);
        g.link(time, "Out", clock, "A").link(rate, "Out", clock, "B").link(clock, "Out", slow, "In");
        g.link(time, "Out", sine, "In").link(sine, "Out", denominator, "A").value(denominator, "B", 2.0);
        g.link(slow, "Out", ratio, "A").link(denominator, "Out", ratio, "B").link(ratio, "Out", small, "A").value(small, "B", 0.1);
        g.link(amplitude, "Out", both, "A").link(small, "Out", both, "B").link(both, "Out", lift, "A").value(lift, "B", 0.5);
        g.link(lift, "Out", output, "Displacement");
        let bound = compile(&g.build()).unwrap().displacement_speed(&[]).unwrap() as f64;
        let mut worst = 0.0f64;
        for i in 0..20000 {
            let t = i as f64 * 0.00731;
            let x = (i % 97) as f64 / 97.0;
            let dt = 1e-5;
            worst = worst.max(((model(t + dt, x) - model(t, x)) / dt).abs());
        }
        assert!(worst <= bound && bound < worst * 3.0, "sampled {worst}, bound {bound}");
    }
}
