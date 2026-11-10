//! Where the CPU work of Lumen's displaced variants (`lumen_displaced.rs`) runs: the refinement
//! of a mesh's triangles into points, and the bake of a variant's distance field and cards.
//!
//! Each piece of work is a [`Work`] value that carries all its inputs and computes an [`Output`]
//! from them alone, so it can run anywhere:
//!
//! * **natively** on a worker thread per task;
//! * **in the browser** on Web Workers that run a second instance of the wasm module. Threads in
//!   wasm need `SharedArrayBuffer`, i.e. a cross-origin isolated page (COOP/COEP headers, which
//!   GitHub Pages cannot send); workers that share nothing need neither. The page installs them
//!   with [`set_bake_offload`] (the demo's `web_bake.rs`); the work crosses as bytes
//!   ([`Work::encode`]), the worker runs [`run_bake_task`] (the same code as natively, so the
//!   result is bit-identical), and the result comes back as bytes ([`Output::decode`]);
//! * **on the render thread**, one task per frame, in a browser without workers (or when a
//!   worker fails: its task is run here instead).
//!
//! The wire format is a flat little-endian record: scalars as `u32` / `f32` bits, arrays as a
//! count and their raw elements. It never leaves the page (both ends are the same build), so it
//! has no version; every struct is written and read field by field with exhaustive patterns, so
//! a field added to a baked type does not compile until it is carried here too.

use std::cell::RefCell;

use bytemuck::Pod;
use glam::{Vec2, Vec3};
use helios_geometry::cards::{Card, MeshCards, TexelSource};
use helios_geometry::{Aabb, MeshSdf, SdfKind};

use crate::cards::SurfaceCapture;
use crate::lumen_displaced::{bake, refine, Corner, DisplacedVariant, Refined, SurfacePoint};

/// A displaced variant's CPU work.
pub(crate) enum Work {
    /// [`refine`] a mesh's triangles (a LOD cut) with a point spacing; without `triangles` the
    /// output keeps the points only (the page needs them for the GPU, the bake refines again).
    Refine { corners: Vec<[Corner; 3]>, spacing: f32, triangles: bool },
    /// [`bake`] the displaced `positions` of a refinement's points with `base`'s kind of field at
    /// its voxel size (`base` carries no voxels: only its kind and voxel size are read).
    Bake { base: MeshSdf, mesh: BakeMesh, positions: Vec<Vec3>, card_scale: f32, serial: u64 },
}

/// The refinement a bake connects its positions with (its triangles) and evaluates its cards'
/// material at (its points' undisplaced positions and UVs).
pub(crate) enum BakeMesh {
    /// The refinement itself.
    Refined(Refined),
    /// The refinement made again from its input (LOD cut, spacing): an offloaded bake gets these
    /// few kilobytes instead of megabytes of points and triangles that would cross to the page and back.
    /// [`refine`] is deterministic, so the triangles are the same.
    Refine(Vec<[Corner; 3]>, f32),
}

/// What a [`Work`] computed.
#[derive(Debug, PartialEq)]
pub(crate) enum Output {
    Refined(Refined),
    /// The variant (`None`: nothing to bake) and the CPU time of the bake (ms).
    Baked(Option<DisplacedVariant>, f32),
}

impl Work {
    pub fn run(self) -> Output {
        match self {
            Work::Refine { corners, spacing, triangles } => {
                let mut refined = refine(&corners, spacing);
                if !triangles {
                    refined.triangles = Vec::new();
                }
                Output::Refined(refined)
            }
            Work::Bake { base, mesh, positions, card_scale, serial } => {
                let start = web_time::Instant::now();
                let refined = match mesh {
                    BakeMesh::Refined(refined) => refined,
                    BakeMesh::Refine(corners, spacing) => refine(&corners, spacing),
                };
                let variant = bake(&base, &refined, &positions, card_scale, serial);
                Output::Baked(variant, start.elapsed().as_secs_f32() * 1000.0)
            }
        }
    }

    #[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
    pub fn encode(&self) -> Vec<u8> {
        // Sized up front: a bake's positions are megabytes, which growing would copy again.
        let size = match self {
            Work::Refine { corners, .. } => corners.len() * 96,
            Work::Bake { mesh: BakeMesh::Refined(r), positions, .. } => r.triangles.len() * 12 + r.points.len() * 32 + positions.len() * 12,
            Work::Bake { mesh: BakeMesh::Refine(c, _), positions, .. } => c.len() * 96 + positions.len() * 12,
        };
        let mut w = Writer(Vec::with_capacity(size + 128));
        match self {
            Work::Refine { corners, spacing, triangles } => {
                w.u32(0);
                w.corners(corners);
                w.f32(*spacing);
                w.u32(*triangles as u32);
            }
            Work::Bake { base, mesh, positions, card_scale, serial } => {
                w.u32(1);
                w.sdf(base);
                match mesh {
                    BakeMesh::Refined(refined) => {
                        w.u32(0);
                        w.refined(refined);
                    }
                    BakeMesh::Refine(corners, spacing) => {
                        w.u32(1);
                        w.corners(corners);
                        w.f32(*spacing);
                    }
                }
                w.array(positions);
                w.f32(*card_scale);
                w.u64(*serial);
            }
        }
        w.0
    }

    #[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader { bytes, at: 0 };
        let work = match r.u32()? {
            0 => Work::Refine { corners: r.corners()?, spacing: r.f32()?, triangles: r.u32()? != 0 },
            1 => {
                let base = r.sdf()?;
                let mesh = match r.u32()? {
                    0 => BakeMesh::Refined(r.refined()?),
                    1 => BakeMesh::Refine(r.corners()?, r.f32()?),
                    _ => return None,
                };
                Work::Bake { base, mesh, positions: r.array()?, card_scale: r.f32()?, serial: r.u64()? }
            }
            _ => return None,
        };
        r.end(work)
    }
}

impl Output {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        match self {
            Output::Refined(refined) => {
                w.u32(0);
                w.refined(refined);
            }
            Output::Baked(variant, ms) => {
                w.u32(1);
                w.f32(*ms);
                w.u32(variant.is_some() as u32);
                if let Some(DisplacedVariant { sdf, aabb, triangles, cards, serial }) = variant {
                    w.sdf(sdf);
                    w.aabb(aabb);
                    let flat: Vec<[f32; 18]> = triangles.iter().map(|t| bytemuck::cast(t.map(|(p, n)| [p.x, p.y, p.z, n.x, n.y, n.z]))).collect();
                    w.array(&flat);
                    w.u32(cards.is_some() as u32);
                    if let Some((scale, texel, SurfaceCapture { cards, attributes })) = cards {
                        w.f32(*scale);
                        w.f32(*texel);
                        w.cards(cards);
                        w.u32(attributes.len() as u32);
                        for texels in attributes {
                            w.array(texels);
                        }
                    }
                    w.u64(*serial);
                }
            }
        }
        w.0
    }

    #[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader { bytes, at: 0 };
        let output = match r.u32()? {
            0 => Output::Refined(r.refined()?),
            1 => {
                let ms = r.f32()?;
                let variant = match r.u32()? {
                    0 => None,
                    _ => {
                        let sdf = r.sdf()?;
                        let aabb = r.aabb()?;
                        // Straight from the bytes (no intermediate array: a bed's are 9 MB).
                        let count = r.u32()? as usize;
                        let triangles = r
                            .take(count.checked_mul(72)?)?
                            .chunks_exact(72)
                            .map(|t| {
                                let t: [[f32; 6]; 3] = bytemuck::pod_read_unaligned(t);
                                t.map(|v| (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
                            })
                            .collect();
                        let cards = match r.u32()? {
                            0 => None,
                            _ => Some((r.f32()?, r.f32()?, r.capture()?)),
                        };
                        Some(DisplacedVariant { sdf, aabb, triangles, cards, serial: r.u64()? })
                    }
                };
                Output::Baked(variant, ms)
            }
            _ => return None,
        };
        r.end(output)
    }
}

/// The worker's entry: decodes a [`Work`], runs it, encodes its [`Output`] (empty for input that
/// does not decode, which the page then runs itself).
pub fn run_bake_task(input: &[u8]) -> Vec<u8> {
    Work::decode(input).map(|work| work.run().encode()).unwrap_or_default()
}

/// A finished offloaded task's state (see [`BakeOffload::poll`]).
pub enum BakePoll {
    Pending,
    Done(Vec<u8>),
    /// The worker failed (it could not start, or the task trapped): the page runs it itself.
    Failed,
}

/// Runs [`run_bake_task`] off the render thread: the browser's Web Workers (installed by the
/// page with [`set_bake_offload`]).
pub trait BakeOffload {
    /// Starts a task with `input` ([`Work::encode`]); `None` if no worker can take it.
    fn submit(&self, input: &[u8]) -> Option<u32>;
    /// The state of task `id` (a finished task's result is returned once).
    fn poll(&self, id: u32) -> BakePoll;
    /// Task `id` is no longer wanted (its result, if any, is dropped).
    fn cancel(&self, id: u32);
}

thread_local! {
    static OFFLOAD: RefCell<Option<Box<dyn BakeOffload>>> = const { RefCell::new(None) };
}

/// Installs (or removes) where displaced-variant work runs in the browser (natively it runs on
/// threads and this is not consulted).
pub fn set_bake_offload(offload: Option<Box<dyn BakeOffload>>) {
    OFFLOAD.with(|o| *o.borrow_mut() = offload);
}

/// Whether work is offloaded (an offload is installed: in the browser, with Web Workers).
pub(crate) fn offloading() -> bool {
    OFFLOAD.with(|o| o.borrow().is_some())
}

/// Submits `work` to the installed offload; `None` without one (or if it declines).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn submit(work: &Work) -> Option<Offloaded> {
    OFFLOAD.with(|o| {
        let offload = o.borrow();
        let offload = offload.as_ref()?;
        offload.submit(&work.encode()).map(Offloaded)
    })
}

/// A task running on the offload; dropped unfinished (a scene left), it is cancelled.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) struct Offloaded(u32);

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl Offloaded {
    /// Its state ([`BakePoll::Failed`] if the offload was removed).
    pub fn poll(&self) -> BakePoll {
        OFFLOAD.with(|o| o.borrow().as_ref().map_or(BakePoll::Failed, |offload| offload.poll(self.0)))
    }
}

impl Drop for Offloaded {
    fn drop(&mut self) {
        // (After a result was taken this finds nothing to cancel.)
        let _ = OFFLOAD.try_with(|o| {
            if let Ok(Some(offload)) = o.try_borrow().as_deref() {
                offload.cancel(self.0);
            }
        });
    }
}

fn corner_floats(c: Corner) -> [f32; 8] {
    [c.0.x, c.0.y, c.0.z, c.1.x, c.1.y, c.1.z, c.2.x, c.2.y]
}

fn floats_corner(c: &[f32; 8]) -> Corner {
    (Vec3::new(c[0], c[1], c[2]), Vec3::new(c[3], c[4], c[5]), Vec2::new(c[6], c[7]))
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }

    fn vec3(&mut self, v: Vec3) {
        for x in v.to_array() {
            self.f32(x);
        }
    }

    fn array<T: Pod>(&mut self, items: &[T]) {
        self.u32(items.len() as u32);
        self.0.extend_from_slice(bytemuck::cast_slice(items));
    }

    fn refined(&mut self, Refined { points, triangles }: &Refined) {
        self.array(points);
        self.array(triangles);
    }

    fn corners(&mut self, corners: &[[Corner; 3]]) {
        let flat: Vec<[f32; 24]> = corners.iter().map(|t| bytemuck::cast(t.map(corner_floats))).collect();
        self.array(&flat);
    }

    fn sdf(&mut self, MeshSdf { kind, origin, voxel_size, resolution, band, height_range, two_sided, words }: &MeshSdf) {
        self.u32(*kind as u32);
        self.vec3(*origin);
        self.f32(*voxel_size);
        for &r in resolution {
            self.u32(r);
        }
        self.f32(*band);
        for &h in height_range {
            self.f32(h);
        }
        self.u32(*two_sided as u32);
        self.array(words);
    }

    fn aabb(&mut self, Aabb { min, max }: &Aabb) {
        self.vec3(*min);
        self.vec3(*max);
    }

    fn cards(&mut self, MeshCards { texel_size, cards }: &MeshCards) {
        self.f32(*texel_size);
        self.u32(cards.len() as u32);
        for Card { axis, layer, origin, resolution, plane, depth_range, texels, sources } in cards {
            self.u32(*axis);
            self.u32(*layer);
            self.f32(origin.x);
            self.f32(origin.y);
            for &r in resolution {
                self.u32(r);
            }
            self.f32(*plane);
            for &d in depth_range {
                self.f32(d);
            }
            self.array(texels);
            let sources: Vec<[u32; 3]> = sources.iter().map(|&TexelSource { triangle, barycentrics: [b1, b2] }| [triangle, b1.to_bits(), b2.to_bits()]).collect();
            self.array(&sources);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let slice = self.bytes.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn f32(&mut self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }

    fn vec3(&mut self) -> Option<Vec3> {
        Some(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }

    /// Copied: the bytes need not be aligned for `T`.
    fn array<T: Pod>(&mut self) -> Option<Vec<T>> {
        let n = self.u32()? as usize;
        Some(bytemuck::pod_collect_to_vec(self.take(n.checked_mul(std::mem::size_of::<T>())?)?))
    }

    /// `value` if every byte was read.
    fn end<T>(&self, value: T) -> Option<T> {
        (self.at == self.bytes.len()).then_some(value)
    }

    fn refined(&mut self) -> Option<Refined> {
        Some(Refined { points: self.array::<SurfacePoint>()?, triangles: self.array()? })
    }

    fn corners(&mut self) -> Option<Vec<[Corner; 3]>> {
        let flat: Vec<[f32; 24]> = self.array()?;
        Some(flat.iter().map(|t| bytemuck::cast::<[f32; 24], [[f32; 8]; 3]>(*t).map(|c| floats_corner(&c))).collect())
    }

    fn sdf(&mut self) -> Option<MeshSdf> {
        let kind = match self.u32()? {
            0 => SdfKind::Volume,
            1 => SdfKind::Heightfield,
            _ => return None,
        };
        Some(MeshSdf {
            kind,
            origin: self.vec3()?,
            voxel_size: self.f32()?,
            resolution: [self.u32()?, self.u32()?, self.u32()?],
            band: self.f32()?,
            height_range: [self.f32()?, self.f32()?],
            two_sided: self.u32()? != 0,
            words: self.array()?,
        })
    }

    fn aabb(&mut self) -> Option<Aabb> {
        Some(Aabb { min: self.vec3()?, max: self.vec3()? })
    }

    fn cards(&mut self) -> Option<MeshCards> {
        let texel_size = self.f32()?;
        let count = self.u32()? as usize;
        let mut cards = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            cards.push(Card {
                axis: self.u32()?,
                layer: self.u32()?,
                origin: Vec2::new(self.f32()?, self.f32()?),
                resolution: [self.u32()?, self.u32()?],
                plane: self.f32()?,
                depth_range: [self.f32()?, self.f32()?],
                texels: self.array()?,
                sources: self
                    .array::<[u32; 3]>()?
                    .into_iter()
                    .map(|[triangle, b1, b2]| TexelSource { triangle, barycentrics: [f32::from_bits(b1), f32::from_bits(b2)] })
                    .collect(),
            });
        }
        Some(MeshCards { texel_size, cards })
    }

    fn capture(&mut self) -> Option<SurfaceCapture> {
        let cards = self.cards()?;
        let count = self.u32()? as usize;
        let mut attributes = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            attributes.push(self.array()?);
        }
        Some(SurfaceCapture { cards, attributes })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn corner(x: f32, z: f32) -> Corner {
        (Vec3::new(x, 0.0, z), Vec3::Y, Vec2::new(x, z))
    }

    /// A 1 m sheet (as in `lumen_displaced`'s bake test) and its refinement work.
    fn sheet() -> (Vec<[Corner; 3]>, f32, MeshSdf) {
        let quad = vec![[corner(-0.5, -0.5), corner(0.5, 0.5), corner(0.5, -0.5)], [corner(-0.5, -0.5), corner(-0.5, 0.5), corner(0.5, 0.5)]];
        let flat: Vec<[Vec3; 3]> = quad.iter().map(|t| t.map(|c| c.0)).collect();
        let base = MeshSdf::bake_volume(&flat, 32).unwrap();
        (quad, 0.06, MeshSdf { words: Vec::new(), ..base })
    }

    /// The points "displaced" by a bump in the middle, as the GPU would.
    fn bumped(refined: &Refined) -> Vec<Vec3> {
        refined
            .points
            .iter()
            .map(|p| {
                let q = Vec3::from_slice(&p.object_u[..3]);
                q + Vec3::Y * (0.2 * (1.0 - (q.x * q.x + q.z * q.z) / 0.09).max(0.0))
            })
            .collect()
    }

    fn variant(output: Output) -> DisplacedVariant {
        match output {
            Output::Baked(Some(variant), _) => variant,
            _ => panic!("no variant"),
        }
    }

    /// The offloaded path — refined in a worker without the triangles, baked in a worker that
    /// refines again — bakes exactly what the native path (everything in place) bakes.
    #[test]
    fn offloaded_work_is_bit_identical() {
        let (corners, spacing, base) = sheet();
        let local = match (Work::Refine { corners: corners.clone(), spacing, triangles: true }).run() {
            Output::Refined(refined) => refined,
            _ => panic!("not a refinement"),
        };
        assert!(local.triangles.len() > 100);
        let work = Work::Refine { corners: corners.clone(), spacing, triangles: false };
        let remote = match Output::decode(&run_bake_task(&work.encode())).unwrap() {
            Output::Refined(refined) => refined,
            _ => panic!("not a refinement"),
        };
        assert_eq!(remote.points, local.points);
        assert!(remote.triangles.is_empty());

        let positions = bumped(&local);
        let bake = |mesh| Work::Bake { base: base.clone(), mesh, positions: positions.clone(), card_scale: 1.0, serial: 7 };
        let a = variant(bake(BakeMesh::Refined(Refined { points: local.points.clone(), triangles: local.triangles.clone() })).run());
        let encoded = run_bake_task(&bake(BakeMesh::Refine(corners, spacing)).encode());
        let b = variant(Output::decode(&encoded).unwrap());
        // Everything baked is identical, to the bit (only the bake time differs).
        assert_eq!(a, b);
        assert!(a.cards.as_ref().is_some_and(|c| !c.2.cards.cards.is_empty() && c.2.attributes.len() == c.2.cards.cards.len()));
        assert_eq!(b.serial, 7);
        assert_eq!(Output::decode(&encoded).unwrap().encode(), encoded);
    }

    #[test]
    fn malformed_input_is_rejected() {
        let (corners, spacing, _) = sheet();
        let bytes = (Work::Refine { corners, spacing, triangles: true }).encode();
        assert!(Work::decode(&bytes).is_some());
        assert!(Work::decode(&bytes[..bytes.len() - 1]).is_none());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(Work::decode(&longer).is_none());
        assert!(run_bake_task(&[9, 0, 0, 0]).is_empty());
        assert!(Output::decode(&[]).is_none());
    }
}
