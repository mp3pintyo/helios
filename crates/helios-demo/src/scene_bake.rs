//! The scene bake, spread over Web Workers in the browser.
//!
//! Building a procedural scene is mostly baking its meshes (the cluster DAG, pages and distance
//! field of each; 93 % of the valley's 5.6 s in the browser), and the browser ran it on the
//! page's thread: the page froze for 3–6 s before its first image. Natively it takes a few
//! threads' time; the browser has no threads without cross-origin isolation (see
//! `web_bake.rs`), but it has Web Workers, which the displaced-variant bakes already use.
//!
//! So the page hands the bake to the workers as [`Task`]s, one per worker. Every worker runs
//! the scene's ordinary builder with a [`Shard`] mesh store: all workers see the same meshes in
//! the same order and assign them alike without talking to each other (the least loaded worker
//! takes the next mesh, by source triangles), so each bakes its share and skips the rest. The
//! meshes come back as `.hvm` bytes ([`VirtualMesh::to_bytes`], the streaming format), and the
//! page builds the scene once more with a [`Baked`] store, which takes the baked meshes instead
//! of baking them: materials, instances and lights are the builder's own, and the meshes are
//! bit-identical to a page bake (the same code on the same input). The page's share is the
//! builder without its bakes (≈ 0.3 s for the valley) and decoding the meshes.
//!
//! A worker that fails or stays silent (the pool's watchdog, `bake_worker.js`) leaves its
//! meshes missing, and the page bakes those itself, as without workers. Imported glTF models
//! bake on the page (their meshes would have to cross to the workers first).

use helios_geometry::VirtualMesh;

use crate::mesh_store::{Baked, MeshStore, Shard};
use crate::scenes::{self, Detail, SceneKind, SceneSetup};

/// The first word of a scene-bake task (Lumen's tasks start with 0 or 1).
const TASK_MAGIC: u32 = u32::from_le_bytes(*b"HSBT");
/// The first word of a task's result.
const OUTPUT_MAGIC: u32 = u32::from_le_bytes(*b"HSBM");

/// One worker's share of a scene bake, with what else the builders read (the cathedral's
/// candle lights and lantern are page settings).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Task {
    pub kind: SceneKind,
    pub detail: Detail,
    pub shard: u32,
    pub shards: u32,
    /// The cathedral's candle lights (0: the fixture lights).
    pub candles: u32,
    pub lantern: bool,
}

impl Task {
    /// A task for shard `shard` of `shards` with the page's current scene settings.
    pub fn new(kind: SceneKind, detail: Detail, shard: u32, shards: u32) -> Self {
        let candles = scenes::candles::candle_lights().unwrap_or(0) as u32;
        Self { kind, detail, shard, shards, candles, lantern: scenes::cathedral::lantern_on() }
    }

    pub fn encode(&self) -> Vec<u8> {
        let kind = SceneKind::ALL.iter().position(|&k| k == self.kind).unwrap_or(0) as u32;
        let detail = matches!(self.detail, Detail::High) as u32;
        [TASK_MAGIC, kind, detail, self.shard, self.shards, self.candles, self.lantern as u32].iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let [magic, kind, detail, shard, shards, candles, lantern] = words[..] else { return None };
        if magic != TASK_MAGIC || shard >= shards {
            return None;
        }
        Some(Self {
            kind: *SceneKind::ALL.get(kind as usize)?,
            detail: if detail == 1 { Detail::High } else { Detail::Medium },
            shard,
            shards,
            candles,
            lantern: lantern != 0,
        })
    }
}

/// Whether `input` is a scene-bake task (and not one of Lumen's).
pub fn is_task(input: &[u8]) -> bool {
    input.get(..4).is_some_and(|m| m == TASK_MAGIC.to_le_bytes())
}

/// A worker's entry: bakes the task's shard and returns its meshes ([`encode_meshes`]); empty
/// if the input is not a task.
pub fn run_task(input: &[u8]) -> Vec<u8> {
    let Some(task) = Task::decode(input) else { return Vec::new() };
    scenes::candles::set_candle_lights(task.candles as usize);
    scenes::cathedral::set_lantern(task.lantern);
    encode_meshes(&bake_shard(&task))
}

/// Runs the scene's builder with the task's shard store; returns the meshes it baked.
pub fn bake_shard(task: &Task) -> Vec<(u32, String, VirtualMesh)> {
    let (_, store) = scenes::build_with(task.kind, task.detail, None, MeshStore::Shard(Shard::new(task.shard, task.shards)));
    match store {
        MeshStore::Shard(shard) => shard.baked,
        _ => unreachable!("the store is handed back unchanged"),
    }
}

/// `[magic, count, (index, name bytes, name…, mesh bytes, mesh…)…]`, little-endian, the names
/// padded to whole words.
pub fn encode_meshes(meshes: &[(u32, String, VirtualMesh)]) -> Vec<u8> {
    let encoded: Vec<Vec<u8>> = meshes.iter().map(|(_, _, mesh)| mesh.to_bytes()).collect();
    let size = 8 + meshes.iter().zip(&encoded).map(|((_, name, _), bytes)| 12 + name.len().next_multiple_of(4) + bytes.len()).sum::<usize>();
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(&OUTPUT_MAGIC.to_le_bytes());
    out.extend_from_slice(&(meshes.len() as u32).to_le_bytes());
    for ((index, name, _), bytes) in meshes.iter().zip(&encoded) {
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.resize(out.len().next_multiple_of(4), 0);
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Inverse of [`encode_meshes`]; `None` for a malformed result.
pub fn decode_meshes(bytes: &[u8]) -> Option<Vec<(u32, String, VirtualMesh)>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, tail) = (self.0.get(..n)?, self.0.get(n..)?);
            self.0 = tail;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
    }
    let mut r = Reader(bytes);
    if r.word()? != OUTPUT_MAGIC {
        return None;
    }
    let count = r.word()?;
    let mut meshes = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let index = r.word()?;
        let length = r.word()? as usize;
        let name = String::from_utf8(r.take(length.next_multiple_of(4))?[..length].to_vec()).ok()?;
        let length = r.word()? as usize;
        let mesh = VirtualMesh::from_bytes(r.take(length)?).ok()?;
        meshes.push((index, name, mesh));
    }
    Some(meshes)
}

/// The scene with empty stand-ins for its meshes (cheap: nothing is baked), for queueing the
/// pipelines its materials need while the meshes bake.
pub fn stand_in(kind: SceneKind, detail: Detail) -> SceneSetup {
    scenes::build_with(kind, detail, None, MeshStore::Shard(Shard::stand_in())).0
}

/// Builds the scene with the meshes the workers baked (the missing ones baked here).
pub fn build_baked(kind: SceneKind, detail: Detail, meshes: Vec<(u32, String, VirtualMesh)>) -> SceneSetup {
    let (setup, store) = scenes::build_with(kind, detail, None, MeshStore::Baked(Baked::new(meshes)));
    if let MeshStore::Baked(baked) = store {
        if baked.baked_here > 0 {
            log::info!("scene bake: {} meshes baked on the page", baked.baked_here);
        }
    }
    setup
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::build;

#[cfg(target_arch = "wasm32")]
mod web {
    use helios_render::{BakeOffload, BakePoll};

    use super::{build_baked, decode_meshes, stand_in, Task};
    use crate::scenes::{self, model::Model, Detail, SceneKind, SceneSetup};

    fn now() -> f64 {
        web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
    }

    async fn sleep(ms: i32) {
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
            }
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }

    /// Builds a scene with its meshes baked in the bake workers, the page's thread free
    /// meanwhile (`idle` runs every few milliseconds of the wait); on the page if there are no
    /// workers (and for an imported model). `prefetch` gets a stand-in of the scene (its meshes
    /// empty) as the bake starts, to queue the pipelines its materials need.
    pub(crate) async fn build(kind: SceneKind, detail: Detail, model: Option<&Model>, prefetch: impl FnOnce(&SceneSetup), mut idle: impl FnMut()) -> SceneSetup {
        let pool = crate::web_bake::pool().filter(|_| kind != SceneKind::Model);
        let Some((pool, workers)) = pool else { return scenes::build(kind, detail, model) };
        let start = now();
        let mut pending: Vec<u32> = (0..workers).filter_map(|shard| pool.submit(&Task::new(kind, detail, shard, workers).encode())).collect();
        let submitted = pending.len();
        if submitted > 0 {
            prefetch(&stand_in(kind, detail));
        }
        let mut meshes = Vec::new();
        let mut failed = 0;
        while !pending.is_empty() {
            idle();
            sleep(8).await;
            pending.retain(|&id| match pool.poll(id) {
                BakePoll::Pending => true,
                BakePoll::Done(bytes) => {
                    match decode_meshes(&bytes) {
                        Some(baked) => meshes.extend(baked),
                        None => failed += 1,
                    }
                    false
                }
                BakePoll::Failed => {
                    failed += 1;
                    false
                }
            });
        }
        let baked = now();
        log::info!(
            "scene bake: {} meshes from {submitted} of {workers} workers in {:.2} s{}",
            meshes.len(),
            (baked - start) / 1000.0,
            if failed > 0 { format!(" ({failed} failed: their meshes bake on the page)") } else { String::new() }
        );
        let setup = build_baked(kind, detail, meshes);
        log::info!("scene bake: built on the page in {:.2} s", (now() - baked) / 1000.0);
        setup
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of(scene: &helios_render::Scene) -> Vec<Vec<u8>> {
        (0..scene.mesh_count() as u32).map(|i| scene.mesh(scene.mesh_handle(i).unwrap()).to_bytes()).collect()
    }

    /// The scene built from the shards' meshes, through the wire format, equals the scene baked
    /// in one piece: the same meshes (bit for bit, by name) and the same instances.
    fn sharded_bake_matches(kind: SceneKind, shards: u32) {
        let direct = scenes::build(kind, Detail::Medium, None);
        let mut meshes = Vec::new();
        for shard in 0..shards {
            let task = Task { kind, detail: Detail::Medium, shard, shards, candles: 0, lantern: false };
            assert_eq!(Task::decode(&task.encode()), Some(task));
            let baked = bake_shard(&task);
            assert!(!baked.is_empty(), "shard {shard} of {shards} baked nothing");
            meshes.extend(decode_meshes(&encode_meshes(&baked)).expect("decodes"));
        }
        let (rebuilt, store) = scenes::build_with(kind, Detail::Medium, None, MeshStore::Baked(Baked::new(meshes)));
        let MeshStore::Baked(baked) = store else { unreachable!() };
        assert_eq!(baked.baked_here, 0, "every mesh came from a shard");
        assert_eq!(rebuilt.scene.instance_count(), direct.scene.instance_count());
        assert_eq!(bytes_of(&rebuilt.scene), bytes_of(&direct.scene));
        for i in 0..direct.scene.mesh_count() as u32 {
            let handle = direct.scene.mesh_handle(i).unwrap();
            assert_eq!(rebuilt.scene.mesh_name(handle), direct.scene.mesh_name(handle));
        }
    }

    #[test]
    fn sharded_gallery_bake_matches_a_direct_bake() {
        sharded_bake_matches(SceneKind::Gallery, 3);
    }

    /// The forest's assemblies: a base is baked from its sprays, so a unit stays on one worker.
    #[test]
    fn sharded_forest_bake_matches_a_direct_bake() {
        sharded_bake_matches(SceneKind::Forest, 2);
    }

    #[test]
    fn malformed_results_and_tasks_are_rejected() {
        assert!(decode_meshes(&[1, 2, 3]).is_none());
        let mut bytes = encode_meshes(&[]);
        assert_eq!(decode_meshes(&bytes).map(|m| m.len()), Some(0));
        bytes[4] = 1;
        assert!(decode_meshes(&bytes).is_none());
        assert!(Task::decode(&[0; 28]).is_none());
        assert!(!is_task(&[0, 0, 0, 0]));
        assert!(is_task(&Task::new(SceneKind::Valley, Detail::Medium, 0, 2).encode()));
    }
}
