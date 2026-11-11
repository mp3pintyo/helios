//! Bakes in Web Workers: the scene's meshes (`scene_bake.rs`) and Lumen's displaced variants.
//!
//! Natively the refinement and the bake of a displaced variant (its distance field and cards)
//! run on worker threads; the browser has no threads without cross-origin isolation, which
//! GitHub Pages cannot provide, so they ran on the render thread, one 35–760 ms task per frame.
//! Here a small pool of Web Workers (`bake_worker.js`) each runs a second instance of this very
//! module — the compiled `WebAssembly.Module` is sent to them, so nothing is downloaded or
//! compiled again — and [`helios_bake_task`] runs the work there with the same code as on the
//! render thread. The render thread only copies the inputs out and the results in.
//!
//! A variant must never stay missing because of a worker: the pool fails a task whose worker
//! errors, does not report its instance ready within seconds or stays silent for 10 s (and
//! terminates that worker), and the engine then runs the task on the render thread itself. A
//! long bake is not silent: the worker's bakes report progress (`helios_geometry::progress`),
//! which the worker passes on to the page as a heartbeat at most once a second.
//!
//! `?bake-workers=0` turns the workers off (the bakes then run on the render thread, as they also
//! do when a worker cannot start).

use std::cell::RefCell;

use helios_render::{BakeOffload, BakePoll};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen(module = "/src/bake_worker.js")]
extern "C" {
    type BakePool;

    #[wasm_bindgen(js_name = bakePoolCreate)]
    fn bake_pool_create(module: JsValue, size: u32) -> Option<BakePool>;

    #[wasm_bindgen(method)]
    fn submit(this: &BakePool, input: &[u8]) -> u32;

    #[wasm_bindgen(method)]
    fn poll(this: &BakePool, id: u32) -> JsValue;

    #[wasm_bindgen(method)]
    fn cancel(this: &BakePool, id: u32);

    /// In a worker: tells the page that the running task is alive (rate-limited there).
    #[wasm_bindgen(js_name = bakeBeat)]
    fn bake_beat();
}

pub(crate) struct WorkerPool(BakePool);

impl Clone for WorkerPool {
    fn clone(&self) -> Self {
        let pool: &JsValue = self.0.as_ref();
        Self(pool.clone().unchecked_into())
    }
}

impl BakeOffload for WorkerPool {
    fn submit(&self, input: &[u8]) -> Option<u32> {
        Some(self.0.submit(input)).filter(|&id| id != 0)
    }

    fn poll(&self, id: u32) -> BakePoll {
        let result = self.0.poll(id);
        if result.is_undefined() {
            BakePoll::Pending
        } else if let Some(bytes) = result.dyn_ref::<js_sys::Uint8Array>() {
            BakePoll::Done(bytes.to_vec())
        } else {
            BakePoll::Failed
        }
    }

    fn cancel(&self, id: u32) {
        self.0.cancel(id);
    }
}

/// Workers in the pool: a gallery's or a valley's variants bake side by side, leaving a core to
/// the page and the GPU process.
fn pool_size() -> u32 {
    let cores = web_sys::window().map(|w| w.navigator().hardware_concurrency()).unwrap_or(2.0) as u32;
    cores.saturating_sub(2).clamp(1, 4)
}

thread_local! {
    /// The page's pool and its size, for the scene bake.
    static POOL: RefCell<Option<(WorkerPool, u32)>> = const { RefCell::new(None) };
}

/// Installs the worker pool (unless `enabled` is false); the workers start with the first bake.
pub(crate) fn install(enabled: bool) {
    let size = pool_size();
    let pool = if enabled { bake_pool_create(wasm_bindgen::module(), size).map(WorkerPool) } else { None };
    if pool.is_none() {
        log::info!("the scene and Lumen's displaced variants bake on the page (no bake workers)");
    }
    POOL.with(|p| *p.borrow_mut() = pool.clone().map(|pool| (pool, size)));
    helios_render::set_bake_offload(pool.map(|p| Box::new(p) as Box<dyn BakeOffload>));
}

/// The bake workers and their number (`None` without workers).
pub(crate) fn pool() -> Option<(WorkerPool, u32)> {
    POOL.with(|p| p.borrow().clone())
}

/// A bake worker's entry (`bake_worker.js`): runs one task, a share of the scene bake or one of
/// Lumen's displaced variants.
#[wasm_bindgen(js_name = heliosBakeTask)]
pub fn helios_bake_task(input: &[u8]) -> Vec<u8> {
    helios_geometry::progress::set_hook(Some(bake_beat));
    if crate::scene_bake::is_task(input) {
        crate::scene_bake::run_task(input)
    } else {
        helios_render::run_bake_task(input)
    }
}
