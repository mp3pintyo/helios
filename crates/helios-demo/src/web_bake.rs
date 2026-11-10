//! Lumen's displaced-variant bakes in Web Workers.
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
//! errors, does not report its instance ready within seconds or does not answer within 10 s (and
//! terminates that worker), and the engine then runs the task on the render thread itself.
//!
//! `?bake-workers=0` turns the workers off (the bakes then run on the render thread, as they also
//! do when a worker cannot start).

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
}

struct WorkerPool(BakePool);

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

/// Installs the worker pool (unless `enabled` is false); the workers start with the first bake.
pub(crate) fn install(enabled: bool) {
    let pool = if enabled { bake_pool_create(wasm_bindgen::module(), pool_size()) } else { None };
    if pool.is_none() {
        log::info!("lumen: displaced variants bake on the page (no bake workers)");
    }
    helios_render::set_bake_offload(pool.map(|p| Box::new(WorkerPool(p)) as Box<dyn BakeOffload>));
}

/// A bake worker's entry (`bake_worker.js`): runs one task of Lumen's displaced variants.
#[wasm_bindgen(js_name = heliosBakeTask)]
pub fn helios_bake_task(input: &[u8]) -> Vec<u8> {
    helios_render::run_bake_task(input)
}
