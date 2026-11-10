// Web Workers for Lumen's displaced-variant bakes (web_bake.rs; helios-render's lumen_offload.rs).
//
// This one file is both ends. Imported by the engine's glue on the page (wasm-bindgen copies it to
// pkg/snippets/), it exports the pool the engine submits work to. Started as a module worker, it
// imports the same glue, instantiates the engine's already-compiled WebAssembly.Module (sent by
// the page, so nothing is fetched or compiled again) and runs the work it is sent with
// heliosBakeTask. Workers that share no memory need no SharedArrayBuffer, so this works without
// cross-origin isolation (GitHub Pages cannot send COOP/COEP headers).

const IN_WORKER = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;

// pkg/snippets/helios-demo-<hash>/src/bake_worker.js -> pkg/helios_demo.js
const GLUE = new URL("../../../helios_demo.js", import.meta.url).href;
// A worker without tasks for this long is retired.
const IDLE_MS = 20000;

class BakePool {
  constructor(module, size) {
    this.module = module;
    this.size = size;
    this.workers = [];
    // Task id -> undefined (running), a Uint8Array (done) or null (failed).
    this.results = new Map();
    this.next = 1;
  }

  // Records the outcome of a task the engine still waits for.
  settle(id, result) {
    if (this.results.has(id)) this.results.set(id, result);
  }

  // A module worker from this file's URL; a page whose origin differs from the script's (a
  // sandboxed frame) may only start one through a blob that imports it.
  spawn() {
    const url = import.meta.url;
    let worker;
    try {
      worker = new Worker(url, { type: "module", name: "helios-bake" });
    } catch {
      const blob = URL.createObjectURL(new Blob([`import ${JSON.stringify(url)};`], { type: "text/javascript" }));
      worker = new Worker(blob, { type: "module", name: "helios-bake" });
    }
    const entry = { worker, tasks: new Set(), dead: false, idle: 0 };
    worker.onmessage = ({ data }) => {
      entry.tasks.delete(data.id);
      this.settle(data.id, data.output ? new Uint8Array(data.output) : null);
      if (data.error) console.warn(`helios: bake worker task failed (${data.error}); running it on the page`);
      // A wasm instance keeps the memory its largest bake grew it to: a worker idle for a while
      // is retired (a later bake starts a new one, from the compiled module, in milliseconds).
      if (entry.tasks.size === 0) {
        entry.idle = setTimeout(() => {
          worker.terminate();
          this.workers = this.workers.filter((w) => w !== entry);
        }, IDLE_MS);
      }
    };
    // A worker that cannot start (or dies) fails its tasks, which the engine then runs itself.
    worker.onerror = (e) => {
      e.preventDefault?.();
      console.warn(`helios: bake worker failed (${e.message || "error"}); bakes run on the page`);
      entry.dead = true;
      for (const id of entry.tasks) this.settle(id, null);
      entry.tasks.clear();
      worker.terminate();
    };
    worker.postMessage({ module: this.module });
    this.workers.push(entry);
    return entry;
  }

  // Starts a task with a copy of `input` (a view of the engine's memory); 0 if no worker can.
  submit(input) {
    let entry = null;
    const live = this.workers.filter((w) => !w.dead);
    const idle = live.find((w) => w.tasks.size === 0);
    if (idle) {
      entry = idle;
    } else if (this.workers.length < this.size) {
      try {
        entry = this.spawn();
      } catch (e) {
        console.warn(`helios: no bake worker (${e}); bakes run on the page`);
        this.size = this.workers.length;
      }
    }
    entry ??= live.reduce((a, b) => (b.tasks.size < a.tasks.size ? b : a), live[0] ?? null);
    if (!entry) return 0;
    const id = this.next++;
    const buffer = input.slice().buffer;
    clearTimeout(entry.idle);
    entry.tasks.add(id);
    this.results.set(id, undefined);
    entry.worker.postMessage({ id, input: buffer }, [buffer]);
    return id;
  }

  // undefined while task `id` runs, then once its result (Uint8Array), or null if it failed.
  poll(id) {
    if (!this.results.has(id)) return null;
    const result = this.results.get(id);
    if (result !== undefined) this.results.delete(id);
    return result;
  }

  // The engine no longer waits for task `id` (it left the scene): its result is not kept.
  cancel(id) {
    this.results.delete(id);
  }
}

// The page's pool: up to `size` workers, started on the first task (a scene without displacement
// starts none); null where workers cannot exist.
export function bakePoolCreate(module, size) {
  if (IN_WORKER || typeof Worker === "undefined" || !(module instanceof WebAssembly.Module)) return null;
  return new BakePool(module, size);
}

if (IN_WORKER) {
  let engine = null;
  self.onmessage = async ({ data }) => {
    if (data.module) {
      engine = import(GLUE).then(async (m) => {
        await m.default({ module_or_path: data.module });
        return m;
      });
      return;
    }
    try {
      const m = await engine;
      const output = m.heliosBakeTask(new Uint8Array(data.input));
      if (output.length === 0) throw new Error("the task did not decode");
      self.postMessage({ id: data.id, output: output.buffer }, [output.buffer]);
    } catch (e) {
      self.postMessage({ id: data.id, error: String(e && e.message ? e.message : e) });
    }
  };
}
