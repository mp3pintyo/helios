// Web Workers for the scene bake (scene_bake.rs) and Lumen's displaced-variant bakes (web_bake.rs;
// helios-render's lumen_offload.rs).
//
// This one file is both ends. Imported by the engine's glue on the page (wasm-bindgen copies it to
// pkg/snippets/), it exports the pool the engine submits work to. Started as a module worker, it
// imports the same glue, instantiates the engine's already-compiled WebAssembly.Module (sent by
// the page, so nothing is fetched or compiled again) and runs the work it is sent with
// heliosBakeTask. Workers that share no memory need no SharedArrayBuffer, so this works without
// cross-origin isolation (GitHub Pages cannot send COOP/COEP headers).
//
// Nothing a worker does may leave a variant missing: a task the page gets no answer for is failed,
// and the engine then runs it on the render thread itself (lumen_displaced.rs). Hence the
// handshake and the watchdog below — a worker reports when its instance is ready, it is only
// handed a task while idle, and one that is not ready in time or falls silent is terminated. A
// running bake reports progress (bakeBeat, at most once a second), so a long one is not mistaken
// for a hung one.

const IN_WORKER = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;

// pkg/snippets/helios-demo-<hash>/src/bake_worker.js -> pkg/helios_demo.js
const GLUE = new URL("../../../helios_demo.js", import.meta.url).href;
// A worker without tasks for this long is retired.
const IDLE_MS = 20000;
// A worker must report its instance ready within this (it takes milliseconds: the module is
// compiled already), and while it runs a task give a sign of life (its heartbeat, or the answer)
// within this (a bake beats several times a second). Past either it is terminated and its task
// runs on the page.
const READY_MS = 8000;
const TASK_MS = 10000;
// A worker's heartbeat while it runs a task.
const BEAT_MS = 1000;
// When a watchdog timer fires this late, the page's thread was blocked and a worker's answer may
// be queued behind the timer: it looks again a little later, after the queue has run.
const LATE_MS = 250;
const RECHECK_MS = 100;
// After this many workers failed (did not start, errored, timed out), none is started again and
// every task runs on the page.
const MAX_FAILURES = 3;

class BakePool {
  constructor(module, size) {
    this.module = module;
    this.size = size;
    // { worker, ready, task (id or 0), timer, idle, deadline }
    this.workers = [];
    // Task ids waiting for an idle, ready worker, and their inputs.
    this.queue = [];
    this.inputs = new Map();
    // Task id -> undefined (running), a Uint8Array (done) or null (failed: the page runs it).
    this.results = new Map();
    this.next = 1;
    this.failures = 0;
  }

  // Records the outcome of a task the engine still waits for.
  settle(id, result) {
    this.inputs.delete(id);
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
    const entry = { worker, ready: false, task: 0, timer: 0, idle: 0, deadline: 0 };
    worker.onmessage = ({ data }) => this.message(entry, data);
    // A worker that cannot load its script (or dies) fails its task, which the engine then runs.
    worker.onerror = (e) => {
      e.preventDefault?.();
      this.fail(entry, `failed (${e.message || "error"})`);
    };
    worker.onmessageerror = () => this.fail(entry, "sent a message that could not be read");
    // A message that reaches a module worker before its script has set the handler is dropped
    // (Chromium does so as soon as the script awaits anything on its way there), so the worker asks
    // for the module once it can receive it ("hello") instead of the page posting it now.
    this.arm(entry, READY_MS, "did not start");
    this.workers.push(entry);
    return entry;
  }

  // (Re)starts `entry`'s watchdog: past `ms` it fails with `reason`.
  arm(entry, ms, reason) {
    clearTimeout(entry.timer);
    entry.deadline = performance.now() + ms;
    const check = () => {
      if (!this.workers.includes(entry)) return;
      // Fired late: the page was busy, so let the answers queued meanwhile arrive first.
      if (performance.now() - entry.deadline > LATE_MS) {
        entry.deadline = performance.now() + RECHECK_MS;
        entry.timer = setTimeout(check, RECHECK_MS);
        return;
      }
      this.fail(entry, reason);
    };
    entry.timer = setTimeout(check, ms);
  }

  message(entry, data) {
    if (!this.workers.includes(entry)) return;
    if (data.hello) {
      entry.worker.postMessage({ module: this.module });
      return;
    }
    if (data.ready !== undefined) {
      if (!data.ready) return this.fail(entry, `could not instantiate the engine (${data.error})`);
      clearTimeout(entry.timer);
      entry.ready = true;
      this.dispatch();
      return;
    }
    if (data.beat !== undefined) {
      // Still working: the watchdog starts over.
      if (data.beat === entry.task) this.arm(entry, TASK_MS, "fell silent");
      return;
    }
    if (data.id !== entry.task) return;
    // A task that trapped may have left the instance broken: the worker is replaced.
    if (data.error) return this.fail(entry, `could not run a bake (${data.error})`);
    clearTimeout(entry.timer);
    entry.task = 0;
    this.settle(data.id, new Uint8Array(data.output));
    this.dispatch();
  }

  // Terminates a worker that failed; its task is failed, so the engine runs it on the page (and so
  // are the queued ones once the pool gives up).
  fail(entry, reason) {
    if (!this.workers.includes(entry)) return;
    console.warn(`helios: bake worker ${reason}; its bake runs on the page`);
    clearTimeout(entry.timer);
    clearTimeout(entry.idle);
    entry.worker.terminate();
    this.workers = this.workers.filter((w) => w !== entry);
    if (entry.task) this.settle(entry.task, null);
    if (++this.failures >= MAX_FAILURES && this.size > 0) {
      console.warn("helios: bake workers keep failing; displaced variants bake on the page");
      this.size = 0;
      // Workers still starting would most likely fail the same way: not waited for.
      for (const w of this.workers.filter((w) => !w.ready)) {
        clearTimeout(w.timer);
        w.worker.terminate();
      }
      this.workers = this.workers.filter((w) => w.ready);
    }
    this.dispatch();
  }

  // Hands queued tasks to idle, ready workers, starting workers up to the pool's size.
  dispatch() {
    // Drop the tasks the engine no longer waits for.
    this.queue = this.queue.filter((id) => this.results.has(id));
    for (const entry of this.workers) {
      if (!entry.ready || entry.task || this.queue.length === 0) continue;
      const id = this.queue.shift();
      const buffer = this.inputs.get(id);
      this.inputs.delete(id);
      clearTimeout(entry.idle);
      entry.idle = 0;
      entry.task = id;
      this.arm(entry, TASK_MS, "fell silent");
      entry.worker.postMessage({ id, input: buffer }, [buffer]);
    }
    const starting = this.workers.filter((w) => !w.ready).length;
    let wanted = Math.min(this.queue.length - starting, this.size - this.workers.length);
    for (; wanted > 0; wanted--) {
      try {
        this.spawn();
      } catch (e) {
        console.warn(`helios: no bake worker (${e}); bakes run on the page`);
        this.size = 0;
        break;
      }
    }
    // Given up on workers (or none could start): the page runs what is queued (tasks already
    // running may still answer).
    if (this.size === 0) {
      for (const id of this.queue) this.settle(id, null);
      this.queue = [];
    }
    // A worker without work is retired after a while: its wasm instance keeps the memory its
    // largest bake grew it to (a later bake starts a new one, from the compiled module, quickly).
    for (const entry of this.workers) {
      if (entry.ready && !entry.task && !entry.idle) {
        entry.idle = setTimeout(() => {
          entry.idle = 0;
          if (entry.task || !this.workers.includes(entry)) return;
          clearTimeout(entry.timer);
          entry.worker.terminate();
          this.workers = this.workers.filter((w) => w !== entry);
        }, IDLE_MS);
      }
    }
  }

  // Starts a task with a copy of `input` (a view of the engine's memory); 0 if no worker can.
  submit(input) {
    if (this.size === 0) return 0;
    const id = this.next++;
    this.results.set(id, undefined);
    this.inputs.set(id, input.slice().buffer);
    this.queue.push(id);
    this.dispatch();
    // Failed at once (no worker could start): the engine runs it itself.
    if (this.results.get(id) === null) {
      this.results.delete(id);
      return 0;
    }
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
    this.inputs.delete(id);
  }
}

// The page's pool: up to `size` workers, started on the first task; null where workers cannot
// exist.
export function bakePoolCreate(module, size) {
  if (IN_WORKER || typeof Worker === "undefined" || !(module instanceof WebAssembly.Module)) return null;
  return new BakePool(module, size);
}

// The task this worker runs (0: none) and when it last said so.
let running = 0;
let lastBeat = 0;

// Called by the engine's bakes (helios_geometry::progress) while a task runs in this worker.
export function bakeBeat() {
  if (!IN_WORKER || !running) return;
  const now = performance.now();
  if (now - lastBeat < BEAT_MS) return;
  lastBeat = now;
  self.postMessage({ beat: running });
}

if (IN_WORKER) {
  let engine = null;
  self.onmessage = async ({ data }) => {
    if (data.module) {
      try {
        engine = await import(GLUE);
        await engine.default({ module_or_path: data.module });
        self.postMessage({ ready: true });
      } catch (e) {
        self.postMessage({ ready: false, error: String(e && e.message ? e.message : e) });
      }
      return;
    }
    try {
      running = data.id;
      lastBeat = performance.now();
      const output = engine.heliosBakeTask(new Uint8Array(data.input));
      if (output.length === 0) throw new Error("the task did not decode");
      self.postMessage({ id: data.id, output: output.buffer }, [output.buffer]);
    } catch (e) {
      self.postMessage({ id: data.id, error: String(e && e.message ? e.message : e) });
    } finally {
      running = 0;
    }
  };
  // The handler exists now: the page may send the module.
  self.postMessage({ hello: true });
}
