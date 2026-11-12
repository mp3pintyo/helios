// helios-boot.js: starts the Helios engine (Rust + WebGPU, compiled to wasm) in a web page and
// waits until it shows an image. It also holds small helpers for the editor API.
//
// The original lives in D:\AI\Unreal\ittkezdjed\. `sync.py` copies it into a site next to the
// engine's pkg/ folder. Change the original, not the copy: the next sync overwrites the copy.
//
//   import { startHelios } from "./helios/helios-boot.js";
//   const H = await startHelios({ config: { scene: "gallery" }, onProgress: (p) => (veil.textContent = p.text) });
//   H.heliosLookAt(0, 1.7, 8, 0, 1, 0);
//
// It works with any Helios build. Builds older than 2026-11-12 have no heliosReady / heliosError;
// for those it falls back to heliosScene and the text of the status element.

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** An error from startHelios. `code`: "no-webgpu", "file-url", "canvas", "twice", "engine" or "timeout". */
export class HeliosError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "HeliosError";
    this.code = code;
  }
}

let started = false;

/**
 * Loads the engine module, starts it on the canvas and resolves to the module's exports
 * (`heliosLookAt`, `heliosEditorApply`, …) once the wait chosen by `waitFor` is over.
 * The page can have only one engine (the wasm module is a singleton).
 *
 * options:
 *   module       URL of helios_demo.js, relative to this file (default "./pkg/helios_demo.js")
 *   canvas       the <canvas> element or its id (default "helios-canvas"). The engine draws only into
 *                the canvas whose id is helios-canvas, and needs it to have a CSS size
 *   config       the fields of globalThis.heliosConfig: scene, gltf, sceneFile, stream, gpuBudgetMs,
 *                screenPercentage, gi, megaLights, lights, worldCache, bakeWorkers, … (see API.md).
 *                `embedded` defaults to true here: keys go only to the focused canvas, and the
 *                mouse wheel scrolls the page
 *   waitFor      "ready": the scene and a scene file's edits are in (camera / editor calls are safe)
 *                "first-image" (default): the first frame is on the canvas
 *                "full-quality": every feature's shaders are compiled too, or fullQualityMs passed
 *   fullQualityMs  how long "full-quality" waits after the first image at most (default 45000)
 *   timeoutMs    give up (HeliosError "timeout") after this long (default 120000)
 *   onProgress   called about every 100 ms with { stage, text, compiled, total, waiting, ms }.
 *                stage is "loading", "compiling", "refining" or "done", and text is the engine's
 *                English progress line
 *   pauseWhenHidden  stop rendering while the tab is hidden or the canvas is off screen (default true)
 */
export async function startHelios(options = {}) {
  const {
    module = "./pkg/helios_demo.js",
    canvas = "helios-canvas",
    config = {},
    waitFor = "first-image",
    fullQualityMs = 45000,
    timeoutMs = 120000,
    onProgress = () => {},
    pauseWhenHidden = true,
  } = options;

  if (location.protocol === "file:") {
    throw new HeliosError("file-url", "Helios needs the page served over HTTP(S) (python -m http.server), not opened as a file.");
  }
  if (!navigator.gpu) {
    throw new HeliosError("no-webgpu", "This browser does not expose WebGPU. Use Chrome/Edge 113+, Safari 26+ or Firefox 141+.");
  }
  if (started) throw new HeliosError("twice", "Helios is already running on this page (one engine per page).");
  const element = typeof canvas === "string" ? document.getElementById(canvas) : canvas;
  if (!(element instanceof HTMLCanvasElement)) throw new HeliosError("canvas", `No <canvas> "${canvas}" on the page.`);
  // The engine looks the canvas up by this id. Renaming another canvas would break the page's
  // #id styles, so refuse instead.
  if (element.id !== "helios-canvas") {
    throw new HeliosError("canvas", 'The engine draws into <canvas id="helios-canvas">: give the canvas that id (style it with #helios-canvas or a class).');
  }
  if (!element.clientWidth || !element.clientHeight) {
    console.warn("Helios: #helios-canvas has no CSS size yet; the engine renders at the canvas's CSS size (give it width/height).");
  }
  if (!element.hasAttribute("tabindex")) element.tabIndex = 0;
  started = true;

  // The engine writes progress and errors into this element. It stays hidden: the page shows
  // its own (translated) text from onProgress.
  const statusId = config.statusElement || "helios-status";
  let status = document.getElementById(statusId);
  if (!status) {
    status = document.createElement("span");
    status.id = statusId;
    status.hidden = true;
    document.body.append(status);
  }
  globalThis.heliosConfig = { embedded: true, ...config, statusElement: statusId };

  const H = await import(new URL(module, import.meta.url).href);
  await H.default();

  const t0 = performance.now();
  let firstImageAt = 0;
  for (;;) {
    const ms = performance.now() - t0;
    const error = H.heliosError ? H.heliosError() : "";
    if (error) throw new HeliosError("engine", error);
    const ready = H.heliosReady ? H.heliosReady() : !!H.heliosScene();
    const frames = ready ? H.heliosFrames() : 0;
    if (frames > 0 && !firstImageAt) firstImageAt = ms;
    let shaders = { compiled: 0, total: 0, waiting: [] };
    try { shaders = JSON.parse(H.heliosShaders()); } catch { /* not running yet */ }
    const busy = H.heliosBusy();
    const stage = !ready ? "loading" : frames === 0 ? "compiling" : shaders.waiting.length || busy ? "refining" : "done";
    const done =
      (waitFor === "ready" && ready) ||
      (waitFor === "first-image" && frames > 0) ||
      (waitFor === "full-quality" && frames > 0 && (stage === "done" || ms - firstImageAt > fullQualityMs));
    onProgress({ stage: done ? "done" : stage, text: busy || status.textContent || "", compiled: shaders.compiled, total: shaders.total, waiting: shaders.waiting, ms });
    if (done) break;
    if (!H.heliosError && !ready && /^(Could not|Failed|missing|.*error)/i.test(status.textContent || "")) {
      throw new HeliosError("engine", status.textContent);
    }
    if (ms > timeoutMs) throw new HeliosError("timeout", `Helios did not start within ${Math.round(timeoutMs / 1000)} s: ${status.textContent || busy}`);
    await sleep(100);
  }

  if (pauseWhenHidden) {
    let onScreen = true;
    const update = () => H.heliosSetPaused(document.hidden || !onScreen);
    document.addEventListener("visibilitychange", update);
    new IntersectionObserver((entries) => {
      onScreen = entries[entries.length - 1].isIntersecting;
      update();
    }).observe(element);
  }
  return H;
}

/**
 * Applies one editor operation (an object, see API.md "Editor operations") and returns what it
 * created ({ actors, instances, lights, pcg }). Throws on an error, which the raw
 * heliosEditorApply only returns as JSON.
 */
export function apply(H, op) {
  const result = JSON.parse(H.heliosEditorApply(JSON.stringify(op)));
  if (result.error) throw new Error(`Helios ${op.op}: ${result.error}`);
  return result.created;
}

/**
 * The editable scene (heliosEditorState). The actor columns { id: [], name: [], parent: [],
 * flags: [] } are also returned as `actorList`: [{ id, name, parent }] (parent -1: top level).
 */
export function sceneState(H) {
  const state = JSON.parse(H.heliosEditorState());
  const a = state.actors;
  state.actorList = a.id.map((id, i) => ({ id, name: a.name[i], parent: a.parent[i], flags: a.flags[i] }));
  return state;
}

/** A rotation quaternion [x, y, z, w] about the vertical axis, in degrees (for `transform.r`). */
export function yaw(degrees) {
  const half = (degrees * Math.PI) / 360;
  return [0, Math.sin(half), 0, Math.cos(half)];
}
