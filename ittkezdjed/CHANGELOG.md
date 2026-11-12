# Helios changes that matter to a web site

Newest first. `sync.py` prints the entries newer than a site's last sync. Each entry says what
changed and what a site has to do.

## 2026-11-12

* **New: `heliosConfig.gltf`, `heliosConfig.sceneFile`, `heliosConfig.stream`.**
  * Until now these were URL parameters only (`?gltf=`, `?scene-file=`, `?stream=`), and sites
    rewrote the page URL with `history.replaceState` before loading the engine.
  * **Site:** drop that workaround and pass them in the config.
* **New: `heliosReady()`.** It is true once the scene, including a scene file's edits, is in.
  * Until now sites polled `heliosScene()`. That is non-empty before a scene file's edits are applied.
* **New: `heliosError()`.** It returns the start-up error or the lost-device message; it is empty while all is well.
  * Until now sites guessed errors from the status element's text.
* **New: this folder.** It holds `helios-boot.js` (`startHelios` replaces the hand-written start-up
  loops), `sync.py`, `API.md`, `catalog/` and `template/`.
* **Fix: Firefox no longer loses the GPU device after about 800 frames** ("Not enough memory left"; it
  froze after about 5–15 s). The engine now reuses its bind groups, and Firefox gets an `rgba8unorm`
  canvas. **Site:** sync. Any site built before this date freezes in Firefox.

## 2026-11-11 (round 8)

* **Scene preprocessing runs in Web Workers.** `pkg/snippets/…/bake_worker.js` and `gpu_memory.js`
  ship with the engine. **Site:** copy the whole `pkg/` including `snippets/`, or the module fails to
  load. Pages without workers can set `bakeWorkers: "0"`.
* **The editor has actors with a hierarchy.**
  * `heliosEditorState().actors` is now columnar: `{ id: [], name: [], parent: [], flags: [] }`
    (`parent` −1: top level).
  * New operations: `actor_transform`, `actor_local`, `actor_visibility`, `rename`, `reparent`,
    `group`, `ungroup`, `component_*`.
  * The older instance- and light-addressed operations still work.
  * Scene files are version 2, and version 1 files are migrated on load.
  * **Site:** code that read actors as a list of objects must read the columns (or use
    `sceneState(H).actorList`). To hide a built-in scene's objects, use `actor_visibility`.
* **Shaders compile progressively.**
  * The first image comes before every feature's shaders are compiled.
  * New: `heliosBusy()`, `heliosFrames()`, `heliosShaders()`.
  * **Site:** keep a loading veil until `heliosFrames() > 0` (or use `startHelios`).
* **Streamed geometry (`?stream=`) needs `.hvm` version 7 files.** **Site:** re-preprocess them
  with the engine of the same version.
* **New: `heliosPhoto(samples)` / `heliosPhotoProgress()`.** They accumulate a still with
  film-quality depth of field.
