# Helios changes that matter to a web site

Newest first. `sync.py` prints the entries newer than a site's last sync. Each entry says what
changed and what a site has to do.

## 2026-11-13: fewer shader compiles per visit

* **Firefox: the full image arrives sooner.**
  * Firefox compiles every shader on every visit. The engine now compiles only what the scene
    uses.
  * Pipelines left out: the lighting variant a scene does not use, the tessellator when no
    displaced surface is shown, the culling variant for motion the scene does not have, and the
    debug views.
  * The Opus55 site: full quality after 32.6 s instead of 49.1 s, first image 18.4 s instead of
    20.9 s. The demo scenes are 2–12 % faster.
  * The image is unchanged.
  * **Site:** sync.
* **A page that hides content at start** (the Opus55 site hides the gallery but its floor) now
  skips the tessellation of what it hides. The material shaders of hidden objects still compile
  (they are queued before the page's edits arrive).
* `?pipeline-usage=1` logs, 10 s after the last compile, the shaders no frame has used (for
  finding waste).

## 2026-11-12 (late night): Firefox holds the image still while it compiles

* **Firefox: no more jumping view after the first image.**
  * Until now the view moved about once a second for ~25 s while the remaining shaders
    compiled.
  * Now the canvas holds its image and moves only for a moment as each feature (shadows, GI,
    fog…) arrives.
  * A 2 px bar at the canvas's bottom edge shows the progress and fades out at the end.
  * Firefox cannot draw while it compiles, so this is the remaining choice; it adds ~2.6 s to the
    full set.
  * New config keys: `compileView` (`"still"`, the Firefox default, or `"live"`, the old
    behaviour) and `compileIndicator` (`false` hides the bar).
  * `heliosShaders()` gains `still`.
  * **Site:** sync. A site that pauses the engine once settled still pauses only when
    `heliosBusy() === ""` (unchanged); a paused engine stops compiling.

## 2026-11-12 (night): lower levels start faster

* **The features a quality level switches off are no longer compiled** until switched on.
  * Firefox at `low`: the first image after 18 s instead of 29 s, and the stutter after it
    ~14 s instead of ~25 s. `high` is unchanged.
  * Switching up a level later (`heliosSetQuality`, or the engine stepping up) compiles them
    then: in Firefox ~20 s of stutter, in Chromium a few seconds.
  * **Site:** sync. A page with a quality menu: expect that delay after a switch up.
  * `heliosShaders()` gains `held`, the features waiting to be switched on.

## 2026-11-12 (evening): Firefox at 60 fps on fast GPUs

* **Fix: Firefox ran at ~27 fps even on an RTX 3090/4090.** Firefox starts a frame only once
  the last one is on screen, so the default 14 ms GPU budget left it at half the refresh rate
  with the GPU 40 % busy. In Firefox the dynamic resolution's budget is now at most 8 ms (also
  when the page sets a higher `gpuBudgetMs`): 60 fps at ~60 % resolution, which TSR brings back
  to full. **Site:** sync.

## 2026-11-12 (later): quality levels for weak GPUs

* **New: the engine lowers its quality on integrated GPUs** (`heliosConfig.quality`, URL
  `?quality=`, default `"auto"`; `heliosQuality()`, `heliosSetQuality(level)`).
  * Visitors reported stutter. On an integrated GPU the full renderer ran at 2–6 fps.
  * `auto` starts integrated and mobile GPUs at `low`, Apple silicon at `medium`, the rest at
    `high`, and moves one level a step when the frame times say so.
  * **Site:** sync. Look at each section with `?quality=low`: the volumetric clouds and fog,
    coloured glass light and local lights' shadow maps are off there. A page that must look the
    same everywhere sets `quality: "high"` (and accepts the stutter).
* **Dynamic resolution reacts faster on slow GPUs** (it reached 50 % only after half a minute at
  2 fps). Nothing to do.

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
  * `heliosSetPaused(true)` also stops the compiling, because it advances in the frame loop.
    **Site:** a page that pauses a still view on its own timer must wait until
    `heliosFrames() > 0` and `heliosBusy() === ""`. Otherwise the canvas stays empty: a site that
    paused 1.8 s after `heliosScene()` stuck at "Compiling GPU shaders… (46 / 50)".
* **Streamed geometry (`?stream=`) needs `.hvm` version 7 files.** **Site:** re-preprocess them
  with the engine of the same version.
* **New: `heliosPhoto(samples)` / `heliosPhotoProgress()`.** They accumulate a still with
  film-quality depth of field.
