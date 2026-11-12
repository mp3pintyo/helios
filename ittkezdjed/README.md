# Start here: building a web page on the Helios engine

This folder is for anyone (human or AI agent) who builds a web site with Helios as its 3D engine.
Read this file first. It covers what Helios is, how to put it into a page in a few minutes, how to
make your own scene, and how to update a site when the engine changes.

| File | What it is |
|---|---|
| `README.md` | this guide |
| [`API.md`](API.md) | **reference**, generated from the engine source: start options, every `helios*` function, editor operations, scene file format |
| [`CHANGELOG.md`](CHANGELOG.md) | engine changes that matter to a site, newest first |
| [`catalog/`](catalog/) | what each built-in scene contains: material, mesh and actor names, environment, render settings, content items |
| [`helios-boot.js`](helios-boot.js) | start-up helper: `startHelios()`, `apply()`, `sceneState()`, `yaw()` |
| [`sync.py`](sync.py) | copies the engine into a site and reports what changed since the last copy |
| [`template/`](template/) | a working starter page (full-window 3D background, loading veil, chapters that move the camera) |
| `tools/catalog.py` | regenerates `catalog/` |

Don't read the repository's `README.md` whole (40 KB). Grep it for a heading if you need more depth.
Details per subsystem are in `docs/architecture/*.md`.

## What Helios is

Helios is a real-time renderer in Rust on wgpu, modelled on Unreal Engine 5:
- Nanite-style virtual geometry
- Lumen-style global illumination
- virtual shadow maps, MegaLights
- TSR
- sky, clouds and fog
- Substrate-style materials and material graphs
- foliage with wind, water
- a level editor with actors, PCG volumes and scene files

In the browser it runs on **WebGPU** as a wasm module of about 7 MiB, which you embed in your page.

* **Browsers:** Chrome / Edge 113+, Safari 26+, Firefox 141+. The page must be served over
  `http(s)://`; a `file://` page cannot load the engine. WebGPU works on `localhost` and on HTTPS.
* **Built-in scenes:**

  | Scene | What it is |
  |---|---|
  | `valley` | landscape, lake, river, cliff with a cave, stone path |
  | `cathedral` | Gothic interior: candles, chandeliers, stained glass, incense |
  | `forest` | trees, ferns, a campfire with smoke |
  | `gallery` | courtyard of material specimens on plinths |

  The engine can also show your own **glTF** model (`.glb`); skinned animations play.
* **Start-up:**
  * The first image takes about 3–10 s: the scene is preprocessed in Web Workers while the
    browser compiles the shaders.
  * Shadows, GI and the other features join over the next seconds as their shaders arrive.
  * In Firefox, full quality takes about 45 s.
* **Units and axes:** Y is up, and lengths are in metres.

## The fast path (5 minutes)

1. Copy the engine into the site. This creates `MySite/helios/pkg/`, `helios-boot.js` and `helios-engine.json`:

   ```powershell
   python D:/AI/Unreal/ittkezdjed/sync.py D:/AI/MySite/helios --build
   ```

   `--build` rebuilds the wasm first if the engine's sources are newer than it. For that you need
   `rustup target add wasm32-unknown-unknown` and `cargo install wasm-bindgen-cli --version 0.2.130`.
2. Start from `template/`: copy `index.html` and `main.js` into the site. Or embed it yourself:

   ```html
   <canvas id="helios-canvas" tabindex="0" style="position:fixed; inset:0; width:100vw; height:100vh"></canvas>
   <script type="module">
     import { startHelios } from "./helios/helios-boot.js";
     const H = await startHelios({
       config: { scene: "gallery" },                     // or { gltf: "models/x.glb" } / { sceneFile: "x.helios.json" }
       onProgress: (p) => (veil.textContent = p.stage),  // "loading" → "compiling" → "done"
     });
     H.heliosLookAt(0, 1.7, 9, 0, 1, 0);                 // eye x, height above ground, z → target x, height, z
     H.heliosSetSun(200, 25);                            // azimuth, elevation in degrees (negative: night)
   </script>
   ```
3. Serve the site and open it: `python -m http.server 8080` → `http://localhost:8080/`.

`startHelios` does the following:
* checks for WebGPU and for a `file://` page;
* sets `globalThis.heliosConfig`, with `embedded: true` by default;
* loads `pkg/helios_demo.js` and waits until the engine is ready;
* reports progress;
* pauses rendering while the tab is hidden or the canvas is off screen;
* throws a `HeliosError` whose `code` is `no-webgpu`, `file-url`, `canvas`, `twice`, `engine` or
  `timeout`. Catch it and leave the page usable without the 3D view.

## Choosing how to make your scene

| You want | Use | Notes |
|---|---|---|
| A nice 3D backdrop, quickly | a built-in scene + `heliosLookAt` / `heliosSetSun` | `config: { scene: "valley" }`. Camera spots: `catalog/<scene>.json` → `camera`, `topLevelActors` |
| Your own object or character | `config: { gltf: "models/thing.glb" }` | The model on a ground plane under the sky. glTF units are used as metres: a model authored in cm is 100× too big, so scale it in a scene file or with an `actor_transform` |
| An authored scene: props, lights, materials, fog, settings | a **scene file**, `config: { sceneFile: "my.helios.json" }` | Easiest to make in the engine's editor: `python D:/AI/Unreal/serve.py 8080 D:/AI/Unreal/web` → `http://localhost:8080/editor.html?scene=gallery`, edit, Ctrl+S saves the file. Format: API.md → `SceneDoc` |
| A scene built or changed by code at run time (chapters, interaction) | editor operations: `apply(H, { op: "place", … })` | Every operation in API.md → *Editor operations*. Example: `D:/AI/JensenHuang_Opus55/js/scene.js` |
| To switch scenes later | `heliosLoadScene("forest")`, `heliosLoadModel(url)` | The new scene is preprocessed while the old one keeps running |

Names you can use:
* **Built-in scenes:** in `catalog/<scene>.json`. Materials are prefixed, e.g. `gallery/flagstone`,
  `valley/...`. In a glTF scene, meshes and materials are `MODEL/NAME`.
* **Engine content** for `place`:
  * meshes: `engine/cube`, `engine/sphere`, `engine/cylinder`, `engine/plane` (unit size,
    centred: −0.5…0.5), `engine/rock`, `engine/boulder`, `engine/grass`, `engine/fern`,
    `engine/flower-red|yellow|blue|white`, `engine/bush`, `engine/tree`, `engine/conifer`;
  * materials: `engine/default`, `engine/white`, `engine/rock`, `engine/gold`, `engine/glass`,
    `engine/emissive`, … (full list in a catalog's `content.materials`).
* **Live state:** read it with `sceneState(H)` (wraps `heliosEditorState()`). Actors come as
  columns, `{ id: [], name: [], parent: [], flags: [] }`; `sceneState` also gives you
  `actorList: [{ id, name, parent }]`.

Recipes (`H` is what `startHelios` returned; `apply`, `sceneState`, `yaw` come from helios-boot.js):

```js
// Hide everything but the floor of the gallery, then build your own set on it.
const s = sceneState(H);
apply(H, { op: "actor_visibility", actors: s.actorList.filter((a) => a.parent === -1 && a.name !== "Courtyard").map((a) => a.id), visible: false });
apply(H, { op: "delete", lights: s.lights.map((l) => l.index) });

// A golden box, 2 × 0.5 × 2 m, turned 30°.
const { instances } = apply(H, { op: "place", item: "engine/cube", transform: { t: [0, 0.25, 0], r: yaw(30), s: [2, 0.5, 2] } });
apply(H, { op: "assign_material", instances, material: "engine/gold" });

// A material tweak, a light, fog and a render setting.
apply(H, { op: "material", name: "gallery/flagstone", params: { base_color: [0.05, 0.05, 0.06], roughness: 0.2 } });
apply(H, { op: "add_light", def: { shape: "point", position: [0, 3, 2], color: [1, 0.6, 0.3], intensity: 40, range: 12, source_radius: 0.05, cast_shadows: true } });
const env = s.environment; env.fog.density = 0.02; env.sun_elevation_deg = -20;   // night
apply(H, { op: "environment", env });
apply(H, { op: "render", key: "motion_blur", value: 0 });   // keys: catalog/<scene>.json → render.schema
```

Use the `render` op to set features: it sets a value. `heliosKey("KeyG")` presses a hotkey, which only toggles.

## Living with the engine in a page

* **One engine per page.** The wasm module is a singleton and draws only into
  `<canvas id="helios-canvas">`.
  * The canvas needs a CSS size. The engine renders at that size × devicePixelRatio (capped at 1.5)
    and follows resizes by itself.
  * Style it with `#helios-canvas` or a class.
  * If you need the engine twice, put it in an `<iframe>`; `JensenHuang_GPT6Astra` has an `engine.html` like that.
* **Input with `embedded: true`:**
  * Dragging on the canvas looks around.
  * Keys reach the engine only while the canvas has focus. Letter keys then toggle features
    (G GI, H shadows, …, N next scene), so blur the canvas after a click if the page has its own keys.
  * The mouse wheel scrolls the page.
  * For a camera the user can't move, put `pointer-events: none` on the canvas.
* **Status text.** The engine writes English progress and errors into the element named by
  `heliosConfig.statusElement` (default `#status`). `startHelios` points it at a hidden span. A page
  with its own `#status` element must never load the engine without `statusElement`.
* **URL parameters** such as `?scene=`, `?gltf=` and `?sp=` also reach the engine (config wins).
  Don't reuse those names for the page's own parameters.
* **Performance:**
  * Dynamic resolution keeps the GPU frame under `gpuBudgetMs`: 14 by default, 10–12 for a
    background behind text.
  * In Firefox the budget is at most 8 ms, whatever the page asks for. Firefox starts a frame
    only once the last one is on screen, so 14 ms of GPU work ran at ~27 fps there even on an
    RTX 3090/4090.
  * Firefox compiles shaders one at a time and keeps no cache between visits: after the first
    image the view stutters for ~25 s while the rest compile. Don't time an intro animation to
    the first image in Firefox.
  * `screenPercentage` fixes the resolution instead.
  * Rendering a still view costs GPU time every frame. Pause it with `heliosSetPaused(true)` once
    the image has settled; `startHelios` already pauses it when the canvas is hidden.
  * A paused engine also stops compiling shaders. Pause a visible canvas only when
    `heliosFrames() > 0 && heliosBusy() === ""`, or it stays empty.
* **Weak GPUs (quality levels).** Most visitors don't have a gaming GPU: on a laptop's or a
  desktop CPU's integrated GPU the full renderer runs at a few frames a second.
  * The engine picks a level by itself (`heliosConfig.quality`, default `"auto"`). It reads the
    GPU from the browser: integrated (Intel UHD / Iris / Arc iGPU, AMD Radeon Graphics) and
    mobile GPUs start at `low`, Apple silicon at `medium`, the rest at `high`.
  * While the page runs it moves one level down when the GPU stays over its budget even at the
    lowest resolution for 3 s, and one level up after 8 s far under it.
  * `medium` uses Lumen Lite's cheaper GI, the cheaper transparency and coarser geometry. `low`
    also drops the volumetric clouds and fog, coloured light through glass, local lights'
    shadow maps, short-range AO and motion blur, and renders down to 35 % resolution.
  * Design the scene so that it still reads at `low`: don't make a fog beam or a cloud the only
    thing a section shows. Check with `?quality=low`.
  * `heliosQuality()` says the level and the GPU (`{"level": "low", "auto": true, "gpu":
    {"class": "integrated", …}}`); `heliosSetQuality("low" | "medium" | "high" | "auto")` sets it,
    e.g. from a quality menu. A fixed `quality` in the config turns the automatic moves off.
* **Readiness:**
  * `heliosReady()`: the scene, including a scene file's edits, is in, so camera and editor calls
    are safe.
  * `heliosFrames() > 0`: the first image is on the canvas.
  * `heliosShaders().waiting`: the features still compiling.
  * `heliosError()`: why the engine stopped (start-up error or lost GPU device).
* **Telemetry for a HUD.** `heliosHud()` returns text such as `… (60 fps) …`;
  `heliosGpuPasses(false)` gives per-pass GPU times.

## Deploying

* Ship the whole `helios/` folder, including `pkg/snippets/`, which holds the bake worker and the
  GPU memory hook. Without it the module fails to load.
* Any static host works. On GitHub Pages add an empty `.nojekyll` at the root. HTTPS is required
  (Pages has it).
* `.wasm` must be served as `application/wasm`: GitHub Pages, `python -m http.server` and Vite do
  this. To stream `.hvm` geometry the server must honour Range requests (`python D:/AI/Unreal/serve.py`).
* **With a bundler (Vite etc.):** put `helios/` under `public/` so it is copied as is and never
  bundled. Load it at run time:

  ```js
  const { startHelios } = await import(/* @vite-ignore */ `${import.meta.env.BASE_URL}helios/helios-boot.js`);
  ```

* Commit `helios/`, including `helios-engine.json`, into the site's repository. The manifest says
  which engine the site runs.

## Updating a site to a newer engine

```powershell
python D:/AI/Unreal/ittkezdjed/sync.py --check D:/AI/MySite/helios      # what would change (copies nothing)
python D:/AI/Unreal/ittkezdjed/sync.py D:/AI/MySite/helios --build      # rebuild if stale, copy, report
```

The report compares the site's `helios-engine.json` with the engine now. It shows:
* functions added, removed or changed, and new `heliosConfig` keys;
* the [CHANGELOG](CHANGELOG.md) entries since the last sync;
* the engine commits since then.

After syncing:
1. Fix whatever the report names.
2. Grep the site for removed functions.
3. Load the page and check the browser console. Editor operations return `{error}` instead of
   throwing; `apply()` throws.

Old sites:
* A site with no `helios-engine.json` has an unknown engine. Its first sync reports
  "First sync into this folder"; read the CHANGELOG from the top.
* Older sites copied `D:/AI/Unreal/web/pkg` by hand. `sync.py` replaces their `pkg/` folder. Point
  it at the folder that contains `pkg/`.

## For engine maintainers

When a change touches what pages use, do these steps in the same commit:
1. Regenerate the reference: `python ittkezdjed/sync.py --api`. Pages use the exports in
   `web.rs` / `web_editor.rs`, `heliosConfig`, the URL parameters, editor operations and the
   scene file format.
2. Add a dated line to `CHANGELOG.md`: what changed and what a site has to do.
3. If scene contents changed (material, mesh or actor names), regenerate `catalog/`:
   `uv run --with playwright==1.56.0 python -I ittkezdjed/tools/catalog.py`.
4. Keep `template/` working.
