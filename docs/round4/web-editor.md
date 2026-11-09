# Round 4 — web-editor: the Helios editor, scene files and PCG

README text for the merge (Architecture bullets, "Status vs. Unreal Engine 5" gaps, flags and hotkeys).

## Quick start (README "Quick start" additions)

```powershell
./build-web.ps1; python serve.py 8080 web
#   http://localhost:8080/editor.html?scene=valley        the level editor (or Tab in index.html)
#   http://localhost:8080/editor.html?scene-file=my.helios.json   open a saved scene
cargo run -p helios-demo --release -- --scene-file my.helios.json   # natively (Tab: editor tools)
```

## Architecture (new bullets for section 2, after *Profiling*)

* **Editor** (`web/editor.html`, `web/editor/*.js|css`, `crates/helios-demo/src/editor/`, `web_editor.rs`,
  `app_editor.rs`; UE's Level Editor). A dark, UE5-style editor around the live WebGPU viewport, plain ES modules
  without a build step or CDN: toolbar (file, undo / redo, W/E/R gizmo modes, world / local axes, grid, rotation and
  scale snapping, scene switch, Play), a Content browser (engine shapes and nature prefabs, point / spot lights, PCG
  presets, the scene's materials and meshes, glTF import by URL or by dropping a `.glb`), a World Outliner (World →
  Environment / Render settings, lights, PCG volumes, the instances grouped by mesh or material, search, eye toggles;
  virtualised rows, so the forest's 8 261 instances scroll smoothly) and a Details panel (transform with scrubbable
  X/Y/Z fields and a uniform-scale lock, mesh, material slot and the material's parameters, rendering; light colour,
  intensity, attenuation radius, source radius, shadows, spot aim and cones; PCG volume, surface, seed and every
  node's parameters; sun / fog / clouds / wind; render settings). The engine owns the scene: the UI reads it as JSON
  and sends edit operations through the wasm page API (`heliosEditorState`, `heliosEditorApply`,
  `heliosEditorPreview`, `heliosEditorPick`, `heliosEditorCamera`, `heliosEditorSetSelection`, `heliosEditorSave`,
  `heliosEditorLoad`, `heliosEditorImport`, …). **Edits are documents**: every operation produces a new
  `SceneDoc` (one undo step; slider drags carry a merge key and form one step), and `Editor::sync` reconciles the
  scene with it, touching only what differs (an instance's transform, a material's parameters, one PCG volume's
  instances), so undo / redo are document swaps and loading a file is the same reconcile from an empty document.
  Instances are never removed — `Scene::set_visible` hides them (a new `INSTANCE_HIDDEN` flag culled for every view;
  translucency, water bodies and Lumen skip hidden instances) and hidden slots are reused for the next instance of
  the same mesh — so handles in the outliner, the selection and picking stay valid. **Picking** reads the visibility
  buffer: a request maps the clicked output pixel to the internal resolution, a one-thread compute pass
  (`pick.wgsl`) decodes the visible-cluster entry into its instance and reconstructs the world position and normal,
  and the 48-byte result is read back asynchronously (it arrives 1–2 frames later) — exact per pixel, so
  alpha-tested holes, swaying leaves and software-rasterised clusters pick what is drawn; translucent surfaces (not
  in the visibility buffer) are tested against their oriented bounds on the CPU and win when nearer. PCG-spawned
  instances select their volume. The **selection outline** (`selection.rs`, `selection.wgsl`) is a full-screen pass
  after tone mapping that tints selected instances and draws a 2-px orange outline from eight visibility taps per
  output pixel, against a per-instance selection bit set; it is only recorded while something is selected. The
  **gizmo** (translate arrows + plane handles + screen-plane centre, rotation rings with angle readout, local-axis
  scale handles + uniform centre) is drawn in a 2D overlay canvas from the engine's unjittered view-projection and
  hit-tested in screen space; drags apply a delta to every selected object around the primary's pivot, preview
  through `heliosEditorPreview` (no undo step) and commit one transform edit on release (Esc cancels); axes pointing
  at the camera switch to screen-space dragging. Overlay also shows light icons (clickable), spot cones, light
  ranges, PCG volume boxes with labels and the bounds of selected instances. Camera: RMB + WASD/QE flies (the engine's
  fly camera), the wheel dollies, F frames the selection. Placing content picks the surface under the viewport centre
  (or the drop point) on the GPU. Native: Tab switches the same editor core on without panels (click to select with
  the outline, Del, Ctrl+D, F, Ctrl+Z / Ctrl+Y, Ctrl+S saves the scene file).
  Cost (RTX 3090, shared with other agents, 1600×900): selection outline 0.05 ms (three runs, the first version with
  20 taps cost 0.34–0.57 ms), pick ≈ 0 (one thread, once per click); with nothing selected the frame is unchanged
  (valley vs. the branch base: max pixel difference 5/255 = temporal noise, GPU 9.83 → 9.04 ms median of 3,
  min 5.02 → 5.18 ms, noise). Gizmo drags and every edit currently bump the instance version, which re-uploads the
  instance buffer, Lumen's distance-field instances and invalidates cached shadow pages — the `dynamic-world`
  branch's per-instance invalidation makes them cheap; the editor only calls `Scene::set_transform`.
* **Scene files** (`.helios.json`, `editor/doc.rs`, serde). A base scene — a procedural scene and detail
  (both change the instance list) or a glTF URL — plus only the edits: base-instance overrides keyed by index with
  the mesh name for validation (transform as TRS, visibility, material, deleted), added instances by mesh and
  material *name*, material parameter overrides by name, base light definitions by index and added lights, the
  environment, render-setting overrides by key, imported models (name + URL), PCG volumes (volume, surface, graph)
  **and their generated output** (parts + per-instance TRS), so a file reloads with exactly the instances it was saved
  with. Meshes get stable names where scenes create them (`mesh_store::add` names every baked mesh, duplicates
  become `rock #2`; glTF imports are `MODEL/MESH`), materials get explicit names in the valley and the forest and
  `<shading model> <n>` elsewhere. Save downloads the file (Ctrl+S), Open uses a file picker (Ctrl+O), `?scene-file=URL`
  builds the file's base scene directly; natively `--scene-file PATH` (relative glTF paths resolve against the file).
  The render settings the Details panel shows come from a data-driven registry (`editor/settings.rs`: key, label,
  group, kind, getter / setter) that also stores them in scene files — a new setting is one entry.
* **PCG** (`crates/helios-pcg`, `editor/pcg.rs`; UE 5.7's Procedural Content Generation framework). A deterministic
  point graph: `SurfaceSampler` (on the landscape heightfield, one jittered candidate per cell of a world-anchored
  grid of `1/density` m², or on any mesh instance's triangles, area-weighted), filters (`SlopeFilter`,
  `HeightFilter`, `NoiseDensity` — world-space fractal value noise —, `DensityFilter` with stochastic thinning,
  `DistanceFilter` to points / splines with falloff, `BoundsFilter`), `SelfPruning` (Poisson-disk thinning by
  priority = density then seed, hash-grid accelerated), `Difference` and `Merge` (graphs are DAGs: nodes read any
  earlier nodes), `Transform` (scale range and jitter, random yaw, tilt, align to normal, offset) and a weighted
  `Spawner` of prefabs (several mesh / material parts per entry: trunk + foliage, stem + petals). Every random
  number is a hash of the graph seed, the node and the point's own seed (from its world grid cell or triangle), never
  of processing order, so output is bit-identical run to run and moving a volume keeps the points in the area it still
  covers (unit tests: determinism, pruning distance and order independence, slope / distance / bounds filters,
  density vs. area, weights, DAG difference, JSON round trip, presets). In the editor, a PCG volume (oriented box,
  movable / scalable / yaw-rotatable with the gizmo) runs a preset graph — *Rocks on slopes*, *Flowers in meadows*,
  *Forest edge* (trees thickening away from an edge spline, bushes and ferns along it, cleared around the trunks)
  — on built-in engine content (`editor/content.rs`, baked on first use), and regenerates live while a node's
  parameter is dragged (at most once per frame, one undo step per gesture); spawned instances reuse the volume's
  instance slots per mesh. Browser (wasm, headless Chromium): flowers in a 28×28 m volume 3 170 instances, regenerate
  15–20 ms; forest edge (80×50 m) 408 instances, 7–12 ms; rocks on slopes 3–8 ms.

## Status vs. Unreal Engine 5 (new gaps)

8. **Editor**: no actors or hierarchy — a tree's trunk and foliage are two instances, selected separately; deleted
   instances and lights stay as hidden / disabled slots (reused, never compacted); undo history is in memory (200
   steps); no landscape sculpting or painting, no multi-viewport, no editor-only billboards in the renderer (light and
   volume icons are a 2D overlay without depth test), the gizmo draws over everything; typed transform values apply to
   the primary selection only; the native window has the editor core without panels. Every edit re-uploads all
   instances and rebuilds Lumen's instance lists until per-instance invalidation (`dynamic-world`) lands.
   Scene files address base instances by build order (a changed scene builder is detected by mesh name and the edit
   skipped with a warning); dropped `.glb` files are referenced by file name and must be hosted next to the page to
   reload; glTF imports bake on the main thread in the browser.
9. **PCG**: a fixed node set without a graph canvas (nodes and their parameters are edited in the Details panel; the
   graph's structure comes from the presets / the file), no attribute system or custom nodes, no spline editing in
   the viewport (distance targets are volume-local data), spawners reference meshes and materials by name, generation
   runs on the main thread (no partitioned / hierarchical generation, no runtime generation around the camera), and
   spawned instances are not individually editable. Native and wasm builds may differ in the last bits of `sin` /
   `cos`, so a point exactly at a filter threshold could flip between them; saved output sidesteps it.

## Flags and hotkeys

* `web/editor.html` (same query parameters as `index.html`) and **Tab** in `index.html` open the editor; Tab in the
  editor toggles play mode. `?scene-file=URL` opens a scene file.
* Editor (viewport focused, right mouse button not held): **W / E / R** move / rotate / scale, **Space** cycles them,
  **Ctrl+`** world / local axes, **F** focus, **Del** delete, **Ctrl+D** duplicate, **Ctrl+Z / Ctrl+Y** undo / redo,
  **Ctrl+S** save, **Ctrl+O** open, **Esc** cancel drag / deselect; RMB + WASD/QE fly, wheel dolly; Shift / Ctrl
  click add / toggle selection; drag content into the viewport; drop a material onto an object to assign it.
* Native: `--scene-file PATH`, `--select i,j` (outline instances in headless captures), **Tab** editor tools.

Screenshots (headless Chromium, WebGPU): `docs/round4/web-editor/` — editor with gizmo and selection outline
(valley), the cathedral's lights in the outliner and overlay, a selected conifer in the forest, PCG flowers in
meadows and a forest edge in the valley.
