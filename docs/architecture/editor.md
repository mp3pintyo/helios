# Editor, scene files and PCG — `helios-demo`, `helios-pcg`, `web/editor`

Back to the [README](../../README.md).

* **Editor** (`web/editor.html`, `web/editor/*.js|css`, `crates/helios-demo/src/editor/`, `web_editor.rs`,
  `app_editor.rs`; UE's Level Editor). A dark, UE5-style editor around the live WebGPU viewport, plain ES modules
  without a build step or CDN: toolbar (file, undo / redo, W/E/R gizmo modes, world / local axes, grid, rotation and
  scale snapping, scene switch, Play, Material Graph), a Content browser (engine shapes and nature prefabs, point /
  spot lights, PCG presets, the scene's materials and meshes, glTF import by URL or by dropping a `.glb`), a World
  Outliner (World → Environment / Render settings, lights, PCG volumes, the instances grouped by mesh or material,
  search, eye toggles; virtualised rows, so the forest's 8 278 instances scroll smoothly) and a Details panel
  (transform with scrubbable X/Y/Z fields and a uniform-scale lock, mesh, material slot and the material's
  parameters, graph, Substrate slab and displacement (magnitude, centre, height channel), rendering; light colour, intensity, attenuation radius, source radius,
  shadows, spot aim and cones; PCG volume, surface, seed and every node's parameters; sun / fog / clouds / wind;
  render settings). The engine owns the scene: the UI reads it as JSON and sends edit operations through the wasm
  page API (`heliosEditorState`, `heliosEditorApply`, `heliosEditorPreview`, `heliosEditorPick`, `heliosEditorCamera`,
  `heliosEditorSetSelection`, `heliosEditorSave`, `heliosEditorLoad`, `heliosEditorImport`, …). **Edits are
  documents**: every operation produces a new `SceneDoc` (one undo step; slider drags carry a merge key and form one
  step), and `Editor::sync` reconciles the scene with it, touching only what differs (an instance's transform, a
  material's parameters, one PCG volume's instances), so undo / redo are document swaps and loading a file is the
  same reconcile from an empty document. Instances are never removed — `Scene::set_visible` hides them (the
  `HIDDEN` instance flag, culled for every view; translucency, water bodies, Lumen and assembly expansion skip hidden
  instances) and hidden slots are reused for the next instance of the same mesh — so handles in the outliner, the
  selection and picking stay valid. Transform edits and gizmo previews go through `Scene::set_transform`, i.e. the
  per-instance path of *Dynamic world* (edited instances stay `Static`, so the shadow pages under them are rendered
  in full); hiding, deleting, duplicating, reassigning materials and PCG regeneration are structural changes.
  **Picking** reads the visibility buffer: a request maps the clicked output pixel to the internal resolution, a
  one-thread compute pass (`pick.wgsl`) decodes the visible-cluster entry into its instance (an assembly part into
  its owning tree) and reconstructs the world position and normal, and the 48-byte result is read back
  asynchronously (it arrives 1–2 frames later) — exact per pixel, so alpha-tested holes, swaying leaves, voxels and
  software-rasterised clusters pick what is drawn; translucent surfaces (not in the visibility buffer) are tested
  against their oriented bounds on the CPU and win when nearer. PCG-spawned instances select their volume. The
  **selection outline** (`selection.rs`, `selection.wgsl`) is a full-screen pass after tone mapping that tints
  selected instances and draws a 2-px orange outline from eight visibility taps per output pixel, against a
  per-instance selection bit set; it is only recorded while something is selected. The **gizmo** (translate arrows
  + plane handles + screen-plane centre, rotation rings with angle readout, local-axis scale handles + uniform
  centre) is drawn in a 2D overlay canvas from the engine's unjittered view-projection and hit-tested in screen
  space; drags apply a delta to every selected object around the primary's pivot, preview through
  `heliosEditorPreview` (no undo step) and commit one transform edit on release (Esc cancels); axes pointing at the
  camera switch to screen-space dragging. The overlay also shows light icons (clickable), spot cones, light ranges,
  PCG volume boxes with labels and the bounds of selected instances. Placing content picks the surface under the
  viewport centre (or the drop point) on the GPU. Natively, Tab switches the same editor core on without panels.
  Cost (shared GPU): selection outline 0.05 ms (a first version with 20 taps cost 0.34–0.57 ms), pick ≈ 0 (one
  thread, once per click); with nothing selected the frame is unchanged.
* **Scene files** (`.helios.json`, `editor/doc.rs`, serde). A base scene — a procedural scene and detail (both change
  the instance list) or a glTF URL — plus only the edits: base-instance overrides keyed by index with the mesh name
  for validation (transform as TRS, visibility, material, deleted), added instances by mesh and material *name*,
  material parameter overrides by name, material graphs (`graphs`) and slabs, base light definitions by index and
  added lights, the environment, render-setting overrides by key, imported models (name + URL), PCG volumes (volume,
  surface, graph) **and their generated output** (parts + per-instance TRS), so a file reloads with exactly the
  instances it was saved with. Meshes get stable names where scenes create them (`mesh_store::add` names every baked
  mesh, duplicates become `rock #2`; glTF imports are `MODEL/MESH`), materials get explicit names in the valley and
  the forest and `<shading model> <n>` elsewhere. Save downloads the file (Ctrl+S), Open uses a file picker (Ctrl+O),
  `?scene-file=URL` builds the file's base scene directly; natively `--scene-file PATH` (relative glTF paths resolve
  against the file). The render settings the Details panel shows come from a data-driven registry
  (`editor/settings.rs`: key, label, group, kind, getter / setter) that also stores them in scene files — a new
  setting is one entry.
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
* **Material graph editor** (`web/editor/graph.js`, `editor/materials.rs`). A dark node canvas docks over the
  viewport (toolbar *Material Graph*): pan (right / middle drag), zoom (wheel), drag nodes, drag pin to pin to link
  (an input takes one link), Alt+click unlinks, right-click or double-click opens a searchable palette built from the
  engine's catalog (`heliosMaterialNodes`), inline editors for constants, colours, parameter names and defaults, UV
  sets, channels; *New from preset* (lava, wet stone, stylised water, brick pavement, blank), *Assign to material*; compile errors
  mark their nodes and list below. Every change is a scene-document edit (`{op: "graph"}`: undo / redo, saved in
  `.helios.json` under `graphs`, gestures merged into one undo step) and recompiles live. The Details panel shows a
  material's graph (assign / open), its parameters as sliders and colour pickers (material instance values, saved
  per material), its Substrate slab (clear coat, sheen, subsurface and radius, transmission, anisotropy, IOR,
  specular, unlit) and its displacement — magnitude, centre and the height channel, or *Driven by graph* (opening
  it) while one of the graph's displacement pins is linked: *Height* or *Vector* (no centre), marked *animated*
  when the graph reads the time; linking a node to such a pin re-dices the surface live. The Output node sets its
  two geometry pins (Displacement, VectorDisplacement) apart with a tooltip each, and the status line names what
  the graph does to geometry ("vector displacement", "animated displacement"; round 6: the spaces a TransformVector
  turns into world space — "tangent-space vector displacement" — and an animation's speed bound, "≤ 1.64 ×
  magnitude/s", or "no speed bound"). TransformVector nodes pick their spaces from two drop-downs; the Details
  panel's note on an animated material gives its speed bound in cm/s (or says its shadows refresh every frame).
