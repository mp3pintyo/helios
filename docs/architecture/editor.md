# Editor, scene files and PCG — `helios-demo`, `helios-pcg`, `web/editor`

Back to the [README](../../README.md).

* **Editor** (`web/editor.html`, `web/editor/*.js|css`, `crates/helios-demo/src/editor/`, `web_editor.rs`,
  `app_editor.rs`; UE's Level Editor). A dark, UE5-style editor around the live WebGPU viewport, plain ES modules
  without a build step or CDN: toolbar (file, undo / redo, W/E/R gizmo modes, world / local axes, grid, rotation and
  scale snapping, scene switch, Play, Material Graph), a Content browser (engine shapes and nature prefabs, point /
  spot lights, PCG presets, the scene's materials and meshes, glTF import by URL or by dropping a `.glb`), a World
  Outliner (World → Environment / Render settings, PCG volumes and the **actor hierarchy** — see *Actors* below —
  with search, eye toggles, drag-and-drop re-parenting and inline renaming; virtualised rows, so the forest's 6 218
  actors scroll smoothly) and a Details panel (an actor's name, parent, key, transform relative to its parent and
  its component list; a mesh component's relative transform, mesh, material slot and the material's
  parameters, graph, Substrate slab and displacement (magnitude, centre, height channel), rendering; a light component's colour, intensity, attenuation radius, source radius,
  shadows, spot aim and cones; PCG volume, surface, seed and every node's parameters; sun / fog / clouds / wind;
  render settings). The engine owns the scene: the UI reads it as JSON and sends edit operations through the wasm
  page API (`heliosEditorState`, `heliosEditorApply`, `heliosEditorPreview`, `heliosEditorPick`, `heliosEditorCamera`,
  `heliosEditorSetSelection`, `heliosEditorSave`, `heliosEditorLoad`, `heliosEditorImport`, …). **Edits are
  documents**: every operation produces a new `SceneDoc` (one undo step; slider drags carry a merge key and form one
  step), and `Editor::sync` reconciles the scene with it, touching only what differs (an instance's transform, a
  material's parameters, one PCG volume's instances), so undo / redo are document swaps and loading a file is the
  same reconcile from an empty document. A removed component's instance is first hidden (`Scene::set_visible`: the
  `HIDDEN` instance flag, culled for every view; translucency, water bodies, Lumen and assembly expansion skip hidden
  instances) and put on a free list per mesh, which the next component of the same mesh takes; once the editor has
  been idle for 1.5 s with at least 32 free instance or 8 free light slots (or for 30 s with any), `Scene::compact`
  removes them and every holder of handles (the editor, the demo motion, the selection outline) maps them through the
  returned `Compaction` — documents never hold slots, so undo after a compaction simply takes new ones. Transform
  edits and gizmo previews go through `Scene::set_transform`, i.e. the per-instance path of *Dynamic world*; a drag
  (gizmo preview or the commits of one gesture) makes the moved instances `Movable` — dynamic casters over the cached
  static shadow pages — and gives them their own mobility back 30 frames after the last move, so only the release
  re-renders the static layer under them once. Hiding, deleting, duplicating, reassigning materials and PCG
  regeneration are structural changes.
  **Picking** reads the visibility buffer: a request maps the clicked output pixel to the internal resolution, a
  one-thread compute pass (`pick.wgsl`) decodes the visible-cluster entry into its instance (an assembly part into
  its owning tree) and reconstructs the world position and normal, and the 48-byte result is read back
  asynchronously (it arrives 1–2 frames later) — exact per pixel, so alpha-tested holes, swaying leaves, voxels and
  software-rasterised clusters pick what is drawn; translucent surfaces (not in the visibility buffer) are tested
  against their oriented bounds on the CPU and win when nearer. A click selects the picked instance's actor,
  Alt+click its component; PCG-spawned instances select their volume. The
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
* **Actors** (`editor/actors.rs`, `scenes/outline.rs`; UE's actors, components and attachment). An actor is a named
  node with a transform relative to an optional parent and components: mesh instances with their materials, lights,
  volumes and particle emitters. A component's world transform is its actor's world transform times its own, an
  actor's is its parent's times its own, so moving an actor moves its components and its children (instances through
  the per-instance path, lights and emitters with them; a fire keeps driving its light). The scene builders record
  their actors while they build (`outline::begin` / `end` around the instances, lights and emitters of one actor,
  `folder` / `end_folder` around groups): the forest's trees are one actor each (trunk + foliage) in a *Trees* folder,
  bushes, ferns, grass, rocks and fallen logs (log + mushrooms) in theirs, the campfire one actor of 18 instances, its
  light, fire volume, sparks and embers; the cathedral's nave wall bays (wall, arcade, clerestory glass and tracery,
  triforium), piers, vault bays, aisle bays, buttresses, west front, towers, apse, pews, the altar, candelabra and
  altar candles with their lights, chandeliers with theirs, choir screens, glass screens, the censer with its incense
  and the dust motes; the valley's landscape, lake, sculpture on its plinth, boulders, mist, raft, translucency
  corner and stone path; the gallery's specimens (plinth + objects + lamp), terraces, beds, turntable and sign.
  Whatever a builder leaves out becomes an actor of its own (forest: 6 218 actors over 8 278 instances). Every actor
  has a stable **key** — its folder path, name and, for actors, its pivot in centimetres
  (`Trees/Conifer@1250,-30,-4510`) — that scene files address it by; the UI uses live ids. Operations: actor
  transforms (world, from the gizmo; relative, from Details), visibility (with everything below), rename (F2 or a
  double click in the outliner), re-parent keeping the world transform (drag rows onto an actor, onto *Level* to
  detach; cycles are refused), group (Ctrl+G: a new actor at the centre of the selection) and ungroup (Shift+G),
  duplicate and delete (an actor with everything below it), and per component transform, visibility, material,
  shadow casting, light. Older operations addressed by instance or light index still work (they edit the component,
  or the actor of a single component). `Editor::sync` re-resolves only the actors whose definition changed and places
  them and everything below them; the outline's parent / child links are rebuilt only when parents change or actors
  come and go. Costs (wasm, forest): moving one actor 4 ms, all 1 500 trees in one edit 33 ms; dragging the campfire
  costs 10.95 ms GPU against 9.32 ms without a drag (13.93 ms when its instances stayed static: 214 instead of 123
  shadow pages, 109 k instead of 31 k shadow draws per frame); a pew in the cathedral 7.33 against 6.93 ms (9.12 ms
  static). `--drag-actor NAME [--drag-static]` (native) and `heliosEditorMovableDrags(false)` (browser) reproduce the
  measurement; see `docs/round8/editor-hierarchy.md`.
* **Scene files** (`.helios.json`, `editor/doc.rs`, serde; version 2). A base scene — a procedural scene and detail
  (both change the instance list) or a glTF URL — plus only the edits: base-actor overrides by key (name, parent,
  relative transform, visibility, deleted, and per component name its transform, visibility, material, shadow
  casting, light), added actors (`+ID` keys: name, parent key, relative transform and their components — meshes by
  mesh and material *name*, lights), material parameter overrides by name, material graphs (`graphs`) and slabs, the
  environment, render-setting overrides by key, imported models (name + URL), PCG volumes (volume,
  surface, graph) **and their generated output** (parts + per-instance TRS), so a file reloads with exactly the
  instances it was saved with. Version 1 files (base instances and lights by build order, added instances and
  lights) are migrated on load: an edit of a single-component actor's instance becomes an actor edit, one of a
  multi-component actor a component edit, added instances and lights become added actors, a PCG surface given by
  instance index a component reference. Meshes get stable names where scenes create them (`mesh_store::add` names every baked
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
