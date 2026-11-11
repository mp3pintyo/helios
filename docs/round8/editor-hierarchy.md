# Round 8 — editor actors and hierarchy (`editor-hierarchy`)

## Problem

The browser editor edited *instances* and *lights*. A tree's trunk and its foliage were two instances, selected
and moved separately; the campfire was 18 instances, a light, a fire volume and two particle emitters with nothing
tying them together. Deleted instances and lights stayed in the scene forever as hidden / disabled slots. A dragged
object stayed `Static`, so every preview frame re-rendered the cached shadow pages under it in full. Scene files
addressed base instances by build order, so any builder change before an edited instance broke the file.

## UE reference

UE's level is a set of **actors** (`AActor`) owning **components** (`UStaticMeshComponent`, `UPointLightComponent`,
`UNiagaraComponent`, …) in a scene-component tree; actors attach to other actors (`AttachToActor`, keeping or
snapping the world transform); the World Outliner shows the attachment hierarchy (plus folders), with drag-and-drop
attachment, renaming (F2), visibility eyes; Ctrl+G / Shift+G group and ungroup (`AGroupActor`); duplicating or deleting
an actor takes attached children along in the outliner's sense of the selection. Moving an actor in the editor moves
its components (render-thread transform updates; virtual shadow map pages under a moving primitive are invalidated
per primitive). Levels save actors by name / GUID, not by spawn order.

## Design

* **Actor model** (`editor/actors.rs`). An actor: key, name, parent key, transform relative to the parent,
  visibility, components. A component: name (unique in its actor), transform relative to the actor, visibility, and
  a kind — mesh (mesh, material, shadow casting, instance slot), light (its definition in actor space, light slot),
  volume or particle emitter (index of the base scene's emitter). World transforms compose parent ∘ local; a
  component's is actor world ∘ component local. Visibility composes the same way.
* **Base actors** come from the builders (`scenes/outline.rs`): a thread-local recorder active during
  `scenes::build_with` collects the instances, lights, volumes and particle emitters added between
  `outline::begin(scene, name, pivot)` and `outline::end`, folders from `outline::folder` / `end_folder`, and lights
  made later with `outline::add_lights` (the cathedral adds its fixtures' lights after all fixtures). Outside a
  recording every call is a no-op, so builders stay usable from tests. Forest, cathedral, valley and gallery are
  outlined; anything not claimed becomes an actor of its own (model scenes: one actor per instance).
* **Stable keys.** Folder path + name, plus the pivot position in centimetres for actors
  (`Fallen logs/Fallen log@-3410,212,880`), duplicates numbered. Inserting or removing other actors in a builder
  does not change a key (build-order indices did); a builder that moves an object changes its key and the edit is
  reported as not applying — the same failure mode the mesh-name check gave before, but local to that object.
* **Document v2** (`editor/doc.rs`). `actors: {key → ActorEdit}` (name, parent key or `""` for the root, relative
  transform, visible, deleted, `components: {name → ComponentEdit}`), `added_actors: [AddedActor]` (id, name, parent,
  transform, components by mesh / material name or light definition). Undo is still a document swap.
* **Reconcile** (`Editor::sync_actors`). Only keys whose override or added definition changed are re-resolved; a
  re-resolved actor keeps the slots of components that stay (by name and mesh), takes slots for new ones from the
  free lists, retires removed ones; parent / child links are carried over unless parents changed or actors came or
  went (then rebuilt in O(n), cycles broken with a warning); then the changed actors and everything below them are
  placed (world transforms, instance transforms / materials / shadow casting / visibility, world-space lights; the
  scene's volume and particle lists are rebuilt from the live emitter components in base order, so a fire keeps its
  simulation state and the light it drives).
* **Interactive moves.** `Editor::preview_actors` moves an actor's subtree without touching the document;
  every instance it moves becomes `Movable` (remembering its own mobility) — a dynamic caster over the cached static
  shadow pages, the round-4 per-instance path. Commits that belong to a gesture (a merge key: the gizmo's release,
  scrubbed fields) count as interactive too. `Editor::tick` (every frame, from `DemoApp::update`) gives each instance
  its mobility back 30 frames after its last move, which re-renders the static layer under it once.
* **Lazy compaction.** Retired slots wait on free lists (instances per mesh, lights). When the editor has been idle
  for 90 frames with ≥ 32 free instances or ≥ 8 free lights (or 1800 frames with any), `DemoApp::compact_editor`
  calls the new `Scene::compact` (helios-render `scene_edit.rs`), which removes them, remaps the volumes' driven
  lights, bumps the instance version and a new instance *layout* version (the motion tracker drops its per-index
  previous transforms), and returns a `Compaction` (old → new index) that the editor, the demo motion and the
  selection outline map their handles through. Documents never contain slots, so undo after a compaction just
  takes new slots.
* **Operations** (`editor/ops.rs`): `actor_transform` (world; actors whose ancestor is listed move with it),
  `actor_local`, `actor_visibility`, `rename`, `reparent` (keeps the world transform, refuses cycles), `group`
  (new actor at the centre, common parent kept), `ungroup`, `component_transform` / `component_local` /
  `component_visibility` / `component_light`, `assign_material` and `cast_shadow` for components or actors,
  `delete` / `duplicate` of actors with their subtrees, `place` (a prefab becomes one actor: tree = trunk +
  foliage). Instance- and light-index operations remain (native editor, older scripts) and map to the component or
  the single-component actor.
* **Migration** of version-1 files happens in `Editor::load` against the base actors (built-index → actor and
  component tables).
* **UI** (`web/editor/*`): the outliner is the actor tree (expand / collapse, Ctrl / Shift selection, HTML5 drag onto
  an actor to attach, onto *Level* to detach, eye per actor, F2 / double-click rename, search shows matches with
  their ancestors); a viewport click selects the actor, Alt+click the component; Details shows name, parent (with
  *Detach*), key, relative transform, the component list (click selects a component, eye toggles it), the mesh /
  material sections of a one-mesh actor and the light of a one-light actor; Ctrl+G / Shift+G group / ungroup.
  The engine side: `heliosEditorActor`, `heliosEditorSelect`, `heliosEditorPreviewActors`,
  `heliosEditorPreviewComponents`, `heliosEditorCompact`.

## Alternatives considered

* *Actors in `helios_render::Scene`.* The renderer needs instances, not authoring structure; keeping actors in
  the demo crate leaves the renderer's data model alone (its only change: `Scene::compact` and the layout version).
* *Explicit actor lists returned by every builder.* Would change every builder's signature and the mesh-library
  recording path; the scoped recorder adds a few lines per builder and keeps tests that build scenes directly working.
* *Build-order keys with a migration table.* Still breaks when a builder inserts objects; position keys break only
  for the moved object.
* *Compaction on every delete.* Each compaction is a structural change; deleting several things in a row would pay
  it every time. Free lists plus idle-time compaction pay it once, and only when it frees enough.
* *Keeping dragged instances movable forever.* A movable instance is redrawn into every page it touches each time
  those pages refresh; restoring `Static` after the drag returns it to the cached layer.

## Measurements

RTX 3090 shared with other agents' builds and renders. Native runs stalled for minutes at their first frame for
every binary (main's too) while the GPU was saturated by others, so the A/B was taken in the browser (headless
Chromium, WebGPU/D3D12, `web/editor.html`, 1600×900 page, dynamic resolution at 100 %): the actor nearest the start
view is dragged through `heliosEditorPreviewActors` on a 1 m circle (one turn per 2 s) for 180 frames, sampling the
HUD over the last 120; `heliosEditorMovableDrags(false)` gives the path before actors (static instances). Three
interleaved rounds of none / movable / static; GPU = minimum over rounds of the per-round median.

| Drag | GPU ms | shadow pages rendered / frame (dynamic-casters-only) | shadow draws / frame |
|---|---|---|---|
| Forest, *Campfire* (18 instances + light + fire): no drag | 9.32 | 96 (96) | 11–18 k |
| — movable while dragged (actors) | **10.95** | 123 (95) | 31–33 k |
| — static (before) | 13.93 | 214 (95) | 109 k |
| Cathedral, *Pew* (1 instance): no drag | 6.93 | 43 (41) | 1.8–1.9 k |
| — movable while dragged (actors) | **7.33** | 97 (95) | 2.0–2.4 k |
| — static (before) | 9.12 | 174 (28) | 23 k |

A static mover invalidates the cached pages under its old and new bounds every frame and re-renders them with
everything in them (the forest's canopy over the clearing: 109 k draws); as a movable it is redrawn over the cached
static layer of those pages (the pages count, the draws are its own). The forest's ~96 dynamic-only pages are the
swaying foliage, present without a drag. Native, before the GPU was saturated (one run each, forest, 240 frames):
movable 121 pages / 38.9 k draws, static 169 pages / 114 k draws.

Editing and loading (wasm, forest medium, 6 218 actors over 8 278 instances): one edit moving all 1 500 trees
33 ms (undo 10 ms, redo 11 ms); moving one actor 4 ms; the state JSON for the outliner 5.4 ms (166 KB); loading a
file of 1 501 actor edits (531 KB) 9.5–12.5 ms after a reset to the base (6–9 ms); the same tree moves as a
version-1 file of 3 000 instance edits, migrated on load, 16–19 ms; the 2-edit file from the UI test 10–16 ms.
Compacting 80 free slots (40 deleted trees) 7–10 ms of CPU plus one structural change; undo after it restores all.
Browser UI test (`pw_editor.py`, Playwright): editor ready 6.7 s after navigation; a gizmo X-arrow drag moved the
campfire 1 m with its 18 instances movable during the drag and 0 after it settled; drag-and-drop re-parenting onto
*Camp lantern* kept the world position; Ctrl+Z / Ctrl+Y restored parent and position; Ctrl+G / Shift+G grouped and
ungrouped; Ctrl+S saved a version-2 file (989 bytes, keys `Campfire@700,-150,-100`), which reloaded to the same
place; a version-1 file migrated without warnings; no console errors. Screenshots: `editor-hierarchy/browser-*.jpg`.

## Remaining limits

See `docs/status.md` item 12.
