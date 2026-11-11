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

See the bottom of this note (filled in with the measurement runs).

## Remaining limits

See `docs/status.md` item 12.
