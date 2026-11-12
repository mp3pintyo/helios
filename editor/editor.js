// Helios editor: an Unreal-style level editor around the live WebGPU viewport.
//
// The engine owns the scene; this UI talks to it through the wasm page API (`heliosEditor*`,
// see crates/helios-demo/src/web_editor.rs): it reads the scene as JSON, sends edit operations
// (each one undo step on the engine side), previews gizmo drags, picks through the visibility
// buffer and draws its overlay (gizmo, light icons, PCG volumes, selection bounds) from the
// engine's camera matrices. `mountEditor()` builds the UI around an existing #helios-canvas,
// so the demo page can switch into it with Tab and back.

import * as engine from "../pkg/helios_demo.js";
import { h, icon, button, select } from "./ui.js";
import { View, add, sub, scale, quatMul, quatRotate, quatIdentity, quatAxisAngle, mat4Compose, mat4TransformPoint, DEG, normalize } from "./math.js";
import { Gizmo } from "./gizmo.js";
import { createOutliner } from "./outliner.js";
import { createDetails } from "./details.js";
import { createContentBrowser } from "./content.js";
import { createGraphEditor } from "./graph.js";

export const DEFAULT_LIGHT = { shape: "point", position: [0, 0, 0], direction: [0, -1, 0], color: [1, 0.82, 0.62], intensity: 5, range: 12, source_radius: 0.05, cast_shadows: true, inner_deg: 25, outer_deg: 35, enabled: true };

class Editor {
  constructor() {
    this.state = null;
    this.revision = -1;
    this.selection = [];
    this.gizmo = new Gizmo();
    this.snapping = { grid: 0.5, angle: 15, scale: 0.1, gridOn: true, angleOn: true, scaleOn: false };
    this.instanceCache = new Map();
    this.actorCache = new Map();
    this.mounted = false;
    this.visible = false;
    this.pending = new Map();
    this.picks = new Set();
    this.dragCounter = 0;
  }

  // ---- Engine access ---------------------------------------------------------------------

  // quiet: a live edit (slider, scrub) that must not rebuild the panel being dragged.
  apply(op, { merge = null, quiet = false } = {}) {
    if (merge) op.merge = merge;
    this.liveEdit = quiet;
    const result = JSON.parse(engine.heliosEditorApply(JSON.stringify(op)));
    if (result.error) {
      this.status(result.error, "error");
      return null;
    }
    if (!quiet) this.refresh();
    return result.created;
  }

  // Applies the latest op for `key` once per animation frame (live sliders: the PCG graph
  // regenerates at most once a frame however fast the input events come).
  applyLater(key, op, merge) {
    this.pending.set(key, { op, merge });
  }

  flushPending() {
    if (!this.pending.size) return;
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const { op, merge } of pending) this.apply(op, { merge, quiet: true });
    this.refresh({ keepDetails: true });
  }

  refresh({ keepDetails = false } = {}) {
    const revision = engine.heliosEditorRevision();
    if (revision === this.revision && this.state) return;
    this.revision = revision;
    this.state = JSON.parse(engine.heliosEditorState());
    this.instanceCache.clear();
    this.actorCache.clear();
    this.indexActors();
    // Drop selected things that no longer exist (undo of an add, deletes).
    this.selection = this.selection.filter((ref) => this.exists(ref));
    this.outliner.update();
    if (!keepDetails) this.details.render();
    this.content.update();
    this.graphEditor.update();
    this.toolbarState();
    this.sendSelection();
    for (const w of this.state.warnings || []) this.status(w, "warn");
  }

  // Actor lookups from the state's parallel arrays: index by id, children by parent.
  indexActors() {
    const a = this.state.actors;
    this.actorIndex = new Map();
    this.actorChildren = new Map();
    for (let k = 0; k < a.id.length; k++) {
      this.actorIndex.set(a.id[k], k);
      const p = a.parent[k];
      if (!this.actorChildren.has(p)) this.actorChildren.set(p, []);
      this.actorChildren.get(p).push(a.id[k]);
    }
  }

  actorName(id) {
    const k = this.actorIndex?.get(id);
    return k === undefined ? "?" : this.state.actors.name[k];
  }

  actorParent(id) {
    const k = this.actorIndex?.get(id);
    return k === undefined ? -1 : this.state.actors.parent[k];
  }

  // Details of one actor (transforms, bounds, components), cached until the next refresh.
  actor(id) {
    if (!this.actorCache.has(id)) this.actorCache.set(id, JSON.parse(engine.heliosEditorActor(id) || "null"));
    return this.actorCache.get(id);
  }

  component(ref) {
    const { actor, component } = parseComponent(ref);
    return this.actor(actor)?.components.find((c) => c.name === component) || null;
  }

  // Outlines the selected actors (with their children) and components in the viewport.
  sendSelection() {
    const components = this.selection.filter((r) => r.startsWith("c:")).map((r) => {
      const c = parseComponent(r);
      return [c.actor, c.component];
    });
    engine.heliosEditorSelect(JSON.stringify({ actors: this.selectedIds("a"), components }));
  }

  selectedComponents() {
    return this.selection.filter((r) => r.startsWith("c:")).map(parseComponent);
  }

  exists(ref) {
    const [kind, id] = splitRef(ref);
    const s = this.state;
    if (kind === "a") return this.actorIndex.has(id);
    if (kind === "c") return !!this.component(ref);
    if (kind === "p") return s.pcg.some((v) => v.def.id === id);
    if (kind === "m") return s.materials.some((m) => m.name === id) || s.content.materials.includes(id);
    return true;
  }

  instance(i) {
    if (!this.instanceCache.has(i)) this.instanceCache.set(i, JSON.parse(engine.heliosEditorInstance(i) || "null"));
    return this.instanceCache.get(i);
  }

  volume(id) {
    return this.state.pcg.find((v) => v.def.id === id);
  }

  // ---- Selection --------------------------------------------------------------------------

  // mode: "set" | "add" | "toggle"
  select(refs, mode = "set") {
    refs = refs.filter(Boolean);
    if (mode === "set") this.selection = [...new Set(refs)];
    else if (mode === "add") this.selection = [...new Set([...this.selection.filter((r) => !refs.includes(r)), ...refs])];
    else for (const r of refs) this.selection = this.selection.includes(r) ? this.selection.filter((x) => x !== r) : [...this.selection, r];
    this.sendSelection();
    this.outliner.update();
    this.details.render();
    this.footerSelection();
  }

  get primary() {
    return this.selection[this.selection.length - 1] || null;
  }

  selectedIds(kind) {
    return this.selection.filter((r) => r.startsWith(kind + ":")).map((r) => splitRef(r)[1]);
  }

  // Transformable targets of the selection: { ref, t, r, s }.
  targets() {
    const out = [];
    for (const ref of this.selection) {
      const [kind, id] = splitRef(ref);
      if (kind === "a") {
        const info = this.actor(id);
        if (info) out.push({ ref, ...info.world });
      } else if (kind === "c") {
        const c = this.component(ref);
        if (c) out.push({ ref, ...c.world });
      } else if (kind === "p") {
        const v = this.volume(id);
        if (v) {
          const vol = v.def.volume;
          out.push({ ref, t: vol.center, r: quatAxisAngle([0, 1, 0], (vol.yaw_deg || 0) * DEG), s: vol.half_extents.map((x) => x * 2) });
        }
      }
    }
    return out;
  }

  // ---- Gizmo drags ------------------------------------------------------------------------

  beginDrag(handle, mouse) {
    const targets = this.targets();
    if (!targets.length) return false;
    const primary = targets[targets.length - 1];
    this.gizmo.begin(handle, this.view, primary.t, primary.r, mouse);
    this.drag = { targets, pivot: primary.t, rotation: primary.r, key: `drag-${++this.dragCounter}`, result: null };
    return true;
  }

  updateDrag(mouse) {
    const d = this.drag;
    const snapping = { grid: this.snapping.gridOn ? this.snapping.grid : 0, angle: this.snapping.angleOn ? this.snapping.angle : 0, scale: this.snapping.scaleOn ? this.snapping.scale : 0 };
    const delta = this.gizmo.update(this.view, mouse, snapping);
    if (!delta) return;
    d.result = d.targets.map((t) => {
      let { t: pos, r: rot, s: size } = t;
      if (delta.translate) pos = add(pos, delta.translate);
      if (delta.rotate) {
        rot = quatMul(delta.rotate, rot);
        pos = add(d.pivot, quatRotate(delta.rotate, sub(pos, d.pivot)));
      }
      if (delta.scale) {
        size = size.map((v, i) => v * delta.scale[i]);
        // Offsets from the pivot scale along the pivot's axes.
        const inverse = [-d.rotation[0], -d.rotation[1], -d.rotation[2], d.rotation[3]];
        const offset = quatRotate(inverse, sub(pos, d.pivot)).map((v, i) => v * delta.scale[i]);
        pos = add(d.pivot, quatRotate(d.rotation, offset));
      }
      return { ref: t.ref, t: pos, r: rot, s: size };
    });
    this.previewDrag();
  }

  // Moves the dragged actors (children follow) and components in the engine, without an undo
  // step; their instances are movable until the drag settles.
  previewDrag() {
    const results = this.drag.result || [];
    const actors = results.filter((r) => r.ref.startsWith("a:"));
    if (actors.length) {
      const flat = new Float32Array(actors.length * 10);
      actors.forEach((r, k) => flat.set([...r.t, ...r.r, ...r.s], k * 10));
      engine.heliosEditorPreviewActors(new Uint32Array(actors.map((r) => splitRef(r.ref)[1])), flat);
      for (const r of actors) {
        const info = this.actor(splitRef(r.ref)[1]);
        if (info) info.preview = r;
      }
    }
    const components = results.filter((r) => r.ref.startsWith("c:"));
    if (components.length) {
      engine.heliosEditorPreviewComponents(JSON.stringify(components.map((r) => ({ ...parseComponent(r.ref), t: r.t, r: r.r, s: r.s }))));
      for (const r of components) {
        const c = this.component(r.ref);
        if (c) c.preview = r;
      }
    }
  }

  endDrag(cancel = false) {
    const d = this.drag;
    this.drag = null;
    this.gizmo.end();
    if (!d) return;
    const results = d.result || [];
    const actors = results.filter((r) => r.ref.startsWith("a:"));
    const components = results.filter((r) => r.ref.startsWith("c:"));
    if (cancel) {
      engine.heliosEditorRevertPreview(new Uint32Array(0));
      this.actorCache.clear();
      this.refresh();
      return;
    }
    const trs = (r) => ({ t: r.t, r: r.r, s: r.s });
    if (actors.length) this.apply({ op: "actor_transform", actors: actors.map((r) => splitRef(r.ref)[1]), transforms: actors.map(trs) }, { merge: d.key, quiet: true });
    if (components.length) this.apply({ op: "component_transform", components: components.map((r) => parseComponent(r.ref)), transforms: components.map(trs) }, { merge: d.key, quiet: true });
    for (const r of results.filter((r) => r.ref.startsWith("p:"))) {
      const v = this.volume(splitRef(r.ref)[1]);
      if (!v) continue;
      const def = structuredClone(v.def);
      def.volume.center = r.t;
      def.volume.half_extents = r.s.map((x) => Math.max(0.5, x / 2));
      def.volume.yaw_deg = (Math.atan2(2 * (r.r[3] * r.r[1] + r.r[0] * r.r[2]), 1 - 2 * (r.r[1] * r.r[1] + r.r[2] * r.r[2])) / DEG) % 360;
      this.apply({ op: "pcg", def }, { merge: d.key, quiet: true });
    }
    this.refresh();
  }

  // ---- Commands -----------------------------------------------------------------------------

  deleteSelection() {
    if (!this.selection.length) return;
    const n = this.selection.length;
    this.apply({ op: "delete", actors: this.selectedIds("a"), components: this.selectedComponents(), pcg: this.selectedIds("p") });
    this.select([]);
    this.status(`Deleted ${n} object${n > 1 ? "s" : ""}`);
  }

  duplicateSelection() {
    // A selected component duplicates its actor.
    const actors = [...new Set([...this.selectedIds("a"), ...this.selectedComponents().map((c) => c.actor)])], pcg = this.selectedIds("p");
    if (!actors.length && !pcg.length) return;
    const step = this.snapping.gridOn ? Math.max(this.snapping.grid, 0.5) : 1;
    const created = this.apply({ op: "duplicate", actors, pcg, offset: [step, 0, 0] });
    if (created) this.select([...created.actors.map((a) => "a:" + a), ...created.pcg.map((p) => "p:" + p)]);
  }

  // Ctrl+G: a new actor at the selection's centre with the selected actors as children.
  groupSelection() {
    const actors = this.selectedIds("a");
    if (!actors.length) return;
    const created = this.apply({ op: "group", actors });
    if (created) {
      this.select(created.actors.map((a) => "a:" + a));
      this.status(`Grouped ${actors.length} actor${actors.length > 1 ? "s" : ""}`);
    }
  }

  // Shift+G: the children of the selected actors move up a level; empty groups go.
  ungroupSelection() {
    const actors = this.selectedIds("a");
    if (!actors.length) return;
    const children = actors.flatMap((a) => this.actorChildren.get(a) || []);
    if (this.apply({ op: "ungroup", actors })) this.select(children.map((c) => "a:" + c));
  }

  // Attaches actors to `parent` (null: the world root), keeping where they are.
  reparent(actors, parent) {
    if (!actors.length) return;
    if (this.apply({ op: "reparent", actors, parent })) this.status(parent === null ? "Detached" : `Attached to ${this.actorName(parent)}`);
  }

  rename(actor, name) {
    if (name && name !== this.actorName(actor)) this.apply({ op: "rename", actor, name });
  }

  undo() {
    if (engine.heliosEditorUndo()) this.status("Undo");
    this.instanceCache.clear();
    this.actorCache.clear();
    this.refresh();
  }

  redo() {
    if (engine.heliosEditorRedo()) this.status("Redo");
    this.instanceCache.clear();
    this.actorCache.clear();
    this.refresh();
  }

  // World bounds of a reference ([lo, hi]).
  bounds(ref) {
    const [kind, id] = splitRef(ref);
    if (kind === "a") return this.actor(id)?.bounds;
    if (kind === "c") {
      const c = this.component(ref);
      if (!c) return null;
      if (c.instance !== undefined && c.instance !== null) return this.instance(c.instance)?.bounds;
      const r = c.light ? Math.max(1, c.light.range * 0.3) : 0.5;
      return [sub(c.world.t, [r, r, r]), add(c.world.t, [r, r, r])];
    }
    if (kind === "p") {
      const v = this.volume(id)?.def.volume;
      if (!v) return null;
      const e = Math.hypot(v.half_extents[0], v.half_extents[2]);
      return [sub(v.center, [e, v.half_extents[1], e]), add(v.center, [e, v.half_extents[1], e])];
    }
    return null;
  }

  focusSelection() {
    const boxes = this.selection.map((r) => this.bounds(r)).filter(Boolean);
    if (!boxes.length) return;
    const lo = boxes.reduce((a, b) => a.map((v, i) => Math.min(v, b[0][i])), [Infinity, Infinity, Infinity]);
    const hi = boxes.reduce((a, b) => a.map((v, i) => Math.max(v, b[1][i])), [-Infinity, -Infinity, -Infinity]);
    engine.heliosEditorFocus(new Float32Array(lo), new Float32Array(hi));
  }

  // Where to put new things under canvas point (x, y), by default the viewport's centre: the
  // surface a GPU pick finds there, else the ground, else 10 m ahead of the camera.
  placementPoint(x, y) {
    const rect = this.canvas.getBoundingClientRect();
    const px = x ?? rect.width / 2, py = y ?? rect.height / 2;
    const near = (p) => p && p.length === 3 && this.view && Math.hypot(...sub([...p], this.view.eye)) < 400;
    const fallback = () => {
      const ground = engine.heliosEditorGroundAt(px, py);
      if (near(ground)) return [...ground];
      return this.view ? add(this.view.eye, scale(this.view.ray(px, py).dir, 10)) : [0, 0, 0];
    };
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(fallback()), 600);
      const done = (result) => {
        clearTimeout(timer);
        resolve(result.instance !== null && near(result.position) ? result.position : near(result.ground) ? result.ground : fallback());
      };
      this.picks.add({ id: engine.heliosEditorPick(px, py), resolve: done });
    });
  }

  place(item, at) {
    const created = this.apply({ op: "place", item, transform: { t: at, r: [0, 0, 0, 1], s: [1, 1, 1] } });
    if (created) {
      this.select(created.actors.map((a) => "a:" + a));
      this.status(`Placed ${item}`);
    }
  }

  addLight(shape, at) {
    const def = { ...DEFAULT_LIGHT, shape, position: add(at, [0, shape === "spot" ? 4 : 2.5, 0]) };
    const created = this.apply({ op: "add_light", def });
    if (created) this.select(created.actors.map((a) => "a:" + a));
  }

  addPcg(preset, at) {
    this.status("Generating…");
    setTimeout(() => {
      const created = this.apply({ op: "add_pcg", preset, center: at });
      if (created) {
        this.select(created.pcg.map((p) => "p:" + p));
        const v = this.volume(created.pcg[0]);
        this.status(`${v?.def.name}: ${v?.instances ?? 0} instances`);
      }
    }, 30);
  }

  // ---- Files ----------------------------------------------------------------------------------

  save() {
    const json = engine.heliosEditorSave();
    const base = this.state?.base;
    const name = (base?.scene || base?.url?.split("/").pop()?.split(".")[0] || "scene") + ".helios.json";
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = h("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    this.lastSaved = json;
    this.status(`Saved ${name} (${(json.length / 1024).toFixed(1)} KiB)`);
  }

  async load(text, label = "scene file") {
    this.status(`Loading ${label}…`);
    this.busy(true, `Loading ${label}…`);
    await new Promise((r) => setTimeout(r, 30));
    try {
      await engine.heliosEditorLoad(text);
      this.select([]);
      this.instanceCache.clear();
      this.revision = -1;
      this.refresh();
      this.status(`Loaded ${label}`);
    } catch (e) {
      this.status(`Could not load ${label}: ${e}`, "error");
    } finally {
      this.busy(false);
    }
  }

  openFile() {
    const input = h("input", { type: "file", accept: ".json,.helios.json,application/json" });
    input.addEventListener("change", async () => {
      const file = input.files[0];
      if (file) await this.load(await file.text(), file.name);
    });
    input.click();
  }

  async newScene(sceneName) {
    const base = sceneName ? { kind: "procedural", scene: sceneName, detail: this.state?.base?.detail || "medium" } : this.state.base;
    await this.load(JSON.stringify({ format: "helios-scene", version: 2, base }), sceneName || "new scene");
  }

  async importModel(url, bytes, at) {
    this.busy(true, `Importing ${url}…`);
    await new Promise((r) => setTimeout(r, 30));
    try {
      const item = await engine.heliosEditorImport(url, bytes || undefined);
      this.revision = -1;
      this.refresh();
      this.place(item, at || (await this.placementPoint()));
      this.status(`Imported ${item}`);
    } catch (e) {
      this.status(`Import failed: ${e}`, "error");
    } finally {
      this.busy(false);
    }
  }

  // ---- Chrome -----------------------------------------------------------------------------

  status(message, kind = "info") {
    if (!this.statusEl) return;
    this.statusEl.textContent = message;
    this.statusEl.className = "ed-status-message " + kind;
    clearTimeout(this.statusTimer);
    this.statusTimer = setTimeout(() => (this.statusEl.textContent = ""), kind === "error" ? 9000 : 4000);
    if (kind !== "info") console[kind === "error" ? "error" : "warn"]("[editor]", message);
  }

  busy(on, text = "") {
    this.busyEl.textContent = text;
    this.busyEl.classList.toggle("on", on);
  }

  footerSelection() {
    const n = this.selection.length;
    this.selectionEl.textContent = n ? `${n} selected` : "Nothing selected";
  }

  toolbarState() {
    if (!this.state) return;
    this.undoButton.disabled = !this.state.can_undo;
    this.redoButton.disabled = !this.state.can_redo;
    for (const [mode, b] of Object.entries(this.modeButtons)) b.classList.toggle("active", this.gizmo.mode === mode);
    this.spaceButton.replaceChildren(icon(this.gizmo.space === "world" ? "world" : "local"), h("span", {}, this.gizmo.space === "world" ? "World" : "Local"));
    for (const [k, b] of Object.entries(this.snapButtons)) b.classList.toggle("active", this.snapping[k + "On"]);
    const base = this.state.base;
    this.sceneLabel.textContent = base.kind === "procedural" ? `${base.scene} · ${base.detail}` : base.url;
  }

  setMode(mode) {
    this.gizmo.mode = mode;
    this.toolbarState();
  }

  buildToolbar() {
    const snapButton = (key, label, values, unit) => {
      const toggle = button("", () => {
        this.snapping[key + "On"] = !this.snapping[key + "On"];
        this.toolbarState();
      }, { iconName: key === "grid" ? "grid" : key === "angle" ? "rotate" : "scale", cls: "ed-snap", title: `${label} snapping` });
      const values_ = select(values.map((v) => [String(v), `${v}${unit}`]), String(this.snapping[key]), (v) => (this.snapping[key] = parseFloat(v)));
      return [toggle, h("div", { class: "ed-snap-group" }, toggle, values_)];
    };
    const [gridToggle, gridGroup] = snapButton("grid", "Grid", [0.1, 0.25, 0.5, 1, 2, 5, 10], " m");
    const [angleToggle, angleGroup] = snapButton("angle", "Rotation", [5, 10, 15, 30, 45, 90], "°");
    const [scaleToggle, scaleGroup] = snapButton("scale", "Scale", [0.05, 0.1, 0.25, 0.5], "");
    this.snapButtons = { grid: gridToggle, angle: angleToggle, scale: scaleToggle };
    this.modeButtons = {
      translate: button("", () => this.setMode("translate"), { iconName: "move", cls: "ed-tool", title: "Move (W)" }),
      rotate: button("", () => this.setMode("rotate"), { iconName: "rotate", cls: "ed-tool", title: "Rotate (E)" }),
      scale: button("", () => this.setMode("scale"), { iconName: "scale", cls: "ed-tool", title: "Scale (R)" }),
    };
    this.spaceButton = button("World", () => {
      this.gizmo.space = this.gizmo.space === "world" ? "local" : "world";
      this.toolbarState();
    }, { iconName: "world", cls: "ed-space", title: "World / local axes (Ctrl+`)" });
    this.undoButton = button("", () => this.undo(), { iconName: "undo", title: "Undo (Ctrl+Z)" });
    this.redoButton = button("", () => this.redo(), { iconName: "redo", title: "Redo (Ctrl+Y)" });
    this.sceneLabel = h("span", { class: "ed-scene-label" });
    const scenes = select([["", "Switch scene…"], ["valley", "Valley"], ["cathedral", "Cathedral"], ["forest", "Forest"], ["gallery", "Material gallery"]], "", async (v) => {
      if (v && confirm(`Load the ${v} scene? Unsaved edits are lost.`)) await this.newScene(v);
    });
    return h(
      "header",
      { class: "ed-toolbar" },
      h("div", { class: "ed-brand" }, h("span", { class: "ed-logo" }), h("span", {}, "HELIOS"), h("span", { class: "ed-brand-sub" }, "Editor")),
      h("div", { class: "ed-group" },
        button("", () => this.newScene(), { iconName: "file", title: "New: discard the edits of this scene" }),
        button("", () => this.openFile(), { iconName: "open", title: "Open a scene file (Ctrl+O)" }),
        button("", () => this.save(), { iconName: "save", title: "Save the scene file (Ctrl+S)" })),
      h("div", { class: "ed-group" }, this.undoButton, this.redoButton),
      h("div", { class: "ed-group" }, Object.values(this.modeButtons), this.spaceButton),
      h("div", { class: "ed-group" }, gridGroup, angleGroup, scaleGroup),
      h("div", { class: "ed-group" }, button("Material Graph", () => (this.graphEditor.visible ? this.graphEditor.close() : this.graphEditor.open()), { iconName: "material", title: "Material graph editor" })),
      h("div", { class: "ed-spacer" }),
      h("div", { class: "ed-group" }, this.sceneLabel, scenes),
      button("Play", () => toggleEditor(), { iconName: "play", cls: "ed-play", title: "Leave the editor (Tab)" }),
    );
  }

  // ---- Mount ------------------------------------------------------------------------------

  mount() {
    if (!document.querySelector("link[data-helios-editor]")) {
      document.head.append(h("link", { rel: "stylesheet", href: new URL("./editor.css", import.meta.url).href, "data-helios-editor": true }));
    }
    if (!document.getElementById("ed-play-hint")) document.body.append(h("div", { id: "ed-play-hint" }, "Tab — back to the editor"));
    this.canvas = document.getElementById("helios-canvas");
    this.overlay = h("canvas", { class: "ed-overlay" });
    this.viewportStats = h("div", { class: "ed-viewport-stats" });
    this.busyEl = h("div", { class: "ed-busy" });
    this.viewport = h("main", { class: "ed-viewport", tabindex: "-1" }, this.overlay, this.viewportStats, this.busyEl,
      h("div", { class: "ed-viewport-hint" }, "LMB select actor · Alt+LMB component · RMB + WASD/QE fly · wheel dolly · W/E/R move/rotate/scale · F focus · Del delete · Ctrl+D duplicate · Ctrl+G / Shift+G group / ungroup · F2 rename · Tab play"));
    this.statusEl = h("span", { class: "ed-status-message" });
    this.selectionEl = h("span", { class: "ed-status-selection" });
    this.gpuEl = h("span", { class: "ed-status-gpu" });
    this.outliner = createOutliner(this);
    this.details = createDetails(this);
    this.content = createContentBrowser(this);
    this.graphEditor = createGraphEditor(this);
    this.viewport.append(this.graphEditor.element);
    this.root = h("div", { class: "ed-root" },
      this.buildToolbar(),
      h("aside", { class: "ed-left" }, this.content.element),
      this.viewport,
      h("aside", { class: "ed-right" }, this.outliner.element, this.details.element),
      h("footer", { class: "ed-statusbar" }, this.statusEl, h("span", { class: "ed-spacer" }), this.selectionEl, this.gpuEl));
    document.body.append(this.root);
    this.mounted = true;
    this.installViewport();
    this.installKeyboard();
    requestAnimationFrame(() => this.frame());
  }

  show(on) {
    this.visible = on;
    document.body.classList.toggle("ed-active", on);
    this.root.classList.toggle("hidden", !on);
    if (on) {
      this.viewport.prepend(this.canvas);
      engine.heliosSetEditorMode(true);
      this.revision = -1;
      this.refresh();
      this.viewport.focus();
    } else {
      document.body.prepend(this.canvas);
      engine.heliosSetEditorMode(false);
    }
  }

  // ---- Viewport input -------------------------------------------------------------------------

  mouse(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  installViewport() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      if (!this.visible) return;
      this.viewport.focus();
      if (e.button === 2) {
        this.looking = true;
        return;
      }
      if (e.button !== 0) return;
      const m = this.mouse(e);
      if (this.view && this.gizmo.hover && this.beginDrag(this.gizmo.hover, m)) {
        c.setPointerCapture(e.pointerId);
        return;
      }
      this.click = { m, shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, alt: e.altKey };
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.visible || !this.view) return;
      const m = this.mouse(e);
      if (this.drag) {
        this.updateDrag(m);
        return;
      }
      if (this.looking) return;
      const targets = this.targets();
      const primary = targets[targets.length - 1];
      this.gizmo.hover = primary ? this.gizmo.hit(this.view, primary.t, primary.r, m) : null;
      c.style.cursor = this.gizmo.hover ? "pointer" : "default";
    });
    c.addEventListener("pointerup", (e) => {
      if (e.button === 2) this.looking = false;
      if (!this.visible) return;
      if (this.drag) {
        this.endDrag();
        return;
      }
      const click = this.click;
      this.click = null;
      if (!click || e.button !== 0) return;
      const m = this.mouse(e);
      if (Math.hypot(m[0] - click.m[0], m[1] - click.m[1]) > 4) return;
      const mode = click.ctrl ? "toggle" : click.shift ? "add" : "set";
      const icon = this.iconAt(m);
      if (icon) {
        this.select([icon], mode);
        return;
      }
      this.picks.add({ id: engine.heliosEditorPick(m[0], m[1]), mode, component: click.alt });
    });
    // Drag and drop: content items and .glb files.
    this.viewport.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    });
    this.viewport.addEventListener("drop", (e) => {
      e.preventDefault();
      // The drop's data is only readable during the event.
      const file = [...(e.dataTransfer.files || [])].find((f) => /\.glb$/i.test(f.name));
      const data = e.dataTransfer.getData("application/x-helios");
      this.drop(this.mouse(e), file, data ? JSON.parse(data) : null);
    });
  }

  async drop(m, file, item) {
    if (item?.kind === "material") {
      this.picks.add({ id: engine.heliosEditorPick(m[0], m[1]), assign: item.id });
      return;
    }
    if (!file && !item) return;
    const at = await this.placementPoint(m[0], m[1]);
    if (file) this.importModel(file.name, new Uint8Array(await file.arrayBuffer()), at);
    else if (item.kind === "item") this.place(item.id, at);
    else if (item.kind === "light") this.addLight(item.id, at);
    else if (item.kind === "pcg") this.addPcg(item.id, at);
  }

  installKeyboard() {
    window.addEventListener("keydown", (e) => {
      if (!this.mounted) return;
      const typing = e.target.closest && e.target.closest("input, textarea, select");
      if (e.code === "Tab" && !typing) {
        e.preventDefault();
        toggleEditor();
        return;
      }
      if (!this.visible || typing || this.looking) return;
      const ctrl = e.ctrlKey || e.metaKey;
      const handled = () => e.preventDefault();
      if (ctrl) {
        if (e.code === "KeyZ" && !e.shiftKey) this.undo();
        else if (e.code === "KeyY" || (e.code === "KeyZ" && e.shiftKey)) this.redo();
        else if (e.code === "KeyS") this.save();
        else if (e.code === "KeyO") this.openFile();
        else if (e.code === "KeyD") this.duplicateSelection();
        else if (e.code === "KeyG") this.groupSelection();
        else if (e.code === "Backquote") {
          this.gizmo.space = this.gizmo.space === "world" ? "local" : "world";
          this.toolbarState();
        } else return;
        return handled();
      }
      switch (e.code) {
        case "KeyW": this.setMode("translate"); break;
        case "KeyE": this.setMode("rotate"); break;
        case "KeyR": this.setMode("scale"); break;
        case "Space": this.setMode({ translate: "rotate", rotate: "scale", scale: "translate" }[this.gizmo.mode]); break;
        case "KeyF": this.focusSelection(); break;
        case "KeyG": if (e.shiftKey) this.ungroupSelection(); else return; break;
        case "F2": if (this.primary?.startsWith("a:")) this.outliner.rename(splitRef(this.primary)[1]); else return; break;
        case "Delete": case "Backspace": this.deleteSelection(); break;
        case "Escape":
          if (this.drag) this.endDrag(true);
          else this.select([]);
          break;
        default: return;
      }
      handled();
    });
  }

  // ---- Overlay ------------------------------------------------------------------------------

  iconAt(m) {
    for (const hit of this.iconHits || []) if (Math.hypot(m[0] - hit.p[0], m[1] - hit.p[1]) < hit.r) return hit.ref;
    return null;
  }

  frame() {
    requestAnimationFrame(() => this.frame());
    if (!this.visible) return;
    this.flushPending();
    if (engine.heliosEditorRevision() !== this.revision) this.refresh({ keepDetails: !!this.drag || this.liveEdit });
    let text;
    while (this.picks.size && (text = engine.heliosEditorPickResult())) {
      const result = JSON.parse(text);
      for (const p of [...this.picks]) {
        // A newer request replaces older ones the engine had not run yet.
        if (p.id < result.id) this.picks.delete(p);
        if (p.id === result.id) {
          this.picks.delete(p);
          this.handlePick(result, p);
        }
      }
    }
    const cam = engine.heliosEditorCamera();
    if (cam.length < 25) return;
    this.view = new View(cam);
    const dpr = window.devicePixelRatio || 1;
    const w = this.viewport.clientWidth, hgt = this.viewport.clientHeight;
    if (this.overlay.width !== Math.round(w * dpr) || this.overlay.height !== Math.round(hgt * dpr)) {
      this.overlay.width = Math.round(w * dpr);
      this.overlay.height = Math.round(hgt * dpr);
    }
    const ctx = this.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    this.drawOverlay(ctx);
    this.statsTimer = (this.statsTimer || 0) + 1;
    if (this.statsTimer % 20 === 0) {
      const first = (engine.heliosHud() || "").split("\n")[0].split(" · ");
      this.viewportStats.textContent = first.slice(1, 3).join(" · ");
      this.gpuEl.textContent = first.slice(1, 4).join(" · ");
    }
  }

  handlePick(result, request) {
    if (request.resolve) {
      request.resolve(result);
      return;
    }
    if (request.assign) {
      if (result.owner?.kind === "actor") {
        this.apply({ op: "assign_material", components: [{ actor: result.owner.actor, component: result.owner.component }], material: request.assign });
        this.status(`Assigned ${request.assign}`);
      }
      return;
    }
    if (result.instance === null || result.instance === undefined) {
      if (request.mode === "set") this.select([]);
      return;
    }
    const o = result.owner;
    if (!o || o.kind === "deleted") return;
    // A click selects the actor; Alt+click the component under the cursor.
    const ref = o.kind === "pcg" ? "p:" + o.id : request.component ? componentRef(o.actor, o.component) : "a:" + o.actor;
    this.select([ref], request.mode);
  }

  drawOverlay(ctx) {
    const view = this.view;
    this.iconHits = [];
    const s = this.state;
    if (!s) return;
    const selected = new Set(this.selection);
    // PCG volumes: dashed green boxes with a label.
    for (const v of s.pcg) {
      const ref = "p:" + v.def.id;
      const preview = this.drag?.result?.find((r) => r.ref === ref);
      const vol = v.def.volume;
      const t = preview ? preview.t : vol.center;
      const r = preview ? preview.r : quatAxisAngle([0, 1, 0], (vol.yaw_deg || 0) * DEG);
      const size = preview ? preview.s : vol.half_extents.map((x) => x * 2);
      const on = selected.has(ref);
      drawBox(ctx, view, mat4Compose(t, r, size), [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], on ? "#9cf0b4" : "rgba(95, 211, 141, 0.75)", on ? 2 : 1.2, [6, 4]);
      const label = view.project(add(t, [0, size[1] / 2, 0]));
      if (label) {
        drawTag(ctx, label, `${v.def.name}  ·  ${v.instances}`, on ? "#2f8f55" : "rgba(26, 64, 42, 0.85)", v.error ? "#ff8a80" : "#d9fbe5");
        this.iconHits.push({ p: label, r: 14, ref });
      }
    }
    // Selected actors: their bounds with everything below them (moved with a drag's preview);
    // selected mesh components: oriented bounds. Shows translucent and occluded selections too.
    let boxes = 0;
    for (const ref of this.selection) {
      if (boxes++ > 200) break;
      if (ref.startsWith("a:")) {
        const info = this.actor(splitRef(ref)[1]);
        if (!info?.bounds) continue;
        const shift = info.preview ? sub(info.preview.t, info.world.t) : [0, 0, 0];
        drawBox(ctx, view, mat4Compose(shift, quatIdentity(), [1, 1, 1]), info.bounds[0], info.bounds[1], "rgba(255, 170, 60, 0.55)", 1, []);
      } else if (ref.startsWith("c:")) {
        const c = this.component(ref);
        if (!c?.local_bounds) continue;
        const tr = c.preview || c.world;
        drawBox(ctx, view, mat4Compose(tr.t, tr.r, tr.s), c.local_bounds[0], c.local_bounds[1], "rgba(255, 170, 60, 0.75)", 1.4, [4, 3]);
      }
    }
    // Lights: icons (and the cone of a selected spot light); a click selects the light's actor.
    for (const l of s.lights) {
      if (!l.def.enabled) continue;
      const ref = "a:" + l.actor;
      const p = view.project(l.def.position);
      if (!p) continue;
      const on = selected.has(ref) || selected.has(componentRef(l.actor, l.component));
      drawLightIcon(ctx, p, l.def, on);
      this.iconHits.push({ p, r: 13, ref });
      if (on) {
        if (l.def.shape === "spot") drawCone(ctx, view, l.def);
        drawRange(ctx, view, l.def);
      }
    }
    const targets = this.drag ? null : this.targets();
    const primary = this.drag ? { t: this.drag.pivot, r: this.drag.rotation } : targets[targets.length - 1];
    if (primary) {
      const pivot = this.drag?.result ? this.drag.result[this.drag.result.length - 1].t : primary.t;
      const rotation = this.drag?.result && this.gizmo.mode !== "rotate" ? this.drag.result[this.drag.result.length - 1].r : primary.r;
      this.gizmo.draw(ctx, view, pivot, rotation);
    }
  }
}

function splitRef(ref) {
  const i = ref.indexOf(":");
  if (i < 0) return [ref, null];
  const kind = ref.slice(0, i), id = ref.slice(i + 1);
  return [kind, kind === "m" || kind === "c" ? id : Number(id)];
}

// A component's reference: `c:ACTOR/NAME`.
function componentRef(actor, name) {
  return `c:${actor}/${name}`;
}

function parseComponent(ref) {
  const id = ref.slice(2);
  const slash = id.indexOf("/");
  return { actor: Number(id.slice(0, slash)), component: id.slice(slash + 1) };
}

function drawBox(ctx, view, m, lo, hi, color, width, dash) {
  const corners = [];
  for (let k = 0; k < 8; k++) {
    const p = [k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]];
    const w = mat4TransformPoint(m, p);
    corners.push(view.project([w[0], w[1], w[2]]));
  }
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  for (const [a, b] of edges) {
    if (!corners[a] || !corners[b]) continue;
    ctx.moveTo(corners[a][0], corners[a][1]);
    ctx.lineTo(corners[b][0], corners[b][1]);
  }
  ctx.stroke();
  ctx.restore();
}

function drawTag(ctx, p, text, background, color) {
  ctx.save();
  ctx.font = "600 11px 'Segoe UI', system-ui, sans-serif";
  const w = ctx.measureText(text).width + 14;
  ctx.fillStyle = background;
  roundRect(ctx, p[0] - w / 2, p[1] - 22, w, 18, 4);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(text, p[0], p[1] - 9);
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawLightIcon(ctx, p, def, on) {
  const peak = Math.max(...def.color, 1e-3);
  const rgb = def.color.map((c) => Math.round(Math.pow(c / peak, 1 / 2.2) * 255));
  ctx.save();
  ctx.beginPath();
  ctx.arc(p[0], p[1], 11, 0, Math.PI * 2);
  ctx.fillStyle = on ? "rgba(255, 166, 40, 0.95)" : "rgba(20, 22, 26, 0.78)";
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = on ? "#fff3d6" : "rgba(255,255,255,0.55)";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p[0], p[1] - 1.5, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = `rgb(${rgb.join(",")})`;
  ctx.fill();
  ctx.fillStyle = on ? "#3b2400" : "#d8dde6";
  ctx.fillRect(p[0] - 2.2, p[1] + 3.5, 4.4, 2.6);
  if (def.shape === "spot") {
    ctx.strokeStyle = on ? "#3b2400" : "#d8dde6";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(p[0] - 7, p[1] + 8);
    ctx.lineTo(p[0] + 7, p[1] + 8);
    ctx.stroke();
  }
  ctx.restore();
}

function drawCone(ctx, view, def) {
  const dir = normalize(def.direction);
  const helper = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = normalize([dir[1] * helper[2] - dir[2] * helper[1], dir[2] * helper[0] - dir[0] * helper[2], dir[0] * helper[1] - dir[1] * helper[0]]);
  const v = [dir[1] * u[2] - dir[2] * u[1], dir[2] * u[0] - dir[0] * u[2], dir[0] * u[1] - dir[1] * u[0]];
  const length = Math.min(def.range, 8);
  const radius = Math.tan(def.outer_deg * DEG) * length;
  const center = add(def.position, scale(dir, length));
  const apex = view.project(def.position);
  ctx.save();
  ctx.strokeStyle = "rgba(255, 196, 92, 0.8)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i <= 32; i++) {
    const a = (i / 32) * Math.PI * 2;
    const p = view.project(add(center, add(scale(u, Math.cos(a) * radius), scale(v, Math.sin(a) * radius))));
    if (!p) continue;
    if (first) ctx.moveTo(p[0], p[1]);
    else ctx.lineTo(p[0], p[1]);
    first = false;
    if (apex && i % 8 === 0) {
      ctx.moveTo(apex[0], apex[1]);
      ctx.lineTo(p[0], p[1]);
      ctx.moveTo(p[0], p[1]);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function drawRange(ctx, view, def) {
  ctx.save();
  ctx.strokeStyle = "rgba(255, 196, 92, 0.35)";
  ctx.setLineDash([4, 4]);
  for (const [a, b] of [[[1, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]]) {
    ctx.beginPath();
    let first = true;
    for (let i = 0; i <= 48; i++) {
      const t = (i / 48) * Math.PI * 2;
      const p = view.project(add(def.position, add(scale(a, Math.cos(t) * def.range), scale(b, Math.sin(t) * def.range))));
      if (!p) {
        first = true;
        continue;
      }
      if (first) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
      first = false;
    }
    ctx.stroke();
  }
  ctx.restore();
}

// ---- Public entry points ------------------------------------------------------------------------

let editor = null;

async function engineReady() {
  for (let i = 0; i < 6000; i++) {
    if (engine.heliosScene && engine.heliosScene()) return;
    await new Promise((r) => setTimeout(r, 50));
  }
}

// Builds the editor around the running engine and shows it.
export async function mountEditor() {
  await engineReady();
  if (!editor) {
    editor = new Editor();
    editor.mount();
    globalThis.heliosEditorMounted = true;
    // For the console and automated tests.
    globalThis.heliosEditor = editor;
  }
  editor.show(true);
  return editor;
}

// Tab: switches between the editor and the plain demo view.
export function toggleEditor() {
  if (!editor) return mountEditor();
  editor.show(!editor.visible);
  const hint = document.getElementById("ed-play-hint");
  if (hint) hint.classList.toggle("on", !editor.visible);
}

export { splitRef, componentRef, parseComponent };
