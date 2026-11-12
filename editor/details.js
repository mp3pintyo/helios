// Details panel: the properties of the primary selection — an instance (transform, material
// and its parameters, graph parameters and Substrate slab, rendering), a light, a PCG volume
// (volume, surface, seed and the graph's nodes), the environment (sun, fog, clouds, wind), the
// render settings (built from the engine's schema) or a material.

import { h, icon, section, row, numberField, vectorField, slider, checkbox, select, colorField, button } from "./ui.js";
import { quatFromEuler, quatToEuler } from "./math.js";
import { splitRef, componentRef, parseComponent } from "./editor.js";
import { materialParams } from "./graph.js";

export function createDetails(ed) {
  const title = h("span", { class: "ed-panel-title" }, "Details");
  const subtitle = h("span", { class: "ed-panel-count" });
  const body = h("div", { class: "ed-details-body" });
  const element = h("section", { class: "ed-panel ed-details" }, h("div", { class: "ed-panel-head" }, icon("gear"), title, subtitle), body);

  function render() {
    body.replaceChildren();
    subtitle.textContent = ed.selection.length > 1 ? `${ed.selection.length} selected` : "";
    const ref = ed.primary;
    if (!ref || !ed.state) {
      body.append(h("div", { class: "ed-empty" }, h("div", {}, "Select an object in the viewport or the outliner"), h("div", { class: "ed-empty-hint" }, "Environment and render settings are under World")));
      return;
    }
    const [kind, id] = splitRef(ref);
    if (kind === "a") actorDetails(id);
    else if (kind === "c") componentDetails(ref);
    else if (kind === "p") pcgDetails(id);
    else if (kind === "m") body.append(...materialSections(id));
    else if (ref === "env") environmentDetails();
    else if (ref === "render") renderDetails();
  }

  function header(iconName, name, kind, actions = []) {
    body.append(h("div", { class: "ed-details-header" }, icon(iconName, "ed-details-icon"), h("div", { class: "ed-details-name" }, h("div", {}, name), h("div", { class: "ed-details-kind" }, kind)), h("div", { class: "ed-details-actions" }, actions)));
  }

  const commonActions = () => [
    button("", () => ed.focusSelection(), { iconName: "focus", title: "Focus (F)" }),
    button("", () => ed.duplicateSelection(), { iconName: "copy", title: "Duplicate (Ctrl+D)" }),
    button("", () => ed.deleteSelection(), { iconName: "trash", title: "Delete (Del)", cls: "danger" }),
  ];

  // Location / rotation / scale rows for a transform {t, r, s}; `commit(trs, merge, done)`.
  function transformSection(trs, commit, { scaleLabel = "Scale", rotation = true, uniformScale = true, title = "Transform" } = {}) {
    const state = { t: [...trs.t], euler: quatToEuler(trs.r), s: [...trs.s] };
    let lock = uniformScale && Math.abs(trs.s[0] - trs.s[1]) < 1e-4 && Math.abs(trs.s[1] - trs.s[2]) < 1e-4;
    const key = `xf-${Math.random()}`;
    const send = (done) => commit({ t: state.t, r: quatFromEuler(state.euler), s: state.s }, key, done);
    const location = vectorField(state.t, { step: 0.05, precision: 3, onInput: (i, v) => ((state.t[i] = v), send(false)), onCommit: (i, v) => ((state.t[i] = v), send(true)) });
    const rows = [row("Location", location)];
    if (rotation) {
      const rot = vectorField(state.euler, { step: 0.5, precision: 2, onInput: (i, v) => ((state.euler[i] = v), send(false)), onCommit: (i, v) => ((state.euler[i] = v), send(true)) });
      rows.push(row("Rotation", rot));
    }
    let scaleField;
    const setScale = (i, v, done) => {
      if (lock) {
        const ratio = v / (state.s[i] || 1);
        state.s = state.s.map((x, k) => (k === i ? v : x * ratio));
        scaleField.setValues(state.s);
      } else state.s[i] = v;
      send(done);
    };
    scaleField = vectorField(state.s, { step: 0.01, precision: 3, min: 0.001, onInput: (i, v) => setScale(i, v, false), onCommit: (i, v) => setScale(i, v, true) });
    const lockBox = uniformScale ? h("label", { class: "ed-lock", title: "Uniform scale" }, checkbox(lock, (v) => (lock = v)), "🔒") : null;
    rows.push(row(scaleLabel, scaleField, lockBox));
    return section(title, rows);
  }

  // ---- Actors ----------------------------------------------------------------------------

  const KIND_ICON = { mesh: "cube", light: "light", volume: "volume", particles: "volume" };

  function actorDetails(id) {
    const info = ed.actor(id);
    if (!info) return;
    const what = info.folder ? "Folder" : info.added ? "Added actor" : "Scene actor";
    const counts = [`${info.components.length} component${info.components.length === 1 ? "" : "s"}`];
    if (info.descendants) counts.push(`${info.descendants} below`);
    header(info.components.length ? KIND_ICON[info.components[0].kind] || "cube" : "folder", info.name, `${what} · ${counts.join(" · ")}`, [
      button("", () => ed.groupSelection(), { iconName: "folder", title: "Group (Ctrl+G)" }),
      ...commonActions(),
    ]);
    const parentRow = info.parent === null
      ? h("span", { class: "ed-value-text" }, "World")
      : h("span", { class: "ed-value-text" }, info.parent_name, " ", button("", () => ed.reparent([id], null), { iconName: "close", title: "Detach (keep where it is)" }));
    body.append(section("Actor", [
      row("Name", textField(info.name, (t) => ed.rename(id, t.trim()))),
      row("Parent", parentRow),
      row("Visible", checkbox(info.visible, (v) => ed.apply({ op: "actor_visibility", actors: ed.selectedIds("a").includes(id) ? ed.selectedIds("a") : [id], visible: v }))),
      row("Key", h("span", { class: "ed-value-text ed-key", title: "How scene files address this actor" }, info.key)),
    ]));
    // Relative to the parent, as UE's Details; the gizmo works in world space.
    if (!info.folder) {
      body.append(transformSection(info.local, (trs, merge, done) => ed.apply({ op: "actor_local", actors: [id], transforms: [trs] }, { merge, quiet: !done }), { title: info.parent === null ? "Transform" : "Transform (relative to parent)" }));
    }
    if (info.components.length) {
      body.append(section(`Components · ${info.components.length}`, info.components.map((c) => componentRow(id, c))));
    }
    // An actor of one mesh (or one light) shows it directly.
    const meshes = info.components.filter((c) => c.kind === "mesh");
    const lights = info.components.filter((c) => c.kind === "light");
    if (meshes.length) body.append(meshSection(id, meshes, true));
    if (meshes.length === 1) body.append(...materialSections(meshes[0].material, true));
    if (lights.length === 1) lightSection(id, lights[0]);
  }

  function componentRow(actor, c) {
    const ref = componentRef(actor, c.name);
    const label = c.kind === "mesh" ? `${c.mesh} · ${c.material}` : c.kind;
    const eye = icon(c.visible ? "eye" : "eyeOff", "ed-eye");
    eye.addEventListener("click", (e) => {
      e.stopPropagation();
      ed.apply({ op: "component_visibility", components: [{ actor, component: c.name }], visible: !c.visible });
    });
    const el = h("div", { class: "ed-component" + (ed.selection.includes(ref) ? " selected" : "") + (c.visible ? "" : " dim"), title: "Select the component" },
      icon(KIND_ICON[c.kind] || "cube", "ed-row-icon"), h("span", { class: "ed-component-name" }, c.name), h("span", { class: "ed-tree-sub" }, label), eye);
    el.addEventListener("click", () => ed.select([ref]));
    return el;
  }

  // Mesh and material of mesh components (all of an actor's, or one).
  function meshSection(actor, meshes, many) {
    const materials = [...new Set([...ed.state.materials.map((m) => m.name).filter(Boolean), ...ed.state.content.materials])];
    const first = meshes[0];
    const targets = many ? { actors: [actor] } : { components: [{ actor, component: first.name }] };
    const rows = [
      row("Mesh", h("span", { class: "ed-value-text" }, meshes.length > 1 ? `${meshes.length} meshes` : first.mesh)),
      row("Triangles", h("span", { class: "ed-value-text" }, meshes.reduce((n, c) => n + c.triangles, 0).toLocaleString())),
      row("Material", select(materials.map((m) => [m, m]), first.material, (m) => ed.apply({ op: "assign_material", ...targets, material: m }))),
    ];
    if (!many) rows.push(row("Cast shadows", checkbox(first.cast_shadow, (v) => ed.apply({ op: "cast_shadow", components: [{ actor, component: first.name }], cast_shadow: v }))));
    return section(many && meshes.length > 1 ? "Static meshes" : "Static mesh", rows);
  }

  function componentDetails(ref) {
    const { actor, component } = parseComponent(ref);
    const info = ed.actor(actor);
    const c = info?.components.find((x) => x.name === component);
    if (!c) return;
    header(KIND_ICON[c.kind] || "cube", c.name, `Component of ${info.name}`, [
      button("", () => ed.select(["a:" + actor]), { iconName: "folder", title: "Select the actor" }),
      ...commonActions(),
    ]);
    if (c.kind !== "light") {
      body.append(transformSection(c.local, (trs, merge, done) => ed.apply({ op: "component_local", components: [{ actor, component }], transforms: [trs] }, { merge, quiet: !done }), { title: "Transform (relative to the actor)" }));
    }
    body.append(section("Rendering", [row("Visible", checkbox(c.visible, (v) => ed.apply({ op: "component_visibility", components: [{ actor, component }], visible: v })))]));
    if (c.kind === "mesh") {
      body.append(meshSection(actor, [c], false));
      body.append(...materialSections(c.material, true));
    } else if (c.kind === "light") lightSection(actor, c);
    else body.append(section("Emitter", [h("div", { class: "ed-note" }, `A ${c.kind} emitter of the base scene: it moves with its actor and hides with it.`)]));
  }

  // ---- Materials ---------------------------------------------------------------------------

  function materialSections(name, nested = false) {
    const m = ed.state.materials.find((x) => x.name === name);
    if (!m) {
      return [section("Material", [h("div", { class: "ed-note" }, `${name} is created the first time something uses it.`)])];
    }
    // An edit replaces the material's whole entry: send every field (graph, parameters and slab too).
    const params = materialParams(m);
    const key = `mat-${name}-${Math.random()}`;
    const send = (done) => ed.apply({ op: "material", name, params }, { merge: key, quiet: !done });
    const rows = [
      row("Base colour", colorField(m.base_color, (c) => ((params.base_color = c), send(false)), (c) => ((params.base_color = c), send(true)))),
      row("Roughness", slider(m.roughness, { min: 0, max: 1, step: 0.01, onInput: (v) => ((params.roughness = v), send(false)), onCommit: (v) => ((params.roughness = v), send(true)) })),
      row("Metallic", slider(m.metallic, { min: 0, max: 1, step: 0.01, onInput: (v) => ((params.metallic = v), send(false)), onCommit: (v) => ((params.metallic = v), send(true)) })),
      row("Emissive", slider(m.emissive, { min: 0, max: 50, step: 0.1, precision: 1, onInput: (v) => ((params.emissive = v), send(false)), onCommit: (v) => ((params.emissive = v), send(true)) })),
    ];
    if (["glass", "translucent"].includes(m.kind)) {
      rows.push(row("Opacity", slider(m.opacity, { min: 0, max: 1, step: 0.01, onInput: (v) => ((params.opacity = v), send(false)), onCommit: (v) => ((params.opacity = v), send(true)) })));
    }
    if (!nested) header("material", name, `Material · ${m.kind}`);
    const title = nested ? `Material · ${name}` : "Parameters";
    const sections = [section(title, [h("div", { class: "ed-note" }, `${m.kind} shading model · shared by every instance using it`), ...rows])];
    if (!["glass", "water", "translucent"].includes(m.kind)) sections.push(graphSection(m, params, send), slabSection(m, params, send), displacementSection(m, params, send));
    return sections;
  }

  // The material graph and this material's values of its parameters (a material instance).
  function graphSection(m, params, send) {
    const names = (ed.state.graphs || []).map((g) => g.name);
    const rows = [row("Graph", select([["", "None (shading model)"], ...names.map((n) => [n, n])], m.graph || "", (g) => ((params.graph = g), send(true))),
      button("", () => ed.graphEditor.open(m.graph || names[0]), { iconName: "material", title: "Open the material graph editor" }))];
    const graph = (ed.state.graphs || []).find((g) => g.name === m.graph);
    if (graph?.errors?.length) rows.push(h("div", { class: "ed-error" }, `${graph.errors.length} compile error(s): the material renders without its graph`));
    for (const p of m.parameters || []) {
      if (p.kind === "scalar") {
        const [min, max] = p.range[0] < p.range[1] ? p.range : [0, 1];
        const set = (v, done) => ((params.parameters[p.name] = v), send(done));
        rows.push(row(p.name, slider(p.value, { min, max, step: (max - min) / 200, precision: 3, onInput: (v) => set(v, false), onCommit: (v) => set(v, true) })));
      } else if (p.kind === "vector") {
        const set = (c, done) => ((params.parameters[p.name] = [...c, p.value[3] ?? 1]), send(done));
        rows.push(row(p.name, colorField(p.value.slice(0, 3), (c) => set(c, false), (c) => set(c, true))));
      } else rows.push(row(p.name, h("span", { class: "ed-value-text" }, p.value)));
    }
    return section("Material graph", rows, { open: !!m.graph });
  }

  // Substrate slab: clear coat, sheen, subsurface, transmission, anisotropy, IOR and specular.
  function slabSection(m, params, send) {
    const slab = params.slab;
    const field = (label, prop, opts) => row(label, slider(slab[prop], { ...opts, onInput: (v) => ((slab[prop] = v), send(false)), onCommit: (v) => ((slab[prop] = v), send(true)) }));
    const color = (label, prop) => row(label, colorField(slab[prop], (c) => ((slab[prop] = c), send(false)), (c) => ((slab[prop] = c), send(true))));
    const on = Object.entries(slab).some(([k, v]) => (k === "clearcoat" || k === "subsurface" || k === "transmission") && v > 0) || slab.sheen_color.some((c) => c > 0) || slab.anisotropy !== 0 || slab.unlit;
    return section("Substrate slab", [
      h("div", { class: "ed-note" }, "PageUp toggles slabs; debug view 24 shows their features"),
      field("Clear coat", "clearcoat", { min: 0, max: 1, step: 0.01 }),
      field("Coat roughness", "clearcoat_roughness", { min: 0, max: 1, step: 0.01 }),
      color("Sheen colour", "sheen_color"),
      field("Sheen roughness", "sheen_roughness", { min: 0, max: 1, step: 0.01 }),
      field("Subsurface", "subsurface", { min: 0, max: 1, step: 0.01 }),
      color("Scatter colour", "subsurface_color"),
      field("Scatter radius (m)", "subsurface_radius", { min: 0, max: 0.1, step: 0.001, precision: 3 }),
      field("Transmission", "transmission", { min: 0, max: 1, step: 0.01 }),
      field("Anisotropy", "anisotropy", { min: -1, max: 1, step: 0.01 }),
      field("IOR", "ior", { min: 1, max: 3, step: 0.01 }),
      field("Specular", "specular", { min: 0, max: 1, step: 0.01 }),
      row("Unlit", checkbox(slab.unlit, (v) => ((slab.unlit = v), send(true)))),
    ], { open: on });
  }

  // Displacement (Nanite tessellation): the surface moves at render time — along its normal by a
  // height (a texture channel, or the material graph's Displacement output when it is connected),
  // scaled by the magnitude around the centre, or by the graph's VectorDisplacement output, a
  // vector in units of the magnitude (no centre: zero stays in place). A graph that reads the
  // time animates it.
  function displacementSection(m, params, send) {
    const d = params.displacement;
    const field = (label, prop, opts) => row(label, slider(d[prop], { ...opts, onInput: (v) => ((d[prop] = v), send(false)), onCommit: (v) => ((d[prop] = v), send(true)) }));
    const vector = m.displacement_mode === "vector";
    const what = vector ? "Vector" : "Height";
    const pin = vector ? "VectorDisplacement" : "Displacement";
    const driven = `Driven by graph · ${m.graph}` + (m.displacement_animated ? " · animated" : "");
    const source = m.displacement_graph
      ? row(what, button(driven, () => ed.graphEditor.open(m.graph), { iconName: "material", title: `The graph's ${pin} output gives the ${what.toLowerCase()}: open it` }))
      : row("Height channel", select([["0", "R"], ["1", "G"], ["2", "B"], ["3", "A (procedural height)"]], String(d.channel), (v) => ((d.channel = Number(v)), send(true))));
    const toggles = "PageDown toggles tessellation; debug view 25 shows it";
    const note = m.displacement_graph && !(d.magnitude > 0)
      ? `The graph drives the ${what.toLowerCase()}: set a magnitude to displace. ${toggles}`
      : vector
        ? `The graph's vector moves the surface by up to the magnitude in any direction (overhangs). ${toggles}`
        : `Magnitude 0: none. ${toggles}`;
    const rows = [h("div", { class: "ed-note" }, note), field("Magnitude (m)", "magnitude", { min: 0, max: 0.5, step: 0.005, precision: 3 })];
    if (!vector) rows.push(field("Centre", "center", { min: 0, max: 1, step: 0.01 }));
    rows.push(source);
    if (m.displacement_graph && m.displacement_animated) {
      // The material compiler's speed bound (m/s at unit instance scale) paces the shadow refreshes.
      const speed = m.displacement_speed;
      const pace = speed == null
        ? "the graph's motion has no speed bound, so they refresh every frame"
        : `they refresh once it may have moved two shadow texels (at most ${(speed * 100).toFixed(1)} cm/s)`;
      rows.push(h("div", { class: "ed-note" }, `Animated: its instances' shadows are re-rendered as the surface moves; ${pace}`));
    }
    return section("Displacement", rows, { open: d.magnitude > 0 || m.displacement_graph });
  }

  // ---- Lights ------------------------------------------------------------------------------

  function lightSection(actor, c) {
    const def = structuredClone(c.light);
    const key = `light-${actor}-${c.name}-${Math.random()}`;
    const send = (done) => ed.apply({ op: "component_light", component: { actor, component: c.name }, def }, { merge: key, quiet: !done });
    const field = (label, prop, opts) => row(label, slider(def[prop], { ...opts, onInput: (v) => ((def[prop] = v), send(false)), onCommit: (v) => ((def[prop] = v), send(true)) }));
    const rows = [
      row("Location", vectorField(def.position, { step: 0.05, onInput: (i, v) => ((def.position[i] = v), send(false)), onCommit: (i, v) => ((def.position[i] = v), send(true)) })),
      row("Type", select([["point", "Point"], ["spot", "Spot"]], def.shape, (v) => ((def.shape = v), send(true), render()))),
      row("Colour", colorField(def.color, (x) => ((def.color = x), send(false)), (x) => ((def.color = x), send(true)))),
      field("Intensity", "intensity", { min: 0, max: 100, step: 0.1, precision: 1 }),
      field("Attenuation radius", "range", { min: 0.5, max: 100, step: 0.1, precision: 1 }),
      field("Source radius", "source_radius", { min: 0, max: 2, step: 0.01 }),
      row("Cast shadows", checkbox(def.cast_shadows, (v) => ((def.cast_shadows = v), send(true)))),
    ];
    if (def.shape === "spot") {
      const d = def.direction;
      const aim = { pitch: Math.asin(Math.max(-1, Math.min(1, d[1]))) * 57.2958, yaw: Math.atan2(d[0], d[2]) * 57.2958 };
      const setAim = (done) => {
        const p = (aim.pitch * Math.PI) / 180, y = (aim.yaw * Math.PI) / 180;
        def.direction = [Math.cos(p) * Math.sin(y), Math.sin(p), Math.cos(p) * Math.cos(y)];
        send(done);
      };
      rows.push(
        row("Aim pitch", slider(aim.pitch, { min: -90, max: 90, step: 1, precision: 0, onInput: (v) => ((aim.pitch = v), setAim(false)), onCommit: (v) => ((aim.pitch = v), setAim(true)) })),
        row("Aim yaw", slider(aim.yaw, { min: -180, max: 180, step: 1, precision: 0, onInput: (v) => ((aim.yaw = v), setAim(false)), onCommit: (v) => ((aim.yaw = v), setAim(true)) })),
        field("Inner cone", "inner_deg", { min: 1, max: 89, step: 0.5, precision: 1 }),
        field("Outer cone", "outer_deg", { min: 1, max: 89, step: 0.5, precision: 1 }),
      );
    }
    body.append(section(`Light · ${c.name}`, rows));
  }

  // ---- PCG volumes -------------------------------------------------------------------------

  function pcgDetails(id) {
    const v = ed.volume(id);
    if (!v) return;
    const def = structuredClone(v.def);
    header("volume", def.name, `PCG volume · ${v.instances.toLocaleString()} instances`, [
      button("", () => regenerate(), { iconName: "refresh", title: "Regenerate with a new seed" }),
      ...commonActions(),
    ]);
    if (v.error) body.append(h("div", { class: "ed-error" }, v.error));
    const key = `pcg-${id}-${Math.random()}`;
    // Live: regenerate at most once a frame while dragging, one undo step per gesture.
    const send = (done) => {
      if (done) ed.apply({ op: "pcg", def: structuredClone(def) }, { merge: key });
      else ed.applyLater(key, { op: "pcg", def: structuredClone(def) }, key);
    };
    const regenerate = () => {
      def.graph.seed = (Math.random() * 0xffffffff) >>> 0;
      send(true);
    };
    const vol = def.volume;
    body.append(section("Volume", [
      row("Name", textField(def.name, (t) => ((def.name = t), send(true)))),
      row("Center", vectorField(vol.center, { step: 0.1, onInput: (i, x) => ((vol.center[i] = x), send(false)), onCommit: (i, x) => ((vol.center[i] = x), send(true)) })),
      row("Size", vectorField(vol.half_extents.map((x) => x * 2), { step: 0.1, min: 1, onInput: (i, x) => ((vol.half_extents[i] = x / 2), send(false)), onCommit: (i, x) => ((vol.half_extents[i] = x / 2), send(true)) })),
      row("Yaw", slider(vol.yaw_deg || 0, { min: -180, max: 180, step: 1, precision: 0, onInput: (x) => ((vol.yaw_deg = x), send(false)), onCommit: (x) => ((vol.yaw_deg = x), send(true)) })),
      row("Surface", select([["landscape", "Landscape"], ["component", "Selected mesh"]], def.surface.kind === "landscape" ? "landscape" : "component", (k) => {
        // The selected mesh component, or the first mesh of a selected actor.
        const c = ed.selectedComponents()[0];
        const a = ed.selectedIds("a")[0];
        const actor = c ? ed.actor(c.actor) : a !== undefined ? ed.actor(a) : null;
        const mesh = c ? actor?.components.find((x) => x.name === c.component && x.kind === "mesh") : actor?.components.find((x) => x.kind === "mesh");
        if (k === "component" && !mesh) {
          ed.status("Select a mesh actor or component too (Ctrl+click), then choose it as the surface", "warn");
          render();
          return;
        }
        def.surface = k === "landscape" ? { kind: "landscape" } : { kind: "component", actor: actor.key, component: mesh.name };
        send(true);
      })),
      def.surface.kind !== "landscape" ? row("Surface mesh", h("span", { class: "ed-value-text" }, def.surface.kind === "component" ? `${def.surface.actor} · ${def.surface.component}` : `#${def.surface.instance}`)) : null,
      row("Seed", numberField(def.graph.seed, { step: 1, precision: 0, min: 0, onCommit: (x) => ((def.graph.seed = Math.round(x) >>> 0), send(true)) }), button("", regenerate, { iconName: "dice", title: "Random seed" })),
    ]));
    body.append(section(`Graph · ${def.graph.nodes.length} nodes`, def.graph.nodes.map((node, k) => nodeCard(node, k, def, send))));
  }

  function textField(value, onCommit) {
    const input = h("input", { class: "ed-text", value });
    input.addEventListener("change", () => onCommit(input.value));
    input.addEventListener("keydown", (e) => e.stopPropagation());
    return input;
  }

  const NODE_FIELDS = {
    SurfaceSampler: { points_per_m2: [0.01, 20, 0.01], jitter: [0, 1, 0.01] },
    SlopeFilter: { min_deg: [0, 90, 1], max_deg: [0, 90, 1] },
    HeightFilter: { min: [-500, 500, 0.5], max: [-500, 500, 0.5] },
    NoiseDensity: { scale: [0.5, 200, 0.5], threshold: [0, 1, 0.01], softness: [0, 0.5, 0.01] },
    DensityFilter: { min: [0, 1, 0.01], max: [0, 1, 0.01] },
    DistanceFilter: { min: [0, 200, 0.5], max: [0, 1000, 0.5], falloff: [0, 50, 0.5] },
    SelfPruning: { radius: [0, 50, 0.05] },
    Difference: { radius: [0, 50, 0.05] },
    Transform: { scale_min: [0.05, 5, 0.01], scale_max: [0.05, 5, 0.01], scale_jitter: [0, 1, 0.01], tilt_deg: [0, 90, 1], align_to_normal: [0, 1, 0.01], offset_min: [-5, 5, 0.01], offset_max: [-5, 5, 0.01] },
  };

  function nodeCard(node, k, def, send) {
    const fields = NODE_FIELDS[node.type] || {};
    const rows = [];
    for (const [prop, [min, max, step]] of Object.entries(fields)) {
      if (typeof node[prop] !== "number") continue;
      rows.push(row(prop.replace(/_/g, " "), slider(node[prop], { min, max, step, precision: step >= 1 ? 0 : step >= 0.1 ? 1 : 2, onInput: (v) => ((node[prop] = v), send(false)), onCommit: (v) => ((node[prop] = v), send(true)) })));
    }
    for (const prop of ["random", "invert", "random_yaw"]) {
      if (typeof node[prop] === "boolean") rows.push(row(prop.replace(/_/g, " "), checkbox(node[prop], (v) => ((node[prop] = v), send(true)))));
    }
    if (node.type === "BoundsFilter") rows.push(h("div", { class: "ed-note" }, `box ${JSON.stringify(node.min)} – ${JSON.stringify(node.max)} (volume units)`));
    if (node.type === "DistanceFilter") rows.push(h("div", { class: "ed-note" }, `${node.splines?.length || 0} spline(s), ${node.points?.length || 0} point(s), volume-local`));
    if (node.type === "Spawner") {
      node.entries.forEach((entry, e) => {
        rows.push(row(`weight ${e + 1}`, slider(entry.weight, { min: 0, max: 10, step: 0.1, precision: 1, onInput: (v) => ((entry.weight = v), send(false)), onCommit: (v) => ((entry.weight = v), send(true)) }),
          h("span", { class: "ed-node-parts", title: entry.parts.map((p) => `${p.mesh} · ${p.material}`).join("\n") }, entry.parts.map((p) => p.mesh.replace("engine/", "")).join(" + "))));
      });
    }
    const inputs = node.inputs?.length ? `← ${node.inputs.join(", ")}` : k > 0 && node.type !== "SurfaceSampler" ? `← ${k - 1}` : "";
    const head = h("div", { class: "ed-node-head" },
      checkbox(node.enabled !== false, (v) => ((node.enabled = v), send(true))),
      h("span", { class: "ed-node-index" }, k),
      h("span", { class: "ed-node-name" }, node.name || node.type),
      h("span", { class: "ed-node-type" }, node.type),
      h("span", { class: "ed-node-inputs" }, inputs));
    return h("div", { class: "ed-node" + (node.enabled === false ? " disabled" : "") }, head, rows.length ? h("div", { class: "ed-node-body" }, rows) : null);
  }

  // ---- Environment -------------------------------------------------------------------------

  function environmentDetails() {
    const env = structuredClone(ed.state.environment);
    header("sun", "Environment", "Sun, sky, fog, clouds and wind");
    const key = `env-${Math.random()}`;
    const send = (done) => ed.apply({ op: "environment", env }, { merge: key, quiet: !done });
    const field = (obj, label, prop, opts) => row(label, slider(obj[prop], { ...opts, onInput: (v) => ((obj[prop] = v), send(false)), onCommit: (v) => ((obj[prop] = v), send(true)) }));
    body.append(section("Directional light (sun)", [
      field(env, "Azimuth", "sun_azimuth_deg", { min: -180, max: 360, step: 0.5, precision: 1 }),
      field(env, "Elevation", "sun_elevation_deg", { min: -60, max: 90, step: 0.25, precision: 2 }),
      field(env, "Intensity", "sun_intensity", { min: 0, max: 30, step: 0.1, precision: 1 }),
      h("div", { class: "ed-note" }, "Below about −11° the moon and stars take over (night)."),
    ]));
    body.append(section("Volumetric fog", [
      field(env.fog, "Density", "density", { min: 0, max: 0.05, step: 0.0005, precision: 4 }),
      field(env.fog, "Height falloff", "height_falloff", { min: 0, max: 0.5, step: 0.005, precision: 3 }),
      field(env.fog, "Base height", "base_height", { min: -200, max: 200, step: 0.5, precision: 1 }),
      field(env.fog, "Anisotropy", "anisotropy", { min: 0, max: 0.95, step: 0.01 }),
      field(env.fog, "Noise", "noise", { min: 0, max: 1, step: 0.01 }),
      field(env.fog, "Distance", "distance", { min: 20, max: 1000, step: 5, precision: 0 }),
      row("Albedo", colorField(env.fog.albedo, (c) => ((env.fog.albedo = c), send(false)), (c) => ((env.fog.albedo = c), send(true)))),
    ]));
    body.append(section("Volumetric clouds", [
      field(env.clouds, "Coverage", "coverage", { min: 0, max: 1, step: 0.01 }),
      field(env.clouds, "Density", "density", { min: 0, max: 60, step: 0.5, precision: 1 }),
      field(env.clouds, "Base altitude", "base_altitude", { min: 200, max: 5000, step: 10, precision: 0 }),
      field(env.clouds, "Thickness", "thickness", { min: 100, max: 5000, step: 10, precision: 0 }),
      field(env.clouds, "Shadow strength", "shadow_strength", { min: 0, max: 1, step: 0.01 }),
    ], { open: false }));
    const windAngle = { deg: (Math.atan2(env.wind.direction[1], env.wind.direction[0]) * 180) / Math.PI };
    body.append(section("Wind", [
      field(env.wind, "Strength", "strength", { min: 0, max: 3, step: 0.05 }),
      row("Direction", slider(windAngle.deg, { min: -180, max: 180, step: 1, precision: 0, onInput: (v) => {
        env.wind.direction = [Math.cos((v * Math.PI) / 180), Math.sin((v * Math.PI) / 180)];
        send(false);
      }, onCommit: () => send(true) })),
      field(env.wind, "Gusts / s", "gust_frequency", { min: 0, max: 1, step: 0.01 }),
      field(env.wind, "Gust wavelength", "gust_wavelength", { min: 1, max: 200, step: 1, precision: 0 }),
      field(env.wind, "Turbulence", "turbulence", { min: 0, max: 1, step: 0.01 }),
    ], { open: false }));
  }

  // ---- Render settings (data-driven: the engine sends the schema) -------------------------

  function renderDetails() {
    const { schema, values } = ed.state.render;
    header("gear", "Render settings", "Post process & engine scalability");
    const groups = new Map();
    for (const s of schema) {
      if (!groups.has(s.group)) groups.set(s.group, []);
      const key = `render-${s.key}`;
      const send = (value, done) => ed.apply({ op: "render", key: s.key, value }, { merge: key, quiet: !done });
      let control;
      if (s.kind.type === "bool") control = checkbox(values[s.key], (v) => send(v, true));
      else if (s.kind.type === "number") control = slider(values[s.key], { min: s.kind.min, max: s.kind.max, step: s.kind.step, precision: s.kind.step >= 1 ? 0 : 3, onInput: (v) => send(v, false), onCommit: (v) => send(v, true) });
      else control = select(s.kind.options, values[s.key], (v) => send(v, true));
      groups.get(s.group).push(row(s.label, control));
    }
    for (const [group, rows] of groups) body.append(section(group, rows));
  }

  return { element, render };
}
