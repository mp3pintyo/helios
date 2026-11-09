// Details panel: the properties of the primary selection — an instance (transform, material
// and its parameters, graph parameters and Substrate slab, rendering), a light, a PCG volume
// (volume, surface, seed and the graph's nodes), the environment (sun, fog, clouds, wind), the
// render settings (built from the engine's schema) or a material.

import { h, icon, section, row, numberField, vectorField, slider, checkbox, select, colorField, button } from "./ui.js";
import { quatFromEuler, quatToEuler } from "./math.js";
import { splitRef } from "./editor.js";
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
    if (kind === "i") instanceDetails(id);
    else if (kind === "l") lightDetails(id);
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
  function transformSection(trs, commit, { scaleLabel = "Scale", rotation = true, uniformScale = true } = {}) {
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
    return section("Transform", rows);
  }

  // ---- Instance --------------------------------------------------------------------------

  function instanceDetails(i) {
    const info = ed.instance(i);
    if (!info) return;
    const owner = info.owner.kind === "added" ? "Added instance" : info.owner.kind === "pcg" ? "Spawned by PCG" : "Scene instance";
    header("cube", `${info.mesh} #${i}`, owner, commonActions());
    const ids = ed.selectedIds("i");
    // With several selected, typed values move the primary only (the gizmo moves them all).
    body.append(transformSection(info.transform, (trs, merge, done) => ed.apply({ op: "transform", instances: [i], transforms: [trs] }, { merge, quiet: !done })));
    const materials = [...new Set([...ed.state.materials.map((m) => m.name).filter(Boolean), ...ed.state.content.materials])];
    body.append(section("Static mesh", [
      row("Mesh", h("span", { class: "ed-value-text" }, info.mesh)),
      row("Triangles", h("span", { class: "ed-value-text" }, info.triangles.toLocaleString())),
      row("Material", select(materials.map((m) => [m, m]), info.material, (m) => {
        ed.apply({ op: "assign_material", instances: ids.length ? ids : [i], material: m });
      })),
    ]));
    body.append(...materialSections(info.material, true));
    const rendering = [row("Visible", checkbox(info.visible, (v) => ed.apply({ op: "visibility", instances: ids.length ? ids : [i], visible: v })))];
    if (info.owner.kind === "added") rendering.push(row("Cast shadows", checkbox(info.cast_shadow, (v) => ed.apply({ op: "cast_shadow", instances: [i], cast_shadow: v }))));
    body.append(section("Rendering", rendering));
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
    if (!["glass", "water", "translucent"].includes(m.kind)) sections.push(graphSection(m, params, send), slabSection(m, params, send));
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

  // ---- Lights ------------------------------------------------------------------------------

  function lightDetails(index) {
    const light = ed.light(index);
    if (!light) return;
    const def = structuredClone(light.def);
    header(def.shape === "spot" ? "spot" : "light", `${def.shape === "spot" ? "Spot" : "Point"} light ${index}`, light.added ? "Added light" : "Scene light", commonActions());
    const key = `light-${index}-${Math.random()}`;
    const send = (done) => ed.apply({ op: "light", light: index, def }, { merge: key, quiet: !done });
    const field = (label, prop, opts) => row(label, slider(def[prop], { ...opts, onInput: (v) => ((def[prop] = v), send(false)), onCommit: (v) => ((def[prop] = v), send(true)) }));
    body.append(section("Transform", [row("Location", vectorField(def.position, { step: 0.05, onInput: (i, v) => ((def.position[i] = v), send(false)), onCommit: (i, v) => ((def.position[i] = v), send(true)) }))]));
    const rows = [
      row("Type", select([["point", "Point"], ["spot", "Spot"]], def.shape, (v) => ((def.shape = v), send(true), render()))),
      row("Colour", colorField(def.color, (c) => ((def.color = c), send(false)), (c) => ((def.color = c), send(true)))),
      field("Intensity", "intensity", { min: 0, max: 100, step: 0.1, precision: 1 }),
      field("Attenuation radius", "range", { min: 0.5, max: 100, step: 0.1, precision: 1 }),
      field("Source radius", "source_radius", { min: 0, max: 2, step: 0.01 }),
      row("Cast shadows", checkbox(def.cast_shadows, (v) => ((def.cast_shadows = v), send(true)))),
    ];
    if (def.shape === "spot") {
      const yawPitch = () => {
        const d = def.direction;
        return [Math.asin(Math.max(-1, Math.min(1, d[1]))) * 57.2958, Math.atan2(d[0], d[2]) * 57.2958];
      };
      const [pitch, yaw] = yawPitch();
      const aim = { pitch, yaw };
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
    body.append(section("Light", rows));
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
      row("Surface", select([["landscape", "Landscape"], ["instance", "Selected mesh instance"]], def.surface.kind, (k) => {
        const instance = ed.selectedIds("i")[0];
        if (k === "instance" && instance === undefined) {
          ed.status("Select a mesh instance too (Ctrl+click), then choose it as the surface", "warn");
          render();
          return;
        }
        def.surface = k === "landscape" ? { kind: "landscape" } : { kind: "instance", instance };
        send(true);
      })),
      def.surface.kind === "instance" ? row("Surface instance", h("span", { class: "ed-value-text" }, `#${def.surface.instance}`)) : null,
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
