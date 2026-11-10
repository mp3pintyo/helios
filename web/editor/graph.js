// Material graph editor (UE's Material Editor): a node canvas over the viewport. Nodes come from
// the engine's catalog (heliosMaterialNodes: pins, defaults, properties); every change is an
// edit of the scene document ({op: "graph"}), so it is undoable, saved in scene files and
// recompiled live by the renderer (a graph's WGSL is cached by hash). Compile errors come back
// with the scene state and mark their nodes.
//
// Mouse: drag a node's title to move it, drag from a pin to another to link them (an input
// takes one link; dropping on a linked input replaces it), Alt+click a pin to unlink it,
// right-click or double-click the canvas for the node palette, right-drag or middle-drag to pan,
// wheel to zoom. Keys (canvas focused): Delete removes the selected node, F frames the graph.

import * as engine from "../pkg/helios_demo.js";
import { h, icon, button, select, colorField } from "./ui.js";

const CATEGORY_COLORS = {
  Constants: "#5b7a47", Parameters: "#3d6f92", Textures: "#8a5a2b", Inputs: "#7a3f6e",
  Math: "#4a5260", Vector: "#4f5d73", Patterns: "#6a5a8c", Utility: "#56606b", Output: "#9b3b30",
};
const NODE_WIDTH = 168;

export function createGraphEditor(ed) {
  const catalog = JSON.parse(engine.heliosMaterialNodes());
  const specs = new Map(catalog.nodes.map((n) => [n.type, n]));
  const state = { name: null, graph: null, selected: null, pan: [380, 160], zoom: 0.85, drag: null, seq: 0 };

  const title = h("span", { class: "mg-title" }, "Material Graph");
  const graphSelect = h("span");
  const presetSelect = h("span");
  const assignSelect = h("span");
  const statusEl = h("span", { class: "mg-status" });
  const errorsEl = h("div", { class: "mg-errors" });
  const links = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  links.setAttribute("class", "mg-links");
  const world = h("div", { class: "mg-world" }, links);
  const canvas = h("div", { class: "mg-canvas", tabindex: "0" }, world);
  const palette = h("div", { class: "mg-palette hidden" });
  const element = h("section", { class: "mg-panel hidden" },
    h("div", { class: "mg-head" }, icon("material"), title, graphSelect, presetSelect, assignSelect, statusEl, h("span", { class: "ed-spacer" }),
      button("", () => frameAll(), { iconName: "focus", title: "Frame the graph (F)" }),
      button("", () => close(), { iconName: "plus", cls: "mg-close", title: "Close" })),
    h("div", { class: "mg-body" }, canvas, palette, errorsEl));

  // ---- Document access ----------------------------------------------------------------------

  const graphs = () => ed.state?.graphs || [];
  const current = () => graphs().find((g) => g.name === state.name);

  function open(name) {
    element.classList.remove("hidden");
    if (name && graphs().some((g) => g.name === name)) load(name);
    else if (!state.graph) {
      const first = graphs()[0];
      if (first) load(first.name);
    }
    update();
    canvas.focus();
  }

  function close() {
    element.classList.add("hidden");
    hidePalette();
  }

  function load(name) {
    state.name = name;
    state.graph = structuredClone(current()?.graph || { name, nodes: [], links: [] });
    state.selected = null;
    render();
    requestAnimationFrame(frameAll);
  }

  // Sends the working copy as one edit (merged while a gesture lasts).
  function commit(merge = null) {
    if (!state.name) return;
    state.graph.name = state.name;
    ed.apply({ op: "graph", name: state.name, graph: state.graph }, { merge, quiet: !!merge });
    renderStatus();
  }

  // Called when the scene state changes (undo, redo, loads): pick up the document's version.
  function update() {
    if (element.classList.contains("hidden")) return;
    const g = current();
    if (g && !state.drag && JSON.stringify(g.graph) !== JSON.stringify(state.graph)) {
      state.graph = structuredClone(g.graph);
      render();
    }
    renderHead();
    renderStatus();
  }

  // ---- Header -------------------------------------------------------------------------------

  function renderHead() {
    const names = graphs().map((g) => g.name);
    graphSelect.replaceChildren(select([["", names.length ? "Graph…" : "No graphs"], ...names.map((n) => [n, n])], state.name || "", (n) => n && load(n)));
    presetSelect.replaceChildren(select([["", "New from preset…"], ...catalog.presets.map((p) => [p.id, p.label])], "", (id) => {
      const preset = catalog.presets.find((p) => p.id === id);
      if (!preset) return;
      const base = preset.graph.name;
      let name = base, k = 2;
      while (names.includes(name)) name = `${base} ${k++}`;
      ed.apply({ op: "graph", name, graph: { ...preset.graph, name } });
      load(name);
      renderHead();
    }));
    const materials = (ed.state?.materials || []).filter((m) => !["glass", "water", "translucent"].includes(m.kind)).map((m) => m.name);
    assignSelect.replaceChildren(select([["", "Assign to material…"], ...materials.map((m) => [m, m])], "", (m) => {
      if (!m || !state.name) return;
      const material = ed.state.materials.find((x) => x.name === m);
      ed.apply({ op: "material", name: m, params: { ...materialParams(material), graph: state.name } });
      ed.status(`${m} now uses ${state.name}`);
      renderHead();
    }));
  }

  function renderStatus() {
    const g = current();
    const errors = g?.errors || [];
    // What the graph does to geometry, next to what it compiled to.
    const displaces = !g?.displacement ? "" : ` · ${g.displacement_animated ? "animated " : ""}${g.displacement === "vector" ? "vector displacement" : "displacement"}`;
    statusEl.textContent = !g ? "" : errors.length ? `${errors.length} error${errors.length > 1 ? "s" : ""}` : `compiled · ${g.parameters.length} parameter${g.parameters.length === 1 ? "" : "s"}${displaces} · ${g.hash || ""}`;
    statusEl.className = "mg-status" + (errors.length ? " error" : "");
    errorsEl.replaceChildren(...errors.map((e) => h("div", { class: "mg-error", onclick: () => e.node && selectNode(e.node, true) }, e.node ? `#${e.node}: ${e.message}` : e.message)));
    for (const el of world.querySelectorAll(".mg-node")) {
      const bad = errors.filter((e) => e.node === Number(el.dataset.id));
      el.classList.toggle("error", bad.length > 0);
      el.title = bad.map((e) => e.message).join("\n");
    }
  }

  // ---- Canvas -------------------------------------------------------------------------------

  function applyView() {
    world.style.transform = `translate(${state.pan[0]}px, ${state.pan[1]}px) scale(${state.zoom})`;
    canvas.style.backgroundPosition = `${state.pan[0]}px ${state.pan[1]}px`;
    canvas.style.backgroundSize = `${24 * state.zoom}px ${24 * state.zoom}px`;
  }

  function render() {
    world.querySelectorAll(".mg-node").forEach((n) => n.remove());
    if (!state.graph) return;
    for (const node of state.graph.nodes) world.append(nodeElement(node));
    applyView();
    drawLinks();
    renderStatus();
  }

  const linkInto = (node, pin) => state.graph.links.find((l) => l.to.node === node && l.to.pin === pin);

  // Output pins that move geometry (Nanite tessellation) rather than shade it: set apart on the
  // node and explained on hover. A graph connects one of the two.
  const DISPLACEMENT_PINS = {
    Displacement: "Height 0..1 along the normal, around the material's displacement centre, scaled by its magnitude (float). Time animates it.",
    VectorDisplacement: "World-space offset in units of the material's displacement magnitude, length up to 1 (float3): overhangs and sideways shifts. VertexNormal × height displaces as the Displacement pin does. Time animates it.",
  };

  function nodeElement(node) {
    const spec = specs.get(node.type) || { category: "Utility", inputs: [], outputs: [] };
    const label = node.name || node.parameter ? `${node.type} · ${node.name || node.parameter}` : node.type;
    const head = h("div", { class: "mg-node-head", style: `background:${CATEGORY_COLORS[spec.category] || "#555"}` }, label, h("span", { class: "mg-node-id" }, `#${node.id}`));
    const el = h("div", { class: "mg-node" + (state.selected === node.id ? " selected" : ""), "data-id": node.id, style: `left:${node.pos[0]}px;top:${node.pos[1]}px;width:${NODE_WIDTH}px` }, head);
    head.addEventListener("pointerdown", (e) => startMove(e, node, el));
    el.addEventListener("pointerdown", (e) => {
      if (e.button === 0) selectNode(node.id);
    });
    const props = propertyEditors(node);
    if (props.length) el.append(h("div", { class: "mg-props" }, props));
    const rows = Math.max(spec.inputs.length, spec.outputs.length);
    for (let r = 0; r < rows; r++) {
      const input = spec.inputs[r], output = spec.outputs[r];
      const row = h("div", { class: "mg-row" });
      if (input) {
        const linked = linkInto(node.id, input.name);
        const pin = pinElement(node.id, input.name, "in", linked);
        const hint = node.type === "Output" ? DISPLACEMENT_PINS[input.name] : undefined;
        const parts = [pin, h("span", hint ? { class: "mg-pin-label", title: hint } : { class: "mg-pin-label" }, input.name)];
        if (hint) row.classList.add("mg-row-geometry");
        if (!linked && input.default.kind === "scalar") parts.push(valueField(node, input));
        row.append(h("div", { class: "mg-in" }, parts));
      } else row.append(h("div", { class: "mg-in" }));
      if (output) {
        const used = state.graph.links.some((l) => l.from.node === node.id && l.from.pin === output);
        row.append(h("div", { class: "mg-out" }, h("span", { class: "mg-pin-label" }, output), pinElement(node.id, output, "out", used)));
      }
      el.append(row);
    }
    return el;
  }

  function pinElement(node, pin, dir, linked) {
    const el = h("span", { class: `mg-pin ${dir}` + (linked ? " linked" : ""), "data-node": node, "data-pin": pin, "data-dir": dir });
    el.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      if (e.button !== 0) return;
      if (e.altKey) {
        state.graph.links = state.graph.links.filter((l) => !(dir === "in" ? l.to.node === node && l.to.pin === pin : l.from.node === node && l.from.pin === pin));
        commit();
        render();
        return;
      }
      startLink(e, { node, pin, dir });
    });
    return el;
  }

  // An unconnected scalar input's constant (UE's "Const A").
  function valueField(node, input) {
    const value = node.values?.[input.name] ?? input.default.value;
    const field = h("input", { class: "mg-value", value: String(value), spellcheck: "false" });
    field.addEventListener("pointerdown", (e) => e.stopPropagation());
    field.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") field.blur();
    });
    field.addEventListener("change", () => {
      const v = parseFloat(field.value);
      if (!Number.isFinite(v)) return;
      node.values = { ...(node.values || {}), [input.name]: v };
      commit();
    });
    return field;
  }

  // Editors for a node's own properties (constants, parameter names and defaults, …).
  function propertyEditors(node) {
    const out = [];
    const stop = (el) => {
      el.addEventListener("pointerdown", (e) => e.stopPropagation());
      el.addEventListener("keydown", (e) => e.stopPropagation());
      return el;
    };
    const number = (key, i = null, step = 0.01) => {
      const value = i === null ? node[key] : node[key][i];
      const field = stop(h("input", { class: "mg-value", value: String(value), title: key, spellcheck: "false" }));
      field.addEventListener("change", () => {
        const v = parseFloat(field.value);
        if (!Number.isFinite(v)) return;
        if (i === null) node[key] = key === "octaves" || key === "set" ? Math.max(0, Math.round(v)) : v;
        else node[key][i] = v;
        commit();
      });
      return field;
    };
    const text = (key, rerender = false) => {
      const field = stop(h("input", { class: "mg-text", value: node[key], title: key, spellcheck: "false" }));
      field.addEventListener("change", () => {
        node[key] = field.value;
        commit();
        if (rerender) render();
      });
      return field;
    };
    const line = (label, ...controls) => h("div", { class: "mg-prop" }, h("span", { class: "mg-prop-label" }, label), ...controls);
    const color = (key) => {
      const rgb = node[key].slice(0, 3);
      return stop(colorField(rgb, (c) => (node[key].splice(0, 3, ...c), commit(`mg-color-${node.id}`)), (c) => (node[key].splice(0, 3, ...c), commit())));
    };
    switch (node.type) {
      case "Constant": out.push(line("value", number("value"))); break;
      case "Constant2": out.push(line("value", number("value", 0), number("value", 1))); break;
      case "Constant3": out.push(line("colour", color("value")), line("value", number("value", 0), number("value", 1), number("value", 2))); break;
      case "Constant4": out.push(line("colour", color("value")), line("alpha", number("value", 3))); break;
      case "ScalarParameter": out.push(line("name", text("name", true)), line("default", number("default")), line("range", number("min"), number("max"))); break;
      case "VectorParameter": out.push(line("name", text("name", true)), line("default", color("default"))); break;
      case "TextureSample": out.push(line("texture", text("parameter", true))); break;
      case "Triplanar": out.push(line("texture", text("parameter", true)), line("scale", number("scale"))); break;
      case "TexCoord": out.push(line("UV set", number("set", null, 1)), line("tiling", number("tiling", 0), number("tiling", 1))); break;
      case "ComponentMask": out.push(line("channels", text("channels"))); break;
      case "Noise": out.push(line("scale", number("scale")), line("octaves", number("octaves", null, 1))); break;
      case "Fresnel": out.push(line("exponent", number("exponent")), line("base", number("base_reflect"))); break;
      case "Panner": out.push(line("speed", number("speed", 0), number("speed", 1))); break;
    }
    return out;
  }

  // Pin centre in world coordinates.
  function pinPoint(node, pin, dir) {
    const el = world.querySelector(`.mg-pin.${dir}[data-node="${node}"][data-pin="${CSS.escape(pin)}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect(), w = world.getBoundingClientRect();
    return [(r.left + r.width / 2 - w.left) / state.zoom, (r.top + r.height / 2 - w.top) / state.zoom];
  }

  const curve = (a, b) => {
    const dx = Math.max(40, Math.abs(b[0] - a[0]) * 0.5);
    return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
  };

  function drawLinks(temp = null) {
    const paths = [];
    for (const l of state.graph?.links || []) {
      const a = pinPoint(l.from.node, l.from.pin, "out"), b = pinPoint(l.to.node, l.to.pin, "in");
      if (a && b) paths.push(`<path d="${curve(a, b)}"/>`);
    }
    if (temp) paths.push(`<path class="temp" d="${curve(temp[0], temp[1])}"/>`);
    links.innerHTML = paths.join("");
  }

  const worldPoint = (e) => {
    const w = canvas.getBoundingClientRect();
    return [(e.clientX - w.left - state.pan[0]) / state.zoom, (e.clientY - w.top - state.pan[1]) / state.zoom];
  };

  function selectNode(id, frame = false) {
    state.selected = id;
    for (const el of world.querySelectorAll(".mg-node")) el.classList.toggle("selected", Number(el.dataset.id) === id);
    if (frame) {
      const node = state.graph.nodes.find((n) => n.id === id);
      if (node) {
        const r = canvas.getBoundingClientRect();
        state.pan = [r.width / 2 - (node.pos[0] + NODE_WIDTH / 2) * state.zoom, r.height / 2 - node.pos[1] * state.zoom];
        applyView();
      }
    }
  }

  function startMove(e, node, el) {
    if (e.button !== 0) return;
    e.stopPropagation();
    selectNode(node.id);
    const start = [e.clientX, e.clientY], origin = [...node.pos];
    const key = `mg-move-${++state.seq}`;
    let moved = false;
    state.drag = "move";
    const move = (m) => {
      moved = true;
      node.pos = [origin[0] + (m.clientX - start[0]) / state.zoom, origin[1] + (m.clientY - start[1]) / state.zoom];
      el.style.left = `${node.pos[0]}px`;
      el.style.top = `${node.pos[1]}px`;
      drawLinks();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      state.drag = null;
      if (moved) commit(key);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function startLink(e, from) {
    const anchor = pinPoint(from.node, from.pin, from.dir);
    if (!anchor) return;
    state.drag = "link";
    const move = (m) => {
      const p = worldPoint(m);
      drawLinks(from.dir === "out" ? [anchor, p] : [p, anchor]);
    };
    const up = (u) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      state.drag = null;
      const target = document.elementFromPoint(u.clientX, u.clientY)?.closest(".mg-pin");
      if (target && target.dataset.dir !== from.dir) {
        const other = { node: Number(target.dataset.node), pin: target.dataset.pin };
        const [out, inp] = from.dir === "out" ? [from, other] : [other, from];
        if (out.node !== inp.node) {
          state.graph.links = state.graph.links.filter((l) => !(l.to.node === inp.node && l.to.pin === inp.pin));
          state.graph.links.push({ from: { node: out.node, pin: out.pin }, to: { node: inp.node, pin: inp.pin } });
          commit();
          render();
          return;
        }
      }
      drawLinks();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function frameAll() {
    const nodes = state.graph?.nodes || [];
    if (!nodes.length) return;
    const lo = [Math.min(...nodes.map((n) => n.pos[0])), Math.min(...nodes.map((n) => n.pos[1]))];
    const hi = [Math.max(...nodes.map((n) => n.pos[0] + NODE_WIDTH)), Math.max(...nodes.map((n) => n.pos[1] + 160))];
    const r = canvas.getBoundingClientRect();
    state.zoom = Math.min(1.2, Math.max(0.3, Math.min(r.width / (hi[0] - lo[0] + 80), r.height / (hi[1] - lo[1] + 80))));
    state.pan = [r.width / 2 - ((lo[0] + hi[0]) / 2) * state.zoom, r.height / 2 - ((lo[1] + hi[1]) / 2) * state.zoom];
    applyView();
    drawLinks();
  }

  // ---- Palette ------------------------------------------------------------------------------

  function showPalette(e) {
    const at = worldPoint(e);
    const box = canvas.getBoundingClientRect();
    const search = h("input", { class: "mg-search", placeholder: "Search nodes…", spellcheck: "false" });
    const list = h("div", { class: "mg-palette-list" });
    const fill = () => {
      const q = search.value.toLowerCase();
      list.replaceChildren();
      const byCategory = new Map();
      for (const n of catalog.nodes) {
        if (q && !n.type.toLowerCase().includes(q)) continue;
        if (n.type === "Output" && state.graph.nodes.some((x) => x.type === "Output")) continue;
        if (!byCategory.has(n.category)) byCategory.set(n.category, []);
        byCategory.get(n.category).push(n);
      }
      for (const [category, nodes] of byCategory) {
        list.append(h("div", { class: "mg-palette-cat" }, category));
        for (const n of nodes) list.append(h("div", { class: "mg-palette-item", onclick: () => addNode(n, at) }, n.type));
      }
    };
    search.addEventListener("input", fill);
    search.addEventListener("keydown", (k) => {
      k.stopPropagation();
      if (k.key === "Escape") hidePalette();
      if (k.key === "Enter") list.querySelector(".mg-palette-item")?.click();
    });
    fill();
    palette.replaceChildren(search, list);
    palette.style.left = `${Math.min(e.clientX - box.left, box.width - 220)}px`;
    palette.style.top = `${Math.min(e.clientY - box.top, box.height - 300)}px`;
    palette.classList.remove("hidden");
    search.focus();
  }

  function hidePalette() {
    palette.classList.add("hidden");
  }

  function addNode(entry, at) {
    hidePalette();
    if (!state.graph) return;
    const id = Math.max(0, ...state.graph.nodes.map((n) => n.id)) + 1;
    state.graph.nodes.push({ ...structuredClone(entry.template), id, pos: [Math.round(at[0]), Math.round(at[1])] });
    commit();
    render();
    selectNode(id);
  }

  function deleteSelected() {
    const id = state.selected;
    const node = state.graph?.nodes.find((n) => n.id === id);
    if (!node || node.type === "Output") return;
    state.graph.nodes = state.graph.nodes.filter((n) => n.id !== id);
    state.graph.links = state.graph.links.filter((l) => l.from.node !== id && l.to.node !== id);
    state.selected = null;
    commit();
    render();
  }

  // ---- Canvas input -------------------------------------------------------------------------

  canvas.addEventListener("pointerdown", (e) => {
    canvas.focus();
    if (!palette.contains(e.target)) hidePalette();
    if (e.button === 0 && e.target === canvas) selectNode(null);
    if (e.button !== 1 && e.button !== 2) return;
    e.preventDefault();
    const start = [e.clientX, e.clientY], origin = [...state.pan];
    let moved = false;
    const move = (m) => {
      if (Math.hypot(m.clientX - start[0], m.clientY - start[1]) > 4) moved = true;
      state.pan = [origin[0] + m.clientX - start[0], origin[1] + m.clientY - start[1]];
      applyView();
    };
    const up = (u) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (!moved && e.button === 2) showPalette(u);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("dblclick", (e) => {
    if (e.target === canvas || e.target === world || e.target === links) showPalette(e);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const p = [e.clientX - r.left, e.clientY - r.top];
    const zoom = Math.min(2, Math.max(0.25, state.zoom * Math.exp(-e.deltaY * 0.0015)));
    state.pan = [p[0] - ((p[0] - state.pan[0]) * zoom) / state.zoom, p[1] - ((p[1] - state.pan[1]) * zoom) / state.zoom];
    state.zoom = zoom;
    applyView();
  }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    if (e.target !== canvas) return;
    e.stopPropagation();
    if (e.code === "Delete" || e.code === "Backspace") deleteSelected();
    else if (e.code === "KeyF") frameAll();
    else if (e.code === "Escape") hidePalette();
    else if ((e.ctrlKey || e.metaKey) && e.code === "KeyZ") ed.undo();
    else if ((e.ctrlKey || e.metaKey) && e.code === "KeyY") ed.redo();
    else return;
    e.preventDefault();
  });

  return { element, open, close, update, get visible() { return !element.classList.contains("hidden"); } };
}

// The document parameters of a material as the Details panel shows it (so an edit of one
// field keeps the others: an edit replaces the material's whole entry).
export function materialParams(m) {
  const parameters = {};
  for (const p of m.parameters || []) if (p.kind !== "texture") parameters[p.name] = p.value;
  return {
    base_color: m.base_color, roughness: m.roughness, metallic: m.metallic, emissive: m.emissive, opacity: m.opacity,
    graph: m.graph || "", parameters, slab: m.slab, displacement: m.displacement,
  };
}
