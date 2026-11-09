// World Outliner: environment, lights, PCG volumes and the instances grouped by mesh or
// material, with search. Rows are virtualised (the forest has thousands of instances).

import { h, icon, select } from "./ui.js";

const ROW_HEIGHT = 24;

export function createOutliner(ed) {
  const expanded = new Set(["world", "lights", "pcg", "meshes"]);
  let groupBy = "mesh";
  let query = "";
  let rows = [];
  let anchor = null;

  const searchInput = h("input", { class: "ed-search", placeholder: "Search…", spellcheck: "false" });
  searchInput.addEventListener("input", () => {
    query = searchInput.value.trim().toLowerCase();
    update();
  });
  searchInput.addEventListener("keydown", (e) => e.stopPropagation());
  const groupSelect = select([["mesh", "By mesh"], ["material", "By material"]], groupBy, (v) => {
    groupBy = v;
    update();
  });
  const count = h("span", { class: "ed-panel-count" });
  const spacer = h("div", { class: "ed-tree-spacer" });
  const body = h("div", { class: "ed-tree-rows" });
  const scroller = h("div", { class: "ed-tree" }, spacer, body);
  scroller.addEventListener("scroll", () => render());
  const element = h("section", { class: "ed-panel ed-outliner" },
    h("div", { class: "ed-panel-head" }, icon("folder"), h("span", { class: "ed-panel-title" }, "Outliner"), count),
    h("div", { class: "ed-panel-tools" }, h("div", { class: "ed-search-wrap" }, icon("search"), searchInput), groupSelect),
    scroller);

  const matches = (text) => !query || text.toLowerCase().includes(query);

  function build() {
    const s = ed.state;
    rows = [];
    if (!s) return;
    const group = (key, label, iconName, n, depth = 0) => {
      rows.push({ key, group: key, label, icon: iconName, count: n, depth, open: expanded.has(key) || !!query });
      return expanded.has(key) || !!query;
    };
    if (group("world", "World", "globe", 2)) {
      for (const [ref, label, iconName] of [["env", "Environment", "sun"], ["render", "Render settings", "gear"]]) {
        if (matches(label)) rows.push({ key: ref, ref, label, icon: iconName, depth: 1 });
      }
    }
    const lights = s.lights.filter((l) => l.def.enabled);
    if (group("lights", "Lights", "light", lights.length)) {
      for (const l of lights) {
        const label = `${l.def.shape === "spot" ? "Spot" : "Point"} light ${l.index}`;
        if (matches(label)) rows.push({ key: "l:" + l.index, ref: "l:" + l.index, label, icon: l.def.shape === "spot" ? "spot" : "light", depth: 1, badge: l.added ? "+" : null });
      }
    }
    if (group("pcg", "PCG volumes", "volume", s.pcg.length)) {
      for (const v of s.pcg) {
        if (matches(v.def.name)) rows.push({ key: "p:" + v.def.id, ref: "p:" + v.def.id, label: v.def.name, sub: `${v.instances}`, icon: "volume", depth: 1, error: v.error });
      }
    }
    // Instances (not PCG-spawned or deleted), grouped.
    const inst = s.instances;
    const groups = new Map();
    let total = 0;
    for (let i = 0; i < inst.mesh.length; i++) {
      if (inst.visible[i] === 2 || inst.owner[i] === 2 || inst.owner[i] === 3) continue;
      total++;
      const key = groupBy === "mesh" ? inst.mesh[i] : inst.material[i];
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(i);
    }
    if (group("meshes", "Static meshes", "cube", total)) {
      const name = (k) => (groupBy === "mesh" ? s.meshes[k]?.name : s.materials[k]?.name) || "unnamed";
      for (const key of [...groups.keys()].sort((a, b) => name(a).localeCompare(name(b)))) {
        const members = groups.get(key);
        const label = name(key);
        const groupMatch = matches(label);
        const shown = members.filter((i) => groupMatch || matches(`${ed.meshName(i)} #${i}`));
        if (!shown.length) continue;
        const gkey = `g:${groupBy}:${key}`;
        if (group(gkey, label, groupBy === "mesh" ? "mesh" : "material", members.length, 1)) {
          for (const i of shown) {
            const other = groupBy === "mesh" ? s.materials[inst.material[i]]?.name : ed.meshName(i);
            rows.push({ key: "i:" + i, ref: "i:" + i, label: `${ed.meshName(i)} #${i}`, sub: other, icon: "cube", depth: 2, hidden: inst.visible[i] === 0, badge: inst.owner[i] === 1 ? "+" : null, eye: true, index: i });
          }
        }
      }
    }
    count.textContent = `${total} instances`;
  }

  function render() {
    spacer.style.height = rows.length * ROW_HEIGHT + "px";
    const top = scroller.scrollTop;
    const first = Math.max(0, Math.floor(top / ROW_HEIGHT) - 8);
    const last = Math.min(rows.length, Math.ceil((top + scroller.clientHeight) / ROW_HEIGHT) + 8);
    const selected = new Set(ed.selection);
    const primary = ed.primary;
    body.replaceChildren();
    for (let k = first; k < last; k++) {
      const r = rows[k];
      const el = h("div", {
        class: "ed-tree-row" + (r.group ? " group" : "") + (selected.has(r.ref) ? " selected" : "") + (r.ref && r.ref === primary ? " primary" : "") + (r.hidden ? " dim" : ""),
        style: `top:${k * ROW_HEIGHT}px;padding-left:${8 + r.depth * 16}px`,
        "data-index": k,
      });
      if (r.group) el.append(icon("chevron", "ed-chevron" + (r.open ? " open" : "")));
      el.append(icon(r.icon, "ed-row-icon"), h("span", { class: "ed-tree-label" }, r.label));
      if (r.badge) el.append(h("span", { class: "ed-badge" }, r.badge));
      if (r.error) el.append(h("span", { class: "ed-badge error", title: r.error }, "!"));
      if (r.sub) el.append(h("span", { class: "ed-tree-sub" }, r.sub));
      if (r.count !== undefined) el.append(h("span", { class: "ed-tree-count" }, r.count));
      if (r.eye) {
        const eye = icon(r.hidden ? "eyeOff" : "eye", "ed-eye");
        eye.addEventListener("click", (e) => {
          e.stopPropagation();
          ed.apply({ op: "visibility", instances: [r.index], visible: r.hidden });
        });
        el.append(eye);
      }
      body.append(el);
    }
  }

  body.addEventListener("click", (e) => {
    const el = e.target.closest(".ed-tree-row");
    if (!el) return;
    const k = Number(el.dataset.index);
    const r = rows[k];
    if (r.group) {
      if (expanded.has(r.group)) expanded.delete(r.group);
      else expanded.add(r.group);
      update();
      return;
    }
    if (e.shiftKey && anchor !== null) {
      const [a, b] = [Math.min(anchor, k), Math.max(anchor, k)];
      ed.select(rows.slice(a, b + 1).filter((x) => x.ref).map((x) => x.ref), "add");
    } else {
      ed.select([r.ref], e.ctrlKey || e.metaKey ? "toggle" : "set");
      anchor = k;
    }
  });
  body.addEventListener("dblclick", (e) => {
    const el = e.target.closest(".ed-tree-row");
    if (el && rows[Number(el.dataset.index)].ref) ed.focusSelection();
  });

  function update() {
    build();
    render();
  }

  // Scrolls the primary selection into view (after a viewport pick).
  function reveal(ref) {
    const k = rows.findIndex((r) => r.ref === ref);
    if (k < 0) return;
    const y = k * ROW_HEIGHT;
    if (y < scroller.scrollTop || y > scroller.scrollTop + scroller.clientHeight - ROW_HEIGHT) scroller.scrollTop = y - scroller.clientHeight / 2;
  }

  return {
    element,
    update() {
      // Expand the group of a primary instance picked in the viewport, then show it.
      const primary = ed.primary;
      if (primary?.startsWith("i:") && ed.state) {
        const i = Number(primary.slice(2));
        const key = groupBy === "mesh" ? ed.state.instances.mesh[i] : ed.state.instances.material[i];
        expanded.add("meshes");
        expanded.add(`g:${groupBy}:${key}`);
      }
      update();
      if (primary) reveal(primary);
    },
  };
}
