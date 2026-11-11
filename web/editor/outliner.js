// World Outliner: environment and render settings, PCG volumes and the actor hierarchy (UE's
// outliner): expand / collapse, click to select (Ctrl toggles, Shift selects a range), drag rows
// onto an actor to attach them (onto "Level" to detach), an eye per actor, F2 or a double click
// on the name renames, search. Rows are virtualised (the forest has thousands of actors).

import { h, icon } from "./ui.js";
import { splitRef } from "./editor.js";

const ROW_HEIGHT = 24;
// Actor flags (editor/describe.rs).
const VISIBLE = 1, SHOWN = 2, ADDED = 4, FOLDER = 8, LIGHT = 16, MESH = 32, EMITTER = 64, EMPTY = 128;

export function createOutliner(ed) {
  const expanded = new Set(["world", "pcg", "level"]);
  let query = "";
  let rows = [];
  let anchor = null;
  let renaming = null;
  let dragged = null;

  const searchInput = h("input", { class: "ed-search", placeholder: "Search actors…", spellcheck: "false" });
  searchInput.addEventListener("input", () => {
    query = searchInput.value.trim().toLowerCase();
    update();
  });
  searchInput.addEventListener("keydown", (e) => e.stopPropagation());
  const count = h("span", { class: "ed-panel-count" });
  const spacer = h("div", { class: "ed-tree-spacer" });
  const body = h("div", { class: "ed-tree-rows" });
  const scroller = h("div", { class: "ed-tree" }, spacer, body);
  scroller.addEventListener("scroll", () => render());
  const element = h("section", { class: "ed-panel ed-outliner" },
    h("div", { class: "ed-panel-head" }, icon("folder"), h("span", { class: "ed-panel-title" }, "Outliner"), count),
    h("div", { class: "ed-panel-tools" }, h("div", { class: "ed-search-wrap" }, icon("search"), searchInput)),
    scroller);

  const matches = (text) => !query || text.toLowerCase().includes(query);

  function actorIcon(flags) {
    if (flags & LIGHT && !(flags & MESH)) return "light";
    if (flags & EMITTER && !(flags & MESH)) return "volume";
    if (flags & FOLDER || flags & EMPTY) return "folder";
    return "cube";
  }

  function build() {
    const s = ed.state;
    rows = [];
    if (!s) return;
    const group = (key, label, iconName, n, depth = 0) => {
      const open = expanded.has(key) || !!query;
      rows.push({ key, group: key, label, icon: iconName, count: n, depth, open });
      return open;
    };
    if (group("world", "World", "globe", 2)) {
      for (const [ref, label, iconName] of [["env", "Environment", "sun"], ["render", "Render settings", "gear"]]) {
        if (matches(label)) rows.push({ key: ref, ref, label, icon: iconName, depth: 1 });
      }
    }
    if (s.pcg.length && group("pcg", "PCG volumes", "volume", s.pcg.length)) {
      for (const v of s.pcg) {
        if (matches(v.def.name)) rows.push({ key: "p:" + v.def.id, ref: "p:" + v.def.id, label: v.def.name, sub: `${v.instances}`, icon: "volume", depth: 1, error: v.error });
      }
    }
    const a = s.actors;
    count.textContent = `${a.id.length.toLocaleString()} actors`;
    // The level: the actor tree. With a search, matching actors and their ancestors.
    let keep = null;
    if (query) {
      keep = new Set();
      for (let k = 0; k < a.id.length; k++) {
        if (!a.name[k].toLowerCase().includes(query)) continue;
        for (let id = a.id[k]; id !== -1 && !keep.has(id); id = ed.actorParent(id)) keep.add(id);
      }
    }
    const roots = ed.actorChildren.get(-1) || [];
    if (!group("level", "Level", "world", roots.length)) return;
    const walk = (id, depth) => {
      if (keep && !keep.has(id)) return;
      const k = ed.actorIndex.get(id);
      const flags = a.flags[k];
      const children = ed.actorChildren.get(id) || [];
      const open = children.length > 0 && (expanded.has("a:" + id) || !!keep);
      rows.push({
        key: "a:" + id,
        ref: "a:" + id,
        actor: id,
        label: a.name[k],
        icon: actorIcon(flags),
        depth,
        hidden: !(flags & SHOWN),
        visible: !!(flags & VISIBLE),
        badge: flags & ADDED ? "+" : null,
        eye: true,
        children: children.length,
        open,
      });
      if (open) for (const c of children) walk(c, depth + 1);
    };
    for (const id of roots) walk(id, 1);
  }

  function render() {
    spacer.style.height = rows.length * ROW_HEIGHT + "px";
    const top = scroller.scrollTop;
    const first = Math.max(0, Math.floor(top / ROW_HEIGHT) - 8);
    const last = Math.min(rows.length, Math.ceil((top + scroller.clientHeight) / ROW_HEIGHT) + 8);
    const selected = new Set(ed.selection);
    // Components selected: their actor's row is marked too.
    for (const ref of ed.selection) if (ref.startsWith("c:")) selected.add("a:" + ref.slice(2).split("/")[0]);
    const primary = ed.primary;
    body.replaceChildren();
    for (let k = first; k < last; k++) {
      const r = rows[k];
      const el = h("div", {
        class: "ed-tree-row" + (r.group ? " group" : "") + (selected.has(r.ref) ? " selected" : "") + (r.ref && r.ref === primary ? " primary" : "") + (r.hidden ? " dim" : ""),
        style: `top:${k * ROW_HEIGHT}px;padding-left:${8 + r.depth * 14}px`,
        "data-index": k,
        "data-ref": r.ref || r.group,
        draggable: r.actor !== undefined ? "true" : null,
      });
      if (r.group) el.append(icon("chevron", "ed-chevron" + (r.open ? " open" : "")));
      else if (r.children) el.append(icon("chevron", "ed-chevron ed-toggle" + (r.open ? " open" : "")));
      else el.append(h("span", { class: "ed-chevron-gap" }));
      el.append(icon(r.icon, "ed-row-icon"));
      if (renaming !== null && r.actor === renaming) {
        const input = h("input", { class: "ed-rename", value: r.label, spellcheck: "false" });
        const done = (commit) => {
          if (renaming === null) return;
          renaming = null;
          if (commit) ed.rename(r.actor, input.value.trim());
          update();
        };
        input.addEventListener("keydown", (e) => {
          e.stopPropagation();
          if (e.key === "Enter") done(true);
          if (e.key === "Escape") done(false);
        });
        input.addEventListener("blur", () => done(true));
        el.append(input);
        requestAnimationFrame(() => {
          input.focus();
          input.select();
        });
      } else el.append(h("span", { class: "ed-tree-label" }, r.label));
      if (r.badge) el.append(h("span", { class: "ed-badge" }, r.badge));
      if (r.error) el.append(h("span", { class: "ed-badge error", title: r.error }, "!"));
      if (r.sub) el.append(h("span", { class: "ed-tree-sub" }, r.sub));
      if (r.count !== undefined) el.append(h("span", { class: "ed-tree-count" }, r.count));
      else if (r.children) el.append(h("span", { class: "ed-tree-count" }, r.children));
      if (r.eye) {
        const eye = icon(r.visible ? "eye" : "eyeOff", "ed-eye");
        eye.title = r.visible ? "Hide (with everything below)" : "Show";
        eye.addEventListener("click", (e) => {
          e.stopPropagation();
          ed.apply({ op: "actor_visibility", actors: [r.actor], visible: !r.visible });
        });
        el.append(eye);
      }
      body.append(el);
    }
  }

  const rowOf = (e) => {
    const el = e.target.closest(".ed-tree-row");
    return el ? [el, rows[Number(el.dataset.index)]] : [null, null];
  };

  body.addEventListener("click", (e) => {
    const [, r] = rowOf(e);
    if (!r || e.target.closest(".ed-rename")) return;
    const toggle = (key) => {
      if (expanded.has(key)) expanded.delete(key);
      else expanded.add(key);
      update();
    };
    if (r.group) return toggle(r.group);
    if (r.children && e.target.closest(".ed-toggle")) return toggle(r.key);
    const k = rows.indexOf(r);
    if (e.shiftKey && anchor !== null) {
      const [a, b] = [Math.min(anchor, k), Math.max(anchor, k)];
      ed.select(rows.slice(a, b + 1).filter((x) => x.ref).map((x) => x.ref), "add");
    } else {
      ed.select([r.ref], e.ctrlKey || e.metaKey ? "toggle" : "set");
      anchor = k;
    }
  });
  body.addEventListener("dblclick", (e) => {
    const [, r] = rowOf(e);
    if (!r?.ref) return;
    if (r.actor !== undefined && e.target.closest(".ed-tree-label")) rename(r.actor);
    else ed.focusSelection();
  });

  // Drag rows onto an actor to attach them to it, onto "Level" to detach them.
  body.addEventListener("dragstart", (e) => {
    const [, r] = rowOf(e);
    if (r?.actor === undefined) return;
    // Dragging a selected row moves the whole selection's actors.
    dragged = ed.selection.includes(r.ref) ? ed.selectedIds("a") : [r.actor];
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("application/x-helios-actors", JSON.stringify(dragged));
  });
  const target = (e) => {
    const [el, r] = rowOf(e);
    if (!r || !dragged) return [null, undefined];
    if (r.group === "level") return [el, null];
    if (r.actor === undefined || dragged.includes(r.actor)) return [null, undefined];
    return [el, r.actor];
  };
  body.addEventListener("dragover", (e) => {
    const [el, parent] = target(e);
    body.querySelectorAll(".drop").forEach((x) => x.classList.remove("drop"));
    if (parent === undefined) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    el.classList.add("drop");
  });
  body.addEventListener("dragleave", () => body.querySelectorAll(".drop").forEach((x) => x.classList.remove("drop")));
  body.addEventListener("drop", (e) => {
    const [, parent] = target(e);
    e.preventDefault();
    const actors = dragged || [];
    dragged = null;
    if (parent === undefined) return;
    if (parent !== null) expanded.add("a:" + parent);
    ed.reparent(actors, parent);
  });
  body.addEventListener("dragend", () => {
    dragged = null;
    body.querySelectorAll(".drop").forEach((x) => x.classList.remove("drop"));
  });

  function update() {
    build();
    render();
  }

  // Scrolls a row into view.
  function reveal(ref) {
    const k = rows.findIndex((r) => r.ref === ref);
    if (k < 0) return;
    const y = k * ROW_HEIGHT;
    if (y < scroller.scrollTop || y > scroller.scrollTop + scroller.clientHeight - ROW_HEIGHT) scroller.scrollTop = y - scroller.clientHeight / 2;
  }

  // Expands the ancestors of an actor so its row exists.
  function expandTo(id) {
    expanded.add("level");
    for (let p = ed.actorParent(id); p !== -1 && p !== undefined; p = ed.actorParent(p)) expanded.add("a:" + p);
  }

  function rename(id) {
    expandTo(id);
    renaming = id;
    update();
    reveal("a:" + id);
  }

  return {
    element,
    rename,
    update() {
      // Show the primary selection (picked in the viewport: expand its ancestors).
      let primary = ed.primary;
      if (primary?.startsWith("c:")) primary = "a:" + primary.slice(2).split("/")[0];
      if (primary?.startsWith("a:") && ed.state) expandTo(splitRef(primary)[1]);
      update();
      if (primary) reveal(primary);
    },
  };
}
