// Content browser / Place Actors: engine content (shapes, nature prefabs), lights, PCG presets,
// the scene's own meshes, materials and imported glTF models. Click places in front of the
// camera, drag drops at the cursor; materials drop onto instances. glTF: import by URL or drop
// a .glb anywhere on the editor.

import { h, icon, button } from "./ui.js";

const TILE_ICONS = { Basic: "cube", Nature: "mesh", Imported: "import", Lights: "light", PCG: "volume", Scene: "mesh" };

export function createContentBrowser(ed) {
  let tab = "place";
  let query = "";
  const tabs = {};
  const list = h("div", { class: "ed-content-list" });
  const search = h("input", { class: "ed-search", placeholder: "Search content…", spellcheck: "false" });
  search.addEventListener("input", () => {
    query = search.value.trim().toLowerCase();
    render();
  });
  search.addEventListener("keydown", (e) => e.stopPropagation());
  for (const [id, label] of [["place", "Place"], ["materials", "Materials"], ["meshes", "Meshes"]]) {
    tabs[id] = h("button", { class: "ed-tab", onclick: () => ((tab = id), render()) }, label);
  }
  const urlInput = h("input", { class: "ed-text", placeholder: "models/Duck.glb", spellcheck: "false" });
  urlInput.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") importUrl();
  });
  const importUrl = () => {
    const url = urlInput.value.trim();
    if (url) ed.importModel(url, null);
  };
  const element = h("section", { class: "ed-panel ed-content" },
    h("div", { class: "ed-panel-head" }, icon("folder"), h("span", { class: "ed-panel-title" }, "Content")),
    h("div", { class: "ed-tabs" }, Object.values(tabs)),
    h("div", { class: "ed-panel-tools" }, h("div", { class: "ed-search-wrap" }, icon("search"), search)),
    list,
    h("div", { class: "ed-import" },
      h("div", { class: "ed-import-title" }, icon("import"), "Import glTF"),
      h("div", { class: "ed-import-row" }, urlInput, button("Import", importUrl, { cls: "primary" })),
      h("div", { class: "ed-import-hint" }, "…or drop a .glb onto the viewport")));

  const matches = (text) => !query || text.toLowerCase().includes(query);

  function tile({ label, kind, id, category, swatch = null, sub = null }) {
    const el = h("div", { class: "ed-tile", draggable: "true", title: `${label}\nClick: place in front of the camera · drag: drop into the viewport` },
      swatch ? h("span", { class: "ed-swatch", style: `background:${swatch}` }) : icon(kind === "light" ? (id === "spot" ? "spot" : "light") : kind === "pcg" ? "volume" : TILE_ICONS[category] || "cube", "ed-tile-icon"),
      h("span", { class: "ed-tile-label" }, label),
      sub ? h("span", { class: "ed-tile-sub" }, sub) : null);
    el.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("application/x-helios", JSON.stringify({ kind, id }));
      e.dataTransfer.effectAllowed = "copy";
    });
    el.addEventListener("click", async () => {
      if (kind === "material") {
        ed.select(["m:" + id]);
        return;
      }
      const at = await ed.placementPoint();
      if (kind === "item") ed.place(id, at);
      else if (kind === "light") ed.addLight(id, at);
      else if (kind === "pcg") ed.addPcg(id, at);
    });
    return el;
  }

  function group(title, tiles) {
    tiles = tiles.filter((t) => matches(t.label) || matches(t.id));
    if (!tiles.length) return null;
    return h("div", { class: "ed-content-group" }, h("div", { class: "ed-content-group-title" }, title), h("div", { class: "ed-tiles" }, tiles.map(tile)));
  }

  const swatchOf = (m) => {
    const peak = Math.max(1, ...m.base_color);
    return `rgb(${m.base_color.map((c) => Math.round(Math.pow(Math.min(1, c / peak), 1 / 2.2) * 255)).join(",")})`;
  };

  function render() {
    for (const [id, el] of Object.entries(tabs)) el.classList.toggle("active", id === tab);
    const s = ed.state;
    list.replaceChildren();
    if (!s) return;
    const sections = [];
    if (tab === "place") {
      const byCategory = new Map();
      for (const item of s.content.items) {
        if (!byCategory.has(item.category)) byCategory.set(item.category, []);
        byCategory.get(item.category).push({ label: item.label, kind: "item", id: item.id, category: item.category, sub: item.parts.length > 1 ? `${item.parts.length} parts` : null });
      }
      sections.push(group("Basic", byCategory.get("Basic") || []));
      sections.push(group("Lights", [{ label: "Point light", kind: "light", id: "point" }, { label: "Spot light", kind: "light", id: "spot" }]));
      sections.push(group("PCG volumes", s.content.presets.map((p) => ({ label: p.label, kind: "pcg", id: p.id, category: "PCG" }))));
      sections.push(group("Nature", byCategory.get("Nature") || []));
      sections.push(group("Imported", byCategory.get("Imported") || []));
    } else if (tab === "materials") {
      const sceneMaterials = s.materials.filter((m) => m.name && !m.name.startsWith("engine/"));
      const engineMaterials = s.content.materials.map((name) => s.materials.find((m) => m.name === name) || { name, base_color: [0.5, 0.5, 0.5], kind: "not created yet" });
      const asTile = (m) => ({ label: m.name.replace("engine/", ""), kind: "material", id: m.name, swatch: swatchOf(m), sub: m.kind });
      sections.push(group("Scene materials", sceneMaterials.map(asTile)));
      sections.push(group("Engine materials", engineMaterials.map(asTile)));
      sections.push(h("div", { class: "ed-note" }, "Drag a material onto an object to assign it; click to edit its parameters."));
    } else {
      const meshes = s.meshes.filter((m) => m.name && !m.name.startsWith("engine/"));
      sections.push(group("Scene meshes", meshes.map((m) => ({ label: m.name, kind: "item", id: "mesh:" + m.name, category: "Scene", sub: `${(m.triangles / 1000).toFixed(m.triangles < 10000 ? 1 : 0)}k tris` }))));
    }
    list.append(...sections.filter(Boolean));
  }

  return { element, update: render };
}
