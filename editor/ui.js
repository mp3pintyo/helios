// DOM helpers for the editor panels: elements, collapsible sections, property rows and
// UE-style numeric fields (drag the label to scrub the value).

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style") el.style.cssText = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "html") el.innerHTML = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const ICONS = {
  cube: '<path d="M8 1.5 14 4.75v6.5L8 14.5 2 11.25v-6.5z M2 4.75 8 8l6-3.25 M8 8v6.5"/>',
  light: '<circle cx="8" cy="6.5" r="3.5"/><path d="M6.5 11.5h3 M7 14h2 M8 0.8v1 M2.6 3l.8.6 M13.4 3l-.8.6"/>',
  spot: '<path d="M5 2h6l-1.5 4h-3z M6.5 6 3 14h10L9.5 6"/>',
  volume: '<rect x="2" y="2" width="12" height="12" rx="1" stroke-dasharray="2.2 1.6"/><circle cx="6" cy="9" r="1.2"/><circle cx="10" cy="6" r="1.2"/><circle cx="10.5" cy="11" r="1"/>',
  globe: '<circle cx="8" cy="8" r="6"/><path d="M2 8h12 M8 2c2 2 2 10 0 12 M8 2c-2 2-2 10 0 12"/>',
  sun: '<circle cx="8" cy="8" r="3"/><path d="M8 1v2 M8 13v2 M1 8h2 M13 8h2 M3 3l1.4 1.4 M11.6 11.6 13 13 M3 13l1.4-1.4 M11.6 4.4 13 3"/>',
  gear: '<circle cx="8" cy="8" r="2.2"/><path d="M8 1.5v2 M8 12.5v2 M1.5 8h2 M12.5 8h2 M3.4 3.4l1.4 1.4 M11.2 11.2l1.4 1.4 M3.4 12.6l1.4-1.4 M11.2 4.8l1.4-1.4"/>',
  eye: '<path d="M1 8s2.5-4.5 7-4.5S15 8 15 8s-2.5 4.5-7 4.5S1 8 1 8z"/><circle cx="8" cy="8" r="2"/>',
  eyeOff: '<path d="M1 8s2.5-4.5 7-4.5S15 8 15 8s-2.5 4.5-7 4.5S1 8 1 8z" opacity=".35"/><path d="M2 2l12 12"/>',
  folder: '<path d="M1.5 4.5v8h13v-7H8L6.5 3.5h-5z"/>',
  chevron: '<path d="M6 4l4 4-4 4"/>',
  material: '<circle cx="8" cy="8" r="6"/><path d="M4 6.5c2-2 6-2 8 0" opacity=".6"/>',
  mesh: '<path d="M2 12 8 2l6 10z M5 7h6 M8 2v10"/>',
  undo: '<path d="M5 3 2 6l3 3 M2 6h7.5a4 4 0 0 1 0 8H6"/>',
  redo: '<path d="M11 3l3 3-3 3 M14 6H6.5a4 4 0 0 0 0 8H10"/>',
  move: '<path d="M8 1v14 M1 8h14 M8 1 6 3 M8 1l2 2 M8 15l-2-2 M8 15l2-2 M1 8l2-2 M1 8l2 2 M15 8l-2-2 M15 8l-2 2"/>',
  rotate: '<path d="M13 8a5 5 0 1 1-1.5-3.6 M13 2v3h-3"/>',
  scale: '<rect x="2" y="7" width="7" height="7"/><path d="M9 7l5-5 M10 2h4v4"/>',
  magnet: '<path d="M3 2v6a5 5 0 0 0 10 0V2h-3v6a2 2 0 0 1-4 0V2z M3 5h3 M10 5h3"/>',
  save: '<path d="M2 2h9.5L14 4.5V14H2z M5 2v4h5V2 M4.5 14V9.5h7V14"/>',
  open: '<path d="M1.5 4.5v8.5h11l2-6.5H4L2.5 12 M1.5 4.5V3h4l1.5 1.5h5V6.5"/>',
  file: '<path d="M3.5 1.5h6l3 3v10h-9z M9.5 1.5v3h3"/>',
  play: '<path d="M4.5 2.5v11l9-5.5z"/>',
  plus: '<path d="M8 2.5v11 M2.5 8h11"/>',
  trash: '<path d="M2.5 4h11 M6 4V2.5h4V4 M3.5 4l.8 10h7.4l.8-10"/>',
  copy: '<rect x="5" y="5" width="9" height="9" rx="1"/><path d="M11 5V2H2v9h3"/>',
  focus: '<path d="M1.5 5V1.5H5 M11 1.5h3.5V5 M14.5 11v3.5H11 M5 14.5H1.5V11"/><circle cx="8" cy="8" r="2"/>',
  search: '<circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/>',
  dice: '<rect x="2" y="2" width="12" height="12" rx="2"/><circle cx="5.5" cy="5.5" r=".9"/><circle cx="10.5" cy="10.5" r=".9"/><circle cx="8" cy="8" r=".9"/>',
  refresh: '<path d="M13.5 8A5.5 5.5 0 1 1 12 4.2 M13.5 1.8v3h-3"/>',
  world: '<path d="M8 1.5 14 5v6l-6 3.5L2 11V5z"/>',
  local: '<rect x="3" y="3" width="10" height="10" transform="rotate(20 8 8)"/>',
  grid: '<path d="M1.5 5.5h13 M1.5 10.5h13 M5.5 1.5v13 M10.5 1.5v13"/>',
  close: '<path d="M4.5 4.5l7 7 M11.5 4.5l-7 7"/>',
  import: '<path d="M8 1.5v9 M4.5 7 8 10.5 11.5 7 M2 12v2.5h12V12"/>',
};

export function icon(name, cls = "") {
  const span = document.createElement("span");
  span.className = "ed-icon " + cls;
  span.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
  return span;
}

// A collapsible Details section; open state survives re-renders.
const openSections = new Map();
export function section(title, body, { open = true, actions = null } = {}) {
  const key = title;
  const isOpen = openSections.has(key) ? openSections.get(key) : open;
  const el = h("div", { class: "ed-section" + (isOpen ? " open" : "") });
  const head = h("div", { class: "ed-section-head" }, icon("chevron", "ed-chevron"), h("span", {}, title), actions ? h("span", { class: "ed-section-actions" }, actions) : null);
  head.addEventListener("click", (e) => {
    if (e.target.closest(".ed-section-actions")) return;
    const now = !el.classList.contains("open");
    el.classList.toggle("open", now);
    openSections.set(key, now);
  });
  el.append(head, h("div", { class: "ed-section-body" }, body));
  return el;
}

export function row(label, ...controls) {
  return h("div", { class: "ed-row" }, h("label", { class: "ed-row-label", title: label }, label), h("div", { class: "ed-row-value" }, ...controls));
}

const format = (v, precision) => (Number.isFinite(v) ? Number(v.toFixed(precision)).toString() : "0");

// A numeric field; dragging its label (or the field's left edge tag) scrubs the value.
// onInput fires while editing (live), onCommit when done.
export function numberField(value, { step = 0.1, min = -Infinity, max = Infinity, precision = 3, tag = null, tagClass = "", onInput, onCommit } = {}) {
  const input = h("input", { class: "ed-number", type: "text", inputmode: "decimal", spellcheck: "false", value: format(value, precision) });
  const clamp = (v) => Math.min(max, Math.max(min, v));
  let current = value;
  const emit = (v, done) => {
    current = clamp(v);
    input.value = format(current, precision);
    (done ? onCommit || onInput : onInput || onCommit)?.(current);
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") {
      input.value = format(current, precision);
      input.blur();
    }
    e.stopPropagation();
  });
  input.addEventListener("change", () => {
    const v = parseFloat(input.value.replace(",", "."));
    if (Number.isFinite(v)) emit(v, true);
    else input.value = format(current, precision);
  });
  const wrap = h("div", { class: "ed-number-wrap" + (tag ? " tagged" : "") });
  if (tag) {
    const t = h("span", { class: "ed-number-tag " + tagClass }, tag);
    scrub(t, () => current, step, (v, done) => emit(v, done));
    wrap.append(t);
  }
  wrap.append(input);
  wrap.setValue = (v) => {
    current = v;
    if (document.activeElement !== input) input.value = format(v, precision);
  };
  return wrap;
}

// Horizontal drag on `el` changes a value by `step` per pixel (Shift: ×10, Alt: ×0.1).
export function scrub(el, get, step, set) {
  el.classList.add("ed-scrub");
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    const start = e.clientX;
    const base = get();
    let moved = false;
    const move = (m) => {
      const k = m.shiftKey ? 10 : m.altKey ? 0.1 : 1;
      moved = true;
      set(base + (m.clientX - start) * step * k, false);
    };
    const up = () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      if (moved) set(get(), true);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
}

// Three numeric fields tagged X / Y / Z (red, green, blue tags as in UE).
export function vectorField(values, options) {
  const fields = ["X", "Y", "Z"].map((tag, i) =>
    numberField(values[i], {
      ...options,
      tag,
      tagClass: "axis-" + tag.toLowerCase(),
      onInput: options.onInput && ((v) => options.onInput(i, v)),
      onCommit: options.onCommit && ((v) => options.onCommit(i, v)),
    }),
  );
  const el = h("div", { class: "ed-vector" }, fields);
  el.setValues = (v) => fields.forEach((f, i) => f.setValue(v[i]));
  return el;
}

export function slider(value, { min = 0, max = 1, step = 0.01, precision = 2, onInput, onCommit }) {
  const range = h("input", { class: "ed-slider", type: "range", min, max, step, value });
  const number = numberField(value, { step, min, max, precision, onCommit: (v) => {
    range.value = v;
    (onCommit || onInput)?.(v);
  } });
  const fill = () => range.style.setProperty("--fill", `${((range.value - min) / (max - min)) * 100}%`);
  fill();
  range.addEventListener("input", () => {
    fill();
    number.setValue(parseFloat(range.value));
    onInput?.(parseFloat(range.value));
  });
  range.addEventListener("change", () => (onCommit || onInput)?.(parseFloat(range.value)));
  return h("div", { class: "ed-slider-wrap" }, range, number);
}

export function checkbox(value, onChange) {
  const input = h("input", { type: "checkbox", class: "ed-check" });
  input.checked = !!value;
  input.addEventListener("change", () => onChange(input.checked));
  return input;
}

export function select(options, value, onChange) {
  const el = h("select", { class: "ed-select" }, options.map((o) => {
    const [v, label] = Array.isArray(o) ? o : [o, o];
    const opt = h("option", { value: v }, label);
    if (v === value) opt.selected = true;
    return opt;
  }));
  el.addEventListener("change", () => onChange(el.value));
  el.addEventListener("keydown", (e) => e.stopPropagation());
  return el;
}

// Linear RGB <-> sRGB hex for colour pickers (the engine's colours are linear).
const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
export const linearToHex = (rgb) =>
  "#" + rgb.map((c) => Math.round(Math.min(1, Math.max(0, toSrgb(c))) * 255).toString(16).padStart(2, "0")).join("");
export const hexToLinear = (hex) => [1, 3, 5].map((i) => toLinear(parseInt(hex.slice(i, i + 2), 16) / 255));

// A colour swatch (linear RGB in and out; values above 1 are scaled into the picker).
export function colorField(rgb, onInput, onCommit) {
  const peak = Math.max(1, ...rgb);
  const input = h("input", { type: "color", class: "ed-color", value: linearToHex(rgb.map((c) => c / peak)) });
  input.addEventListener("input", () => onInput(hexToLinear(input.value).map((c) => c * peak)));
  input.addEventListener("change", () => (onCommit || onInput)(hexToLinear(input.value).map((c) => c * peak)));
  return h("div", { class: "ed-color-wrap" }, input, h("span", { class: "ed-color-hex" }, input.value.toUpperCase()));
}

export function button(label, onClick, { iconName = null, cls = "", title = "" } = {}) {
  return h("button", { class: "ed-button " + cls, title, onclick: onClick }, iconName ? icon(iconName) : null, label ? h("span", {}, label) : null);
}
