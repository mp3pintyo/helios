// Transform gizmo (UE style): translate arrows with plane handles, rotation rings, scale handles,
// drawn into the 2D overlay canvas from the engine's camera and hit-tested in screen space.
// Dragging yields a delta (translation, rotation quaternion or scale factors) that the editor
// applies to every selected object around the pivot.

import {
  add, sub, scale, dot, cross, normalize, quatRotate, quatAxisAngle, rayPlane, closestOnAxis, segmentDistance, snap, DEG,
} from "./math.js";

const COLORS = { x: "#ec4a41", y: "#7bd44c", z: "#3f8cf5", hover: "#ffd54a", view: "#e8e8e8" };
const AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const PLANES = { xy: ["x", "y", "z"], xz: ["x", "z", "y"], yz: ["y", "z", "x"] };
const SIZE_PX = 105;
const HIT_PX = 8;

export class Gizmo {
  constructor() {
    this.mode = "translate"; // translate | rotate | scale
    this.space = "world"; // world | local
    this.hover = null;
    this.drag = null;
  }

  // Axis directions in world space for the pivot rotation (scale handles are always local, as in UE).
  basis(rotation) {
    const local = this.space === "local" || this.mode === "scale";
    const axis = (name) => (local ? normalize(quatRotate(rotation, AXES[name])) : AXES[name]);
    return { x: axis("x"), y: axis("y"), z: axis("z") };
  }

  // Screen-space shapes of the handles.
  shapes(view, pivot, rotation) {
    const center = view.project(pivot);
    if (!center) return null;
    const len = SIZE_PX * view.pixelSize(pivot);
    const basis = this.basis(rotation);
    const ends = {};
    for (const a of ["x", "y", "z"]) ends[a] = view.project(add(pivot, scale(basis[a], len)));
    const shapes = { center, len, basis, ends, rings: {}, planes: {} };
    if (this.mode === "rotate") {
      for (const a of ["x", "y", "z"]) {
        const [u, v] = perpendicular(basis[a]);
        const ring = [];
        for (let i = 0; i <= 64; i++) {
          const t = (i / 64) * Math.PI * 2;
          const p = add(pivot, add(scale(u, Math.cos(t) * len * 0.9), scale(v, Math.sin(t) * len * 0.9)));
          const front = dot(sub(p, pivot), sub(view.eye, pivot)) >= -1e-3;
          ring.push({ s: view.project(p), front });
        }
        shapes.rings[a] = ring;
      }
    } else if (this.mode === "translate") {
      for (const [name, [a, b]] of Object.entries(PLANES)) {
        const corner = (s, t) => view.project(add(pivot, add(scale(basis[a], s * len), scale(basis[b], t * len))));
        shapes.planes[name] = [corner(0.22, 0.22), corner(0.42, 0.22), corner(0.42, 0.42), corner(0.22, 0.42)];
      }
    }
    return shapes;
  }

  hit(view, pivot, rotation, mouse) {
    const s = this.shapes(view, pivot, rotation);
    if (!s) return null;
    let best = null;
    let bestDistance = HIT_PX;
    const consider = (handle, distance) => {
      if (distance < bestDistance) {
        bestDistance = distance;
        best = handle;
      }
    };
    if (this.mode === "rotate") {
      for (const [a, ring] of Object.entries(s.rings)) {
        for (let i = 0; i + 1 < ring.length; i++) {
          if (ring[i].s && ring[i + 1].s && ring[i].front) consider("r" + a, segmentDistance(mouse, ring[i].s, ring[i + 1].s));
        }
      }
      return best;
    }
    if (this.mode === "translate") {
      for (const [name, quad] of Object.entries(s.planes)) {
        if (quad.every(Boolean) && insideQuad(mouse, quad)) return name;
      }
      consider("view", Math.hypot(mouse[0] - s.center[0], mouse[1] - s.center[1]) - 2);
    } else {
      consider("uniform", Math.max(Math.abs(mouse[0] - s.center[0]), Math.abs(mouse[1] - s.center[1])) - 5);
    }
    for (const a of ["x", "y", "z"]) {
      if (s.ends[a]) consider(a, segmentDistance(mouse, s.center, s.ends[a]));
    }
    return best;
  }

  draw(ctx, view, pivot, rotation) {
    const s = this.shapes(view, pivot, rotation);
    if (!s) return;
    const active = this.drag ? this.drag.handle : this.hover;
    const color = (handle, base) => (handle === active ? COLORS.hover : base);
    ctx.save();
    ctx.lineCap = "round";
    if (this.mode === "rotate") {
      for (const [a, ring] of Object.entries(s.rings)) {
        for (const front of [false, true]) {
          ctx.beginPath();
          let pen = false;
          for (const p of ring) {
            if (!p.s || p.front !== front) {
              pen = false;
              continue;
            }
            if (pen) ctx.lineTo(p.s[0], p.s[1]);
            else ctx.moveTo(p.s[0], p.s[1]);
            pen = true;
          }
          ctx.strokeStyle = color("r" + a, COLORS[a]);
          ctx.globalAlpha = front ? 1 : 0.25;
          ctx.lineWidth = "r" + a === active ? 3.5 : 2.5;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      if (this.drag && this.drag.angle !== undefined) {
        ctx.fillStyle = "#fff";
        ctx.font = "600 12px Inter, system-ui, sans-serif";
        ctx.fillText(`${(this.drag.angle / DEG).toFixed(1)}°`, s.center[0] + 14, s.center[1] - 14);
      }
    } else {
      if (this.mode === "translate") {
        for (const [name, quad] of Object.entries(s.planes)) {
          if (!quad.every(Boolean)) continue;
          const [, , normalAxis] = PLANES[name];
          ctx.beginPath();
          quad.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          ctx.closePath();
          ctx.fillStyle = color(name, COLORS[normalAxis]);
          ctx.globalAlpha = name === active ? 0.75 : 0.35;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
      for (const a of ["x", "y", "z"]) {
        const end = s.ends[a];
        if (!end) continue;
        ctx.strokeStyle = ctx.fillStyle = color(a, COLORS[a]);
        ctx.lineWidth = a === active ? 3.5 : 2.5;
        ctx.beginPath();
        ctx.moveTo(s.center[0], s.center[1]);
        ctx.lineTo(end[0], end[1]);
        ctx.stroke();
        const dir = normalize([end[0] - s.center[0], end[1] - s.center[1], 0]);
        if (this.mode === "translate") {
          const n = [-dir[1], dir[0]];
          ctx.beginPath();
          ctx.moveTo(end[0] + dir[0] * 12, end[1] + dir[1] * 12);
          ctx.lineTo(end[0] + n[0] * 5, end[1] + n[1] * 5);
          ctx.lineTo(end[0] - n[0] * 5, end[1] - n[1] * 5);
          ctx.closePath();
          ctx.fill();
        } else {
          ctx.fillRect(end[0] - 5, end[1] - 5, 10, 10);
        }
      }
      ctx.fillStyle = color(this.mode === "translate" ? "view" : "uniform", COLORS.view);
      if (this.mode === "translate") {
        ctx.beginPath();
        ctx.arc(s.center[0], s.center[1], 4.5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillRect(s.center[0] - 6, s.center[1] - 6, 12, 12);
      }
    }
    ctx.restore();
  }

  begin(handle, view, pivot, rotation, mouse) {
    const basis = this.basis(rotation);
    const ray = view.ray(mouse[0], mouse[1]);
    const drag = { handle, pivot, basis, start: mouse, startRay: ray };
    if (handle in AXES) {
      drag.axis = basis[handle];
      drag.t0 = closestOnAxis(ray, pivot, drag.axis);
    } else if (handle in PLANES) {
      drag.normal = basis[PLANES[handle][2]];
      drag.hit0 = rayPlane(ray, pivot, drag.normal);
    } else if (handle === "view") {
      drag.normal = scale(view.forward, -1);
      drag.hit0 = rayPlane(ray, pivot, drag.normal);
    } else if (handle[0] === "r") {
      drag.axis = basis[handle[1]];
      drag.hit0 = rayPlane(ray, pivot, drag.axis);
      drag.edgeOn = Math.abs(dot(view.forward, drag.axis)) < 0.15 || !drag.hit0;
    }
    if (handle in AXES) {
      const s = this.shapes(view, pivot, rotation);
      const end = s.ends[handle] || s.center;
      const d = [end[0] - s.center[0], end[1] - s.center[1]];
      const screen = Math.hypot(d[0], d[1]);
      drag.screenLength = Math.max(20, screen);
      drag.screenDir = screen > 1e-3 ? [d[0] / screen, d[1] / screen] : [1, 0];
      // An axis pointing at the camera: the ray / axis intersection runs off to huge distances,
      // so drag along its short projection on screen instead.
      drag.worldPerPixel = s.len / Math.max(screen, 40);
      drag.grazing = screen < 40;
    }
    this.drag = drag;
  }

  // The drag's delta for the mouse at `mouse`: { translate }, { rotate } or { scale }.
  update(view, mouse, snapping) {
    const d = this.drag;
    if (!d) return null;
    const ray = view.ray(mouse[0], mouse[1]);
    if (this.mode === "translate") {
      let delta = [0, 0, 0];
      if (d.handle in AXES && d.grazing) {
        const m = [mouse[0] - d.start[0], mouse[1] - d.start[1]];
        delta = scale(d.axis, (m[0] * d.screenDir[0] + m[1] * d.screenDir[1]) * d.worldPerPixel);
      } else if (d.handle in AXES) {
        const t = closestOnAxis(ray, d.pivot, d.axis);
        if (t !== null && d.t0 !== null) delta = scale(d.axis, t - d.t0);
      } else if (d.hit0) {
        const hit = rayPlane(ray, d.pivot, d.normal);
        if (hit) delta = sub(hit, d.hit0);
      }
      if (snapping.grid > 0) delta = this.snapTranslation(delta, d, snapping.grid);
      return { translate: delta };
    }
    if (this.mode === "rotate") {
      let angle;
      if (d.edgeOn) {
        angle = (mouse[0] - d.start[0]) * 0.012;
      } else {
        const hit = rayPlane(ray, d.pivot, d.axis);
        if (!hit) return { rotate: quatAxisAngle(d.axis, d.angle || 0) };
        const v0 = sub(d.hit0, d.pivot), v = sub(hit, d.pivot);
        angle = Math.atan2(dot(cross(v0, v), d.axis), dot(v0, v));
        // Unwrap past ±180° while dragging around the ring.
        if (d.angle !== undefined) {
          while (angle - d.angle > Math.PI) angle -= 2 * Math.PI;
          while (angle - d.angle < -Math.PI) angle += 2 * Math.PI;
        }
      }
      d.rawAngle = angle;
      d.angle = snapping.angle > 0 ? snap(angle, snapping.angle * DEG) : angle;
      return { rotate: quatAxisAngle(d.axis, d.angle), axis: d.axis };
    }
    // Scale.
    let factor;
    if (d.handle === "uniform") {
      factor = 1 + (mouse[0] - d.start[0] - (mouse[1] - d.start[1])) / 120;
    } else {
      const m = [mouse[0] - d.start[0], mouse[1] - d.start[1]];
      factor = 1 + (m[0] * d.screenDir[0] + m[1] * d.screenDir[1]) / d.screenLength;
    }
    if (snapping.scale > 0) factor = 1 + snap(factor - 1, snapping.scale);
    factor = Math.max(0.01, factor);
    const factors = d.handle === "uniform" ? [factor, factor, factor] : ["x", "y", "z"].map((a) => (a === d.handle ? factor : 1));
    return { scale: factors };
  }

  // Grid snapping: world axes snap the pivot's coordinates to the grid, local axes the distance.
  snapTranslation(delta, d, grid) {
    if (this.space === "local") {
      if (d.axis) return scale(d.axis, snap(dot(delta, d.axis), grid));
      return delta.map((v) => snap(v, grid));
    }
    const target = add(d.pivot, delta);
    return target.map((v, i) => (Math.abs(delta[i]) > 1e-9 ? snap(v, grid) - d.pivot[i] : 0));
  }

  end() {
    this.drag = null;
  }
}

// Two unit vectors spanning the plane perpendicular to n.
function perpendicular(n) {
  const helper = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = normalize(cross(n, helper));
  return [u, cross(n, u)];
}

function insideQuad(p, quad) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i], b = quad[(i + 1) % 4];
    const c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (c !== 0) {
      if (sign === 0) sign = Math.sign(c);
      else if (Math.sign(c) !== sign) return false;
    }
  }
  return true;
}

