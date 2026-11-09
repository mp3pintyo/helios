// Small vector / quaternion / matrix helpers for the editor overlay and gizmo.
// Vectors are [x, y, z] arrays, quaternions [x, y, z, w], matrices column-major Float32Array(16)
// (the engine's convention: right-handed, +Y up, reverse-Z projection).

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const mul = (a, b) => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const length = (a) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a) => {
  const l = length(a);
  return l > 1e-12 ? scale(a, 1 / l) : [0, 0, 0];
};
export const lerp = (a, b, t) => add(a, scale(sub(b, a), t));
export const DEG = Math.PI / 180;

// ---- Quaternions ------------------------------------------------------------------------------

export const quatIdentity = () => [0, 0, 0, 1];

export function quatMul(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function quatAxisAngle(axis, angle) {
  const s = Math.sin(angle / 2);
  const n = normalize(axis);
  return [n[0] * s, n[1] * s, n[2] * s, Math.cos(angle / 2)];
}

export function quatNormalize(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return q.map((v) => v / l);
}

export function quatRotate(q, v) {
  const u = [q[0], q[1], q[2]];
  const t = scale(cross(u, v), 2);
  return add(add(v, scale(t, q[3])), cross(u, t));
}

// Euler angles in degrees, applied as yaw (Y), then pitch (X), then roll (Z): q = Ry · Rx · Rz.
export function quatFromEuler([x, y, z]) {
  return quatMul(quatMul(quatAxisAngle([0, 1, 0], y * DEG), quatAxisAngle([1, 0, 0], x * DEG)), quatAxisAngle([0, 0, 1], z * DEG));
}

export function quatToEuler(q) {
  const m = mat3FromQuat(q);
  // m = Ry Rx Rz: m[7] (row 1, col 2) = -sin(x).
  const sx = Math.max(-1, Math.min(1, -m[7]));
  const x = Math.asin(sx);
  let y, z;
  if (Math.abs(sx) < 0.9999) {
    y = Math.atan2(m[6], m[8]);
    z = Math.atan2(m[1], m[4]);
  } else {
    y = Math.atan2(-m[2], m[0]);
    z = 0;
  }
  return [x / DEG, y / DEG, z / DEG];
}

// Column-major 3x3 rotation matrix of a quaternion.
export function mat3FromQuat([x, y, z, w]) {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
  ];
}

// ---- Matrices ---------------------------------------------------------------------------------

export function mat4Compose(t, r, s) {
  const m = mat3FromQuat(r);
  return new Float32Array([
    m[0] * s[0], m[1] * s[0], m[2] * s[0], 0,
    m[3] * s[1], m[4] * s[1], m[5] * s[1], 0,
    m[6] * s[2], m[7] * s[2], m[8] * s[2], 0,
    t[0], t[1], t[2], 1,
  ]);
}

export function mat4TransformPoint(m, p) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return [x, y, z, w];
}

export function mat4Invert(m) {
  const inv = new Float32Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-20) return inv;
  for (let i = 0; i < 16; i++) inv[i] /= det;
  return inv;
}

// ---- The engine camera ------------------------------------------------------------------------

// heliosEditorCamera(): viewProj (16), eye (3), forward (3), fov, CSS width, CSS height.
export class View {
  constructor(data) {
    this.viewProj = data.subarray(0, 16);
    this.inverse = mat4Invert(this.viewProj);
    this.eye = [data[16], data[17], data[18]];
    this.forward = [data[19], data[20], data[21]];
    this.fov = data[22];
    this.width = data[23];
    this.height = data[24];
  }

  // World point -> [x, y] in CSS pixels, plus whether it is in front of the camera.
  project(p) {
    const c = mat4TransformPoint(this.viewProj, p);
    if (c[3] <= 1e-6) return null;
    return [((c[0] / c[3]) * 0.5 + 0.5) * this.width, (0.5 - (c[1] / c[3]) * 0.5) * this.height];
  }

  // The world ray through CSS pixel (x, y).
  ray(x, y) {
    const nx = (x / this.width) * 2 - 1;
    const ny = 1 - (y / this.height) * 2;
    const unproject = (z) => {
      const p = mat4TransformPoint(this.inverse, [nx, ny, z]);
      return [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
    };
    const near = unproject(1);
    const far = unproject(0.5);
    return { origin: this.eye, dir: normalize(sub(far, near)) };
  }

  // World units per CSS pixel at point p (for constant screen-size handles).
  pixelSize(p) {
    const depth = Math.max(0.01, dot(sub(p, this.eye), this.forward));
    return (2 * depth * Math.tan(this.fov / 2)) / this.height;
  }
}

// Ray–plane intersection: the hit point or null.
export function rayPlane(ray, point, normal) {
  const d = dot(ray.dir, normal);
  if (Math.abs(d) < 1e-6) return null;
  const t = dot(sub(point, ray.origin), normal) / d;
  return t > 0 ? add(ray.origin, scale(ray.dir, t)) : null;
}

// Parameter along the line (point, axis) closest to the ray.
export function closestOnAxis(ray, point, axis) {
  const w = sub(ray.origin, point);
  const a = dot(axis, axis), b = dot(axis, ray.dir), c = dot(ray.dir, ray.dir);
  const d = dot(axis, w), e = dot(ray.dir, w);
  const den = a * c - b * b;
  if (Math.abs(den) < 1e-9) return null;
  return (c * d - b * e) / den;
}

// Distance from point p to segment ab (2D).
export function segmentDistance(p, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx * abx + aby * aby || 1)));
  return Math.hypot(p[0] - a[0] - abx * t, p[1] - a[1] - aby * t);
}

export const snap = (v, step) => (step > 0 ? Math.round(v / step) * step : v);
