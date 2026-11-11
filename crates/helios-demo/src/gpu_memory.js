// GPU memory accounting for the browser (`heliosGpuMemory()`): every GPUBuffer / GPUTexture the
// page creates is recorded under its label with its size until it is destroyed or garbage
// collected. wgpu's WebGPU backend never calls `destroy()` when a resource is dropped, so a
// dropped one still holds its memory until the JS collector finalizes the wrapper; the engine
// destroys what it replaces explicitly (`helios_render::gpu_memory`), and this ledger shows both.
//
// `diagnose` (`?gpu-memory=1`) also wraps each creation in an out-of-memory error scope and logs
// the allocation that failed, with the live total at that moment.

const BYTES = {
  r8unorm: 1, r8snorm: 1, r8uint: 1, r8sint: 1, stencil8: 1,
  r16uint: 2, r16sint: 2, r16float: 2, rg8unorm: 2, rg8snorm: 2, rg8uint: 2, rg8sint: 2, depth16unorm: 2,
  r32uint: 4, r32sint: 4, r32float: 4, rg16uint: 4, rg16sint: 4, rg16float: 4, rgba8unorm: 4,
  "rgba8unorm-srgb": 4, rgba8snorm: 4, rgba8uint: 4, rgba8sint: 4, bgra8unorm: 4, "bgra8unorm-srgb": 4,
  rgb9e5ufloat: 4, rgb10a2uint: 4, rgb10a2unorm: 4, rg11b10ufloat: 4, depth24plus: 4,
  "depth24plus-stencil8": 4, depth32float: 4, "depth32float-stencil8": 8,
  rg32uint: 8, rg32sint: 8, rg32float: 8, rgba16uint: 8, rgba16sint: 8, rgba16float: 8,
  rgba32uint: 16, rgba32sint: 16, rgba32float: 16,
};

// Bytes of one 4×4 block of a compressed format.
function blockBytes(format) {
  if (/^(bc1|bc4|etc2-rgb8|etc2-rgb8a1|eac-r11)/.test(format)) return 8;
  if (/^(bc|etc2|eac|astc-4x4)/.test(format)) return 16;
  return 0;
}

function extent(size) {
  if (Array.isArray(size)) return [size[0] ?? 1, size[1] ?? 1, size[2] ?? 1];
  return [size.width ?? 1, size.height ?? 1, size.depthOrArrayLayers ?? 1];
}

export function textureBytes(desc) {
  const [w, h, d] = extent(desc.size);
  const mips = desc.mipLevelCount ?? 1;
  const samples = desc.sampleCount ?? 1;
  const is3d = desc.dimension === "3d";
  const block = blockBytes(desc.format);
  const texel = BYTES[desc.format] ?? 4;
  let total = 0;
  for (let m = 0; m < mips; m++) {
    const mw = Math.max(1, w >> m), mh = Math.max(1, h >> m);
    const md = is3d ? Math.max(1, d >> m) : d;
    const plane = block ? Math.ceil(mw / 4) * Math.ceil(mh / 4) * block : mw * mh * texel;
    total += plane * md;
  }
  return total * samples;
}

export function installGpuMemoryHook(diagnose) {
  if (globalThis.__heliosGpuMem || typeof GPUDevice === "undefined") return;
  const live = new Map(); // id → [label, bytes, kind]
  const ids = new WeakMap(); // resource → id
  const failures = [];
  let next = 1, total = 0, peak = 0;
  const finalizer = new FinalizationRegistry((id) => forget(id));
  function forget(id) {
    const entry = live.get(id);
    if (entry) {
      total -= entry[1];
      live.delete(id);
    }
  }
  // Creations per label (diagnose: shows resources made again every frame).
  const made = new Map();
  function track(resource, label, bytes, kind) {
    const id = next++;
    if (diagnose) made.set(label, (made.get(label) ?? 0) + 1);
    live.set(id, [label || "(unlabeled)", bytes, kind]);
    ids.set(resource, id);
    finalizer.register(resource, id);
    total += bytes;
    peak = Math.max(peak, total);
  }
  function wrapCreate(name, kind, size) {
    const create = GPUDevice.prototype[name];
    GPUDevice.prototype[name] = function (desc) {
      const bytes = size(desc);
      if (diagnose) this.pushErrorScope("out-of-memory");
      const resource = create.call(this, desc);
      if (diagnose) {
        const live = total;
        this.popErrorScope().then((e) => {
          if (e) {
            failures.push({ label: desc.label, kind, bytes, liveBytes: live, message: e.message });
            console.error(`GPU out of memory creating ${kind} "${desc.label}" (${(bytes / 1048576).toFixed(2)} MiB, ` +
              `${(live / 1048576).toFixed(1)} MiB live): ${e.message}`);
          }
        });
      }
      track(resource, desc.label, bytes, kind);
      return resource;
    };
  }
  function wrapDestroy(proto) {
    const destroy = proto.destroy;
    proto.destroy = function () {
      const id = ids.get(this);
      if (id !== undefined) {
        forget(id);
        ids.delete(this);
      }
      return destroy.call(this);
    };
  }
  wrapCreate("createBuffer", "buffer", (desc) => Number(desc.size));
  wrapCreate("createTexture", "texture", textureBytes);
  wrapDestroy(GPUBuffer.prototype);
  wrapDestroy(GPUTexture.prototype);
  globalThis.__heliosGpuMem = {
    // Live allocations aggregated by label and kind: [[label, kind, bytes, count], …].
    entries() {
      const byLabel = new Map();
      for (const [label, bytes, kind] of live.values()) {
        const key = kind + "\u0000" + label;
        const e = byLabel.get(key) ?? [label, kind, 0, 0];
        e[2] += bytes;
        e[3] += 1;
        byLabel.set(key, e);
      }
      return [...byLabel.values()];
    },
    total: () => total,
    peak: () => peak,
    created: () => next - 1,
    made: () => [...made.entries()].sort((a, b) => b[1] - a[1]),
    failures: () => failures,
  };
}

// The ledger as JSON for the engine: {"entries": [[label, kind, bytes, count], …], "peak": n,
// "failures": n}; empty without the hook.
export function gpuMemoryEntries() {
  const m = globalThis.__heliosGpuMem;
  if (!m) return "";
  return JSON.stringify({ entries: m.entries(), peak: m.peak(), failures: m.failures().length });
}
