// What the browser tells about the GPU, for the quality level (quality.rs): the WebGPU adapter's
// info (vendor and architecture, e.g. "intel" / "xe-lpg") and the WebGL renderer string, which
// names the model in Chromium ("ANGLE (AMD, AMD Radeon(TM) Graphics (0x164E) Direct3D11 …)").
// Firefox reports a generic model ("…, or similar") and Safari "Apple GPU"; nothing here fails.

export async function gpuInfo() {
  const out = { vendor: "", architecture: "", description: "", renderer: "", fallback: false };
  try {
    const adapter = navigator.gpu && (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }));
    const info = adapter && (adapter.info || (adapter.requestAdapterInfo && (await adapter.requestAdapterInfo())));
    if (info) {
      out.vendor = info.vendor || "";
      out.architecture = info.architecture || "";
      out.description = info.description || "";
      out.fallback = !!(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
    }
  } catch (e) {
    // No info: the level comes from the frame times alone.
  }
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (gl) {
      const debug = gl.getExtension("WEBGL_debug_renderer_info");
      out.renderer = String(gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "");
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    }
  } catch (e) {
    // WebGL off: the WebGPU info alone.
  }
  return JSON.stringify(out);
}
