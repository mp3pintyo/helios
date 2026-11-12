// Helios starter: start the engine, show progress, and let the chapters move the camera.
import { startHelios, HeliosError } from "./helios/helios-boot.js";

// Camera shots: eye x, height above ground, z, then target x, height, z (heights relative to the
// ground under each point). The gallery's courtyard is around the origin.
const SHOTS = {
  intro: { eye: [0, 2.2, 14], at: [0, 1.2, 0], sun: [200, 35] },
  detail: { eye: [3.5, 1.6, 5], at: [0, 1.0, 0], sun: [120, 18] },
  wide: { eye: [-14, 6, 18], at: [0, 1, 0], sun: [250, 8] },
};

const veil = document.getElementById("veil");
const veilText = document.getElementById("veil-text");
const STAGES = { loading: "A jelenet felépítése…", compiling: "Shaderek fordítása…", refining: "Finomítás…" };

let H = null;
function shoot(name) {
  const s = SHOTS[name];
  if (!H || !s) return;
  H.heliosLookAt(...s.eye, ...s.at);
  H.heliosSetSun(...s.sun);
}

try {
  H = await startHelios({
    config: { scene: "gallery", gpuBudgetMs: 12 }, // or { gltf: "models/thing.glb" } / { sceneFile: "my.helios.json" }
    waitFor: "first-image",
    onProgress: (p) => {
      if (p.stage === "loading" || p.stage === "compiling") {
        veilText.textContent = STAGES[p.stage] + (p.total ? ` ${p.compiled} / ${p.total}` : "");
      }
    },
  });
  shoot("intro");
  veil.classList.add("gone");
} catch (e) {
  // No WebGPU, file://, or an engine error: the page stays readable without the 3D background.
  veilText.textContent = e instanceof HeliosError && e.code === "no-webgpu"
    ? "Ez a böngésző nem támogatja a WebGPU-t, ezért a 3D háttér nem jelenik meg."
    : `A 3D jelenet nem indult el: ${e.message}`;
  setTimeout(() => veil.classList.add("gone"), 4000);
}

// The chapter in the middle of the screen chooses the shot.
const chapters = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) shoot(e.target.dataset.camera);
}, { rootMargin: "-45% 0px -45% 0px" });
document.querySelectorAll("section[data-camera]").forEach((s) => chapters.observe(s));

// Frame rate from the engine's HUD text, once a second.
setInterval(() => {
  const fps = H && /\((\d+) fps\)/.exec(H.heliosHud());
  document.getElementById("fps").textContent = fps ? `Helios · WebGPU · ${fps[1]} fps` : "";
}, 1000);
