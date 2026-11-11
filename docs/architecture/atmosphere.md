# Clouds, fog and time of day

Part of the `helios-render` GPU pipeline; the pipeline diagram and the index of these pages are in the
[README](../../README.md). Costs are GPU times at 1600×900 on an RTX 3090 unless stated otherwise
([measurement method and per-round frame times](../performance.md)).
* **Volumetric clouds** (`clouds.rs`). A cumulus layer around the planet, ray-marched where the depth buffer shows
  sky: tileable Perlin–Worley shape and Worley detail volumes generated once on the GPU, a weather field for coverage,
  adaptive stepping (long steps through clear air, short ones inside clouds), sun light marched through the cloud with
  multiple-scattering octaves and a dual-lobe phase, ambient from the sky's SH. Half resolution, checkerboarded, with
  temporal reprojection by cloud distance; a 16 km cloud shadow map darkens the ground under moving clouds.
* **Volumetric fog** (`fog.rs`). Height fog with drifting noise in a 16 px × 64-slice froxel volume: each froxel
  scatters the sun (virtual shadow map visibility, cloud shadows), the clustered local lights (with MegaLights: the
  in-scattering of MegaLights' froxel volume, every light shadowed, [lights](lights.md)) and Lumen's radiance
  cache (so interiors don't glow with sky light); jittered, temporally reprojected and integrated front to back.

* **Time of day.** `Scene::set_time_of_day` keeps the sun as the light through twilight, then hands over to a
  (brightened, bluish) full moon opposite it and fades in a procedural star field (power-law magnitudes, colour
  temperatures, scintillation) behind the clouds; the demo lowers the exposure at night so it stays a night.
