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
  scatters the sun (virtual shadow map visibility, cloud shadows), the clustered local lights and Lumen's radiance
  cache (so interiors don't glow with sky light); jittered, temporally reprojected and integrated front to back.
* **Fog screen space scattering** (`fog_scattering.rs`, `fog_scattering.wgsl`; UE 5.8's experimental Fog Screen Space
  Scattering). The froxel fog only *removes* the light coming from visible surfaces (`colour × transmittance +
  in-scattering`); the part it scatters just slightly off the view ray still reaches the camera from beside the pixel —
  the glow of lamps, candles and lit windows in fog and the soft loss of contrast through dense haze. A post step after
  TSR, at output resolution: per pixel the integrated fog volume at its depth gives the optical depth τ and the
  in-scattering S, so the light seen directly is D = colour − S = L·T. Of what the fog scatters, the share F within
  the blur's reach (from a Henyey–Greenstein lobe, anisotropy 0.85 as for real droplets and dust — or the fog
  volume's, if larger), scattered any number of times and attenuated on the rest of the way, arrives as
  D·(e^(F·albedo·τ) − 1) ≤ L·(1 − T): spread, not created. It is blurred by a 13-tap pyramid from half to 1/256
  resolution whose levels are weighted by the phase function's light between their angular radii (an apparent angle
  half the scattering angle), so the kernel follows the forward lobe, and added back; a pixel only receives in
  proportion min(1, τ / τ of the light's sources), so a near post in fog does not glow with the haze behind it.
  `RenderSettings::fog_scattering: Option<FogScattering { intensity, anisotropy }>` (off by default, as in UE;
  `--fog-scattering 1`, Home, editor "Fog screen space scattering"); intensity scales the share, never beyond all
  the scattered light. 0.11 ms at 1600×900 (RTX 3090), 19 MiB of targets. Volumetric fog only (Helios has no local fog
  volumes; heterogeneous volumes are not included, as in UE).

* **Time of day.** `Scene::set_time_of_day` keeps the sun as the light through twilight, then hands over to a
  (brightened, bluish) full moon opposite it and fades in a procedural star field (power-law magnitudes, colour
  temperatures, scintillation) behind the clouds; the demo lowers the exposure at night so it stays a night.
