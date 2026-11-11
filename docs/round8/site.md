# Round 8 on the showcase site

Item `site-round8-showcase` (order 240). The showcase page (`site/index.html`, HU/EN through `data-en`)
gets a round-8 part after the editor chapter, like round 4's: an era header (`#round8`, nav link "8. kör") and
eight chapters (17–24) with live presets that drive the embedded engine, toggles whose state the page reads from
the HUD, before/after sliders, videos and measurements. The "How it was built" timeline, the debugging stories,
the one-frame pipeline and the honest-limits list are extended with round 8.

## Chapters

| # | id | Content | Presets / toggles | Media |
|---|----|---------|-------------------|-------|
| 17 | `world-cache` | Lumen World Cache (leak through the aisle wall, path-traced reference), Lumen Lite 5.8 screen reflections, light-grid sampling of the surface cache | `wc-aisle`, `wc-cells`, `wc-lit`, `lite-puddles`; `5` (three modes shown by name), `6` | `wc_off/on` (indirect view 7 at a fixed exposure), `lite_refl_off/on`, `wc_cells` |
| 18 | `megalights-58` | MegaLights froxel volume for fog and glass, shared rays, HZB contact traces, hierarchical light culling; live readout "light culling" | `ml-fog` (300 candles), `ml-1000`, `candles-off`; `3`, `=` | `ml_volume_off/on` |
| 19 | `avboit` | AVBOIT (lake panels, sparks in smoke); local lights through coloured glass | `lake-panels`, `campfire`, `glass-candles` (1008 candles); `4`, `End` | `avboit_off/on`, `glass_candles_off/on` |
| 20 | `shadows-8` | VSM refresh budget, distant levels at their own pace + prefiltered moments, shadow overview (26); volume shadows of a smoky campfire | `vsm-overview`, `P` (turning sun), `smoke-shadow` (forest rebuilt with soot ×8); `2`, volume shadows (`7`, pressed until the HUD flips) | `vsm_overview`, `smoke_shadow_off/on` |
| 21 | `photo` | Fog screen-space scattering, photo mode (accumulated lens samples; button with progress readout), TSR depth history (one sentence) | `fog-glow`, `photo-nave`, photo (128 frames), `photo-off`; `Home` | `fog_glow_off/on` (intensity 4), `photo_realtime/accum` (f/1.2, 256 samples) |
| 22 | `wind` | Grounded foliage (the owner's reports: floating grass, glued plants, no fern shadows at the fire), short-range AO; dynamic wind with bones, push capsule, gust | `grass-close`, `ferns-fire`, `campfire`, `gust-view`; `9`, `Numpad1` (gust), `L` | `grounded_off/on`, `campfire_plants_off/on` (cut from docs/round8/foliage-grounding), `wind_gust.mp4`, `ferns_photo` |
| 23 | `terrain` | Mesh terrain (cliff, ledge, cave, arch), shallow-water river; the owner's fixes: cliff rock material, cave mouth slabs / blue lines / dots | `cliff`, `arch`, `cave`, `river`, `valley`; `Numpad2` | `river_flow.mp4`, `cliff`, `arch`, `cave_river`, `river_close` (photo mode), `rock_off/on` (docs/round8/rockmat), `cave_fix_off/on` (docs/round8/riverfix) |
| 24 | `editor-8` | Editor actors / hierarchy / outliner (link to `editor.html?scene=forest`); browser: worker preprocessing, faster scene switches, entropy-coded streaming, the Firefox GPU-process leak as an honest limit | — | `editor_actor_drag`, `editor_outliner`, `editor_component` (browser screenshots from docs/round8/editor-hierarchy) |

Wide sections: timeline entry "8. kör · Az UE 5.8 és 5.9 nyomán, három hullámban" (research → three waves →
the owner's reports → closing perf pass, numbers from `docs/performance.md`) with 22 chips; unit-test count 270 →
380; six debugging stories tagged "8. kör" (campfire patches, the lake-edge line, blue lines at the cave, wind
bones' register cost while off, the short-range AO's register cost, the Firefox leak); pipeline boxes (plant bones,
refresh budget, prefiltered distant, volume shadows, world cache, short-range AO, river simulation, MegaLights
volume, AVBOIT, fog scattering, photo mode); limits rewritten from `docs/status.md` (world cache instead of
cards, coloured glass light for 8 local lights, first visit 7–10 s / Firefox ~45 s, the Firefox leak, river and
mesh terrain, AVBOIT, volume shadows, editor actors, dynamic wind).

## Engine-side changes (small)

- `app.rs` HUD: a new line `round 8 features: MegaLights volume · short range AO · volume shadows · prefiltered
  distant shadows · shadow refresh budget · Lite screen reflections · river simulation · wind bones · depth of
  field` (on/off each; labels without digits or hyphens so the page's parser reads them). The OIT method and the
  world cache mode were already in the HUD (`OIT AVBOIT`, `GI Lumen, world cache replaces cards`).
- `web.rs`: `heliosSetCampfireSmoke(scale)` — the native `--campfire-smoke`, applied at the forest's next build
  (`heliosLoadScene`), as `heliosSetCandleLights` is for the cathedral.
- Page script: `loadScene` rebuilds the forest when the soot scale changes (forest presets without `smoke` bring the
  default fire back); `ensure` presses a key up to three times until the HUD agrees (cycling keys `5`, `7`);
  `data-feature` buttons flip such a feature; world cache state shown by mode name; photo button + `#t-photo`
  progress from `heliosPhotoProgress`; GPU pass categories for `plant`, `short-range ao`, `river`, `avboit`.

## Media

`site/capture.py` gains the round-8 cameras (same as the presets), 25 stills (beauty stills in photo mode,
`--photo 64`), two 8 s videos (`wind_gust`, `river_flow`), and a `docs` mode that cuts the before/after pairs whose
"before" only a pre-fix build could render out of the comparison images in `docs/round8/` (the label strip is
cropped away) plus the editor screenshots. Media stay in `site/media-src` (gitignored): 34 webp stills
(25–377 KB, ≈ 5.0 MiB), `wind_gust.mp4` 5.5 MiB and `river_flow.mp4` 2.1 MiB with their posters — 12.5 MiB for
round 8. `site/dist`: 113 media files, 94.8 MiB (the artifact host needs more than one publish batch at ≤ 64 MB,
and its copy ships the fox as embedded glTF `media/Fox.json`, as before).

Notes for the capture: the native `--exposure EV` sets the exposure compensation, but `DemoApp::set_sun`
overwrites it (`day_exposure`), so the flag acts as "auto exposure off at EV 0" — fine for the world-cache pair,
which needs only a fixed exposure. The `--fov` of the glass-candle and lake-panel stills (34° / 30°) is not
reproduced by the page presets (the page API has no field of view), so their live views are wider.

## Verification

`./build-web.ps1`, `python site/build.py`, `python serve.py <port> site/dist` and a Playwright run in headless
Chromium (WebGPU, D3D11): every round-8 preset loads in the live engine, every round-8 toggle flips its HUD state,
the photo button accumulates and finishes, both languages render, 390 px width has no horizontal overflow.
Firefox was not driven (Playwright's headless Firefox has no WebGPU canvas); its GPU-process leak is documented in
`docs/round8/gpumem.md` and listed on the page as a limit.
