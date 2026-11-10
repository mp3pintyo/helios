# Coloured sunlight through glass (round 7: `glass-shadows`)

Design note. The README carries the user-facing text (the *Coloured light through glass* bullet of the GPU
pipeline section, gap 16 of "Status vs. Unreal Engine 5"); this file records why it is built the way it is, and the
measurements.

## What

The cathedral's stained-glass windows let the sun in, but the light landed **white**: glass is translucent, the
panes are excluded from shadow casting (`Scene::set_cast_shadow(pane, false)`) so the virtual shadow map lets the
sun through, and nothing coloured the light on the way. Seen from the aisle, the piers and the floor showed sharp
white patches under windows full of red, blue and gold glass (`glass-shadows/aisle-off.jpg`). Only Lumen's indirect
light carried the colour: window cards emit the sky's light tinted by the glass.

Now the sun reaching a point behind a pane is filtered by the pane's colour, as with UE5's translucent shadows:

* **The map** (`glass_shadow.rs`): an orthographic view along the sunlight, fitted to the world boxes of every visible
  `Glass` instance (square texels; 256² to 2048² texels as the panes' extent needs, 2 cm texels at best — the
  cathedral's 84 m of windows get a 2048² map with 4.1 cm texels, the valley's one pane 256²). The panes are drawn
  with the translucency pass's decoded meshes (`vs_glass_shadow` / `fs_glass_shadow` in `translucency.wgsl`): each
  texel keeps the pane nearest the sun (16-bit depth test), and stores in an rgba16f target its **transmittance** —
  what `shade_glass` transmits: the base colour × texture, × `GLASS_TRANSMISSION` (0.5) for stained glass, × the
  Fresnel transmission at the sun's incidence, × the glass coverage (lead came passes nothing) — and its **depth**
  along the sunlight (0..1 over the panes). Texels without glass hold white and depth 0: being "behind" them changes
  nothing, and bilinear filtering across a pane's edge blends towards white.
* **The receivers** (`glass_shadow.wgsl`): a point whose depth lies more than 5 cm behind its texel's pane gets the
  sun × that transmittance.
  * deferred lighting (`lighting.wgsl`): `sun_light` itself is filtered, so the diffuse and specular sun, the
    Substrate slabs and foliage transmission all follow. Six Vogel taps rotated per pixel and frame (TSR resolves
    them) blur the projection by 1.2 % of the distance behind the pane (the sun's disc plus some of the hammered
    glass's scattering): ~10 cm under a pane 10 m up, between 1 and 8 texels;
  * volumetric fog (`fog.wgsl`): one tap per froxel — the light shafts under the windows take the glass's colour;
  * Lumen's mesh-card lighting (`lumen_card_light.wgsl`): one tap per texel — the coloured patches bounce coloured
    light.
* **When it is drawn**: only when its projection changes (the sun, the panes' boxes), the scene changes (geometry,
  material or instance versions) or a pane moves (`Translucency::move_instances`); with a fixed sun, once.
* **Switches**: `RenderSettings::glass_shadows` (default on; needs `shadows`), `End` in the demo, `--no-glass-shadows`,
  the editor's *Coloured light through glass* setting. Its pipeline belongs to the translucency group: queued for the
  browser's background compile as soon as the scene has glass (`Translucency::prefetch`).

## Why this way

* **A separate map, not the virtual shadow map.** The VSM stores one depth per texel: it can say *whether* light
  arrives, not *what colour*. UE5 likewise keeps translucent shadows out of its virtual shadow maps. Glass is rare
  and simple (the cathedral: 57 translucent instances, 876 triangles), so one full redraw is cheap (0.034 ms with
  the sun animated every frame) and needs no paging.
* **Nearest pane only.** Multiplying every layer (`dst × src` blending) would also tint a receiver with panes
  *behind* it along the light: a pier lit through the south clerestory would take the colour of a north-aisle
  window further down the same sun ray. One depth per texel cannot order several layers; the nearest pane is right
  for every receiver behind one pane and only wrong where light crosses two panes in a row (the aisle's free-standing
  glass screens under the west window), which then get the nearer one's colour.
* **Depth in the colour target's alpha.** The receivers sample one texture with one bilinear fetch per tap
  (lighting's compute pass already samples 16–17 textures). fp16 depth over the panes' range resolves ~3 cm at the
  far end of the cathedral's 60 m, well inside the 5 cm bias, and receivers far behind the panes compare correctly
  (their depth exceeds 1).
* **The full transmitted light, diffused part included.** `shade_glass` diffuses 70 % of what stained glass
  transmits and shows 30 % as a blurred view; a projection that kept only the 30 % would put a faint glow where
  real cathedrals show distinct coloured patches. The diffused light still arrives, blurred a little more
  (`GLASS_SHADOW_SPREAD`).

## Measurements

RTX 3090, 2560×1440 at 100 %, cathedral south aisle (`--eye 45,1.7,10.3 --at 5,4,10.8`), 3 alternating runs of 200
frames, GPU frame medians:

| | off (`--no-glass-shadows`) | on |
|---|---|---|
| GPU frame | 9.69 / 9.96 / 9.75 ms | 9.87 / 9.92 / 9.78 ms |
| lighting | 1.921 / 1.956 / 1.940 ms | 1.971 / 2.017 / 1.989 ms |
| volumetric fog | 0.238 / 0.239 / 0.234 ms | 0.241 / 0.237 / 0.239 ms |
| lumen cards | 0.080 / 0.079 / 0.078 ms | 0.083 / 0.083 / 0.078 ms |

The lighting pays ~+0.05 ms for the six taps (only inside the map and where the sun reaches), fog and card lighting
nothing measurable; the map is drawn once (with `--animate-sun`: 0.034 ms a frame). Memory: 40 MiB for the
cathedral's 2048² map (rgba16f + depth16), 0.6 MiB for the valley's 256²; scenes without glass allocate nothing
(a 1×1 white placeholder is bound).

Browser (headless Chromium, WebGPU on D3D11): the shaders compile in Tint, no console errors; `End` toggles it
(`glass-shadows/browser-on.jpg`, `browser-off.jpg`).

## Remaining limits

* One pane per texel: light through two panes in a row takes the nearer one's colour.
* The map spans every visible pane: glass spread over a large world gets coarse texels (2048² over 1 km: 49 cm).
  A camera-centred fit would need texel snapping and a redraw as the camera moves.
* Lumen's voxel surface cache (far field, instances beyond the cards) still lights with white sun; the translucent
  surfaces themselves (glass behind glass, blended meshes) are lit without the map.
* Local lights shine through glass white (only the sun has a map).
