# Round 8 — the river at the cave: sheets, blue slivers, polka-dot ground, milky water

Backlog item 237 `river-cave-artifacts` (fix). How the result works is described in
[architecture/terrain.md](../architecture/terrain.md) (Shading: Ground; Shallow-water rivers: Scene
description, Surface, Shores); this note is the causes, the alternatives and the numbers.

## Problem

From the valley's cave view (`--scene valley --eye -14,2.2,-28 --at -12.5,1.8,-40`), and in the
river close-up (`--eye -2,3,-6 --at -9,0,-20`) and the initial view, four things looked wrong
(`riverfix/cave-before-after.jpg`, `river-before-after.jpg`, `mouth-before-after.jpg`):

1. light, vertically smeared "sheets" or blocks standing at the cave's mouth where the river leaves
   it;
2. thin, saturated blue lines in the water, along the banks and on the cave's floor;
3. the ground — the banks, the meadow's dirt, the river's bed — covered in an even polka-dot of
   round, dark-ringed pebbles;
4. the shallow river reading as a milky grey sheet over the gravel instead of clear water.

## Causes

Found with temporary debug colours in `shade_water` (steep geometry, long refraction paths, no
opaque depth behind) and the G-buffer debug views (albedo, normals, indirect light, reflections).

**1. The sheets were opaque ground, dressed as submerged lake bed, in blocks of whole river cells.**
The albedo view showed them: they are the rock face of the step under the cave's lip, not water.
Two things painted them. `surface_step` gives every dry cell within three cells of water the
*highest* neighbouring water level (so a bank above the water gets the wet band); next to the step
that is the cave pool's level, so the face below the lip and the channel floor at its foot counted as
lying under up to a metre of water, and `shore_surface` replaced the rock by sand, pebbles and silt —
whose pattern is projected on the horizontal plane, so on the face it smeared into vertical streaks,
and whose extent followed the nearest-texel level lookup, so it ended in staircase outlines one cell
(0.25 m) wide. A second, smaller contributor: the river's dry corners hid at the *heightfield's*
ground (`River::hide`), meant for rock walls, but in the cave that ground lies a metre under the
floor, so the edges of the pool's sheet plunged through the floor and stood as water sheets in
front of the lip (red in the steepness debug view).

**2. The blue slivers were back faces of the river's sheet, shaded as if seen from under water.**
A dry corner was drawn 12 cm under its own bed, so the triangles at the shore tilted down into the
bank; where such a triangle rose out of the ground and the view was flatter than the triangle, its
back faced the camera, and `shade_water` sends back faces to `shade_water_below` (the camera in the
lake), whose deep-water colour is a dark saturated blue (R and G near 0). None of the river's front
faces had a long refraction path or sky behind them (checked) and no opaque view showed the lines.

**3. The polka dots were the dirt texture.** `TexturePattern::Dirt` placed one pebble size in 30 % of
the cells of a 40 × 40 Voronoi grid per 4 m tile — every pebble the same disc, evenly spread, and its
height a dome on a step, which the generated normal map turns into a dark ring. The shore's own
pebbles (`shore.wgsl pebbles`) had the same flaw at 5 and 22 cm.

**4. The milky water was the bed under it.** Splitting the screen into thirds — river without
reflections, with everything, without foam — changed little: the water itself is the lake's clear
medium (absorption 0.38 / 0.05 / 0.035 per metre). What showed through it was the lake's shore
sediment: pale sand (0.42, 0.37, 0.29) with grey pebbles, under water only darkened (×0.55) — the same
pale strip as the dry bank beside it, so the water was invisible except for its sheen and foam.

## UE reference

UE's water renders rivers and lakes with one scattering/absorption medium; the bed is the landscape's
material, blended by the water body's influence (UE 5 `WaterBodyRiver` with the landscape's
"water weight"/wetness layers), usually rocks and gravel from Megascans with wet darkening and
saturation, and algae or silt layers painted by depth. Shorelines come from the water mesh meeting
the terrain; the water's surface is not seen from below unless the camera is in the water
(`Water Info` / underwater post-process). Ground materials avoid visible repetition with several
scales, macro variation and scattered decals or meshes rather than one regular pattern.

## Fixes

* **Shore levels** (`river_sim.wgsl surface_step`): a dry cell takes only water levels that lie
  below its own ground (+5 cm). The ground under the step is no longer the pool's bed; its level is
  the channel's (or none).
* **Sediment on level ground only** (`shore.wgsl`): sand, gravel, silt and ripples fade out between a
  normal y of 0.65 and 0.35. Steep faces keep their rock (cliff rock on the mesh terrain), wet.
* **Hiding dry cells** (`valley_terrain::add_river`): just under their own bed (the cave's floor),
  the heightfield's ground only under walls.
* **Dry corners carry the water level on** (`river_sim.wgsl drawn_height`): the highest water level
  among the 3 × 3 neighbouring cells, or 12 cm under the cell's ground if that is lower. The bank
  cuts a level sheet at the true water line instead of the sheet dipping into the bank; every quad
  sharing the corner computes the same height, so there are no cracks.
* **No underwater view of the river** (`translucency.wgsl shade_water`): a back face of the
  simulated surface (a fold that still shows) is shaded as water seen from above, with the normal
  flipped; `shade_water_below` stays for the lake.
* **White water over the step**: steep river surface (vertex normal y < 0.85, fully at 0.55) is
  aerated — the foam lace and patches with at least that amount — whatever the simulated foam.
* **Natural dirt** (`texgen.wgsl dirt`, `stone_dome`): lumpy soil; gravel (60 cells per tile) and
  stones (23) in patches of a low-frequency density field; per-stone size (a few large, many small),
  squash, turn and hue; each stone sunk by its own share into the soil's mean level, so its outline
  is where it rises out of the clods; dusted with the soil and coated where it disappears. All
  scenes using the dirt pattern (valley, forest, cathedral) get it.
* **Shore pebbles** (`shore.wgsl pebbles`): sizes 0.18–0.48 cells (skewed small), burial per stone
  plus a field over the ground (gravel gathers in patches, stones sink by varying amounts), warm to
  cool hues.
* **A stream bed under the river** (where its level is the higher): a gravel matrix (0.25, 0.22,
  0.17) packed with gravel and cobbles (denser, less buried: an armoured bed), an olive-brown film
  of algae and silt that thickens over the first 0.4 m of depth and in patches, and wet stones
  saturated (+35 %) as well as darkened. The water stays the lake's medium, so the river and the
  lake match; the bed now shows through it, darker and greener where it is deeper.

## Alternatives considered

* *Bilinear river-level lookup* to soften the blocky outline: four loads per terrain pixel in the
  river's rectangle, and it would only have blurred a level that was wrong there in the first place.
* *Culling the river's back faces*: removes the slivers but leaves holes (the ground or the lake bed
  showing through the fold's other side); shading them as front faces keeps the sheet continuous.
* *Lowering the WET threshold or smoothing depths* so fewer corners are dry: the shoreline's folds
  are geometric (a level sheet meeting a bank between cell centres), not a threshold artefact.
* *A dedicated waterfall mesh at the step*: the right thing for a tall fall; this step is under a
  metre and the simulated sheet already has the shape, it only lacked the white.
* *Tinting the river's water green* to fight the milky look: the river would no longer match the
  lake; the colour belongs to the bed (algae, wet gravel), which is what real clear streams show.
* *Scattering decal stones or PCG pebble meshes* over the banks: more convincing up close but
  thousands of instances for a ground texture's job; the texture now carries the variety.

## Measurements

RTX 3090 shared with other agents' builds and renders; 1600×900, 240 frames, the GPU minima the demo
logs at the end of a run (whole frame and per pass); three interleaved runs of main 7888e3c
(`before`) and this branch (`after`) per view; the lowest of the three is given.

| View | Frame (before → after) | Material classify + resolve | Translucency | River simulation |
| --- | --- | --- | --- | --- |
| Initial view (second set of 3) | 5.71 → 5.78 ms | — | 0.658 → 0.665 ms | 0.045 → 0.044 ms |
| Cave `--eye -14,2.2,-28 --at -12.5,1.8,-40` | 5.53 → 5.43 ms | 0.999 → 0.925 ms | 0.364 → 0.421 ms (others 0.433–0.449 → 0.421–0.431) | 0.043 → 0.044 ms |
| River `--eye -2,3,-6 --at -9,0,-20` | 7.27 → 6.63 ms | 0.969 → 0.996 ms | 0.904 → 0.903 ms | 0.043 → 0.043 ms |

The GPU was heavily shared during these runs (single frames' minima ranged from 5.4 to 19 ms between
runs of the same build, and the material resolve swings by 2× with the load), so only the passes the
change touches are meaningful and they are flat within the noise: translucency within ±0.01 ms in
the initial and river views and lower in the cave view's typical runs; the material resolve (the
shore's extra stream-bed terms and the larger pebble set) within its run-to-run spread; the river's
simulation including the vertex and shore-surface steps unchanged at 0.043–0.045 ms. A first set in
the initial view showed translucency +0.04 ms; a second interleaved set with a third build (only the
dry-corner change reverted) put all three within 0.007 ms of each other, so that was load. The dirt
texture is generated once at start (two Voronoi layers instead of one: no measurable change).

Saturated blue pixels (B > 80, B > 2R, B > 1.6G) in the screenshots: cave view 166 → 0, river view
444 → 1, the step close-up 361 → 0.

Browser (Chromium headless, WebGPU, D3D11, `--detail medium`): no WGSL or validation errors, the
valley renders the cave, the step and the stream bed as natively (`riverfix/browser-chromium.jpg`).
Tests: `cargo test --workspace --release` 382 passed (all_shaders and bitwise included).

## Images (`docs/round8/riverfix/`)

* `cave-before-after.jpg` — the cave view, before (top) and after (bottom).
* `mouth-before-after.jpg` — close to the step (`--eye -11,1.4,-33 --at -11.5,0,-40`).
* `river-before-after.jpg` — the river close-up (`--eye -2,3,-6 --at -9,0,-20`).
* `default-before-after.jpg` — the initial view.
* `browser-chromium.jpg` — the cave view in headless Chromium (WebGPU).

## Remaining limits

See `docs/status.md` item 18. The step's white water is a steepness heuristic, not an aeration
model, and has no spray; the river's sheet is still a height per 0.25 m cell, so a fold can remain
where a level sheet meets ground between cell centres (now shaded from above); the ground and bed
are procedural textures (no scanned materials, no deposition from the simulated flow, the dirt tile
still repeats every 4 m under the landscape's macro variation).
