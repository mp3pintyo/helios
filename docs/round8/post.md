# Round 8 — post-processing: fog screen space scattering, photo mode, TSR depth history

Three items of the round's first wave, all after the lighting: a fog post step, an accumulated still mode and a
disocclusion test in TSR. Measurements on an RTX 3090 shared with other agents' builds, 1600×900, per-pass GPU minima
of headless runs (`--frames 240`, minima over the last two thirds), interleaved on / off.

## 1. Fog screen space scattering (backlog 211)

**Problem.** The froxel fog (`fog.wgsl`) applies `colour × T + S`: the light from the visible surfaces is only
extinguished, the in-scattering S is single scattering of the sun, the local lights and the ambient light at 16-pixel
froxels. What the fog scatters slightly off a view ray is lost although it reaches the camera from beside the pixel:
lamps, candles and windows do not glow through the haze, dense fog does not soften what is behind it.

**UE reference.** UE 5.8 adds *Fog Screen Space Scattering* (experimental): multiple scattering of light in fog, smoke
and dust approximated in screen space, with volumetric fog and local fog volumes, not heterogeneous volumes.

**Design** (`fog_scattering.rs`, `shaders/fog_scattering.wgsl`, after TSR, before the camera effects; fog.wgsl is
untouched, the pass reads the integrated volume and the fog uniform the lighting reads).

* Per output pixel the integrated volume at the pixel's depth gives T (optical depth τ = −ln T) and S; the light seen
  directly is D = colour − S = L·T.
* Forward scattering: of the light the fog scatters, a share F stays within the reach of the blur. Scattered k times
  and attenuated on the rest of the ray, it adds (F·albedo·τ)^k / k! · L·T, in all D·(e^(F·albedo·τ) − 1), which is at
  most L·(1 − T), what the fog extinguished — energy is spread, not created. The intensity setting scales this, but
  never beyond D·(e^(albedo·τ) − 1), all the scattered light.
* The spread: a 13-tap downsample pyramid (half to 1/256 resolution, 8 levels at 1600×900) and tent upsampling that
  adds each level with a weight: the Henyey–Greenstein phase function's light between that level's angular radius and
  the previous one's (`forward_fraction`, normalised; light scattered at θ somewhere along the ray appears at about
  θ/2 from its source). The anisotropy is 0.85 by default (fog droplets and dust are strongly forward-scattering),
  or the fog volume's when larger: the volumes' 0.6–0.7 are chosen for visible sun shafts and make an overly broad
  glow.
* Occlusion: a pixel only receives what scattered in front of its own surface, in proportion min(1, τ / τ̄) with τ̄
  the luminance-weighted mean τ of the light reaching it (carried in the pyramid's alpha), so a near pillar in fog
  does not glow with the haze behind it.
* Setting `RenderSettings::fog_scattering: Option<FogScattering { intensity: 1, anisotropy: 0.85 }>`, off by default
  (experimental in UE too); `--fog-scattering N`, hotkey **Home**, editor "Fog screen space scattering" (0–4).
  Pipelines in the Fog group (compiled with the fog in the browser).

**Alternatives.** A per-pixel lerp to a blurred image (`mix(c, blur(c), s)`) is not energy-conserving where s varies.
Blurring the in-scattered S as well double-counts the single scattering the froxels already spread. Spreading inside
fog.wgsl (scattering between froxels) would be a real multiple-scattering solve but at 16-pixel froxels the glow of a
candle flame — the point of the feature — is lost.

**Measurements.** fog scattering pass 0.110 / 0.115 ms (cathedral, two interleaved runs; 0.111 in the night forest,
0.110 with the lantern); frame minima 5.72 / 5.65 ms off vs 5.80 / 6.15 ms on (the shared GPU's noise is larger than the
pass). Targets 19 MiB at 1600×900. At the default intensity the effect is physically small in the demo scenes'
light haze (0.01–0.012 extinction per metre: τ ≈ 0.05 to the campfire, 0.3–0.5 down the nave): the far chandeliers
and candelabra glow softly and the nave loses some contrast (`round8/post/fog-scattering-cathedral.jpg`: off,
intensity 1, intensity 2, cathedral at night with the lantern, `--lights 300 --sun 200,-30 --lantern`).

**Limits.** Single-scattering angles for every order (many orders widen the lobe), the kernel's width is per screen
level and only its amount varies with τ, no aerial perspective or underwater fog, no local fog volumes (Helios has
none).

## 2. Photo mode — accumulated stills (backlog 212)

**Problem.** Stills for the showcase come from the real-time image: TSR's temporal accumulation and the post-process
depth of field (a 48-tap gather over a disc), which dims bright out-of-focus highlights and approximates occlusion.

**UE reference.** Movie Render Graph accumulates many temporal / spatial samples per output frame; Accumulation DOF
renders a lens sample per sample for film-quality depth of field.

**Design** (`photo.rs`, `shaders/photo.wgsl`, a `PHOTO` permutation of `tsr.wgsl`).

* `Renderer::start_photo(n)`: from the next frame's camera, every frame gets a Halton (2, 3) sub-pixel jitter (no
  cycle) and, with depth of field on, a lens sample: Halton (5, 7) mapped concentrically onto the aperture disc
  (radius focal length / f-number / 2, the real-time depth of field's thin lens), the camera moved there and the
  projection sheared by `scale · lens / focus` so the focus plane projects to the same pixel through the whole lens
  (unit test `lens_samples_keep_the_focus_plane_and_blur_the_rest`). The frame treats the lens camera as its camera
  (unjittered matrices included); only the sub-pixel offset is jitter.
* Focus: the setting's distance, or with autofocus the post-process depth of field's smoothed autofocus distance read
  back once (`PostFxPass::request_focus` / `take_focus`; one frame natively).
* TSR's photo permutation writes this frame's Gaussian reconstruction alone with its sample weight (a valid one-frame
  history for when the photo ends); `photo.wgsl` sums colour × weight and weight in two `rgba32float` targets and
  writes the mean, which goes through exposure, local exposure, bloom and tone mapping. Fog scattering passes the
  weight through.
* While it accumulates: post-process depth of field and motion blur off, the renderer's clock stands still (wind,
  fog drift, exposure adaptation), and the demo holds the sun, props and skeletal animation. Lumen, MegaLights and the
  fog keep converging and their remaining noise averages over the samples. The finished still stays on screen until
  the camera moves or the photo is stopped.
* Interfaces: **Enter** (128 frames; again to go back), `--photo N` with `--screenshot` (native; logs the progress),
  `heliosPhoto(n)` / `heliosPhoto(0)` and `heliosPhotoProgress()` (JSON) in the browser page API; the HUD shows
  the progress.

**Found on the way.** The real-time depth of field spread its taps over the thin-lens circle's *diameter* as a radius:
twice the physical blur. Photo mode's lens samples, which blur exactly as the lens does, showed it; it now uses the
radius (`post_fx.wgsl`), so real-time and photo agree (`round8/post/photo-vs-realtime-dof.jpg`: before, after, photo).

**Measurements.** Sum pass 0.048 ms a frame; cathedral 128 frames with autofocus 2.24 s (one frame waiting for the
focus), 256 frames at night 8.3 s; targets 58 MiB at 1600×900, made on the first photo. `round8/post/photo-full.jpg`:
cathedral at night, f/1.2 at 6 m, 40° — the chandeliers' flames stay bright bokeh discs in the photo, the gather
dims them.

**Limits.** One instant (no temporal sub-samples, so no accumulated motion blur), no per-sample warm-up of
history-based effects (they converge over the photo instead), round aperture, output at window resolution only.

## 3. TSR depth history (backlog 213)

**Problem.** TSR judged disocclusion by motion vectors only: a history pixel is rejected where the surface's motion
differs by more than 1 px (fully at 3 px) from the motion the history pixel was written with. Slower relative motion
— a sub-pixel parallax between a pillar and the wall behind it, a swinging chandelier, a swaying branch — relies on the
neighbourhood clip, which is wide on a nearly still image.

**Design** (`tsr.wgsl`, `passes/tsr.rs`). The history state texture's free channel keeps the view depth of the closest
sample each output pixel was written with (sky: 60 000). For the reprojected history pixel, the surface point it
stood for is rebuilt through its centre with the previous frame's inverse view-projection (a new `TsrUniform` field)
and must lie on the plane of this pixel's closest surface — normal from the depth neighbours, per axis the nearer one —
within 2 % of the depth: a plane distance, so surfaces at grazing angles, where the history pixel's sub-pixel offset
alone changes the depth a lot, don't fail it. The depth test acts in proportion to the motion difference from 0.05 to
0.5 px and not at all where surface and history moved alike: a twig that this frame's jittered samples miss looks
exactly like a disocclusion (the history is nearer), and on a still image or a camera pan that is all it is. A debug
render of the test (red mismatch, green motion weight) showed mismatches at every silhouette, which is why the gate
is needed.

**Measurements.** TSR 0.104 / 0.103 ms with, 0.104 / 0.112 ms without (no measurable cost; the test runs only where
motions differ by 0.05–1 px). Frame-to-frame flicker of the still forest in the wind (16 frames, pixels changing by more
than 24/255): 0.118 % with, 0.115 % without — no regression. Ghosting against a reference (photo mode at the same
instant, 64 frames: the converged image without history lag): swinging chandeliers in a 25° view, pixels off by more
than 16/255 1.332 % → 1.280 %, mean error 2.509 → 2.493 (`round8/post/tsr-depth-history-chandelier.jpg`, error × 4);
the windy forest 5.849 → 5.831 %; a slow sideways move in the nave (0.15 m/s, error against a converged still render
at the end position) unchanged at 0.688. In these scenes the motion test and the clip box already left little
ghosting; the depth history removes part of what remains on thin moving objects.

**Limits.** One closest depth per output pixel (UE scatters a dilated depth forward), the gate leaves equal-motion
disocclusions to the clip box, no thin-geometry detection as UE's.
