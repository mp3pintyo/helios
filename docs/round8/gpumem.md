# Round 8 — GPU memory: accounting, the Firefox "Not enough memory left", and what the scene doesn't use

Backlog `gpu-memory-budget` (238, fix). The user-facing description is in
[`docs/architecture/browser.md`](../architecture/browser.md#gpu-memory-round-8-gpu-memory-budget).

## Problem

In headed Firefox 155 (Playwright, `?compile-batch=1`) the valley run ended with "Uncaptured WebGPU error: Not
enough memory left", also on older builds. Round 8 had added many GPU resources (AVBOIT grid, VSM moment atlas, world
cache, glass light cubes, MegaLights volume, …), and nothing told how much the renderer held, per subsystem, nor
whether a scene switch gave anything back.

## UE reference

UE accounts GPU memory in the RHI (`stat RHI`, `stat GPUMem`, `rhi.DumpMemory`, `memreport -full`: render targets,
textures, buffers per pool and per resource name). Render targets come from a pool that frees targets unused for a
number of frames, and RDG's transient allocator aliases targets whose lifetimes within a frame don't overlap.
Features allocate when used (e.g. the VSM / Nanite pools, Lumen's surface cache when Lumen is on).

## Design

**Accounting from the allocator** (`gpu_memory.rs`). A ledger kept by the create helpers would miss the 229 create
sites that don't go through them, and couldn't see a resource die: textures live as long as any of their views, and
wgpu gives no drop hook. So the ledger is the allocator's own:

* *Native:* `Device::generate_allocator_report()` (D3D12 and Vulkan sub-allocators) lists every live allocation
  under its label, with the driver's padding. `Report::from_device` groups them; `--gpu-memory` logs the table (each
  category, then its labels of ≥ 1 MiB) at the end of a headless run and before each `--switch-to` scene (now a
  list, switched at evenly spaced frames).
* *Browser:* `helios-demo/src/gpu_memory.js`, installed before the device exists, wraps `GPUDevice.createBuffer` /
  `createTexture` and `GPUBuffer` / `GPUTexture.destroy`, and keeps each live resource's label and size (texel size ×
  extent × mips) until it is destroyed or collected (`FinalizationRegistry`). `heliosGpuMemory(table)` gives JSON or
  the native table; `?gpu-memory=1` wraps each creation in an out-of-memory error scope and logs a failing one.
* *Categories:* `CATEGORIES` files labels by keyword into 21 subsystems (`category("vsm moments")` →
  "shadows: virtual shadow maps"); unknown labels land in "other" (6 MiB: the headless screenshot target).

**Destroying what is replaced** (browser). wgpu's WebGPU backend never calls `destroy()` on drop: a dropped buffer or
texture holds its memory until the JS collector finalizes it, which memory pressure on the GPU doesn't trigger (the
first Firefox ledger showed Lumen's card atlases and buffers four times over after the valley's card uploads).
`retire_buffer` / `retire_view` / `retire_texture` queue a resource; `destroy_retired()` destroys the queue right after
the frame's submission (WebGPU and wgpu keep a destroyed resource alive for submitted work). Retired: the frame
targets on a resize (`FrameTargets::retire`), the page's frame target, the volume march targets, OIT targets, AVBOIT
grid, translucency scene colour and peeled layers, post-processing targets when replaced, Lumen's card capture /
albedo atlases, card data / pages / pair sums on a scene upload, the RVT atlases. Natively, destroy is what dropping
did anyway (and a destroyed resource used later would be a validation error: the native runs, the resize and the
switch chains found none).

**Allocated when used, given back when not** (`gpu_memory::Idle`, `RELEASE_AFTER_FRAMES` = 240, 4 s at 60 Hz):

| resource | allocated | released after 240 frames without |
|---|---|---|
| glass light cube maps (`glass_light.rs`, 24 MiB) | a local light reaches a pane | a light through glass |
| volume march + history (`volumes.rs`, 6 × internal res. rgba16f) | a volume is visible | a visible volume (also with no emitters at all) |
| OIT targets + AVBOIT grid (`oit.rs`, `avboit.rs`) | effects or AVBOIT surfaces drawn | anything composited |
| translucency scene colour + peeled layers | translucent draws or a water body around the camera | translucent draws |
| post fx colour / tile targets | motion blur or depth of field | both off |

Re-created resources start empty, as on their first use (the glass maps redraw every light, the volume history
restarts). Two allocations were also right-sized: the **RVT atlases** (2 × 41 MiB) exist only while the scene has a
landscape (1×1 stand-ins otherwise; the G-buffer samples them only with `rvt.enabled`), and the **post-processing**
pass keeps its second colour target only while depth of field and motion blur both run (motion blur alone writes
straight from the input to `post fx 0`).

**A lost device** (`web.rs`): `set_device_lost_callback` stores the reason; the next frame logs it with the ledger,
shows "The GPU device was lost (the browser's GPU process may have run out of memory). Reload the page to continue."
and stops rendering (before, the canvas froze silently).

## What the Firefox error is

Reproduced with the original script (`?scene=valley&compile-batch=1&compile-log=1`, headed Firefox 155): every run
logs the error once. Polling from the page (Firefox delivers console messages to Playwright late, often only when the
browser closes) showed:

* The page's own ledger was flat at 1.28–1.40 GB from the full compile on, with no allocation failing (the OOM error
  scope around every create caught nothing) and no resource made per frame (541 creations in total).
* The device was then **lost**: `GPUDevice.lost` "Out of memory" together with the uncaptured "Not enough memory left"
  (once, `In a set_bind_group … invalid` follows), 20–30 s after the frames ran at full rate (t = 71–88 s).
* Windows' `GPU Process Memory\Dedicated Usage` of Firefox's GPU process meanwhile grew from 1.3 GB to **12.8 GB**, at
  ≈ 340 MB/s once all pipelines were compiled (≈ 45 presented frames a second): ≈ 7–13 MB per presented frame.
* A 30-line page that clears an offscreen texture, copies it to the canvas and presents (or renders straight into the
  canvas, with or without `COPY_DST` on the canvas, with or without waiting for `onSubmittedWorkDone` each frame, with
  `dom.webgpu.allow-present-without-readback` either way) grows the same: 1.7 → 20.9 GB in 27 s at 60 Hz
  (≈ 12.5 MB per frame at 1600×900; ≈ 2.7 MB at 800×450 — two canvas-sized BGRA8 surfaces per frame). The
  installed release Firefox 160, in its own profile outside Playwright, grows the same (2.2 → 8.6 GB in 17 s).

So Firefox's GPU process (D3D12 on this RTX 3090 / Windows 11 machine) leaks the presented surfaces, and wgpu's
memory-budget guard in Firefox (`MemoryBudgetThresholds`: resource creation fails above 95 %, the device is lost above
99 % of the process's DXGI budget) ends the device once the leak fills the budget. Helios's ≈ 1.3 GB only moves the
moment a little; a page that presents fewer frames lasts longer, and none survives indefinitely. Not a per-buffer
limit (no single allocation exceeds `maxBufferSize` / `maxStorageBufferBindingSize`: the largest is the 192 MiB
geometry pool), not Helios's total, not a leak on scene switches (below). Recorded for a Firefox bug report; what
Helios does about it is the device-lost message.

## Measurements

Native (Vulkan, RTX 3090, headless 90 frames, `--gpu-memory`; MiB live, the allocator's report):

| scene | 1600×900 high | 1600×900 medium | 2560×1440 high | 2560×1440 medium |
|---|---|---|---|---|
| valley | 1312.8 → 1300.3 | 1307.3 → 1294.8 | 1859.0 → 1829.0 | 1853.5 → 1823.5 |
| forest | 1325.0 → 1312.5 | 1286.1 → 1273.6 | 1830.2 → 1800.2 | 1791.3 → 1761.3 |
| cathedral | 1340.3 → 1327.8 | 1334.2 → 1321.7 | 1869.0 → 1839.0 | 1862.9 → 1832.9 |
| gallery | 1096.0 → 1000.6 | 1095.4 → 1000.0 | 1502.3 → 1389.4 | 1501.7 → 1388.8 |

Within a scene, everything else is in use: the second post-processing target (12.5 / 30 MiB) and, in the gallery, the
RVT atlases (83 MiB). The largest items per scene (1600×900 high): virtual shadow maps 267 (software depth 80, physical
pages 64, moments 43, draws 20, 16 cull views 32), Nanite 233–257 (geometry pool 192), translucency and OIT 0–211
(volume march 75, OIT 50, AVBOIT 51, scene colour 17), Lumen 134–216, virtual textures 110 (RVT 83 + its page cache
27), post-processing 85, frame targets 71, tessellation 0–38.

**Scene switches** (native chain valley → gallery → forest → cathedral → valley, twice, 300 frames each; MiB before
each switch):

| | valley | gallery | forest | cathedral | valley | gallery | forest | cathedral | valley |
|---|---|---|---|---|---|---|---|---|---|
| base | 1312.8 | 1318.9 | 1407.4 | 1406.8 | 1345.0 | 1351.1 | 1432.0 | 1406.8 | 1345.0 |
| new | 1300.3 | **1017.4** | 1372.0 | 1388.0 | 1308.3 | **1025.4** | 1372.3 | 1388.0 | 1308.3 |

No leak in either (the second loop equals the first); the new build gives back what the gallery doesn't use after the
valley (volumes, OIT, AVBOIT, scene colour, RVT: −300 MiB) and the cathedral's glass light maps. What stays across
scenes: buffers that grow to the largest scene (culling work +19.5 MiB, plant bones 4 MiB).

**Browser**, Chromium 1248 headless, the same chain twice (each scene 7 s after full quality) then a resize to
1280×720 and back; MiB live in the page (the base build measured with the same ledger injected before the page's
scripts):

| | valley | gallery | forest | cathedral | valley | gallery | forest | cathedral | valley | resized | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1240 | 1242 | 1297 | 1372 | 1312 | 1316 | 1367 | 1372 | 1312 | 1314 | 1964 |
| new | 1227 | **1024** | 1376 | 1360 | 1277 | **1031** | 1311 | 1356 | 1278 | 1278 | **1561** |

No WebGPU error and no lost device in either. The base's resize peak is the replaced frame targets waiting for the
collector; the forest's first-visit figure varies with when the collector runs (1297–1376 MiB).

**Images** (native, 1600×900, base vs new, frame 90): valley 57.7 dB, cathedral 60.9 dB, gallery 63.8 dB, forest
39.1 dB — the forest's run-to-run noise (base vs base 39.2 dB; its particles and wind aren't deterministic).
**Frame time** (native, valley, 1600×900, 300 frames, interleaved base / new, GPU frame minimum): base 5.69, 5.69,
5.67 ms; new 5.72, 5.71, 5.70, 5.71 ms — +0.02–0.03 ms, at the run-to-run noise (no pass changes; per frame the CPU
adds `Idle::tick`s and an empty `destroy_retired`).

**Firefox with the new build** (headed 155, the valley, `?compile-batch=1`): the page holds 1282 MiB, steady; at
t = 59 s, 820 frames in, the device is lost as before ("Not enough memory left", lost "Out of memory") and the page
now says so (`gpumem/firefox-device-lost.jpg`). A chain valley → gallery → forest → cathedral → valley ran with the
page at 1276 / 1067 / 1313 / 1416 / 1449 MiB until the device was lost back in the valley.

## Alternatives considered

* *A `Tracked<Buffer>` handle type with a drop guard* for every resource: exact, but 175 buffer fields and the views
  that keep textures alive would change type across the renderer (and other agents' branches); the allocator's report
  needs no change at the create sites.
* *Destroy in a `Drop` of each owner struct*: unsafe in general — some resources are shared (`CullView::share_work`
  clones a work buffer into 16 VSM levels); retiring at the replacement sites keeps it explicit.
* *Releasing feature resources at once on a scene switch*: the idle counter covers switches, feature toggles and
  views where a feature leaves the screen, without each subsystem knowing about scenes.
* *Throttling frames in flight* for Firefox: the minimal page leaks the same with one frame in flight.

## Remaining limits

No memory budget the renderer adapts to, no transient aliasing within a frame, growable buffers never shrink, 240
frames before a release (a feature toggled off and on within 4 s keeps its memory), and the browser ledger estimates a
texture's size from its format (no padding) while the native report includes the driver's. Firefox's per-frame growth
on Windows remains (a browser bug; the device is lost after tens of seconds at full frame rate).
