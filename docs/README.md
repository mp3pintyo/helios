# Helios documentation

The [top-level README](../README.md) is the overview: quick start, the demo scenes, using the library, the GPU
pipeline diagram and an index of the pages below.

| Where | What |
|---|---|
| [`architecture/`](architecture/) | One page per subsystem of the GPU pipeline (Nanite core, materials, virtual textures, shadows, motion, foliage, tessellation, Lumen, local lights and MegaLights, atmosphere, translucency and water, terrain and rivers, volumes and particles, post-processing, the browser platform) and the editor |
| [`performance.md`](performance.md) | How costs are measured and the default views' frame times, round by round |
| [`status.md`](status.md) | What is still simpler than Unreal Engine 5 |
| [`round4/`](round4/) … [`round9/`](round9/) | Design notes per feature: why it is built that way, the full measurement tables, before / after images |

Until 2026-11-11 the README held every subsystem description and the status list. Design notes written before then
say "the README carries the user-facing text (the *X* bullet …)": that bullet is now in the matching
`architecture/` page, and the numbered gaps are in `status.md` with the same numbers.
