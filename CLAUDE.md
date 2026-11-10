# Helios — notes for Claude

## Documentation layout (keep context small)

- `README.md` is the overview only (~40 KB): intro, quick start, scenes, library use, virtual geometry table,
  the GPU pipeline diagram, an index table of subsystem pages, the WebGPU constraints table.
- Subsystem detail lives in `docs/architecture/*.md` (nanite, materials, virtual-textures, shadows, motion, foliage,
  tessellation, lumen, lights, atmosphere, translucency, volumes-particles, post, editor).
- Per-round frame times: `docs/performance.md`. Differences from UE5: `docs/status.md`.
  Design notes per feature: `docs/roundN/`. `docs/README.md` indexes all of it.

Rules:
- Never Read `README.md` or a large doc whole. Grep for the heading / bullet first, then Read with `offset`/`limit`.
- A new feature's description goes into the matching `docs/architecture/*.md` page (new page + README index row if
  no page fits); the README gets at most one line (index row, quick-start flag, diagram step).
- New measurements: `docs/performance.md`; new UE5 gaps: `docs/status.md`.
- Never run `cargo fmt` (the crate is not rustfmt-formatted).
