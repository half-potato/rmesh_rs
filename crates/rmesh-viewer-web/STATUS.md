# rmesh-viewer-web — current state

## TL;DR

The web viewer **renders** in Chrome on macOS via headless Metal:

```
canvas region (322x85 at 16,332): nonzero=27370/27370 distinct=7684 max_rgb_sum=710
--- title ---
PASS
```

`./test.sh` is green. The full viewer (`web/index.html`) should work in any
browser with WebGPU — open `http://localhost:8000/`.

## Testing on macOS

```bash
cd crates/rmesh-viewer-web
./build.sh                    # cargo build wasm32 + wasm-bindgen
./serve.sh &                  # python http.server on :8000
PORT=8000 ./test.sh           # headless Chrome smoketest -> PASS/FAIL
```

Prereqs (one-time):
- `rustup target add wasm32-unknown-unknown`  (Homebrew rust on PATH lacks
  cross targets, so the scripts prepend `~/.rustup/.../bin`)
- `cargo install wasm-bindgen-cli --version 0.2.113 --locked`
- `web/test.rmesh` — a current-format file. Symlink convention:
  `ln -s ~/Downloads/room_pbr_refined.rmesh web/test.rmesh`.

`test.sh` drives Chrome through the DevTools protocol (not
`--screenshot --virtual-time-budget`) because:
- `--virtual-time-budget` advances JS time without waiting on real GPU work
  so any `setTimeout` race against `queue.onSubmittedWorkDone()` always wins.
- `<canvas>.drawImage()` from JS does *not* reliably snapshot a WebGPU
  surface in headless mode — the page can't reliably sample its own pixels.

The script: starts Chrome with `--remote-debugging-port`, polls the page
title for `PASS`/`FAIL`, then asks DevTools `Page.captureScreenshot` and
checks pixels in the canvas region of the resulting PNG (via Pillow).

## What was broken (fixed in this pass)

1. **API drift** — `rmesh-compositor` API changed but the web viewer wasn't
   updated. `PrimitivePipeline::new` now takes a material BGL; and
   `record_primitive_pass` takes an extra `Option<&MaterialRegistry>`.

2. **`std::time::Instant::now()` panics on wasm** — `rmesh-data` uses it
   throughout parse paths. Replaced with `web_time::Instant` (new
   workspace dep; drop-in API).

3. **WGSL strict-mode atomics in radix sort** — Chrome's WGSL parser is
   stricter than naga-native. Plain assignment to/from `atomic<u32>` cells
   was rejected. Fixed 1 store + 2 loads across
   `radix_sort_count.wgsl` and `radix_sort_scatter.wgsl`.

4. **`maxStorageBuffersPerShaderStage = 10` on Chrome WebGPU.** Three
   compute layouts exceeded it (14, 11, 11 buffers across all bind groups
   in the COMPUTE stage). Fix shipped:

   - `ForwardPipelines` compute layouts split into bg0/bg1 (8+6 and 7+4)
     for cleaner code organization, though the per-stage cap still counts
     across both groups so the split alone does not lower validation count.
   - `project_compute_hw.wgsl` `uniforms` switched from `var<storage, read>`
     to `var<uniform>` — uniforms count toward `maxUniformBuffersPerShaderStage`
     (default 12) instead, so the hw_compute path is now 10 storage + 1
     uniform → fits.
   - `SceneBuffers.uniforms` BufferUsages now include `UNIFORM | STORAGE`
     so the same buffer can be bound either way.
   - Web viewer now passes `hw_compute_bg = Some(...)` to
     `record_sorted_forward_pass`, dispatching `hw_compute_pipeline` instead
     of the still-too-big `compute_pipeline`.
   - `record_sorted_forward_pass`, `record_project_compute(_and_sort)`,
     `record_forward_pass`, `record_sorted_mesh_forward_pass`,
     `record_sorted_interval_forward_pass`,
     `record_sorted_compute_interval_forward_pass` now take small carrier
     structs (`ForwardComputeBindGroups`, `ForwardHwComputeBindGroups`) so
     all consumers (viewer, viewer-web, train, python, tests, benches) get
     a mechanical type change instead of fan-out edits.

## Known cosmetic noise

The `compute_pipeline` (14 storage buffers) still fails to validate on
Chrome — the error scope picks up
*"The number of storage buffers (14) in the Compute stage exceeds the
maximum per-stage limit (10)"*. **The web viewer does not use this
pipeline at runtime** (it uses `hw_compute_pipeline`), so this is purely a
log-noise issue and rendering proceeds normally. See `TODO.md` for the
clean-up option (pack buffers in `project_compute.wgsl` or gate creation
behind a non-wasm cfg).

## Known follow-ups (also in `TODO.md`)

- Strip `compute_pipeline` to ≤10 storage buffers to clear the validation
  noise.
- Apply the uniform-buffer trick to `compute_interval_gen_bg_layout` (11
  buffers) before migrating the web viewer to the active interval shading
  path. The web viewer is currently on the legacy forward path; matching
  the native viewer's compute-interval pipeline is the obvious next step
  but is out of scope for this fix.
- Mouse/keyboard interaction is wired but not visually exercised. Open
  `http://localhost:8000/` in a real browser to drive the UX.

## Test infrastructure files

- `build.sh` — cargo build wasm32 + wasm-bindgen output to `web/pkg/`
- `serve.sh` — `python3 -m http.server` on `$PORT` (default 8000)
- `test.sh` — DevTools-driven headless smoketest; exit 0 on PASS, 1 on FAIL
- `web/smoketest.html` — programmatic test page (sets `document.title=PASS|FAIL`)
- `STATUS.md` (this file)
