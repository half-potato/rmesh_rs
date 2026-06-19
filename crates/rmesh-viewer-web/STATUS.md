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

4. **`maxStorageBuffersPerShaderStage = 10` on Chrome WebGPU.** Two
   compute pipelines on the web viewer's active path exceeded it. Fix
   uses the **uniform-buffer trick**: `var<uniform> uniforms` instead of
   `var<storage, read> uniforms` drops one storage binding without
   restructuring anything else.

   | Pipeline | Storage / Uniform | Status |
   |---|---|---|
   | `hw_compute_pipeline` | 10 / 1 | fits |
   | `compute_interval_gen_pipeline` | 10 / 1 | fits |
   | `compute_interval_render_*` | 5 / 1 | fits |
   | `compute_interval_indirect_convert` | 2 / 0 | fits |

   - `project_compute_hw.wgsl`, `interval_compute.wgsl` now declare
     `@group(0) @binding(0) var<uniform> uniforms: Uniforms;`.
   - `hw_compute_bind_group_layout` and `compute_interval_gen_bg_layout`
     binding 0 use `BufferBindingType::Uniform`.
   - `SceneBuffers.uniforms` BufferUsages include `UNIFORM | STORAGE` so
     the same buffer can be bound either way.

5. **Active rendering path: interval shading.** The web viewer now uses
   `record_sorted_compute_interval_forward_pass` — same as the native
   viewer's default for PBR scenes. The legacy `record_sorted_forward_pass`
   call site is gone; render bind groups (`render_bg`/`render_bg_b`) are
   replaced by `gen_bg_a` / `gen_bg_b` / `ci_render_bg` / `ci_convert_bg`.
   `mrt_enabled=false` because there's no deferred pass.

6. **Non-PBR color fallback in `interval_fragment.wgsl`.** Previously
   `out.color` always read `albedo` from `aux_data`. For non-PBR scenes
   (which is every `.rmesh` the web viewer loads today) `aux_data` is
   bound to a 4-byte dummy, so every tet wrote `vec4(0,0,0,alpha)` and
   the canvas composited to black. Fix detects the dummy via
   `arrayLength(&aux_data) > 1u` and uses the proper volume-rendering
   integral of `c_front` / `c_back` (matching `forward_fragment.wgsl`'s
   `compute_integral`) so the color-only path produces the same output
   as Regular mode.

## Known cosmetic noise

`compute_pipeline` (the 14-storage-buffer projection shader used by
training and the native viewer's fallback) still fails to validate on
Chrome. **The web viewer does not dispatch it** — interval shading uses
`hw_compute_pipeline` for the project step — so the
*"Invalid ComputePipeline 'project_compute_pipeline'"* log line is
harmless. See `TODO.md` for the clean-up option (pack buffers in
`project_compute.wgsl` or gate creation behind a non-wasm cfg).

## Known follow-ups (also in `TODO.md`)

- Strip `compute_pipeline` to ≤10 storage buffers to clear the
  validation noise.
- Mouse/keyboard interaction is wired but not visually exercised. Open
  `http://localhost:8000/` in a real browser to drive the UX.

## Test infrastructure files

- `build.sh` — cargo build wasm32 + wasm-bindgen output to `web/pkg/`
- `serve.sh` — `python3 -m http.server` on `$PORT` (default 8000)
- `test.sh` — DevTools-driven headless smoketest; exit 0 on PASS, 1 on FAIL
- `web/smoketest.html` — programmatic test page (sets `document.title=PASS|FAIL`)
- `STATUS.md` (this file)
