# Web viewer status

The browser and desktop share the interval renderer, single HDR color target,
GPU shell/Hi-Z, light-bake shaders and scheduling, torch physics and attached
lights, importance scoring and masks, camera/interactions, glTF decoding, animation evaluation, and material
upload. `rmesh-viewer-core` owns the common viewer lighting, torch, and asset code;
platform crates own their window/canvas and UI integration.

## Run locally

```sh
cd crates/rmesh-viewer-web
./build.sh
./serve.sh
```

Open `http://localhost:8000/` and select a `.rmesh` file. The build packages the
existing desktop torch and its license. Animation import accepts GLB or glTF
with embedded resources. Its node tree supports selection, G/R/S transforms,
numeric axis input, Enter to commit, Escape to cancel, and deleting groups with
their descendants. Shared node-edit offsets survive animation playback; group
previews include children and restore the hierarchy when cancelled. External glTF buffers/images must be embedded first.
Hold Shift while left-dragging to pan; ordinary left-drag orbits.
Select a scene light to edit position, intensity and linear RGB; spotlights also
expose their direction. Press L or use Throw torch. Enable Grab vertices (V) and drag a dense vertex;
release to commit or press Escape to cancel. The shared GPU PBD solver drives
the preview. Commit reads vertices asynchronously, refreshes depth-sort bounds,
shell and importance data, invalidates collision geometry, and rebakes lighting.
Geometry-cache rebuilds still run on the browser main thread and can pause
interaction after a large deformation. Moving lights rebake using the same fixed-snapshot
scheduler as desktop, retaining the last completed bake until a replacement
finishes.

For a local bench regression, make `web/bench.rmesh` available from your local
asset and open `smoketest.html?mesh=bench.rmesh&gpusort=1`. Do not publish a local
asset symlink as part of a deployment. Add `&benchmark=1` for the 1080p benchmark.
The smoke page checks linear HDR pixels and reports GPU validation errors.

## Browser compatibility and performance

Verified with Chromium WebGPU / Metal on an M1 Pro. The current pipeline needs
at least 8 storage buffers per shader stage. The device explicitly requests 8,
even when the adapter exposes more. Unused tile projection pipelines
are not created on devices that cannot support them. The baked HW projection binds cached radiance directly in the drawing passes;
its projection shader only handles geometry and sort keys. Baked interval
generation omits screen-space normal and zero-gradient bindings. The shell
omits SH inputs; baking and recombination bind only their own resources.
These variants preserve the lighting model while fitting eight storage buffers.
Wasm parsing is serial and clocks use `web-time`.

GPU radix sorting is enabled by default. The CPU fallback uses matching
projection culling; the old sphere-only fallback could submit rejected cells
with stale projected colors. On bench, GPU vs CPU linear RGB maximum difference
was 0.000306 with shell Hi-Z enabled. This tests sorting equivalence, not
agreement between the approximate vertex bake and the sdir reference renderer.

Measured 1920×1080 bench frames (24 frames per case, first 4 discarded, queue
completion included; M1 Pro Chromium Metal, before importance culling):

| Sort | Camera | Median | p95 |
|---|---|---:|---:|
| CPU | Stationary | 73.1 ms | 81.7 ms |
| CPU | Orbiting | 364.8 ms | 377.8 ms |
| GPU | Stationary | 78.1 ms | 86.3 ms |
| GPU | Orbiting | 70.3 ms | 79.5 ms |

With the shared desktop importance threshold of 0.01 enabled, the same browser
benchmark measured GPU stationary 52.6 ms (p95 60.8) and orbiting 54.0 ms
(p95 59.4). CPU stationary was 57.2 ms and orbiting 294.7 ms. Bench retains
3,382,659 of 4,207,572 tets; the starting-view GPU sort sees 1,936,791 after
projection and shell Hi-Z. CPU/GPU maximum RGB difference remains 0.000306.
The smoke test also verifies disabling and restoring the importance mask.

A subsequent native run on the same machine and bench at 1920×1080, using
the default interval renderer, shell and 0.01 importance threshold, reported
18 FPS after warmup. Browser queue-completion measurements correspond to
about 19 FPS stationary and 18.5 FPS orbiting. These are close, but the timing
methods differ (native displayed FPS vs browser serial queue completion), so
they do not establish a precise native/browser ratio.
The browser uses the portable radix pipeline; the native subgroup sorter is
not exposed by the current wgpu WebGPU backend on this setup.

The Rendering controls panel exposes the desktop shell opacity/erosion,
entry/exit surface, cull enable/bias, shell-only view, show-scene toggle, and
importance threshold. Collision density and collision-mesh display use the shared
torch geometry and debug rendering. Browser regression verifies both thresholds
and rendered pixels, including restoring the normal scene afterward. Shell geometry reuses cached adjacency and opacity;
importance masks reuse cached scores. Counts are cached rather than scanned
every UI frame. Threshold edits survive deformation commits and scene reloads.

Desktop and browser quality presets share their Rust definitions. High remains
32-bit GPU sorting with shell and importance culling; Low/Medium use 16-bit
linear distance keys. The browser's lean HW projection supports both precisions
within the eight-storage-buffer limit. All five presets pass the bench rendering
regression and returning to High restores the original pixels. The 16-bit sort
preserved all 1,936,791 visible cells with ordered keys, but its maximum linear
RGB difference from 32-bit was 0.166 in the tested view. It is a quality tradeoff,
not an epsilon-0.01 approximation guarantee. A follow-up 1080p run measured
32-bit GPU stationary/orbit medians of 54.4/54.3 ms, versus 39.7/38.5 ms for
16-bit (p95 44.6/43.3 ms). Each case uses 24 serial queue-completed frames,
discarding the first four; orbit cases advance the camera, so these are nearby
views rather than an exactly matched trajectory. High remains the default.

The browser exposes the legacy Quad mode. Face and quad drawing now use the
same `record_forward_after_sort` implementation as desktop, including the quad
prepass, color-only pipelines, and opaque depth attachment. The browser smoke
suite verifies finite nonempty quad output with CPU, GPU 32-bit, and GPU 16-bit
sorting and checks GPU validation scopes. Quad and face output differ on bench
(max linear RGB difference 0.105 with 32-bit sorting in the tested view); this
is a legacy diagnostic mode, not an equivalence claim. Interval stays default.

## Remaining parity work

Ray-trace mode has been removed from both viewers, including its GPU pipelines,
output buffers/textures, resize handling and viewer-only shared wrapper. The
underlying tracing library remains for tests, reference validation and geometry
utilities used by baking/collision. It is no longer a viewer parity requirement.

Broader browser/device validation and a matched desktop performance comparison
also remain. The web port is not yet full desktop feature parity.

## Diffuse lighting

Desktop and web viewers share the diffuse light bake. The separate interactive
energy renderer and Specular checkbox have been removed. The standalone native
reference-volume validation tools retain their material math and data support.

Diffuse baking uses disconnected-component shadow recovery. Missing adjacency triggers a BVH re-entry search through the
remaining light segment. Regression tests cover a detached dense occluder and
an occluder beyond the light, on an eight-storage-buffer device. Shadow batches
are 16,384 vertices to bound BVH traversal work on Metal.

Legacy re-entry now indexes exposed boundary faces only, retaining the fast
adjacency walk within connected components (including zero-density cells).
BVH queries stop at the light distance. `cargo run --release -p rmesh-viewer-core --example bake_benchmark -- ASSET.rmesh`
measures preparation and two completed bakes with GPU completion included,
independently of UI frame scheduling.

On the local M1 Pro, `bench.rmesh` (650,000 vertices, 4,207,572 cells) measured
19.85 s for the previous warmed full-face legacy bake and 4.74 s after the
boundary/adjacency optimization; preparation measured 3.58 s and 0.83 s,
respectively. These are native queue-completion timings, not browser timings
or frame-rate measurements. Both use 40 batches of 16,384 vertices.

Legacy shadow boundaries now use a binned surface-area BVH split instead of
fixed Morton midpoints. Coincident triangle hits use deterministic face-ID
tie-breaking, so tree layout does not change which face wins. On a subsequent
paired native bench run, warmed baking measured 7.14 s with the Morton tree
and 2.29 s with the new tree; all 12,622,716 baked RGB components were bit-identical
between those layouts with the same tie-breaking rule. These are separate runs
from the earlier 4.74 s measurement and remain native, not browser, timings.
