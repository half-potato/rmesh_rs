# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**rmeshvk** — A wgpu-based differentiable tetrahedral volume renderer for Radiance Meshes. This is the Rust/WebGPU port of the rendering and training backend, designed to replace the Slang/CUDA pipeline from the parent `delaunay_rasterization` project.

It renders Delaunay tetrahedral meshes with spherical harmonics (SH) color, supports forward and backward (gradient) passes entirely on GPU via WGSL compute shaders, and exposes a Python module for PyTorch integration.

## Build & Test

```bash
# Build entire workspace
cargo build

# Run all tests (requires GPU — tests gracefully skip if no adapter found)
cargo test

# Run a specific test
cargo test -p rmesh-render test_center_view

# Run tests for a specific crate
cargo test -p rmesh-render
cargo test -p rmesh-trainable

# Build the Python extension (maturin + PyO3)
cd crates/rmesh-python && maturin develop
```

Toolchain: Rust stable (see `rust-toolchain.toml`). No special features or nightly required.

## Workspace Crates (Dependency Order)

```
rmesh-util          ← Shared CPU/GPU types (Uniforms, etc.), WGSL utility shaders (common, intersect, math, SH)
rmesh-sort          ← GPU radix sort pipelines (basic & DRS modes)
rmesh-tile          ← Tile infrastructure (fill, ranges, scan, RTS prefix scan)
    ↓
rmesh-data          ← .rmesh file loading, PCA decompression, circumsphere computation
    ↓
rmesh-render        ← Interactive rendering: interval shader (main) + legacy mesh-style/HW raster paths
    ↓
rmesh-trainable     ← Trainable pipeline: tile shader (compute-based tiled forward + backward + gradient accumulation)
rmesh-error         ← Per-tet error statistics accumulation
    ↓
rmesh-train         ← Training loop (forward + loss + backward + Adam, all GPU)
rmesh-dsm           ← Deep shadow maps (power moments, per-light cubemap atlas)
rmesh-compositor    ← Opaque primitive rendering (cube, sphere, plane, cylinder) + depth compositing
rmesh-pbd           ← GPU XPBD distance-constraint solver for interactive vertex grabs
rmesh-anim          ← Animation clock, keyframe evaluation, glTF scene loader
rmesh-interact      ← Input handling & transform interactions (translate, rotate, scale, vertex select)
rmesh-viewer-core   ← Shared desktop/web light baking, importance culling, vertex picking/PBD preparation, torch simulation, glTF GPU upload
rmesh-viewer        ← Interactive winit/wgpu viewer (orbit camera, egui UI, loads .rmesh files)
rmesh-viewer-web    ← WebAssembly viewer
rmesh-python        ← PyO3/maturin bindings exposing RMeshRenderer to Python/PyTorch
```

## Architecture

The codebase has **three rendering pathways**, each tuned for a different use case:

| Pathway | Crate | Purpose | Status |
|---------|-------|---------|--------|
| **Interval shader** | `rmesh-render` | Interactive viewer / Python forward inference | **Main** — production renderer |
| **Mesh-style shader** | `rmesh-render` | Original vertex/fragment & mesh shader paths | **Old** — kept for reference, viewer toggles |
| **Tile shader** | `rmesh-trainable` | Differentiable forward + backward for training | **Trainable** — used by `rmesh-train` and `rmesh-python` |

### Interval Shader (main pathway)

The interval shading path is the active rendering approach for the viewer and forward-only inference. It decomposes each tet into non-overlapping screen-space triangles with interpolated front/back NDC depths so the hardware rasterizer handles overdraw without sorting fragments.

1. **Project compute** — Projects tet vertices, evaluates SH color, generates sort keys (`project_compute.wgsl`, or the lean `project_compute_hw.wgsl` when tile counts are not needed)
2. **Radix sort** — GPU radix sort of tets by depth (`rmesh-sort`, 5-pass: count → reduce → scan → scan_add → scatter). Falls back to CPU `radsort` + sphere-frustum cull on backends without subgroups (Chrome WebGPU).
3. **Interval compute** — Generates screen-space triangles per tet and writes vertex + per-tet data to fixed-slot buffers (`interval_compute.wgsl`)
4. **Interval vertex/fragment** — The desktop viewer uses `main_color_only` to integrate cached baked radiance into one HDR RGBA target (`interval_vertex.wgsl`, `interval_fragment.wgsl`). PBR albedo remains separate in scene data; it is not substituted for radiance on this path.
5. **Present** — Blit integrated radiance to the display. The desktop viewer no longer runs deferred PBR, DSM lighting, GTAO, SSGI, SSR, or their temporal/blur passes. Hardware depth remains for opaque occlusion. The density shell, its Hi-Z, and importance culling are enabled in the normal quality presets. On macOS, shell color vertices are prepared in compute and rasterized from a conventional vertex buffer to avoid intermittent black frames in Metal’s procedural vertex path; Linux retains the direct vertex path. The compute preparation uses the same shell geometry, facing tests, colors, and depth pipeline, with no CPU readback. Legacy MRT APIs remain available to other consumers/tests; the viewer allocates their unused textures as 1×1 placeholders. For temporal regression captures, set `RMESH_SHOT_COUNT=N` with `RMESH_LINEAR_SHOT=/tmp/frame`; exports are `/tmp/frame.0.npy`, etc., from consecutive frames after bake completion.

The desktop and web viewers share `rmesh-viewer-core` and retain albedo separately and automatically bakes scene point/spot lights with zero ambient. Its `light_bake.rs` and `wgsl/light_bake.wgsl` trace volumetric transmittance through tet adjacency at shared vertices, apply the material's SGGX projected area, and combine the cached irradiance with per-tet albedo into radiance. Light changes start an incremental bake in batches of 16,384 vertices ordered spatially (Morton order) for coherent shadow traversal. Each in-flight bake finishes a fixed light snapshot, then picks up the latest state so moving torches cannot starve completion. Single-light color/intensity edits reuse completed cached transport and only recombine radiance. Flare models carry an explicit point light through movement, fade, and removal. The diffuse bake uses shared BVH recovery to cross culled gaps and re-enter disconnected components; diffuse baking indexes exposed boundary faces; missing adjacency is not treated as escape to the light. Diffuse baking packs BVH nodes and faces into one buffer to stay within eight storage bindings. Unchanged lights require no baking work. The previous completed bake remains visible until its replacement is ready. Legacy assets without albedo use an approximate SH DC fallback. Learned vertex normals remain bake inputs; there is no screen-space normal attachment. This diffuse bake does not reproduce view-dependent specular terms.

The desktop and web viewers use the shared diffuse light bake. The separate interactive energy renderer and its specular toggle have been removed. Energy material metadata remains available to data loading and the standalone reference-volume validation tools.

Accuracy validation is separate from that approximate viewer bake. `scripts/sdir_checkpoint_reference.py` executes sdir's actual CPU Warp volume quadrature and M3 shading with CPU-evaluated checkpoint materials, and can export a full-precision `RMLIT001` scene. `rmesh-viewer --reference-volume` runs the native float32 per-ray equivalent (including M3 and face-BVH shadow recovery), without opening a window. `scripts/validate_reference_volume.py` renders the exported rays and gates maximum absolute linear RGB error against the sdir image, with default epsilon 0.01. This command-line path is not yet an interactive bake replacement. A passing comparison covers only its matched camera, lights, quadrature, and CPU-evaluated inputs; it does not certify the vertex-average bake or equivalence to original CUDA/TF32 checkpoint evaluation. The optimized `.rmesh` density quantization and missing M3 channels must not be silently treated as full-precision reference data.

The two pipeline structs that drive this path:
- `ComputeIntervalPipelines` — compute → vertex/fragment draw. **Universal** — works on every backend (Metal/Vulkan/DX12/WebGPU).
- `IntervalPipelines` — single mesh-shader dispatch replacing steps 3–4 (`interval_mesh.wgsl` + `interval_fragment.wgsl`). Requires `Features::EXPERIMENTAL_MESH_SHADER`; currently untested due to lack of HW access.

#### Deep Shadow Maps (rmesh-dsm)

Shadows use power moments stored in a per-light cubemap atlas:
- `DsmPipeline` renders interval geometry from each light's perspective
- Fragment shader accumulates power moments (m_0..m_4 = α·z^k) into 3 Rgba16Float MRT targets
- `DsmResolvePipeline` reconstructs transmittance T(z) via Hamburger 2-atom moment reconstruction
- The deferred shading pass samples the DSM atlas for shadow evaluation

#### Compositing (rmesh-compositor)

Opaque primitives (cubes, spheres, planes, cylinders) are rendered via `PrimitivePipeline` and depth-composited with translucent tet volumes via the compositor.

### Mesh-Style Shader (old pathway)

The original vertex/fragment and mesh-shader forward path predates interval shading. It still compiles and the viewer exposes it via `RenderMode::Regular`, `RenderMode::Quad`, and `RenderMode::MeshShader`, but is no longer the default and is not used by training or the Python bindings:

- **Hardware rasterization** (`forward_vertex.wgsl`, `forward_fragment.wgsl`, `ForwardPipelines`) — the original 12-tri-per-tet draw with sorted indices.
- **Quad billboard** (`forward_vertex_quad.wgsl`, `forward_fragment_quad.wgsl`, `forward_prepass_compute.wgsl`) — 4 verts/tet via triangle strip, reads a precomputed per-tet buffer.
- **Mesh-shader forward** (`forward_mesh.wgsl`, `MeshForwardPipelines`) — non-interval mesh-shader path.
- **Ray-tracing compute** (`raytrace_compute.wgsl`, `RayTracePipeline`) — software ray-tet intersection retained for tests and reference validation; removed from both interactive viewers.

Also retained but not wired into the active viewer modes: `project_compute_16bit.wgsl` (half-precision projection) and `shadow_ray_gen.wgsl` (pre-DSM shadow approach).

### Tile Shader (trainable pathway)

`rmesh-trainable` implements the differentiable forward + backward pass used by `rmesh-train` and the Python bindings. Both halves are pure compute — no fixed-function rasterizer — so the forward intermediates are exactly the ones the backward kernel needs.

**Forward (compute-based tiled rasterizer):**
1. **Interval generate** — One thread per tet writes the same interval vertex/per-tet data the interactive path produces (`interval_generate.wgsl`, `IntervalGeneratePipeline`).
2. **Interval tiled rasterize** — One workgroup per tile composites the visible intervals front-to-back into the output image (`interval_tiled_rasterize.wgsl`, `IntervalTiledRasterizePipeline`). Uses subgroup ops for warp-level shuffles.

The older `rasterize_compute.wgsl` (`RasterizeComputePipeline`) — a pre-interval software rasterizer — is kept alongside as a baseline.

**Backward:**
1. **Loss compute** — L1/L2/SSIM loss + per-pixel dL/d(image) (`rmesh-train`)
2. **Backward interval tiled** — Reverse-order traversal of the same intervals computing dL/d(params) (`backward_interval_tiled.wgsl`, `backward_tiled_compute.wgsl`, with `interval_chain_back.wgsl` for the SH/color chain)
3. **Error accumulation** — Per-tet error statistics (`rmesh-error`)
4. **Adam compute** — Per-parameter-group Adam optimizer update (`rmesh-train`)

### Interaction & Physics

- **rmesh-interact** — Input handling, transform manipulators (translate/rotate/scale) and vertex-select picking for the viewer.
- **rmesh-pbd** — GPU XPBD distance-constraint solver for localized vertex grabs in both desktop and web viewers. Shared picking, drag math, island preparation and asynchronous readback live in `rmesh-viewer-core::vertices`; browser controls use V or Grab vertices, release to commit, and Escape to cancel. CPU BFS island construction + greedy edge coloring feed a per-step compute pipeline (`apply_handles`, `predict`, `solve_distance` × iters × colors, `finalize`). Ported from DelTetRenderer's `PBDMove.hpp`.
- **rmesh-anim** — Animation clock, keyframe interpolation (step/linear/cubic spline), scene hierarchy, glTF loader.

### Key Design Patterns

- **Shared types** (`rmesh-util/src/shared.rs`): `#[repr(C)]` structs with `bytemuck::Pod` matching WGSL `struct` layouts byte-for-byte (std430). All CPU↔GPU data flows through these.
- **WGSL as `include_str!`**: Shaders are embedded at compile time in each crate's `lib.rs`. No runtime shader compilation — change a `.wgsl` file and `cargo build` picks it up.
- **WebGPU 8-buffer limit**: Bind groups split across 2 groups to stay within limits.
- **Subgroup operations**: Tiled/interval shaders use `enable subgroups;` for warp-level shuffles.
- **GPU-only training loop**: `rmesh-train` does forward+loss+backward+Adam without CPU readback in the loop.
- **Indirect dispatch**: Compute shaders write `DrawIndirectCommand` buffers for dynamic dispatch counts.

### Python Integration

`rmesh-python` builds a `_native.cpython-*.so` via maturin. The Python package `rmesh_wgpu` provides:
- `RMeshRenderer` — Rust class with `forward()`, `backward()`, `train_step()`, `update_params()`, `get_params()`
- `autograd.py` — `torch.autograd.Function` wrapper (`RMeshForward`) and `nn.Module` wrapper (`RMeshModule`)

### Tests

Tests live in `crates/rmesh-render/tests/` with a shared `common/mod.rs` module providing:
- CPU reference renderer for comparison
- Random scene generators (`random_single_tet_scene`)
- GPU render helper (gracefully returns `None` if no adapter)
- Image comparison utilities

Interval-shader (forward) tests: `single_tet_test.rs`, `multi_tet_test.rs`, `cross_renderer_test.rs`, `mrt_test.rs`, `overdraw_stats.rs`.

Tile-shader (trainable) tests: `gradient_test.rs` (finite-difference gradient checks), `kernel_tests.rs`, and `crates/rmesh-trainable/tests/kernel_tests.rs`.

### Symbolic Backward Derivation

`scripts/derive_backward.py` uses SymPy to derive and validate the backward pass math (volume rendering integral, alpha blending, ray-plane intersection, SH/color chain). Run with `python scripts/derive_backward.py`.

## Data Format

`.rmesh` files are gzip-compressed binary:
- Header: `[vertex_count, tet_count, sh_degree, k_components]` as u32
- Start pose: 8× f32 (position + quaternion + pad)
- Vertices: `[N × 3]` f32, Indices: `[M × 4]` u32
- Densities: `[M]` u8 log-encoded → `exp((val-100)/20)`
- SH: PCA-compressed (mean + basis + weights as f16) → decompressed to direct coefficients on load
- Color gradients: `[M × 3]` f16
