# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
cargo test -p rmesh-backward

# Build the Python extension (maturin + PyO3)
cd crates/rmesh-python && maturin develop
```

Toolchain: Rust stable (see `rust-toolchain.toml`). No special features or nightly required.

## Workspace Crates (Dependency Order)

```
rmesh-shaders       ← WGSL shader source strings + shared CPU/GPU types (Uniforms, etc.)
    ↓
rmesh-data          ← .rmesh file loading, PCA decompression, circumsphere computation
    ↓
rmesh-render        ← Forward pipeline: compute → sort → rasterize (wgpu orchestration)
    ↓
rmesh-backward      ← Backward pipeline: loss → backward compute → Adam optimizer
    ↓
rmesh-train         ← Training loop (forward + loss + backward + Adam, all GPU)
rmesh-viewer        ← Interactive winit/wgpu viewer (orbit camera, loads .rmesh files)
rmesh-python        ← PyO3/maturin bindings exposing RMeshRenderer to Python/PyTorch
```

## Architecture

### Rendering Pipeline (Tiled)

The tiled path is the active development focus (see git status — `forward_tiled_compute.wgsl`, `backward_tiled_compute.wgsl`, `tile_ranges_compute.wgsl`):

1. **Forward compute** — Projects tet vertices, evaluates SH color, generates sort keys
2. **Tile generation** — Assigns tets to screen tiles (4×4 pixel blocks), generates (tile_id, depth) pairs
3. **Radix sort** — GPU radix sort of tile-tet pairs (5-pass: count → reduce → scan → scan_add → scatter)
4. **Tile ranges** — Finds start/end indices per tile in the sorted array
5. **Forward tiled compute** — Warp-per-tile (32 threads/tile): threads 0-15 own pixels, loads 2 tets/iteration, ray-tet intersection + alpha compositing
6. **Loss compute** — L1/L2/SSIM loss + per-pixel dL/d(image)
7. **Backward tiled compute** — Reverse-order traversal computing dL/d(params)
8. **Adam compute** — Per-parameter-group Adam optimizer update

### Legacy (Non-Tiled) Pipeline

Also present: `forward_compute.wgsl` → bitonic sort → `forward_vertex.wgsl`/`forward_fragment.wgsl` (hardware rasterization path). The tiled compute path replaces this.

### Key Design Patterns

- **Shared types** (`rmesh-shaders/src/shared.rs`): `#[repr(C)]` structs with `bytemuck::Pod` matching WGSL `struct` layouts byte-for-byte (std430). All CPU↔GPU data flows through these.
- **WGSL as `include_str!`**: Shaders are embedded at compile time via `rmesh-shaders/src/lib.rs`. No runtime shader compilation — change a `.wgsl` file and `cargo build` picks it up.
- **WebGPU 8-buffer limit**: Backward pass splits across 2 bind groups to stay within limits.
- **Subgroup operations**: Tiled shaders use `enable subgroups;` for warp-level shuffles.
- **GPU-only training loop**: `rmesh-train` does forward+loss+backward+Adam without CPU readback in the loop.

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

Test files: `single_tet_test.rs` (forward rendering), `multi_tet_test.rs` (multiple tets), `gradient_test.rs` (finite-difference gradient checks).

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
