# TODO

## Vertex Normal Interpolation
Precompute normals on vertices and interpolate them across the surfaces of each tet, rather than using flat face normals. This should produce smoother normal maps for supervision.

## Unit Tests for AUX Outputs
Add unit tests validating:
- Normals (entry face outward normal, T*alpha weighted)
- Depth (transmittance-weighted depth integral)
- Entropy (sum_w, sum_wlogw, sum_wc, sum_wc_logwc)
- Variable aux pass-through
- Verify compute (raytrace + tiled) and HW raster normals/depth agree

## Raytracer Backward Pass
Implement the backward pass for the raytrace pipeline (`raytrace_compute.wgsl`), analogous to `backward_tiled_compute.wgsl`. This enables gradient-based optimization using the raytracing path.

## Web viewer: reduce compute_pipeline storage-buffer count
`compute_bind_group_layout` (`crates/rmesh-render/src/lib.rs` `ForwardPipelines::new`) uses 14 storage buffers across both bg0+bg1; Chrome WebGPU caps `maxStorageBuffersPerShaderStage` at 10. The web viewer doesn't dispatch this pipeline (it uses `hw_compute_pipeline` for the project step and `record_sorted_compute_interval_forward_pass` for everything else), but `ForwardPipelines::new` still creates the invalid pipeline at init, producing one harmless `Invalid ComputePipeline "project_compute_pipeline"` validation error in Chrome's GPU log.

To silence: either strip `compute_pipeline` to ≤10 storage buffers (pack `densities + circumdata`, `base_colors + color_grads`, `tiles_touched + compact_tet_ids` into combined buffers, and switch `uniforms` to `var<uniform>` — touches many shaders + CPU buffer layouts) or gate `compute_pipeline` creation behind a non-wasm cfg so the web viewer never tries to make it.

