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
