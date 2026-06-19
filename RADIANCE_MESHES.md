# Radiance Meshes — the sort order this codebase is built on

Reference: Mai, Hedstrom, Kopanas, Kontkanen, Kuester, Barron — *Radiance Meshes
for Volumetric Reconstruction* ([arXiv:2512.04076](https://arxiv.org/abs/2512.04076),
[project page](https://half-potato.gitlab.io/rm/)). Section 3 (Preliminaries §
"Delaunay Triangulation") and the radical-axis figure in the supplement are the
relevant parts.

This note distills the single property the rest of `rmesh_rs` relies on, and
calls out where the code currently leaves performance on the table because it
doesn't use it.

## The property

For a Delaunay tetrahedralization, the back-to-front order from a fixed camera
origin **o** is given by a per-tet scalar — the **power of the camera origin
with respect to the tet's circumsphere** — and the order depends *only on the
origin*, not on the viewing direction. From the paper (Eq. 1):

```
P(T) = ‖C(T) − o‖²  −  r(T)²
```

where `C(T)`, `r(T)` are the circumcenter and circumradius of tet `T`. Sorting
all tets ascending by `P(T)` yields a back-to-front order that is
**simultaneously valid for every ray leaving o** — so a single radix sort
serves any image plane, frustum, projection, or distortion attached to that
origin.

The paper cites Karasick et al. 1997 for the proof; Edelsbrunner 1989 and Max
1990 for the underlying acyclicity result on Delaunay meshes.

## Radical-axis intuition (from the project page)

Two circumcircles `A`, `B` have equal power along their **radical axis** (the
line through their two intersection points; extends to non-intersecting
circles). For a viewpoint `C` on the radical axis, `|A − C|` and `|B − C|` move
in opposite directions as you slide along the line, so crossing the axis flips
which power is larger — i.e. flips the depth order of the two tets.

Combining this with the Delaunay empty-sphere property (every circumball is
empty of other vertices) lifts the pairwise statement to a globally consistent
ordering: there exists a single permutation of the tets such that every ray
from `o` visits them in that order. The supplement carries the figure; the
paper's body cites Karasick 1997 for the full proof.

The non-Delaunay case is the only thing that can break this — a non-empty
circumsphere can yield a cyclic per-ray order. The data pipeline guarantees
Delaunay; `compute_circumspheres_parallel` in `rmesh-data` handles sliver /
near-degenerate tets explicitly.

## Why this matters for distorted projections

Because `P(T)` is a function of the origin alone, the sort is valid for **any
camera model that shares that origin**. The paper highlights this with a
fisheye-lens demo. The corollaries that matter for `rmesh_rs`:

| Frustum / projection shape | Same origin? | Reuse the sort? |
|---|---|---|
| Fisheye / wide-FOV lens | yes | ✓ |
| Off-axis / sheared projection | yes | ✓ |
| Cubemap (six 90° frusta) | yes — all 6 faces share the eye point | ✓ |
| Spotlight cone | yes | ✓ |
| Directional light (parallel rays) | **no** — no shared finite origin | ✗ — use `dot(direction, C(T))` instead |

## Implementation map

The codebase is structured so that **sorting is origin-only** and **culling is
per-view** — they live in different shader stages.

* **Precompute**, once at load: `crates/rmesh-data/src/lib.rs::compute_circumspheres_parallel`
  writes `circumdata[i] = (Cx, Cy, Cz, r²)` in double precision (matches
  `stable_power.slang`).
* **Sort key**, per frame: origin-only.
  * Forward path: `crates/rmesh-render/src/wgsl/project_compute_hw.wgsl`.
  * DSM:         `crates/rmesh-dsm/src/wgsl/dsm_project_compute.wgsl`.

  Both compute the same stable form

  ```wgsl
  let depth_raw = dot(cam - v0, cam + v0 - 2.0 * center);
  sort_keys[tet_id] = ~bitcast<u32>(clamp(depth_raw, -1e20, 1e20));
  ```

  algebraically `|cam − C|² − r²` from Eq. 1, but cancellation-stable when
  `|cam − C|² ≈ r²`. Thread 0 publishes `indirect_args.instance_count =
  tet_count` — every tet stays in the sorted list.
* **Sort**: `crates/rmesh-sort` (radix). Inverting the bit-cast (`~`) flips it
  to ascending order under an unsigned radix.
* **Per-view cull**, at rasterization time:
  `crates/rmesh-render/src/wgsl/interval_generate.wgsl` runs an AABB +
  sub-pixel test after the perspective divide and emits degenerate vertices
  for off-frustum tets — the hardware rasterizer drops them. Shared between
  forward and DSM, which is what lets a single origin-sort serve any number
  of frusta sharing that origin (cubemap faces, fisheye, wide-FOV).
* **Rasterize**: `crates/rmesh-render/src/wgsl/interval_*.wgsl` consumes the
  sorted permutation; premul-alpha blending of intervals = the closed-form
  segment integral in Eq. 8 / Eq. 9 of the paper.
* **Backward**: `crates/rmesh-trainable/` walks the same permutation in reverse.

The legacy tiled compute path (`project_compute.wgsl`) still carries its own
visibility logic because it also writes `tiles_touched` for prefix-scan tile
binning — that's a different downstream contract than the interval path.

## How the DSM crate uses this

`crates/rmesh-dsm/src/lib.rs::generate_dsm_for_lights` runs one project +
radix sort + indirect convert per light, using `cam_pos = light.position`,
and shares the resulting permutation across all six cubemap faces. Each face
then only needs to write its own VP into `scratch_uniforms`, run `interval_gen`
with that VP, and draw.

The project step uses a minimal dedicated shader,
`crates/rmesh-dsm/src/wgsl/dsm_project_compute.wgsl`, which:

* computes the stable circumsphere-power key `(cam − v0)·(cam + v0 − 2C)` and
  writes it to `sort_keys[i]`,
* skips frustum culling and SH evaluation (DSM stores depth moments, not
  view-dependent color), and
* publishes `indirect_args.instance_count = tet_count` from thread 0 — every
  tet stays in the sorted list.

Per-view visibility is then handled at rasterization time by an AABB +
sub-pixel cull at the top of `crates/rmesh-render/src/wgsl/interval_generate.wgsl`:
once each tet's vertices are projected through the per-face VP, off-frustum
tets emit degenerate vertices and the hardware rasterizer drops them. This is
the same predicate the old `project_compute_hw.wgsl` applied before sorting;
moving it downstream of the sort is what makes one origin-sort serve any
number of frusta sharing that origin (cubemap faces, fisheye, wide-FOV).

For directional lights the key would need to change to `dot(direction, C(T))`
(no shared finite origin), but those don't go through the cubemap path anyway.

## History

* `2358f4b "tried moment based deep shadow maps, failed"` — went the other
  direction: avoided sorting entirely by using order-independent Fourier
  moments with additive blending (`sort_values = identity`). Failed because
  Fourier reconstruction rings and costs 9 channels per texel; reverted to a
  sorted 2-moment power MSM (`MSM.md`).
* `6ad95bb "depth sort dsm thing"` — correctly identified that the 2-moment
  power MSM needs a back-to-front order, but over-corrected to per-face
  sorts. The Karasick property allows one sort per light; the cubemap-face
  difference is just a different VP, not a different origin.

The current code is the corrected version: one sort per light + per-view
AABB cull inside `interval_generate.wgsl`.

## References (paper, §2 / §3)

* Karasick, Lieber, Nackman, Rajan — *Visualization of three-dimensional
  Delaunay meshes*, 1997. The original power-sort proof.
* Edelsbrunner — *An acyclicity theorem for cell complexes in d dimensions*,
  1989.
* Max, Hanrahan, Crawfis — *Area and volume coherence for efficient
  visualization of 3D scalar functions*, 1990.
* Peters & Klein — *Moment Shadow Maps*, I3D 2015. The Cantelli bound the
  current DSM uses once the back-to-front order is in hand (see `MSM.md`).

Sources:
- [Radiance Meshes for Volumetric Reconstruction (arXiv:2512.04076)](https://arxiv.org/abs/2512.04076)
- [Project page](https://half-potato.gitlab.io/rm/)
