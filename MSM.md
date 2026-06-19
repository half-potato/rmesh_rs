# 2-Moment Shadow Maps (MSM)

α-weighted depth moments for cached per-light volumetric transmittance.
Renders the scene once per light face using premultiplied-alpha back-to-front
blending, stores three scalars per texel — `(E[α·z], E[α·z²], α)` — and
queries with a Cantelli (one-sided Chebyshev) bound to attenuate light in the
deferred shader.

## Problem

We need to know how much light reaches a point at depth `z` along a ray from
a light source. The scene contains volumetric tetrahedra with varying density.
Each tet absorbs some light; total transmittance along the ray is the product
of individual transmittances.

A standard shadow map stores a single depth — useless for volumes. A
trigonometric / Fourier representation works but spends 9 channels per pixel
(see git history of TMSM.md). This compromise stores only the first two
moments of the α-weighted termination depth distribution plus the total α,
and accepts a conservative bound at query time.

## Representation

Per cubemap texel the render pass accumulates, via premul-alpha back-to-front
blending:

```
α       = 1 − T_total                    (.a channel)
E[α·z]  = Σᵢ wᵢ·αᵢ·z̄ᵢ                    (.r channel)
E[α·z²] = Σᵢ wᵢ·αᵢ·z̄²ᵢ                   (.g channel)
.b      = 0
```

where each interval `i` contributes its α-weighted view-volume moments
weighted by the standard forward-pass termination distribution
(`w0·zback + w1·zfront`, `w0 = φ(od) − exp(−od)`, `w1 = 1 − φ(od)`,
`φ(x) = (1 − e^−x)/x`).

Stored in a single Rgba16Float MRT. With 6 cubemap faces per point light
this is one cubemap (6 layers, 4 channels × 16 bits) per light.

## Generation (per light face, single pass)

Render all visible tet intervals from the light's viewpoint. **Premul-alpha
back-to-front blending**, depth pre-test against opaque-primitive depth.

The back-to-front order comes from the circumsphere power sort with
`P = light.position`; for point and spot lights one sort suffices across all
six cubemap faces because they share an origin. See `RADIANCE_MESHES.md`.

```
fragment.out = vec4(depth_premul, depth_sq_premul, 0, α)

color.out = src.rgba ⊕ dst.rgba · (1 − src.a)
```

Primitives (opaque blockers) write `vec4(z, z², 0, 1)` at their NDC depth
with depth-write enabled in a pre-pass; tets blend over those with depth-read.

## Query (deferred shading)

For a receiver world point projected into the light's space at distance `z`:

1. Sample the moments cubemap in the receiver direction.
2. Compute `μ = .r / .a`, `σ² = (.g − .r²/.a) / .a`.
3. Below the mean ⇒ fully lit:
   ```
   if z ≤ μ:    T(z) = 1
   ```
4. Above the mean ⇒ **Cantelli one-sided Chebyshev bound**:
   ```
   T(z) = σ² / (σ² + (z − μ)²)
   ```
5. Floor at `(1 − α)` so partially-transparent occluders don't claim more
   shadow than the total alpha permits.

Reference: see `crates/rmesh-render/src/wgsl/deferred_shade_frag.wgsl::evaluate_transmittance`.

## Conservativeness

The Cantelli bound is exact only when the absorber distribution along the ray
is a 2-spike δ; for smoother distributions it over-shadows. Where the bound
matters most — directions with multiple well-separated absorbers and high
α-weighted depth variance — `T(z)` reads as larger than the true
transmittance, so receivers darken more than physically correct.

`crates/rmesh-dsm/tests/probe_vs_cpu_test.rs` renders error maps comparing
this bound to the exact transmittance computed by CPU ray traversal at
several query depths, per cubemap face.

## Storage cost

|   | Per-texel | Per-light @ 256³ | Notes |
|---|-----------|------------------|-------|
| Trigonometric (N=4) | 9 scalars (3 Rgba16F) | 9 MiB | Removed |
| **2-moment α-weighted** | **3 scalars (1 Rgba16F)** | **3 MiB** | Current |

## References

- Peters & Klein, "Moment Shadow Maps", I3D 2015 — power-moment shadow maps.
- Donnelly & Lauritzen, "Variance Shadow Maps", I3D 2006 — the original
  one-moment / Chebyshev variant this 2-moment formulation generalizes.
- For volumetric integration the α-weighted moments are accumulated via the
  same forward-pass weighting `w0/w1` used by `rmesh-render` so that mean and
  variance are taken over the **termination** distribution, not raw depth.
