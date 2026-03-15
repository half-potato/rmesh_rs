# WGSL Shader Numerical Stability Audit

**Date**: 2026-03-15
**Scope**: All rendering/training WGSL shaders in `rmeshvk/`
**Reference**: `safe-math.slang` — `safe_div` (den >= TINY_VAL=1e-20, result clamped to +/-1e20), `safe_exp` (input clamped to +/-46), `safe_sqrt` (input >= TINY_VAL), `safe_clip`
**Existing WGSL port**: `crates/rmesh-util/src/wgsl/safe_math.wgsl` — already has `safe_div_f32`, `safe_exp_f32`, `safe_sqrt_f32`, `safe_clip_f32`, etc. but none of the rendering shaders use them yet.

---

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL | 3 | Causes NaN/Inf in normal first-iteration training |
| HIGH | 8 | Causes NaN with common edge-case inputs (large grads, thin tets) |
| MEDIUM | 3 | Theoretical concern for unusual camera configurations |
| LOW | 6 | Already partially guarded or extremely unlikely |

### Root Cause Chain (Known Symptom)

```
Forward: large color_grads → unbounded base_color → huge c_premul → image values ~1e27
         ↓
Loss:    diff = 1e27 - gt → dl_d_image ~ 1e27
         ↓
Backward: t_prev = exp(prev_log_t) overflows → Inf × anything = NaN
          1.0/den with small den → 1e20 × large grads → Inf
          ↓
Adam:     NaN grads → NaN moments → NaN params → all subsequent iters NaN
```

---

## CRITICAL Priority

### C1. `backward_tiled_compute.wgsl:385` — `exp(prev_log_t)` overflow

```wgsl
let prev_log_t = log_t + od;        // line 384
let t_prev = exp(prev_log_t);       // line 385
```

**Problem**: In back-to-front traversal, `prev_log_t` accumulates `od` from each undone tet. For a pixel seeing N tets each with od=1, `prev_log_t` grows to N. `exp(89)` overflows f32 (max ~3.4e38). With 100+ tets per ray, this routinely overflows.

**Fix**: Replace with `safe_exp_f32(prev_log_t)` which clamps input to `[-1e20, log(1e20)] ≈ [-1e20, 46.05]`.

**Downstream**: This overflow propagates to lines 386, 396, 398:
```wgsl
let prev_color = color - c_premul * t_prev;    // line 386: Inf * anything = Inf/NaN
let d_c_premul = d_color * t_prev;             // line 396: same
let d_old_log_t = d_log_t + dot(d_color, c_premul) * t_prev;  // line 398: same
```

### C2. `backward_tiled_compute.wgsl:396-398` — Gradient explosion from `t_prev`

```wgsl
let d_c_premul = d_color * t_prev;                              // line 396
let d_od_state = -d_log_t;                                      // line 397
let d_old_log_t = d_log_t + dot(d_color, c_premul) * t_prev;   // line 398
```

**Problem**: Even if t_prev is clamped (fixing C1), `d_color` can be ~1e27 (from forward producing huge values), and t_prev * d_color remains huge. The accumulation via `d_old_log_t` compounds across tets.

**Fix**:
1. Clamp `t_prev` via `safe_exp_f32` (fixes C1).
2. Clamp `d_c_premul = safe_clip_v3f(d_color * t_prev, MIN_VAL, MAX_VAL)`.
3. Clamp `d_old_log_t = safe_clip_f32(d_log_t + dot(d_color, c_premul) * t_prev, MIN_VAL, MAX_VAL)`.

### C3. `adam_compute.wgsl:43-46` — Division by zero at step=0

```wgsl
let beta1_pow = exp(step_f * log(uniforms.beta1));   // line 43
let beta2_pow = exp(step_f * log(uniforms.beta2));   // line 44
let m_hat = m_new / (1.0 - beta1_pow);              // line 45
let v_hat = v_new / (1.0 - beta2_pow);              // line 46
```

**Problem**: If `step_f = 0` (step 0), `beta1_pow = exp(0) = 1.0`, so `1.0 - beta1_pow = 0.0` → division by zero → `m_hat = Inf/NaN`. Same for `v_hat`. Then `sqrt(NaN)` on line 49 → NaN propagates to params.

**Fix**: Use `safe_div_f32(m_new, 1.0 - beta1_pow)` and `safe_div_f32(v_new, 1.0 - beta2_pow)`. Or ensure `step` is always >= 1 on the CPU side (defensive approach: do both).

---

## HIGH Priority

### H1. `backward_tiled_compute.wgsl:458-460,471-473` — `1.0 / den` in dt/dv gradients

```wgsl
// Lines 458-460 (min_face gradient):
let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);

// Lines 471-473 (max_face gradient):
let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);
```

**Problem**: `den = dot(n, ray_dir)` where `n = cross(vc - va, vb - va)` (unflipped). The intersection loop guaranteed `abs(den) >= 1e-20` for this face, so `1/den` can be up to `1e20`. When multiplied by `d_t_min` or `d_t_max` (which can themselves be large), the vertex gradients can overflow.

**Fix**: Replace `(1.0 / den)` with `safe_div_f32(1.0, den)` which clamps the result to `[-1e20, 1e20]`. Then clamp each `dt_dv*` result: `safe_clip_v3f(dt_dva * d_t_min, MIN_VAL, MAX_VAL)`.

### H2. `rasterize_compute.wgsl:377-391` — Unbounded forward color values

```wgsl
let base_offset = dot(grad_vec, ray_o - verts[0]);              // line 377
let base_color = colors_tet + vec3<f32>(base_offset);           // line 378
...
let c_start = max(base_color + vec3<f32>(dc_dt * t_min_val), vec3<f32>(0.0));  // line 381
let c_end = max(base_color + vec3<f32>(dc_dt * t_max_val), vec3<f32>(0.0));    // line 382
...
let c_premul = c_end * w0 + c_start * w1;                      // line 391
```

**Problem**: `grad_vec` (from `color_grads_buf`) has no bounds. If `grad_vec` components are ~1e10 and `(ray_o - verts[0])` has magnitude ~10, `base_offset` ~ 1e11. After adding to `colors_tet`, `c_premul` can be ~1e11+. This is the direct cause of "image values up to 1e27" in the known symptom.

**Fix**: Clamp `c_premul`:
```wgsl
let c_premul = safe_clip_v3f(c_end * w0 + c_start * w1, 0.0, MAX_VAL);
```
Or more aggressively, clamp `base_color`:
```wgsl
let base_color = safe_clip_v3f(colors_tet + vec3<f32>(base_offset), MIN_VAL, MAX_VAL);
```

### H3. `raytrace_compute.wgsl:187-201` — Unbounded forward color values (same as H2)

```wgsl
let base_offset = dot(grad, cam - verts[0]);                    // line 187
let base_color = colors_tet + vec3<f32>(base_offset);           // line 188
...
let c_start = max(base_color + vec3<f32>(dc_dt * t_min_val), vec3<f32>(0.0));  // line 191
let c_end = max(base_color + vec3<f32>(dc_dt * t_max_val), vec3<f32>(0.0));    // line 192
...
let c_premul = c_end * w0 + c_start * w1;                      // line 201
```

**Fix**: Same as H2 — clamp `c_premul` (and optionally `base_color`).

### H4. `backward_tiled_compute.wgsl:360-375` — Unbounded backward color values

```wgsl
let base_offset = dot(grad_vec, cam - verts[0]);                // line 360
let base_color = colors_tet + vec3<f32>(base_offset);           // line 361
...
let c_premul = c_end * w0 + c_start * w1;                      // line 375
```

**Problem**: Same as H2 but in the backward pass. The c_premul value is used in state undoing (line 386) and gradient computation (line 410), amplifying the instability.

**Fix**: Clamp `c_premul` to `[-MAX_VAL, MAX_VAL]`:
```wgsl
let c_premul = safe_clip_v3f(c_end * w0 + c_start * w1, MIN_VAL, MAX_VAL);
```

### H5. `forward_fragment.wgsl:84-85` — Unbounded HW raster color values

```wgsl
let c_start = max(base_color + vec3<f32>(dc_dt * t_min), vec3<f32>(0.0));  // line 84
let c_end = max(base_color + vec3<f32>(dc_dt * t_max), vec3<f32>(0.0));    // line 85
```

**Fix**: Same — clamp after compute_integral:
```wgsl
out.color = safe_clip_v4f(compute_integral(c_end, c_start, od), MIN_VAL, MAX_VAL);
```

### H6. `forward_vertex.wgsl:119-120` — Unbounded base_color passed to fragment

```wgsl
let offset = dot(grad, cam - verts[0]);         // line 119
out.base_color = color + vec3<f32>(offset);     // line 120
```

**Problem**: Unbounded `offset` makes `base_color` huge, which then arrives in the fragment shader (H5).

**Fix**: Clamp the offset or base_color:
```wgsl
out.base_color = safe_clip_v3f(color + vec3<f32>(offset), MIN_VAL, MAX_VAL);
```

### H7. `backward_tiled_compute.wgsl:482-487` — Unbounded gradient accumulators

```wgsl
d_density_accum += d_density_local;      // line 482
d_vert_accum[0] += d_vert_local[0];     // line 483
...
d_grad_accum += d_grad_local;            // line 487
```

**Problem**: Each pixel adds its contribution. With 256 pixels per tile and large per-pixel gradients (from C1/C2/H1), the accumulation can overflow f32.

**Fix**: Clamp each per-pixel contribution before accumulation:
```wgsl
d_density_accum += safe_clip_f32(d_density_local, MIN_VAL, MAX_VAL);
d_vert_accum[vi] += safe_clip_v3f(d_vert_local[vi], MIN_VAL, MAX_VAL);
d_grad_accum += safe_clip_v3f(d_grad_local, MIN_VAL, MAX_VAL);
d_base_colors_accum += safe_clip_v3f(d_base_color, MIN_VAL, MAX_VAL);
```

### H8. `adam_compute.wgsl:31-39` — NaN propagation from backward into moments

```wgsl
let grad = grads[idx];                                          // line 31
let m_new = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * grad;        // line 35
let v_new = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * grad * grad; // line 39
```

**Problem**: If `grad` is NaN (from backward NaN), `m_new` and `v_new` become NaN. Once moments are NaN, all subsequent steps produce NaN parameters forever.

**Fix**: Sanitize grad before use:
```wgsl
let grad_raw = grads[idx];
let grad = select(grad_raw, 0.0, grad_raw != grad_raw);  // NaN → 0
```
Or clamp: `let grad = safe_clip_f32(grads[idx], MIN_VAL, MAX_VAL);`

---

## MEDIUM Priority

### M1. `backward_tiled_compute.wgsl:321-322` — Perspective unprojection w-divide

```wgsl
let near_world = near_clip.xyz / near_clip.w;    // line 321
let far_world = far_clip.xyz / far_clip.w;       // line 322
```

**Problem**: `near_clip.w` or `far_clip.w` could be near-zero for extreme inv_vp matrices.

**Fix**: `let near_world = near_clip.xyz / max(abs(near_clip.w), 1e-20) * sign(near_clip.w);`
Or use `safe_div_v3f(near_clip.xyz, near_clip.w)`.

### M2. `rasterize_compute.wgsl:332-333` — Same perspective unprojection

```wgsl
let near_world = near_clip.xyz / near_clip.w;    // line 332
let far_world = far_clip.xyz / far_clip.w;       // line 333
```

**Fix**: Same as M1.

### M3. `raytrace_compute.wgsl:337-338` — Same perspective unprojection

```wgsl
let near_world = near_clip.xyz / near_clip.w;    // line 337
let far_world = far_clip.xyz / far_clip.w;       // line 338
```

**Fix**: Same as M1.

---

## LOW Priority

### L1. `math.wgsl:7` — `softplus()` exp overflow edge case

```wgsl
fn softplus(x: f32) -> f32 {
    if (x > 8.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}
```

**Status**: Already guarded — for x <= 8.0, `exp(80)` ~ 5.5e34 is within f32 range. **No fix needed.**

### L2. `math.wgsl:10-16` — `dsoftplus()` exp overflow edge case

**Status**: Same guard at x > 8.0 means exp(80) max. Ratio `e/(1+e)` approaches 1.0 smoothly. **No fix needed.**

### L3. All `phi()` implementations — Taylor-guarded division

```wgsl
fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return (1.0 - exp(-x)) / x;
}
```

**Status**: For |x| >= 1e-6, the function is well-behaved: numerator bounded by [0,1], denominator >= 1e-6, result <= 1e6 (but in practice much smaller since numerator also shrinks). **No fix needed.**

### L4. All `dphi_dx()` implementations — Taylor-guarded division by x*x

**Status**: For |x| >= 1e-6, numerator approaches 0 as fast as denominator, ratio is smooth. **No fix needed.**

### L5. `raytrace_compute.wgsl:255` — BVH `inv_dir` infinity

```wgsl
let inv_dir = vec3<f32>(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);
```

**Status**: Standard BVH practice. Inf values in inv_dir produce correct AABB slab test results via IEEE 754 Inf arithmetic. **No fix needed.**

### L6. `forward_fragment.wgsl:106` — `normalize()` of potentially-zero vector

```wgsl
let entry_normal = normalize(-entry_normal_raw);
```

**Status**: Degenerate only for zero-area faces, extremely unlikely. If it happens, only affects normal output, not color/alpha. **No fix needed** (or optionally use `l2_normalize_v3f`).

---

## Recommended Fix Priority Order

1. **C1 + C2**: Replace `exp(prev_log_t)` with `safe_exp_f32(prev_log_t)` and clamp downstream products. This is the #1 cause of NaN in backward.

2. **H2 + H3 + H4**: Clamp `c_premul` in all three forward shaders and the backward shader. This prevents the forward pass from producing 1e27-scale values that make the backward pass impossible.

3. **H1**: Replace `1.0 / den` with `safe_div_f32(1.0, den)` in backward dt/dv computations.

4. **C3**: Fix Adam bias correction division-by-zero at step=0.

5. **H7**: Clamp per-pixel gradient contributions before accumulation.

6. **H8**: Sanitize NaN gradients in Adam to prevent moment poisoning.

7. **H5 + H6**: Clamp colors in HW raster path (forward_vertex + forward_fragment).

8. **M1-M3**: Use safe division in perspective unprojection (low urgency, rarely triggered).

---

## Files Requiring Changes (Ordered by Urgency)

| File | Changes | Lines Affected |
|------|---------|----------------|
| `backward_tiled_compute.wgsl` | safe_exp, safe_div, safe_clip on gradients | 385, 386, 396, 398, 458-460, 471-473, 482-487 |
| `rasterize_compute.wgsl` | safe_clip on c_premul, safe_div on w-divide | 321-322, 332-333, 377-391 |
| `raytrace_compute.wgsl` | safe_clip on c_premul, safe_div on w-divide | 187-201, 337-338 |
| `adam_compute.wgsl` | safe_div on bias correction, NaN guard on grad | 31, 35, 39, 43-46 |
| `forward_fragment.wgsl` | safe_clip on color output | 84-88 |
| `forward_vertex.wgsl` | safe_clip on base_color | 119-120 |
| `math.wgsl` | No changes needed (already stable) | — |

All fixes should import/inline functions from `safe_math.wgsl` (`safe_div_f32`, `safe_exp_f32`, `safe_clip_f32`, `safe_clip_v3f`).
