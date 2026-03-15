// safe_math.wgsl — Complete port of safe-math.slang
// Safe math utilities: division, sqrt, exp, log, normalize with NaN/Inf protection.
// Self-contained, no external dependencies. Copy-paste into shaders as needed.

// ============================================================================
// Constants
// ============================================================================

// Found experimentally in Slang — used as near-zero threshold
const TINY_VAL: f32 = 1.0754944e-20;
const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;

// ============================================================================
// nextafter — bitwise float manipulation
// ============================================================================

// Returns the next representable f32 value after x in the direction of y.
// WGSL has no intrinsic for this; implemented via bitcast.
fn nextafter_f32(x: f32, y: f32) -> f32 {
    if (x == y) { return y; }
    if (is_nan_f32(x) || is_nan_f32(y)) { return x + y; } // propagate NaN
    if (x == 0.0) {
        // Smallest subnormal toward y
        if (y > 0.0) {
            return bitcast<f32>(1u);
        } else {
            return bitcast<f32>(0x80000001u);
        }
    }
    let xi = bitcast<i32>(x);
    if ((x < y && x > 0.0) || (x > y && x < 0.0)) {
        return bitcast<f32>(xi + 1i);
    } else {
        return bitcast<f32>(xi - 1i);
    }
}

// ============================================================================
// NaN detection
// ============================================================================

fn is_nan_f32(x: f32) -> bool {
    return x != x;
}

fn is_nan_v3f(x: vec3<f32>) -> bool {
    return !((x.x == x.x) && (x.y == x.y) && (x.z == x.z));
}

fn is_nan_v4f(x: vec4<f32>) -> bool {
    return !((x.x == x.x) && (x.y == x.y) && (x.z == x.z) && (x.w == x.w));
}

// ============================================================================
// L1 / Linf norms
// ============================================================================

fn l1_v3f(v: vec3<f32>) -> f32 {
    return abs(v.x) + abs(v.y) + abs(v.z);
}

fn linf_v3f(v: vec3<f32>) -> f32 {
    return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

// ============================================================================
// Safe clip (clamp)
// ============================================================================

fn safe_clip_f32(v: f32, minv: f32, maxv: f32) -> f32 {
    return max(min(v, maxv), minv);
}

fn safe_clip_v3f(v: vec3<f32>, minv: f32, maxv: f32) -> vec3<f32> {
    return vec3<f32>(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv)
    );
}

fn safe_clip_v4f(v: vec4<f32>, minv: f32, maxv: f32) -> vec4<f32> {
    return vec4<f32>(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv),
        max(min(v.w, maxv), minv)
    );
}

// ============================================================================
// Safe division — tiny denominator protection + result clamping
// ============================================================================

// safe_div(float, float)
fn safe_div_f32(a: f32, b: f32) -> f32 {
    if (abs(b) < TINY_VAL) {
        return safe_clip_f32(a / TINY_VAL, MIN_VAL, MAX_VAL);
    } else {
        return safe_clip_f32(a / b, MIN_VAL, MAX_VAL);
    }
}

// safe_div(float2, float)
fn safe_div_v2f(a: vec2<f32>, b: f32) -> vec2<f32> {
    return vec2<f32>(
        safe_div_f32(a.x, b),
        safe_div_f32(a.y, b)
    );
}

// safe_div(float3, float)
fn safe_div_v3f(a: vec3<f32>, b: f32) -> vec3<f32> {
    return vec3<f32>(
        safe_div_f32(a.x, b),
        safe_div_f32(a.y, b),
        safe_div_f32(a.z, b)
    );
}

// safe_div(float4, float)
fn safe_div_v4f(a: vec4<f32>, b: f32) -> vec4<f32> {
    return vec4<f32>(
        safe_div_f32(a.x, b),
        safe_div_f32(a.y, b),
        safe_div_f32(a.z, b),
        safe_div_f32(a.w, b)
    );
}

// safe_div(float3, float3) — element-wise
fn safe_div_v3v3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        safe_div_f32(a.x, b.x),
        safe_div_f32(a.y, b.y),
        safe_div_f32(a.z, b.z)
    );
}

// ============================================================================
// Safe sqrt
// ============================================================================

fn safe_sqrt_f32(a: f32) -> f32 {
    if (a < TINY_VAL) {
        return 0.0;
    } else {
        return sqrt(a);
    }
}

// ============================================================================
// L2 normalize
// ============================================================================

fn l2_normalize_v4f(x: vec4<f32>) -> vec4<f32> {
    return safe_div_v4f(x, sqrt(max(dot(x, x), TINY_VAL)));
}

fn l2_normalize_v3f(x: vec3<f32>) -> vec3<f32> {
    return safe_div_v3f(x, sqrt(max(dot(x, x), TINY_VAL)));
}

// ============================================================================
// Safe exp
// ============================================================================

fn safe_exp_f32(v: f32) -> f32 {
    return exp(safe_clip_f32(v, MIN_VAL, log(MAX_VAL)));
}

// ============================================================================
// log1p — log(1 + v), more accurate for small v
// WGSL has no log1p intrinsic; use identity for small values, direct for large.
// ============================================================================

fn log1p_f32(v: f32) -> f32 {
    // For small v, log(1+v) loses precision. Use Taylor: v - v^2/2 + v^3/3 ...
    // Threshold where log(1+v) == v within f32 precision
    if (abs(v) < 1e-4) {
        return v - 0.5 * v * v + (1.0 / 3.0) * v * v * v;
    }
    return log(1.0 + v);
}

// ============================================================================
// Safe log
// ============================================================================

fn safe_log_f32(v: f32) -> f32 {
    return log(safe_clip_f32(v, TINY_VAL, MAX_VAL));
}

// ============================================================================
// expm1 — exp(v) - 1, more accurate for small v
// WGSL has no expm1 intrinsic; use Taylor expansion for small values.
// ============================================================================

fn expm1_f32(v: f32) -> f32 {
    // For small v, exp(v)-1 loses precision. Use Taylor: v + v^2/2 + v^3/6
    if (abs(v) < 1e-4) {
        return v + 0.5 * v * v + (1.0 / 6.0) * v * v * v;
    }
    return exp(v) - 1.0;
}

// safe_expm1 — same as expm1 (no separate backward needed in WGSL)
fn safe_expm1_f32(v: f32) -> f32 {
    return expm1_f32(v);
}

// ============================================================================
// log1mexp — log(1 - exp(-x)), numerically stable for all x > 0
// Uses log1p branch for large x, log(-expm1(-x)) for small x.
// Threshold 0.30102999566 ≈ log10(2) is the crossover point.
// ============================================================================

fn log1mexp_f32(x: f32) -> f32 {
    if (x > 0.30102999566) {
        return log1p_f32(max(-exp(-x), -1.0 + 1e-5));
    } else {
        return log(max(-expm1_f32(-x), 1e-20));
    }
}

// ============================================================================
// 4x4 matrix inverse
// ============================================================================

fn inverse_mat4(m: mat4x4<f32>) -> mat4x4<f32> {
    // Slang uses column-major m[col][row], WGSL also uses m[col][row]
    let n11 = m[0][0]; let n12 = m[1][0]; let n13 = m[2][0]; let n14 = m[3][0];
    let n21 = m[0][1]; let n22 = m[1][1]; let n23 = m[2][1]; let n24 = m[3][1];
    let n31 = m[0][2]; let n32 = m[1][2]; let n33 = m[2][2]; let n34 = m[3][2];
    let n41 = m[0][3]; let n42 = m[1][3]; let n43 = m[2][3]; let n44 = m[3][3];

    let t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    let t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    let t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    let t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    let det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    let idet = 1.0 / det;

    return mat4x4<f32>(
        // column 0
        vec4<f32>(
            t11 * idet,
            (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet,
            (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet,
            (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet
        ),
        // column 1
        vec4<f32>(
            t12 * idet,
            (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet,
            (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet,
            (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet
        ),
        // column 2
        vec4<f32>(
            t13 * idet,
            (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet,
            (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet,
            (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet
        ),
        // column 3
        vec4<f32>(
            t14 * idet,
            (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet,
            (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet,
            (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet
        )
    );
}

// ============================================================================
// Quaternion rotation (conjugate quaternion)
// Adapted from John D Cook: https://www.johndcook.com/blog/2021/06/16/faster-quaternion-rotations/
// ============================================================================

// rotate_vector(float3, float4) — conjugate quaternion rotation
fn rotate_vector_v3f(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(-q.yzw, v);
    return v + q.x * t + cross(-q.yzw, t);
}

// rotate_vector_T(float3, float4) — transpose (non-conjugate) quaternion rotation
fn rotate_vector_transpose_v3f(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

// rotate_vector(half3, half4) — conjugate quaternion rotation, f16 version
// Requires: enable f16;
// fn rotate_vector_v3h(v: vec3<f16>, q: vec4<f16>) -> vec3<f16> {
//     let t = 2.0h * cross(-q.yzw, v);
//     return v + q.x * t + cross(-q.yzw, t);
// }

// ============================================================================
// Tukey power ladder
// sign(x) * |p-1|/p * ((|x|/|p-1| + 1)^p - 1)
// ============================================================================

fn tukey_power_ladder_f32(x: f32, p: f32) -> f32 {
    let xp = abs(x);
    let xs = xp / max(TINY_VAL, abs(p - 1.0));
    let y = sign(x) * abs(p - 1.0) / p * (pow(xs + 1.0, p) - 1.0);
    return y;
}

// ============================================================================
// Softplus — log(1 + exp(beta * x)) / beta, with overflow guard
// ============================================================================

fn softplus_f32(x: f32, beta: f32) -> f32 {
    if (x * beta > 20.0) {
        return x;
    } else {
        return log(1.0 + exp(beta * x)) / beta;
    }
}

fn softplus_v3f(x: vec3<f32>, beta: f32) -> vec3<f32> {
    return vec3<f32>(
        softplus_f32(x.x, beta),
        softplus_f32(x.y, beta),
        softplus_f32(x.z, beta)
    );
}

// ============================================================================
// Half-precision helpers
// WGSL f16 requires "enable f16;" at top of shader.
// These are provided as f32 equivalents since f16 support is optional.
// ============================================================================

// hsign(half) — sign function that returns -1 or +1 (never 0)
fn hsign_f32(x: f32) -> f32 {
    if (x < 0.0) { return -1.0; } else { return 1.0; }
}

// f16 versions (uncomment when f16 is enabled in your shader):
// fn hsign_f16(x: f16) -> f16 {
//     if (x < 0.0h) { return -1.0h; } else { return 1.0h; }
// }
//
// fn hsqrt_f16(x: f16) -> f16 {
//     return sqrt(x);
// }
//
// fn f32_to_f16(x: f32) -> f16 {
//     return f16(x);
// }
//
// fn f16_to_f32(x: f16) -> f32 {
//     return f32(x);
// }
//
// fn f32_to_f16_v3(v: vec3<f32>) -> vec3<f16> {
//     return vec3<f16>(f16(v.x), f16(v.y), f16(v.z));
// }
//
// fn f16_to_f32_v3(v: vec3<f16>) -> vec3<f32> {
//     return vec3<f32>(f32(v.x), f32(v.y), f32(v.z));
// }
//
// fn f16_to_f32_v4(v: vec4<f16>) -> vec4<f32> {
//     return vec4<f32>(f32(v.x), f32(v.y), f32(v.z), f32(v.w));
// }

// ============================================================================
// phi and dphi_dx — (1 - exp(-x))/x with Taylor guard for small x
// ============================================================================

fn phi_f32(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return safe_div_f32(1.0 - exp(-x), x);
}

fn dphi_dx_f32(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return -0.5 + x / 3.0; }
    let ex = exp(-x);
    return safe_div_f32(ex * (x + 1.0) - 1.0, x * x);
}
