#define_import_path rmesh::math

// Safe math constants
const TINY_VAL: f32 = 1.0754944e-20;
const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;

fn safe_clip_f32(v: f32, minv: f32, maxv: f32) -> f32 {
    return max(min(v, maxv), minv);
}

fn safe_div_f32(a: f32, b: f32) -> f32 {
    if (abs(b) < TINY_VAL) {
        return safe_clip_f32(a / TINY_VAL, MIN_VAL, MAX_VAL);
    } else {
        return safe_clip_f32(a / b, MIN_VAL, MAX_VAL);
    }
}

fn softplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) {
        return x;
    }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn dsoftplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) {
        return 1.0;
    }
    let e = exp(10.0 * x);
    return safe_div_f32(e, 1.0 + e);
}

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return safe_div_f32(1.0 - exp(-x), x);
}

fn dphi_dx(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return -0.5 + x / 3.0; }
    let ex = exp(-x);
    return safe_div_f32(ex * (x + 1.0) - 1.0, x * x);
}

fn project_to_ndc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let inv_w = safe_div_f32(1.0, clip.w);
    return vec4<f32>(clip.xyz * inv_w, clip.w);
}
