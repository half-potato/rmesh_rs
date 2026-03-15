// Adam optimizer compute shader.
//
// One dispatch per parameter group (different learning rates for
// SH coefficients, vertices, densities, color gradients).
// Each thread updates one parameter element.

struct AdamUniforms {
    param_count: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    _pad: vec2<u32>,
};

// --- Safe math utilities (subset of safe_math.wgsl) ---
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

fn safe_sqrt_f32(a: f32) -> f32 {
    if (a < TINY_VAL) {
        return 0.0;
    } else {
        return sqrt(a);
    }
}
// --- End safe math utilities ---

@group(0) @binding(0) var<storage, read> uniforms: AdamUniforms;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = global_id.x + global_id.y * nwg.x * 256u;
    if (idx >= uniforms.param_count) {
        return;
    }

    let grad_raw = grads[idx];
    let grad = safe_clip_f32(select(grad_raw, 0.0, grad_raw != grad_raw), MIN_VAL, MAX_VAL);
    let step_f = f32(uniforms.step);

    // Update biased first moment
    let m_new = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment
    let v_new = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction: pow(a, n) = exp(n * ln(a))
    let beta1_pow = exp(step_f * log(uniforms.beta1));
    let beta2_pow = exp(step_f * log(uniforms.beta2));
    let m_hat = safe_div_f32(m_new, 1.0 - beta1_pow);
    let v_hat = safe_div_f32(v_new, 1.0 - beta2_pow);

    // Parameter update
    params[idx] -= uniforms.lr * safe_div_f32(m_hat, safe_sqrt_f32(v_hat) + uniforms.epsilon);
}
