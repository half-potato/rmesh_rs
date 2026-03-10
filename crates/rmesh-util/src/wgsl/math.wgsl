#define_import_path rmesh::math

fn softplus(x: f32) -> f32 {
    if (x > 8.0) {
        return x;
    }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn dsoftplus(x: f32) -> f32 {
    if (x > 8.0) {
        return 1.0;
    }
    let e = exp(10.0 * x);
    return e / (1.0 + e);
}

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return (1.0 - exp(-x)) / x;
}

fn dphi_dx(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return -0.5 + x / 3.0; }
    let ex = exp(-x);
    return (ex * (x + 1.0) - 1.0) / (x * x);
}

fn project_to_ndc(pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(pos, 1.0);
    let inv_w = 1.0 / (clip.w + 1e-6);
    return vec4<f32>(clip.xyz * inv_w, clip.w);
}
