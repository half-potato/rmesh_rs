// SH evaluation compute shader.
// One thread per tet: evaluates SH basis, adds gradient offset, applies softplus.

#import rmesh::sh::{C0, C1, C2_0, C2_1, C2_2, C2_3, C2_4, C3_0, C3_1, C3_2, C3_3, C3_4, C3_5, C3_6}
#import rmesh::math::softplus

struct ShEvalUniforms {
    cam_pos: vec4<f32>,
    tet_count: u32,
    sh_degree: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: ShEvalUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(4) var<storage, read> color_grads: array<f32>;
@group(0) @binding(5) var<storage, read_write> base_colors: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // Linearize 2D dispatch grid: row-major index
    let t = gid.x + gid.y * nwg.x * 256u;
    if (t >= u.tet_count) {
        return;
    }

    let nc = (u.sh_degree + 1u) * (u.sh_degree + 1u);
    let stride = nc * 3u;

    // Load vertex indices
    let i0 = indices[t * 4u];
    let i1 = indices[t * 4u + 1u];
    let i2 = indices[t * 4u + 2u];
    let i3 = indices[t * 4u + 3u];

    // Load vertex positions
    let v0 = vec3<f32>(vertices[i0 * 3u], vertices[i0 * 3u + 1u], vertices[i0 * 3u + 2u]);
    let v1 = vec3<f32>(vertices[i1 * 3u], vertices[i1 * 3u + 1u], vertices[i1 * 3u + 2u]);
    let v2 = vec3<f32>(vertices[i2 * 3u], vertices[i2 * 3u + 1u], vertices[i2 * 3u + 2u]);
    let v3 = vec3<f32>(vertices[i3 * 3u], vertices[i3 * 3u + 1u], vertices[i3 * 3u + 2u]);

    let centroid = (v0 + v1 + v2 + v3) * 0.25;

    // Direction: centroid - cam_pos (point FROM camera TO tet)
    let raw_dir = centroid - u.cam_pos.xyz;
    let len = length(raw_dir);
    var dir = vec3<f32>(0.0);
    if (len > 0.0) {
        dir = raw_dir / len;
    }

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let sh_base = t * stride;

    // Evaluate SH per channel
    var rgb = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let ch_base = sh_base + c * nc;

        // Degree 0
        var val = C0 * sh_coeffs[ch_base];

        // Degree 1
        if (u.sh_degree >= 1u) {
            val -= C1 * y * sh_coeffs[ch_base + 1u];
            val += C1 * z * sh_coeffs[ch_base + 2u];
            val -= C1 * x * sh_coeffs[ch_base + 3u];
        }

        // Degree 2
        if (u.sh_degree >= 2u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;
            val += C2_0 * xy * sh_coeffs[ch_base + 4u];
            val += C2_1 * yz * sh_coeffs[ch_base + 5u];
            val += C2_2 * (2.0 * zz - xx - yy) * sh_coeffs[ch_base + 6u];
            val += C2_3 * xz * sh_coeffs[ch_base + 7u];
            val += C2_4 * (xx - yy) * sh_coeffs[ch_base + 8u];
        }

        // Degree 3
        if (u.sh_degree >= 3u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            val += C3_0 * y * (3.0 * xx - yy) * sh_coeffs[ch_base + 9u];
            val += C3_1 * x * y * z * sh_coeffs[ch_base + 10u];
            val += C3_2 * y * (4.0 * zz - xx - yy) * sh_coeffs[ch_base + 11u];
            val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[ch_base + 12u];
            val += C3_4 * x * (4.0 * zz - xx - yy) * sh_coeffs[ch_base + 13u];
            val += C3_5 * z * (xx - yy) * sh_coeffs[ch_base + 14u];
            val += C3_6 * x * (xx - 3.0 * yy) * sh_coeffs[ch_base + 15u];
        }

        rgb[c] = val + 0.5;
    }

    // Gradient offset at v0
    let grad = vec3<f32>(color_grads[t * 3u], color_grads[t * 3u + 1u], color_grads[t * 3u + 2u]);
    let offset = dot(grad, v0 - centroid);
    rgb += vec3<f32>(offset);

    // Softplus activation (beta=10)
    base_colors[t * 3u]     = softplus(rgb.x);
    base_colors[t * 3u + 1u] = softplus(rgb.y);
    base_colors[t * 3u + 2u] = softplus(rgb.z);
}
