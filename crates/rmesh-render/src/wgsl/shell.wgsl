// Density shell: the boundary surface of the density-thresholded region,
// rendered as opaque geometry.
//
// Only *back* faces are emitted — the point where a ray leaves the dense region
// is a bound on how far light can travel, so the depth this pass writes is a
// conservative "everything past here is occluded" plane. The interval pass
// already depth-tests LessEqual against the same buffer, so writing it here is
// all that is needed to kill occluded fragments; `shell_hiz` turns the same
// depth into a per-tet cull that runs before the sort.
//
// The color it writes is the second half of the idea: it sits *under* the
// volume, so residual transmittance reveals the model's own surface color
// instead of the background. Without it, raising the threshold washes the image
// out toward the clear color.
//
// Faces arrive as packed `(tet_id << 2) | face_idx` ids (see
// `flare::extract_boundary_faces`), so positions and normals are fetched live
// from the scene buffers — the shell follows animation and PBD vertex edits
// without re-extraction.
//
// MRT layout matches primitive_mrt / interval_fragment exactly:
//   location(0): albedo.rgb, 1
//   location(1): roughness, f0_dielectric, metallic, 1
//   location(2): -n * 0.5 + 0.5, 1   (Rgba8Unorm bias-encoded)
//   location(3): view_depth, occlusion(=1), view_depth^2, 1

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    c2w_col0: vec4<f32>,
    c2w_col1: vec4<f32>,
    c2w_col2: vec4<f32>,
    intrinsics: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size_u: u32,
    ray_mode: u32,
    min_t: f32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    shell_cull_bias: f32,
    shell_cull_flags: u32,
};

// Per-pass view. Kept separate from `Uniforms` so the DSM can draw the same
// shell through a cube-face VP with the light as the eye, reusing this shader
// unchanged.
struct ShellView {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    /// xyz = eye (camera or light) position. Back-facing is decided against
    /// this, in world space, so no NDC winding convention is involved.
    eye: vec4<f32>,
    /// x = face count, y = near plane, z = far plane, w reserved.
    /// near/far are only read by the DSM entry point.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> view: ShellView;
@group(0) @binding(1) var<storage, read> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> vertices: array<f32>;        // [V * 3]
@group(0) @binding(3) var<storage, read> indices: array<u32>;         // [M * 4]
@group(0) @binding(4) var<storage, read> shell_faces: array<u32>;     // packed face ids
@group(0) @binding(5) var<storage, read> vertex_normals: array<f32>;  // [V * 3]
@group(0) @binding(6) var<storage, read> sh_coeffs: array<u32>;
@group(0) @binding(7) var<storage, read> color_grads: array<f32>;     // [M * 3]
@group(0) @binding(8) var<storage, read> aux_data: array<f32>;        // [M * AUX_DIM]
@group(0) @binding(9) var<storage, read> base_colors_buf: array<f32>; // [M * 3]

const AUX_DIM: u32 = 6u;

// Face table: (a, b, c, opposite_vertex) — matches rmesh-util TET_FACE_INDICES
// and the CPU-side table in flare.rs.
const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2_0: f32 = 1.0925484305920792;
const C2_1: f32 = -1.0925484305920792;
const C2_2: f32 = 0.31539156525252005;
const C2_3: f32 = -1.0925484305920792;
const C2_4: f32 = 0.5462742152960396;
const C3_0: f32 = -0.5900435899266435;
const C3_1: f32 = 2.890611442640554;
const C3_2: f32 = -0.4570457994644658;
const C3_3: f32 = 0.3731763325901154;
const C3_4: f32 = -0.4570457994644658;
const C3_5: f32 = 1.445305721320277;
const C3_6: f32 = -0.5900435899266435;

fn softplus(x: f32) -> f32 {
    if (x * 10.0 > 20.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn read_sh_f16(base: u32, i: u32) -> f32 {
    let word = sh_coeffs[base + i / 2u];
    let unpacked = unpack2x16float(word);
    return select(unpacked.x, unpacked.y, (i & 1u) != 0u);
}

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

fn load_normal(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertex_normals[i], vertex_normals[i + 1u], vertex_normals[i + 2u]);
}

// Same evaluation as project_compute.wgsl, sampled at the face center rather
// than the tet centroid so a large tet's shell face gets the color of the part
// actually on the surface.
fn eval_sh_color(tet_id: u32, dir: vec3<f32>, at: vec3<f32>, centroid: vec3<f32>) -> vec3<f32> {
    if (arrayLength(&sh_coeffs) <= 1u) {
        // Training path: no SH coefficients loaded, fall back to base colors.
        return vec3<f32>(
            base_colors_buf[tet_id * 3u],
            base_colors_buf[tet_id * 3u + 1u],
            base_colors_buf[tet_id * 3u + 2u],
        );
    }

    let x = dir.x; let y = dir.y; let z = dir.z;
    let nc = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u);
    let stride = nc * 3u;
    let sh_base = tet_id * ((stride + 1u) / 2u);

    var rgb = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let ch_offset = c * nc;
        var val = C0 * read_sh_f16(sh_base, ch_offset);

        if (uniforms.sh_degree >= 1u) {
            val -= C1 * y * read_sh_f16(sh_base, ch_offset + 1u);
            val += C1 * z * read_sh_f16(sh_base, ch_offset + 2u);
            val -= C1 * x * read_sh_f16(sh_base, ch_offset + 3u);
        }
        if (uniforms.sh_degree >= 2u) {
            let xx = x * x; let yy = y * y; let zz = z * z;
            let xy = x * y; let yz = y * z; let xz = x * z;
            val += C2_0 * xy * read_sh_f16(sh_base, ch_offset + 4u);
            val += C2_1 * yz * read_sh_f16(sh_base, ch_offset + 5u);
            val += C2_2 * (2.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 6u);
            val += C2_3 * xz * read_sh_f16(sh_base, ch_offset + 7u);
            val += C2_4 * (xx - yy) * read_sh_f16(sh_base, ch_offset + 8u);
        }
        if (uniforms.sh_degree >= 3u) {
            let xx = x * x; let yy = y * y; let zz = z * z;
            val += C3_0 * y * (3.0 * xx - yy) * read_sh_f16(sh_base, ch_offset + 9u);
            val += C3_1 * x * y * z * read_sh_f16(sh_base, ch_offset + 10u);
            val += C3_2 * y * (4.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 11u);
            val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * read_sh_f16(sh_base, ch_offset + 12u);
            val += C3_4 * x * (4.0 * zz - xx - yy) * read_sh_f16(sh_base, ch_offset + 13u);
            val += C3_5 * z * (xx - yy) * read_sh_f16(sh_base, ch_offset + 14u);
            val += C3_6 * x * (xx - 3.0 * yy) * read_sh_f16(sh_base, ch_offset + 15u);
        }
        rgb[c] = val + 0.5;
    }

    let grad = vec3<f32>(
        color_grads[tet_id * 3u],
        color_grads[tet_id * 3u + 1u],
        color_grads[tet_id * 3u + 2u],
    );
    rgb += vec3<f32>(dot(grad, at - centroid));

    return vec3<f32>(softplus(rgb.x), softplus(rgb.y), softplus(rgb.z));
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    /// Volume-integrated SH radiance. This is the *color-only* fallback, used
    /// when the scene carries no PBR aux data. It is NOT a PBR base color —
    /// feeding it to the deferred pass as albedo double-counts lighting and
    /// blows the shell out to white — so `fs_main` prefers `aux_data` when it
    /// exists, exactly as `interval_fragment.wgsl` does.
    @location(0) @interpolate(flat) sh_color: vec3<f32>,
    @location(1) view_depth: f32,
    /// Interpolated learned vertex normal. These are gradient-like (they point
    /// *into* the density, same as the volume path's `field_gradient`), so the
    /// outward surface normal is its negation.
    @location(2) grad: vec3<f32>,
    /// Geometric outward face normal — the fallback, and the reference the
    /// learned normal's sign is checked against.
    @location(3) @interpolate(flat) geo_outward: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
};

// A triangle collapsed to a point produces no fragments. Same trick
// interval_compute.wgsl uses to drop tets it doesn't want to draw.
fn degenerate() -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.sh_color = vec3<f32>(0.0);
    out.view_depth = 0.0;
    out.grad = vec3<f32>(0.0);
    out.geo_outward = vec3<f32>(0.0, 0.0, 1.0);
    out.tet_id = 0u;
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let tri = vid / 3u;
    let corner = vid % 3u;

    if (tri >= u32(view.params.x)) {
        return degenerate();
    }

    let packed = shell_faces[tri];
    let tet_id = packed >> 2u;
    let face = TET_FACES[packed & 3u];

    let base = tet_id * 4u;
    let ia = indices[base + face.x];
    let ib = indices[base + face.y];
    let ic = indices[base + face.z];
    let iopp = indices[base + face.w];

    let pa = load_vertex(ia);
    let pb = load_vertex(ib);
    let pc = load_vertex(ic);
    let popp = load_vertex(iopp);

    let n_raw = cross(pb - pa, pc - pa);
    let n_len = length(n_raw);
    if (n_len < 1e-12) {
        return degenerate(); // degenerate face
    }

    let center = (pa + pb + pc) * (1.0 / 3.0);

    // Orient outward: away from the tet's own opposite vertex. Identical rule
    // to the CPU extractor in flare.rs.
    var outward = n_raw / n_len;
    if (dot(outward, center - popp) < 0.0) {
        outward = -outward;
    }

    // Keep only faces the ray leaves through: outward pointing away from the
    // eye. This is the exit of the dense region, which is the bound we want.
    // Decided in world space, so it is independent of NDC winding and works
    // unchanged when `eye` is a light position.
    if (dot(outward, center - view.eye.xyz) <= 0.0) {
        return degenerate();
    }

    var p = pa;
    var vn_idx = ia;
    if (corner == 1u) { p = pb; vn_idx = ib; }
    else if (corner == 2u) { p = pc; vn_idx = ic; }

    let vp = mat4x4<f32>(view.vp_col0, view.vp_col1, view.vp_col2, view.vp_col3);
    let clip = vp * vec4<f32>(p, 1.0);

    let raw_dir = center - view.eye.xyz;
    let dir_len = length(raw_dir);
    var dir = vec3<f32>(0.0);
    if (dir_len > 0.0) { dir = raw_dir / dir_len; }

    let centroid = (pa + pb + pc + popp) * 0.25;

    var out: VsOut;
    out.pos = clip;
    out.sh_color = eval_sh_color(tet_id, dir, center, centroid);
    out.view_depth = clip.w;
    if (arrayLength(&vertex_normals) > 1u) {
        out.grad = load_normal(vn_idx);
    } else {
        out.grad = -outward;
    }
    out.geo_outward = outward;
    out.tet_id = tet_id;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) expected_depth: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> FragmentOutput {
    // Learned normals are gradient-like (inward), so negate for the outward
    // surface normal. Fall back to the face normal when the learned one is
    // unusable, and flip it if it disagrees with the geometric orientation —
    // a per-vertex normal shared with neighbouring tets can point the wrong way
    // on a shell face.
    var n = in.geo_outward;
    let g_len = length(in.grad);
    if (g_len > 1e-6) {
        n = -in.grad / g_len;
        if (dot(n, in.geo_outward) < 0.0) {
            n = -n;
        }
    }

    // Per-tet PBR channels, same defaults as interval_fragment when the scene
    // carries no aux data.
    let has_pbr = arrayLength(&aux_data) > 1u;
    let aux_base = in.tet_id * AUX_DIM;
    let roughness     = select(0.5,  aux_data[aux_base + 0u], has_pbr);
    let metallic      = select(0.0,  aux_data[aux_base + 1u], has_pbr);
    let f0_dielectric = select(0.04, aux_data[aux_base + 2u], has_pbr);

    // Albedo comes from the PBR aux channels, NOT from SH. The SH color is
    // volume-integrated *radiance* — already lit — so handing it to the
    // deferred pass as a base color makes it get lit a second time and the
    // shell saturates to white. `interval_fragment.wgsl` makes exactly this
    // distinction; the shell has to agree with it or the backstop will not
    // match the volume it sits behind.
    let albedo = select(in.sh_color,
                        vec3<f32>(aux_data[aux_base + 3u],
                                  aux_data[aux_base + 4u],
                                  aux_data[aux_base + 5u]),
                        has_pbr);

    var out: FragmentOutput;
    out.color = vec4<f32>(albedo, 1.0);
    out.aux0 = vec4<f32>(roughness, f0_dielectric, metallic, 1.0);
    // Bias-encoded for the Rgba8Unorm normals target; the negation matches
    // primitive_mrt so the deferred pass's un-negate recovers `n`.
    out.normals = vec4<f32>(-n * 0.5 + 0.5, 1.0);
    // .g = baked occlusion (1.0 — the shell has no occlusion map).
    // .b = E[z²], which for an opaque surface at z is exactly z².
    out.expected_depth = vec4<f32>(in.view_depth, 1.0, in.view_depth * in.view_depth, 1.0);
    return out;
}

// DSM entry: an alpha=1 absorber moment at the shell's depth, matching
// dsm_primitive.wgsl and dsm_moment_fragment.wgsl.
//
// Writing this matters for correctness, not just fidelity. The shell's depth
// already culls tet intervals behind it out of the DSM, so if the shell itself
// contributed no occlusion, an opaque wall would end up casting a *lighter*
// shadow than before — occluders removed with nothing standing in for them.
@fragment
fn fs_dsm_moment(in: VsOut) -> @location(0) vec4<f32> {
    let near = view.params.y;
    let far = view.params.z;
    let z_linear = near * far / (far - in.pos.z * (far - near));
    let z = clamp((z_linear - near) / (far - near), 0.0, 1.0);
    return vec4<f32>(z, z * z, 0.0, 1.0);
}
