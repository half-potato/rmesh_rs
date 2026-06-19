// MRT primitive rendering with full PBR material support.
//
// Bind group 0: per-primitive uniforms (dynamic offset)
// Bind group 1: material textures (base_color, metallic_roughness, normal, occlusion, sampler)
//
// MRT outputs (all Rgba16Float):
//   location(0): albedo.rgb, alpha=1
//   location(1): roughness, 0.0, metallic, alpha=1
//   location(2): world_normal.xyz, alpha=1
//   location(3): view_depth, occlusion, 0.0, alpha=1

struct PrimitiveUniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    model_col0: vec4<f32>,
    model_col1: vec4<f32>,
    model_col2: vec4<f32>,
    model_col3: vec4<f32>,
    color: vec4<f32>,
    roughness_factor: f32,
    metallic_factor: f32,
    occlusion_strength: f32,
    normal_scale: f32,
    tex_flags: u32,
}

@group(0) @binding(0) var<uniform> u: PrimitiveUniforms;

@group(1) @binding(0) var base_color_tex: texture_2d<f32>;
@group(1) @binding(1) var metallic_roughness_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var occlusion_tex: texture_2d<f32>;
@group(1) @binding(4) var mat_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) view_depth: f32,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) tangent_w: f32,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
) -> VsOut {
    let model = mat4x4<f32>(u.model_col0, u.model_col1, u.model_col2, u.model_col3);
    let vp = mat4x4<f32>(u.vp_col0, u.vp_col1, u.vp_col2, u.vp_col3);

    let world_pos = model * vec4<f32>(position, 1.0);
    let clip = vp * world_pos;

    let model3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let wn = normalize(model3 * normal);
    let wt = normalize(model3 * tangent.xyz);

    var out: VsOut;
    out.pos = clip;
    out.world_normal = wn;
    out.view_depth = clip.w;
    out.uv = uv;
    out.world_tangent = wt;
    out.tangent_w = tangent.w;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) expected_depth: vec4<f32>,
}

@fragment
fn fs_main(in: VsOut) -> FragmentOutput {
    // Base color
    var base_color = u.color;
    if (u.tex_flags & 1u) != 0u {
        base_color = base_color * textureSample(base_color_tex, mat_sampler, in.uv);
    }

    // Roughness + metallic
    var roughness = u.roughness_factor;
    var metallic = u.metallic_factor;
    if (u.tex_flags & 2u) != 0u {
        let mr = textureSample(metallic_roughness_tex, mat_sampler, in.uv);
        roughness = roughness * mr.g; // glTF: green = roughness
        metallic = metallic * mr.b;   // glTF: blue = metallic
    }

    // Normal
    var n = normalize(in.world_normal);
    if (u.tex_flags & 4u) != 0u {
        let ts_normal_raw = textureSample(normal_tex, mat_sampler, in.uv).rgb;
        var ts_normal = ts_normal_raw * 2.0 - 1.0;
        ts_normal.x = ts_normal.x * u.normal_scale;
        ts_normal.y = ts_normal.y * u.normal_scale;
        ts_normal = normalize(ts_normal);

        // Build TBN matrix
        let T = normalize(in.world_tangent);
        let N = n;
        let B = normalize(cross(N, T)) * in.tangent_w;
        let tbn = mat3x3<f32>(T, B, N);
        n = normalize(tbn * ts_normal);
    }

    // Occlusion
    var occlusion = 1.0;
    if (u.tex_flags & 8u) != 0u {
        let ao_sample = textureSample(occlusion_tex, mat_sampler, in.uv).r;
        occlusion = mix(1.0, ao_sample, u.occlusion_strength);
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(base_color.rgb, 1.0);
    out.aux0 = vec4<f32>(roughness, 0.0, metallic, 1.0);
    // Bias-encoded for Rgba8Unorm normals target. Consumers undo with *2-1.
    out.normals = vec4<f32>(-n * 0.5 + 0.5, 1.0);
    out.expected_depth = vec4<f32>(in.view_depth, occlusion, 0.0, 1.0);
    return out;
}
