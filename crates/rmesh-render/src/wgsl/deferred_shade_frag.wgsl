// Deferred PBR shading — fullscreen triangle render pass.
//
// Reads MRT textures from the forward rasterization pass:
//   color_tex:   albedo.rgb * a, a
//   aux0_tex:    roughness * a, env_f0 * a, metallic * a, a
//   normals_tex: gradient.xyz * a, a
//   depth_tex:   expected_depth * a, occlusion * a, env_f3 * a, a
// Plus hw depth buffer for world-position reconstruction.
// Group 1: cached Fourier DSM shadow atlas for transmittance lookup.

struct DeferredUniforms {
    inv_vp: mat4x4f,
    cam_pos: vec3f,
    num_lights: u32,
    width: u32,
    height: u32,
    ambient: f32,
    debug_mode: u32,
    near_plane: f32,
    far_plane: f32,
    dsm_enabled: u32,
    _pad: u32,
}

struct Light {
    position: vec3f,
    light_type: u32,    // 0=point, 1=spot, 2=directional
    color: vec3f,
    intensity: f32,
    direction: vec3f,
    inner_angle: f32,
    outer_angle: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct ShadowLight {
    vp0: mat4x4f,
    vp1: mat4x4f,
    vp2: mat4x4f,
    vp3: mat4x4f,
    vp4: mat4x4f,
    vp5: mat4x4f,
    face_offset: u32,
    face_count: u32,
    near: f32,
    far: f32,
    light_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Group 0: MRT textures + lights
@group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;     // albedo RGBA (premul alpha)
@group(0) @binding(2) var aux0_tex: texture_2d<f32>;      // roughness, env_f0, metallic, alpha
@group(0) @binding(3) var normals_tex: texture_2d<f32>;   // raw field gradient.xyz * alpha, alpha
@group(0) @binding(4) var depth_tex: texture_2d<f32>;     // expected depth, occlusion, env_f3, alpha
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var hw_depth_tex: texture_depth_2d; // hardware depth buffer

// Group 1: DSM shadow cubemap
@group(1) @binding(0) var dsm_rt0: texture_cube<f32>;
@group(1) @binding(1) var dsm_rt1: texture_cube<f32>;
@group(1) @binding(2) var dsm_rt2: texture_cube<f32>;
@group(1) @binding(3) var<storage, read> shadow_meta: array<ShadowLight>;
@group(1) @binding(4) var dsm_sampler: sampler;

// Debug mode constants
const DBG_FINAL:       u32 = 0u;
const DBG_RAW_ALBEDO:  u32 = 1u;
const DBG_TRUE_ALBEDO: u32 = 2u;
const DBG_NORMALS:     u32 = 3u;
const DBG_ROUGHNESS:   u32 = 4u;
const DBG_ENV_FEATURE: u32 = 5u;
const DBG_DEPTH:       u32 = 6u;
const DBG_SPECULAR:    u32 = 7u;
const DBG_DIFFUSE:     u32 = 8u;
const DBG_SHADOW:      u32 = 9u;
const DBG_METALLIC:    u32 = 10u;
const DBG_OCCLUSION:   u32 = 11u;
const DBG_PLASTER:     u32 = 12u;
const DBG_ALPHA:       u32 = 13u;
const DBG_PRIMITIVES:  u32 = 14u;
const DBG_DSM:         u32 = 15u;
const DBG_DSM_DEPTH:   u32 = 16u;
const DBG_DSM_FACE:    u32 = 17u;
const DBG_DSM_UV:      u32 = 18u;

const PI: f32 = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// DSM shadow helpers
// ---------------------------------------------------------------------------

fn get_shadow_vp(sm: ShadowLight, face: u32) -> mat4x4f {
    switch face {
        case 0u: { return sm.vp0; }
        case 1u: { return sm.vp1; }
        case 2u: { return sm.vp2; }
        case 3u: { return sm.vp3; }
        case 4u: { return sm.vp4; }
        default: { return sm.vp5; }
    }
}

/// Select cubemap face from a direction vector (world-space, light → surface).
fn select_cubemap_face(dir: vec3f) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, dir.x > 0.0);
    } else if a.y >= a.z {
        return select(3u, 2u, dir.y > 0.0);
    } else {
        return select(5u, 4u, dir.z > 0.0);
    }
}

/// Evaluate transmittance T(world_pos) using variance shadow map (Chebyshev bound).
fn evaluate_transmittance(world_pos: vec3f, li: u32, NdotL: f32) -> f32 {
    let sm = shadow_meta[li];

    let dir = world_pos - lights[li].position;
    let dist = length(dir);
    if dist < 1e-6 { return 1.0; }

    let face = select_cubemap_face(dir);
    let vp = get_shadow_vp(sm, face);
    let clip = vp * vec4f(world_pos, 1.0);
    if clip.w <= 0.0 { return 1.0; }

    let c0 = textureSample(dsm_rt0, dsm_sampler, dir);
    let c1 = textureSample(dsm_rt1, dsm_sampler, dir);
    let shadow_alpha = c0.a;

    if shadow_alpha < 0.01 { return 1.0; }

    let inv_alpha = 1.0 / shadow_alpha;
    let mean = c0.r * inv_alpha;
    let mean_sq = c1.r * inv_alpha;

    let z = (clip.w - sm.near) / (sm.far - sm.near);

    if z <= mean { return 1.0; }

    let variance = max(mean_sq - mean * mean, 3e-5);
    let d = z - mean;
    let p_max = variance / (variance + d * d);

    let T_total = 1.0 - shadow_alpha;
    return max(p_max, T_total);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let coords = vec2u(u32(frag_coord.x), u32(frag_coord.y));

    // Load MRT textures
    let color_raw = textureLoad(color_tex, coords, 0);
    let aux0_raw = textureLoad(aux0_tex, coords, 0);
    let normals_raw = textureLoad(normals_tex, coords, 0);
    let depth_raw = textureLoad(depth_tex, coords, 0);
    let hw_depth = textureLoad(hw_depth_tex, coords, 0);

    let alpha = color_raw.a;

    if (alpha < 0.01) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    // Un-premultiply all channels
    let inv_alpha = 1.0 / max(alpha, 1e-6);
    let albedo = color_raw.rgb * inv_alpha;
    let roughness = aux0_raw.r * inv_alpha;
    let env_f1 = aux0_raw.b * inv_alpha;

    let raw_gradient = normals_raw.rgb * inv_alpha;
    let normal = -normalize(vec3f(raw_gradient.x, raw_gradient.y, raw_gradient.z));

    let z_expected = depth_raw.r * inv_alpha;
    let env_f2 = depth_raw.g * inv_alpha;

    let near = uniforms.near_plane;
    let far = uniforms.far_plane;

    // Use hw depth where an opaque primitive wrote to it (hw_depth < 1.0),
    // otherwise use the volume's expected termination depth.
    var z_final = max(z_expected, near);
    let is_prim = hw_depth < 0.999;
    if is_prim {
        let z_hw = near * far / (far - hw_depth * (far - near));
        z_final = min(z_final, z_hw);
    }

    // Reconstruct world position from depth + inverse VP
    let ndc_z = (far * (z_final - near)) / (z_final * (far - near));
    let ndc_x = frag_coord.x / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - frag_coord.y / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, ndc_z, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    let view_dir = normalize(uniforms.cam_pos - world_pos);

    // PBR material properties (primitive pixels encode metallic/occlusion in MRT)
    let metallic = select(0.0, env_f1, is_prim);
    let ao = select(1.0, env_f2, is_prim);

    let f0 = mix(vec3f(0.04), albedo, metallic);
    let diffuse_color = albedo * (1.0 - metallic);
    let NdotV = max(dot(normal, view_dir), 1e-4);

    // Accumulate lighting
    var total_diffuse = vec3f(0.0);
    var total_specular = vec3f(0.0);
    var total_contribution = vec3f(0.0);

    for (var li = 0u; li < uniforms.num_lights; li++) {
        let light = lights[li];
        var to_light: vec3f;
        var atten: f32;

        if (light.light_type == 2u) {
            to_light = normalize(-light.direction);
            atten = 1.0;
        } else {
            let to_light_raw = light.position - world_pos;
            let dist = max(length(to_light_raw), 1e-6);
            to_light = to_light_raw / dist;
            atten = 1.0 / (dist * dist);
        }

        if (light.light_type == 1u) {
            let l_fwd = normalize(light.direction);
            let cos_a = dot(-to_light, l_fwd);
            let inner_cos = cos(light.inner_angle);
            let outer_cos = cos(light.outer_angle);
            let spot = clamp((cos_a - outer_cos) / (inner_cos - outer_cos + 1e-8), 0.0, 1.0);
            atten *= spot;
        }

        let NdotL = max(dot(normal, to_light), 0.0);
        let l_color = light.color * light.intensity;

        // Shadow transmittance from cached DSM
        var T = 1.0;
        if uniforms.dsm_enabled != 0u {
            T = evaluate_transmittance(world_pos, li, 1.0);
        }

        // Cook-Torrance GGX BRDF
        let half_v = normalize(to_light + view_dir);
        let NdotH = max(dot(normal, half_v), 0.0);

        // GGX distribution
        let a = roughness * roughness;
        let a2 = a * a;
        let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        let D = a2 / (PI * denom * denom + 1e-7);

        // Schlick-GGX geometry
        let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        let G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));

        // Fresnel-Schlick
        let F = f0 + (1.0 - f0) * pow(1.0 - max(dot(half_v, view_dir), 0.0), 5.0);

        let spec = D * G * F / max(4.0 * NdotV * NdotL, 0.001);
        let kD = (vec3f(1.0) - F) * (1.0 - metallic);

        total_contribution += (kD * diffuse_color / PI + spec) * NdotL * T * atten * l_color * ao;
        total_diffuse += kD * diffuse_color / PI * NdotL * T * atten * l_color * ao;
        total_specular += spec * NdotL * T * atten * l_color * ao;
    }

    var final_color = uniforms.ambient * albedo * ao + total_contribution;

    // Debug mode overrides
    let dm = uniforms.debug_mode;
    if (dm == DBG_RAW_ALBEDO)       { final_color = albedo; }
    else if (dm == DBG_TRUE_ALBEDO) { final_color = albedo; }
    else if (dm == DBG_NORMALS)     { final_color = normal * 0.5 + 0.5; }
    else if (dm == DBG_ROUGHNESS)   { final_color = vec3f(roughness); }
    else if (dm == DBG_ENV_FEATURE) { final_color = vec3f(metallic, ao, 0.0); }
    else if (dm == DBG_DEPTH)       { final_color = vec3f(z_expected * 0.1); }
    else if (dm == DBG_SPECULAR)    { final_color = total_specular; }
    else if (dm == DBG_DIFFUSE)     { final_color = total_diffuse; }
    else if (dm == DBG_SHADOW) {
        var T_total = 0.0;
        if uniforms.dsm_enabled != 0u {
            for (var li = 0u; li < uniforms.num_lights; li++) {
                T_total += evaluate_transmittance(world_pos, li, 1.0);
            }
            final_color = vec3f(T_total / f32(uniforms.num_lights));
        } else {
            final_color = vec3f(1.0);
        }
    }
    else if (dm == DBG_METALLIC)    { final_color = vec3f(metallic); }
    else if (dm == DBG_OCCLUSION)   { final_color = vec3f(ao); }
    else if (dm == DBG_PLASTER)     { final_color = albedo; }
    else if (dm == DBG_ALPHA)       { final_color = vec3f(alpha); }
    else if (dm == DBG_PRIMITIVES)  { /* final_color already correct */ }
    else if (dm == DBG_DSM_DEPTH) {
        if uniforms.dsm_enabled != 0u && uniforms.num_lights > 0u {
            let dir = world_pos - lights[0u].position;
            let c0 = textureSample(dsm_rt0, dsm_sampler, dir);
            let shadow_alpha = c0.a;
            let shadow_depth = select(0.0, c0.r / shadow_alpha, shadow_alpha > 0.01);
            final_color = vec3f(shadow_depth);
        } else {
            final_color = vec3f(0.5);
        }
    }
    else if (dm == DBG_DSM_FACE) {
        if uniforms.dsm_enabled != 0u && uniforms.num_lights > 0u {
            let to_surface = world_pos - lights[0u].position;
            let face = select_cubemap_face(to_surface);
            switch face {
                case 0u: { final_color = vec3f(1.0, 0.0, 0.0); }
                case 1u: { final_color = vec3f(0.0, 1.0, 1.0); }
                case 2u: { final_color = vec3f(0.0, 1.0, 0.0); }
                case 3u: { final_color = vec3f(1.0, 0.0, 1.0); }
                case 4u: { final_color = vec3f(0.0, 0.0, 1.0); }
                default: { final_color = vec3f(1.0, 1.0, 0.0); }
            }
        } else {
            final_color = vec3f(0.5);
        }
    }
    else if (dm == DBG_DSM_UV) {
        if uniforms.dsm_enabled != 0u && uniforms.num_lights > 0u {
            let sm = shadow_meta[0u];
            let to_surface = world_pos - lights[0u].position;
            var face = 0u;
            if sm.light_type == 0u {
                face = select_cubemap_face(to_surface);
            }
            let vp = get_shadow_vp(sm, face);
            let clip = vp * vec4f(world_pos, 1.0);
            if clip.w > 0.0 {
                let ndc_xy = clip.xy / clip.w;
                let uv = vec2f(ndc_xy.x * 0.5 + 0.5, 0.5 - ndc_xy.y * 0.5);
                final_color = vec3f(uv.x, uv.y, 0.0);
            } else {
                final_color = vec3f(1.0, 0.0, 1.0);
            }
        } else {
            final_color = vec3f(0.5);
        }
    }

    return vec4f(max(final_color, vec3f(0.0)), alpha);
}
