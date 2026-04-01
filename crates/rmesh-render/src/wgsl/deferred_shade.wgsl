// Deferred PBR shading compute shader.
//
// Reads the G-buffer (plaster, normals, depth, aux material channels) and
// neural material outputs (specular, lambda, retroreflectivity) to produce
// the final lit image with per-light shadow modulation.
//
// Reference: renderer/pbr_renderer.py render_relit()

struct DeferredUniforms {
    inv_vp: mat4x4f,
    cam_pos: vec3f,
    num_lights: u32,
    width: u32,
    height: u32,
    ambient: f32,
    debug_mode: u32,
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

// Group 0: G-buffer inputs + neural material outputs
@group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
@group(0) @binding(1) var<storage, read> rendered_image: array<f32>;       // plaster RGBA [W*H*4]
@group(0) @binding(2) var<storage, read> xyzd_image: array<f32>;           // normal.xyz + depth [W*H*4]
@group(0) @binding(3) var<storage, read> aux_image: array<f32>;            // [W*H*AUX_DIM]
@group(0) @binding(4) var<storage, read> shadow_transmittance: array<f32>; // [num_lights*W*H]
@group(0) @binding(5) var<storage, read> specular_buffer: array<f32>;      // [num_lights*W*H*3]
@group(0) @binding(6) var<storage, read> lambda_buffer: array<f32>;        // [num_lights*W*H]
@group(0) @binding(7) var<storage, read> retro_buffer: array<f32>;         // [W*H]

// Group 1: lights + output
@group(1) @binding(0) var<storage, read> lights: array<Light>;
@group(1) @binding(1) var<storage, read_write> output_image: array<f32>;   // [W*H*4]

const AUX_DIM: u32 = 8u;

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
const DBG_RETRO:       u32 = 10u;
const DBG_LAMBDA:      u32 = 11u;
const DBG_PLASTER:     u32 = 12u;
const DBG_ALPHA:       u32 = 13u;

fn linear_to_srgb(x: f32) -> f32 {
    if (x <= 0.0031308) {
        return 12.92 * max(x, 0.0);
    }
    return 1.055 * pow(max(x, 1e-7), 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb3(c: vec3f) -> vec3f {
    return vec3f(
        linear_to_srgb(c.x),
        linear_to_srgb(c.y),
        linear_to_srgb(c.z),
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let x = gid.x;
    let y = gid.y;
    if (x >= uniforms.width || y >= uniforms.height) { return; }

    let pixel_idx = y * uniforms.width + x;
    let wh = uniforms.width * uniforms.height;

    // Read plaster RGBA (alpha in log-space)
    let plaster_r = rendered_image[pixel_idx * 4u + 0u];
    let plaster_g = rendered_image[pixel_idx * 4u + 1u];
    let plaster_b = rendered_image[pixel_idx * 4u + 2u];
    let log_alpha = rendered_image[pixel_idx * 4u + 3u];
    let alpha = 1.0 - exp(log_alpha);

    if (alpha < 0.01) {
        output_image[pixel_idx * 4u + 0u] = 0.0;
        output_image[pixel_idx * 4u + 1u] = 0.0;
        output_image[pixel_idx * 4u + 2u] = 0.0;
        output_image[pixel_idx * 4u + 3u] = 0.0;
        return;
    }

    // Read aux channels: [roughness, env_feat(4), albedo(3)]
    let aux_base = pixel_idx * AUX_DIM;
    let roughness = aux_image[aux_base + 0u];
    let env_feat = vec4f(
        aux_image[aux_base + 1u],
        aux_image[aux_base + 2u],
        aux_image[aux_base + 3u],
        aux_image[aux_base + 4u],
    );
    let raw_albedo = vec3f(
        aux_image[aux_base + 5u],
        aux_image[aux_base + 6u],
        aux_image[aux_base + 7u],
    );

    // Recover true albedo by dividing out plaster (env lighting)
    let plaster = vec3f(plaster_r, plaster_g, plaster_b);
    let plaster_lum = max(0.2126 * plaster_r + 0.7152 * plaster_g + 0.0722 * plaster_b, 1e-3);
    let albedo = raw_albedo / plaster_lum;

    // Read normal
    let normal = normalize(vec3f(
        xyzd_image[pixel_idx * 4u + 0u],
        xyzd_image[pixel_idx * 4u + 1u],
        xyzd_image[pixel_idx * 4u + 2u],
    ));

    // Reconstruct world position from depth
    let depth = xyzd_image[pixel_idx * 4u + 3u];
    let ndc_x = (f32(x) + 0.5) / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(y) + 0.5) / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, depth, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    let view_dir = normalize(uniforms.cam_pos - world_pos);
    let retro = retro_buffer[pixel_idx];

    var total_contribution = vec3f(0.0);
    var total_diffuse = vec3f(0.0);
    var total_specular = vec3f(0.0);
    var first_shadow = 1.0;
    var first_lam = 0.5;

    for (var li = 0u; li < uniforms.num_lights; li++) {
        let light = lights[li];
        var to_light: vec3f;
        var atten: f32;

        if (light.light_type == 2u) {  // directional
            to_light = normalize(-light.direction);
            atten = 1.0;
        } else {  // point or spot
            let to_light_raw = light.position - world_pos;
            let dist = max(length(to_light_raw), 1e-6);
            to_light = to_light_raw / dist;
            atten = 1.0 / (dist * dist);
        }

        // Spot falloff
        if (light.light_type == 1u) {
            let l_fwd = normalize(light.direction);
            let cos_a = dot(-to_light, l_fwd);
            let inner_cos = cos(light.inner_angle);
            let outer_cos = cos(light.outer_angle);
            let spot = clamp((cos_a - outer_cos) / (inner_cos - outer_cos + 1e-8), 0.0, 1.0);
            atten *= spot;
        }

        let NdotL = max(dot(normal, to_light), 0.0);
        let shadow = shadow_transmittance[li * wh + pixel_idx];

        // Neural material results from burn
        let spec_base = li * wh * 3u + pixel_idx * 3u;
        let spec_color = vec3f(
            specular_buffer[spec_base + 0u],
            specular_buffer[spec_base + 1u],
            specular_buffer[spec_base + 2u],
        );
        let lam = lambda_buffer[li * wh + pixel_idx];

        // Retroreflectivity modulates NdotL
        let ndotl_mod = (1.0 - retro) + retro * NdotL;

        let l_color = light.color * light.intensity;
        let diffuse = albedo * ndotl_mod * atten * shadow * l_color;
        let specular = spec_color * atten * shadow * l_color;

        total_diffuse += (1.0 - lam) * diffuse;
        total_specular += lam * specular;
        total_contribution += (1.0 - lam) * diffuse + lam * specular;

        if (li == 0u) {
            first_shadow = shadow;
            first_lam = lam;
        }
    }

    var final_color = uniforms.ambient * albedo + total_contribution;

    // Apply debug mode overrides
    let dm = uniforms.debug_mode;
    if (dm == DBG_RAW_ALBEDO)  { final_color = raw_albedo; }
    else if (dm == DBG_TRUE_ALBEDO) { final_color = albedo; }
    else if (dm == DBG_NORMALS)     { final_color = normal * 0.5 + 0.5; }
    else if (dm == DBG_ROUGHNESS)   { final_color = vec3f(roughness); }
    else if (dm == DBG_ENV_FEATURE) { final_color = env_feat.xyz * 0.5 + 0.5; }
    else if (dm == DBG_DEPTH)       { final_color = vec3f(depth * 0.1); }
    else if (dm == DBG_SPECULAR)    { final_color = total_specular; }
    else if (dm == DBG_DIFFUSE)     { final_color = total_diffuse; }
    else if (dm == DBG_SHADOW)      { final_color = vec3f(first_shadow); }
    else if (dm == DBG_RETRO)       { final_color = vec3f(retro); }
    else if (dm == DBG_LAMBDA)      { final_color = vec3f(first_lam); }
    else if (dm == DBG_PLASTER)     { final_color = plaster; }
    else if (dm == DBG_ALPHA)       { final_color = vec3f(alpha); }

    // Tone map for final composite and some debug modes
    if (dm == DBG_FINAL || dm == DBG_SPECULAR || dm == DBG_DIFFUSE || dm == DBG_TRUE_ALBEDO || dm == DBG_RAW_ALBEDO || dm == DBG_PLASTER) {
        final_color = linear_to_srgb3(max(final_color, vec3f(0.0)));
    }

    output_image[pixel_idx * 4u + 0u] = final_color.x;
    output_image[pixel_idx * 4u + 1u] = final_color.y;
    output_image[pixel_idx * 4u + 2u] = final_color.z;
    output_image[pixel_idx * 4u + 3u] = alpha;
}
