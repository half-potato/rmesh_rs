// Deferred PBR shading — fullscreen triangle render pass.
//
// Reads MRT textures from the forward rasterization pass (plaster, aux0,
// normals, depth+albedo), computes per-pixel lighting, outputs lit color
// in linear space (blit handles sRGB conversion).
//
// Phase A: Lambertian diffuse only (no neural specular, no shadows).

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

@group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;    // plaster RGBA (premul alpha)
@group(0) @binding(2) var aux0_tex: texture_2d<f32>;     // roughness, env_feat[0..2]
@group(0) @binding(3) var normals_tex: texture_2d<f32>;  // normal.xyz * alpha, env_feat[3] * alpha
@group(0) @binding(4) var depth_tex: texture_2d<f32>;    // depth * alpha, albedo.rgb * alpha
@group(0) @binding(5) var<storage, read> lights: array<Light>;

// Debug mode constants
const DBG_FINAL:       u32 = 0u;
const DBG_RAW_ALBEDO:  u32 = 1u;
const DBG_TRUE_ALBEDO: u32 = 2u;
const DBG_NORMALS:     u32 = 3u;
const DBG_ROUGHNESS:   u32 = 4u;
const DBG_ENV_FEATURE: u32 = 5u;
const DBG_DEPTH:       u32 = 6u;
const DBG_PLASTER:     u32 = 12u;
const DBG_ALPHA:       u32 = 13u;

struct VsOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle: 3 vertices cover clip space [-1,1]^2.
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

    // Alpha from plaster (premultiplied — .a channel is alpha directly for Rgba16Float blend)
    // The forward pass uses premultiplied alpha blending: rgb *= alpha, a = alpha.
    // After all tets are blended, total_alpha = 1 - product(1 - alpha_i) is in .a.
    let alpha = color_raw.a;

    if (alpha < 0.01) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    // Un-premultiply all channels
    let inv_alpha = 1.0 / max(alpha, 1e-6);
    let plaster = color_raw.rgb * inv_alpha;
    let roughness = aux0_raw.r * inv_alpha;
    let env_feat = vec4f(aux0_raw.gba * inv_alpha, normals_raw.a * inv_alpha);
    let normal = normalize(normals_raw.rgb * inv_alpha);
    let depth = depth_raw.r * inv_alpha;
    let raw_albedo = depth_raw.gba * inv_alpha;

    // Recover true albedo by dividing out plaster (env lighting)
    let plaster_lum = max(0.2126 * plaster.r + 0.7152 * plaster.g + 0.0722 * plaster.b, 1e-3);
    let albedo = raw_albedo / plaster_lum;

    // Reconstruct world position from depth + inverse VP
    let ndc_x = (frag_coord.x + 0.5) / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (frag_coord.y + 0.5) / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, depth, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    // Accumulate lighting
    var total_contribution = vec3f(0.0);

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
        let l_color = light.color * light.intensity;

        // Phase A: pure Lambertian, no shadow, no specular
        total_contribution += albedo * NdotL * atten * l_color;
    }

    var final_color = uniforms.ambient * albedo + total_contribution;

    // Debug mode overrides
    let dm = uniforms.debug_mode;
    if (dm == DBG_RAW_ALBEDO)       { final_color = raw_albedo; }
    else if (dm == DBG_TRUE_ALBEDO) { final_color = albedo; }
    else if (dm == DBG_NORMALS)     { final_color = normal * 0.5 + 0.5; }
    else if (dm == DBG_ROUGHNESS)   { final_color = vec3f(roughness); }
    else if (dm == DBG_ENV_FEATURE) { final_color = env_feat.xyz * 0.5 + 0.5; }
    else if (dm == DBG_DEPTH)       { final_color = vec3f(depth * 0.1); }
    else if (dm == DBG_PLASTER)     { final_color = plaster; }
    else if (dm == DBG_ALPHA)       { final_color = vec3f(alpha); }

    // Output linear — blit handles sRGB conversion
    return vec4f(max(final_color, vec3f(0.0)), alpha);
}
