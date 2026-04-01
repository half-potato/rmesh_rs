// Shadow ray generation compute shader.
//
// For each visible pixel × each active light, generates a shadow ray
// (origin biased along surface normal, direction toward light).
// These rays are then traced by raytrace_compute.wgsl for transmittance.

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
@group(0) @binding(1) var<storage, read> xyzd_image: array<f32>;           // [W*H*4] normal.xyz + depth
@group(0) @binding(2) var<storage, read> rendered_image: array<f32>;       // [W*H*4] plaster RGBA (alpha in .a)
@group(0) @binding(3) var<storage, read> lights: array<Light>;
@group(0) @binding(4) var<storage, read_write> shadow_ray_origins: array<f32>;  // [num_lights*W*H*3]
@group(0) @binding(5) var<storage, read_write> shadow_ray_dirs: array<f32>;     // [num_lights*W*H*3]

const SHADOW_BIAS: f32 = 2e-3;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let x = gid.x;
    let y = gid.y;
    if (x >= uniforms.width || y >= uniforms.height) { return; }

    let pixel_idx = y * uniforms.width + x;

    // Alpha is stored in log-space: alpha = 1 - exp(log_alpha)
    let log_alpha = rendered_image[pixel_idx * 4u + 3u];
    let alpha = 1.0 - exp(log_alpha);
    if (alpha < 0.01) { return; }

    // Read normal from xyzd_image
    let nx = xyzd_image[pixel_idx * 4u + 0u];
    let ny = xyzd_image[pixel_idx * 4u + 1u];
    let nz = xyzd_image[pixel_idx * 4u + 2u];
    let normal = normalize(vec3f(nx, ny, nz));

    // Reconstruct world position from depth + inverse VP
    let depth = xyzd_image[pixel_idx * 4u + 3u];
    let ndc_x = (f32(x) + 0.5) / f32(uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(y) + 0.5) / f32(uniforms.height) * 2.0;
    let clip_pos = vec4f(ndc_x, ndc_y, depth, 1.0);
    let world_h = uniforms.inv_vp * clip_pos;
    let world_pos = world_h.xyz / world_h.w;

    let biased_pos = world_pos + SHADOW_BIAS * normal;
    let wh = uniforms.width * uniforms.height;

    for (var li = 0u; li < uniforms.num_lights; li++) {
        let light = lights[li];
        var ray_dir: vec3f;

        if (light.light_type == 2u) {  // directional
            ray_dir = normalize(-light.direction);
        } else {  // point or spot
            ray_dir = normalize(light.position - world_pos);
        }

        let ray_idx = li * wh + pixel_idx;
        shadow_ray_origins[ray_idx * 3u + 0u] = biased_pos.x;
        shadow_ray_origins[ray_idx * 3u + 1u] = biased_pos.y;
        shadow_ray_origins[ray_idx * 3u + 2u] = biased_pos.z;
        shadow_ray_dirs[ray_idx * 3u + 0u] = ray_dir.x;
        shadow_ray_dirs[ray_idx * 3u + 1u] = ray_dir.y;
        shadow_ray_dirs[ray_idx * 3u + 2u] = ray_dir.z;
    }
}
