// Fullscreen triangle blit: samples an Rgba16Float texture and outputs to the swapchain.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle: 3 vertices cover clip space [-1,1]^2.
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // Map clip [-1,1] to UV [0,1], flip Y so top-left is (0,0).
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// Linear → sRGB transfer function (exact per sRGB spec)
fn linear_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(src_tex, src_sampler, in.uv);
    // Apply linear → sRGB gamma and set alpha=1 for opaque swapchain.
    return vec4<f32>(
        linear_to_srgb(color.r),
        linear_to_srgb(color.g),
        linear_to_srgb(color.b),
        1.0,
    );
}
