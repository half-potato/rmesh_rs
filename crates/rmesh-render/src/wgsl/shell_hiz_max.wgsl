// Max-reduce Hi-Z over the shell/opaque depth buffer.
//
// Deliberately a near-twin of rmesh-postprocess's hiz_downsample.wgsl, with two
// differences that matter:
//
//   * it reduces with max(), not min(). A conservative occlusion test needs the
//     *farthest* bound over a tet's screen footprint: cull only when the tet's
//     nearest point is past every exit plane it could cover.
//   * it stays in raw NDC depth. No linearization, so nothing here needs the
//     near/far planes, and an untouched texel keeps the cleared 1.0 — which
//     max()es to 1.0 and culls nothing. Background is safe by construction.
//
// It can't live in rmesh-postprocess: that crate's Hi-Z is built after the
// volume pass, and rmesh-render sits below it in the dependency order. The
// duplication is intentional — see the note on hiz_downsample.wgsl.

@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var src_color: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

// Mip 0: copy the hardware depth buffer into R32Float.
@fragment
fn fs_init(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let dst = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let dim = textureDimensions(src_depth);
    let clamped = min(dst, dim - vec2u(1u, 1u));
    return vec4f(textureLoad(src_depth, clamped, 0), 0.0, 0.0, 0.0);
}

// Mips 1..n: 2x2 max of the parent. Out-of-bounds reads (odd parent extents)
// clamp to the edge rather than pulling in zeros, which would make the bound
// *nearer* than reality and over-cull.
@fragment
fn fs_downsample(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let dst = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let src_dim = textureDimensions(src_color);
    let src = dst * 2u;
    let max_xy = src_dim - vec2u(1u, 1u);
    let z00 = textureLoad(src_color, min(src + vec2u(0u, 0u), max_xy), 0).r;
    let z10 = textureLoad(src_color, min(src + vec2u(1u, 0u), max_xy), 0).r;
    let z01 = textureLoad(src_color, min(src + vec2u(0u, 1u), max_xy), 0).r;
    let z11 = textureLoad(src_color, min(src + vec2u(1u, 1u), max_xy), 0).r;
    return vec4f(max(max(z00, z10), max(z01, z11)), 0.0, 0.0, 0.0);
}
