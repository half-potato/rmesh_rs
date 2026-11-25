// Multi-mip compute max-reduce Hi-Z over the shell/opaque depth buffer.
//
// Writes up to 4 mip levels per compute pass (staying within WebGPU's default
// 4-storage-texture-per-stage cap) using shared-memory reductions instead of
// one render pass per mip.
//
// Reduction is conservative `max()` in raw NDC depth: untouched texels stay 1.0
// (far plane) and max() to 1.0 so background never over-culls. Odd parent
// extents include the trailing row/column in the last child texel.

struct Params {
    src_w: u32,
    src_h: u32,
    levels: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src_depth: texture_depth_2d;
@group(0) @binding(2) var out0: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var out1: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var out2: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var out3: texture_storage_2d<r32float, write>;

@group(0) @binding(6) var src_color: texture_2d<f32>;

var<workgroup> smem0: array<f32, 64>;
var<workgroup> smem1: array<f32, 16>;
var<workgroup> smem2: array<f32, 4>;

// Pass 1: copy src_depth -> mip 0 and reduce up to 3 additional mips in shared memory.
// Caller only passes levels >= 2 when every fused parent extent is even.
@compute @workgroup_size(8, 8)
fn cs_init(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let w0 = params.src_w;
    let h0 = params.src_h;
    let max0 = vec2u(w0 - 1u, h0 - 1u);
    let base = vec2u(wid.x * 16u + lid.x * 2u, wid.y * 16u + lid.y * 2u);

    let z00 = textureLoad(src_depth, min(base + vec2u(0u, 0u), max0), 0);
    let z10 = textureLoad(src_depth, min(base + vec2u(1u, 0u), max0), 0);
    let z01 = textureLoad(src_depth, min(base + vec2u(0u, 1u), max0), 0);
    let z11 = textureLoad(src_depth, min(base + vec2u(1u, 1u), max0), 0);

    if base.x < w0 && base.y < h0 {
        textureStore(out0, vec2i(base), vec4f(z00, 0.0, 0.0, 0.0));
    }
    if base.x + 1u < w0 && base.y < h0 {
        textureStore(out0, vec2i(base + vec2u(1u, 0u)), vec4f(z10, 0.0, 0.0, 0.0));
    }
    if base.x < w0 && base.y + 1u < h0 {
        textureStore(out0, vec2i(base + vec2u(0u, 1u)), vec4f(z01, 0.0, 0.0, 0.0));
    }
    if base.x + 1u < w0 && base.y + 1u < h0 {
        textureStore(out0, vec2i(base + vec2u(1u, 1u)), vec4f(z11, 0.0, 0.0, 0.0));
    }

    if params.levels < 2u {
        return;
    }

    let m1 = max(max(z00, z10), max(z01, z11));
    let w1 = max(w0 >> 1u, 1u);
    let h1 = max(h0 >> 1u, 1u);
    let p1 = vec2u(wid.x * 8u + lid.x, wid.y * 8u + lid.y);
    if p1.x < w1 && p1.y < h1 {
        textureStore(out1, vec2i(p1), vec4f(m1, 0.0, 0.0, 0.0));
    }
    smem0[lid.y * 8u + lid.x] = m1;
    workgroupBarrier();

    if params.levels < 3u {
        return;
    }

    if (lid.x & 1u) == 0u && (lid.y & 1u) == 0u {
        let i0 = lid.y * 8u + lid.x;
        let i1 = i0 + 8u;
        let m2 = max(max(smem0[i0], smem0[i0 + 1u]), max(smem0[i1], smem0[i1 + 1u]));
        let w2 = max(w1 >> 1u, 1u);
        let h2 = max(h1 >> 1u, 1u);
        let p2 = vec2u(wid.x * 4u + (lid.x >> 1u), wid.y * 4u + (lid.y >> 1u));
        if p2.x < w2 && p2.y < h2 {
            textureStore(out2, vec2i(p2), vec4f(m2, 0.0, 0.0, 0.0));
        }
        smem1[(lid.y >> 1u) * 4u + (lid.x >> 1u)] = m2;
    }
    workgroupBarrier();

    if params.levels < 4u {
        return;
    }

    if lid.x < 2u && lid.y < 2u {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        let m3 = max(
            max(smem1[sy * 4u + sx], smem1[sy * 4u + sx + 1u]),
            max(smem1[(sy + 1u) * 4u + sx], smem1[(sy + 1u) * 4u + sx + 1u]),
        );
        let w3 = max(w1 >> 2u, 1u);
        let h3 = max(h1 >> 2u, 1u);
        let p3 = vec2u(wid.x * 2u + lid.x, wid.y * 2u + lid.y);
        if p3.x < w3 && p3.y < h3 {
            textureStore(out3, vec2i(p3), vec4f(m3, 0.0, 0.0, 0.0));
        }
    }
}

fn load_src_conservative(p: vec2u, dst_dim: vec2u, src_dim: vec2u) -> f32 {
    let max_xy = src_dim - vec2u(1u, 1u);
    let s = p * 2u;
    var m = max(
        max(
            textureLoad(src_color, min(s + vec2u(0u, 0u), max_xy), 0).r,
            textureLoad(src_color, min(s + vec2u(1u, 0u), max_xy), 0).r,
        ),
        max(
            textureLoad(src_color, min(s + vec2u(0u, 1u), max_xy), 0).r,
            textureLoad(src_color, min(s + vec2u(1u, 1u), max_xy), 0).r,
        ),
    );
    let extra_x = (p.x + 1u == dst_dim.x) && ((src_dim.x & 1u) == 1u) && (src_dim.x > 1u);
    let extra_y = (p.y + 1u == dst_dim.y) && ((src_dim.y & 1u) == 1u) && (src_dim.y > 1u);
    if extra_x {
        m = max(m, textureLoad(src_color, vec2u(max_xy.x, min(s.y, max_xy.y)), 0).r);
        m = max(m, textureLoad(src_color, vec2u(max_xy.x, min(s.y + 1u, max_xy.y)), 0).r);
    }
    if extra_y {
        m = max(m, textureLoad(src_color, vec2u(min(s.x, max_xy.x), max_xy.y), 0).r);
        m = max(m, textureLoad(src_color, vec2u(min(s.x + 1u, max_xy.x), max_xy.y), 0).r);
    }
    if extra_x && extra_y {
        m = max(m, textureLoad(src_color, max_xy, 0).r);
    }
    return m;
}

// Grid downsample: reduces src_color into 1..=4 output mips.
// levels >= 2 is only used when every fused parent extent is even.
@compute @workgroup_size(8, 8)
fn cs_downsample(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let src_dim = vec2u(params.src_w, params.src_h);
    let d0 = vec2u(max(src_dim.x >> 1u, 1u), max(src_dim.y >> 1u, 1u));
    let p0 = vec2u(wid.x * 8u + lid.x, wid.y * 8u + lid.y);

    var m0 = 1.0;
    if p0.x < d0.x && p0.y < d0.y {
        m0 = load_src_conservative(p0, d0, src_dim);
        textureStore(out0, vec2i(p0), vec4f(m0, 0.0, 0.0, 0.0));
    }

    if params.levels < 2u {
        return;
    }

    smem0[lid.y * 8u + lid.x] = m0;
    workgroupBarrier();

    if (lid.x & 1u) == 0u && (lid.y & 1u) == 0u {
        let i0 = lid.y * 8u + lid.x;
        let i1 = i0 + 8u;
        let m1 = max(max(smem0[i0], smem0[i0 + 1u]), max(smem0[i1], smem0[i1 + 1u]));
        let d1 = vec2u(max(d0.x >> 1u, 1u), max(d0.y >> 1u, 1u));
        let p1 = vec2u(wid.x * 4u + (lid.x >> 1u), wid.y * 4u + (lid.y >> 1u));
        if p1.x < d1.x && p1.y < d1.y {
            textureStore(out1, vec2i(p1), vec4f(m1, 0.0, 0.0, 0.0));
        }
        smem1[(lid.y >> 1u) * 4u + (lid.x >> 1u)] = m1;
    }
    workgroupBarrier();

    if params.levels < 3u {
        return;
    }

    if lid.x < 2u && lid.y < 2u {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        let m2 = max(
            max(smem1[sy * 4u + sx], smem1[sy * 4u + sx + 1u]),
            max(smem1[(sy + 1u) * 4u + sx], smem1[(sy + 1u) * 4u + sx + 1u]),
        );
        let d2 = vec2u(max(d0.x >> 2u, 1u), max(d0.y >> 2u, 1u));
        let p2 = vec2u(wid.x * 2u + lid.x, wid.y * 2u + lid.y);
        if p2.x < d2.x && p2.y < d2.y {
            textureStore(out2, vec2i(p2), vec4f(m2, 0.0, 0.0, 0.0));
        }
        smem2[lid.y * 2u + lid.x] = m2;
    }
    workgroupBarrier();

    if params.levels < 4u {
        return;
    }

    if lid.x == 0u && lid.y == 0u {
        let m3 = max(max(smem2[0], smem2[1]), max(smem2[2], smem2[3]));
        let d3 = vec2u(max(d0.x >> 3u, 1u), max(d0.y >> 3u, 1u));
        let p3 = vec2u(wid.x, wid.y);
        if p3.x < d3.x && p3.y < d3.y {
            textureStore(out3, vec2i(p3), vec4f(m3, 0.0, 0.0, 0.0));
        }
    }
}

// Single-workgroup tail downsample for small mips (dst0_w * dst0_h <= 2048).
// Reduces up to 4 levels in one workgroup with full conservative odd-border max.
var<workgroup> tail_a: array<f32, 2048>;
var<workgroup> tail_b: array<f32, 1024>;

fn tail_load_a(x: u32, y: u32, w: u32, h: u32) -> f32 {
    return tail_a[min(y, h - 1u) * w + min(x, w - 1u)];
}
fn tail_load_b(x: u32, y: u32, w: u32, h: u32) -> f32 {
    return tail_b[min(y, h - 1u) * w + min(x, w - 1u)];
}

@compute @workgroup_size(8, 8)
fn cs_tail(@builtin(local_invocation_index) tid: u32) {
    let src_dim = vec2u(params.src_w, params.src_h);
    let d0 = vec2u(max(src_dim.x >> 1u, 1u), max(src_dim.y >> 1u, 1u));
    let n0 = d0.x * d0.y;
    for (var i = tid; i < n0; i += 64u) {
        let p = vec2u(i % d0.x, i / d0.x);
        let m = load_src_conservative(p, d0, src_dim);
        tail_a[i] = m;
        textureStore(out0, vec2i(p), vec4f(m, 0.0, 0.0, 0.0));
    }
    workgroupBarrier();

    if params.levels < 2u {
        return;
    }

    let d1 = vec2u(max(d0.x >> 1u, 1u), max(d0.y >> 1u, 1u));
    let n1 = d1.x * d1.y;
    for (var i = tid; i < n1; i += 64u) {
        let p = vec2u(i % d1.x, i / d1.x);
        let sx = p.x * 2u;
        let sy = p.y * 2u;
        var m = max(
            max(tail_load_a(sx, sy, d0.x, d0.y), tail_load_a(sx + 1u, sy, d0.x, d0.y)),
            max(tail_load_a(sx, sy + 1u, d0.x, d0.y), tail_load_a(sx + 1u, sy + 1u, d0.x, d0.y)),
        );
        let ex = (p.x + 1u == d1.x) && ((d0.x & 1u) == 1u) && (d0.x > 1u);
        let ey = (p.y + 1u == d1.y) && ((d0.y & 1u) == 1u) && (d0.y > 1u);
        if ex {
            m = max(m, max(tail_load_a(d0.x - 1u, sy, d0.x, d0.y), tail_load_a(d0.x - 1u, sy + 1u, d0.x, d0.y)));
        }
        if ey {
            m = max(m, max(tail_load_a(sx, d0.y - 1u, d0.x, d0.y), tail_load_a(sx + 1u, d0.y - 1u, d0.x, d0.y)));
        }
        if ex && ey {
            m = max(m, tail_load_a(d0.x - 1u, d0.y - 1u, d0.x, d0.y));
        }
        tail_b[i] = m;
        textureStore(out1, vec2i(p), vec4f(m, 0.0, 0.0, 0.0));
    }
    workgroupBarrier();

    if params.levels < 3u {
        return;
    }

    let d2 = vec2u(max(d1.x >> 1u, 1u), max(d1.y >> 1u, 1u));
    let n2 = d2.x * d2.y;
    for (var i = tid; i < n2; i += 64u) {
        let p = vec2u(i % d2.x, i / d2.x);
        let sx = p.x * 2u;
        let sy = p.y * 2u;
        var m = max(
            max(tail_load_b(sx, sy, d1.x, d1.y), tail_load_b(sx + 1u, sy, d1.x, d1.y)),
            max(tail_load_b(sx, sy + 1u, d1.x, d1.y), tail_load_b(sx + 1u, sy + 1u, d1.x, d1.y)),
        );
        let ex = (p.x + 1u == d2.x) && ((d1.x & 1u) == 1u) && (d1.x > 1u);
        let ey = (p.y + 1u == d2.y) && ((d1.y & 1u) == 1u) && (d1.y > 1u);
        if ex {
            m = max(m, max(tail_load_b(d1.x - 1u, sy, d1.x, d1.y), tail_load_b(d1.x - 1u, sy + 1u, d1.x, d1.y)));
        }
        if ey {
            m = max(m, max(tail_load_b(sx, d1.y - 1u, d1.x, d1.y), tail_load_b(sx + 1u, d1.y - 1u, d1.x, d1.y)));
        }
        if ex && ey {
            m = max(m, tail_load_b(d1.x - 1u, d1.y - 1u, d1.x, d1.y));
        }
        tail_a[i] = m;
        textureStore(out2, vec2i(p), vec4f(m, 0.0, 0.0, 0.0));
    }
    workgroupBarrier();

    if params.levels < 4u {
        return;
    }

    let d3 = vec2u(max(d2.x >> 1u, 1u), max(d2.y >> 1u, 1u));
    let n3 = d3.x * d3.y;
    for (var i = tid; i < n3; i += 64u) {
        let p = vec2u(i % d3.x, i / d3.x);
        let sx = p.x * 2u;
        let sy = p.y * 2u;
        var m = max(
            max(tail_load_a(sx, sy, d2.x, d2.y), tail_load_a(sx + 1u, sy, d2.x, d2.y)),
            max(tail_load_a(sx, sy + 1u, d2.x, d2.y), tail_load_a(sx + 1u, sy + 1u, d2.x, d2.y)),
        );
        let ex = (p.x + 1u == d3.x) && ((d2.x & 1u) == 1u) && (d2.x > 1u);
        let ey = (p.y + 1u == d3.y) && ((d2.y & 1u) == 1u) && (d2.y > 1u);
        if ex {
            m = max(m, max(tail_load_a(d2.x - 1u, sy, d2.x, d2.y), tail_load_a(d2.x - 1u, sy + 1u, d2.x, d2.y)));
        }
        if ey {
            m = max(m, max(tail_load_a(sx, d2.y - 1u, d2.x, d2.y), tail_load_a(sx + 1u, d2.y - 1u, d2.x, d2.y)));
        }
        if ex && ey {
            m = max(m, tail_load_a(d2.x - 1u, d2.y - 1u, d2.x, d2.y));
        }
        textureStore(out3, vec2i(p), vec4f(m, 0.0, 0.0, 0.0));
    }
}
