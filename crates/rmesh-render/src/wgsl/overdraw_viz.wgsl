// Overdraw heatmap.
//
// Reads the Rg16Float probe target written by `interval_fragment::fs_overdraw`
// (.r = fragments shaded at this pixel, .g = sum of their individual alphas)
// together with the G-buffer's blended alpha, and renders one of three views.
//
// The third is the interesting one. Back-to-front hardware blending cannot stop
// early: every fragment on a ray is blended whether or not the result is already
// opaque. From the mean per-fragment alpha a = sum(alpha)/count, a front-to-back
// walk would reach T < 0.01 after n = ln(0.01)/ln(1-a) fragments, so everything
// past n is work an early-out would have skipped.
//
// Note this is NOT 1 - alpha_final/sum(alpha): opacity composes multiplicatively,
// so that ratio counts a ray with 100 fragments of alpha 0.05 as almost entirely
// wasted when in fact it needs ~90 of them. Getting that wrong overstates the
// available win by about 2x.

@group(0) @binding(0) var probe: texture_2d<f32>;
@group(0) @binding(1) var gbuf_color: texture_2d<f32>;

struct VizUniforms {
    /// Fragment count that maps to the top of the colour ramp.
    scale: f32,
    /// 0 = count heatmap, 1 = final alpha, 2 = fraction of fragments a
    /// front-to-back early-out would skip.
    mode: u32,
    _pad: vec2<u32>,
};
@group(0) @binding(2) var<uniform> viz: VizUniforms;

struct VsOut { @builtin(position) pos: vec4f }

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4f(x, y, 0.0, 1.0);
    return out;
}

// Google Turbo, polynomial fit. Perceptually ordered, so a reader can rank two
// regions by eye — greyscale cannot be ranked and the usual blue→red ramp has a
// false boundary in the middle.
fn turbo(t_in: f32) -> vec3f {
    let t = clamp(t_in, 0.0, 1.0);
    let r = 0.1357 + t * (4.5974 + t * (-42.3277 + t * (130.5887 + t * (-150.5666 + t * 58.1375))));
    let g = 0.0914 + t * (2.1856 + t * (4.8052 + t * (-14.0195 + t * (4.2109 + t * 2.7747))));
    let b = 0.1067 + t * (12.5925 + t * (-60.1097 + t * (109.0745 + t * (-88.5066 + t * 26.8183))));
    return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let px = vec2u(u32(frag_coord.x), u32(frag_coord.y));
    let p = textureLoad(probe, px, 0);
    let count = p.r;
    let alpha_sum = p.g;
    let alpha_final = textureLoad(gbuf_color, px, 0).a;

    if (viz.mode == 1u) {
        return vec4f(turbo(alpha_final), 1.0);
    }
    if (viz.mode == 2u) {
        // Fraction of shaded fragments a front-to-back early-out would skip.
        if (count < 1.0) {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        let a = clamp(alpha_sum / count, 1e-6, 0.999999);
        let needed = log(0.01) / log(1.0 - a);
        return vec4f(turbo(1.0 - min(needed, count) / count), 1.0);
    }
    return vec4f(turbo(count / max(viz.scale, 1.0)), 1.0);
}
