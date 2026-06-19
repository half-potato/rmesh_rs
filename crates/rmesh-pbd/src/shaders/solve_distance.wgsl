// XPBD distance constraint solve, one dispatch per graph-coloring batch.
//
// Bind group binds a *slice* of the global constraints buffer corresponding to
// the current color (via wgpu BufferBinding offset+size). All threads in a
// dispatch hold constraints with vertex-disjoint endpoints — safe to write
// p.predicted with ordinary stores.

struct Particle {
    position:  vec4<f32>,
    predicted: vec4<f32>,
    velocity:  vec4<f32>,
};

struct DistanceConstraint {
    p1: u32,
    p2: u32,
    rest_length: f32,
    alpha: f32,
};

struct PbdUniforms {
    dt: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read>       constraints: array<DistanceConstraint>;
@group(0) @binding(2) var<uniform>             u: PbdUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&constraints) { return; }
    let c = constraints[i];

    let p1 = particles[c.p1];
    let p2 = particles[c.p2];

    let inv_m1 = p1.position.w;
    let inv_m2 = p2.position.w;
    let w_sum = inv_m1 + inv_m2;
    if w_sum < 1e-6 { return; }

    let delta = p2.predicted.xyz - p1.predicted.xyz;
    let len = length(delta);
    if len < 1e-6 { return; }

    let alpha_tilde = c.alpha / (u.dt * u.dt);
    let err = len - c.rest_length;
    let lambda = -err / (w_sum + alpha_tilde);
    let corr = (delta / len) * lambda;

    particles[c.p1].predicted = vec4<f32>(p1.predicted.xyz - corr * inv_m1, 0.0);
    particles[c.p2].predicted = vec4<f32>(p2.predicted.xyz + corr * inv_m2, 0.0);
}
