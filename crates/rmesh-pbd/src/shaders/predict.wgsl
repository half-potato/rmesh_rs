// Symplectic-Euler predict step. One thread per particle.
//
// Active particles (inverse_mass > 0): predicted = position + velocity * dt.
// Fixed particles (handles, boundary, inverse_mass == 0): predicted = position.
// Gravity is intentionally omitted — matches the C++ reference, where the
// only driver is handle motion.

struct Particle {
    position:  vec4<f32>,
    predicted: vec4<f32>,
    velocity:  vec4<f32>,
};

struct PbdUniforms {
    dt: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform>             u: PbdUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&particles) { return; }
    var p = particles[i];
    let inv_m = p.position.w;
    if inv_m > 0.0 {
        p.predicted = vec4<f32>(p.position.xyz + p.velocity.xyz * u.dt, 0.0);
    } else {
        p.predicted = vec4<f32>(p.position.xyz, 0.0);
    }
    particles[i] = p;
}
