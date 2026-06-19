// Recover velocity from (predicted - position) / dt, commit predicted -> position,
// and scatter the final position back into the global vertex buffer.

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
@group(0) @binding(1) var<storage, read>       global_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(3) var<uniform>             u: PbdUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&particles) { return; }
    var p = particles[i];
    let inv_m = p.position.w;
    if inv_m > 0.0 {
        let new_vel = (p.predicted.xyz - p.position.xyz) / u.dt;
        p.velocity = vec4<f32>(new_vel, 0.0);
        p.position = vec4<f32>(p.predicted.xyz, inv_m);
        particles[i] = p;
    }
    let gi = global_indices[i];
    vertices[gi * 3u + 0u] = p.position.x;
    vertices[gi * 3u + 1u] = p.position.y;
    vertices[gi * 3u + 2u] = p.position.z;
}
