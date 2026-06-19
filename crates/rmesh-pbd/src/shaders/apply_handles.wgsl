// Apply kinematic handle positions before predict.
//
// One thread per handle: writes the current handle world position into the
// corresponding particle.position.xyz. Inverse-mass (stored in position.w)
// is preserved so the predict step keeps treating the handle as fixed.

struct Particle {
    position:  vec4<f32>, // xyz, w = inverse_mass
    predicted: vec4<f32>, // xyz, w = pad
    velocity:  vec4<f32>, // xyz, w = pad
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read>       handle_local_indices: array<u32>;
@group(0) @binding(2) var<storage, read>       handle_positions: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&handle_local_indices) { return; }
    let li = handle_local_indices[i];
    var p = particles[li];
    p.position = vec4<f32>(handle_positions[i].xyz, p.position.w);
    particles[li] = p;
}
