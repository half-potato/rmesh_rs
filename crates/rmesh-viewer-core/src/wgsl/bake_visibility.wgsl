// Pack BVH nodes and face IDs in one storage buffer to keep the baker at
// WebGPU's eight-storage-buffer limit. Header: node count plus three pads.
struct Node { lo:vec3f,left:i32,hi:vec3f,right:i32 }
@group(0) @binding(10) var<storage,read> shadow_geometry:array<vec4u>;
fn shadow_node(i:u32)->Node {
    let b=1u+i*2u;
    let n0=shadow_geometry[b];
    let n1=shadow_geometry[b+1u];
    return Node(vec3f(bitcast<f32>(n0.x),bitcast<f32>(n0.y),bitcast<f32>(n0.z)),bitcast<i32>(n0.w),
                vec3f(bitcast<f32>(n1.x),bitcast<f32>(n1.y),bitcast<f32>(n1.z)),bitcast<i32>(n1.w));
}
fn shadow_face(i:u32)->u32 { return shadow_geometry[1u+shadow_geometry[0].x*2u+(i>>2u)][i&3u]; }
fn shadow_epsilon()->f32 { return bitcast<f32>(params.control.x); }
fn shadow_limit()->u32 { return params.control.y; }
fn shadow_cutoff()->f32 { return 12.0; }
fn shadow_boundary_only()->bool { return true; }
fn active_tet_count()->u32 { return params.tet_count; }
