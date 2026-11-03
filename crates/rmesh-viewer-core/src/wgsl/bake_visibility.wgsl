// Pack BVH nodes and face IDs in one storage buffer to keep the baker at
// WebGPU's eight-storage-buffer limit. Header: node count plus three pads.
struct Node { lo:vec3f,left:i32,hi:vec3f,right:i32 }
@group(0) @binding(10) var<storage,read> shadow_geometry:array<u32>;
fn shadow_node(i:u32)->Node {
    let b=4u+i*8u;
    return Node(vec3f(bitcast<f32>(shadow_geometry[b]),bitcast<f32>(shadow_geometry[b+1u]),bitcast<f32>(shadow_geometry[b+2u])),bitcast<i32>(shadow_geometry[b+3u]),
                vec3f(bitcast<f32>(shadow_geometry[b+4u]),bitcast<f32>(shadow_geometry[b+5u]),bitcast<f32>(shadow_geometry[b+6u])),bitcast<i32>(shadow_geometry[b+7u]));
}
fn shadow_face(i:u32)->u32 { return shadow_geometry[4u+shadow_geometry[0]*8u+i]; }
fn shadow_epsilon()->f32 { return bitcast<f32>(params.control.x); }
fn shadow_limit()->u32 { return params.control.y; }
fn shadow_cutoff()->f32 { return 12.0; }
fn shadow_boundary_only()->bool { return true; }
