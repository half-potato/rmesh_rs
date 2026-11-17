// Integrate complete cell-corner radiance using one consistent cell material.
// One HDR target; no G-buffer or shell.
@group(1) @binding(0) var<storage,read> positions:array<f32>;
@group(1) @binding(1) var<storage,read> corner_radiance:array<vec2u>;
struct Frame {eye:vec4f,probe:vec4f,counts:vec4u,components:vec4f, probes:array<vec4f,5>, influence:vec4f}
@group(1) @binding(3) var<uniform> frame:Frame;
fn corner(i:u32)->vec3f {let packed=corner_radiance[i];return vec3f(unpack2x16float(packed.x),unpack2x16float(packed.y).x);}
fn point(i:u32)->vec3f {return vec3f(positions[3u*i],positions[3u*i+1u],positions[3u*i+2u]);}
@fragment
fn main_color_only(@builtin(position) pixel:vec4f,in:FragmentInput)->@location(0) vec4f {
    let sc=shade_common(pixel,in,false);
    if(sc.alpha<=0.0){return vec4f(0.0);}
    let camera_ray=vec3f((pixel.x-uniforms.intrinsics.z)/uniforms.intrinsics.x,(pixel.y-uniforms.intrinsics.w)/uniforms.intrinsics.y,1.0);
    let ray=uniforms.c2w_col0.xyz*camera_ray.x+uniforms.c2w_col1.xyz*camera_ray.y+uniforms.c2w_col2.xyz;
    let ids=vec4u(tet_indices[4u*in.tet_id],tet_indices[4u*in.tet_id+1u],tet_indices[4u*in.tet_id+2u],tet_indices[4u*in.tet_id+3u]);
    let a=point(ids.x);let b=point(ids.y)-a;let c=point(ids.z)-a;let d=point(ids.w)-a;
    let det=dot(b,cross(c,d));if(abs(det)<1e-30){return vec4f(0.0);}
    // Barycentric coordinates are affine along a ray: evaluate at the exact
    // absorption-weighted mean position instead of a quadrature loop.
    let z=(sc.w0*sc.z_b+sc.w1*sc.z_f)/sc.alpha;
    let p=uniforms.cam_pos_pad.xyz+ray*z-a;
    let xyz=vec3f(dot(p,cross(c,d)),dot(p,cross(d,b)),dot(p,cross(b,c)))/det;
    var w=max(vec4f(1.0-xyz.x-xyz.y-xyz.z,xyz),vec4f(0.0));w/=max(dot(w,vec4f(1.0)),1e-30);
    let base=4u*in.tet_id;
    let rgb=w.x*corner(base)+w.y*corner(base+1u)+w.z*corner(base+2u)+w.w*corner(base+3u);
    return vec4f(max(rgb,vec3f(0.0))*sc.alpha,sc.alpha)*in.coverage;
}
