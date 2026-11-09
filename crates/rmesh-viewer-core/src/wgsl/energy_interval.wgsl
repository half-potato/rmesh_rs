// Material-free, per-light vertex visibility is baked; all material and
// inverse-square/spotlight factors are evaluated at the actual camera sample.
struct Lamp { position:vec4f, color:vec4f, direction:vec4f }
struct BakeSnapshot { offset:u32,count:u32,lights:u32,tet_count:u32,control:vec4u,lamps:array<Lamp,16> }
struct CellMaterial { albedo:vec3f,roughness:f32,f0:vec3f,retro:f32 }
@group(1) @binding(0) var<storage,read> positions:array<f32>;
@group(1) @binding(1) var<storage,read> materials:array<CellMaterial>;
@group(1) @binding(2) var<storage,read> transport:array<f32>;
@group(1) @binding(3) var<uniform> snapshot:BakeSnapshot;
@group(1) @binding(4) var energy_lut:texture_2d<f32>;
fn point(i:u32)->vec3f {return vec3f(positions[3u*i],positions[3u*i+1u],positions[3u*i+2u]);}
fn geom_normal(i:u32)->vec3f {return vec3f(vertex_normals[3u*i],vertex_normals[3u*i+1u],vertex_normals[3u*i+2u]);}
const ENERGY_SAMPLES:u32=16u;
@fragment
fn main_color_only(@builtin(position) pixel:vec4f,in:FragmentInput)->@location(0) vec4f {
    let sc=shade_common(pixel,in,false);
    if(sc.alpha<=0.0 || snapshot.lights==0u){return vec4f(0.0,0.0,0.0,sc.alpha)*in.coverage;}
    let camera_ray=vec3f((pixel.x-uniforms.intrinsics.z)/uniforms.intrinsics.x,
                        (pixel.y-uniforms.intrinsics.w)/uniforms.intrinsics.y,1.0);
    let ray=uniforms.c2w_col0.xyz*camera_ray.x+uniforms.c2w_col1.xyz*camera_ray.y+uniforms.c2w_col2.xyz;
    let ids=vec4u(tet_indices[4u*in.tet_id],tet_indices[4u*in.tet_id+1u],tet_indices[4u*in.tet_id+2u],tet_indices[4u*in.tet_id+3u]);
    let n0=geom_normal(ids.x);let n1=geom_normal(ids.y);let n2=geom_normal(ids.z);let n3=geom_normal(ids.w);
    let view=-energy_unit(ray);
    let a=point(ids.x);let b=point(ids.y)-a;let c=point(ids.z)-a;let d=point(ids.w)-a;
    let determinant=dot(b,cross(c,d));
    if(abs(determinant)<1e-30){return vec4f(0.0);}
    // Both canonical GGX lobes (including retro) vanish for N.V <= 0.
    // Barycentric weights are nonnegative, so this vertex bound proves the
    // entire interval is dark without changing its absorption contribution.
    // The margin also guarantees normalization is outside its epsilon branch.
    if(all(vec4f(dot(-n0,view),dot(-n1,view),dot(-n2,view),dot(-n3,view))<=vec4f(-1e-8))){
        return vec4f(0.0,0.0,0.0,sc.alpha)*in.coverage;
    }
    let mat=materials[in.tet_id];
    var result=vec3f(0.0);
    // Keep the canonical absorption-coordinate quadrature unchanged. Put
    // lights outside the sample loop so each vertex cache value is fetched
    // once per fragment/light, rather than once per quadrature node.
    let count=max(1u,u32(ceil(sc.alpha*f32(ENERGY_SAMPLES))));
    let bc=cross(c,d);let ca=cross(d,b);let ab=cross(b,c);
    for(var li=0u;li<snapshot.lights;li++){
        let lamp=snapshot.lamps[li];
        let vertex_visibility=vec4f(transport[16u*ids.x+li],transport[16u*ids.y+li],transport[16u*ids.z+li],transport[16u*ids.w+li]);
        // Exactly dark light/transport contributions need no material work.
        if(all(vertex_visibility==vec4f(0.0)) || lamp.color.w==0.0){continue;}
        let illumination=lamp.color.rgb*lamp.color.w;
        for(var k=0u;k<count;k++){
            let u=(f32(k)+0.5)/f32(count);
            let fraction=select(u,-log(max(1.0-u*sc.alpha,1e-30))/max(sc.od,1e-30),sc.od>1e-5);
            let p=uniforms.cam_pos_pad.xyz+ray*mix(sc.z_f,sc.z_b,fraction);
            let relative=p-a;
            let xyz=vec3f(dot(relative,bc),dot(relative,ca),dot(relative,ab))/determinant;
            var weights=max(vec4f(1.0-xyz.x-xyz.y-xyz.z,xyz),vec4f(0.0));
            weights/=max(dot(weights,vec4f(1.0)),1e-30);
            let normal=-energy_unit(weights.x*n0+weights.y*n1+weights.z*n2+weights.w*n3);
            let delta=lamp.position.xyz-p;
            let distance2=max(dot(delta,delta),1e-6);let light=delta/max(length(delta),1e-12);
            // Every canonical lobe includes max(N.L,0), including retro.
            if(dot(normal,light)<=0.0){continue;}
            var emission=1.0;
            if(lamp.position.w>0.5){emission=smoothstep(lamp.direction.w,min(1.0,lamp.direction.w+0.1),dot(-light,lamp.direction.xyz));}
            if(emission==0.0){continue;}
            let visibility=dot(weights,vertex_visibility);
            result+=energy_source(normal,light,view,mat.roughness,mat.f0,mat.albedo,mat.retro)*illumination*(visibility*emission/distance2);
        }
    }
    return vec4f(result*(sc.alpha/f32(count)),sc.alpha)*in.coverage;
}
