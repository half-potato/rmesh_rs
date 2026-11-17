// Camera-independent probe capture bake plus complete per-cell-corner shading.
struct Frame { eye:vec4f, probe:vec4f, counts:vec4u, components:vec4f, probes:array<vec4f,5>, influence:vec4f }
struct Lamp { position:vec4f, color:vec4f, direction:vec4f }
struct BakeSnapshot { offset:u32,count:u32,lights:u32,tet_count:u32,control:vec4u,lamps:array<Lamp,16> }
struct Material { albedo:vec3f,roughness:f32,f0:vec3f,retro:f32 }
@group(0) @binding(0) var<uniform> frame:Frame;
@group(0) @binding(1) var<storage,read> positions:array<f32>;
@group(0) @binding(2) var<storage,read> normals:array<f32>;
@group(0) @binding(4) var<storage,read> transport:array<f32>;
@group(0) @binding(5) var<uniform> snapshot:BakeSnapshot;
@group(0) @binding(6) var<storage,read_write> diffuse:array<vec4f>;
@group(0) @binding(7) var<storage,read_write> radiance:array<vec2u>;
@group(0) @binding(8) var energy_lut:texture_2d<f32>;
@group(0) @binding(11) var<storage,read> cell_materials:array<Material>;
@group(0) @binding(12) var<storage,read> indices:array<u32>;
@group(0) @binding(9) var probe_maps:texture_cube_array<f32>;
@group(0) @binding(10) var probe_sampler:sampler;
@group(0) @binding(13) var<storage,read> probe_weights:array<f32>;
fn probe_specular(i:u32,n:vec3f,v:vec3f,m:Material)->vec3f {
    let co=dot(n,v);if(co<=0.0 || frame.components.z<=0.0){return vec3f(0.0);}
    let reflected=reflect(-v,n);
    let level=clamp((m.roughness-0.04)/0.96,0.0,1.0)*f32(frame.counts.z-1u);
    var single_light=vec3f(0.0);var broad_light=vec3f(0.0);
    for(var j=0u;j<5u;j++){
        let w=probe_weights[5u*i+j];
        single_light+=w*textureSampleLevel(probe_maps,probe_sampler,reflected,i32(j),level).rgb;
        broad_light+=w*textureSampleLevel(probe_maps,probe_sampler,n,i32(j),f32(frame.counts.z-1u)).rgb;
    }
    let lut=energy_lookup(m.roughness,co);let allocation=energy_allocation(m.roughness,co,m.f0);
    return (1.0-m.retro)*((m.f0*lut.x+(1.0-m.f0)*lut.y)*single_light+
        allocation.compensation*(1.0-lut.x)*broad_light);
}
fn position(i:u32)->vec3f {return vec3f(positions[3u*i],positions[3u*i+1u],positions[3u*i+2u]);}
fn vertex_normal(i:u32)->vec3f {return -energy_unit(vec3f(normals[3u*i],normals[3u*i+1u],normals[3u*i+2u]));}
fn light_power(i:u32,li:u32,p:vec3f)->vec3f {
    let lamp=snapshot.lamps[li];let delta=lamp.position.xyz-p;
    var emission=1.0;
    if(lamp.position.w>0.5){emission=smoothstep(lamp.direction.w,min(1.0,lamp.direction.w+0.1),dot(-energy_unit(delta),lamp.direction.xyz));}
    return lamp.color.rgb*(lamp.color.w*emission*transport[16u*i+li]/max(dot(delta,delta),1e-6));
}
@compute @workgroup_size(64)
fn bake_diffuse(@builtin(global_invocation_id) id:vec3u){
    let cell=id.x+id.y*65535u*64u;if(cell>=frame.counts.y){return;}
    let m=cell_materials[cell];var value=vec3f(0.0);
    // Probe capture uses each cell's diffuse albedo and energy allocation.
    // Display shading evaluates the complete closure separately at its corners.
    for(var k=0u;k<4u;k++){
        let i=indices[4u*cell+k];let p=position(i);let n=vertex_normal(i);
        for(var li=0u;li<snapshot.lights;li++){
            value+=energy_diffuse_source(n,energy_unit(snapshot.lamps[li].position.xyz-p),m.roughness,m.f0,m.albedo,m.retro)*light_power(i,li,p)*0.25;
        }
    }
    diffuse[2u*cell]=vec4f(value,m.roughness);
    diffuse[2u*cell+1u]=vec4f(m.f0,m.retro);
}
@compute @workgroup_size(64)
fn shade_corners(@builtin(global_invocation_id) id:vec3u){
    let cell=id.x+id.y*65535u*64u;if(cell>=frame.counts.y){return;}
    let m=cell_materials[cell];
    for(var k=0u;k<4u;k++){
    let i=indices[4u*cell+k];
    let p=position(i);let n=vertex_normal(i);let v=energy_unit(frame.eye.xyz-p);
    var direct_diffuse=vec3f(0.0);var direct_specular=vec3f(0.0);
    // Preserve the canonical GGX/backscatter closure for explicit lights.
    // Both lobes use THIS cell material. No shared-owner diffuse subtraction.
    for(var li=0u;li<snapshot.lights;li++){
        let l=energy_unit(snapshot.lamps[li].position.xyz-p);
        let power=light_power(i,li,p);let ci=dot(n,l);let co=dot(n,v);
        var view_diffuse=vec3f(0.0);
        if(ci>0.0 && co>0.0){
            let ei=energy_allocation(m.roughness,ci,m.f0);let eo=energy_allocation(m.roughness,co,m.f0);
            view_diffuse=(1.0-m.retro)*m.albedo*(1.0-ei.total)*(1.0-eo.total)/max(1.0-ei.mean,vec3f(1e-6))*ci;
        }
        direct_diffuse+=view_diffuse*power;
        direct_specular+=(energy_source(n,l,v,m.roughness,m.f0,m.albedo,m.retro)-view_diffuse)*power;
    }
    let rgb=clamp(frame.components.x*direct_diffuse+frame.components.y*direct_specular+frame.components.z*probe_specular(i,n,v,m),vec3f(0.0),vec3f(65504.0));
    // Two packed half pairs keep four corners per tet within WebGPU buffer limits.
    radiance[4u*cell+k]=vec2u(pack2x16float(rgb.rg),pack2x16float(vec2f(rgb.b,0.0)));
    }
}
