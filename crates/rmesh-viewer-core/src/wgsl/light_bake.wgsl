// Shared-vertex incident lighting, combined into per-cell radiance for rendering.
struct Light { position: vec4f, color: vec4f, direction: vec4f }
struct Params { offset: u32, count: u32, lights: u32, tet_count: u32, control:vec4u, lamps: array<Light, 16> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> density: array<f32>;
@group(0) @binding(4) var<storage, read> neighbors: array<i32>;
@group(0) @binding(5) var<storage, read> albedo: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> irradiance: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> radiance: array<f32>;
@group(0) @binding(8) var<storage, read> normals: array<f32>;
@group(0) @binding(9) var<storage, read> vertex_meta: array<vec4u>;
@compute @workgroup_size(64)
fn bake(@builtin(global_invocation_id) id:vec3u) {
    let work=params.offset+id.x;
    if(work>=params.count){return;}
    let sample_meta=vertex_meta[work];
    let i=sample_meta.z;
    let cell=sample_meta.x;
    if(cell==0xffffffffu){irradiance[i]=vec4f(0.0);return;}
    let v=tet(cell);
    let center=(v[0]+v[1]+v[2]+v[3])*0.25;
    // A tiny relative inset makes the starting cell unambiguous at shared
    // vertices. Unlike a receiver-depth bias it shrinks with the cell size.
    let p=mix(vertex(i),center,0.0001);
    if(params.control.z==1u){
        // No material response, intensity, or distance attenuation in this cache.
        for(var block=0u;block<4u;block++){
            var values=vec4f(0.0);
            for(var lane=0u;lane<4u;lane++){
                let li=block*4u+lane;
                if(li<params.lights){
                    let delta=params.lamps[li].position.xyz-p;let d=length(delta);
                    values[lane]=visibility(p,delta/max(d,1e-8),d,i32(cell));
                }
            }
            irradiance[4u*i+block]=values;
        }
        return;
    }
    let raw_n=-normal(i);
    let roughness=bitcast<f32>(sample_meta.y);
    var incoming=vec3f(0.0);
    for(var li=0u;li<params.lights;li++){
        let light=params.lamps[li];
        let delta=light.position.xyz-p;
        let d=max(length(delta),0.01);
        let l=delta/d;
        var emission=1.0;
        if(light.position.w>0.5){emission=smoothstep(light.direction.w,min(1.0,light.direction.w+0.1),dot(-l,light.direction.xyz));}
        if(emission==0.0){continue;}
        var sigma=1.0;
        if(dot(raw_n,raw_n)>1e-12){
            let cosine=clamp(dot(normalize(raw_n),l),-1.0,1.0);
            let a=roughness*roughness;
            sigma=sqrt(max(cosine*cosine+a*a*(1.0-cosine*cosine),1e-8));
            if(cosine<0.0){sigma*=a;}
        }
        let transport=visibility(p,l,d,i32(cell))*emission*sigma/(d*d+0.04);
        if(params.lights==1u){incoming=vec3f(transport);}
        else {incoming+=light.color.rgb*light.color.w*transport;}
    }
    irradiance[i]=vec4f(incoming,1.0);
}
@compute @workgroup_size(64)
fn combine(@builtin(global_invocation_id) id:vec3u) {
    let i=id.x+id.y*65535u*64u;
    if(i>=params.tet_count){return;}
    let b=i*4u;
    var incoming=(irradiance[indices[b]].rgb+irradiance[indices[b+1]].rgb+irradiance[indices[b+2]].rgb+irradiance[indices[b+3]].rgb)*0.25;
    if(params.lights==1u){incoming*=params.lamps[0].color.rgb*params.lamps[0].color.w;}
    let result=albedo[i].rgb*incoming;
    radiance[3*i]=result.r; radiance[3*i+1]=result.g; radiance[3*i+2]=result.b;
}
