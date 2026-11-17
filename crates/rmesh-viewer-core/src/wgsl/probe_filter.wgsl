// GGX importance prefilter, not ordinary box-filtered cubemap mips.
struct Filter { size:u32, level:u32, levels:u32, samples:u32 }
@group(0) @binding(0) var<uniform> settings:Filter;
@group(0) @binding(1) var input:texture_cube_array<f32>;
@group(0) @binding(2) var linear_sampler:sampler;
@group(0) @binding(3) var output:texture_storage_2d_array<rgba16float,write>;
@compute @workgroup_size(8,8)
fn prefilter(@builtin(global_invocation_id) id:vec3u){
    if(id.x>=settings.size || id.y>=settings.size){return;}
    let n=cube_direction(id.xy,settings.size,id.z%6u);
    let r=0.04+0.96*f32(settings.level)/f32(settings.levels-1u);let a=r*r;
    let up=select(vec3f(0.0,0.0,1.0),vec3f(1.0,0.0,0.0),abs(n.z)>0.999);
    let tangent=normalize(cross(up,n));let bitangent=cross(n,tangent);
    var sum=vec3f(0.0);var weight=0.0;
    for(var i=0u;i<settings.samples;i++){
        let u=(f32(i)+0.5)/f32(settings.samples);let v=f32(reverseBits(i))*2.3283064365386963e-10;
        let phi=6.28318530718*v;let ct=sqrt((1.0-u)/(1.0+(a*a-1.0)*u));let st=sqrt(max(1.0-ct*ct,0.0));
        let h=tangent*(cos(phi)*st)+bitangent*(sin(phi)*st)+n*ct;
        let l=2.0*dot(n,h)*h-n;let nl=max(dot(n,l),0.0);
        sum+=textureSampleLevel(input,linear_sampler,l,i32(id.z/6u),0.0).rgb*nl;weight+=nl;
    }
    textureStore(output,vec2i(id.xy),i32(id.z),vec4f(sum/max(weight,1e-8),1.0));
}
