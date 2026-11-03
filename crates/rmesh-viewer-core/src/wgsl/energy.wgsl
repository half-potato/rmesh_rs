// sdir renderer/energy_material.py: direct-light energy model (mode 4).
// Caller supplies energy_lut: texture_2d<f32>, RGBA = Eu,E0,average Eu,average E0.
// Explicit bilinear loads reproduce grid_sample(align_corners=True, border).
fn energy_lookup(roughness:f32, cosine:f32)->vec4f {
    let xy=vec2f(clamp(cosine,0.0,1.0),(clamp(roughness,0.04,1.0)-0.04)/0.96)*64.0;
    let lo=vec2i(floor(xy));let hi=min(lo+vec2i(1),vec2i(64));let f=fract(xy);
    return mix(mix(textureLoad(energy_lut,lo,0),textureLoad(energy_lut,vec2i(hi.x,lo.y),0),f.x),
               mix(textureLoad(energy_lut,vec2i(lo.x,hi.y),0),textureLoad(energy_lut,hi,0),f.x),f.y);
}
struct EnergyAllocation { total:vec3f, mean:vec3f, unit:f32, average:f32, compensation:vec3f }
fn energy_allocation(r:f32,c:f32,f0:vec3f)->EnergyAllocation {
    let lut=energy_lookup(r,c);let fb=f0+(1.0-f0)/21.0;
    let C=fb*fb*lut.z/max(1.0-fb*(1.0-lut.z),vec3f(1e-6));
    let total=clamp(f0*lut.x+(1.0-f0)*lut.y+C*(1.0-lut.x),vec3f(0.0),vec3f(1.0));
    let mean=clamp(f0*lut.z+(1.0-f0)*lut.w+C*(1.0-lut.z),vec3f(0.0),vec3f(1.0));
    return EnergyAllocation(total,mean,lut.x,lut.z,C);
}
fn energy_unit(v:vec3f)->vec3f { return v/max(length(v),1e-8); }
fn energy_ggx(n:vec3f,l:vec3f,v:vec3f,r:f32)->f32 {
    let ci=dot(n,l);let co=dot(n,v);
    if(ci<=0.0 || co<=0.0){return 0.0;}
    let h=energy_unit(l+v);let nh=clamp(dot(n,h),0.0,1.0);
    let a2=pow(clamp(r,0.04,1.0),4.0);let t=cross(n,h);
    let denom=dot(t,t)+a2*nh*nh;
    let D=a2/(3.141592653589793*max(denom*denom,1e-20));
    let gi=2.0/max(max(ci,0.0)+sqrt(a2+(1.0-a2)*ci*ci),1e-8);
    let go=2.0/max(max(co,0.0)+sqrt(a2+(1.0-a2)*co*co),1e-8);
    return 0.25*D*gi*go;
}
// Full direct-light source, including pi*N.L and retroreflection exactly as
// volumetric_relight.shade_volume; lamp RGB, inverse-square and visibility outside.
fn energy_source(n:vec3f,l:vec3f,v:vec3f,r:f32,f0:vec3f,albedo:vec3f,retro:f32)->vec3f {
    let ci=dot(n,l);let co=dot(n,v);var brdf=vec3f(0.0);
    if(ci>0.0 && co>0.0){
        let ei=energy_allocation(r,ci,f0);let eo=energy_allocation(r,co,f0);
        let h=energy_unit(l+v);let F=f0+(1.0-f0)*pow(1.0-clamp(dot(l,h),0.0,1.0),5.0);
        let ss=energy_ggx(n,l,v,r)*F;
        let ms=ei.compensation*(1.0-ei.unit)*(1.0-eo.unit)/(3.141592653589793*max(1.0-ei.average,1e-6));
        let diffuse=albedo*(1.0-ei.total)*(1.0-eo.total)/(3.141592653589793*max(1.0-ei.mean,vec3f(1e-6)));
        brdf=diffuse+ss+ms;
    }
    let mirrored=2.0*dot(n,v)*n-v;
    let back=energy_ggx(n,l,mirrored,0.25);
    return ((1.0-retro)*brdf+vec3f(retro*back))*max(ci,0.0)*3.141592653589793;
}

// Fast diffuse preview: hemispherical average over outgoing directions cancels
// (1-Eo)/(1-Eavg). Retain incoming energy allocation; omit specular and retro.
fn energy_diffuse_source(n:vec3f,l:vec3f,r:f32,f0:vec3f,albedo:vec3f,retro:f32)->vec3f {
    let ci=max(dot(n,l),0.0);
    return (1.0-retro)*albedo*(1.0-energy_allocation(r,ci,f0).total)*ci;
}
