// Camera-independent probe capture bake plus complete per-cell-corner shading.
// - `prepare_vertex_lights` + coalesced `bake_diffuse` (with `cell_seen` bitset skip)
//   shades only cells touched by the 5 reflection probes while keeping sequential reads/writes.
// - `prepare_vertex_view` deduplicates per-vertex geometry, visibility, retroreflection, inline
//   light-0 terms (16 B/vert hot region), and 5-probe cubemaps packed as RGB9E5 (32 B/vert).
// - `shade_corners` uses scalarized 4-corner evaluation with 16-byte `vec4u` index/radiance I/O,
//   keeping every cell's diffuse/specular closure 100% consistent with its own exported material.
struct Frame { eye:vec4f, probe:vec4f, counts:vec4u, components:vec4f, probes:array<vec4f,5>, influence:vec4f }
struct Lamp { position:vec4f, color:vec4f, direction:vec4f }
struct BakeSnapshot { offset:u32,count:u32,lights:u32,tet_count:u32,control:vec4u,lamps:array<Lamp,16> }
struct Material { albedo:vec3f,roughness:f32,f0:vec3f,retro:f32 }

@group(0) @binding(0) var<uniform> frame:Frame;
@group(0) @binding(1) var<storage,read> positions:array<f32>;
@group(0) @binding(2) var<storage,read> normals:array<f32>;
@group(0) @binding(4) var<storage,read> transport:array<f32>;
@group(0) @binding(5) var<uniform> snapshot:BakeSnapshot;
@group(0) @binding(6) var<storage,read_write> diffuse:array<vec2u>;
@group(0) @binding(7) var<storage,read_write> radiance:array<vec4u>;
@group(0) @binding(8) var energy_lut:texture_2d<f32>;
@group(0) @binding(9) var probe_maps:texture_cube_array<f32>;
@group(0) @binding(10) var probe_sampler:sampler;
@group(0) @binding(11) var<storage,read> cell_materials:array<vec4u>;
@group(0) @binding(12) var<storage,read> indices:array<vec4u>;
@group(0) @binding(13) var<storage,read> probe_weights:array<f32>;
@group(0) @binding(15) var<storage,read> cell_seen:array<u32>;

// Per-vertex light cache (written by prepare_vertex_lights, read by bake_diffuse & prepare_vertex_view):
// 1 vec4u per vertex: (f32(power0.r), f32(power0.g), f32(power0.b), lit_count | (li0 << 5) | (u16unorm(ci0) << 16))
@group(0) @binding(16) var<storage,read_write> vertex_light_rw:array<vec4u>;
@group(0) @binding(21) var<storage,read> vertex_light:array<vec4u>;

// De-interleaved per-vertex view cache (`2 * vertex_count` vec4u):
//   Hot region [0..vertex_count): 16 bytes/vertex:
//     .x = pack2x16unorm(vec2f(co, ci0))  (0u when co <= 0)
//     .y = pack2x16unorm(vec2f(sin_nh0, lh5_0))
//     .z = pack2x16float(power0.rg)
//     .w = (pack2x16float(vec2f(power0.b, 0.0)) & 0xffffu) | (flags << 16u)
//   Cold retro tail [vertex_count..2*vertex_count): read only if m.retro > 0 && retro_nonzero.
@group(0) @binding(17) var<storage,read_write> vertex_view_rw:array<vec4u>;
@group(0) @binding(18) var<storage,read> vertex_view:array<vec4u>;

// Per-vertex pre-blended 5-probe cubemap levels packed as RGB9E5 (8 u32s = 32 bytes/vertex):
// slots 0..6 = GGX mips 0..6 along reflected dir, slot 7 = broad_light along normal at max mip.
@group(0) @binding(19) var<storage,read_write> vertex_probe_rw:array<u32>;
@group(0) @binding(20) var<storage,read> vertex_probe:array<u32>;

fn unpack_material(packed:vec4u)->Material {
    let a=unpack2x16unorm(packed.x);
    let b=unpack2x16unorm(packed.y);
    let c=unpack2x16unorm(packed.z);
    let d=unpack2x16unorm(packed.w);
    return Material(vec3f(a.x,a.y,b.x),b.y,vec3f(c.x,c.y,d.x),d.y);
}
fn position(i:u32)->vec3f {return vec3f(positions[3u*i],positions[3u*i+1u],positions[3u*i+2u]);}
fn vertex_normal(i:u32)->vec3f {return -energy_unit(vec3f(normals[3u*i],normals[3u*i+1u],normals[3u*i+2u]));}
fn light_power_vis(vis:f32,lamp:Lamp,delta:vec3f,dist2:f32)->vec3f {
    var emission=1.0;
    if(lamp.position.w>0.5){emission=smoothstep(lamp.direction.w,min(1.0,lamp.direction.w+0.1),dot(-delta/max(sqrt(dist2),1e-8),lamp.direction.xyz));}
    return lamp.color.rgb*(lamp.color.w*emission*vis/max(dist2,1e-6));
}

fn pack_rgb9e5(c_in:vec3f)->u32 {
    let c=clamp(c_in,vec3f(0.0),vec3f(65408.0));
    let m=max(c.r,max(c.g,c.b));
    if(m<=0.0){return 0u;}
    let exp_raw=i32((bitcast<u32>(m)>>23u) & 0xffu)-127;
    let e=u32(clamp(exp_raw+16,0,31));
    let inv_scale=bitcast<f32>((151u-e)<<23u);
    let q=min(vec3u(c*inv_scale+0.5),vec3u(511u));
    return q.x | (q.y<<9u) | (q.z<<18u) | (e<<27u);
}

fn unpack_rgb9e5(w:u32)->vec3f {
    if(w==0u){return vec3f(0.0);}
    let e=w>>27u;
    let scale=bitcast<f32>((e+103u)<<23u);
    return vec3f(f32(w & 0x1ffu),f32((w>>9u) & 0x1ffu),f32((w>>18u) & 0x1ffu))*scale;
}

fn load_retro_rgb(i:u32,vcount:u32)->vec3f {
    let packed=vertex_view[vcount+i];
    return vec3f(unpack2x16float(packed.x),unpack2x16float(packed.y).x);
}

fn energy_alloc_fb(r:f32,c:f32,f0:vec3f,fb:vec3f)->EnergyAllocation {
    let lut=energy_lookup(r,c);
    let C=fb*fb*lut.z/max(1.0-fb*(1.0-lut.z),vec3f(1e-6));
    let total=clamp(f0*lut.x+(1.0-f0)*lut.y+C*(1.0-lut.x),vec3f(0.0),vec3f(1.0));
    let mean=clamp(f0*lut.z+(1.0-f0)*lut.w+C*(1.0-lut.z),vec3f(0.0),vec3f(1.0));
    return EnergyAllocation(total,mean,lut.x,lut.z,C);
}

// Stage 1a (per-vertex, runs on light generation change): precompute (power0, ci0, li0) once per vertex.
@compute @workgroup_size(64)
fn prepare_vertex_lights(@builtin(global_invocation_id) id:vec3u){
    let i=id.x+id.y*65535u*64u;if(i>=frame.counts.x){return;}
    let n_lights=snapshot.lights;
    if(n_lights==0u){
        vertex_light_rw[i]=vec4u(0u);
        return;
    }
    let p=position(i);let n=vertex_normal(i);
    let tbase=16u*i;
    var lit_count=0u;
    var li0=0u;
    var power0=vec3f(0.0);
    var ci0=0.0;
    for(var li=0u;li<n_lights;li++){
        let vis=transport[tbase+li];
        if(vis<=0.0){continue;}
        let lamp=snapshot.lamps[li];
        let delta=lamp.position.xyz-p;
        let dist2=dot(delta,delta);
        let l=delta/max(sqrt(dist2),1e-8);
        let ci=dot(n,l);
        if(ci<=0.0){continue;}
        if(lit_count==0u){
            power0=light_power_vis(vis,lamp,delta,dist2);
            ci0=ci;
            li0=li;
        }
        lit_count+=1u;
    }
    let ci_q=u32(clamp(ci0,0.0,1.0)*65535.0+0.5);
    vertex_light_rw[i]=vec4u(
        bitcast<u32>(power0.r),
        bitcast<u32>(power0.g),
        bitcast<u32>(power0.b),
        lit_count | (li0<<5u) | (ci_q<<16u),
    );
}

// Stage 1b (linear coalesced over cells, skips untouched cells via 407 KB cell_seen bitset).
@compute @workgroup_size(64)
fn bake_diffuse(@builtin(global_invocation_id) id:vec3u){
    let cell=id.x+id.y*65535u*64u;if(cell>=frame.counts.y){return;}
    if((cell_seen[cell>>5u] & (1u<<(cell & 31u)))==0u){return;}
    let n_lights=snapshot.lights;
    if(n_lights==0u){
        diffuse[cell]=vec2u(0u,0u);
        return;
    }
    let m=unpack_material(cell_materials[cell]);
    let fb=m.f0+(1.0-m.f0)/21.0;
    let diff_scale=(1.0-m.retro)*m.albedo*0.25;
    let ids=indices[cell];
    var value=vec3f(0.0);
    for(var k=0u;k<4u;k++){
        let i=ids[k];
        let vl=vertex_light[i];
        let lit_count=vl.w & 0x1fu;
        if(lit_count==0u){continue;}
        if(lit_count==1u){
            let ci=f32(vl.w>>16u)*(1.0/65535.0);
            let power=vec3f(bitcast<f32>(vl.x),bitcast<f32>(vl.y),bitcast<f32>(vl.z));
            let lut=energy_lookup(m.roughness,ci);
            let C=fb*fb*lut.z/max(1.0-fb*(1.0-lut.z),vec3f(1e-6));
            let ei_total=clamp(m.f0*lut.x+(1.0-m.f0)*lut.y+C*(1.0-lut.x),vec3f(0.0),vec3f(1.0));
            value+=diff_scale*(1.0-ei_total)*(ci*power);
        } else {
            let p=position(i);let n=vertex_normal(i);
            let tbase=16u*i;
            for(var li=0u;li<n_lights;li++){
                let vis=transport[tbase+li];
                if(vis<=0.0){continue;}
                let lamp=snapshot.lamps[li];
                let delta=lamp.position.xyz-p;
                let dist2=dot(delta,delta);
                let l=delta/max(sqrt(dist2),1e-8);
                let ci=dot(n,l);
                if(ci<=0.0){continue;}
                let lut=energy_lookup(m.roughness,ci);
                let C=fb*fb*lut.z/max(1.0-fb*(1.0-lut.z),vec3f(1e-6));
                let ei_total=clamp(m.f0*lut.x+(1.0-m.f0)*lut.y+C*(1.0-lut.x),vec3f(0.0),vec3f(1.0));
                value+=diff_scale*(1.0-ei_total)*(ci*light_power_vis(vis,lamp,delta,dist2));
            }
        }
    }
    let rgb=clamp(value,vec3f(0.0),vec3f(65504.0));
    diffuse[cell]=vec2u(pack2x16float(rgb.rg),pack2x16float(vec2f(rgb.b,0.0)));
}

// Stage 2a (per-vertex, runs on camera or light change): precompute compact 16 B hot
// view/light cache, cold retro tail, and RGB9E5-packed 5-probe cubemaps (32 B/vert).
@compute @workgroup_size(64)
fn prepare_vertex_view(@builtin(global_invocation_id) id:vec3u){
    let vcount=frame.counts.x;
    let i=id.x+id.y*65535u*64u;if(i>=vcount){return;}
    let p=position(i);let n=vertex_normal(i);
    let v=energy_unit(frame.eye.xyz-p);
    let co=dot(n,v);
    if(co<=0.0){
        vertex_view_rw[i]=vec4u(0u);
        return;
    }

    let vl=vertex_light[i];
    let lit_count=vl.w & 0x1fu;
    let mirrored=2.0*co*n-v;
    var power0=vec3f(0.0);
    var ci0=0.0;
    var sin_nh0=0.0;
    var lh5_0=0.0;
    var retro_rgb=vec3f(0.0);

    if(lit_count==1u){
        let li0=(vl.w>>5u) & 0xfu;
        let lamp=snapshot.lamps[li0];
        let delta=lamp.position.xyz-p;
        let l=delta/max(length(delta),1e-8);
        power0=vec3f(bitcast<f32>(vl.x),bitcast<f32>(vl.y),bitcast<f32>(vl.z));
        ci0=f32(vl.w>>16u)*(1.0/65535.0);
        retro_rgb=vec3f(energy_ggx(n,l,mirrored,0.25)*ci0*3.141592653589793)*power0;
        let h=energy_unit(l+v);
        let t=cross(n,h);
        sin_nh0=clamp(length(t),0.0,1.0);
        let om=1.0-clamp(dot(l,h),0.0,1.0);
        let om2=om*om;
        lh5_0=om2*om2*om;
    } else if(lit_count>1u){
        let n_lights=snapshot.lights;
        let tbase=16u*i;
        for(var li=0u;li<n_lights;li++){
            let vis=transport[tbase+li];
            if(vis<=0.0){continue;}
            let lamp=snapshot.lamps[li];
            let delta=lamp.position.xyz-p;
            let dist2=dot(delta,delta);
            let l=delta/max(sqrt(dist2),1e-8);
            let ci=dot(n,l);
            if(ci<=0.0){continue;}
            let power=light_power_vis(vis,lamp,delta,dist2);
            retro_rgb+=vec3f(energy_ggx(n,l,mirrored,0.25)*ci*3.141592653589793)*power;
        }
    }

    var probe_active=0u;
    if(frame.components.z>0.0){
        let wbase=5u*i;
        let w0=probe_weights[wbase];
        let w1=probe_weights[wbase+1u];
        let w2=probe_weights[wbase+2u];
        let w3=probe_weights[wbase+3u];
        let w4=probe_weights[wbase+4u];
        if(w0+w1+w2+w3+w4>0.0){
            probe_active=1u;
            let reflected=reflect(-v,n);
            let max_level=f32(frame.counts.z-1u);
            var s0=vec3f(0.0);var s1=vec3f(0.0);var s2=vec3f(0.0);var s3=vec3f(0.0);
            var s4=vec3f(0.0);var s5=vec3f(0.0);var s6=vec3f(0.0);var broad=vec3f(0.0);
            if(w0>0.0){
                s0+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,0.0).rgb;
                s1+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,1.0).rgb;
                s2+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,2.0).rgb;
                s3+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,3.0).rgb;
                s4+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,4.0).rgb;
                s5+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,5.0).rgb;
                s6+=w0*textureSampleLevel(probe_maps,probe_sampler,reflected,0,6.0).rgb;
                broad+=w0*textureSampleLevel(probe_maps,probe_sampler,n,0,max_level).rgb;
            }
            if(w1>0.0){
                s0+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,0.0).rgb;
                s1+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,1.0).rgb;
                s2+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,2.0).rgb;
                s3+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,3.0).rgb;
                s4+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,4.0).rgb;
                s5+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,5.0).rgb;
                s6+=w1*textureSampleLevel(probe_maps,probe_sampler,reflected,1,6.0).rgb;
                broad+=w1*textureSampleLevel(probe_maps,probe_sampler,n,1,max_level).rgb;
            }
            if(w2>0.0){
                s0+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,0.0).rgb;
                s1+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,1.0).rgb;
                s2+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,2.0).rgb;
                s3+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,3.0).rgb;
                s4+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,4.0).rgb;
                s5+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,5.0).rgb;
                s6+=w2*textureSampleLevel(probe_maps,probe_sampler,reflected,2,6.0).rgb;
                broad+=w2*textureSampleLevel(probe_maps,probe_sampler,n,2,max_level).rgb;
            }
            if(w3>0.0){
                s0+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,0.0).rgb;
                s1+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,1.0).rgb;
                s2+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,2.0).rgb;
                s3+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,3.0).rgb;
                s4+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,4.0).rgb;
                s5+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,5.0).rgb;
                s6+=w3*textureSampleLevel(probe_maps,probe_sampler,reflected,3,6.0).rgb;
                broad+=w3*textureSampleLevel(probe_maps,probe_sampler,n,3,max_level).rgb;
            }
            if(w4>0.0){
                s0+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,0.0).rgb;
                s1+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,1.0).rgb;
                s2+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,2.0).rgb;
                s3+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,3.0).rgb;
                s4+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,4.0).rgb;
                s5+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,5.0).rgb;
                s6+=w4*textureSampleLevel(probe_maps,probe_sampler,reflected,4,6.0).rgb;
                broad+=w4*textureSampleLevel(probe_maps,probe_sampler,n,4,max_level).rgb;
            }
            let pbase=8u*i;
            vertex_probe_rw[pbase+0u]=pack_rgb9e5(s0);
            vertex_probe_rw[pbase+1u]=pack_rgb9e5(s1);
            vertex_probe_rw[pbase+2u]=pack_rgb9e5(s2);
            vertex_probe_rw[pbase+3u]=pack_rgb9e5(s3);
            vertex_probe_rw[pbase+4u]=pack_rgb9e5(s4);
            vertex_probe_rw[pbase+5u]=pack_rgb9e5(s5);
            vertex_probe_rw[pbase+6u]=pack_rgb9e5(s6);
            vertex_probe_rw[pbase+7u]=pack_rgb9e5(broad);
        }
    }

    var code=lit_count;
    if(code==1u && max(power0.r,max(power0.g,power0.b))>65504.0){
        code=17u;
    }
    let retro_nonzero=select(0u,1u,max(retro_rgb.r,max(retro_rgb.g,retro_rgb.b))>0.0);
    if(retro_nonzero!=0u){
        let retro_c=clamp(retro_rgb,vec3f(0.0),vec3f(65504.0));
        vertex_view_rw[vcount+i]=vec4u(
            pack2x16float(retro_c.rg),
            pack2x16float(vec2f(retro_c.b,0.0)),
            0u,
            0u,
        );
    }
    let flags=code | (probe_active<<5u) | (retro_nonzero<<6u);
    let pb_lo=pack2x16float(vec2f(min(power0.b,65504.0),0.0)) & 0xffffu;
    let co_q=max(co,1.0/65535.0);
    vertex_view_rw[i]=vec4u(
        pack2x16unorm(vec2f(co_q,ci0)),
        pack2x16unorm(vec2f(sin_nh0,lh5_0)),
        pack2x16float(min(power0.rg,vec2f(65504.0))),
        pb_lo | (flags<<16u),
    );
}

fn shade_one_corner(
    i:u32,
    vcount:u32,
    n_lights:u32,
    m:Material,
    one_minus_retro:f32,
    fb:vec3f,
    a2:f32,
    one_minus_a2:f32,
    mip_lo:u32,
    mip_hi:u32,
    mip_f:f32,
    comp_d:f32,
    comp_s:f32,
    comp_p:f32,
)->vec2u {
    let v0=vertex_view[i];
    if(v0.x==0u){return vec2u(0u,0u);}
    let flags=v0.w>>16u;
    let lit_code=flags & 0x1fu;
    let probe_active=(flags & 0x20u)!=0u;
    if(lit_code==0u && !probe_active){return vec2u(0u,0u);}

    let co_ci=unpack2x16unorm(v0.x);
    let co=co_ci.x;
    let lut_o=energy_lookup(m.roughness,co);
    let C_o=fb*fb*lut_o.z/max(1.0-fb*(1.0-lut_o.z),vec3f(1e-6));
    let eo_total=clamp(m.f0*lut_o.x+(1.0-m.f0)*lut_o.y+C_o*(1.0-lut_o.x),vec3f(0.0),vec3f(1.0));
    let eo_mean=clamp(m.f0*lut_o.z+(1.0-m.f0)*lut_o.w+C_o*(1.0-lut_o.z),vec3f(0.0),vec3f(1.0));
    let eo=EnergyAllocation(eo_total,eo_mean,lut_o.x,lut_o.z,C_o);

    var direct_diffuse=vec3f(0.0);
    var direct_specular=vec3f(0.0);

    if(lit_code==1u){
        let ci=co_ci.y;
        let sl=unpack2x16unorm(v0.y);
        let tt=sl.x*sl.x;
        let nh2=max(1.0-tt,0.0);
        let lh5=sl.y;
        let prg=unpack2x16float(v0.z);
        let pb=unpack2x16float(v0.w & 0xffffu).x;
        let power=vec3f(prg.x,prg.y,pb);

        let ei=energy_alloc_fb(m.roughness,ci,m.f0,fb);
        let view_diffuse=one_minus_retro*m.albedo*(1.0-ei.total)*(1.0-eo.total)/max(1.0-ei.mean,vec3f(1e-6))*ci;
        let F=m.f0+(1.0-m.f0)*lh5;
        let denom=tt+a2*nh2;
        let D=a2/(3.141592653589793*max(denom*denom,1e-20));
        let gi=2.0/max(ci+sqrt(a2+one_minus_a2*ci*ci),1e-8);
        let go=2.0/max(co+sqrt(a2+one_minus_a2*co*co),1e-8);
        let ss=(0.25*D*gi*go)*F;
        let ms=ei.compensation*(1.0-ei.unit)*(1.0-eo.unit)/(3.141592653589793*max(1.0-ei.average,1e-6));
        direct_diffuse=view_diffuse*power;
        direct_specular=(one_minus_retro*(ss+ms)*ci*3.141592653589793)*power;
        if(m.retro>0.0 && (flags & 0x40u)!=0u){
            direct_specular+=m.retro*load_retro_rgb(i,vcount);
        }
    } else if(lit_code>1u){
        let p=position(i);let n=vertex_normal(i);let v=energy_unit(frame.eye.xyz-p);
        let tbase=16u*i;
        let go=2.0/max(co+sqrt(a2+one_minus_a2*co*co),1e-8);
        for(var li=0u;li<n_lights;li++){
            let vis=transport[tbase+li];
            if(vis<=0.0){continue;}
            let lamp=snapshot.lamps[li];
            let delta=lamp.position.xyz-p;
            let dist2=dot(delta,delta);
            let l=delta/max(sqrt(dist2),1e-8);
            let ci=dot(n,l);
            if(ci<=0.0){continue;}
            let power=light_power_vis(vis,lamp,delta,dist2);
            let ei=energy_alloc_fb(m.roughness,ci,m.f0,fb);
            let view_diffuse=one_minus_retro*m.albedo*(1.0-ei.total)*(1.0-eo.total)/max(1.0-ei.mean,vec3f(1e-6))*ci;
            let h=energy_unit(l+v);
            let om=1.0-clamp(dot(l,h),0.0,1.0);
            let om2=om*om;
            let F=m.f0+(1.0-m.f0)*(om2*om2*om);
            let nh=clamp(dot(n,h),0.0,1.0);
            let t=cross(n,h);
            let denom=dot(t,t)+a2*nh*nh;
            let D=a2/(3.141592653589793*max(denom*denom,1e-20));
            let gi=2.0/max(ci+sqrt(a2+one_minus_a2*ci*ci),1e-8);
            let ss=(0.25*D*gi*go)*F;
            let ms=ei.compensation*(1.0-ei.unit)*(1.0-eo.unit)/(3.141592653589793*max(1.0-ei.average,1e-6));
            direct_diffuse+=view_diffuse*power;
            direct_specular+=(one_minus_retro*(ss+ms)*ci*3.141592653589793)*power;
        }
        if(m.retro>0.0 && (flags & 0x40u)!=0u){
            direct_specular+=m.retro*load_retro_rgb(i,vcount);
        }
    }

    var probe_spec=vec3f(0.0);
    if(probe_active){
        let pbase=8u*i;
        let single_lo=unpack_rgb9e5(vertex_probe[pbase+mip_lo]);
        let single_light=select(
            mix(single_lo,unpack_rgb9e5(vertex_probe[pbase+mip_hi]),mip_f),
            single_lo,
            mip_lo==mip_hi,
        );
        let broad_light=unpack_rgb9e5(vertex_probe[pbase+7u]);
        probe_spec=one_minus_retro*((m.f0*lut_o.x+(1.0-m.f0)*lut_o.y)*single_light+
            eo.compensation*(1.0-lut_o.x)*broad_light);
    }

    let rgb=clamp(comp_d*direct_diffuse+comp_s*direct_specular+comp_p*probe_spec,vec3f(0.0),vec3f(65504.0));
    return vec2u(pack2x16float(rgb.rg),pack2x16float(vec2f(rgb.b,0.0)));
}

// Stage 2b (per-cell, runs on camera or light change): scalarized 4-corner shading with
// 16-byte coalesced index loads and radiance writes.
@compute @workgroup_size(64)
fn shade_corners(@builtin(global_invocation_id) id:vec3u){
    let cell=id.x+id.y*65535u*64u;if(cell>=frame.counts.y){return;}
    let m=unpack_material(cell_materials[cell]);
    let ids=indices[cell];
    let vcount=frame.counts.x;
    let n_lights=snapshot.lights;
    let one_minus_retro=1.0-m.retro;
    let fb=m.f0+(1.0-m.f0)/21.0;
    let r_clamped=clamp(m.roughness,0.04,1.0);
    let a=r_clamped*r_clamped;
    let a2=a*a;
    let one_minus_a2=1.0-a2;
    let max_mip=frame.counts.z-1u;
    let level=clamp((m.roughness-0.04)/0.96,0.0,1.0)*f32(max_mip);
    let mip_lo=u32(floor(level));
    let mip_hi=min(mip_lo+1u,max_mip);
    let mip_f=fract(level);
    let comp_d=frame.components.x;
    let comp_s=frame.components.y;
    let comp_p=frame.components.z;

    let r0=shade_one_corner(ids.x,vcount,n_lights,m,one_minus_retro,fb,a2,one_minus_a2,mip_lo,mip_hi,mip_f,comp_d,comp_s,comp_p);
    let r1=shade_one_corner(ids.y,vcount,n_lights,m,one_minus_retro,fb,a2,one_minus_a2,mip_lo,mip_hi,mip_f,comp_d,comp_s,comp_p);
    let r2=shade_one_corner(ids.z,vcount,n_lights,m,one_minus_retro,fb,a2,one_minus_a2,mip_lo,mip_hi,mip_f,comp_d,comp_s,comp_p);
    let r3=shade_one_corner(ids.w,vcount,n_lights,m,one_minus_retro,fb,a2,one_minus_a2,mip_lo,mip_hi,mip_f,comp_d,comp_s,comp_p);
    radiance[2u*cell]=vec4u(r0,r1);
    radiance[2u*cell+1u]=vec4u(r2,r3);
}
