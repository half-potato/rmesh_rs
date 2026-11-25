struct Frame { eye:vec4f, probe:vec4f, counts:vec4u, components:vec4f, probes:array<vec4f,5>, influence:vec4f }
@group(0) @binding(0) var<uniform> frame:Frame;
@group(0) @binding(1) var<storage,read> vertices:array<f32>;
@group(0) @binding(2) var<storage,read> indices:array<u32>;
@group(0) @binding(3) var<storage,read> density:array<f32>;
@group(0) @binding(4) var<storage,read> neighbors:array<i32>;
@group(0) @binding(5) var<storage,read> diffuse:array<vec2u>;
@group(0) @binding(6) var<storage,read> shadow_geometry:array<vec4u>;
@group(0) @binding(7) var output:texture_storage_2d_array<rgba16float,write>;
@group(0) @binding(10) var<storage,read_write> probe_hits_rw:array<vec2u>;
@group(0) @binding(11) var<storage,read> probe_hits:array<vec2u>;
@group(0) @binding(12) var<storage,read_write> cell_seen:array<atomic<u32>>;

const PROBE_HIT_CAP:u32 = 96u;
const PROBE_HIT_END:u32 = 0xffffffffu;
const PROBE_HIT_OVERFLOW:u32 = 0xfffffffeu;
const PROBE_MIN_W:f32 = 2e-4;
const PROBE_MIN_TRANS:f32 = 0.002;

struct Node {lo:vec3f,left:i32,hi:vec3f,right:i32}
fn shadow_node(i:u32)->Node {
    let b=1u+i*2u;
    let n0=shadow_geometry[b];
    let n1=shadow_geometry[b+1u];
    return Node(vec3f(bitcast<f32>(n0.x),bitcast<f32>(n0.y),bitcast<f32>(n0.z)),bitcast<i32>(n0.w),
    vec3f(bitcast<f32>(n1.x),bitcast<f32>(n1.y),bitcast<f32>(n1.z)),bitcast<i32>(n1.w));
}
fn shadow_face(i:u32)->u32{return shadow_geometry[1u+shadow_geometry[0].x*2u+(i>>2u)][i&3u];}
fn shadow_epsilon()->f32{return frame.probe.w;}
fn shadow_limit()->u32{return 4096u;}
fn shadow_cutoff()->f32{return 12.0;}
fn shadow_boundary_only()->bool{return true;}
fn active_tet_count()->u32{return frame.counts.y;}

// Precompute static SoA ray-segment (cell, weight) lists for all 5x6x64x64 probe rays
// once per geometry revision, and collect the compact unique set of touched cells so
// bake_diffuse only shades cells actually seen by the reflection probes.
@compute @workgroup_size(8,8)
fn capture_cache(@builtin(global_invocation_id) id:vec3u){
    let dim=u32(frame.influence.z);
    if(id.x>=dim || id.y>=dim){return;}
    let probe=frame.probes[id.z/6u];
    let d=cube_direction(id.xy,dim,id.z%6u);let o=probe.xyz;
    let stride=dim*dim*30u;
    let ray=(id.z*dim+id.y)*dim+id.x;
    var t=0.0;var current=bitcast<i32>(probe.w);var trans=1.0;
    var count=0u;var overflow=false;
    for(var step=0u;step<4096u;step++){
        let seg=segment(o,d,t,frame.eye.w,current);
        if(seg.cell>=0 && u32(seg.cell)<frame.counts.y){
            let cell=u32(seg.cell);
            let rho=max(density[cell],0.0);
            let dt=max(seg.end-t,0.0);
            if(rho>0.0 && dt>0.0){
                let next_trans=trans*exp(-rho*dt);
                let w=trans-next_trans;
                trans=next_trans;
                if(w>PROBE_MIN_W){
                    let word=cell>>5u;
                    let bit=1u<<(cell&31u);
                    atomicOr(&cell_seen[word],bit);
                    if(!overflow){
                        if(count<PROBE_HIT_CAP){
                            probe_hits_rw[count*stride+ray]=vec2u(cell,bitcast<u32>(w));
                            count+=1u;
                        } else {
                            overflow=true;
                        }
                    }
                }
            }
        }
        if(!seg.hit || seg.end>=frame.eye.w || trans<PROBE_MIN_TRANS){break;}
        t=seg.end;current=seg.next;
    }
    if(overflow){
        probe_hits_rw[ray]=vec2u(PROBE_HIT_OVERFLOW,0u);
    } else if(count<PROBE_HIT_CAP){
        probe_hits_rw[count*stride+ray]=vec2u(PROBE_HIT_END,0u);
    }
}

// Replay SoA-coalesced cached probe ray segments against updated cell diffuse radiance.
// Overflow rays execute the exact volume raymarch fallback.
@compute @workgroup_size(8,8)
fn capture(@builtin(global_invocation_id) id:vec3u){
    let size=textureDimensions(output);if(id.x>=size.x || id.y>=size.y){return;}
    let stride=size.x*size.y*30u;
    let ray=(id.z*size.y+id.y)*size.x+id.x;
    let h0=probe_hits[ray];
    var rgb=vec3f(0.0);
    if(h0.x==PROBE_HIT_OVERFLOW){
        let probe=frame.probes[id.z/6u];
        let d=cube_direction(id.xy,size.x,id.z%6u);let o=probe.xyz;
        var t=0.0;var current=bitcast<i32>(probe.w);var trans=1.0;
        for(var step=0u;step<4096u;step++){
            let seg=segment(o,d,t,frame.eye.w,current);
            if(seg.cell>=0 && u32(seg.cell)<frame.counts.y){
                let cell=u32(seg.cell);
                let rho=max(density[cell],0.0);
                let dt=max(seg.end-t,0.0);
                if(rho>0.0 && dt>0.0){
                    let next_trans=trans*exp(-rho*dt);
                    let w=trans-next_trans;
                    trans=next_trans;
                    if(w>PROBE_MIN_W){
                        let packed=diffuse[cell];
                        let c=vec3f(unpack2x16float(packed.x),unpack2x16float(packed.y).x);
                        rgb+=w*c;
                    }
                }
            }
            if(!seg.hit || seg.end>=frame.eye.w || trans<PROBE_MIN_TRANS){break;}
            t=seg.end;current=seg.next;
        }
    } else if(h0.x!=PROBE_HIT_END){
        let p0=diffuse[h0.x];
        rgb+=bitcast<f32>(h0.y)*vec3f(unpack2x16float(p0.x),unpack2x16float(p0.y).x);
        for(var s=1u;s<PROBE_HIT_CAP;s++){
            let h=probe_hits[s*stride+ray];
            if(h.x==PROBE_HIT_END){break;}
            let packed=diffuse[h.x];
            let c=vec3f(unpack2x16float(packed.x),unpack2x16float(packed.y).x);
            rgb+=bitcast<f32>(h.y)*c;
        }
    }
    textureStore(output,vec2i(id.xy),i32(id.z),vec4f(min(rgb,vec3f(65504.0)),1.0));
}

@group(0) @binding(8) var<storage,read_write> weights:array<f32>;
@group(0) @binding(9) var<storage,read> vertex_meta:array<vec4u>;
// Normalize spatial influence before visibility. A blocked set must stay dark.
// Dispatched in Morton order with id.z = probe index so each workgroup traces
// coherent rays from nearby vertices toward the same probe.
@compute @workgroup_size(64)
fn cache_visibility(@builtin(global_invocation_id) id:vec3u){
    let work=id.x+id.y*65535u*64u;if(work>=frame.counts.x){return;}
    let j=id.z;
    let sample_meta=vertex_meta[work];
    let i=sample_meta.z;
    let vpos=vertex(i);
    let inv_r=1.0/frame.influence.x;
    var total=0.0;var spatial_j=0.0;
    for(var k=0u;k<5u;k++){
        let q=length(vpos-frame.probes[k].xyz)*inv_r;
        let falloff=max(1.0-q*q,0.0);
        let s=falloff*falloff;
        total+=s;
        if(k==j){spatial_j=s;}
    }
    var weight=0.0;
    if(spatial_j>0.0){
        let probe=frame.probes[j];
        let anchor=sample_meta.x;
        var p=vpos;
        var cell=-1;
        if(anchor!=0xffffffffu){
            let tv=tet(anchor);
            let center=(tv[0]+tv[1]+tv[2]+tv[3])*0.25;
            p=mix(vpos,center,0.0001);
            cell=i32(anchor);
        }
        let delta=probe.xyz-p;
        let distance=length(delta);
        let far=max(distance-shadow_epsilon(),0.0);
        var tau=0.0;var t=0.0;var completed=far<=0.0;
        let dir=delta/max(distance,1e-30);
        for(var step=0u;step<4096u && !completed;step++){
            let seg=segment(p,dir,t,far,cell);
            if(seg.cell>=0 && u32(seg.cell)<frame.counts.y){tau+=max(density[u32(seg.cell)],0.0)*max(seg.end-t,0.0);}
            completed=seg.end>=far || !seg.hit || tau>=12.0;
            t=seg.end;cell=seg.next;
        }
        let trans=select(0.0,exp(-tau),completed);
        // Apply attenuation AFTER blend normalization so overlapping probes cannot cancel it.
        let r=frame.influence.y;
        var attenuation=1.0;
        if(r>0.0){
            let dist_v=length(vpos-probe.xyz);
            attenuation=1.0/(1.0+(dist_v/r)*(dist_v/r));
        }
        weight=spatial_j/max(total,1.0)*trans*attenuation;
    }
    weights[5u*i+j]=weight;
}
