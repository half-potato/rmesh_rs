struct Frame { eye:vec4f, probe:vec4f, counts:vec4u, components:vec4f, probes:array<vec4f,5>, influence:vec4f }
@group(0) @binding(0) var<uniform> frame:Frame;
@group(0) @binding(1) var<storage,read> vertices:array<f32>;
@group(0) @binding(2) var<storage,read> indices:array<u32>;
@group(0) @binding(3) var<storage,read> density:array<f32>;
@group(0) @binding(4) var<storage,read> neighbors:array<i32>;
@group(0) @binding(5) var<storage,read> diffuse:array<vec4f>;
@group(0) @binding(6) var<storage,read> shadow_geometry:array<u32>;
@group(0) @binding(7) var output:texture_storage_2d_array<rgba16float,write>;
struct Node {lo:vec3f,left:i32,hi:vec3f,right:i32}
fn shadow_node(i:u32)->Node {
    let b=4u+i*8u;
    return Node(vec3f(bitcast<f32>(shadow_geometry[b]),bitcast<f32>(shadow_geometry[b+1u]),bitcast<f32>(shadow_geometry[b+2u])),bitcast<i32>(shadow_geometry[b+3u]),
    vec3f(bitcast<f32>(shadow_geometry[b+4u]),bitcast<f32>(shadow_geometry[b+5u]),bitcast<f32>(shadow_geometry[b+6u])),bitcast<i32>(shadow_geometry[b+7u]));
}
fn shadow_face(i:u32)->u32{return shadow_geometry[4u+shadow_geometry[0]*8u+i];}
fn shadow_epsilon()->f32{return frame.probe.w;}
fn shadow_limit()->u32{return 4096u;}
fn shadow_cutoff()->f32{return 12.0;}
fn shadow_boundary_only()->bool{return true;}
@compute @workgroup_size(8,8)
fn capture(@builtin(global_invocation_id) id:vec3u){
    let size=textureDimensions(output);if(id.x>=size.x || id.y>=size.y){return;}
    let probe=frame.probes[id.z/6u];
    let d=cube_direction(id.xy,size.x,id.z%6u);let o=probe.xyz;
    var t=0.0;var current=bitcast<i32>(probe.w);var trans=1.0;var rgb=vec3f(0.0);
    for(var step=0u;step<4096u;step++){
        let seg=segment(o,d,t,frame.eye.w,current);
        if(seg.cell>=0){
            let cell=u32(seg.cell);let c=diffuse[2u*cell].rgb;
            let alpha=1.0-exp(-max(density[cell],0.0)*max(seg.end-t,0.0));
            rgb+=trans*alpha*c;trans*=1.0-alpha;
        }
        if(!seg.hit || seg.end>=frame.eye.w || trans<0.00001){break;}
        t=seg.end;current=seg.next;
    }
    textureStore(output,vec2i(id.xy),i32(id.z),vec4f(min(rgb,vec3f(65504.0)),1.0));
}

@group(0) @binding(8) var<storage,read_write> weights:array<f32>;
// Normalize spatial influence before visibility. A blocked set must stay dark.
@compute @workgroup_size(64)
fn cache_visibility(@builtin(global_invocation_id) id:vec3u){
    let i=id.x+id.y*65535u*64u;if(i>=frame.counts.x){return;}
    let p=vertex(i);var spatial:array<f32,5>;var total=0.0;
    for(var j=0u;j<5u;j++){
        let q=length(p-frame.probes[j].xyz)/frame.influence.x;
        let falloff=max(1.0-q*q,0.0);
        spatial[j]=falloff*falloff;total+=spatial[j];
    }
    for(var j=0u;j<5u;j++){
        var weight=0.0;
        if(spatial[j]>0.0){
            let probe=frame.probes[j];let delta=p-probe.xyz;let distance=length(delta);
            let far=max(distance-shadow_epsilon(),0.0);
            // Trace from the probe's known containing cell to just before the vertex.
            var trans=1.0;var t=0.0;var cell=bitcast<i32>(probe.w);var completed=far<=0.0;
            for(var step=0u;step<4096u && !completed;step++){
                let seg=segment(probe.xyz,unit(delta),t,far,cell);
                if(seg.cell>=0){trans*=exp(-max(density[u32(seg.cell)],0.0)*max(seg.end-t,0.0));}
                completed=seg.end>=far || !seg.hit || trans<0.00001;
                t=seg.end;cell=seg.next;
            }
            if(!completed){trans=0.0;}
            // Apply attenuation AFTER blend normalization so overlapping probes cannot cancel it.
            let r=frame.influence.y;
            var attenuation=1.0;
            if(r>0.0){attenuation=1.0/(1.0+(distance/r)*(distance/r));}
            weight=spatial[j]/max(total,1.0)*trans*attenuation;
        }
        weights[5u*i+j]=weight;
    }
}
