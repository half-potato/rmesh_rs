// Direct optical-depth quadrature matching sdir/renderer/volumetric_relight.py.
// This is the accuracy fallback, not the vertex-average bake. Linear float32.
struct Params { count:u32, samples:u32, limit:u32, mode:u32, eps:f32, near:f32, offset:u32, pad:u32, light:vec4f, color:vec4f }
struct Node { lo:vec3f, left:i32, hi:vec3f, right:i32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> vertices:array<f32>;
@group(0) @binding(2) var<storage,read> indices:array<u32>;
@group(0) @binding(3) var<storage,read> densities:array<f32>;
@group(0) @binding(4) var<storage,read> normals:array<f32>;
@group(0) @binding(5) var<storage,read> materials:array<f32>;
@group(0) @binding(6) var<storage,read> neighbors:array<i32>;
@group(0) @binding(7) var<storage,read> nodes:array<Node>;
@group(0) @binding(8) var<storage,read> faces:array<u32>;
@group(1) @binding(0) var<storage,read> rays:array<f32>;
@group(1) @binding(1) var<storage,read_write> output:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> failures:array<atomic<u32>>;
@group(0) @binding(9) var energy_lut:texture_2d<f32>;
const FACES=array<vec4u,4>(vec4u(0,2,1,3),vec4u(1,2,3,0),vec4u(0,3,2,1),vec4u(3,0,1,2));
fn v(i:u32)->vec3f { return vec3f(vertices[3*i],vertices[3*i+1],vertices[3*i+2]); }
fn n(i:u32)->vec3f { return vec3f(normals[3*i],normals[3*i+1],normals[3*i+2]); }
fn tet(t:u32)->array<vec3f,4> { return array<vec3f,4>(v(indices[4*t]),v(indices[4*t+1]),v(indices[4*t+2]),v(indices[4*t+3])); }
fn unit(a:vec3f)->vec3f { return a/max(length(a),1e-30); }
fn inside(p:vec3f,t:i32)->bool {
    if(t<0){return false;}
    let vs=tet(u32(t));
    for(var f=0u;f<4u;f++){
        let ids=FACES[f]; let nn=cross(vs[ids.y]-vs[ids.x],vs[ids.z]-vs[ids.x]);
        if(dot(nn,p-vs[ids.x])*dot(nn,vs[ids.w]-vs[ids.x])<0.0){return false;}
    }
    return true;
}
fn box_hit(o:vec3f,d:vec3f,lo:vec3f,hi:vec3f,far:f32)->bool {
    var a=0.0; var b=far;
    for(var k=0u;k<3u;k++){
        if(abs(d[k])<1e-30){if(o[k]<lo[k] || o[k]>hi[k]){return false;}}
        else {let t0=(lo[k]-o[k])/d[k];let t1=(hi[k]-o[k])/d[k];a=max(a,min(t0,t1));b=min(b,max(t0,t1));}
    }
    return a<=b;
}
fn triangle(o:vec3f,d:vec3f,a:vec3f,b:vec3f,c:vec3f)->f32 {
    let e1=b-a; let e2=c-a; let h=cross(d,e2); let det=dot(e1,h);
    if(abs(det)<1e-30){return -1.0;}
    let s=o-a;let u=dot(s,h)/det;if(u<0.0 || u>1.0){return -1.0;}
    let q=cross(s,e1);let vv=dot(d,q)/det;if(vv<0.0 || u+vv>1.0){return -1.0;}
    return dot(e2,q)/det;
}
struct Hit { distance:f32, face:i32 }
fn bvh(o:vec3f,d:vec3f)->Hit {
    var hit=Hit(1e30,-1); var stack:array<u32,64>; var sp=1u; stack[0]=0u;
    loop {
        if(sp==0u){break;} sp-=1u;let node=nodes[stack[sp]];
        if(!box_hit(o,d,node.lo,node.hi,hit.distance)){continue;}
        if(node.left<0){
            let begin=u32(-node.left-1);
            for(var i=0u;i<u32(node.right);i++){
                let packed=faces[begin+i];let vs=tet(packed>>2u);let f=FACES[packed&3u];
                let t=triangle(o,d,vs[f.x],vs[f.y],vs[f.z]);
                if(t>=0.0 && t<hit.distance){hit=Hit(t,i32(packed));}
            }
        } else {
            if(sp+2u>64u){atomicAdd(&failures[0],1u);return Hit(0.0,-1);}
            stack[sp]=u32(node.left);stack[sp+1]=u32(node.right);sp+=2u;
        }
    }
    return hit;
}
fn classify(p:vec3f,packed:i32)->i32 {
    if(packed<0){return -1;}
    let t=u32(packed)>>2u; if(inside(p,i32(t))){return i32(t);}
    let next=neighbors[t*4u+(u32(packed)&3u)]; if(inside(p,next)){return next;}
    return -1;
}
struct Segment { end:f32, cell:i32, next:i32, hit:bool }
fn segment(o:vec3f,d:vec3f,t:f32,far:f32,current:i32)->Segment {
    if(current>=0){
        let vs=tet(u32(current)); var best=1e30;var face=-1;
        for(var f=0u;f<4u;f++){
            let ids=FACES[f];var nn=cross(vs[ids.y]-vs[ids.x],vs[ids.z]-vs[ids.x]);
            if(dot(nn,vs[ids.w]-vs[ids.x])>0.0){nn=-nn;}
            let den=dot(nn,d);
            if(den>0.0){let t0=dot(nn,vs[ids.x]-o)/den;if(t0>=t && t0<best){best=t0;face=i32(f);}}
        }
        if(face>=0 && best>t){
            let end=min(best,far);var next=-1;let candidate=neighbors[u32(current)*4u+u32(face)];
            if(inside(o+d*(end+params.eps),candidate)){next=candidate;}
            return Segment(end,current,next,true);
        }
    }
    let hit=bvh(o+d*(t+params.eps),d);
    if(hit.face>=0){
        let crossing=t+params.eps+hit.distance;let end=min(crossing,far);
        return Segment(end,classify(o+d*((t+end)*0.5),hit.face),classify(o+d*(crossing+params.eps),hit.face),true);
    }
    return Segment(far,current,-1,false);
}
fn visibility(p:vec3f,d:vec3f,far:f32,cell:i32)->f32 {
    var t=0.0;var tau=0.0;var current=cell;
    for(var step=0u;step<params.limit;step++){
        if(current>=0 && densities[u32(current)]<=0.0){current=-1;}
        let seg=segment(p,d,t,far,current);
        if(!seg.hit){return exp(-tau);}
        if(seg.cell>=0){tau+=max(densities[u32(seg.cell)],0.0)*(seg.end-t);}
        t=seg.end;current=seg.next;
        if(t>=far || tau>=5.54){return exp(-tau);}
    }
    atomicAdd(&failures[1],1u);return 0.0;
}
fn normal_at(p:vec3f,t:u32)->vec3f {
    let vs=tet(t);let e1=vs[1]-vs[0];let e2=vs[2]-vs[0];let e3=vs[3]-vs[0];let delta=p-vs[0];
    let det=dot(e1,cross(e2,e3));if(abs(det)<=1e-20){return vec3f(0.0);}
    let b1=dot(delta,cross(e2,e3))/det;let b2=dot(delta,cross(e3,e1))/det;let b3=dot(delta,cross(e1,e2))/det;
    return -unit(n(indices[4*t])*(1.0-b1-b2-b3)+n(indices[4*t+1])*b1+n(indices[4*t+2])*b2+n(indices[4*t+3])*b3);
}
fn ggx(nn:vec3f,l:vec3f,vv:vec3f,r:f32)->f32 {
    let nl=dot(nn,l);let nv=dot(nn,vv);if(nl<=0.0 || nv<=0.0 || length(l+vv)<1e-8){return 0.0;}
    let h=unit(l+vv);let nh=clamp(dot(nn,h),0.0,1.0);let a=pow(clamp(r,0.04,1.0),2.0);let a2=a*a;
    let tangent=cross(nn,h);let dd=dot(tangent,tangent)+a2*nh*nh;
    let D=a2/(3.141592653589793*dd*dd);
    let gl=2.0/(nl+sqrt(a2+(1.0-a2)*nl*nl));let gv=2.0/(nv+sqrt(a2+(1.0-a2)*nv*nv));
    return 0.25*D*gl*gv;
}
fn shade(p:vec3f,cell:u32,eye:vec3f)->vec3f {
    let delta=params.light.xyz-p;let ray_length=length(delta);let dist=max(ray_length,1e-3);let l=delta/max(ray_length,1e-12);
    let nn=normal_at(p,cell);let c=clamp(dot(nn,l),-1.0,1.0);let base=cell*9u;
    let r=max(materials[base],0.22);let a=r*r;var sigma=sqrt(max(c*c+a*a*(1.0-c*c),1e-8));if(c<0.0){sigma*=a;}
    var source=vec3f(materials[base+1],materials[base+2],materials[base+3])*sigma;
    if(params.mode==3u){
        let retro=clamp(materials[base+8],0.0,1.0);let sw=clamp(vec3f(materials[base+5],materials[base+6],materials[base+7]),vec3f(0.0),vec3f(1.0));
        source*= (vec3f(1.0)-sw)*(1.0-retro);let view=unit(eye-p);let normal=unit(nn);let nl=max(dot(normal,l),0.0);
        let spec=ggx(normal,l,view,materials[base+4])*nl*3.141592653589793;
        let mirrored=2.0*dot(normal,view)*normal-view;let back=ggx(normal,l,mirrored,0.25)*nl*3.141592653589793;
        source+=sw*((1.0-retro)*spec)+vec3f(retro*back);
    }
    if(params.mode==4u){
        source=energy_source(energy_unit(nn),energy_unit(delta),energy_unit(eye-p),materials[base+4],
            vec3f(materials[base+5],materials[base+6],materials[base+7]),
            vec3f(materials[base+1],materials[base+2],materials[base+3]),materials[base+8]);
    }
    return source*params.color.rgb*(visibility(p,l,ray_length,i32(cell))/(dist*dist));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3u){
    let i=params.offset+id.x;if(i>=params.count){return;}
    let base=i*6u;let eye=vec3f(rays[base],rays[base+1],rays[base+2]);let d=unit(vec3f(rays[base+3],rays[base+4],rays[base+5]));
    var t=params.near;var current=-1;var trans=1.0;var color=vec3f(0.0);var done=false;
    for(var step=0u;step<params.limit;step++){
        if(current>=0 && densities[u32(current)]<=0.0){current=-1;}
        let seg=segment(eye,d,t,1e20,current);if(!seg.hit){done=true;break;}
        current=seg.next;
        if(seg.cell>=0){
            let rho=max(densities[u32(seg.cell)],0.0);let tau=rho*(seg.end-t);let alpha=1.0-exp(-tau);
            if(alpha>1e-8){
                let count=max(1u,u32(ceil(alpha*f32(params.samples))));
                for(var j=0u;j<count;j++){
                    let u=(f32(j)+0.5)/f32(count);let distance=t-log(max(1.0-u*alpha,1e-12))/rho;
                    let p=eye+d*clamp(distance,t,seg.end);color+=shade(p,u32(seg.cell),eye)*(trans*alpha/f32(count));
                }
            }
            trans*=exp(-tau);
        }
        t=seg.end;if(trans<0.004){done=true;break;}
    }
    if(!done){atomicAdd(&failures[2],1u);}
    output[i]=vec4f(color,1.0-trans);
}
