// Shared full-face BVH recovery/re-entry for legacy and energy light baking.
const FACES=array<vec4u,4>(vec4u(0,2,1,3),vec4u(1,2,3,0),vec4u(0,3,2,1),vec4u(3,0,1,2));
fn vertex(i:u32)->vec3f { return vec3f(vertices[3*i],vertices[3*i+1],vertices[3*i+2]); }
fn normal(i:u32)->vec3f { return vec3f(normals[3*i],normals[3*i+1],normals[3*i+2]); }
fn tet(t:u32)->array<vec3f,4> { return array<vec3f,4>(vertex(indices[4*t]),vertex(indices[4*t+1]),vertex(indices[4*t+2]),vertex(indices[4*t+3])); }
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
fn box_entry(o:vec3f,d:vec3f,lo:vec3f,hi:vec3f,far:f32)->f32 {
    var a=0.0; var b=far;
    for(var k=0u;k<3u;k++){
        if(abs(d[k])<1e-30){if(o[k]<lo[k] || o[k]>hi[k]){return 1e30;}}
        else {let t0=(lo[k]-o[k])/d[k];let t1=(hi[k]-o[k])/d[k];a=max(a,min(t0,t1));b=min(b,max(t0,t1));}
    }
    return select(1e30,a,a<=b);
}
fn triangle(o:vec3f,d:vec3f,a:vec3f,b:vec3f,c:vec3f)->f32 {
    let e1=b-a; let e2=c-a; let h=cross(d,e2); let det=dot(e1,h);
    if(abs(det)<1e-30){return -1.0;}
    let s=o-a;let u=dot(s,h)/det;if(u<0.0 || u>1.0){return -1.0;}
    let q=cross(s,e1);let vv=dot(d,q)/det;if(vv<0.0 || u+vv>1.0){return -1.0;}
    return dot(e2,q)/det;
}
struct Hit { distance:f32, face:i32 }
fn bvh(o:vec3f,d:vec3f,far:f32)->Hit {
    var hit=Hit(far,-1); var stack:array<u32,64>; var sp=1u; stack[0]=0u;
    loop {
        if(sp==0u){break;} sp-=1u;let node=shadow_node(stack[sp]);
        if(box_entry(o,d,node.lo,node.hi,hit.distance)>hit.distance){continue;}
        if(node.left<0){
            let begin=u32(-node.left-1);
            for(var i=0u;i<u32(node.right);i++){
                let packed=shadow_face(begin+i);let vs=tet(packed>>2u);let f=FACES[packed&3u];
                let t=triangle(o,d,vs[f.x],vs[f.y],vs[f.z]);
                if(t>=0.0 && (t<hit.distance || (t==hit.distance && packed<bitcast<u32>(hit.face)))){hit=Hit(t,i32(packed));}
            }
        } else {
            if(sp+2u>64u){return Hit(0.0,-1);}
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
        // A ray on a shared face must cross to its neighbor even when this
        // cell contributes zero length. Boundary-only BVH recovery cannot
        // recover an internal crossing, and would otherwise leak light.
        if(face>=0 && best>=t){
            let end=min(best,far);var next=-1;let candidate=neighbors[u32(current)*4u+u32(face)];
            if(candidate>=0 && (shadow_boundary_only() || inside(o+d*(end+shadow_epsilon()),candidate))){next=candidate;}
            return Segment(end,current,next,true);
        }
    }
    let hit=bvh(o+d*(t+shadow_epsilon()),d,max(far-t-shadow_epsilon(),0.0));
    if(hit.face>=0){
        let crossing=t+shadow_epsilon()+hit.distance;let end=min(crossing,far);
        return Segment(end,classify(o+d*((t+end)*0.5),hit.face),classify(o+d*(crossing+shadow_epsilon()),hit.face),true);
    }
    return Segment(far,current,-1,false);
}
fn visibility(p:vec3f,d:vec3f,far:f32,cell:i32)->f32 {
    var t=0.0;var tau=0.0;var current=cell;
    for(var step=0u;step<shadow_limit();step++){
        // Empty cells still have valid adjacency. Keep walking them; dropping
        // the cell here forces an expensive BVH search at every empty cell.
        let seg=segment(p,d,t,far,current);
        if(!seg.hit){return exp(-tau);}
        if(seg.cell>=0){tau+=max(density[u32(seg.cell)],0.0)*(seg.end-t);}
        t=seg.end;current=seg.next;
        if(t>=far || tau>=shadow_cutoff()){return exp(-tau);}
    }
    return 0.0;
}
