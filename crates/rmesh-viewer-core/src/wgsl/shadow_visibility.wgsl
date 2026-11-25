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
fn safe_inv_dir(d:vec3f)->vec3f {
    let sd=select(d,select(vec3f(1e-30),vec3f(-1e-30),d<vec3f(0.0)),abs(d)<vec3f(1e-30));
    return 1.0/sd;
}
fn box_entry(o:vec3f,inv_d:vec3f,lo:vec3f,hi:vec3f,far:f32)->f32 {
    let t0=(lo-o)*inv_d;
    let t1=(hi-o)*inv_d;
    let tmin=min(t0,t1);
    let tmax=max(t0,t1);
    let a=max(max(tmin.x,tmin.y),max(tmin.z,0.0));
    let b=min(min(tmax.x,tmax.y),min(tmax.z,far));
    return select(1e30,a,a<=b);
}
fn triangle(o:vec3f,d:vec3f,a:vec3f,b:vec3f,c:vec3f)->f32 {
    let e1=b-a; let e2=c-a; let h=cross(d,e2); let det=dot(e1,h);
    if(abs(det)<1e-30){return -1.0;}
    let inv_det=1.0/det;
    let s=o-a;let u=dot(s,h)*inv_det;if(u<0.0 || u>1.0){return -1.0;}
    let q=cross(s,e1);let vv=dot(d,q)*inv_det;if(vv<0.0 || u+vv>1.0){return -1.0;}
    return dot(e2,q)*inv_det;
}
struct Hit { distance:f32, face:i32 }
fn bvh(o:vec3f,d:vec3f,far:f32)->Hit {
    var hit=Hit(far,-1);
    let inv_d=safe_inv_dir(d);
    var node=shadow_node(0u);
    if(box_entry(o,inv_d,node.lo,node.hi,far)>far){return hit;}
    var stack_idx:array<u32,64>;
    var stack_t:array<f32,64>;
    var sp=0u;
    loop {
        if(node.left<0){
            let begin=u32(-node.left-1);
            let count=u32(node.right);
            for(var i=0u;i<count;i++){
                let packed=shadow_face(begin+i);
                let base=(packed>>2u)*4u;
                let f=FACES[packed&3u];
                let t=triangle(o,d,vertex(indices[base+f.x]),vertex(indices[base+f.y]),vertex(indices[base+f.z]));
                if(t>=0.0 && (t<hit.distance || (t==hit.distance && packed<bitcast<u32>(hit.face)))){hit=Hit(t,i32(packed));}
            }
        } else {
            let nl=shadow_node(u32(node.left));
            let nr=shadow_node(u32(node.right));
            let tl=box_entry(o,inv_d,nl.lo,nl.hi,hit.distance);
            let tr=box_entry(o,inv_d,nr.lo,nr.hi,hit.distance);
            let hit_l=tl<=hit.distance;
            let hit_r=tr<=hit.distance;
            if(hit_l && hit_r){
                if(sp>=64u){return Hit(0.0,-1);}
                if(tl<=tr){
                    stack_idx[sp]=u32(node.right);
                    stack_t[sp]=tr;
                    sp+=1u;
                    node=nl;
                } else {
                    stack_idx[sp]=u32(node.left);
                    stack_t[sp]=tl;
                    sp+=1u;
                    node=nr;
                }
                continue;
            } else if(hit_l){
                node=nl;
                continue;
            } else if(hit_r){
                node=nr;
                continue;
            }
        }
        var found=false;
        while(sp>0u){
            sp-=1u;
            if(stack_t[sp]<=hit.distance){
                node=shadow_node(stack_idx[sp]);
                found=true;
                break;
            }
        }
        if(!found){break;}
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
        let base=u32(current)*4u;
        let v0=vertex(indices[base]);
        let v1=vertex(indices[base+1u]);
        let v2=vertex(indices[base+2u]);
        let v3=vertex(indices[base+3u]);
        let e1=v1-v0;
        let e2=v2-v0;
        let e3=v3-v0;
        let nn0=cross(e2,e1);
        let s=select(1.0,-1.0,dot(nn0,e3)>0.0);
        let n0=nn0*s;
        let n2=cross(e3,e2)*s;
        let n3=cross(e1,e3)*s;
        let n1=-(n0+n2+n3);
        let vo=v0-o;
        let den0=dot(n0,d);
        let den1=dot(n1,d);
        let den2=dot(n2,d);
        let den3=dot(n3,d);
        var best=1e30;var face=-1;
        // Carry topology across a rounded crossing even if its plane is a
        // few ulps behind t; discarding that exit can leak the remaining ray.
        if(den0>0.0){let t0=dot(n0,vo)/den0;if(t0<best){best=t0;face=0;}}
        if(den1>0.0){let t1=dot(n1,v1-o)/den1;if(t1<best){best=t1;face=1;}}
        if(den2>0.0){let t2=dot(n2,vo)/den2;if(t2<best){best=t2;face=2;}}
        if(den3>0.0){let t3=dot(n3,vo)/den3;if(t3<best){best=t3;face=3;}}
        // A ray on a shared face must cross to its neighbor even when this
        // cell contributes zero length. Boundary-only BVH recovery cannot
        // recover an internal crossing, and would otherwise leak light.
        if(face>=0){
            let end=max(t,min(best,far));var next=-1;let candidate=neighbors[base+u32(face)];
            if(candidate>=0 && (shadow_boundary_only() || inside(o+d*(end+shadow_epsilon()),candidate))){next=candidate;}
            return Segment(end,current,next,true);
        }
    }
    let hit=bvh(o+d*(t+shadow_epsilon()),d,max(far-t-shadow_epsilon(),0.0));
    if(hit.face>=0){
        let crossing=t+shadow_epsilon()+hit.distance;let end=min(crossing,far);
        // Classify the two sides by orientation, not by an epsilon-offset
        // point: that point can jump completely through a thin absorbing cell.
        let cell=u32(hit.face)>>2u;let f=u32(hit.face)&3u;let ids=FACES[f];let vs=tet(cell);
        var n=cross(vs[ids.y]-vs[ids.x],vs[ids.z]-vs[ids.x]);
        if(dot(n,vs[ids.w]-vs[ids.x])>0.0){n=-n;}
        let neighbor=neighbors[4u*cell+f];
        if(dot(n,d)<0.0){return Segment(end,neighbor,i32(cell),true);}
        return Segment(end,i32(cell),neighbor,true);
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
        if(seg.cell>=0 && u32(seg.cell)<active_tet_count()){tau+=max(density[u32(seg.cell)],0.0)*(seg.end-t);}
        t=seg.end;current=seg.next;
        if(t>=far || tau>=shadow_cutoff()){return exp(-tau);}
    }
    return 0.0;
}
