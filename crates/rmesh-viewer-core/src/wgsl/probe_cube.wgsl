fn cube_direction(pixel:vec2u,size:u32,face:u32)->vec3f {
    let uv=2.0*(vec2f(pixel)+vec2f(0.5))/f32(size)-vec2f(1.0);
    var d=vec3f(0.0);
    switch face {
        case 0u:{d=vec3f(1.0,-uv.y,-uv.x);} case 1u:{d=vec3f(-1.0,-uv.y,uv.x);}
        case 2u:{d=vec3f(uv.x,1.0,uv.y);} case 3u:{d=vec3f(uv.x,-1.0,-uv.y);}
        case 4u:{d=vec3f(uv.x,-uv.y,1.0);} default:{d=vec3f(-uv.x,-uv.y,-1.0);}
    }
    return normalize(d);
}
