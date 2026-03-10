#define_import_path rmesh::intersect

// Face winding (inward normals). Matches TET_FACES in Rust camera module.
const FACES: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0u, 2u, 1u),
    vec3<u32>(1u, 2u, 3u),
    vec3<u32>(0u, 3u, 2u),
    vec3<u32>(3u, 0u, 1u),
);
