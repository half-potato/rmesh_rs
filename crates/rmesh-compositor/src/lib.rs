//! rmesh-compositor: Opaque primitive rendering + depth compositing with tet volume rendering.

pub mod compositor_pass;
pub mod geometry;
pub mod material;
pub mod primitive_pass;

pub use compositor_pass::{
    create_compositor_bind_group, record_composite, CompositorPipeline, CompositorTargets,
    CompositorUniforms,
};
pub use geometry::{MeshSlice, PrimitiveGeometry, PrimitiveVertex};
pub use material::{MaterialDef, MaterialRegistry, PbrMaterial, TextureData};
pub use primitive_pass::{
    record_collision_debug_pass, record_primitive_pass, MrtViews, PrimitivePipeline,
    PrimitiveTargets,
};
