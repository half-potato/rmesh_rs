use glam::{Mat4, Quat, Vec3};

/// Spatial transform (position, rotation, scale).
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Kind of geometric primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    Cube,
    Sphere,
    Plane,
    Cylinder,
    PointLight,
    SpotLight,
    /// Custom mesh loaded from glTF. The `usize` indexes into the compositor's custom mesh registry.
    CustomMesh(usize),
}

impl PrimitiveKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::Sphere => "Sphere",
            Self::Plane => "Plane",
            Self::Cylinder => "Cylinder",
            Self::PointLight => "Point Light",
            Self::SpotLight => "Spot Light",
            Self::CustomMesh(_) => "Mesh",
        }
    }

    /// Index into the built-in geometry kinds array. Panics for `CustomMesh`.
    pub fn index(self) -> usize {
        match self {
            Self::Cube => 0,
            Self::Sphere => 1,
            Self::Plane => 2,
            Self::Cylinder => 3,
            Self::PointLight => 1, // Render as sphere gizmo
            Self::SpotLight => 1,  // Render as sphere gizmo
            Self::CustomMesh(_) => panic!("CustomMesh has no built-in geometry index"),
        }
    }

    pub fn is_custom_mesh(self) -> bool {
        matches!(self, Self::CustomMesh(_))
    }

    pub fn custom_mesh_index(self) -> Option<usize> {
        match self {
            Self::CustomMesh(i) => Some(i),
            _ => None,
        }
    }
}

/// A named primitive with a transform and material properties.
#[derive(Debug, Clone)]
pub struct Primitive {
    pub kind: PrimitiveKind,
    pub transform: Transform,
    pub name: String,
    /// Per-instance color override. `None` uses the default color for the primitive kind.
    pub color: Option<[f32; 4]>,
    /// PBR roughness factor (0.0 = mirror, 1.0 = fully rough). Default 1.0.
    pub roughness: f32,
    /// PBR metallic factor (0.0 = dielectric, 1.0 = metal). Default 0.0.
    pub metallic: f32,
    /// Index into the compositor's MaterialRegistry for texture lookups.
    pub material_index: Option<usize>,
}

impl Primitive {
    pub fn new(kind: PrimitiveKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            transform: Transform::default(),
            name: name.into(),
            color: None,
            roughness: 1.0,
            metallic: 0.0,
            material_index: None,
        }
    }
}
