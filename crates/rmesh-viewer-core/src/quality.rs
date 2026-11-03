//! Shared geometry quality settings for desktop and browser viewers.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortMode {
    /// GPU radix sort, 32-bit depth keys. Default; fastest on Metal/Vulkan
    /// with subgroups (Drs backend).
    Gpu32,
    /// GPU radix sort, 16-bit linearly-quantized depth keys. Half as many
    /// radix passes; quantization can change ordering of nearby intervals.
    Gpu16,
    /// CPU projection cull and `radsort` fallback.
    Cpu,
}

impl std::fmt::Display for SortMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SortMode::Gpu32 => write!(f, "GPU 32-bit"),
            SortMode::Gpu16 => write!(f, "GPU 16-bit"),
            SortMode::Cpu => write!(f, "CPU"),
        }
    }
}

/// Geometry quality presets for the baked-radiance viewer. These control
/// sorting and culling only; screen-space lighting is no longer part of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityPreset {
    /// CPU sort with aggressive importance culling. CPU sorting can be
    /// slower than portable GPU sorting while moving the camera.
    Lowest,
    /// 16-bit GPU sort, no screen-space effects.
    Low,
    Medium,
    /// Full-depth GPU sort with shell and importance culling.
    High,
    /// No shell or importance culling.
    Reference,
}

impl QualityPreset {
    pub const ALL: [QualityPreset; 5] = [
        QualityPreset::Lowest,
        QualityPreset::Low,
        QualityPreset::Medium,
        QualityPreset::High,
        QualityPreset::Reference,
    ];

    pub fn label(self) -> &'static str {
        match self {
            QualityPreset::Lowest => "Lowest",
            QualityPreset::Low => "Low",
            QualityPreset::Medium => "Medium",
            QualityPreset::High => "High",
            QualityPreset::Reference => "Reference",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.label().eq_ignore_ascii_case(name.trim()))
    }

    /// The knob values this preset stands for:
    /// (sort, shell, importance, importance threshold).
    pub fn knobs(self) -> (SortMode, bool, bool, f32) {
        match self {
            QualityPreset::Lowest => (SortMode::Cpu, true, true, 0.03),
            QualityPreset::Low => (SortMode::Gpu16, true, true, 0.02),
            QualityPreset::Medium => (SortMode::Gpu16, true, true, 0.01),
            QualityPreset::High => (SortMode::Gpu32, true, true, 0.01),
            QualityPreset::Reference => (SortMode::Gpu32, false, false, 0.01),
        }
    }
}

