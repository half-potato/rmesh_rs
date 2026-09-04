/// Types shared between CPU and GPU code.
/// These match the WGSL struct layouts byte-for-byte (std430/storage buffer layout).
///
/// Matches the FrozenTetModel layout from delaunay_splatting:
///   - Per-tet: density (1), rgb (3), gradient (3×3), sh ((deg+1)^2 × 3)
///   - No iNGP network — direct parameter storage
use bytemuck::{Pod, Zeroable};

/// Per-frame uniforms.
/// Bound as a **storage buffer** (not uniform) to avoid std140 alignment hassles.
/// std430 layout matches Rust's repr(C) directly.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    /// Column-major 4×4 view-projection matrix
    pub vp_col0: [f32; 4],
    pub vp_col1: [f32; 4],
    pub vp_col2: [f32; 4],
    pub vp_col3: [f32; 4],
    /// Camera-to-world rotation matrix columns (xyz, w=0 padding)
    pub c2w_col0: [f32; 4],
    pub c2w_col1: [f32; 4],
    pub c2w_col2: [f32; 4],
    /// Camera intrinsics [fx, fy, cx, cy]
    pub intrinsics: [f32; 4],
    /// Camera position (xyz) + padding (w)
    pub cam_pos_pad: [f32; 4],
    /// Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
    /// Total number of tetrahedra
    pub tet_count: u32,
    /// Current training step
    pub step: u32,
    /// Tile size for tiled pipeline (must match shaders).
    pub tile_size_u: u32,
    /// Ray mode: 0 = derive from c2w + intrinsics, 1 = read from ray buffers.
    pub ray_mode: u32,
    /// Minimum ray-origin offset along view direction (matches Slang camera.min_t).
    pub min_t: f32,
    /// SH degree for inline SH evaluation (0 = use base_colors path, 1-3 = eval SH).
    pub sh_degree: u32,
    /// Near clip plane distance (positive, view-space Z).
    pub near_plane: f32,
    /// Far clip plane distance (positive, view-space Z).
    pub far_plane: f32,
    /// NDC depth slack added to the shell z-cull bound in `project_compute*`.
    /// Positive values cull less (safer); 0 culls exactly at the shell exit.
    pub shell_cull_bias: f32,
    /// Bit 0: shell z-cull enabled. Remaining bits reserved (keeps the struct
    /// at 192 bytes without reintroducing a pad field).
    pub shell_cull_flags: u32,
}

impl Uniforms {
    pub fn vp_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_cols(
            glam::Vec4::from(self.vp_col0),
            glam::Vec4::from(self.vp_col1),
            glam::Vec4::from(self.vp_col2),
            glam::Vec4::from(self.vp_col3),
        )
    }

    pub fn c2w_matrix(&self) -> glam::Mat3 {
        glam::Mat3::from_cols(
            glam::Vec3::new(self.c2w_col0[0], self.c2w_col0[1], self.c2w_col0[2]),
            glam::Vec3::new(self.c2w_col1[0], self.c2w_col1[1], self.c2w_col1[2]),
            glam::Vec3::new(self.c2w_col2[0], self.c2w_col2[1], self.c2w_col2[2]),
        )
    }
}

/// DrawIndirectCommand layout (matches wgpu::DrawIndirectArgs).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Loss computation uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LossUniforms {
    pub width: u32,
    pub height: u32,
    /// 0 = L1, 1 = L2, 2 = SSIM
    pub loss_type: u32,
    /// Weight for SSIM component when using mixed loss
    pub lambda_ssim: f32,
}

/// Adam optimizer push constants / uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AdamUniforms {
    pub param_count: u32,
    pub step: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub _pad: [u32; 2],
}

// Constant face winding for all tets (4 faces × 3 vertex indices)
// 12 vertices total when unrolled for triangle-list topology
pub const TET_FACE_INDICES: [u32; 12] = [
    0, 2, 1, // face 0
    1, 2, 3, // face 1
    0, 3, 2, // face 2
    3, 0, 1, // face 3
];

// --- SH basis constants ---
pub const C0: f32 = 0.282_094_8;
pub const C1: f32 = 0.488_602_52;

pub const C2_0: f32 = 1.092_548_5;
pub const C2_1: f32 = -1.092_548_5;
pub const C2_2: f32 = 0.315_391_57;
pub const C2_3: f32 = -1.092_548_5;
pub const C2_4: f32 = 0.546_274_24;

pub const C3_0: f32 = -0.590_043_6;
pub const C3_1: f32 = 2.890_611_4;
pub const C3_2: f32 = -0.457_045_8;
pub const C3_3: f32 = 0.373_176_34;
pub const C3_4: f32 = -0.457_045_8;
pub const C3_5: f32 = 1.445_305_7;
pub const C3_6: f32 = -0.590_043_6;

/// Tile-based backward pass uniforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TileUniforms {
    pub screen_width: u32,
    pub screen_height: u32,
    pub tile_size: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub num_tiles: u32,
    pub visible_tet_count: u32,
    pub _pad: [u32; 5],
}

/// BVH node for boundary face acceleration structure.
/// Matches the WGSL `BVHNode` struct layout (std430).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BVHNode {
    pub aabb_min: [f32; 3],
    /// >= 0: left child index, < 0: -(leaf_start + 1)
    pub left_or_face: i32,
    pub aabb_max: [f32; 3],
    /// Internal: right child index. Leaf: face count.
    pub right_or_count: i32,
}

/// Per-light data for deferred shading (matches WGSL `Light` struct).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub light_type: u32, // 0=point, 1=spot, 2=directional
    pub color: [f32; 3],
    pub intensity: f32,
    /// Pre-normalized — shader skips the normalize.
    pub direction: [f32; 3],
    /// cos(inner_angle) — pre-computed so the shader avoids a per-pixel cos.
    pub inner_cos: f32,
    /// cos(outer_angle) — pre-computed so the shader avoids a per-pixel cos.
    pub outer_cos: f32,
    pub _pad: [f32; 3],
}

/// Maximum number of lights supported.
pub const MAX_LIGHTS: usize = 16;

// Safe math constants (from safe_math.slang)
pub const SAFE_MIN: f32 = -1e20;
pub const SAFE_MAX: f32 = 1e20;
pub const TINY_VAL: f32 = 1e-20;
/// Early termination threshold: exp(-5.54) ≈ 1/255
pub const LOG_TRANSMITTANCE_THRESHOLD: f32 = -5.54;

// ---------------------------------------------------------------------------
// G-buffer attachment formats
// ---------------------------------------------------------------------------

/// Formats of the four MRT attachments the forward passes render into.
///
/// These were a single shared `color_format` variable, which forced slots 0, 1
/// and 3 to move together and — worse — was the *same* variable the DSM moment
/// target used, so narrowing the G-buffer silently narrowed DSM. Naming each
/// attachment separately decouples them and makes the set a runtime parameter,
/// which is what lets a test render a baseline and a candidate in one process.
///
/// Every producer's `FragmentState.targets` array and the corresponding texture
/// in `RenderTargets` must agree attachment-for-attachment, so both sides read
/// from one of these.
///
/// Constraints on the individual slots:
/// - `color` is sampled by `BlitPipeline` with a filtering sampler, so it must
///   stay float-**filterable**.
/// - `depth` holds linear view-space Z in world units and is the root of the
///   Hi-Z chain, so it must stay a float format — it cannot be a unorm.
/// - All four are blended with premultiplied `over`, so all must be blendable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GBufferFormats {
    /// Slot 0 — `albedo.rgb * a, a`. `.a` is the master alpha gate for the
    /// whole deferred shader, so this slot needs a full-precision alpha.
    pub color: wgpu::TextureFormat,
    /// Slot 1 — `roughness * a, f0_dielectric * a, metallic * a, a`.
    /// Nothing reads this slot's `.a` (the deferred pass un-premultiplies by
    /// slot 0's alpha), so a format with a narrow alpha channel is safe here.
    pub aux0: wgpu::TextureFormat,
    /// Slot 2 — `(n * 0.5 + 0.5) * a, a`, bias-encoded so the unorm range fits
    /// a signed unit vector. Already narrowed; the precedent for the others.
    pub normals: wgpu::TextureFormat,
    /// Slot 3 — `E[z] * a, occlusion * a, E[z^2] * a, a`.
    pub depth: wgpu::TextureFormat,
}

impl Default for GBufferFormats {
    fn default() -> Self {
        Self {
            // Narrowed from Rgba16Float. Albedo is [0,1] in every asset, and in
            // the deferred path this slot holds `albedo * alpha`, so the unorm
            // range is safe. Srgb rather than linear because the stored value is
            // premultiplied albedo: a linear 8-bit step is a flat 1/255 which is
            // ~20% relative error at albedo 0.02 and bands in darks. The sRGB
            // variant measures better on quality — 48.2 dB / 0.9992 vs linear's
            // 45.8 dB / 0.9973, with 2.6x lower worst-case error — but the ROP's
            // sRGB encode/decode on every one of ~118 blends per pixel costs a
            // measured 1.75 ms/frame at 1080p. Linear clears the 35 dB / 0.99
            // bar by ~10 dB, so the time is worth more than the margin here.
            // Switch to Rgba8UnormSrgb if dark-region banding ever shows up in
            // a real scene; it needs no shader changes either way.
            color: wgpu::TextureFormat::Rgba8Unorm,
            // Narrowed from Rgba16Float. roughness/f0/metallic are all [0,1],
            // and the richest asset spans 210 of 255 levels in roughness, so 8
            // bits covers the data. Measured against the f16 baseline on
            // room_pbr_refined at 640x360: PSNR 53.0 dB, SSIM 0.9996, with the
            // difference reading as dither rather than banding.
            aux0: wgpu::TextureFormat::Rgba8Unorm,
            normals: wgpu::TextureFormat::Rgba8Unorm,
            depth: wgpu::TextureFormat::Rgba16Float,
        }
    }
}

impl GBufferFormats {
    /// Parse a G-buffer format override from the environment, starting from
    /// `Default`. Format:
    ///   `RMESH_GBUF="color=rgba8srgb,aux0=rgba16f"`
    /// Slots not named keep their default. Unknown names panic loudly rather
    /// than silently rendering a different configuration than intended.
    ///
    /// This exists so a perf sweep can vary the attachment set without a
    /// rebuild — which matters because the cost is not linear in the number of
    /// bytes saved, so the interesting question is which *combinations* are
    /// cheap, not which single slot is.
    pub fn from_env() -> Self {
        let mut f = Self::default();
        let Ok(spec) = std::env::var("RMESH_GBUF") else {
            return f;
        };
        for part in spec.split(',').filter(|p| !p.trim().is_empty()) {
            let (slot, val) = part
                .split_once('=')
                .unwrap_or_else(|| panic!("RMESH_GBUF: expected slot=format, got {part:?}"));
            let fmt = match val.trim().to_ascii_lowercase().as_str() {
                "rgba16f" | "rgba16float" => wgpu::TextureFormat::Rgba16Float,
                "rgba8" | "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
                "rgba8srgb" | "rgba8unormsrgb" => wgpu::TextureFormat::Rgba8UnormSrgb,
                "rgb10a2" | "rgb10a2unorm" => wgpu::TextureFormat::Rgb10a2Unorm,
                "rg16f" | "rg16float" => wgpu::TextureFormat::Rg16Float,
                other => panic!("RMESH_GBUF: unknown format {other:?}"),
            };
            match slot.trim().to_ascii_lowercase().as_str() {
                "color" => f.color = fmt,
                "aux0" => f.aux0 = fmt,
                "normals" => f.normals = fmt,
                "depth" => f.depth = fmt,
                other => panic!("RMESH_GBUF: unknown slot {other:?}"),
            }
        }
        f
    }

    /// The four attachment formats, in slot order.
    pub fn slots(&self) -> [wgpu::TextureFormat; 4] {
        [self.color, self.aux0, self.normals, self.depth]
    }

    /// Bytes per pixel summed over all four attachments — the per-fragment
    /// blend footprint, which is what the raster pass is bound on.
    ///
    /// Uses the real texel size, NOT `target_pixel_byte_cost`. The latter is
    /// WebGPU's render-target *budget* accounting, which pads every 4-channel
    /// 32-bpp format to 8 — so it reports Rgba8Unorm as 8 bytes and would make
    /// a narrowing look like it saved nothing.
    pub fn bytes_per_pixel(&self) -> u32 {
        self.slots().iter().map(|f| bytes_per_texel(*f)).sum()
    }

    /// ROP slots per fragment. 64-bpp render targets blend at half rate on
    /// current NVIDIA parts, so they cost two slots where a 32-bpp target costs
    /// one. This is the quantity the raster pass's cost actually tracks.
    pub fn rop_slots(&self) -> u32 {
        self.slots()
            .iter()
            .map(|f| if bytes_per_texel(*f) <= 4 { 1 } else { 2 })
            .sum()
    }

    /// Sum of WebGPU's per-attachment budget cost, which must stay within the
    /// `max_color_attachment_bytes_per_sample` limit (32 by default). Note this
    /// does NOT drop when a 4-channel format narrows from 64-bpp to 32-bpp —
    /// both cost 8 here — so it is not a measure of the win.
    pub fn attachment_budget(&self) -> u32 {
        self.slots()
            .iter()
            .map(|f| f.target_pixel_byte_cost().unwrap_or(0))
            .sum()
    }
}

/// Bytes occupied by one texel of `fmt` in a render target.
///
/// Readback paths need this to size the copy and stride correctly. Several
/// hardcoded `8` constants (assuming Rgba16Float everywhere) survived the
/// normals attachment's narrowing to Rgba8Unorm and have been silently
/// decoding garbage ever since — route every readback through this instead.
pub fn bytes_per_texel(fmt: wgpu::TextureFormat) -> u32 {
    // `block_copy_size` is the true bytes-per-texel used by buffer copies.
    // Deliberately NOT `target_pixel_byte_cost`, which is the render-target
    // budget figure and reports 8 for Rgba8Unorm — using that as a readback
    // stride would misread every row.
    fmt.block_copy_size(None)
        .unwrap_or_else(|| panic!("{fmt:?} has no defined block copy size"))
}

/// Decode one texel of `fmt` into linear RGBA, matching exactly what
/// `textureLoad` returns to a shader for the same texel.
///
/// Two details that are easy to get wrong and both matter here:
/// - `Rgba8UnormSrgb` is decoded through the sRGB EOTF, because the hardware
///   does the same on load. The stored value is encoded, the shader sees linear.
/// - Formats with no alpha channel yield `a = 1.0`, which is what `textureLoad`
///   substitutes. This is load-bearing: code that gates on `.a > 0.01` silently
///   becomes always-true under such a format.
pub fn decode_texel(fmt: wgpu::TextureFormat, b: &[u8]) -> [f32; 4] {
    use wgpu::TextureFormat as F;
    let f16 = |i: usize| half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32();
    let u8n = |i: usize| b[i] as f32 / 255.0;
    // sRGB EOTF — the exact inverse of the encode the hardware applies on store.
    let srgb = |c: f32| {
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    match fmt {
        F::Rgba16Float => [f16(0), f16(2), f16(4), f16(6)],
        F::Rgba8Unorm => [u8n(0), u8n(1), u8n(2), u8n(3)],
        F::Rgba8UnormSrgb => [srgb(u8n(0)), srgb(u8n(1)), srgb(u8n(2)), u8n(3)],
        F::Rgb10a2Unorm => {
            let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            [
                (v & 0x3FF) as f32 / 1023.0,
                ((v >> 10) & 0x3FF) as f32 / 1023.0,
                ((v >> 20) & 0x3FF) as f32 / 1023.0,
                ((v >> 30) & 0x3) as f32 / 3.0,
            ]
        }
        // No alpha channel in the format — textureLoad substitutes 1.0.
        F::Rg16Float => [f16(0), f16(2), 0.0, 1.0],
        F::Rg8Unorm => [u8n(0), u8n(1), 0.0, 1.0],
        F::R16Float => [f16(0), 0.0, 0.0, 1.0],
        F::R8Unorm => [u8n(0), 0.0, 0.0, 1.0],
        F::R32Float => [f32::from_le_bytes([b[0], b[1], b[2], b[3]]), 0.0, 0.0, 1.0],
        other => panic!("decode_texel: unhandled format {other:?}"),
    }
}
