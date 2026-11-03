//! GPU state: device, pipelines, buffers, bind groups, and profiling.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rmesh_compositor::{MaterialRegistry, PrimitiveGeometry, PrimitivePipeline, PrimitiveTargets};
use rmesh_pbd::{PbdPipelines, PbdSolver};
use rmesh_render::{
    BlitPipeline, ComputeIntervalPipelines, ForwardPipelines, IntervalPipelines, MaterialBuffers,
    MeshForwardPipelines, RenderTargets, SceneBuffers,
};

#[derive(Clone, Copy, PartialEq)]
pub enum RenderMode {
    Regular,
    Quad,
    MeshShader,
    IntervalShader,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Regular => write!(f, "Regular"),
            RenderMode::Quad => write!(f, "Quad"),
            RenderMode::MeshShader => write!(f, "Mesh Shader"),
            RenderMode::IntervalShader => write!(f, "Interval Shader"),
        }
    }
}

pub use rmesh_viewer_core::quality::SortMode;

/// Per-frame GPU timing breakdown from timestamp queries.
#[derive(Clone, Default)]
pub struct GpuTimings {
    pub sh_eval_ms: f32,
    pub project_ms: f32,
    pub sort_ms: f32,
    pub prepass_ms: f32,
    pub render_ms: f32,
    pub total_ms: f32,
}

/// Per-frame CPU timing breakdown (rolling average).
#[derive(Clone, Default)]
pub struct CpuTimings {
    pub poll_readback_ms: f32,
    pub acquire_ms: f32,
    pub egui_ms: f32,
    pub encode_ms: f32,
    pub submit_ms: f32,
    pub present_ms: f32,
    pub total_ms: f32,
}

pub const TS_QUERY_COUNT: u32 = 8;

/// Number of timestamp slots the frame actually writes: 0..6 (project, sort,
/// gen/prepass, render). Slots 6..8 belong to the standalone SH-eval compute
/// pass (`rmesh_util::sh_eval`), which the viewer never records — the interval
/// path evaluates SH inline in `project_compute.wgsl`.
///
/// Only ever resolve slots that were written this frame. `resolve_query_set`
/// over a never-written query is undefined behavior on Vulkan and **wedges the
/// NVIDIA Linux driver**: the submission never completes, so nothing is ever
/// presented and the window shows a black or frozen image while input keeps
/// working. If the SH-eval pass is ever recorded again, raise this to
/// `TS_QUERY_COUNT` only on the frames that actually dispatch it.
pub const TS_WRITTEN_COUNT: u32 = 6;

/// Timestamp capacity for the shell, occlusion hierarchy, and overdraw probe.
pub const PP_TS_COUNT: u32 = 64;

pub struct GpuState {
    pub light_bake: crate::light_bake::LightBake,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub pipelines: ForwardPipelines,
    pub blit_pipeline: BlitPipeline,
    pub sort_pipelines: rmesh_sort::RadixSortPipelines,
    pub sort_state: rmesh_sort::RadixSortState,
    pub sort_state_16bit: rmesh_sort::RadixSortState,
    pub sort_backend: rmesh_sort::SortBackend,
    /// CPU sphere-frustum cull + sort workspace. Used when
    /// `App.sort_mode == SortMode::Cpu`.
    pub cpu_sorter: rmesh_sort::CpuSorter,
    /// Reused staging-buffer pool for streaming `sort_values` +
    /// `indirect_args.instance_count` writes when `SortMode::Cpu` is active.
    /// Without this, every frame would call `queue.write_buffer` with ~19MB
    /// of sorted indices, which allocates a fresh staging buffer per call
    /// and leaks memory until the next-next submit completes.
    pub staging_belt: wgpu::util::StagingBelt,
    pub buffers: SceneBuffers,
    pub material_buffers: MaterialBuffers,
    pub targets: RenderTargets,
    pub compute_bg: rmesh_render::ForwardComputeBindGroups,
    pub hw_compute_bg: rmesh_render::ForwardHwComputeBindGroups,
    pub render_bg: wgpu::BindGroup,
    pub render_bg_b: wgpu::BindGroup,
    pub blit_bg: wgpu::BindGroup,
    pub tet_count: u32,
    pub sh_coeffs_buf: wgpu::Buffer,
    pub sh_degree: u32,
    pub pending_reconfigure: bool,
    // Mesh shader (optional)
    pub mesh_pipelines: Option<MeshForwardPipelines>,
    pub mesh_render_bg_a: Option<wgpu::BindGroup>,
    pub mesh_render_bg_b: Option<wgpu::BindGroup>,
    pub indirect_convert_bg: Option<wgpu::BindGroup>,
    // Interval shading (optional, requires mesh shader support)
    pub interval_pipelines: Option<IntervalPipelines>,
    pub interval_render_bg_a: Option<wgpu::BindGroup>,
    pub interval_render_bg_b: Option<wgpu::BindGroup>,
    pub interval_indirect_convert_bg: Option<wgpu::BindGroup>,
    // Compute-based interval shading (always available)
    pub compute_interval_pipelines: ComputeIntervalPipelines,
    pub compute_interval_gen_bg_a: wgpu::BindGroup,
    pub compute_interval_gen_bg_b: wgpu::BindGroup,
    pub compute_interval_gen_bg_a_16bit: wgpu::BindGroup,
    pub compute_interval_gen_bg_b_16bit: wgpu::BindGroup,
    pub compute_interval_render_bg: wgpu::BindGroup,
    pub compute_interval_convert_bg: wgpu::BindGroup,
    // Quad prepass bind groups (A/B for sort result location)
    pub prepass_bg_a: wgpu::BindGroup,
    pub prepass_bg_b: wgpu::BindGroup,
    pub quad_render_bg: wgpu::BindGroup,
    // Per-grab PBD solver (re-created on each BeginGrab).
    pub pbd_solver: Option<PbdSolver>,
    /// PBD compute pipelines (one per scene; cheap to create at startup).
    pub pbd_pipelines: PbdPipelines,
    pub tet_neighbors_buf: Option<wgpu::Buffer>,
    // Primitives
    pub primitive_geometry: PrimitiveGeometry,
    pub material_registry: MaterialRegistry,
    pub primitive_pipeline: PrimitivePipeline,
    pub primitive_targets: PrimitiveTargets,
    // Density shell: boundary surface of the density-thresholded region. Drawn
    // between the primitive pass and the volume pass, where it acts as both the
    // volume's opaque color backstop and the z-cull bound.
    pub shell_pipeline: rmesh_render::ShellPipeline,
    pub shell_buffers: rmesh_render::ShellBuffers,
    pub shell_bg: wgpu::BindGroup,
    /// Max-reduce pyramid over the shell/primitive depth. Sampled by
    /// `project_compute*`, so `compute_bg` / `hw_compute_bg` reference it and
    /// must be rebuilt whenever this is recreated (i.e. on resize).
    pub shell_hiz_pipelines: rmesh_render::ShellHiZPipelines,
    pub shell_hiz: rmesh_render::ShellHiZ,
    /// Per-pixel overdraw heatmap. Recreated with the G-buffer on resize, since
    /// its bind group holds the G-buffer colour view.
    pub overdraw_pipelines: rmesh_render::OverdrawPipelines,
    pub overdraw_target: rmesh_render::OverdrawTarget,
    // egui
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,
    // Instance count readback
    pub instance_count_readback: wgpu::Buffer,
    pub visible_instance_count: u32,
    // GPU timestamp profiling
    pub ts_query_set: wgpu::QuerySet,
    pub ts_resolve_buf: wgpu::Buffer,
    pub ts_readback: [wgpu::Buffer; 2],
    pub ts_readback_ready: [Arc<AtomicBool>; 2],
    pub ts_readback_mapped: [Arc<AtomicBool>; 2],
    pub ts_frame: usize,
    pub ts_period_ns: f32,
    // Postprocess-chain profiler (encoder-level timestamps, dynamic slot count)
    pub pp_ts_enabled: bool,
    pub pp_query_set: wgpu::QuerySet,
    pub pp_resolve_buf: wgpu::Buffer,
    pub pp_readback: [wgpu::Buffer; 2],
    pub pp_readback_ready: [Arc<AtomicBool>; 2],
    pub pp_readback_mapped: [Arc<AtomicBool>; 2],
    /// `pp_labels` snapshotted per readback slot, so the async readback two
    /// frames later can still name what it measured.
    pub pp_labels_pending: [Vec<&'static str>; 2],
    /// Last completed readback: (label, milliseconds), in encode order.
    pub pp_times_ms: Vec<(&'static str, f32)>,
    // Fragment-shader invocations in the interval raster pass (None when the
    // adapter lacks PIPELINE_STATISTICS_QUERY).
    pub frag_query_set: Option<wgpu::QuerySet>,
    pub frag_resolve_buf: wgpu::Buffer,
    pub frag_readback: wgpu::Buffer,
    pub frag_ready: Arc<AtomicBool>,
    pub frag_mapped: Arc<AtomicBool>,
    /// Last completed readback of the fragment-invocation counter.
    pub frag_invocations: u64,
    pub gpu_times_ms: GpuTimings,
    pub cpu_times_ms: CpuTimings,
    pub instance_count_ready: Arc<AtomicBool>,
    pub instance_count_mapped: Arc<AtomicBool>,
    // PBR aux data (per-tet material channels, uploaded when PBR data present)
    pub aux_data_buf: Option<wgpu::Buffer>,
}
