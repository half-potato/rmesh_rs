//! Cell-corner viewer approximation with consistent diffuse/specular materials.
//! Lighting publication refreshes diffuse reflection probes; camera motion shades corners.
use glam::Vec3;
use rmesh_data::{PbrData, SceneBuffers, SceneData};
use wgpu::util::DeviceExt;
const PROBE_SIZE: u32 = 64;
const PROBE_LEVELS: u32 = 7;
const PROBE_COUNT: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Frame {
    eye: [f32; 4],
    probe: [f32; 4],
    counts: [u32; 4],
    components: [f32; 4],
    probes: [[f32; 4]; 5],
    influence: [f32; 4],
}
struct Pass {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    dispatch: [u32; 3],
}
impl Pass {
    fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vertex probe lighting"),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.dispatch[0], self.dispatch[1], self.dispatch[2]);
    }
}
fn buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage,
    })
}
fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
fn make_pass(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
    resources: &[(u32, wgpu::BindingResource<'_>)],
    dispatch: [u32; 3],
) -> Pass {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let entries: Vec<_> = resources
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    Pass {
        pipeline,
        group,
        dispatch,
    }
}

fn point_in_cell(scene: &SceneData, t: usize, p: Vec3) -> bool {
    let ids = &scene.indices[t * 4..t * 4 + 4];
    let q = |i| Vec3::from_slice(&scene.vertices[3 * ids[i] as usize..3 * ids[i] as usize + 3]);
    let a = q(0);
    let b = q(1) - a;
    let c = q(2) - a;
    let d = q(3) - a;
    let det = b.dot(c.cross(d));
    if det.abs() < 1e-25 {
        return false;
    }
    let v = p - a;
    let w = Vec3::new(v.dot(c.cross(d)), v.dot(d.cross(b)), v.dot(b.cross(c))) / det;
    w.min_element() >= -1e-5 && w.element_sum() <= 1.00001
}
/// One exported camera plus four local offsets. Avoid dense cells and use dense
/// scene bounds rather than the potentially enormous empty background tets.
fn probe_positions(scene: &SceneData, lo: Vec3, hi: Vec3) -> ([Vec3; 5], f32) {
    let camera = Vec3::from_slice(&scene.start_pose[..3]);
    let alpha = crate::flare::tet_alphas(
        &scene.vertices,
        &scene.indices,
        &scene.densities,
        scene.tet_count as usize,
    );
    let mut dense_lo = Vec3::splat(f32::INFINITY);
    let mut dense_hi = Vec3::splat(f32::NEG_INFINITY);
    let mut nearest_surface = f32::INFINITY;
    for (t, ids) in scene.indices.chunks_exact(4).enumerate() {
        if alpha[t] > 0.1 {
            for &i in ids {
                let p = Vec3::from_slice(&scene.vertices[3 * i as usize..3 * i as usize + 3]);
                nearest_surface = nearest_surface.min(p.distance(camera));
                dense_lo = dense_lo.min(p);
                dense_hi = dense_hi.max(p);
            }
        }
    }
    if !dense_lo.is_finite() {
        dense_lo = lo;
        dense_hi = hi;
    }
    let extent = (dense_hi - dense_lo).length().max(1e-3);
    let center = (dense_lo + dense_hi) * 0.5;
    let forward = (center - camera).try_normalize().unwrap_or(Vec3::Z);
    let axis = if forward.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let right = forward.cross(axis).normalize();
    let up = right.cross(forward).normalize();
    // Background cells can have enormous bounds. Keep offsets local to the
    // capture camera using its nearest dense vertex as the spacing scale.
    let spread = (nearest_surface * 0.5).min(extent * 0.15).max(1e-4);
    let targets = [
        camera,
        camera + right * spread,
        camera - right * spread,
        camera + up * spread,
        camera - up * spread,
    ];
    let exterior = hi + Vec3::splat((hi - lo).length().max(1.0) * 0.05);
    let safe = |p: Vec3| {
        p.is_finite()
            && !(0..scene.tet_count as usize).any(|t| alpha[t] > 0.01 && point_in_cell(scene, t, p))
    };
    let positions = targets.map(|target| {
        // Prefer short moves back toward the camera before falling back outside.
        for step in 0..=16 {
            let p = target.lerp(camera, step as f32 / 16.0);
            if safe(p) {
                return p;
            }
        }
        for step in 1..=16 {
            let p = target.lerp(exterior, step as f32 / 16.0);
            if safe(p) {
                return p;
            }
        }
        exterior
    });
    let radius = (extent * 0.75).max(camera.distance(center) * 1.5).max(1e-3);
    (positions, radius)
}

pub struct EnergyForward {
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    frame: wgpu::Buffer,
    frame_data: Frame,
    diffuse: Pass,
    capture: Pass,
    visibility: Pass,
    visibility_ready: bool,
    filters: Vec<Pass>,
    shade: Pass,
    last: Option<([f32; 3], u64)>,
    // Keep available for GPU regression readback.
    #[allow(dead_code)] // Used by GPU regression readback.
    pub(crate) radiance: wgpu::Buffer,
    pub(crate) probe_raw: wgpu::Texture,
    #[cfg(test)]
    weights: wgpu::Buffer,
}
impl EnergyForward {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &SceneData,
        pbr: &PbrData,
        buffers: &SceneBuffers,
        transport: &wgpu::Buffer,
        snapshot: &wgpu::Buffer,
        neighbors: &wgpu::Buffer,
        shadow_geometry: &wgpu::Buffer,
        _adjacency: &[i32],
    ) -> Self {
        let mut cell_materials = Vec::<f32>::with_capacity(scene.tet_count as usize * 8);
        for (i, m) in pbr.energy_material.chunks_exact(5).enumerate() {
            cell_materials.extend_from_slice(&pbr.albedo[3 * i..3 * i + 3]);
            cell_materials.extend_from_slice(m);
        }
        let cell_materials = buffer(
            device,
            "cell diffuse materials",
            bytemuck::cast_slice(&cell_materials),
            wgpu::BufferUsages::STORAGE,
        );
        let diffuse = storage(
            device,
            "baked cell diffuse/material",
            scene.tet_count as u64 * 32,
        );
        let radiance = storage(
            device,
            "material-consistent cell-corner radiance",
            scene.tet_count as u64 * 32,
        );
        let lut = crate::energy::upload_bytes(device, queue, bytemuck::cast_slice(&pbr.energy_lut));
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in scene.vertices.chunks_exact(3) {
            let p = Vec3::from_slice(p);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let (probes, radius) = probe_positions(scene, lo, hi);
        let probe = probes[0];
        let diameter = (hi - lo).length().max(1e-3);
        log::info!(
            "Vertex reflection probes at {probes:?}, radius {radius}, {PROBE_SIZE}px, {PROBE_LEVELS} GGX levels"
        );
        // Internal contribution captures; default preserves all lighting.
        let components = match std::env::var("RMESH_PROBE_COMPONENTS").as_deref() {
            Ok("no-specular") => [1., 0., 0., 0.],
            Ok("no-probe") => [1., 1., 0., 0.],
            Ok("probe-only") => [0., 0., 1., 0.],
            Ok("all") | Err(_) => [1., 1., 1., 0.],
            Ok(other) => panic!("unknown RMESH_PROBE_COMPONENTS: {other}"),
        };
        // Half-intensity distance in scene units; zero is the matched no-falloff control.
        let falloff_distance = std::env::var("RMESH_PROBE_FALLOFF_DISTANCE")
            .ok().map(|v| v.parse::<f32>().expect("invalid probe falloff distance"))
            .unwrap_or(1.0);
        assert!(falloff_distance.is_finite() && falloff_distance >= 0.0);
        let frame_data = Frame {
            components,
            probes: probes.map(|p| {
                [
                    p.x,
                    p.y,
                    p.z,
                    f32::from_bits(
                        (0..scene.tet_count as usize)
                            .find(|&t| point_in_cell(scene, t, p))
                            .map_or(u32::MAX, |t| t as u32),
                    ),
                ]
            }),
            influence: [radius, falloff_distance, 0., 0.],
            eye: [
                0.,
                0.,
                0.,
                diameter * 4.0 + (probe - (lo + hi) * 0.5).length(),
            ],
            probe: [probe.x, probe.y, probe.z, (diameter * 1e-7).max(1e-7)],
            counts: [
                scene.vertex_count,
                scene.tet_count,
                PROBE_LEVELS,
                (0..scene.tet_count as usize)
                    .find(|&t| point_in_cell(scene, t, probe))
                    .map_or(u32::MAX, |t| t as u32),
            ],
        };
        let frame = buffer(
            device,
            "vertex probe frame",
            bytemuck::bytes_of(&frame_data),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let texture = |label, levels| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: PROBE_SIZE,
                    height: PROBE_SIZE,
                    depth_or_array_layers: 6 * PROBE_COUNT,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let raw = texture("diffuse scene reflection probe", 1);
        let filtered = texture("GGX prefiltered reflection probe", PROBE_LEVELS);
        let cube = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::CubeArray),
                ..Default::default()
            })
        };
        let raw_cube = cube(&raw);
        let filtered_cube = cube(&filtered);
        let layer = |texture: &wgpu::Texture, mip| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let raw_output = layer(&raw, 0);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let source = format!(
            "{}\n{}",
            crate::energy::WGSL,
            include_str!("wgsl/energy_vertex.wgsl")
        );
        let groups = scene.vertex_count.div_ceil(64);
        let dispatch = [groups.min(65535), groups.div_ceil(65535), 1];
        let diffuse_pass = make_pass(
            device,
            &source,
            "bake_diffuse",
            &[
                (0, frame.as_entire_binding()),
                (1, buffers.vertices.as_entire_binding()),
                (2, buffers.vertex_normals.as_entire_binding()),
                (4, transport.as_entire_binding()),
                (5, snapshot.as_entire_binding()),
                (6, diffuse.as_entire_binding()),
                (8, wgpu::BindingResource::TextureView(&lut)),
                (11, cell_materials.as_entire_binding()),
                (12, buffers.indices.as_entire_binding()),
            ],
            [
                scene.tet_count.div_ceil(64).min(65535),
                scene.tet_count.div_ceil(64).div_ceil(65535),
                1,
            ],
        );
        let weights = storage(
            device,
            "cached probe blend weights",
            scene.vertex_count as u64 * PROBE_COUNT as u64 * 4,
        );
        let shade = make_pass(
            device,
            &source,
            "shade_corners",
            &[
                (0, frame.as_entire_binding()),
                (1, buffers.vertices.as_entire_binding()),
                (2, buffers.vertex_normals.as_entire_binding()),
                (4, transport.as_entire_binding()),
                (5, snapshot.as_entire_binding()),
                (7, radiance.as_entire_binding()),
                (8, wgpu::BindingResource::TextureView(&lut)),
                (9, wgpu::BindingResource::TextureView(&filtered_cube)),
                (10, wgpu::BindingResource::Sampler(&sampler)),
                (13, weights.as_entire_binding()),
                (11, cell_materials.as_entire_binding()),
                (12, buffers.indices.as_entire_binding()),
            ],
            [
                scene.tet_count.div_ceil(64).min(65535),
                scene.tet_count.div_ceil(64).div_ceil(65535),
                1,
            ],
        );
        let visibility = include_str!("wgsl/shadow_visibility.wgsl").replace(
            "fn normal(i:u32)->vec3f { return vec3f(normals[3*i],normals[3*i+1],normals[3*i+2]); }",
            "",
        );
        let source = format!(
            "{}\n{}\n{}",
            include_str!("wgsl/probe_cube.wgsl"),
            visibility,
            include_str!("wgsl/probe_capture.wgsl")
        );
        let capture = make_pass(
            device,
            &source,
            "capture",
            &[
                (0, frame.as_entire_binding()),
                (1, buffers.vertices.as_entire_binding()),
                (2, buffers.indices.as_entire_binding()),
                (3, buffers.densities.as_entire_binding()),
                (4, neighbors.as_entire_binding()),
                (5, diffuse.as_entire_binding()),
                (6, shadow_geometry.as_entire_binding()),
                (7, wgpu::BindingResource::TextureView(&raw_output)),
            ],
            [
                PROBE_SIZE.div_ceil(8),
                PROBE_SIZE.div_ceil(8),
                6 * PROBE_COUNT,
            ],
        );
        let visibility_pass = make_pass(
            device,
            &source,
            "cache_visibility",
            &[
                (0, frame.as_entire_binding()),
                (1, buffers.vertices.as_entire_binding()),
                (2, buffers.indices.as_entire_binding()),
                (3, buffers.densities.as_entire_binding()),
                (4, neighbors.as_entire_binding()),
                (6, shadow_geometry.as_entire_binding()),
                (8, weights.as_entire_binding()),
            ],
            dispatch,
        );
        let source = format!(
            "{}\n{}",
            include_str!("wgsl/probe_cube.wgsl"),
            include_str!("wgsl/probe_filter.wgsl")
        );
        let filters = (0..PROBE_LEVELS)
            .map(|level| {
                let size = PROBE_SIZE >> level;
                let settings = buffer(
                    device,
                    "probe filter settings",
                    bytemuck::cast_slice(&[size, level, PROBE_LEVELS, 64u32]),
                    wgpu::BufferUsages::UNIFORM,
                );
                let output = layer(&filtered, level);
                make_pass(
                    device,
                    &source,
                    "prefilter",
                    &[
                        (0, settings.as_entire_binding()),
                        (1, wgpu::BindingResource::TextureView(&raw_cube)),
                        (2, wgpu::BindingResource::Sampler(&sampler)),
                        (3, wgpu::BindingResource::TextureView(&output)),
                    ],
                    [size.div_ceil(8), size.div_ceil(8), 6 * PROBE_COUNT],
                )
            })
            .collect();
        let entries: Vec<_> = [0, 1, 3]
            .into_iter()
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 3 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interpolated vertex radiance"),
            entries: &entries,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("interpolated vertex radiance"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: radiance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frame.as_entire_binding(),
                },
            ],
        });
        Self {
            layout,
            group,
            frame,
            frame_data,
            diffuse: diffuse_pass,
            capture,
            visibility: visibility_pass,
            visibility_ready: false,
            filters,
            shade,
            last: None,
            radiance,
            probe_raw: raw,
            #[cfg(test)]
            weights,
        }
    }
    pub fn probe_positions(&self) -> [[f32; 3]; 5] {
        self.frame_data.probes.map(|p| [p[0], p[1], p[2]])
    }
    pub fn configure(
        &self,
        device: &wgpu::Device,
        pipelines: &mut rmesh_render::ComputeIntervalPipelines,
    ) {
        pipelines.set_point_samples(device, true);
        pipelines.set_color_shader(
            device,
            &self.layout,
            include_str!("wgsl/energy_interval.wgsl"),
            self.group.clone(),
        );
    }
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        eye: [f32; 3],
        generation: u64,
    ) {
        if self.last == Some((eye, generation)) {
            return;
        }
        self.frame_data.eye[..3].copy_from_slice(&eye);
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&self.frame_data));
        if !self.visibility_ready {
            self.visibility.record(encoder);
            self.visibility_ready = true;
        }
        if self.last.is_none_or(|(_, g)| g != generation) {
            self.diffuse.record(encoder);
            self.capture.record(encoder);
            for filter in &self.filters {
                filter.record(encoder);
            }
        }
        self.shade.record(encoder);
        self.last = Some((eye, generation));
    }
    pub fn invalidate(&mut self) {
        self.last = None;
        self.visibility_ready = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmesh_util::test_util::build_test_scene;
    #[test]
    fn probe_weights_preserve_occlusion_and_fade_outside_support() {
        use rmesh_util::test_util::{create_test_device_default, read_buffer};
        let (device, queue) = create_test_device_default().expect("GPU required");
        // One opaque tetrahedron, a target behind it, a clear target beside it,
        // and an out-of-range target. Extra target vertices need no incident cell.
        let scene = build_test_scene(
            vec![
                -0.5, -1., -1., 0.5, -1., -1., 0., 1., -1., 0., 0., 1., 2., 0., 0., -2., 2., 0.,
                20., 0., 0.,
            ],
            vec![0, 1, 2, 3],
            vec![100.],
            vec![0.; 3],
        );
        let buffers = SceneBuffers::upload(&device, &queue, &scene);
        let adjacency = rmesh_raytrace::compute_tet_neighbors(&scene.indices, 1);
        let bvh =
            rmesh_raytrace::build_boundary_bvh_sah(&scene.vertices, &scene.indices, &adjacency, 1);
        let mut packed = vec![bvh.nodes.len() as u32, 0, 0, 0];
        packed.extend_from_slice(bytemuck::cast_slice(&bvh.nodes));
        packed.extend_from_slice(&bvh.boundary_faces);
        let neighbors = buffer(
            &device,
            "neighbors",
            bytemuck::cast_slice(&adjacency),
            wgpu::BufferUsages::STORAGE,
        );
        let geometry = buffer(
            &device,
            "geometry",
            bytemuck::cast_slice(&packed),
            wgpu::BufferUsages::STORAGE,
        );
        let transport = storage(&device, "transport", scene.vertex_count as u64 * 64);
        let snapshot = buffer(
            &device,
            "snapshot",
            &[0u8; 1024],
            wgpu::BufferUsages::UNIFORM,
        );
        let pbr = PbrData {
            roughness: vec![0.3],
            albedo: vec![0.5; 3],
            energy_material: vec![0.3, 0.04, 0.04, 0.04, 0.],
            energy_lut: crate::energy::LUT
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
            geom_normals: vec![],
            render_min_t: None,
            env_feature: vec![],
            vertex_normals: vec![],
            tone_curve: vec![],
            metallic: vec![],
            f0_dielectric: vec![],
        };
        let mut forward = EnergyForward::new(
            &device, &queue, &scene, &pbr, &buffers, &transport, &snapshot, &neighbors, &geometry,
            &adjacency,
        );
        forward.frame_data.probes = [[-2., 0., 0., f32::from_bits(u32::MAX)]; 5];
        forward.frame_data.influence[0] = 10.;
        forward.frame_data.influence[1] = 1.;
        queue.write_buffer(&forward.frame, 0, bytemuck::bytes_of(&forward.frame_data));
        let mut encoder = device.create_command_encoder(&Default::default());
        forward.visibility.record(&mut encoder);
        queue.submit([encoder.finish()]);
        let weights = read_buffer::<f32>(
            &device,
            &queue,
            &forward.weights,
            scene.vertex_count as usize * 5,
        );
        let sum = |v: usize| weights[v * 5..v * 5 + 5].iter().sum::<f32>();
        assert!(
            sum(4) < 1e-4,
            "opaque blocker must suppress every probe: {}",
            sum(4)
        );
        assert!(
            (sum(5) - 0.2).abs() < 1e-5,
            "overlapping probes must retain distance attenuation: {}",
            sum(5)
        );
        assert_eq!(sum(6), 0., "outside influence must be dark");
        assert!(weights
            .iter()
            .all(|w| w.is_finite() && *w >= 0. && *w <= 1.));
        forward.frame_data.influence[1] = 0.;
        queue.write_buffer(&forward.frame, 0, bytemuck::bytes_of(&forward.frame_data));
        let mut encoder = device.create_command_encoder(&Default::default());
        forward.visibility.record(&mut encoder);
        queue.submit([encoder.finish()]);
        let control = read_buffer::<f32>(&device, &queue, &forward.weights, scene.vertex_count as usize * 5);
        assert!((control[25..30].iter().sum::<f32>()-1.).abs()<1e-5,
            "no-falloff control must preserve normalized blend");
        // Light/camera updates reuse visibility; explicit invalidation recomputes it.
        forward.record(
            &queue,
            &mut device.create_command_encoder(&Default::default()),
            [0.; 3],
            1,
        );
        assert!(forward.visibility_ready);
        forward.invalidate();
        assert!(!forward.visibility_ready);
    }
    #[test]
    fn probe_spacing_ignores_distant_background_bounds() {
        let mut scene = build_test_scene(
            vec![
                0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1000., 0., 0., 2000., 0., 0.,
                1000., 1000., 0., 1000., 0., 1000.,
            ],
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![10., 10.],
            vec![0.; 6],
        );
        let camera = Vec3::new(-1., 0., 0.);
        scene.start_pose[..3].copy_from_slice(&camera.to_array());
        let (positions, _) = probe_positions(&scene, Vec3::ZERO, Vec3::splat(2000.));
        assert_eq!(positions[0], camera);
        for (i, p) in positions.iter().enumerate() {
            assert!(
                p.distance(camera) <= 0.5001,
                "probe must stay near camera: {p:?}"
            );
            assert!(!point_in_cell(&scene, 0, *p));
            for q in &positions[..i] {
                assert!(p.distance(*q) > 1e-4);
            }
        }
    }
    #[test]
    fn probe_avoids_dense_interior() {
        let mut scene = build_test_scene(
            vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
            vec![0, 1, 2, 3],
            vec![10.],
            vec![0.; 3],
        );
        scene.start_pose[..3].copy_from_slice(&[0.1, 0.1, 0.1]);
        let p = probe_positions(&scene, Vec3::ZERO, Vec3::ONE).0[0];
        assert!(!point_in_cell(&scene, 0, p));
        scene.start_pose[..3].copy_from_slice(&[-1., 0., 0.]);
        assert_eq!(
            probe_positions(&scene, Vec3::ZERO, Vec3::ONE).0[0],
            Vec3::new(-1., 0., 0.)
        );
    }
}
