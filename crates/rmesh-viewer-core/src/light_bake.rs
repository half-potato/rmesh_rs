//! Automatic light baking; no lighting work is done while lights are unchanged.
use rmesh_data::{MaterialBuffers, PbrData, SceneBuffers, SceneData, ShCoeffs};
use rmesh_interact::{Primitive, PrimitiveKind};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Light {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub direction: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    offset: u32,
    count: u32,
    lights: u32,
    tet_count: u32,
    control: [u32; 4],
    lamps: [Light; 16],
}

/// Upload the common color-only material state: dark radiance, zero gradients,
/// learned normals for baking, and the projection shader's baked-mode sentinel.
pub fn upload_material(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneData,
    pbr: Option<&PbrData>,
    buffers: &SceneBuffers,
) -> (MaterialBuffers, wgpu::Buffer) {
    let zeros = vec![0.0f32; scene.tet_count as usize * 3];
    let material = MaterialBuffers::upload(device, &zeros, &zeros, scene.tet_count);
    let sentinel = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("baked_radiance_mode"),
        contents: bytemuck::bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    if let Some(pbr) = pbr {
        let normals = if pbr.energy_material.is_empty() {
            &pbr.vertex_normals
        } else {
            &pbr.geom_normals
        };
        if !normals.is_empty() {
            queue.write_buffer(&buffers.vertex_normals, 0, bytemuck::cast_slice(normals));
        }
    }
    (material, sentinel)
}

pub struct DiffuseBake {
    pipeline: wgpu::ComputePipeline,
    combine: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    combine_bg: wgpu::BindGroup,
    params: wgpu::Buffer,
    pending: wgpu::Buffer,
    _irradiance: wgpu::Buffer,
    cached: Vec<Light>,
    offset: u32,
    count: u32,
    tet_count: u32,
    initialized: bool,
    pub generation: u64,
    started: web_time::Instant,
    shadow_epsilon: f32,
}

pub fn default_light(scene: &SceneData) -> Primitive {
    let mut p = Primitive::new(PrimitiveKind::PointLight, "Scene light");
    // Match the viewer's -Z up convention: a light above the subject and
    // toward the starting camera makes its visible side readable.
    let camera = glam::Vec3::from_slice(&scene.start_pose[..3]);
    let distance = camera.length().max(1.0);
    p.transform.position = glam::Vec3::new(camera.x * 0.45, camera.y * 0.45, -distance * 0.3);
    p.transform.scale = glam::Vec3::splat(0.06);
    p.color = Some([1.0, 0.88, 0.72, (distance * distance * 0.5).max(1.0)]);
    if let Ok(v) = std::env::var("RMESH_LIGHT_POSITION") {
        let c: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
        if c.len() == 3 {
            p.transform.position = glam::Vec3::from_slice(&c);
        }
    }
    if let Ok(v) = std::env::var("RMESH_LIGHT_INTENSITY") {
        if let Ok(i) = v.parse::<f32>() {
            p.color.as_mut().unwrap()[3] = i.max(0.0);
        }
    }
    p
}

impl DiffuseBake {
    pub fn new(
        device: &wgpu::Device,
        scene: &SceneData,
        sh: &ShCoeffs,
        pbr: Option<&PbrData>,
        buffers: &SceneBuffers,
    ) -> Self {
        let now = web_time::Instant::now();
        let adjacency =
            rmesh_raytrace::compute_tet_neighbors(&scene.indices, scene.tet_count as usize);
        let nc = ((sh.degree + 1) * (sh.degree + 1)) as usize;
        let material: Vec<[f32; 4]> = (0..scene.tet_count as usize)
            .map(|i| {
                let rgb = std::array::from_fn::<_, 3, _>(|c| {
                    if let Some(p) = pbr.filter(|p| p.albedo.len() == scene.tet_count as usize * 3)
                    {
                        p.albedo[i * 3 + c]
                    } else {
                        // Legacy SH stores lit appearance, not intrinsic albedo.
                        // Its DC color is an explicit approximation for relighting.
                        let x = sh.coeffs[i * nc * 3 + c * nc] * 0.2820948;
                        let value = if x > 2.0 {
                            x
                        } else {
                            (10.0 * x).exp().ln_1p() * 0.1
                        };
                        value.clamp(0.0, 1.0)
                    }
                });
                [
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    pbr.and_then(|p| p.roughness.get(i))
                        .copied()
                        .unwrap_or(1.0)
                        .clamp(0.22, 1.0),
                ]
            })
            .collect();
        let mut anchor = vec![u32::MAX; scene.vertex_count as usize];
        let mut roughness = vec![[0.0f32; 2]; scene.vertex_count as usize];
        for (t, ids) in scene.indices.chunks_exact(4).enumerate() {
            for &v in ids {
                let idx = v as usize;
                if anchor[idx] == u32::MAX
                    || scene.densities[t] < scene.densities[anchor[idx] as usize]
                {
                    anchor[idx] = t as u32;
                }
                roughness[idx][0] += material[t][3];
                roughness[idx][1] += 1.0;
            }
        }
        // Morton order groups nearby shadow rays in each GPU workgroup. Keep
        // original vertex IDs so geometry, interpolation and output stay intact.
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in scene.vertices.chunks_exact(3) {
            for c in 0..3 {
                lo[c] = lo[c].min(v[c]);
                hi[c] = hi[c].max(v[c]);
            }
        }
        let spatial_key = |i: usize| {
            let mut key = 0u32;
            for c in 0..3 {
                let q = (((scene.vertices[3 * i + c] - lo[c]) / (hi[c] - lo[c]).max(1e-20))
                    .clamp(0.0, 1.0)
                    * 1023.0) as u32;
                for bit in 0..10 {
                    key |= ((q >> bit) & 1) << (bit * 3 + c);
                }
            }
            key
        };
        let mut meta: Vec<[u32; 4]> = anchor
            .iter()
            .zip(&roughness)
            .enumerate()
            .map(|(i, (&a, r))| {
                [
                    a,
                    (r[0] / r[1].max(1.0)).to_bits(),
                    i as u32,
                    spatial_key(i),
                ]
            })
            .collect();
        meta.sort_unstable_by_key(|m| (m[3], m[2]));
        let upload = |label, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        // Re-entry from empty space only needs exposed faces, not every
        // internal face in the tetrahedralization.
        let bvh = rmesh_raytrace::build_boundary_bvh_sah(&scene.vertices, &scene.indices, &adjacency, scene.tet_count as usize);
        let mut packed = vec![bvh.nodes.len() as u32, 0, 0, 0];
        packed.extend_from_slice(bytemuck::cast_slice(&bvh.nodes));
        packed.extend_from_slice(&bvh.boundary_faces);
        let shadow_geometry = upload("bake_shadow_geometry", bytemuck::cast_slice(&packed));
        let shadow_epsilon = ((glam::Vec3::from_array(hi)-glam::Vec3::from_array(lo)).length()*1e-7).max(1e-7);
        let vertex_meta = upload("bake_vertex_meta", bytemuck::cast_slice(&meta));
        let neighbors = upload("bake_neighbors", bytemuck::cast_slice(&adjacency));
        let albedo = upload("bake_albedo", bytemuck::cast_slice(&material));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pending = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake_radiance"),
            size: scene.tet_count as u64 * 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let irradiance = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("baked_irradiance"),
            size: scene.vertex_count as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light_bake"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("wgsl/bake_visibility.wgsl"),
                    include_str!("wgsl/shadow_visibility.wgsl"),
                    include_str!("wgsl/light_bake.wgsl")
                )
                .into(),
            ),
        });
        let make_pipeline = |entry: &str, bindings: &[u32]| {
            let entries = bindings
                .iter()
                .map(|&i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if i == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: i != 6 && i != 7,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(entry),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[&layout],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let resources = [
                &params,
                &buffers.vertices,
                &buffers.indices,
                &buffers.densities,
                &neighbors,
                &albedo,
                &irradiance,
                &pending,
                &buffers.vertex_normals,
                &vertex_meta,
                &shadow_geometry,
            ];
            let entries = bindings
                .iter()
                .map(|&i| wgpu::BindGroupEntry {
                    binding: i,
                    resource: resources[i as usize].as_entire_binding(),
                })
                .collect::<Vec<_>>();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &layout,
                entries: &entries,
            });
            (pipeline, bg)
        };
        let (pipeline, bg) = make_pipeline("bake", &[0, 1, 2, 3, 4, 6, 8, 9, 10]);
        let (combine, combine_bg) = make_pipeline("combine", &[0, 2, 5, 6, 7]);
        log::info!(
            "Light bake geometry prepared in {:.2}s",
            now.elapsed().as_secs_f32()
        );
        Self {
            pipeline,
            combine,
            bg,
            combine_bg,
            params,
            pending,
            _irradiance: irradiance,
            cached: Vec::new(),
            offset: 0,
            count: scene.vertex_count,
            tet_count: scene.tet_count,
            initialized: false,
            generation: 0,
            started: web_time::Instant::now(),
            shadow_epsilon,
        }
    }
    pub fn invalidate(&mut self) {
        self.initialized = false;
    }
    pub fn ready(&self) -> bool {
        self.initialized && self.offset == self.count
    }
    pub fn progress(&self) -> f32 {
        self.offset as f32 / self.count.max(1) as f32
    }
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        material: &MaterialBuffers,
        primitives: &[Primitive],
    ) {
        self.record_batch(queue, encoder, material, primitives, 16384);
    }

    fn record_batch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        material: &MaterialBuffers,
        primitives: &[Primitive],
        batch_limit: u32,
    ) {
        let lights: Vec<Light> = primitives
            .iter()
            .filter(|p| matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight))
            .filter(|p| p.color.unwrap_or([1.0, 1.0, 1.0, 3.0])[3] > 0.0)
            .take(16)
            .map(|p| {
                let dir = p.transform.rotation * glam::Vec3::NEG_Z;
                Light {
                    position: [
                        p.transform.position.x,
                        p.transform.position.y,
                        p.transform.position.z,
                        if p.kind == PrimitiveKind::SpotLight {
                            1.0
                        } else {
                            0.0
                        },
                    ],
                    color: p.color.unwrap_or([1.0, 1.0, 1.0, 3.0]),
                    direction: [dir.x, dir.y, dir.z, 0.85],
                }
            })
            .collect();
        // A single light's transport is independent of its RGB/intensity.
        // Keep an in-flight snapshot fixed, including its color/intensity.
        let reuse_transport = self.ready()
            && lights.len() == 1
            && self.cached.len() == 1
            && lights[0].position == self.cached[0].position
            && lights[0].direction == self.cached[0].direction;
        let recolor = reuse_transport && lights != self.cached;
        if recolor {
            self.cached = lights.clone();
        }
        // Moving/fading lights must not starve an incremental bake. Finish
        // this snapshot, then sample the newest lights on the following frame.
        if !self.initialized || (self.ready() && lights != self.cached) {
            self.cached = lights;
            self.offset = 0;
            self.initialized = true;
            self.started = web_time::Instant::now();
            log::info!("Rebaking {} lights", self.cached.len());
        }
        if self.offset == self.count && !recolor {
            return;
        }
        if self.cached.is_empty() {
            encoder.clear_buffer(&material.base_colors, 0, None);
            self.offset = self.count;
            self.generation += 1;
            return;
        }
        let mut p = Params {
            offset: self.offset,
            count: self.count,
            lights: self.cached.len() as u32,
            tet_count: self.tet_count,
            control: [self.shadow_epsilon.to_bits(), 4096, 0, 0],
            lamps: [Light::default(); 16],
        };
        p.lamps[..self.cached.len()].copy_from_slice(&self.cached);
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let batch = (self.count - self.offset).min(batch_limit);
        if batch > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bake lighting batch"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg, &[]);
            pass.dispatch_workgroups(batch.div_ceil(64), 1, 1);
        }
        self.offset += batch;
        if self.offset == self.count {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("combine baked lighting"),
                    ..Default::default()
                });
                pass.set_pipeline(&self.combine);
                pass.set_bind_group(0, &self.combine_bg, &[]);
                let groups = self.tet_count.div_ceil(64);
                pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
            }
            encoder.copy_buffer_to_buffer(
                &self.pending,
                0,
                &material.base_colors,
                0,
                self.tet_count as u64 * 12,
            );
            self.generation += 1;
            log::info!(
                "Light bake {} complete in {:.2}s",
                self.generation,
                self.started.elapsed().as_secs_f32()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmesh_util::test_util::{build_test_scene, create_test_device_default, read_buffer};
    #[test]
    fn automatic_bake_is_dark_without_lights_and_rebakes_shadows() {
        let Some((device, queue)) = create_test_device_default() else {
            eprintln!("No GPU: skipping light bake test");
            return;
        };
        // Two tetrahedra sharing x=0. Every ray from the left cell to the
        // lamp inside the right cell traverses that controllable absorbing cell.
        let scene = build_test_scene(
            vec![
                -1.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0,
            ],
            vec![0, 1, 2, 3, 4, 1, 2, 3],
            vec![0.0, 0.0],
            vec![0.0; 6],
        );
        let sh = ShCoeffs {
            degree: 0,
            coeffs: vec![2.0; 6],
        };
        let buffers = SceneBuffers::upload(&device, &queue, &scene);
        let material = MaterialBuffers::upload(&device, &[0.0; 6], &[0.0; 6], 2);
        let mut bake = DiffuseBake::new(&device, &scene, &sh, None, &buffers);
        let run = |bake: &mut DiffuseBake, lights: &[Primitive]| {
            let mut enc = device.create_command_encoder(&Default::default());
            bake.record(&queue, &mut enc, &material, lights);
            queue.submit([enc.finish()]);
            read_buffer::<f32>(&device, &queue, &material.base_colors, 6)
        };
        assert_eq!(run(&mut bake, &[]), vec![0.0; 6]);
        let mut light = default_light(&scene);
        light.transform.position = glam::Vec3::new(2.0, 0.0, 0.0);
        light.color = Some([1.0, 1.0, 1.0, 3.0]);
        let lit = run(&mut bake, &[light.clone()]);
        assert!(lit[0] > 0.05, "unobstructed cell should be lit: {lit:?}");
        let generation = bake.generation;
        let same = run(&mut bake, &[light.clone()]);
        assert_eq!(
            generation, bake.generation,
            "unchanged lights must not rebake"
        );
        assert_eq!(lit, same);
        light.color.as_mut().unwrap()[3] *= 2.0;
        let brighter = run(&mut bake, &[light.clone()]);
        assert_eq!(bake.generation, generation + 1);
        assert!((brighter[0] - 2.0 * lit[0]).abs() < 1e-5);
        // Recoloring must reuse transport even when a channel was previously zero.
        light.color = Some([0.0, 0.5, 1.0, 6.0]);
        let colored = run(&mut bake, &[light.clone()]);
        assert_eq!(colored[0], 0.0);
        assert!((colored[1] - brighter[1] * 0.5).abs() < 1e-5);
        light.color = Some([1.0, 1.0, 1.0, 6.0]);
        assert_eq!(run(&mut bake, &[light.clone()]), brighter);
        queue.write_buffer(&buffers.densities, 0, bytemuck::cast_slice(&[0.0f32, 40.0]));
        // Changing density without invalidation distinguishes a cached recolor
        // from retracing. Explicit invalidation below must then update shadows.
        light.color.as_mut().unwrap()[3] = 3.0;
        let cached = run(&mut bake, &[light.clone()]);
        assert!((cached[0] - lit[0]).abs() < 1e-5);
        light.color.as_mut().unwrap()[3] = 6.0;
        bake.invalidate();
        let shadow = run(&mut bake, &[light.clone()]);
        assert!(
            shadow[0] < brighter[0] * 0.01,
            "occluding volume should shadow receiver: {shadow:?} vs {brighter:?}"
        );
        light.transform.position = glam::Vec3::new(-2.0, 0.0, 0.0);
        let moved = run(&mut bake, &[light.clone()]);
        assert!(
            moved[0] > shadow[0] + 0.05,
            "moving light to receiver side should restore light"
        );
        assert_eq!(run(&mut bake, &[]), vec![0.0; 6]);

        // A moving torch must finish the original snapshot over three batches,
        // leaving the previous result visible until the entire bake is ready.
        let generation = bake.generation;
        for frame in 0..3 {
            let mut encoder = device.create_command_encoder(&Default::default());
            bake.record_batch(&queue, &mut encoder, &material, &[light.clone()], 2);
            queue.submit([encoder.finish()]);
            let output = read_buffer::<f32>(&device, &queue, &material.base_colors, 6);
            if frame < 2 {
                assert_eq!(output, vec![0.0; 6]);
                assert_eq!(bake.generation, generation);
            } else {
                assert_eq!(
                    output, moved,
                    "all batches must use the original light snapshot"
                );
                assert_eq!(bake.generation, generation + 1);
            }
            light.transform.position.x -= 1.0;
            light.color.as_mut().unwrap()[3] *= 0.8;
        }
        // The next bake catches the latest position, including its final fade.
        let latest = run(&mut bake, &[light.clone()]);
        assert_ne!(latest, moved);
        bake.invalidate();
        assert_eq!(run(&mut bake, &[light]), latest);
        assert_eq!(run(&mut bake, &[]), vec![0.0; 6]);
    }
}

/// Shared diffuse lighting path for both frontends.
pub struct LightBake {
    inner: DiffuseBake,
    pub generation: u64,
}
impl LightBake {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, scene: &SceneData,
        sh: &ShCoeffs, pbr: Option<&PbrData>, buffers: &SceneBuffers) -> Self {
        Self { inner: DiffuseBake::new(device, scene, sh, pbr, buffers), generation: 0 }
    }
    pub fn configure_interval(&self, device: &wgpu::Device,
        pipelines: &mut rmesh_render::ComputeIntervalPipelines) {
        pipelines.color_only_override = None;
        pipelines.set_point_samples(device, false);
    }
    pub fn invalidate(&mut self) { self.inner.invalidate(); }
    pub fn ready(&self) -> bool { self.inner.ready() }
    pub fn progress(&self) -> f32 { self.inner.progress() }
    pub fn record(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder,
        material: &MaterialBuffers, primitives: &[Primitive], _eye: [f32; 3]) {
        self.inner.record(queue, encoder, material, primitives);
        self.generation = self.inner.generation;
    }
}
