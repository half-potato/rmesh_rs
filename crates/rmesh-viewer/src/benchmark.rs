//! Reproducible, GPU-completed wall-clock measurements of the actual viewer.
use crate::App;
use std::{io::Write, time::Instant};

pub(crate) struct Benchmark {
    path: std::path::PathBuf,
    warmup: usize,
    samples: Vec<f64>,
}
impl Benchmark {
    pub fn from_env() -> Option<Self> {
        std::env::var_os("RMESH_BENCHMARK").map(|path| Self {
            path: path.into(),
            warmup: 60,
            samples: Vec::new(),
        })
    }
    pub fn after_frame(&mut self, app: &mut App, start: Instant) -> bool {
        let Some(gpu) = app.gpu.as_mut() else {
            return false;
        };
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("benchmark GPU completion");
        let ms = start.elapsed().as_secs_f64() * 1000.;
        if !gpu.light_bake.ready() {
            return false;
        }
        if self.warmup > 0 {
            self.warmup -= 1;
            return false;
        }
        self.samples.push(ms);
        if self.samples.len() < 300 {
            return false;
        }
        let mut out =
            std::io::BufWriter::new(std::fs::File::create(&self.path).expect("benchmark output"));
        writeln!(out, "operation,width,height,sample,ms,batches").unwrap();
        let (w, h) = (gpu.surface_config.width, gpu.surface_config.height);
        for (i, ms) in self.samples.iter().enumerate() {
            writeln!(out, "render,{w},{h},{i},{ms:.6},0").unwrap();
        }
        let lamp = app
            .primitives
            .iter()
            .position(|p| {
                matches!(
                    p.kind,
                    rmesh_interact::PrimitiveKind::PointLight
                        | rmesh_interact::PrimitiveKind::SpotLight
                )
            })
            .expect("benchmark requires a scene light");
        let original = app.primitives[lamp].clone();
        for operation in ["relight_full", "relight_cached"] {
            for i in 0..11 {
                if operation == "relight_full" {
                    // Alternate two fixed positions; invalidate explicitly to force full transport.
                    app.primitives[lamp].transform.position = original.transform.position
                        + glam::Vec3::X * if i % 2 == 0 { 0.1 } else { -0.1 };
                    gpu.light_bake.invalidate();
                } else {
                    app.primitives[lamp].color.as_mut().expect("light color")[3] *=
                        if i % 2 == 0 { 1.1 } else { 1.0 / 1.1 };
                }
                let start = Instant::now();
                let mut batches = 0;
                loop {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    gpu.light_bake.record(
                        &gpu.queue,
                        &mut encoder,
                        &gpu.material_buffers,
                        &app.primitives,
                        app.camera.position.to_array(),
                    );
                    gpu.queue.submit([encoder.finish()]);
                    gpu.device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .expect("bake completion");
                    batches += 1;
                    if gpu.light_bake.ready() {
                        break;
                    }
                    assert!(batches < 100_000, "bake did not converge");
                }
                let ms = start.elapsed().as_secs_f64() * 1000.;
                // First update of each category warms its pipelines/caches.
                if i > 0 {
                    writeln!(out, "{operation},{w},{h},{},{ms:.6},{batches}", i - 1).unwrap();
                }
            }
        }
        out.flush().unwrap();
        log::info!(
            "Benchmark saved: {} ({} vertices, {} tets)",
            self.path.display(),
            app.scene_data.vertex_count,
            app.scene_data.tet_count
        );
        true
    }
}
