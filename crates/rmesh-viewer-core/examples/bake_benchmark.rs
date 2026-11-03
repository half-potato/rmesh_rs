//! Usage: cargo run --release -p rmesh-viewer-core --example bake_benchmark -- scene.rmesh
use rmesh_viewer_core::light_bake::{default_light, upload_material, LightBake};
fn main() {
    let bytes = std::fs::read(std::env::args().nth(1).expect("scene path")).unwrap();
    let (scene, sh, pbr) = rmesh_data::load_rmesh(&bytes).unwrap();
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 8,
                    ..adapter.limits()
                },
                ..Default::default()
            })
            .await
            .unwrap()
    });
    let buffers = rmesh_data::SceneBuffers::upload(&device, &queue, &scene);
    let (material, _) = upload_material(&device, &queue, &scene, pbr.as_ref(), &buffers);
    let start = std::time::Instant::now();
    let mut bake = LightBake::new(&device, &queue, &scene, &sh, pbr.as_ref(), &buffers);
    eprintln!("prepare_ms={:.1}", start.elapsed().as_secs_f64() * 1000.);
    let lights = [default_light(&scene)];
    for run in 0..2 {
        bake.invalidate();
        let start = std::time::Instant::now();
        let mut batches = 0;
        let mut peak = 0f64;
        while !bake.ready() {
            let batch = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            bake.record(&queue, &mut encoder, &material, &lights, [0.; 3]);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            peak = peak.max(batch.elapsed().as_secs_f64() * 1000.);
            batches += 1;
        }
        println!(
            "run={run} vertices={} tets={} batches={batches} bake_ms={:.1} max_batch_ms={peak:.1}",
            scene.vertex_count,
            scene.tet_count,
            start.elapsed().as_secs_f64() * 1000.
        );
    }
    if let Some(path) = std::env::args().nth(2) {
        let colors = rmesh_util::test_util::read_buffer::<f32>(
            &device,
            &queue,
            &material.base_colors,
            scene.tet_count as usize * 3,
        );
        std::fs::write(path, bytemuck::cast_slice(&colors)).unwrap();
    }
}
