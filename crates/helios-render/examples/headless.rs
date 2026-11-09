//! The smallest complete Helios program: build a scene, render it offscreen, save a PNG.
//!
//! ```text
//! cargo run -p helios-render --release --example headless -- out.png
//! ```
//!
//! Everything the demo does goes through the same API: meshes are baked into virtual geometry
//! (`build_virtual_mesh`), textures are procedural or imported images, and `Renderer::render`
//! draws a frame. Temporal effects (TAA, Lumen GI, clouds) converge over a few dozen frames.

use glam::{Mat4, Quat, Vec3};
use helios_render::geometry::{build_virtual_mesh, primitives, BuildConfig, SdfMode};
use helios_render::{Camera, CloudLayer, FogVolume, GpuContext, Light, Material, MaterialKind, Renderer, Scene, Sun, TextureDesc, TextureMapping, TexturePattern};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "helios.png".into());
    let ctx = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)).expect("no WebGPU-capable adapter");

    // ---- Scene ----------------------------------------------------------------------------------
    let mut scene = Scene::new();
    // Terrain gets a heightfield distance field (Lumen's landscape representation): the default
    // volume field would turn a 400 m open sheet into a slab metres thick that swallows lights.
    let terrain_sdf = BuildConfig { sdf: SdfMode::Heightfield { resolution: 512 }, ..Default::default() };
    let ground = scene.add_mesh(build_virtual_mesh(&primitives::terrain(400.0, 128, 6.0, 7), &terrain_sdf));
    let rock = scene.add_mesh(build_virtual_mesh(&primitives::rock(3, 6), &BuildConfig::default()));

    let rock_texture = scene.add_texture(TextureDesc::new(TexturePattern::Rock, Vec3::new(0.34, 0.31, 0.27), Vec3::new(0.16, 0.15, 0.13)));
    let grass_texture = scene.add_texture(TextureDesc::new(TexturePattern::Grass, Vec3::new(0.16, 0.24, 0.06), Vec3::new(0.08, 0.12, 0.03)));
    let rock_material = scene.add_material(Material::textured(MaterialKind::Rock, rock_texture, TextureMapping::ObjectTriplanar, 0.6));
    let grass_material = scene.add_material(Material::textured(MaterialKind::Standard, grass_texture, TextureMapping::WorldTriplanar, 0.25));

    scene.add_instance(ground, grass_material, Mat4::IDENTITY);
    for (i, (x, z, s)) in [(0.0, 0.0, 3.0), (6.0, -3.0, 1.6), (-5.0, 2.5, 1.2), (2.5, 6.0, 0.9)].into_iter().enumerate() {
        let transform = Mat4::from_scale_rotation_translation(Vec3::splat(s), Quat::from_rotation_y(i as f32 * 1.3), Vec3::new(x, s * 0.4, z));
        scene.add_instance(rock, rock_material, transform);
    }
    scene.add_light(Light::point(Vec3::new(1.5, 1.2, 4.0), Vec3::new(1.0, 0.6, 0.3), 30.0, 12.0));
    scene.sun = Sun::from_angles(0.8, 0.35, 10.0);
    scene.clouds = CloudLayer::cumulus(0.45);
    scene.fog = FogVolume { density: 0.004, ..Default::default() };

    // ---- Render -----------------------------------------------------------------------------------
    let (width, height) = (1280, 720);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut renderer = Renderer::new(&ctx.device, &ctx.queue, format, width, height);
    let mut camera = Camera { position: Vec3::new(-12.0, 3.0, 12.0), ..Default::default() };
    camera.look_at(Vec3::new(0.0, 1.0, 0.0));
    for _ in 0..90 {
        renderer.render(&scene, &camera, &view, 1.0 / 60.0);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    // ---- Read back and save ---------------------------------------------------------------------------
    let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) } },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer.get_mapped_range(..).expect("mapped");
    let pixels: Vec<u8> = data.chunks(row as usize).flat_map(|r| &r[..(width * 4) as usize]).copied().collect();
    let mut png = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&path).expect("create file")), width, height);
    png.set_color(png::ColorType::Rgba);
    png.set_depth(png::BitDepth::Eight);
    png.write_header().and_then(|mut w| w.write_image_data(&pixels)).expect("write png");
    println!("{}", renderer.stats().triangles);
    println!("saved {path}");
}
