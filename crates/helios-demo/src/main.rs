//! Native demo: an interactive window, or headless rendering to a PNG.
//!
//! ```text
//! helios                                   interactive window
//! helios --screenshot out.png              render offscreen and save
//!     [--size 1600x900] [--frames 90] [--view 1] [--detail medium|high]
//!     [--camera x,y,z,yaw,pitch] [--screen-percentage N] [--gpu-budget-ms N]
//!     [--no-local-exposure | --local-exposure HIGHLIGHT,SHADOW[,BLEND]]
//! helios --gltf model.glb|model.gltf       a scene built from a glTF 2.0 file [--no-ground]
//! helios --bake-hvm DIR                    write the scene's baked meshes as .hvm files
//! helios --stream-from DIR                 build the scene from them, streaming pages from disk
//!     [--io-latency-ms N]                  (every read takes at least N ms)
//! ```

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"))
        .init();
    let args: Vec<String> = std::env::args().collect();
    let options = native::Options::parse(&args);
    if let Some(dir) = &options.bake_hvm {
        if let Err(e) = helios_demo::mesh_store::bake_to(std::path::Path::new(dir), options.scene, options.detail) {
            log::error!("--bake-hvm {dir}: {e}");
            std::process::exit(1);
        }
    } else if let Some(path) = options.screenshot.clone() {
        native::screenshot(&options, &path);
    } else {
        native::run_window(options);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::Arc;
    use std::time::Duration;

    use glam::{Vec2, Vec3};
    use helios_demo::app::{DemoApp, Input, HELP};
    use helios_demo::mesh_store;
    use helios_demo::scenes::model::Model;
    use helios_demo::scenes::{Detail, SceneKind};
    use helios_render::page_source::FileSource;
    use helios_render::{DebugView, GpuContext};
    use web_time::Instant;
    use winit::application::ApplicationHandler;
    use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::PhysicalKey;
    use winit::window::{Window, WindowId};

    #[derive(Clone)]
    pub struct Options {
        pub screenshot: Option<String>,
        pub size: (u32, u32),
        pub frames: u32,
        pub view: Option<usize>,
        pub detail: Detail,
        pub scene: SceneKind,
        pub camera: Option<[f32; 5]>,
        pub eye: Option<Vec3>,
        pub at: Option<Vec3>,
        pub occlusion: bool,
        pub sw_raster: bool,
        pub rvt: bool,
        /// Stream image textures at native resolution (`--no-svt`: 512² texture arrays only).
        pub svt: bool,
        pub animate_sun: bool,
        pub lod: Option<f32>,
        /// Vertical field of view in degrees.
        pub fov: Option<f32>,
        pub pool_mb: Option<u64>,
        pub texture_pool_mb: Option<u64>,
        /// Block-compress streamed texture tiles where the GPU supports BC (`--no-texture-compression`: RGBA8).
        pub texture_compression: bool,
        /// Headless: switch to this scene halfway through (exercises scene reloading).
        pub switch_to: Option<SceneKind>,
        /// Headless: save the last `dump_count` frames as PNGs into this directory.
        pub dump_dir: Option<String>,
        pub dump_count: u32,
        /// Hotkeys to press after loading (e.g. `KeyG,KeyH`), as in the interactive window.
        pub keys: Vec<String>,
        /// Headless: the frame at which `keys` are pressed instead (e.g. to toggle a feature
        /// halfway through a run).
        pub keys_at: Option<u32>,
        /// Headless: keys held down for the whole run (e.g. `KeyW` flies forward).
        pub hold: Vec<String>,
        /// glTF 2.0 file (`.gltf` or `.glb`) to build the model scene from.
        pub gltf: Option<String>,
        /// Model scene: put a ground plane under the model.
        pub ground: bool,
        /// Write the scene's baked meshes into this directory and exit.
        pub bake_hvm: Option<String>,
        /// Build the scene from the meshes in this directory, streaming pages from disk.
        pub stream_from: Option<String>,
        /// Minimum latency of every streamed read (emulates slow storage).
        pub io_latency_ms: u64,
        /// Sun azimuth and elevation in degrees (negative elevations give twilight and night).
        pub sun: Option<(f32, f32)>,
        /// Headless: the shadow-casting light nearest the camera circles every frame.
        pub move_light: bool,
        /// Internal resolution in percent of the output (TSR upscales).
        pub screen_percentage: Option<f32>,
        /// Dynamic resolution: keep the GPU frame time under this many milliseconds.
        pub gpu_budget_ms: Option<f32>,
        /// Local exposure settings (`--no-local-exposure`: `None`, global exposure only).
        pub local_exposure: Option<helios_render::LocalExposure>,
    }

    impl Options {
        pub fn parse(args: &[String]) -> Self {
            let mut o = Options { screenshot: None, size: (1600, 900), frames: 90, view: None, detail: Detail::High, scene: SceneKind::Valley, camera: None, eye: None, at: None, occlusion: true, sw_raster: true, rvt: true, svt: true, animate_sun: false, lod: None, fov: None, pool_mb: None, texture_pool_mb: None, texture_compression: true, switch_to: None, dump_dir: None, dump_count: 16, keys: Vec::new(), keys_at: None, hold: Vec::new(), gltf: None, ground: true, bake_hvm: None, stream_from: None, io_latency_ms: 0, sun: None, move_light: false, screen_percentage: None, gpu_budget_ms: None, local_exposure: Some(Default::default()) };
            let mut scene_given = false;
            let mut it = args.iter().skip(1);
            while let Some(arg) = it.next() {
                let mut value = || it.next().cloned().unwrap_or_default();
                match arg.as_str() {
                    "--screenshot" => o.screenshot = Some(value()),
                    "--size" => {
                        let v = value();
                        if let Some((w, h)) = v.split_once('x') {
                            o.size = (w.parse().unwrap_or(1600), h.parse().unwrap_or(900));
                        }
                    }
                    "--frames" => o.frames = value().parse().unwrap_or(90),
                    "--view" => o.view = value().parse().ok(),
                    "--scene" => {
                        o.scene = SceneKind::parse(&value()).unwrap_or(SceneKind::Valley);
                        scene_given = true;
                    }
                    "--gltf" => o.gltf = Some(value()),
                    "--no-ground" => o.ground = false,
                    "--detail" => o.detail = if value() == "medium" { Detail::Medium } else { Detail::High },
                    "--camera" => {
                        let v: Vec<f32> = value().split(',').filter_map(|x| x.parse().ok()).collect();
                        if v.len() == 5 {
                            o.camera = Some([v[0], v[1], v[2], v[3], v[4]]);
                        }
                    }
                    "--no-occlusion" => o.occlusion = false,
                    "--no-sw-raster" => o.sw_raster = false,
                    "--no-rvt" => o.rvt = false,
                    "--no-svt" => o.svt = false,
                    "--no-texture-compression" => o.texture_compression = false,
                    "--animate-sun" => o.animate_sun = true,
                    "--move-light" => o.move_light = true,
                    "--lod" => o.lod = value().parse().ok(),
                    "--fov" => o.fov = value().parse().ok(),
                    "--sun" => {
                        let v: Vec<f32> = value().split(',').filter_map(|x| x.parse().ok()).collect();
                        if v.len() == 2 {
                            o.sun = Some((v[0], v[1]));
                        }
                    }
                    "--pool-mb" => o.pool_mb = value().parse().ok(),
                    "--texture-pool-mb" => o.texture_pool_mb = value().parse().ok(),
                    "--switch-to" => o.switch_to = SceneKind::parse(&value()),
                    "--dump-dir" => o.dump_dir = Some(value()),
                    "--dump-count" => o.dump_count = value().parse().unwrap_or(16),
                    "--keys" => o.keys = value().split(',').map(str::to_string).collect(),
                    "--keys-at" => o.keys_at = value().parse().ok(),
                    "--hold" => o.hold = value().split(',').map(str::to_string).collect(),
                    "--eye" => o.eye = parse_vec3(&value()),
                    "--at" => o.at = parse_vec3(&value()),
                    "--bake-hvm" => o.bake_hvm = Some(value()),
                    "--stream-from" => o.stream_from = Some(value()),
                    "--io-latency-ms" => o.io_latency_ms = value().parse().unwrap_or(0),
                    "--screen-percentage" => o.screen_percentage = value().parse().ok(),
                    "--gpu-budget-ms" => o.gpu_budget_ms = value().parse().ok(),
                    "--no-local-exposure" => o.local_exposure = None,
                    // Highlight contrast, shadow contrast and optionally the blurred luminance blend.
                    "--local-exposure" => {
                        let v: Vec<f32> = value().split(',').filter_map(|x| x.parse().ok()).collect();
                        let mut settings = helios_render::LocalExposure::default();
                        if let [highlight, shadow, ref rest @ ..] = v[..] {
                            settings.highlight_contrast = highlight;
                            settings.shadow_contrast = shadow;
                            if let Some(&blend) = rest.first() {
                                settings.blurred_luminance_blend = blend;
                            }
                        }
                        o.local_exposure = Some(settings);
                    }
                    other => log::warn!("ignoring unknown argument {other}"),
                }
            }
            if o.gltf.is_some() && !scene_given {
                o.scene = SceneKind::Model;
            }
            o
        }

        /// Loads `--gltf` (exits with a message if the file cannot be imported).
        fn load_model(&self) -> Option<Model> {
            let path = self.gltf.as_ref()?;
            let start = Instant::now();
            let asset = helios_asset::load_gltf_file(path).unwrap_or_else(|e| {
                log::error!("cannot load {path}: {e}");
                std::process::exit(1);
            });
            log::info!("imported {path} in {:.2} s", start.elapsed().as_secs_f64());
            let name = std::path::Path::new(path).file_stem().map_or_else(|| path.clone(), |s| s.to_string_lossy().into_owned());
            let mut model = Model::new(name, asset);
            model.ground = self.ground;
            Some(model)
        }

        fn renderer_config(&self) -> helios_render::RendererConfig {
            let mut config = helios_render::RendererConfig::default();
            if let Some(mb) = self.pool_mb {
                config.geometry_pool_bytes = mb << 20;
            }
            if let Some(mb) = self.texture_pool_mb {
                config.texture_pool_bytes = mb << 20;
            }
            config.texture_compression = self.texture_compression;
            config
        }

        /// The demo app; with `--stream-from`, the library's scene (overriding `--scene` and
        /// `--detail`) plus the file source its pages stream from.
        fn create_app(&self, ctx: &GpuContext, format: wgpu::TextureFormat, size: (u32, u32)) -> (DemoApp, Option<Arc<FileSource>>) {
            let config = self.renderer_config();
            let Some(dir) = &self.stream_from else {
                let model = self.load_model();
                return (DemoApp::with_config(&ctx.device, &ctx.queue, format, size.0, size.1, self.scene, self.detail, config, model), None);
            };
            match mesh_store::open(std::path::Path::new(dir), Duration::from_millis(self.io_latency_ms)) {
                Ok((library, io)) => (DemoApp::from_library(&ctx.device, &ctx.queue, format, size, library, config), Some(io)),
                Err(e) => {
                    log::error!("--stream-from {dir}: {e}");
                    std::process::exit(1);
                }
            }
        }

        fn apply(&self, app: &mut DemoApp) {
            if let Some(index) = self.view {
                app.renderer.settings.debug_view = DebugView::ALL[index.min(DebugView::ALL.len() - 1)];
            }
            if let Some([x, y, z, yaw, pitch]) = self.camera {
                app.camera.position = Vec3::new(x, y, z);
                app.camera.yaw = yaw;
                app.camera.pitch = pitch;
            }
            app.renderer.settings.occlusion_culling = self.occlusion;
            app.renderer.settings.virtual_texturing = self.rvt;
            app.renderer.settings.texture_streaming = self.svt;
            if let Some(lod) = self.lod {
                app.renderer.settings.lod_threshold = lod;
            }
            if let Some(fov) = self.fov {
                app.camera.fov_y = fov.clamp(1.0, 150.0).to_radians();
            }
            if let Some((azimuth, elevation)) = self.sun {
                app.set_sun(azimuth.to_radians(), elevation.to_radians());
            }
            if let Some(percentage) = self.screen_percentage {
                app.renderer.settings.screen_percentage = percentage;
            }
            app.renderer.settings.local_exposure = self.local_exposure;
            if let Some(budget) = self.gpu_budget_ms {
                app.gpu_budget_ms = budget;
                let mut dynamic = helios_render::DynamicResolution::with_budget(budget);
                // A screen percentage given as well caps the dynamic range.
                if let Some(percentage) = self.screen_percentage {
                    dynamic.max_percentage = percentage;
                }
                app.renderer.settings.dynamic_resolution = Some(dynamic);
            }
            if self.animate_sun {
                app.key_pressed("KeyP");
            }
            if self.keys_at.is_none() {
                for key in &self.keys {
                    app.key_pressed(key);
                }
            }
            if !self.sw_raster {
                app.renderer.settings.sw_raster_threshold = 0.0;
            }
            // --eye / --at take heights relative to the terrain below them.
            if let Some(eye) = self.eye {
                app.camera.position = Vec3::new(eye.x, eye.y + app.ground_height(eye.x, eye.z), eye.z);
            }
            if let Some(at) = self.at {
                let target = Vec3::new(at.x, at.y + app.ground_height(at.x, at.z), at.z);
                app.camera.look_at(target);
            }
        }
    }

    fn parse_vec3(s: &str) -> Option<Vec3> {
        let v: Vec<f32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
        (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
    }

    // ---- Headless ------------------------------------------------------------------------------

    const WARM_UP_FRAMES: usize = 10;

    pub fn screenshot(options: &Options, path: &str) {
        let ctx = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)).expect("GPU context");
        let (width, height) = options.size;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());

        let (mut app, io) = options.create_app(&ctx, format, (width, height));
        options.apply(&mut app);
        let mut cpu_ms = Vec::with_capacity(options.frames as usize);
        let mut start = Instant::now();
        let mut input = Input::default();
        input.held.extend(options.hold.iter().cloned());
        let mut timed_frames = options.frames;
        let mut percentage = app.renderer.screen_percentage();
        if let Some(dir) = &options.dump_dir {
            std::fs::create_dir_all(dir).expect("create dump directory");
        }
        for i in 0..options.frames {
            if let Some(kind) = options.switch_to.filter(|_| i == options.frames / 2) {
                app.load_scene(kind);
                options.apply(&mut app);
                start = Instant::now();
                timed_frames = options.frames - i;
            }
            let cpu = Instant::now();
            if options.move_light {
                app.circle_nearest_light(i);
            }
            if options.keys_at == Some(i) {
                for key in &options.keys {
                    app.key_pressed(key);
                }
            }
            app.update(&mut input, 1.0 / 60.0);
            app.render(&view, 1.0 / 60.0);
            cpu_ms.push(cpu.elapsed().as_secs_f64() * 1000.0);
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            if app.renderer.settings.dynamic_resolution.is_some() && app.renderer.screen_percentage() != percentage {
                percentage = app.renderer.screen_percentage();
                let gpu = app.renderer.gpu_time_ms().unwrap_or(0.0);
                log::info!("frame {i}: dynamic resolution {percentage:.0} % (smoothed GPU {gpu:.2} ms)");
            }
            if let Some(dir) = &options.dump_dir {
                if i + options.dump_count >= options.frames {
                    let s = app.renderer.streaming_stats();
                    log::info!(
                        "frame {i}: resident {} pages, {} loads, {} evictions, {} pending ({} in flight, {} waiting, {:.1} MiB staged)",
                        s.resident_pages,
                        s.loads,
                        s.evictions,
                        s.pending,
                        s.in_flight,
                        s.waiting,
                        s.staged_mb
                    );
                    let pixels = read_back(&ctx, &target, width, height);
                    save_png(&format!("{dir}/frame_{i:04}.png"), width, height, &pixels);
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0 / timed_frames.max(1) as f64;
        log::info!("rendered {} frames, {:.2} ms/frame (incl. sync)", options.frames, elapsed);
        // The first frames compile pipelines and synthesise textures.
        let mut cpu_ms = cpu_ms.split_off(cpu_ms.len().min(WARM_UP_FRAMES));
        cpu_ms.sort_by(f64::total_cmp);
        let percentile = |q: f64| cpu_ms.get(((cpu_ms.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
        log::info!(
            "CPU frame time after {WARM_UP_FRAMES} frames (update + render, without GPU wait): median {:.2} ms, p99 {:.2} ms, max {:.2} ms",
            percentile(0.5),
            percentile(0.99),
            percentile(1.0)
        );
        if let Some(io) = io {
            let s = io.stats();
            log::info!(
                "streaming I/O: {} reads ({} failed), {:.1} MiB, latency mean {:.2} ms / max {:.2} ms, {:.0} MiB/s per read",
                s.reads,
                s.failed,
                s.bytes as f64 / 1048576.0,
                s.mean_latency().as_secs_f64() * 1000.0,
                s.max_latency.as_secs_f64() * 1000.0,
                s.throughput() / 1048576.0
            );
        }
        for line in app.hud().lines() {
            log::info!("{line}");
        }

        let pixels = read_back(&ctx, &target, width, height);
        save_png(path, width, height, &pixels);
        log::info!("saved {path}");
    }

    fn read_back(ctx: &GpuContext, target: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
        let padded = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(height) },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        ctx.queue.submit([encoder.finish()]);
        buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map screenshot"));
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let data = buffer.get_mapped_range(..).expect("mapped range");
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in data.chunks(padded as usize) {
            pixels.extend_from_slice(&row[..(width * 4) as usize]);
        }
        pixels
    }

    /// Saves the swapchain image as `helios-<unix time>.png` in the working directory.
    fn save_window_screenshot(ctx: &GpuContext, texture: &wgpu::Texture, config: &wgpu::SurfaceConfiguration) {
        if !config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            log::warn!("this surface cannot be copied; use --screenshot for headless captures");
            return;
        }
        let mut pixels = read_back(ctx, texture, config.width, config.height);
        if matches!(config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb) {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let path = format!("helios-{seconds}.png");
        save_png(&path, config.width, config.height, &pixels);
        log::info!("saved {path}");
    }

    fn save_png(path: &str, width: u32, height: u32, pixels: &[u8]) {
        let file = std::fs::File::create(path).expect("create screenshot file");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().and_then(|mut w| w.write_image_data(pixels)).expect("write png");
    }

    // ---- Interactive window --------------------------------------------------------------------

    struct State {
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        ctx: GpuContext,
        config: wgpu::SurfaceConfiguration,
        app: DemoApp,
        input: Input,
        looking: bool,
        last_frame: Instant,
        title_timer: f32,
        /// F12 was pressed: save the next presented frame.
        screenshot: bool,
    }

    struct Handler {
        options: Options,
        state: Option<State>,
    }

    pub fn run_window(options: Options) {
        println!("{HELP}");
        let event_loop = EventLoop::new().expect("event loop");
        let mut handler = Handler { options, state: None };
        event_loop.run_app(&mut handler).expect("run");
    }

    impl ApplicationHandler for Handler {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.state.is_some() {
                return;
            }
            let attributes = Window::default_attributes()
                .with_title("Helios — virtual geometry & lighting")
                .with_inner_size(winit::dpi::PhysicalSize::new(self.options.size.0, self.options.size.1));
            let window = Arc::new(event_loop.create_window(attributes).expect("window"));
            let instance = GpuContext::create_instance();
            let surface = instance.create_surface(window.clone()).expect("surface");
            let ctx = pollster::block_on(GpuContext::new(instance, Some(&surface))).expect("GPU context");

            let size = window.inner_size();
            let caps = surface.get_capabilities(&ctx.adapter);
            let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
            let mut config = surface.get_default_config(&ctx.adapter, size.width.max(1), size.height.max(1)).expect("config");
            config.format = format;
            config.present_mode = wgpu::PresentMode::AutoVsync;
            // F12 screenshots copy the swapchain image.
            if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
                config.usage |= wgpu::TextureUsages::COPY_SRC;
            }
            surface.configure(&ctx.device, &config);

            let (mut app, _) = self.options.create_app(&ctx, format, (config.width, config.height));
            self.options.apply(&mut app);
            self.state = Some(State {
                window,
                surface,
                ctx,
                config,
                app,
                input: Input::default(),
                looking: false,
                last_frame: Instant::now(),
                title_timer: 0.0,
                screenshot: false,
            });
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
            let Some(st) = self.state.as_mut() else { return };
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    st.config.width = size.width.max(1);
                    st.config.height = size.height.max(1);
                    st.surface.configure(&st.ctx.device, &st.config);
                    st.app.resize(st.config.width, st.config.height);
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        let name = format!("{code:?}");
                        if event.state == ElementState::Pressed {
                            if name == "Escape" {
                                event_loop.exit();
                            }
                            if name == "F12" {
                                st.screenshot = true;
                            }
                            if !event.repeat {
                                st.app.key_pressed(&name);
                            }
                            st.input.held.insert(name);
                        } else {
                            st.input.held.remove(&name);
                        }
                    }
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left | MouseButton::Right, .. } => {
                    st.looking = state == ElementState::Pressed;
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    st.input.scroll -= y.signum();
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = (now - st.last_frame).as_secs_f32();
                    st.last_frame = now;
                    st.app.update(&mut st.input, dt);
                    let texture = match st.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                            st.surface.configure(&st.ctx.device, &st.config);
                            return;
                        }
                        _ => return,
                    };
                    let view = texture.texture.create_view(&Default::default());
                    st.app.render(&view, dt);
                    if std::mem::take(&mut st.screenshot) {
                        save_window_screenshot(&st.ctx, &texture.texture, &st.config);
                    }
                    st.window.pre_present_notify();
                    st.ctx.queue.present(texture);

                    st.title_timer += dt;
                    if st.title_timer > 0.5 {
                        st.title_timer = 0.0;
                        st.window.set_title(&format!("Helios — {}", st.app.hud().lines().next().unwrap_or_default()));
                        log::debug!("{}", st.app.hud().replace('\n', " | "));
                    }
                }
                _ => {}
            }
        }

        fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
            if let (Some(st), DeviceEvent::MouseMotion { delta }) = (self.state.as_mut(), event) {
                if st.looking {
                    st.input.look += Vec2::new(delta.0 as f32, delta.1 as f32);
                }
            }
        }

        fn about_to_wait(&mut self, _: &ActiveEventLoop) {
            if let Some(st) = &self.state {
                st.window.request_redraw();
            }
        }
    }
}
