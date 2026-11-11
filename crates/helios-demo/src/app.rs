//! Platform-independent demo application: fly camera, hotkeys, sun animation and HUD.

use std::collections::HashSet;

use glam::{Vec2, Vec3};
use helios_render::{Camera, DebugView, DynamicResolution, GiMethod, Renderer, RendererConfig, Scene};

use crate::editor::Editor;
use crate::mesh_store::{Library, MeshStore};
use crate::motion::DemoMotion;
use crate::scenes::model::Model;
use crate::scenes::{self, Detail, SceneKind, SceneSetup};

/// Input collected by the platform layer. Keys use DOM `KeyboardEvent.code` names
/// (`"KeyW"`, `"ShiftLeft"`, …), which match winit's `KeyCode` debug names.
#[derive(Default)]
pub struct Input {
    pub held: HashSet<String>,
    /// Accumulated look delta in pixels since the last frame.
    pub look: Vec2,
    pub scroll: f32,
}

impl Input {
    fn down(&self, code: &str) -> bool {
        self.held.contains(code)
    }
}

pub struct DemoApp {
    pub renderer: Renderer,
    pub scene: Scene,
    pub camera: Camera,
    pub kind: SceneKind,
    pub(crate) detail: Detail,
    /// Imported glTF model, if any (`SceneKind::Model` builds from it).
    pub(crate) model: Option<Model>,
    initial_camera: Camera,
    pub(crate) floor: fn(f32, f32) -> f32,
    pub(crate) floor_below: Option<fn(Vec3) -> f32>,
    /// The scene's moving props (`Digit0` pauses them).
    pub motion: DemoMotion,
    pub(crate) sun_intensity: f32,
    /// Fly speed (m/s; the mouse wheel changes it).
    pub speed: f32,
    pub(crate) sun_azimuth: f32,
    pub(crate) sun_elevation: f32,
    /// Position in the day cycle (radians) and the azimuth it started from.
    day_phase: f32,
    day_azimuth: f32,
    animate_sun: bool,
    /// Time-lapse through a whole day and night.
    animate_day: bool,
    /// A running camera tour and its clock.
    tour: Option<(crate::tour::Tour, f32)>,
    /// Skeletal animation of the model scene's skinned meshes.
    animation: Option<scenes::skeletal::SkeletalAnimation>,
    /// Exposure compensation the scene asked for in daylight.
    day_exposure: f32,
    frame_ms: f32,
    /// GPU time budget dynamic resolution steers to when switched on with the hotkey.
    pub gpu_budget_ms: f32,
    /// Scene edits over the built scene (see `editor`), and whether its tools are active.
    pub editor: Editor,
    pub editing: bool,
}

/// Frames a photo (Enter) accumulates.
pub const PHOTO_SAMPLES: u32 = 128;

/// Screen percentages the `[` / `]` keys step through.
const SCREEN_PERCENTAGES: [f32; 4] = [50.0, 67.0, 75.0, 100.0];

pub const HELP: &str = "WASD/QE fly · drag to look · wheel speed · Shift fast\n\
N next scene · 0 demo motion · V debug view · Insert voxel LODs · PageDown tessellation · G GI · ; GI method (Lumen / Lumen Lite) · H shadows · 1 local light shadow maps · [ ] screen percentage (50/67/75/100 %) · \\ dynamic resolution · T TAA · O occlusion · K software raster · I texture streaming · - animation clip / pause · F fog · C clouds · U translucency · End coloured light through glass · Home fog scattering · 2 shadow refresh budget · 7 prefiltered distant shadows / volume shadows (cycles) · / translucency layers · ' volumes + particles · L wind · M motion blur · B depth of field · Enter photo (accumulated still) · ` local exposure · = MegaLights · 3 MegaLights fog + translucency · 4 OIT (AVBOIT / weighted blended) · 5 Lumen world cache (off / behind cards / replaces cards) · 6 Lumen Lite reflections (screen traces / traced) · 9 short-range AO · Numpad1 gust (dynamic wind) · Numpad2 river simulation · Z/X zoom · , . LOD error · PageUp Substrate materials · ←→↑↓ sun · P animate sun · J day/night cycle · Y camera tour · R reset · Tab editor tools (native)";

impl DemoApp {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        kind: SceneKind,
        detail: Detail,
    ) -> Self {
        Self::with_config(device, queue, format, width, height, kind, detail, RendererConfig::default(), None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        kind: SceneKind,
        detail: Detail,
        config: RendererConfig,
        model: Option<Model>,
    ) -> Self {
        Self::build(Renderer::with_config(device, queue, format, width, height, config), kind, detail, model)
    }

    /// Builds the scene for a renderer made beforehand (the browser creates the renderer first,
    /// so its GPU process compiles shaders while the scene bakes).
    pub fn build(renderer: Renderer, kind: SceneKind, detail: Detail, model: Option<Model>) -> Self {
        let setup = scenes::build(kind, detail, model.as_ref());
        Self::from_setup(renderer, setup, kind, detail, model)
    }

    /// Builds the library's scene with its pre-baked meshes, streaming their pages on demand.
    pub fn from_library(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        (width, height): (u32, u32),
        library: Library,
        config: RendererConfig,
    ) -> Self {
        Self::build_from_library(Renderer::with_config(device, queue, format, width, height, config), library)
    }

    /// [`DemoApp::from_library`] for a renderer made beforehand.
    pub fn build_from_library(renderer: Renderer, library: Library) -> Self {
        let (kind, detail) = (library.manifest.scene, library.manifest.detail);
        let (setup, _) = scenes::build_with(kind, detail, None, MeshStore::Stream(library));
        Self::from_setup(renderer, setup, kind, detail, None)
    }

    /// The app for a scene built beforehand (the browser bakes it in workers).
    pub(crate) fn from_setup(renderer: Renderer, setup: SceneSetup, kind: SceneKind, detail: Detail, model: Option<Model>) -> Self {
        let editor = Editor::new(&setup.scene, crate::app_editor::base_scene(kind, detail, model.as_ref()), &setup.outline);
        Self {
            renderer,
            camera: setup.camera,
            initial_camera: setup.camera,
            kind,
            detail,
            model,
            floor: setup.floor,
            floor_below: setup.floor_below,
            motion: setup.motion,
            sun_intensity: setup.sun.2,
            speed: 8.0,
            sun_azimuth: setup.sun.0,
            sun_elevation: setup.sun.1,
            day_phase: 0.0,
            day_azimuth: setup.sun.0,
            animate_sun: false,
            animate_day: false,
            tour: None,
            animation: setup.animation,
            day_exposure: 0.0,
            frame_ms: 16.0,
            gpu_budget_ms: 14.0,
            scene: setup.scene,
            editor,
            editing: false,
        }
    }

    /// Replaces the scene (bakes the new one; takes a few seconds).
    pub fn load_scene(&mut self, kind: SceneKind) {
        let setup = scenes::build(kind, self.detail, self.model.as_ref());
        self.set_scene(kind, setup);
    }

    /// Replaces the scene with one built beforehand.
    pub(crate) fn set_scene(&mut self, kind: SceneKind, setup: SceneSetup) {
        self.kind = kind;
        self.editor = Editor::new(&setup.scene, crate::app_editor::base_scene(kind, self.detail, self.model.as_ref()), &setup.outline);
        self.scene = setup.scene;
        self.camera = setup.camera;
        self.initial_camera = setup.camera;
        self.floor = setup.floor;
        self.floor_below = setup.floor_below;
        self.animation = setup.animation;
        // A new scene's props start moving or paused as the previous ones were.
        let enabled = self.motion.enabled;
        self.motion = setup.motion;
        self.motion.enabled = enabled;
        self.sun_azimuth = setup.sun.0;
        self.sun_elevation = setup.sun.1;
        self.sun_intensity = setup.sun.2;
        self.tour = None;
        self.renderer.set_selection(&[]);
        self.renderer.reset_history();
    }

    /// The current scene's name (the model's name for an imported model).
    pub fn scene_name(&self) -> &str {
        match (&self.model, self.kind) {
            (Some(model), SceneKind::Model) => &model.name,
            _ => self.kind.name(),
        }
    }

    /// Picks the GI's final gather from a `--gi` / `?gi=` value: `lumen` (screen probes) or
    /// `lite` (Lumen Lite); otherwise (`auto`, absent) Lumen Lite on integrated and software
    /// adapters, and a switch to it should dynamic resolution stay at its minimum.
    pub fn choose_gi(&mut self, value: Option<&str>, adapter: &wgpu::AdapterInfo) {
        let s = &mut self.renderer.settings;
        match value.and_then(GiMethod::parse) {
            Some(method) => {
                s.gi_method = method;
                s.gi_lite_fallback = false;
            }
            None => {
                if let Some(other) = value.filter(|v| *v != "auto") {
                    log::warn!("unknown GI method {other:?} (lumen, lite or auto)");
                }
                s.gi_method = GiMethod::for_adapter(adapter);
                s.gi_lite_fallback = true;
            }
        }
        log::info!("GI final gather: {}{}", s.gi_method.name(), if s.gi_lite_fallback { " (Lumen Lite when the GPU falls behind)" } else { "" });
    }

    /// Places the sun (radians); `update` applies it, including the night-time exposure.
    pub fn set_sun(&mut self, azimuth: f32, elevation: f32) {
        self.sun_azimuth = azimuth;
        self.sun_elevation = elevation;
    }

    pub fn ground_height(&self, x: f32, z: f32) -> f32 {
        (self.floor)(x, z)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    /// One-shot key presses.
    pub fn key_pressed(&mut self, code: &str) {
        let s = &mut self.renderer.settings;
        match code {
            "KeyV" => s.debug_view = s.debug_view.next(),
            "KeyG" => s.global_illumination = !s.global_illumination,
            // Lumen's screen probes <-> Lumen Lite's irradiance field (a manual choice ends the
            // automatic one).
            "Semicolon" => {
                s.gi_method = s.gi_method.toggled();
                s.gi_lite_fallback = false;
            }
            "KeyH" => s.shadows = !s.shadows,
            "KeyT" => s.taa = !s.taa,
            "KeyO" => s.occlusion_culling = !s.occlusion_culling,
            "KeyF" => s.aerial_perspective = !s.aerial_perspective,
            "KeyC" => s.clouds = !s.clouds,
            "KeyU" => s.translucency = !s.translucency,
            "Slash" => s.translucency_layers = !s.translucency_layers,
            // Sunlight through glass coloured by the panes / white.
            "End" => s.glass_shadows = !s.glass_shadows,
            // Photo mode: accumulate a still (sub-pixel and, with B, lens samples); again: back
            // to the real-time image.
            "Enter" | "NumpadEnter" => {
                if self.renderer.photo_progress().is_some() {
                    self.renderer.stop_photo();
                } else {
                    self.renderer.start_photo(PHOTO_SAMPLES);
                }
            }
            // Fog screen space scattering (glow of lights through fog) on / off.
            "Home" => s.fog_scattering = if s.fog_scattering.is_some() { None } else { Some(helios_render::FogScattering::default()) },
            // Shadow page refreshes throttled (refresh budget, distant levels at their own rate)
            // / every invalidated page re-rendered at once.
            "Digit2" => {
                let on = !(s.shadow_refresh_budget > 0 || s.shadow_distant_level > 0);
                s.shadow_refresh_budget = if on { helios_render::DEFAULT_SHADOW_REFRESH_BUDGET } else { 0 };
                s.shadow_distant_level = if on { helios_render::DEFAULT_SHADOW_DISTANT_LEVEL } else { 0 };
            }
            // Adaptive voxel-based OIT / weighted blended OIT and depth peeling (A/B).
            "Digit4" => s.avboit = !s.avboit,
            // Round 8 shadows, cycled: both on -> prefiltered distant shadows off -> volume
            // shadows off (distant prefiltered again) -> both on.
            "Digit7" => match (s.shadow_distant_prefiltered, s.volume_shadows) {
                (true, true) => s.shadow_distant_prefiltered = false,
                (false, true) => {
                    s.shadow_distant_prefiltered = true;
                    s.volume_shadows = false;
                }
                _ => {
                    s.shadow_distant_prefiltered = true;
                    s.volume_shadows = true;
                }
            },
            // Short-range ambient occlusion (grounded foliage, rock and log contact) on / off.
            "Digit9" => s.short_range_ao = !s.short_range_ao,
            // Shallow-water river simulation running / paused.
            "Numpad2" => s.river_simulation = !s.river_simulation,
            // A gust burst in front of the camera (the dynamic wind's simulated tier): a ring
            // that bends the grass, ferns and bushes it passes.
            "Numpad1" => {
                let forward = self.camera.forward();
                let ahead = Vec3::new(forward.x, 0.0, forward.z).normalize_or(Vec3::X) * 5.0;
                self.renderer.wind_gust(self.camera.position + ahead);
            }
            "KeyL" => s.wind = !s.wind,
            "Quote" => {
                let on = !(s.volumes || s.particles);
                s.volumes = on;
                s.particles = on;
            }
            "KeyM" => s.motion_blur = if s.motion_blur > 0.0 { 0.0 } else { 0.5 },
            "KeyB" => s.depth_of_field = if s.depth_of_field.is_some() { None } else { Some(helios_render::DepthOfField::default()) },
            "Backquote" => s.local_exposure = if s.local_exposure.is_some() { None } else { Some(helios_render::LocalExposure::default()) },
            "KeyI" => s.texture_streaming = !s.texture_streaming,
            "KeyK" => s.sw_raster_threshold = if s.sw_raster_threshold > 0.0 { 0.0 } else { 2.5 },
            // Voxel LODs of foliage on / off (the triangle-only cut), for A/B comparisons.
            "Insert" => s.voxel_lods = !s.voxel_lods,
            // Nanite tessellation of displaced materials on / off (undisplaced surfaces).
            "PageDown" => s.tessellation = !s.tessellation,
            // Screen percentage one step down / up. With dynamic resolution on, a press turns it
            // off (back to the fixed percentage).
            "BracketLeft" | "BracketRight" => {
                if s.dynamic_resolution.take().is_none() {
                    let current = s.screen_percentage;
                    s.screen_percentage = if code == "BracketLeft" {
                        SCREEN_PERCENTAGES.iter().rev().copied().find(|&p| p < current).unwrap_or(SCREEN_PERCENTAGES[0])
                    } else {
                        SCREEN_PERCENTAGES.iter().copied().find(|&p| p > current).unwrap_or(current.max(100.0))
                    };
                }
            }
            // Dynamic resolution (GPU budget `gpu_budget_ms`, 50–100 %) on / off.
            "Backslash" => {
                s.dynamic_resolution = match s.dynamic_resolution {
                    Some(_) => None,
                    None => Some(DynamicResolution::with_budget(self.gpu_budget_ms)),
                }
            }
            "Comma" => s.lod_threshold = (s.lod_threshold * 0.5).max(0.125),
            "Period" => s.lod_threshold = (s.lod_threshold * 2.0).min(64.0),
            "KeyP" => self.animate_sun = !self.animate_sun,
            "KeyJ" => self.animate_day = !self.animate_day,
            "KeyY" => self.tour = if self.tour.is_some() { None } else { crate::tour::Tour::for_scene(self.kind).map(|t| (t, 0.0)) },
            // Zoom: narrower fields of view make depth of field (B) much shallower, as with a
            // long lens.
            "KeyZ" => self.camera.fov_y = (self.camera.fov_y / 1.25).max(5f32.to_radians()),
            "KeyX" => self.camera.fov_y = (self.camera.fov_y * 1.25).min(100f32.to_radians()),
            "KeyR" => {
                self.camera = self.initial_camera;
                self.renderer.reset_history();
            }
            "KeyN" => self.load_scene(self.kind.next(self.model.is_some())),
            // Skeletal animation: next clip (crossfading); after the last one, pause.
            "Minus" => {
                if let Some(animation) = &mut self.animation {
                    animation.cycle();
                }
            }
            // All letters are taken: 1 toggles the local lights' virtual shadow maps (off: their
            // distance-field shadows); digits without a toggle of their own pick debug views.
            "Digit1" => s.local_light_shadow_maps = !s.local_light_shadow_maps,
            // 0 pauses and resumes the scene's moving props.
            "Digit0" => self.motion.enabled = !self.motion.enabled,
            // MegaLights (stochastic shadowed lighting from every local light) vs the classic loop.
            "Equal" => s.mega_lights = !s.mega_lights,
            // MegaLights' froxel volume for fog and translucent surfaces vs their clustered loops.
            "Digit3" => s.mega_lights_volume = !s.mega_lights_volume,
            // Lumen world cache: off -> behind the mesh cards -> in place of the cards -> off.
            "Digit5" => s.world_cache = s.world_cache.next(),
            // Lumen Lite's reflections: screen traces with the irradiance field behind them vs
            // Lumen's per-pixel world traces.
            "Digit6" => s.lumen_lite_ssr = !s.lumen_lite_ssr,
            // Substrate slabs on / off (extended materials A/B against their plain base).
            "PageUp" => s.substrate = !s.substrate,
            code if code.starts_with("Digit") => {
                if let Some(index) = code[5..].parse::<usize>().ok().and_then(|d| d.checked_sub(1)) {
                    if let Some(&view) = DebugView::ALL.get(index) {
                        s.debug_view = view;
                    }
                }
            }
            _ => {}
        }
    }

    pub fn update(&mut self, input: &mut Input, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        self.frame_ms = self.frame_ms * 0.95 + dt * 1000.0 * 0.05;

        // Look.
        let sensitivity = 0.0035;
        self.camera.yaw -= input.look.x * sensitivity;
        self.camera.pitch = (self.camera.pitch - input.look.y * sensitivity).clamp(-1.54, 1.54);
        input.look = Vec2::ZERO;

        // Move.
        if input.scroll != 0.0 {
            self.speed = (self.speed * (1.0 - input.scroll.signum() * 0.15)).clamp(0.5, 400.0);
            input.scroll = 0.0;
        }
        let forward = self.camera.forward();
        let right = self.camera.right();
        let mut direction = Vec3::ZERO;
        for (code, axis) in [("KeyW", forward), ("KeyS", -forward), ("KeyD", right), ("KeyA", -right), ("KeyE", Vec3::Y), ("KeyQ", -Vec3::Y)] {
            if input.down(code) {
                direction += axis;
            }
        }
        let boost = if input.down("ShiftLeft") || input.down("ShiftRight") { 5.0 } else { 1.0 };
        self.camera.position += direction.normalize_or_zero() * self.speed * boost * dt;
        let floor = match self.floor_below {
            Some(below) => below(self.camera.position),
            None => (self.floor)(self.camera.position.x, self.camera.position.z),
        } + 0.4;
        self.camera.position.y = self.camera.position.y.max(floor);
        if let Some((tour, time)) = &mut self.tour {
            *time += dt;
            let (eye, target) = tour.sample(*time, self.floor);
            self.camera.position = eye;
            self.camera.look_at(target);
        }

        // While a photo accumulates the scene stands still (the camera may move: that ends it).
        let camera_dt = dt;
        let dt = if self.renderer.photo_progress().is_some_and(|p| !p.done()) { 0.0 } else { dt };

        // Sun.
        let rate = 0.6 * camera_dt;
        if input.down("ArrowLeft") {
            self.sun_azimuth -= rate;
        }
        if input.down("ArrowRight") {
            self.sun_azimuth += rate;
        }
        if input.down("ArrowUp") {
            self.sun_elevation += rate * 0.5;
        }
        if input.down("ArrowDown") {
            self.sun_elevation -= rate * 0.5;
        }
        if self.animate_sun {
            self.sun_azimuth += dt * 0.08;
        }
        if self.animate_day {
            // A day in about two minutes: the sun climbs, sets and the night follows.
            self.day_phase = (self.day_phase + dt * std::f32::consts::TAU / 120.0) % std::f32::consts::TAU;
            self.sun_elevation = 0.95 * self.day_phase.sin();
            self.sun_azimuth = self.day_azimuth + self.day_phase * 0.5;
        } else {
            self.day_phase = self.sun_elevation.clamp(-0.95, 0.95).asin();
            self.day_azimuth = self.sun_azimuth - self.day_phase * 0.5;
        }
        self.sun_elevation = self.sun_elevation.clamp(-1.2, 1.5);
        if let Some(animation) = &mut self.animation {
            animation.update(&mut self.scene, dt);
        }
        self.scene.set_time_of_day(self.sun_azimuth, self.sun_elevation, self.sun_intensity);
        self.motion.update(&mut self.scene, dt);
        self.maintain_editor();
        // Night stays dark: exposure adapts, but not all the way to daylight brightness.
        self.renderer.settings.exposure_compensation = self.day_exposure - 1.6 * Scene::night_factor(self.sun_elevation);
    }

    pub fn render(&mut self, output: &wgpu::TextureView, dt: f32) {
        self.renderer.render(&self.scene, &self.camera, output, dt);
    }

    /// Test path for moving lights (`--move-light`): the shadow-casting light nearest the
    /// camera circles 0.25 m around where it started, one turn every two seconds at 60 fps, so
    /// its shadow map is re-rendered every frame.
    pub fn circle_nearest_light(&mut self, frame: u32) {
        let eye = self.camera.position;
        let lights = self.scene.lights_mut();
        let nearest = lights.iter_mut().filter(|l| l.cast_shadows).min_by(|a, b| a.position.distance(eye).total_cmp(&b.position.distance(eye)));
        if let Some(light) = nearest {
            let angle = |f: u32| f as f32 * std::f32::consts::TAU / 120.0;
            let offset = |f: u32| Vec3::new(angle(f).cos(), 0.0, angle(f).sin()) * 0.25;
            light.position += offset(frame + 1) - offset(frame);
        }
    }

    pub fn hud(&self) -> String {
        let st = self.renderer.stats();
        let stream = self.renderer.streaming_stats();
        let s = &self.renderer.settings;
        let on = |b: bool| if b { "on" } else { "off" };
        let budget = self.renderer.lod_budget_scale();
        let budget = if budget > 1.01 { format!(" (×{budget:.1} budget)") } else { String::new() };
        let gpu = match self.renderer.gpu_time_ms() {
            Some(ms) => format!(" · GPU {ms:.2} ms"),
            None => String::new(),
        };
        let (w, h) = self.renderer.internal_size();
        let mut resolution = format!("{:.0} % ({w}×{h})", self.renderer.screen_percentage());
        if let Some(dynamic) = s.dynamic_resolution {
            resolution.push_str(&format!(" dynamic, budget {:.1} ms", dynamic.gpu_budget_ms));
        }
        let mut text = format!(
            "{} · {:.1} ms ({:.0} fps){gpu} · {resolution} · {} · LOD error {} px{budget}\n\
             visible: {} clusters ({} software, {} alpha-masked), {:.2} M tris (early pass {}) · shadows: {:.2} M tris ({:.2} M software)\n\
             streaming: {}/{} pages, {:.0}/{:.0} MiB resident, {} pending, {} in flight · scene: {} instances, {:.0} M source tris\n\
             GI {} · shadows {} · TAA {} · fog {} · SW raster {} · wind {} · local exposure {} · MegaLights {} ({} lights) · volumes + particles {} · Substrate {} · glass shadows {} · fog scattering {} · OIT {}",
            self.scene_name(),
            self.frame_ms,
            1000.0 / self.frame_ms.max(0.01),
            s.debug_view.name(),
            s.lod_threshold,
            st.clusters,
            st.sw_clusters,
            st.masked_clusters,
            st.triangles as f64 / 1e6,
            st.early_clusters,
            st.shadow_triangles as f64 / 1e6,
            st.shadow_sw_triangles as f64 / 1e6,
            stream.resident_pages,
            stream.total_pages,
            stream.resident_mb,
            stream.pool_mb,
            stream.pending,
            stream.in_flight + stream.decoding,
            self.scene.instance_count(),
            self.scene.source_triangle_count() as f64 / 1e6,
            if !s.global_illumination {
                "off".to_owned()
            } else if s.world_cache == helios_render::WorldCacheMode::Off {
                s.gi_method.name().to_owned()
            } else {
                format!("{}, world cache {}", s.gi_method.name(), s.world_cache.name())
            },
            on(s.shadows),
            on(s.taa),
            on(s.aerial_perspective),
            on(s.sw_raster_threshold > 0.0),
            on(s.wind && self.scene.wind.strength > 0.0),
            on(s.local_exposure.is_some()),
            on(s.mega_lights),
            self.scene.lights().len(),
            on(s.volumes || s.particles),
            on(s.substrate),
            on(s.glass_shadows),
            on(s.fog_scattering.is_some()),
            if s.avboit { "AVBOIT" } else { "weighted blended" },
        );
        text.push_str(&format!(
            "\nshadow pages: {}/{} in use, {} requested, {} rendered ({} dynamic casters only, {} split; {}/{} draws), {} unallocated · local lights with shadow maps: {} ({} moving, cached)",
            st.shadow_pages,
            st.shadow_page_capacity,
            st.shadow_pages_requested,
            st.shadow_dirty_pages,
            st.shadow_dynamic_pages,
            st.shadow_split_pages,
            st.shadow_draws,
            st.shadow_draw_capacity,
            st.shadow_page_overflow,
            self.renderer.shadow_mapped_lights(),
            self.renderer.shadow_held_lights(),
        ));
        if s.shadow_refresh_budget > 0 || s.shadow_distant_level > 0 {
            text.push_str(&format!(
                "
shadow refreshes: budget {} a frame, {} deferred, {} past the budget (waited {} frames) · distant levels from {} every {} frames: {} pages waiting for their turn",
                s.shadow_refresh_budget,
                st.shadow_refresh_deferred,
                st.shadow_refresh_forced,
                s.shadow_max_staleness,
                s.shadow_distant_level,
                s.shadow_distant_interval,
                st.shadow_refresh_held,
            ));
        }
        let pool = self.renderer.local_pool_stats();
        if pool.requested > 0 {
            text.push_str(&format!(
                "\nlocal lights' shadow pages: {} of {} requested resident · maps without a resident page: {} · pool resolution bias {:.2} mips",
                pool.resident, pool.requested, pool.starved, pool.bias
            ));
        }
        if let Some(p) = self.renderer.photo_progress() {
            let state = if p.focusing { "focusing".to_string() } else if p.done() { "done (Enter: back to real time)".to_string() } else { format!("{} / {} frames", p.accumulated, p.samples) };
            let lens = if s.depth_of_field.is_some() { "lens + sub-pixel samples" } else { "sub-pixel samples" };
            text.push_str(&format!("\nphoto: {state} ({lens})"));
        }
        let moving = self.renderer.moving_light_stats();
        if moving.anchors.mapped + moving.anchors.fields > 0 {
            text.push_str(&format!(
                "\nmoving lights: {} with shadow maps ({} at a coarser tolerance), {} on distance fields · {} faces re-anchored ({:.0} pages; held maps need {:.1} of {:.0} pages a frame) · local pages rendered in full: {} ({} of moving lights) · moving lights' pages rendered: {} (dynamic casters only: {})",
                moving.anchors.mapped,
                moving.anchors.relaxed,
                moving.anchors.fields,
                moving.anchors.faces,
                moving.anchors.pages,
                moving.anchors.load,
                s.moving_light_pages,
                moving.full_pages,
                moving.moving_full_pages,
                moving.moving_pages,
                moving.moving_pages.saturating_sub(moving.moving_full_pages),
            ));
        }
        if self.motion.instance_count() > 0 {
            let d = self.renderer.dynamic_stats();
            text.push_str(&format!(
                "\nmoving (0): {} props {} · {} instances moved, {} shadow casters · distance field: {} bricks recomposed, {} queued, {} levels rebuilt",
                self.motion.instance_count(),
                if self.motion.enabled { "moving" } else { "paused" },
                d.moved,
                d.shadow_moves,
                d.bricks,
                d.queued_bricks,
                d.sdf_levels,
            ));
        }
        if self.renderer.has_voxels() {
            text.push_str(&format!(
                "\nvoxel LODs {} (Insert): {:.2} M voxels visible, {:.2} M in shadows · assembly parts drawn: {}",
                on(s.voxel_lods),
                st.voxels as f64 / 1e6,
                st.shadow_voxels as f64 / 1e6,
                st.assembly_parts
            ));
        }
        if let Some(t) = self.renderer.tessellation_stats() {
            let scale = if t.dice_scale > 1.01 { format!(" (×{:.1}: lists full)", t.dice_scale) } else { String::new() };
            text.push_str(&format!(
                "
tessellation {} (PageDown): {:.2} M micro-triangles ({:.2} M in shadows), {} k patches ({} k in shadows), dice rate {} px{scale}, {} split passes",
                on(s.tessellation),
                t.micro_triangles as f64 / 1e6,
                t.shadow_micro_triangles as f64 / 1e6,
                t.patches / 1000,
                t.shadow_patches / 1000,
                s.dice_rate,
                t.split_passes,
            ));
        }
        let tex = self.renderer.texture_streaming_stats();
        if tex.layers > 0 {
            text.push_str(&format!(
                "\ntexture streaming {} ({}): {}/{} tiles resident ({} virtual, {} layers, {:.0} MiB), {} requested ({} over capacity), {} pending, {} uploads, {} evictions, upload {:.2} ms (peak {:.2})",
                on(s.texture_streaming),
                tex.format,
                tex.resident_tiles,
                tex.capacity,
                tex.virtual_tiles,
                tex.layers,
                tex.gpu_mb,
                tex.requested,
                tex.overflow,
                tex.pending,
                tex.uploads,
                tex.evictions,
                tex.upload_ms,
                tex.peak_upload_ms,
            ));
        }
        if let Some(animation) = &self.animation {
            text.push('\n');
            text.push_str(&animation.status());
        }
        let timings = self.renderer.gpu_timings();
        if !timings.is_empty() {
            text.push_str("\nGPU:");
            for (label, ms) in timings {
                text.push_str(&format!(" {label} {ms:.2} ·"));
            }
            text.pop();
        }
        text
    }
}
