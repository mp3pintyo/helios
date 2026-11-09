//! Browser platform layer: canvas surface, DOM input and a requestAnimationFrame loop.
//!
//! A page can embed the engine instead of using `index.html`'s full-window demo: it sets
//! `globalThis.heliosConfig = { scene, screenPercentage, gpuBudgetMs, embedded }` before
//! loading the module (query parameters still work and take part as fallbacks), and drives the
//! running engine through the exported `helios*` functions (scene, camera, sun, hotkeys, HUD).
//! `embedded: true` keeps keyboard input on the focused canvas and leaves the mouse wheel to
//! the page, so the engine can sit inside a scrolling document.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use glam::Vec2;
use helios_render::{DynamicResolution, GpuContext};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, HtmlElement};

use crate::app::{DemoApp, Input, HELP};
use crate::mesh_store;
use crate::scenes::model::Model;
use crate::scenes::{Detail, SceneKind};

struct State {
    app: DemoApp,
    input: Input,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    canvas: HtmlCanvasElement,
    hud: Option<HtmlElement>,
    last_time: f64,
    hud_timer: f64,
    /// Skip rendering (the page hides the canvas or scrolls it out of view).
    paused: bool,
}

thread_local! {
    /// The running engine, for the exported page API.
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

/// Runs `f` on the running engine (`None` until it has started).
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.try_borrow_mut().ok()?;
    Some(f(&mut st))
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no window")
}

fn set_status(text: &str) {
    if let Some(el) = window().document().and_then(|d| d.get_element_by_id("status")) {
        el.set_text_content(Some(text));
    }
}

/// Canvas backing-store size for its CSS size (device pixel ratio capped for performance).
fn canvas_size(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let ratio = window().device_pixel_ratio().min(1.5);
    let w = (canvas.client_width() as f64 * ratio).round().max(1.0) as u32;
    let h = (canvas.client_height() as f64 * ratio).round().max(1.0) as u32;
    (w, h)
}

/// A query parameter of the page URL, still percent-encoded.
fn query_param(name: &str) -> Option<String> {
    let search = window().location().search().unwrap_or_default();
    search.trim_start_matches('?').split('&').filter_map(|kv| kv.split_once('=')).find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
}

/// A field of the page's `globalThis.heliosConfig` object, if the page set one.
fn page_config(key: &str) -> Option<JsValue> {
    let config = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("heliosConfig")).ok()?;
    if config.is_undefined() || config.is_null() {
        return None;
    }
    js_sys::Reflect::get(&config, &JsValue::from_str(key)).ok().filter(|v| !v.is_undefined() && !v.is_null())
}

/// A start option: the page config's `key`, else the URL's `param`.
fn option_string(key: &str, param: &str) -> Option<String> {
    page_config(key).and_then(|v| v.as_string().or_else(|| v.as_f64().map(|n| n.to_string()))).or_else(|| query_param(param))
}

/// Whether the engine is embedded in a larger page (see the module docs).
fn embedded() -> bool {
    page_config("embedded").and_then(|v| v.as_bool()).unwrap_or(false)
}

/// `?texture-compression=0` keeps streamed texture tiles uncompressed (for comparisons).
fn renderer_config() -> helios_render::RendererConfig {
    helios_render::RendererConfig { texture_compression: query_param("texture-compression").as_deref() != Some("0"), ..Default::default() }
}

/// Scene selected with `?scene=valley|cathedral|forest|model`; `?gltf=URL` alone selects the model.
fn scene_from_url() -> SceneKind {
    match option_string("scene", "scene").and_then(|v| SceneKind::parse(&v)) {
        Some(kind) => kind,
        None if query_param("gltf").is_some() => SceneKind::Model,
        None => SceneKind::Valley,
    }
}

/// GPU time budget of the browser's default dynamic resolution: a 60 Hz frame with headroom
/// for the compositor and the CPU side.
const WEB_GPU_BUDGET_MS: f32 = 14.0;

/// Screen percentage and dynamic resolution from the URL. Dynamic resolution is on by default
/// in the browser: the page cannot pick the display's refresh or the GPU it runs on, and with
/// vsync a frame that misses 16.7 ms costs a whole refresh, so rendering fewer pixels (TSR
/// keeps the output resolution) is the better trade on laptops and integrated GPUs; a fast GPU
/// simply stays at 100 %. Without timestamp queries it stays at the maximum.
///   `?sp=N`       fixed screen percentage N (with `budget`: the dynamic maximum)
///   `?budget=MS`  dynamic resolution budget in GPU milliseconds (0 = off)
fn configure_resolution(app: &mut DemoApp) {
    let number = |key, name| option_string(key, name).and_then(|v| v.parse::<f32>().ok());
    let percentage = number("screenPercentage", "sp");
    let budget = number("gpuBudgetMs", "budget").or(if percentage.is_some() { None } else { Some(WEB_GPU_BUDGET_MS) });
    let s = &mut app.renderer.settings;
    if let Some(percentage) = percentage {
        s.screen_percentage = percentage;
    }
    if let Some(budget) = budget.filter(|&b| b > 0.0) {
        app.gpu_budget_ms = budget;
        let mut dynamic = DynamicResolution::with_budget(budget);
        if let Some(percentage) = percentage {
            dynamic.max_percentage = percentage;
        }
        app.renderer.settings.dynamic_resolution = Some(dynamic);
    }
}

fn js_error(e: JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(window().fetch_with_str(url))
        .await
        .map_err(js_error)?
        .dyn_into()
        .map_err(|_| "fetch did not return a Response".to_string())?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer().map_err(js_error)?).await.map_err(js_error)?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Fetches a glTF model (`.glb`, or `.gltf` plus the files it references, resolved relative
/// to it) from the `?gltf=` parameter.
async fn load_model(param: &str) -> Result<Model, String> {
    let decoded = js_sys::decode_uri_component(param).map(String::from).unwrap_or_else(|_| param.to_string());
    let base = window().location().href().map_err(js_error)?;
    let url = web_sys::Url::new_with_base(&decoded, &base).map_err(js_error)?.href();
    set_status(&format!("Loading {decoded}…"));
    let document = helios_asset::GltfDocument::parse(&fetch_bytes(&url).await?).map_err(|e| e.to_string())?;
    let mut resources = HashMap::new();
    for uri in document.external_uris() {
        let resolved = web_sys::Url::new_with_base(&uri, &url).map_err(js_error)?.href();
        resources.insert(uri, fetch_bytes(&resolved).await?);
    }
    set_status("Decoding the model…");
    next_tick().await;
    let asset = document.import(&resources).map_err(|e| e.to_string())?;
    let name = decoded.rsplit('/').next().and_then(|f| f.split('.').next()).filter(|n| !n.is_empty()).unwrap_or("model");
    Ok(Model::new(name, asset))
}

async fn next_tick() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        window().set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 30).unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    wasm_bindgen_futures::spawn_local(async {
        if let Err(message) = run().await {
            log::error!("{message}");
            set_status(&message);
        }
    });
}

async fn run() -> Result<(), String> {
    let document = window().document().ok_or("no document")?;
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("helios-canvas")
        .ok_or("missing #helios-canvas")?
        .dyn_into()
        .map_err(|_| "#helios-canvas is not a canvas")?;

    set_status("Initialising WebGPU…");
    let instance = GpuContext::create_instance();
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| format!("Could not create a WebGPU surface: {e}"))?;
    let ctx = GpuContext::new(instance, Some(&surface))
        .await
        .map_err(|e| format!("{e}. Use a browser with WebGPU enabled (Chrome/Edge 113+, Safari 26+, Firefox 141+)."))?;

    let (width, height) = canvas_size(&canvas);
    canvas.set_width(width);
    canvas.set_height(height);
    let caps = surface.get_capabilities(&ctx.adapter);
    let format = caps.formats.first().copied().ok_or("surface has no formats")?;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
        view_formats: vec![],
        ..surface.get_default_config(&ctx.adapter, width, height).ok_or("surface unsupported")?
    };
    surface.configure(&ctx.device, &config);

    let mut app = match query_param("stream") {
        // `?stream=URL_BASE`: the scene of the mesh library at URL_BASE (written by
        // `helios --bake-hvm`), its geometry streamed with HTTP range requests.
        Some(base) => {
            let base = js_sys::decode_uri_component(&base).map(String::from).unwrap_or(base);
            set_status(&format!("Loading the mesh library {base}…"));
            let library = mesh_store::fetch_library(&base).await?;
            set_status(&format!("Building the {} (textures)…", library.manifest.scene.name()));
            next_tick().await;
            DemoApp::from_library(&ctx.device, &ctx.queue, format, (width, height), library, renderer_config())
        }
        None => {
            let kind = scene_from_url();
            let model = match query_param("gltf") {
                Some(param) if kind == SceneKind::Model => Some(load_model(&param).await?),
                _ => None,
            };
            set_status(&format!("Building the {} (virtual geometry, distance fields, textures)…", kind.name()));
            next_tick().await;
            DemoApp::with_config(&ctx.device, &ctx.queue, format, width, height, kind, Detail::Medium, renderer_config(), model)
        }
    };
    configure_resolution(&mut app);
    // `?gi=lumen|lite` (page config `gi`); by default Lumen Lite once dynamic resolution can't
    // keep the budget even at its minimum.
    app.choose_gi(option_string("gi", "gi").as_deref(), &ctx.adapter.get_info());
    set_status("");

    let hud = document.get_element_by_id("hud").and_then(|e| e.dyn_into::<HtmlElement>().ok());
    if let Some(help) = document.get_element_by_id("help") {
        help.set_text_content(Some(HELP));
    }

    let state = Rc::new(RefCell::new(State {
        app,
        input: Input::default(),
        surface,
        device: ctx.device.clone(),
        queue: ctx.queue.clone(),
        config,
        canvas: canvas.clone(),
        hud,
        last_time: window().performance().map(|p| p.now()).unwrap_or(0.0),
        hud_timer: 0.0,
        paused: false,
    }));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    // Keep the context (instance/adapter/queue) alive for the page lifetime.
    std::mem::forget(ctx);

    install_input(&state, &canvas)?;
    start_loop(state);
    Ok(())
}

fn listen<E: wasm_bindgen::convert::FromWasmAbi + 'static>(target: &web_sys::EventTarget, name: &str, handler: impl FnMut(E) + 'static) -> Result<(), String> {
    let closure = Closure::<dyn FnMut(E)>::new(handler);
    target
        .add_event_listener_with_callback(name, closure.as_ref().unchecked_ref())
        .map_err(|_| format!("failed to listen for {name}"))?;
    closure.forget();
    Ok(())
}

fn install_input(state: &Rc<RefCell<State>>, canvas: &HtmlCanvasElement) -> Result<(), String> {
    let embedded = embedded();
    // Embedded, keys go to the focused canvas only, so the page keeps its own keys and scrolling.
    let keys: web_sys::EventTarget = if embedded { canvas.clone().into() } else { window().into() };
    let dragging = Rc::new(RefCell::new(false));

    let s = state.clone();
    listen(&keys, "keydown", move |e: web_sys::KeyboardEvent| {
        let code = e.code();
        if code.starts_with("Arrow") || code == "Space" {
            e.prevent_default();
        }
        let mut st = s.borrow_mut();
        if code == "KeyN" && embedded {
            // The page cannot be reloaded with another query: rebuild in place.
            let next = st.app.kind.next(false);
            st.app.load_scene(next);
            return;
        }
        if code == "KeyN" {
            // Scene switches re-bake everything: reload with the next scene instead of freezing.
            // The model stays in the cycle (re-fetched when it comes round again).
            let gltf = query_param("gltf");
            let next = st.app.kind.next(gltf.is_some());
            let model = gltf.map(|g| format!("&gltf={g}")).unwrap_or_default();
            let _ = window().location().set_search(&format!("scene={}{model}", next.name()));
            return;
        }
        if !e.repeat() {
            st.app.key_pressed(&code);
        }
        st.input.held.insert(code);
    })?;
    let s = state.clone();
    listen(&keys, "keyup", move |e: web_sys::KeyboardEvent| {
        s.borrow_mut().input.held.remove(&e.code());
    })?;
    let s = state.clone();
    listen(&keys, "blur", move |_: web_sys::Event| s.borrow_mut().input.held.clear())?;

    let target: web_sys::EventTarget = canvas.clone().into();
    let d = dragging.clone();
    let c = canvas.clone();
    listen(&target, "pointerdown", move |e: web_sys::PointerEvent| {
        *d.borrow_mut() = true;
        let _ = c.set_pointer_capture(e.pointer_id());
    })?;
    let d = dragging.clone();
    listen(&target, "pointerup", move |_: web_sys::PointerEvent| *d.borrow_mut() = false)?;
    let s = state.clone();
    let d = dragging;
    listen(&target, "pointermove", move |e: web_sys::PointerEvent| {
        if *d.borrow() {
            s.borrow_mut().input.look += Vec2::new(e.movement_x() as f32, e.movement_y() as f32);
        }
    })?;
    if !embedded {
        let s = state.clone();
        listen(&target, "wheel", move |e: web_sys::WheelEvent| {
            e.prevent_default();
            s.borrow_mut().input.scroll += e.delta_y().signum() as f32;
        })?;
    }
    Ok(())
}

fn start_loop(state: Rc<RefCell<State>>) {
    type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let callback: FrameCallback = Rc::new(RefCell::new(None));
    let next = callback.clone();
    *callback.borrow_mut() = Some(Closure::new(move |now: f64| {
        frame(&mut state.borrow_mut(), now);
        if let Some(cb) = next.borrow().as_ref() {
            let _ = window().request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }));
    let _ = window().request_animation_frame(callback.borrow().as_ref().unwrap().as_ref().unchecked_ref());
}

fn frame(st: &mut State, now: f64) {
    let dt = ((now - st.last_time) / 1000.0) as f32;
    st.last_time = now;
    if st.paused {
        return;
    }

    let (width, height) = canvas_size(&st.canvas);
    if (width, height) != (st.config.width, st.config.height) {
        st.canvas.set_width(width);
        st.canvas.set_height(height);
        st.config.width = width;
        st.config.height = height;
        st.surface.configure(&st.device, &st.config);
        st.app.resize(width, height);
    }

    st.app.update(&mut st.input, dt);
    let texture = match st.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            st.surface.configure(&st.device, &st.config);
            return;
        }
        _ => return,
    };
    let view = texture.texture.create_view(&Default::default());
    st.app.render(&view, dt);
    st.queue.present(texture);

    st.hud_timer += dt as f64;
    if st.hud_timer > 0.25 {
        st.hud_timer = 0.0;
        if let Some(hud) = &st.hud {
            hud.set_inner_text(&st.app.hud());
        }
    }
}

// ---- Page API ---------------------------------------------------------------------------------

/// Rebuilds the engine with another scene (`valley`, `cathedral` or `forest`); blocks for the
/// few seconds the bake takes, so show a notice and yield a frame before calling. Returns
/// whether the scene name was known and the engine running.
#[wasm_bindgen(js_name = heliosLoadScene)]
pub fn helios_load_scene(name: &str) -> bool {
    let Some(kind) = SceneKind::parse(name).filter(|k| *k != SceneKind::Model) else { return false };
    with_state(|st| st.app.load_scene(kind)).is_some()
}

/// Presses a hotkey once, by its `KeyboardEvent.code` (e.g. `KeyG` toggles GI), as if typed.
#[wasm_bindgen(js_name = heliosKey)]
pub fn helios_key(code: &str) {
    with_state(|st| st.app.key_pressed(code));
}

/// Places the camera at `eye` looking at `at`; heights are relative to the ground below each
/// point (as the native `--eye` / `--at` flags).
#[wasm_bindgen(js_name = heliosLookAt)]
pub fn helios_look_at(eye_x: f32, eye_h: f32, eye_z: f32, at_x: f32, at_h: f32, at_z: f32) {
    with_state(|st| {
        let app = &mut st.app;
        app.camera.position = glam::Vec3::new(eye_x, eye_h + app.ground_height(eye_x, eye_z), eye_z);
        let target = glam::Vec3::new(at_x, at_h + app.ground_height(at_x, at_z), at_z);
        app.camera.look_at(target);
    });
}

/// Places the sun (degrees; negative elevations give twilight and night).
#[wasm_bindgen(js_name = heliosSetSun)]
pub fn helios_set_sun(azimuth_deg: f32, elevation_deg: f32) {
    with_state(|st| st.app.set_sun(azimuth_deg.to_radians(), elevation_deg.to_radians()));
}

/// Stops or resumes rendering (for a canvas that is hidden or scrolled out of view).
#[wasm_bindgen(js_name = heliosSetPaused)]
pub fn helios_set_paused(paused: bool) {
    with_state(|st| st.paused = paused);
}

/// The HUD text (timings, statistics, feature states); empty until the engine runs.
#[wasm_bindgen(js_name = heliosHud)]
pub fn helios_hud() -> String {
    with_state(|st| st.app.hud()).unwrap_or_default()
}

/// The current scene's name; empty until the engine runs.
#[wasm_bindgen(js_name = heliosScene)]
pub fn helios_scene() -> String {
    with_state(|st| st.app.scene_name().to_string()).unwrap_or_default()
}

/// Selects a debug view by its index (0 = lit; see the native `--view` list).
#[wasm_bindgen(js_name = heliosSetView)]
pub fn helios_set_view(index: u32) {
    with_state(|st| {
        st.app.renderer.settings.debug_view = helios_render::DebugView::from_index(index as usize).unwrap_or_default();
    });
}
