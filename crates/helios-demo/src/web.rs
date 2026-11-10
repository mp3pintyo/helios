//! Browser platform layer: canvas surface, DOM input and a requestAnimationFrame loop.
//!
//! A page can embed the engine instead of using `index.html`'s full-window demo: it sets
//! `globalThis.heliosConfig = { scene, screenPercentage, gpuBudgetMs, embedded, lights, megaLights, statusElement }` before
//! loading the module (query parameters still work and take part as fallbacks), and drives the
//! running engine through the exported `helios*` functions (scene, glTF model, candle lights,
//! camera, sun, hotkeys, HUD).
//! `embedded: true` keeps keyboard input on the focused canvas and leaves the mouse wheel to
//! the page, so the engine can sit inside a scrolling document.
//!
//! **Shader compilation never blocks the page.** The renderer creates ~130 pipelines up
//! front and more on first use (a scene's material permutations, a debug view, MegaLights).
//! WebGPU's `createComputePipeline` returns at once, but the browser's GPU process compiles each
//! pipeline (WGSL → HLSL → DXC on Windows) before it runs anything queued after it: 36 s for the
//! valley's set in Firefox, which compiles them one at a time and keeps no cache between visits,
//! and as long in Chrome on a first visit. The page's next paint waited for a frame presented
//! behind that queue, so it stopped painting, scrolling and reacting for the whole time ("the
//! browser freezes"). So the engine renders each frame into its own texture and
//! copies it to the canvas only if the frame created no pipeline; otherwise it skips the canvas,
//! waits (`onSubmittedWorkDone`, which resolves once the GPU process has worked through the
//! compiles) while the page keeps running, and shows the wait in the status element.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use glam::Vec2;
use helios_render::{DynamicResolution, GpuContext};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, HtmlElement};

use crate::app::{DemoApp, Input, HELP};
use crate::mesh_store;
use crate::scenes::model::Model;
use crate::scenes::{Detail, SceneKind};

pub(crate) struct State {
    pub(crate) app: DemoApp,
    pub(crate) input: Input,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) canvas: HtmlCanvasElement,
    /// Editor mode: the right mouse button is held (the camera flies with WASD/QE).
    looking: bool,
    hud: Option<HtmlElement>,
    last_time: f64,
    hud_timer: f64,
    /// Skip rendering (the page hides the canvas or scrolls it out of view).
    paused: bool,
    /// The frame is rendered here and copied to the canvas only when it compiled nothing (see
    /// the module docs).
    frame_target: wgpu::Texture,
    /// Waiting for the GPU process to finish compiling pipelines.
    compiling: Option<Compiling>,
    /// Frames in a row that created pipelines (a frame that keeps creating them is presented
    /// anyway after a few, rather than never).
    compiling_frames: u32,
}

/// A wait for the browser's GPU process to work through the pipelines created so far.
struct Compiling {
    /// `GPU_DRAINED` reaches this once the wait is over.
    generation: u32,
    /// When the wait began (ms, `performance.now()`).
    since: f64,
}

/// Generation of the last `onSubmittedWorkDone` that resolved (the callback must be `Send`).
static GPU_DRAINED: AtomicU32 = AtomicU32::new(0);
/// Generation of the last wait asked for.
static GPU_WAIT: AtomicU32 = AtomicU32::new(0);

thread_local! {
    /// The running engine, for the exported page API.
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
    /// Pipelines created on the device so far (counted by `count_pipelines`).
    static PIPELINES: Cell<u32> = const { Cell::new(0) };
    /// What the engine is waiting for (`heliosBusy`); empty while it renders.
    static BUSY: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Counts every pipeline the device creates, by wrapping the `GPUDevice` object's
/// `createComputePipeline` and `createRenderPipeline` (wgpu calls them as methods of the object,
/// so own properties take precedence over the prototype's): the frame loop can then tell whether
/// a frame queued compiles without any bookkeeping in the renderer.
fn count_pipelines(device: &wgpu::Device) {
    let Some(gpu) = device.as_webgpu() else { return };
    let gpu: &JsValue = gpu.as_ref();
    for name in ["createComputePipeline", "createRenderPipeline"] {
        let Ok(original) = js_sys::Reflect::get(gpu, &JsValue::from_str(name)).and_then(|f| f.dyn_into::<js_sys::Function>()) else {
            continue;
        };
        let this = gpu.clone();
        let wrapper = Closure::<dyn Fn(JsValue) -> Result<JsValue, JsValue>>::new(move |descriptor: JsValue| {
            PIPELINES.with(|c| c.set(c.get() + 1));
            original.call1(&this, &descriptor)
        });
        let _ = js_sys::Reflect::set(gpu, &JsValue::from_str(name), wrapper.as_ref());
        wrapper.forget();
    }
}

fn pipelines_created() -> u32 {
    PIPELINES.with(Cell::get)
}

/// Starts a wait for everything queued so far, pipeline compiles included.
fn wait_for_gpu(queue: &wgpu::Queue) -> Compiling {
    let generation = GPU_WAIT.fetch_add(1, Ordering::Relaxed) + 1;
    queue.on_submitted_work_done(move || {
        GPU_DRAINED.fetch_max(generation, Ordering::Relaxed);
    });
    Compiling { generation, since: now() }
}

fn now() -> f64 {
    window().performance().map(|p| p.now()).unwrap_or(0.0)
}

/// The texture a frame is rendered into before it is copied to the canvas.
fn frame_target(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Runs `f` on the running engine (`None` until it has started).
pub(crate) fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.try_borrow_mut().ok()?;
    Some(f(&mut st))
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no window")
}

/// Writes a progress or error line into the page's status element: `#status`, or the id in
/// `heliosConfig.statusElement` (a page that shows its own, translated progress text points it at
/// a hidden element and reads errors from there).
fn set_status(text: &str) {
    let id = page_config("statusElement").and_then(|v| v.as_string()).unwrap_or_else(|| "status".into());
    if let Some(el) = window().document().and_then(|d| d.get_element_by_id(&id)) {
        el.set_text_content(Some(text));
    }
}

/// Shows what the engine waits for in the status element and `heliosBusy` (empty: nothing).
fn set_busy(text: &str) {
    let changed = BUSY.with(|b| {
        let mut b = b.borrow_mut();
        let changed = *b != text;
        if changed {
            *b = text.to_string();
        }
        changed
    });
    if changed {
        set_status(text);
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

/// Scene selected with `?scene=valley|cathedral|forest|gallery|model`; `?gltf=URL` alone selects the model.
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

pub(crate) async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
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
pub(crate) async fn load_model(param: &str) -> Result<Model, String> {
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
    let mut model = Model::new(name, asset);
    model.source = decoded;
    Ok(model)
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
    count_pipelines(&ctx.device);

    let (width, height) = canvas_size(&canvas);
    canvas.set_width(width);
    canvas.set_height(height);
    let caps = surface.get_capabilities(&ctx.adapter);
    let format = caps.formats.first().copied().ok_or("surface has no formats")?;
    let config = wgpu::SurfaceConfiguration {
        // Frames are copied in from `frame_target` (see the module docs).
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
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

    // `?scene-file=URL`: a saved editor scene; its base scene is built instead of `?scene`.
    let scene_file = match query_param("scene-file") {
        Some(url) => {
            let url = js_sys::decode_uri_component(&url).map(String::from).unwrap_or(url);
            set_status(&format!("Loading the scene file {url}…"));
            Some(crate::web_editor::fetch_document(&url).await?)
        }
        None => None,
    };
    // `?lights=N`: the cathedral lit by N individual candles (see scenes/candles.rs).
    if let Some(count) = option_string("lights", "lights").and_then(|v| v.parse().ok()) {
        crate::scenes::candles::set_candle_lights(count);
    }
    // `?lantern=1`: a lantern carried down the cathedral's nave (a fast, sharp moving light).
    if option_string("lantern", "lantern").is_some_and(|v| v != "0") {
        crate::scenes::cathedral::set_lantern(true);
    }
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
            let (mut kind, mut detail) = (scene_from_url(), Detail::Medium);
            let mut gltf = query_param("gltf");
            match scene_file.as_ref().map(|doc| &doc.base) {
                Some(crate::editor::doc::BaseScene::Gltf { url, .. }) => (kind, gltf) = (SceneKind::Model, Some(url.clone())),
                Some(base) => (kind, detail) = crate::app_editor::procedural_base(base).unwrap_or((kind, detail)),
                None => {}
            }
            let model = match gltf {
                Some(param) if kind == SceneKind::Model => Some(load_model(&param).await?),
                _ => None,
            };
            set_status(&format!("Building the {} (virtual geometry, distance fields, textures)…", kind.name()));
            next_tick().await;
            DemoApp::with_config(&ctx.device, &ctx.queue, format, width, height, kind, detail, renderer_config(), model)
        }
    };
    configure_resolution(&mut app);
    // `?gi=lumen|lite` (page config `gi`); by default Lumen Lite once dynamic resolution can't
    // keep the budget even at its minimum.
    app.choose_gi(option_string("gi", "gi").as_deref(), &ctx.adapter.get_info());
    // `?megalights=1`: MegaLights instead of the clustered light loop (key `=` toggles).
    app.renderer.settings.mega_lights = matches!(option_string("megaLights", "megalights").as_deref(), Some("1" | "true"));
    set_status("");

    let hud = document.get_element_by_id("hud").and_then(|e| e.dyn_into::<HtmlElement>().ok());
    if let Some(help) = document.get_element_by_id("help") {
        help.set_text_content(Some(HELP));
    }

    let target = frame_target(&ctx.device, &config);
    let state = Rc::new(RefCell::new(State {
        app,
        input: Input::default(),
        surface,
        device: ctx.device.clone(),
        queue: ctx.queue.clone(),
        config,
        canvas: canvas.clone(),
        hud,
        last_time: now(),
        hud_timer: 0.0,
        paused: false,
        looking: false,
        frame_target: target,
        // The renderer's pipelines are compiling now: wait for them before the first frame.
        compiling: Some(wait_for_gpu(&ctx.queue)),
        compiling_frames: 0,
    }));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    // Keep the context (instance/adapter/queue) alive for the page lifetime.
    std::mem::forget(ctx);

    install_input(&state, &canvas)?;
    start_loop(state);
    if let Some(doc) = scene_file {
        set_status("Applying the scene file…");
        crate::web_editor::load_document(doc).await?;
        set_status("");
    }
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
        // PageDown toggles tessellation instead of scrolling the page.
        if code.starts_with("Arrow") || code == "Space" || code == "PageDown" {
            e.prevent_default();
        }
        let mut st = s.borrow_mut();
        if st.app.editing {
            // The editor UI owns the keyboard; while the right button is held, the camera
            // flies (WASD / QE, Shift) as in UE's viewport.
            if st.looking {
                st.input.held.insert(code);
            }
            return;
        }
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
    let s = state.clone();
    listen(&target, "pointerdown", move |e: web_sys::PointerEvent| {
        let mut st = s.borrow_mut();
        if st.app.editing {
            // Left clicks select and drag gizmos (the editor UI); the right button looks.
            if e.button() != 2 {
                return;
            }
            st.looking = true;
        }
        *d.borrow_mut() = true;
        let _ = c.set_pointer_capture(e.pointer_id());
    })?;
    let d = dragging.clone();
    let s = state.clone();
    listen(&target, "pointerup", move |_: web_sys::PointerEvent| {
        *d.borrow_mut() = false;
        let mut st = s.borrow_mut();
        if std::mem::take(&mut st.looking) {
            st.input.held.clear();
        }
    })?;
    listen(&target, "contextmenu", move |e: web_sys::Event| e.prevent_default())?;
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
            let mut st = s.borrow_mut();
            if st.app.editing && !st.looking {
                // The editor's wheel dollies; with the right button held it sets the speed.
                st.app.dolly(-e.delta_y().signum() as f32);
                return;
            }
            st.input.scroll += e.delta_y().signum() as f32;
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
    if let Some(wait) = &st.compiling {
        let seconds = (now - wait.since) / 1000.0;
        if GPU_DRAINED.load(Ordering::Relaxed) < wait.generation {
            // Short waits (a pipeline or two) pass without a message.
            if seconds > 0.5 {
                set_busy(&format!("Compiling GPU shaders… {seconds:.0} s"));
            }
            return;
        }
        if seconds > 0.5 {
            log::info!("waited {seconds:.1} s for the GPU process to compile pipelines ({} created so far)", pipelines_created());
        }
        st.compiling = None;
        set_busy("");
    }

    let (width, height) = canvas_size(&st.canvas);
    if (width, height) != (st.config.width, st.config.height) {
        st.canvas.set_width(width);
        st.canvas.set_height(height);
        st.config.width = width;
        st.config.height = height;
        st.surface.configure(&st.device, &st.config);
        st.frame_target = frame_target(&st.device, &st.config);
        st.app.resize(width, height);
    }

    st.app.update(&mut st.input, dt);
    let pipelines = pipelines_created();
    let view = st.frame_target.create_view(&Default::default());
    st.app.render(&view, dt);
    if pipelines_created() != pipelines && st.compiling_frames < 8 {
        // This frame queued compiles: presenting it would stall the page until they finish.
        st.compiling_frames += 1;
        st.compiling = Some(wait_for_gpu(&st.queue));
        return;
    }
    st.compiling_frames = 0;
    let texture = match st.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            st.surface.configure(&st.device, &st.config);
            return;
        }
        _ => return,
    };
    let mut encoder = st.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present") });
    encoder.copy_texture_to_texture(
        st.frame_target.as_image_copy(),
        texture.texture.as_image_copy(),
        wgpu::Extent3d { width: st.config.width, height: st.config.height, depth_or_array_layers: 1 },
    );
    st.queue.submit([encoder.finish()]);
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

/// What the engine is waiting for, e.g. "Compiling GPU shaders… 12 s" while the browser compiles
/// the pipelines of the first frame, a new scene or a feature switched on (the canvas keeps its
/// last frame meanwhile); empty while it renders. A page can show it over the canvas.
#[wasm_bindgen(js_name = heliosBusy)]
pub fn helios_busy() -> String {
    BUSY.with(|b| b.borrow().clone())
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

/// Sets how many candle flames light the cathedral from its next build (`heliosLoadScene`):
/// every flame a shadow-casting light, up to 1008 (`heliosConfig.lights` at start); 0 restores
/// the 33 fixture lights.
#[wasm_bindgen(js_name = heliosSetCandleLights)]
pub fn helios_set_candle_lights(count: u32) {
    crate::scenes::candles::set_candle_lights(count as usize);
}

/// Sets whether a lantern is carried down the cathedral's nave from its next build
/// (`heliosLoadScene`; `?lantern=1` at start): a fast, sharp moving light.
#[wasm_bindgen(js_name = heliosSetLantern)]
pub fn helios_set_lantern(on: bool) {
    crate::scenes::cathedral::set_lantern(on);
}

/// Fetches a glTF model (`.glb`, or `.gltf` with its files; relative to the page, as `?gltf=`)
/// and rebuilds the engine with it as the scene; skinned models play their first clip. Resolves
/// to whether the engine was running; rejects with the import error.
#[wasm_bindgen(js_name = heliosLoadModel)]
pub async fn helios_load_model(url: String) -> Result<bool, JsValue> {
    let model = load_model(&url).await.map_err(|e| JsValue::from_str(&e))?;
    Ok(with_state(|st| {
        st.app.model = Some(model);
        st.app.load_scene(SceneKind::Model);
    })
    .is_some())
}
