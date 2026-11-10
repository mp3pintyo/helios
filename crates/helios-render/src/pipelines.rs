//! When pipelines are compiled: up front natively, progressively in the browser.
//!
//! Every pipeline is a shader compile. Natively (D3D12, Vulkan, Metal) a compile takes
//! milliseconds and the renderer simply creates its pipelines when it is built. In the browser
//! each one goes through the GPU process (WGSL → HLSL → DXC on Windows): 0.05–1.5 s, about 36 s
//! for the valley's set in Firefox, which compiles them one at a time even through the
//! asynchronous API, and as long in Chromium through the synchronous `createComputePipeline`
//! (the only one wgpu calls), which Chromium compiles one at a time on the thread that also
//! serves the page. Asking for all of them before the first frame meant a 30–60 s wait for the
//! first image on every visit.
//!
//! So pipelines are made through [`Lazy`], like Unreal's asynchronous shader compilation with
//! fallbacks:
//!
//! * Every pipeline belongs to a [`Group`]: [`Group::Core`] is what a frame needs at all (virtual
//!   geometry culling and rasterisation, the material resolve, sun and sky lighting, TSR, tone
//!   mapping), the others a feature each (shadows, Lumen, clouds, fog, translucency,
//!   tessellation…). A pipeline takes the group of the code that creates it ([`in_group`]).
//! * [`Lazy::background`] pipelines are made at once natively. In the browser ([`deferring`])
//!   they are queued; the page's frame loop [`pump`]s the queue. The renderer turns a feature off
//!   until its group is [`ready`], so the first image needs the core set only and every feature
//!   joins as its pipelines arrive; once all are in, the image is the one a native build renders.
//! * [`Lazy::new`] pipelines are made when a frame first uses them (content a scene may not have:
//!   skinning, alpha-masked or voxel clusters, a translucent shading model; MegaLights; editor
//!   tools and debug views). [`Lazy::prefetch`] queues one in the background once the scene is
//!   known to need it.
//!
//! How the queue is worked depends on the browser ([`browser::install`]):
//!
//! * **Parallel** (Chromium: Dawn compiles `create*PipelineAsync` on a thread pool and dedups
//!   pipelines by content): every queued pipeline is *warmed* — its constructor runs once with
//!   the device's `create*Pipeline` switched to the asynchronous call, and the result is dropped
//!   — and made for real once the compile finished, when the synchronous call returns Dawn's
//!   cached pipeline at once. The whole set compiles in a few seconds in parallel, without ever
//!   blocking the page.
//! * **Serial** (Firefox and the rest): the core set is created at once (it compiles in the GPU
//!   process while the main thread bakes the scene), the rest one pipeline per presented frame,
//!   each frame waiting for the previous compile instead of presenting behind it.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc as Shared, Weak};
#[cfg(target_arch = "wasm32")]
use std::rc::{Rc as Shared, Weak};

/// What a pipeline is needed for, in the order the browser compiles them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    /// Every frame: culling and rasterising the virtual geometry (one culling pass, no
    /// occlusion), the material resolve, sun and sky lighting, TSR, exposure, bloom, tone mapping.
    Core,
    /// Virtual shadow maps ([`crate::RenderSettings::shadows`]).
    Shadows,
    /// Volumetric clouds.
    Clouds,
    /// Volumetric fog.
    Fog,
    /// Lumen (screen probes, radiance cache, surface cache lighting, reflections).
    GlobalIllumination,
    /// Glass, water and blended surfaces.
    Translucency,
    /// Motion blur, depth of field and local exposure.
    CameraEffects,
    /// Nanite tessellation of displaced surfaces (main view and shadow pages).
    Tessellation,
    /// Two-pass HZB occlusion culling (the core set culls in one pass).
    OcclusionCulling,
    /// Heterogeneous volumes and particles.
    Effects,
}

impl Group {
    pub const ALL: [Group; 10] = [
        Group::Core,
        Group::Shadows,
        Group::Clouds,
        Group::Fog,
        Group::GlobalIllumination,
        Group::Translucency,
        Group::CameraEffects,
        Group::Tessellation,
        Group::OcclusionCulling,
        Group::Effects,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Group::Core => "core",
            Group::Shadows => "shadows",
            Group::Clouds => "clouds",
            Group::Fog => "fog",
            Group::GlobalIllumination => "global illumination",
            Group::Translucency => "translucency",
            Group::CameraEffects => "camera effects",
            Group::Tessellation => "tessellation",
            Group::OcclusionCulling => "occlusion culling",
            Group::Effects => "volumes and particles",
        }
    }
}

/// How the browser compiles queued pipelines (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compiler {
    /// One at a time: the core set at once, then one per [`make_next`].
    Serial,
    /// Warmed asynchronously all at once, made when warm ([`pump`]).
    Parallel,
}

// Entry states.
/// Made on first use, not queued.
const IDLE: u8 = 0;
/// Queued.
const WANTED: u8 = 1;
/// Compiling asynchronously (parallel compiler).
const WARMING: u8 = 2;
/// Compiled; making it now returns the browser's cached pipeline.
const WARM: u8 = 3;
const MADE: u8 = 4;

/// The constructor of a [`Lazy`] value (`Send + Sync` natively, where the renderer may move
/// between threads; wgpu's browser objects are neither).
#[cfg(not(target_arch = "wasm32"))]
pub trait MakeFn<T>: Fn() -> T + Send + Sync + 'static {}
#[cfg(not(target_arch = "wasm32"))]
impl<T, F: Fn() -> T + Send + Sync + 'static> MakeFn<T> for F {}
/// The constructor of a [`Lazy`] value (`Send + Sync` natively, where the renderer may move
/// between threads; wgpu's browser objects are neither).
#[cfg(target_arch = "wasm32")]
pub trait MakeFn<T>: Fn() -> T + 'static {}
#[cfg(target_arch = "wasm32")]
impl<T, F: Fn() -> T + 'static> MakeFn<T> for F {}

type Make<T> = Box<dyn MakeFn<T>>;

thread_local! {
    static DEFER: Cell<bool> = const { Cell::new(cfg!(target_arch = "wasm32")) };
    static GROUP: Cell<Group> = const { Cell::new(Group::Core) };
    static COMPILER: Cell<Compiler> = const { Cell::new(Compiler::Serial) };
    /// Queued entries (and those made since, for the progress count), in creation order.
    static QUEUE: RefCell<Vec<Weak<dyn Job>>> = const { RefCell::new(Vec::new()) };
}

/// Whether background pipelines are queued instead of made at once (in the browser).
pub fn deferring() -> bool {
    DEFER.with(Cell::get)
}

/// Queues background pipelines from now on (`true`) or makes them at once. On by default in
/// the browser only; tests switch it on to exercise the queue.
pub fn set_deferring(on: bool) {
    DEFER.with(|d| d.set(on));
}

/// The compiler the queue is worked with (set by [`browser::install`]).
pub fn compiler() -> Compiler {
    COMPILER.with(Cell::get)
}

pub fn set_compiler(compiler: Compiler) {
    COMPILER.with(|c| c.set(compiler));
}

/// Runs `f` with pipelines created in it (and in the constructors of [`Lazy`] values it creates)
/// belonging to `group`.
pub fn in_group<R>(group: Group, f: impl FnOnce() -> R) -> R {
    let previous = GROUP.with(|g| g.replace(group));
    let result = f();
    GROUP.with(|g| g.set(previous));
    result
}

fn current_group() -> Group {
    GROUP.with(Cell::get)
}

/// A value made later: on first use ([`Lazy::new`]) or, for pipelines every frame of a feature
/// needs, in the background ([`Lazy::background`]). Dereferences to the value (making it if it
/// is not made yet), so `pass.set_pipeline(&self.pipeline)` reads as before.
pub struct Lazy<T> {
    entry: Shared<Entry<T>>,
}

struct Entry<T> {
    value: OnceLock<T>,
    make: Make<T>,
    group: Group,
    state: AtomicU8,
}

/// A queue entry, whatever it makes.
trait Job {
    fn group(&self) -> Group;
    fn state(&self) -> u8;
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn set_state(&self, from: u8, to: u8);
    fn make(&self);
    /// Starts an asynchronous compile of what `make` would create; false if there was none (the
    /// constructor made no pipeline, or the browser cannot compile asynchronously).
    fn warm(&self, me: Weak<dyn Job>) -> bool;
}

impl<T> Entry<T> {
    fn get(&self) -> &T {
        self.value.get_or_init(|| {
            self.state.store(MADE, Ordering::Relaxed);
            // Pipelines created by the constructor (nested `Lazy` values) share its group.
            in_group(self.group, || (self.make)())
        })
    }
}

impl<T: 'static> Job for Entry<T> {
    fn group(&self) -> Group {
        self.group
    }

    fn state(&self) -> u8 {
        self.state.load(Ordering::Relaxed)
    }

    fn set_state(&self, from: u8, to: u8) {
        let _ = self.state.compare_exchange(from, to, Ordering::Relaxed, Ordering::Relaxed);
    }

    fn make(&self) {
        self.get();
    }

    #[cfg(target_arch = "wasm32")]
    fn warm(&self, me: Weak<dyn Job>) -> bool {
        let Some(compiled) = browser::warm(|| drop(in_group(self.group, || (self.make)()))) else { return false };
        self.state.store(WARMING, Ordering::Relaxed);
        wasm_bindgen_futures::spawn_local(async move {
            // A failed compile is made anyway: the synchronous call then reports the error.
            let _ = wasm_bindgen_futures::JsFuture::from(compiled).await;
            if let Some(job) = me.upgrade() {
                job.set_state(WARMING, WARM);
            }
        });
        true
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn warm(&self, _me: Weak<dyn Job>) -> bool {
        false
    }
}

impl<T: 'static> Lazy<T> {
    /// Made when first dereferenced (or queued by [`Lazy::prefetch`]).
    pub fn new(make: impl MakeFn<T>) -> Self {
        Self::with_state(Box::new(make), IDLE)
    }

    /// Made now, or queued while [`deferring`].
    pub fn background(make: impl MakeFn<T>) -> Self {
        Self::queued(Box::new(make))
    }

    fn queued(make: Make<T>) -> Self {
        let lazy = Self::with_state(make, IDLE);
        if deferring() {
            lazy.prefetch();
        } else {
            lazy.entry.get();
        }
        lazy
    }

    fn with_state(make: Make<T>, state: u8) -> Self {
        Self { entry: Shared::new(Entry { value: OnceLock::new(), make, group: current_group(), state: AtomicU8::new(state) }) }
    }

    /// Queues a pipeline made on first use for the background, because a frame will need it
    /// soon (the scene has that content). Does nothing natively, or once it is queued or made.
    pub fn prefetch(&self) {
        if deferring() && self.entry.state.compare_exchange(IDLE, WANTED, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            let entry: Shared<dyn Job> = self.entry.clone();
            let job = Shared::downgrade(&entry);
            QUEUE.with(|q| q.borrow_mut().push(job));
        }
    }

    /// Whether the value is made.
    pub fn is_made(&self) -> bool {
        self.entry.value.get().is_some()
    }

    /// The group the value belongs to.
    pub fn group(&self) -> Group {
        self.entry.group
    }
}

impl<T> std::ops::Deref for Lazy<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.entry.get()
    }
}

/// The live queue entries (made ones included), in priority order: by group, then creation.
fn entries() -> Vec<Shared<dyn Job>> {
    let mut live: Vec<Shared<dyn Job>> = QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        q.retain(|j| j.strong_count() > 0);
        q.iter().filter_map(Weak::upgrade).collect()
    });
    live.sort_by_key(|j| j.group());
    live
}

fn pending(state: u8) -> bool {
    matches!(state, WANTED | WARMING | WARM)
}

/// Whether every queued pipeline of `group` is made (always, natively).
pub fn ready(group: Group) -> bool {
    QUEUE.with(|q| q.borrow().iter().filter_map(Weak::upgrade).all(|j| j.group() != group || !pending(j.state())))
}

/// Queued pipelines made so far and in all (made ones stay counted while they live).
pub fn progress() -> (usize, usize) {
    let entries = entries();
    (entries.iter().filter(|j| j.state() == MADE).count(), entries.len())
}

/// Groups with pipelines still to make, in priority order.
pub fn waiting() -> Vec<Group> {
    let mut groups: Vec<Group> = entries().iter().filter(|j| pending(j.state())).map(|j| j.group()).collect();
    groups.dedup();
    groups
}

/// Works the queue once per frame. Parallel: starts warming every queued pipeline and makes the
/// warm ones (no compile: the browser returns the pipeline it compiled). Serial: makes the queued
/// core pipelines, all at once (the first image needs every one). Returns how many it made.
pub fn pump() -> usize {
    let mut made = 0;
    match compiler() {
        Compiler::Parallel => {
            for job in entries() {
                match job.state() {
                    WANTED => {
                        if !job.warm(Shared::downgrade(&job)) {
                            job.make();
                            made += 1;
                        }
                    }
                    WARM => {
                        browser_uncounted(|| job.make());
                        made += 1;
                    }
                    _ => {}
                }
            }
        }
        Compiler::Serial => {
            // Making a value may queue the pipelines it holds: until none are left.
            loop {
                let core: Vec<_> = entries().into_iter().filter(|j| j.group() == Group::Core && j.state() == WANTED).collect();
                if core.is_empty() {
                    break;
                }
                for job in core {
                    job.make();
                    made += 1;
                }
            }
        }
    }
    made
}

/// Serial compiler: makes the next queued pipeline (by group, then creation). Returns whether
/// there was one.
pub fn make_next() -> bool {
    match entries().into_iter().find(|j| j.state() == WANTED) {
        Some(job) => {
            job.make();
            true
        }
        None => false,
    }
}

#[cfg(target_arch = "wasm32")]
fn browser_uncounted(f: impl FnOnce()) {
    browser::uncounted(f);
}

#[cfg(not(target_arch = "wasm32"))]
fn browser_uncounted(f: impl FnOnce()) {
    f();
}

/// The browser side: the device's pipeline creation hooks.
#[cfg(target_arch = "wasm32")]
pub mod browser {
    use std::cell::{Cell, RefCell};

    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    use super::Compiler;

    thread_local! {
        /// Pipelines created on the device so far (warm adoptions excluded).
        static CREATED: Cell<u32> = const { Cell::new(0) };
        /// Inside [`warm`]: creations go to the asynchronous call.
        static WARMING: Cell<bool> = const { Cell::new(false) };
        static UNCOUNTED: Cell<bool> = const { Cell::new(false) };
        static COMPILES: RefCell<Vec<js_sys::Promise>> = const { RefCell::new(Vec::new()) };
    }

    /// Wraps the `GPUDevice` object's `createComputePipeline` and `createRenderPipeline` (wgpu
    /// calls them as methods of the object, so own properties take precedence over the
    /// prototype's): they count the pipelines created, so a frame loop can tell whether a frame
    /// queued compiles ([`created`]), and inside [`warm`] they start the asynchronous compile
    /// instead and return a stand-in the caller drops. `parallel` selects the parallel compiler
    /// (where the browser compiles asynchronous pipelines in parallel and returns a compiled
    /// pipeline for an identical descriptor at once: Chromium's Dawn). Starts deferring.
    pub fn install(device: &wgpu::Device, parallel: bool) {
        let Some(gpu) = device.as_webgpu() else { return };
        let gpu: &JsValue = gpu.as_ref();
        let mut asynchronous = true;
        for name in ["createComputePipeline", "createRenderPipeline"] {
            let get = |name: &str| js_sys::Reflect::get(gpu, &JsValue::from_str(name)).ok().and_then(|f| f.dyn_into::<js_sys::Function>().ok());
            let Some(original) = get(name) else { continue };
            let async_call = get(&format!("{name}Async"));
            asynchronous &= async_call.is_some();
            let this = gpu.clone();
            let wrapper = Closure::<dyn Fn(JsValue) -> Result<JsValue, JsValue>>::new(move |descriptor: JsValue| {
                if let Some(call) = async_call.as_ref().filter(|_| WARMING.with(Cell::get)) {
                    let compiled: js_sys::Promise = call.call1(&this, &descriptor)?.unchecked_into();
                    COMPILES.with(|c| c.borrow_mut().push(compiled));
                    return Ok(js_sys::Object::new().into());
                }
                if !UNCOUNTED.with(Cell::get) {
                    CREATED.with(|c| c.set(c.get() + 1));
                }
                original.call1(&this, &descriptor)
            });
            let _ = js_sys::Reflect::set(gpu, &JsValue::from_str(name), wrapper.as_ref());
            wrapper.forget();
        }
        super::set_compiler(if parallel && asynchronous { Compiler::Parallel } else { Compiler::Serial });
        super::set_deferring(true);
    }

    /// Pipelines created on the device so far (made from a warm compile: not counted).
    pub fn created() -> u32 {
        CREATED.with(Cell::get)
    }

    /// Runs a constructor with pipeline creation switched to the asynchronous calls; the
    /// promise resolves (or rejects) once every pipeline it created is compiled.
    pub(super) fn warm(make: impl FnOnce()) -> Option<js_sys::Promise> {
        WARMING.with(|w| w.set(true));
        make();
        WARMING.with(|w| w.set(false));
        let compiles = COMPILES.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if compiles.is_empty() {
            return None;
        }
        let all: js_sys::Array = compiles.into_iter().collect();
        Some(js_sys::Promise::all(&all))
    }

    pub(super) fn uncounted(f: impl FnOnce()) {
        UNCOUNTED.with(|u| u.set(true));
        f();
        UNCOUNTED.with(|u| u.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    fn counter() -> (Arc<AtomicU32>, impl Fn() -> u32 + Send + Sync + Clone + 'static) {
        let made = Arc::new(AtomicU32::new(0));
        let c = made.clone();
        (made, move || c.fetch_add(1, Ordering::Relaxed) + 1)
    }

    #[test]
    fn lazy_makes_its_value_once_on_first_use() {
        let (made, make) = counter();
        let lazy = Lazy::new(make);
        assert_eq!(made.load(Ordering::Relaxed), 0);
        assert_eq!(*lazy + *lazy, 2);
        assert_eq!(made.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn background_values_are_made_at_once_unless_deferring() {
        let (made, make) = counter();
        let lazy = Lazy::background(make);
        assert!(lazy.is_made());
        assert_eq!(made.load(Ordering::Relaxed), 1);
        assert!(Group::ALL.iter().all(|&g| ready(g)));
    }

    #[test]
    fn serial_compiler_makes_the_core_set_then_one_by_one_in_group_order() {
        set_deferring(true);
        set_compiler(Compiler::Serial);
        let (made, make) = counter();
        let fog = in_group(Group::Fog, || Lazy::background(make.clone()));
        let shadows = in_group(Group::Shadows, || [Lazy::background(make.clone()), Lazy::background(make.clone())]);
        let core = [Lazy::background(make.clone()), Lazy::background(make.clone())];
        let rare = Lazy::new(make.clone());
        assert_eq!(made.load(Ordering::Relaxed), 0);
        assert!(!ready(Group::Core) && !ready(Group::Shadows) && !ready(Group::Fog));
        assert!(ready(Group::Clouds));
        assert_eq!(progress(), (0, 5));

        assert_eq!(pump(), 2);
        assert!(core.iter().all(Lazy::is_made));
        assert!(ready(Group::Core));
        assert_eq!(waiting(), vec![Group::Shadows, Group::Fog]);

        assert!(make_next());
        assert!(shadows[0].is_made() && !shadows[1].is_made() && !ready(Group::Shadows));
        assert!(make_next());
        assert!(ready(Group::Shadows) && !ready(Group::Fog));
        assert!(make_next());
        assert!(fog.is_made());
        assert!(!make_next());
        assert_eq!(progress(), (5, 5));

        // A value made on first use is only queued once the scene needs it.
        assert!(!rare.is_made() && ready(Group::Core));
        rare.prefetch();
        assert!(!ready(Group::Core));
        assert_eq!(pump(), 1);
        assert!(rare.is_made());
        set_deferring(false);
    }

    #[test]
    fn nested_values_inherit_their_group_and_dropped_ones_leave_the_queue() {
        set_deferring(true);
        set_compiler(Compiler::Serial);
        let (_, make) = counter();
        let outer = in_group(Group::Shadows, || {
            let make = make.clone();
            Lazy::background(move || Lazy::background(make.clone()))
        });
        let dropped = in_group(Group::Clouds, || Lazy::background(make.clone()));
        drop(dropped);
        assert!(ready(Group::Clouds));
        assert!(make_next());
        // The inner value was queued, in the outer's group, when the outer was made.
        assert!(outer.is_made() && !outer.is_made_inner());
        assert!(!ready(Group::Shadows));
        assert!(make_next());
        assert!(ready(Group::Shadows));
        set_deferring(false);
    }

    impl Lazy<Lazy<u32>> {
        fn is_made_inner(&self) -> bool {
            self.entry.value.get().is_some_and(|inner| inner.is_made())
        }
    }
}
