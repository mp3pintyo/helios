//! Quality levels for the GPU the page runs on (round 9).
//!
//! Browsers don't report the adapter type: wgpu's WebGPU backend gives every adapter the type
//! `Other`, so [`helios_render::GiMethod::for_adapter`] never sees an integrated GPU there.
//! The page tells the engine what it knows instead: the WebGPU adapter's vendor and
//! architecture and the WebGL renderer string (`gpu_info.js`). [`classify`] turns that into a
//! [`GpuClass`], and the class picks the starting [`Quality`]:
//! * discrete GPUs: `High`, the full renderer;
//! * Apple silicon: `Medium`;
//! * integrated, mobile and software adapters: `Low`.
//!
//! The guess can be wrong both ways. A laptop's discrete GPU may be weak, Firefox names only a
//! generic model, and a large integrated GPU (Radeon 8060S) is as fast as a mid-range card. So
//! [`Governor`] watches the GPU time while the page runs and moves one level down when the GPU
//! cannot keep the budget even at dynamic resolution's minimum. It moves one level up when the
//! frame stays far under the budget at full resolution. It never goes back up to a level it
//! left for being too slow.
//!
//! Each level is a set of render settings applied on top of what the scene and the page asked
//! for ([`Quality::apply`]). The costs behind the choices were measured on an AMD Radeon
//! Graphics iGPU (Ryzen 7600X, 2 RDNA 2 compute units) in Chromium: see
//! `docs/round9/quality-levels.md`.

use helios_render::{DynamicResolution, GiMethod, RenderSettings};

/// What kind of GPU the page runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuClass {
    Discrete,
    /// Shares memory with the CPU (Intel UHD / Iris / Arc in Core Ultra CPUs, AMD Radeon
    /// Graphics in Ryzen CPUs).
    Integrated,
    /// Apple silicon: integrated, but from M1 on about as fast as a mid-range card.
    Apple,
    /// Phones and tablets, and Windows on Arm laptops (Adreno, Mali, PowerVR).
    Mobile,
    /// A CPU rasteriser (SwiftShader, WARP, llvmpipe) or the browser's fallback adapter.
    Software,
    /// The browser said too little.
    Unknown,
}

impl GpuClass {
    pub const ALL: [Self; 6] = [Self::Discrete, Self::Integrated, Self::Apple, Self::Mobile, Self::Software, Self::Unknown];

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == value)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Integrated => "integrated",
            Self::Apple => "apple",
            Self::Mobile => "mobile",
            Self::Software => "software",
            Self::Unknown => "unknown",
        }
    }
}

/// What `gpu_info.js` reports.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct GpuInfo {
    /// `GPUAdapterInfo.vendor`, e.g. "intel", "amd", "nvidia", "apple".
    pub vendor: String,
    /// `GPUAdapterInfo.architecture`, e.g. "xe-lpg", "rdna-2", "ampere".
    pub architecture: String,
    pub description: String,
    /// The WebGL renderer string (Chromium: the model, through ANGLE).
    pub renderer: String,
    pub fallback: bool,
}

/// The vendor named by the adapter, or else found in the renderer string.
fn vendor(info: &GpuInfo, text: &str) -> &'static str {
    let named = info.vendor.to_ascii_lowercase();
    let hay = if named.is_empty() { text } else { named.as_str() };
    const VENDORS: &[(&str, &[&str])] = &[
        ("nvidia", &["nvidia", "geforce", "quadro", "rtx", "gtx", "tegra"]),
        ("amd", &["amd", "ati", "radeon"]),
        ("intel", &["intel"]),
        ("apple", &["apple"]),
        ("qualcomm", &["qualcomm", "adreno"]),
        ("arm", &["arm", "mali"]),
        ("imagination", &["imagination", "powervr"]),
        ("samsung", &["samsung", "xclipse"]),
    ];
    VENDORS.iter().find(|(_, words)| words.iter().any(|w| hay.contains(w))).map_or("", |(v, _)| v)
}

/// True when `text` contains `prefix` followed by a digit (e.g. "arc a7" for "Arc A770").
fn followed_by_digit(text: &str, prefix: &str) -> bool {
    text.match_indices(prefix).any(|(i, _)| text[i + prefix.len()..].starts_with(|c: char| c.is_ascii_digit()))
}

/// Classifies the GPU from what the browser reports.
pub fn classify(info: &GpuInfo) -> GpuClass {
    let text = format!("{} {}", info.renderer, info.description).to_ascii_lowercase().replace("(tm)", "").replace("(r)", "");
    let architecture = info.architecture.to_ascii_lowercase();
    const SOFTWARE: &[&str] = &["swiftshader", "llvmpipe", "lavapipe", "softpipe", "basic render", "microsoft warp", "software"];
    if info.fallback || SOFTWARE.iter().any(|s| text.contains(s)) {
        return GpuClass::Software;
    }
    // Firefox reports a generic model of the same family ("Radeon R9 200 Series, or similar").
    let generic = text.contains("or similar");
    match vendor(info, &text) {
        "nvidia" if text.contains("tegra") => GpuClass::Mobile,
        "nvidia" => GpuClass::Discrete,
        "intel" => {
            // Discrete Arc: A- and B-series cards (Alchemist "xe-hpg", Battlemage "xe2-hpg").
            // "Intel Arc Graphics" without a model number is the iGPU of Core Ultra CPUs.
            if architecture.contains("hpg") || followed_by_digit(&text, "arc a") || followed_by_digit(&text, "arc b") {
                GpuClass::Discrete
            } else {
                GpuClass::Integrated
            }
        }
        "amd" => {
            // Strix Halo (Radeon 8060S / 8050S / 8040S): integrated, but a mid-range card's speed.
            if ["8060s", "8050s", "8040s"].iter().any(|m| text.contains(m)) {
                GpuClass::Discrete
            } else if amd_integrated(&text) {
                GpuClass::Integrated
            } else if generic || text.trim().is_empty() || !text.contains("radeon") && !text.contains("firepro") && !text.contains("amd") {
                GpuClass::Unknown
            } else {
                GpuClass::Discrete
            }
        }
        "apple" => GpuClass::Apple,
        "qualcomm" | "arm" | "imagination" | "samsung" => GpuClass::Mobile,
        _ => GpuClass::Unknown,
    }
}

/// AMD's iGPUs: "AMD Radeon(TM) Graphics" (Ryzen 7000/9000 desktop, 2 CUs; many laptop chips),
/// "Radeon 780M" and the other three-digit M models, "Radeon Vega 8 Graphics", the older APUs'
/// "Radeon R7 Graphics".
fn amd_integrated(text: &str) -> bool {
    if text.contains(" rx ") || text.contains("rx ") && followed_by_digit(text, "rx ") {
        return false;
    }
    let m_model = text.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w.len() == 4 && w.ends_with('m') && w[..3].chars().all(|c| c.is_ascii_digit()));
    m_model
        || text.contains("radeon graphics")
        || text.contains("vega") && text.contains("graphics")
        || ["radeon r2 graphics", "radeon r3 graphics", "radeon r4 graphics", "radeon r5 graphics", "radeon r6 graphics", "radeon r7 graphics"].iter().any(|m| text.contains(m))
}

/// A quality level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Quality {
    Low,
    Medium,
    High,
}

impl Quality {
    pub fn name(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    /// The level a GPU class starts at.
    pub fn for_class(class: GpuClass) -> Self {
        match class {
            GpuClass::Discrete | GpuClass::Unknown => Self::High,
            GpuClass::Apple => Self::Medium,
            GpuClass::Integrated | GpuClass::Mobile | GpuClass::Software => Self::Low,
        }
    }

    fn lower(self) -> Option<Self> {
        match self {
            Self::High => Some(Self::Medium),
            Self::Medium => Some(Self::Low),
            Self::Low => None,
        }
    }

    fn higher(self) -> Option<Self> {
        match self {
            Self::Low => Some(Self::Medium),
            Self::Medium => Some(Self::High),
            Self::High => None,
        }
    }

    /// The level's render settings on top of `base` (what the scene and the page asked for:
    /// `High` returns it unchanged). `keep_gi`: the page chose the GI method itself.
    pub fn apply(self, base: &RenderSettings, keep_gi: bool) -> RenderSettings {
        let mut s = *base;
        if self == Self::High {
            return s;
        }
        // Both lower levels: Lumen Lite's irradiance field instead of the screen probes (the
        // largest cost: 25–160 ms of a 2-CU iGPU's frame, 70 % of it saved), the cheaper OIT,
        // no per-pixel volume shadows, coarser geometry.
        if !keep_gi {
            s.gi_method = GiMethod::LumenLite;
            s.gi_lite_fallback = false;
        }
        s.avboit = false;
        s.volume_shadows = false;
        s.lod_threshold = s.lod_threshold.max(1.5);
        s.local_exposure = None;
        s.fog_scattering = None;
        // The voxel surface cache's relighting costs the same at any resolution: 25 ms of the
        // 2-CU iGPU's frame at every level, the largest cost left at `low` before this.
        s.lumen_surface_interval = s.lumen_surface_interval.max(2);
        if let Some(d) = s.dynamic_resolution.as_mut() {
            d.min_percentage = d.min_percentage.min(50.0);
        }
        if self == Self::Medium {
            return s;
        }
        // Low: what is left of the look is the lit geometry, its shadows and the sky. Each of
        // these cost 2–45 ms at 50 % on the 2-CU iGPU (clouds 24 in the valley; local light
        // shadow maps 46, coloured glass light 27, the volumetric fog 16 in the cathedral).
        s.clouds = false;
        s.volumetric_fog = false;
        s.glass_shadows = false;
        s.short_range_ao = false;
        s.local_light_shadow_maps = false;
        s.mega_lights = false;
        s.translucency_layers = false;
        s.tessellation = false;
        s.motion_blur = 0.0;
        s.depth_of_field = None;
        s.wind_simulation = false;
        s.lod_threshold = s.lod_threshold.max(2.0);
        s.lumen_surface_interval = s.lumen_surface_interval.max(4);
        s.shadow_resolution_bias = s.shadow_resolution_bias.max(1.0);
        if let Some(d) = s.dynamic_resolution.as_mut() {
            d.min_percentage = d.min_percentage.min(35.0);
        }
        s
    }
}

/// Watches the frame times and moves the quality level when the GPU cannot keep up or has
/// plenty to spare. Time-based (seconds, not frames): a slow GPU renders few frames.
#[derive(Clone, Debug)]
pub struct Governor {
    /// Off: the page fixed the level (`heliosConfig.quality`), the governor only reports.
    pub auto: bool,
    pub quality: Quality,
    /// The highest level it may go up to (levels left for being too slow are excluded).
    ceiling: Quality,
    /// Since when the frame has been over (or far under) the budget, in ms of page time.
    over_since: Option<f64>,
    under_since: Option<f64>,
    /// No decisions before this time (after a change, a resume, the start).
    quiet_until: f64,
    last: f64,
    /// Smoothed frame interval (for browsers without timestamp queries).
    interval_ms: f32,
    pub steps_down: u32,
    pub steps_up: u32,
}

/// What the governor sees each frame.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub now_ms: f64,
    /// The renderer's smoothed GPU time (None without timestamp queries).
    pub gpu_ms: Option<f32>,
    /// The screen percentage this frame and the dynamic range (None: fixed resolution).
    pub percentage: f32,
    pub dynamic: Option<DynamicResolution>,
    /// The GPU budget (the dynamic resolution's, or the default one).
    pub budget_ms: f32,
    /// Every pipeline is compiled (the frames show every feature the level has).
    pub complete: bool,
}

impl Governor {
    /// Seconds over the budget before a step down.
    const DOWN_SECONDS: f64 = 3.0;
    /// Seconds far under the budget before a step up.
    const UP_SECONDS: f64 = 8.0;
    /// Seconds after a change (or the start) before the next decision: the new settings' first
    /// frames, pipelines and histories settle.
    const SETTLE_SECONDS: f64 = 3.0;
    /// Over the budget at dynamic resolution's minimum by this factor, or anywhere by `HOPELESS`.
    const OVER: f32 = 1.25;
    const HOPELESS: f32 = 2.5;
    /// Under the budget by this factor at the dynamic maximum: room for the level above, which
    /// cost 1.3–1.8 times as much in the measurements (RTX 3090 gallery: low 5.1, medium 7.0,
    /// high 9.2 ms). A step up that misses is undone 3 s later and not tried again.
    const UNDER: f32 = 0.55;
    /// Without GPU timings: the frame interval that counts as too slow (under 22 fps).
    const SLOW_INTERVAL_MS: f32 = 45.0;

    pub fn new(quality: Quality, auto: bool, now_ms: f64) -> Self {
        Self {
            auto,
            quality,
            ceiling: Quality::High,
            over_since: None,
            under_since: None,
            quiet_until: now_ms + Self::SETTLE_SECONDS * 1000.0,
            last: now_ms,
            interval_ms: 16.7,
            steps_down: 0,
            steps_up: 0,
        }
    }

    /// Sets the level by hand (`heliosSetQuality`): `auto` false fixes it.
    pub fn set(&mut self, quality: Quality, auto: bool, now_ms: f64) {
        self.quality = quality;
        self.auto = auto;
        self.ceiling = Quality::High;
        self.reset(now_ms);
    }

    fn reset(&mut self, now_ms: f64) {
        self.over_since = None;
        self.under_since = None;
        self.quiet_until = now_ms + Self::SETTLE_SECONDS * 1000.0;
    }

    /// Observes a presented frame; returns the new level when it changes.
    pub fn observe(&mut self, s: Sample) -> Option<Quality> {
        let gap = s.now_ms - self.last;
        self.last = s.now_ms;
        // A pause (hidden canvas, background tab) is not a slow frame.
        if gap > 1000.0 {
            self.reset(s.now_ms);
            return None;
        }
        self.interval_ms += (gap as f32 - self.interval_ms) * 0.1;
        if !self.auto || s.now_ms < self.quiet_until {
            return None;
        }
        let (at_min, at_max) = match s.dynamic {
            Some(d) => {
                let min = d.min_percentage.clamp(helios_render::MIN_SCREEN_PERCENTAGE, helios_render::MAX_SCREEN_PERCENTAGE);
                let max = d.max_percentage.clamp(helios_render::MIN_SCREEN_PERCENTAGE, helios_render::MAX_SCREEN_PERCENTAGE).max(min);
                (s.percentage <= min + 0.5, s.percentage >= max - 0.5)
            }
            None => (true, true),
        };
        let budget = s.budget_ms.max(1.0);
        let (over, under) = match s.gpu_ms {
            Some(gpu) => ((gpu > budget * Self::OVER && at_min) || gpu > budget * Self::HOPELESS, gpu < budget * Self::UNDER && at_max),
            None => (self.interval_ms > Self::SLOW_INTERVAL_MS, false),
        };
        self.over_since = if over { self.over_since.or(Some(s.now_ms)) } else { None };
        self.under_since = if under && s.complete { self.under_since.or(Some(s.now_ms)) } else { None };
        if self.over_since.is_some_and(|t| s.now_ms - t >= Self::DOWN_SECONDS * 1000.0) {
            if let Some(lower) = self.quality.lower() {
                self.ceiling = lower;
                self.quality = lower;
                self.steps_down += 1;
                self.reset(s.now_ms);
                return Some(lower);
            }
        }
        if self.under_since.is_some_and(|t| s.now_ms - t >= Self::UP_SECONDS * 1000.0) {
            if let Some(higher) = self.quality.higher().filter(|h| *h <= self.ceiling) {
                self.quality = higher;
                self.steps_up += 1;
                self.reset(s.now_ms);
                return Some(higher);
            }
            self.under_since = None;
        }
        None
    }
}

/// The quality level of a running page: the GPU it found, the governor, and the settings the
/// level is applied to.
#[derive(Clone, Debug)]
pub struct Levels {
    pub info: GpuInfo,
    pub class: GpuClass,
    pub governor: Governor,
    /// The page chose the GI method (`heliosConfig.gi`): the levels leave it alone.
    keep_gi: bool,
    /// What the scene, the page and the editor asked for.
    base: RenderSettings,
    /// What the level made of `base` last time.
    applied: RenderSettings,
}

impl Levels {
    /// `requested`: the page's `heliosConfig.quality` (None or "auto": from the GPU class, and
    /// the governor moves it).
    pub fn new(info: GpuInfo, requested: Option<Quality>, keep_gi: bool, settings: &RenderSettings, now_ms: f64) -> Self {
        let class = classify(&info);
        let quality = requested.unwrap_or_else(|| Quality::for_class(class));
        Self { info, class, governor: Governor::new(quality, requested.is_none(), now_ms), keep_gi, base: *settings, applied: *settings }
    }

    pub fn quality(&self) -> Quality {
        self.governor.quality
    }

    /// Testing (`?gpu-class=`): as if the browser had reported a GPU of `class`.
    pub fn force_class(&mut self, class: GpuClass) {
        self.class = class;
        if self.governor.auto {
            self.governor.quality = Quality::for_class(class);
        }
    }

    /// The render settings for the current level. Settings changed since the level was last
    /// applied (an editor edit, a scene file's overrides, the renderer's own Lumen Lite
    /// fallback) count as asked for and survive every later change of level.
    pub fn settings(&mut self, current: &RenderSettings) -> RenderSettings {
        for setting in crate::editor::settings::all() {
            let now = crate::editor::settings::get(current, setting.key);
            if now != crate::editor::settings::get(&self.applied, setting.key) {
                if let Some(value) = now {
                    let _ = crate::editor::settings::set(&mut self.base, setting.key, &value);
                }
            }
        }
        if current.gi_method != self.applied.gi_method {
            self.base.gi_method = current.gi_method;
            self.base.gi_lite_fallback = current.gi_lite_fallback;
        }
        self.applied = self.governor.quality.apply(&self.base, self.keep_gi);
        self.applied
    }

    /// `heliosQuality()`: the level, whether it is automatic, the GPU and the steps taken.
    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "level": self.quality().name(),
            "auto": self.governor.auto,
            "detected": Quality::for_class(self.class).name(),
            "gpu": { "class": self.class.name(), "vendor": self.info.vendor, "architecture": self.info.architecture, "renderer": self.info.renderer },
            "stepsDown": self.governor.steps_down,
            "stepsUp": self.governor.steps_up,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_changes_keep_edits_made_in_between() {
        let mut base = RenderSettings::default();
        base.dynamic_resolution = Some(DynamicResolution::with_budget(14.0));
        let mut levels = Levels::new(info("amd", "rdna-2", "AMD Radeon(TM) Graphics"), None, false, &base, 0.0);
        assert_eq!(levels.quality(), Quality::Low);
        let mut current = levels.settings(&base);
        assert!(!current.clouds);
        // The editor turns the aerial perspective off and the clouds back on at the low level.
        current.aerial_perspective = false;
        current.clouds = true;
        levels.governor.set(Quality::High, true, 0.0);
        let high = levels.settings(&current);
        assert!(!high.aerial_perspective && high.clouds && high.gi_method == GiMethod::ScreenProbes && high.avboit);
        assert_eq!(high.lumen_surface_interval, 1);
        levels.governor.set(Quality::Low, true, 0.0);
        let low = levels.settings(&high);
        // The edit survives; the level still wins over what it switches.
        assert!(!low.aerial_perspective && !low.clouds && low.lumen_surface_interval == 4);
    }

    fn info(vendor: &str, architecture: &str, renderer: &str) -> GpuInfo {
        GpuInfo { vendor: vendor.into(), architecture: architecture.into(), renderer: renderer.into(), ..Default::default() }
    }

    #[test]
    fn classifies_what_browsers_report() {
        use GpuClass::*;
        let cases = [
            // Chromium on Windows (ANGLE strings).
            (info("amd", "rdna-2", "ANGLE (AMD, AMD Radeon(TM) Graphics (0x0000164E) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("amd", "rdna-3", "ANGLE (AMD, AMD Radeon 780M Graphics (0x000015BF) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("amd", "gcn-5", "ANGLE (AMD, AMD Radeon(TM) Vega 8 Graphics (0x000015D8) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("amd", "rdna-3", "ANGLE (AMD, AMD Radeon RX 7800 XT (0x0000747E) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Discrete),
            (info("amd", "rdna-2", "ANGLE (AMD, AMD Radeon RX 6600M (0x000073FF) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Discrete),
            (info("amd", "rdna-3", "ANGLE (AMD, AMD Radeon(TM) 8060S Graphics (0x00001586) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Discrete),
            (info("intel", "gen-12lp", "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics (0x00009A49) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("intel", "gen-9", "ANGLE (Intel, Intel(R) UHD Graphics 620 (0x00005917) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("intel", "xe-lpg", "ANGLE (Intel, Intel(R) Arc(TM) Graphics (0x00007D55) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("intel", "xe-hpg", "ANGLE (Intel, Intel(R) Arc(TM) A770 Graphics (0x000056A0) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Discrete),
            (info("intel", "xe2-hpg", "ANGLE (Intel, Intel(R) Arc(TM) B580 Graphics (0x0000E20B) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Discrete),
            (info("intel", "xe2-lpg", "ANGLE (Intel, Intel(R) Arc(TM) 140V GPU (16GB) (0x000064A0) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Integrated),
            (info("nvidia", "ampere", "ANGLE (NVIDIA, NVIDIA GeForce RTX 3090 (0x00002204) Direct3D11 vs_5_0 ps_5_0, D3D11)"), Discrete),
            (info("qualcomm", "", "ANGLE (Qualcomm, Qualcomm(R) Adreno(TM) X1-85 GPU, D3D11)"), Mobile),
            (info("google", "swiftshader", "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)"), Software),
            (info("microsoft", "", "ANGLE (Microsoft, Microsoft Basic Render Driver, D3D11)"), Software),
            // macOS, Android.
            (info("apple", "metal-3", "ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)"), Apple),
            (info("arm", "valhall", "ANGLE (ARM, Mali-G710, OpenGL ES 3.2)"), Mobile),
            // Firefox: no WebGPU vendor, a generic WebGL model.
            (info("", "", "Radeon R9 200 Series, or similar"), Unknown),
            (info("", "", "Intel(R) HD Graphics 400, or similar"), Integrated),
            (info("", "", "NVIDIA GeForce GTX 980, or similar"), Discrete),
            // Nothing at all.
            (info("", "", ""), Unknown),
        ];
        for (i, class) in cases {
            assert_eq!(classify(&i), class, "{i:?}");
        }
        assert_eq!(classify(&GpuInfo { fallback: true, ..info("amd", "", "") }), Software);
    }

    #[test]
    fn levels_only_lower_and_respect_the_page_s_gi() {
        let mut base = RenderSettings::default();
        base.dynamic_resolution = Some(DynamicResolution::with_budget(14.0));
        assert_eq!(format!("{:?}", Quality::High.apply(&base, false)), format!("{base:?}"));
        let low = Quality::Low.apply(&base, false);
        assert_eq!(low.gi_method, GiMethod::LumenLite);
        assert!(!low.clouds && !low.avboit && low.dynamic_resolution.unwrap().min_percentage <= 35.0);
        assert_eq!(Quality::Low.apply(&base, true).gi_method, base.gi_method);
        let medium = Quality::Medium.apply(&base, false);
        assert!(medium.clouds && medium.gi_method == GiMethod::LumenLite);
    }

    fn sample(now_ms: f64, gpu: f32, percentage: f32) -> Sample {
        Sample { now_ms, gpu_ms: Some(gpu), percentage, dynamic: Some(DynamicResolution::with_budget(14.0)), budget_ms: 14.0, complete: true }
    }

    /// Feeds frames every `step` ms from `from` to `to`; returns the changes.
    fn run(g: &mut Governor, from: f64, to: f64, step: f64, gpu: f32, percentage: f32) -> Vec<Quality> {
        let mut changes = vec![];
        let mut t = from;
        while t < to {
            changes.extend(g.observe(sample(t, gpu, percentage)));
            t += step;
        }
        changes
    }

    #[test]
    fn steps_down_after_seconds_over_budget_and_not_back_up() {
        let mut g = Governor::new(Quality::High, true, 0.0);
        // Slow GPU, 4 fps, at the minimum: one step after settling + 3 s, the next 6 s later.
        assert_eq!(run(&mut g, 0.0, 6_500.0, 250.0, 120.0, 50.0), vec![Quality::Medium]);
        assert_eq!(run(&mut g, 6_500.0, 13_000.0, 250.0, 60.0, 50.0), vec![Quality::Low]);
        assert_eq!(g.steps_down, 2);
        // Fast now, at full resolution: may not return to the levels it left.
        assert!(run(&mut g, 13_000.0, 40_000.0, 16.7, 3.0, 100.0).is_empty());
    }

    #[test]
    fn steps_up_from_a_cautious_start_and_ignores_pauses() {
        let mut g = Governor::new(Quality::Low, true, 0.0);
        assert_eq!(run(&mut g, 0.0, 12_000.0, 16.7, 3.0, 100.0), vec![Quality::Medium]);
        // A paused canvas resumes with a long gap: no decision from it.
        assert!(g.observe(sample(30_000.0, 100.0, 50.0)).is_none());
        // Within budget at a lower resolution: stays.
        let mut steady = Governor::new(Quality::Medium, true, 0.0);
        assert!(run(&mut steady, 0.0, 30_000.0, 16.7, 12.0, 70.0).is_empty());
        // A fixed level never moves.
        let mut fixed = Governor::new(Quality::High, false, 0.0);
        assert!(run(&mut fixed, 0.0, 30_000.0, 100.0, 200.0, 50.0).is_empty());
    }

    #[test]
    fn without_timings_the_frame_interval_decides() {
        let mut g = Governor::new(Quality::High, true, 0.0);
        let mut changes = vec![];
        for i in 0..40 {
            changes.extend(g.observe(Sample { now_ms: i as f64 * 200.0, gpu_ms: None, percentage: 100.0, dynamic: None, budget_ms: 14.0, complete: true }));
        }
        assert_eq!(changes, vec![Quality::Medium]);
    }
}
