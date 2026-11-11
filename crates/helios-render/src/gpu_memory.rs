//! GPU memory accounting: what the renderer's buffers and textures occupy, by subsystem.
//!
//! The ledger is the allocator's own, not a parallel one the create calls would have to keep
//! up to date: natively the backend's sub-allocator lists every live allocation under the
//! resource's label (`wgpu::Device::generate_allocator_report`, D3D12 and Vulkan); in the
//! browser the page records each `GPUDevice.createBuffer` / `createTexture` by its label until
//! the resource is destroyed or collected (`helios-demo/src/gpu_memory.js`). Both feed
//! [`Report::from_allocations`], which files every label under a [`CATEGORIES`] entry.
//!
//! In the browser a dropped `wgpu::Buffer` / `wgpu::Texture` keeps its memory until the JS
//! collector finalizes the wrapper (wgpu's WebGPU backend doesn't call `destroy()` on drop), so
//! a renderer that replaces large resources while the page keeps allocating can run its GPU
//! process out of memory before any collection happens. [`retire_buffer`] / [`retire_view`]
//! hand a replaced or released resource to [`destroy_retired`], which destroys it right after the
//! frame's submission (in WebGPU, `destroy()` waits for the work already submitted that uses it).
//!
//! Features the scene doesn't use don't hold memory: their resources are allocated on first use
//! and released after [`RELEASE_AFTER_FRAMES`] frames unused ([`Idle`]).

use std::fmt::Write as _;

/// Subsystems, matched against the label in order (the first entry whose keyword the label
/// contains wins; keywords are lower case).
pub const CATEGORIES: &[(&str, &[&str])] = &[
    ("wgpu internal", &["(wgpu internal)", "unlabeled"]),
    ("photo mode", &["photo"]),
    ("shadows: virtual shadow maps", &["vsm", "shadow page", "shadow mask", "local shadow", "moving light"]),
    ("shadows: glass and volumes", &["glass", "volume shadow"]),
    (
        "translucency and OIT",
        &["avboit", "oit ", "translucen", "volume front", "volume back", "refraction", "transmittance"],
    ),
    ("megalights", &["megalights", "mega lights"]),
    ("lumen: world cache", &["world cache"]),
    ("lumen lite", &["lumen lite", "lite "]),
    (
        "lumen: cards, SDF, gather",
        &["card", "sdf", "lumen", "radiance", "screen probe", "irradiance", "gi raw", "gi history", "reflection"],
    ),
    ("textures: virtual / streamed", &["rvt", "svt", "texture pool", "page table", "feedback", "texture tile"]),
    ("textures: material arrays", &["material albedo", "material data", "material normal", "texture array", "pattern"]),
    (
        "geometry (Nanite)",
        &["geometry pool", "main view", "cluster", "page", "instance", "candidate", "nanite", "sw raster", "visible", "draw", "meshlet", "hzb", "occlusion", "cull"],
    ),
    ("tessellation", &["tessellat", "dice", "patch", "displace"]),
    ("foliage and wind", &["plant", "wind", "bone", "foliage"]),
    ("water and terrain", &["river", "water", "terrain", "landscape", "heightfield"]),
    ("fog, clouds, sky", &["fog", "cloud", "sky", "atmosphere", "froxel", "aerial", "scattering"]),
    ("volumes and particles", &["volume", "particle", "fluid", "smoke", "sprite"]),
    ("post-processing", &["bloom", "post fx", "exposure", "tonemap", "tone map", "history", "dof", "luminance", "tsr"]),
    ("frame targets", &["depth", "visibility", "gbuffer", "hdr", "material tiles", "target", "ao"]),
    ("lights and materials", &["light", "material", "substrate", "subsurface", "skin"]),
    ("scene and uniforms", &["scene", "uniform", "frame", "camera", "transform", "params", "globals"]),
];

/// The category of a resource label.
pub fn category(label: &str) -> &'static str {
    let lower = label.to_ascii_lowercase();
    CATEGORIES.iter().find(|(_, keys)| keys.iter().any(|k| lower.contains(k))).map_or("other", |(name, _)| name)
}

/// Live GPU memory under one label.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelUse {
    pub label: String,
    pub category: &'static str,
    pub bytes: u64,
    pub count: u32,
}

/// Live GPU memory by category and label.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Largest first.
    pub labels: Vec<LabelUse>,
    /// What the allocator holds in its blocks, including free space (native only).
    pub reserved: Option<u64>,
    /// The browser ledger's largest total so far.
    pub peak: Option<u64>,
    /// Allocations that failed with out-of-memory (browser, `?gpu-memory=1`).
    pub failures: u32,
}

impl Report {
    /// The report for live allocations given as (label, bytes); equal labels are summed.
    pub fn from_allocations<'a>(allocations: impl IntoIterator<Item = (&'a str, u64)>) -> Self {
        Self::from_counted(allocations.into_iter().map(|(label, bytes)| (label, bytes, 1)))
    }

    /// The report for (label, bytes, resources) groups; equal labels are summed.
    pub fn from_counted<'a>(groups: impl IntoIterator<Item = (&'a str, u64, u32)>) -> Self {
        let mut labels: Vec<LabelUse> = Vec::new();
        let mut index = std::collections::HashMap::<String, usize>::new();
        for (label, bytes, count) in groups {
            let label = if label.is_empty() { "(unlabeled)" } else { label };
            let i = *index.entry(label.to_string()).or_insert_with(|| {
                labels.push(LabelUse { label: label.to_string(), category: category(label), bytes: 0, count: 0 });
                labels.len() - 1
            });
            labels[i].bytes += bytes;
            labels[i].count += count;
        }
        labels.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.label.cmp(&b.label)));
        Self { labels, ..Default::default() }
    }

    /// The backend allocator's report (D3D12, Vulkan); `None` where the backend has none (the
    /// browser: see the page's ledger).
    pub fn from_device(device: &wgpu::Device) -> Option<Self> {
        let report = device.generate_allocator_report()?;
        let mut out = Self::from_allocations(report.allocations.iter().map(|a| (a.name.as_str(), a.size)));
        out.reserved = Some(report.total_reserved_bytes);
        Some(out)
    }

    pub fn total(&self) -> u64 {
        self.labels.iter().map(|l| l.bytes).sum()
    }

    /// (category, bytes, resources), largest first.
    pub fn categories(&self) -> Vec<(&'static str, u64, u32)> {
        let mut out: Vec<(&'static str, u64, u32)> = Vec::new();
        for l in &self.labels {
            match out.iter_mut().find(|c| c.0 == l.category) {
                Some(c) => {
                    c.1 += l.bytes;
                    c.2 += l.count;
                }
                None => out.push((l.category, l.bytes, l.count)),
            }
        }
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    /// The per-category table, each category followed by its labels of at least `min_label` bytes.
    pub fn table(&self, min_label: u64) -> String {
        let mib = |b: u64| b as f64 / 1048576.0;
        let mut s = String::new();
        let _ = write!(s, "GPU memory: {:.1} MiB in {} resources", mib(self.total()), self.labels.iter().map(|l| l.count).sum::<u32>());
        if let Some(reserved) = self.reserved {
            let _ = write!(s, " ({:.1} MiB reserved by the allocator)", mib(reserved));
        }
        if let Some(peak) = self.peak {
            let _ = write!(s, ", peak {:.1} MiB", mib(peak));
        }
        if self.failures > 0 {
            let _ = write!(s, ", {} allocations failed", self.failures);
        }
        for (category, bytes, count) in self.categories() {
            let _ = write!(s, "\n  {:>8.1} MiB  {category} ({count})", mib(bytes));
            for l in self.labels.iter().filter(|l| l.category == category && l.bytes >= min_label) {
                let times = if l.count > 1 { format!(" ×{}", l.count) } else { String::new() };
                let _ = write!(s, "\n  {:>8.1} MiB      {}{times}", mib(l.bytes), l.label);
            }
        }
        s
    }

    /// `{"total": bytes, "categories": {name: bytes, …}, "labels": [[label, category, bytes, count], …]}`.
    pub fn json(&self) -> String {
        let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let categories: Vec<String> = self.categories().iter().map(|(c, b, _)| format!("\"{}\":{b}", escape(c))).collect();
        let labels: Vec<String> =
            self.labels.iter().map(|l| format!("[\"{}\",\"{}\",{},{}]", escape(&l.label), escape(l.category), l.bytes, l.count)).collect();
        let mut extra = String::new();
        if let Some(peak) = self.peak {
            let _ = write!(extra, ",\"peak\":{peak}");
        }
        if let Some(reserved) = self.reserved {
            let _ = write!(extra, ",\"reserved\":{reserved}");
        }
        format!(
            "{{\"total\":{}{extra},\"failures\":{},\"categories\":{{{}}},\"labels\":[{}]}}",
            self.total(),
            self.failures,
            categories.join(","),
            labels.join(",")
        )
    }
}

/// Frames a lazily allocated resource may go unused before it is released (4 s at 60 Hz): a
/// feature that pauses for a moment keeps its targets, one the scene no longer has (glass after
/// leaving the cathedral, volumes out of view) gives them back.
pub const RELEASE_AFTER_FRAMES: u32 = 240;

/// Counts the frames a lazily allocated resource went unused.
#[derive(Clone, Copy, Debug, Default)]
pub struct Idle {
    frames: u32,
}

impl Idle {
    /// Records this frame's use; true once the resource has gone unused for
    /// [`RELEASE_AFTER_FRAMES`] (release it then).
    pub fn tick(&mut self, used: bool) -> bool {
        if used {
            self.frames = 0;
            return false;
        }
        self.frames = self.frames.saturating_add(1);
        self.frames >= RELEASE_AFTER_FRAMES
    }
}

enum Retired {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture),
}

thread_local! {
    /// Resources replaced or released since the last frame's submission.
    static RETIRED: std::cell::RefCell<Vec<Retired>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Destroys `buffer` after this frame's submission (see the module notes): for a buffer that was
/// replaced, released, or used only by this frame's commands. Nothing may use it afterwards.
pub fn retire_buffer(buffer: &wgpu::Buffer) {
    RETIRED.with(|r| r.borrow_mut().push(Retired::Buffer(buffer.clone())));
}

/// [`retire_buffer`] for the texture behind `view` (all its views die with it).
pub fn retire_view(view: &wgpu::TextureView) {
    retire_texture(view.texture());
}

/// [`retire_buffer`] for a texture.
pub fn retire_texture(texture: &wgpu::Texture) {
    RETIRED.with(|r| r.borrow_mut().push(Retired::Texture(texture.clone())));
}

/// Destroys what was retired; call right after the frame's submission (WebGPU and wgpu keep a
/// destroyed resource alive for the work already submitted). Returns how many.
pub fn destroy_retired() -> usize {
    let retired = RETIRED.with(|r| std::mem::take(&mut *r.borrow_mut()));
    for r in &retired {
        match r {
            Retired::Buffer(b) => b.destroy(),
            Retired::Texture(t) => t.destroy(),
        }
    }
    retired.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_and_totals() {
        let r = Report::from_allocations([("avboit cells", 100), ("vsm physical pages", 50), ("avboit cells", 20), ("hdr", 8)]);
        assert_eq!(r.total(), 178);
        assert_eq!(r.labels[0].label, "avboit cells");
        assert_eq!(r.labels[0].count, 2);
        assert_eq!(category("vsm moments"), "shadows: virtual shadow maps");
        assert_eq!(category("hdr"), "frame targets");
        assert!(r.json().contains("\"total\":178"));
    }
}
