//! Alpha masks of alpha-tested ("masked") materials, for programmable rasterisation.
//!
//! Every rasteriser that writes visibility or depth — the hardware and software visibility
//! rasterisers and both virtual shadow map rasterisers — discards texels whose alpha is below
//! the material's cutoff, so foliage cards, chains and grilles are cut out and cast cut-out
//! shadows (see `shaders/alpha_mask.wgsl`). The masks are a separate R8 texture array, one
//! layer per masked base colour image and cutoff, independent of the material texture arrays:
//! the rasterisers bind only this small array, and its mip chain is built differently.
//!
//! **Coverage-preserving mips.** Box-filtering alpha shrinks the area above the cutoff in
//! coarse mips (thin blades and chain links average to below the cutoff), so distant foliage
//! would dissolve. Each mip is therefore rescaled so that the fraction of its texels passing
//! the alpha test equals the full-resolution level's (Castaño, "Computing alpha mipmaps",
//! 2010) — which is why a mask layer belongs to one cutoff.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::scene::{Image, Material, Scene, TextureMapping, TextureSource};
use crate::util::mip_count;

/// Largest mask layer edge; larger images are area-downsampled. All layers share the size of
/// the largest mask (one array), smaller ones are upsampled.
const MAX_MASK_SIZE: u32 = 1024;
const MIN_MASK_SIZE: u32 = 16;

/// The GPU alpha test of one material: its mask layer and the cutoff, quantised to 8 bits
/// (packed into `GpuMaterial::flags`, see `material_flags`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialMask {
    pub layer: u32,
    pub cutoff: u8,
}

/// A mask layer is identified by its source texture and quantised cutoff.
type MaskKey = (u32, u8);

pub struct AlphaMasks {
    /// R8 array, one layer per mask (a 1×1 dummy without masks).
    pub view: wgpu::TextureView,
    /// Repeating, trilinear, anisotropic like the material sampler, so the alpha test and
    /// shading filter the same footprint.
    pub sampler: wgpu::Sampler,
    /// Per scene material.
    materials: Vec<Option<MaterialMask>>,
    /// Every requested mask with its layer (see [`MaskSet::layers`]).
    layers: HashMap<MaskKey, Option<u32>>,
    texture_version: u64,
}

impl AlphaMasks {
    pub fn new(device: &wgpu::Device) -> Self {
        let dummy = device.create_texture(&texture_descriptor(1, 1));
        Self {
            view: dummy.create_view(&view_descriptor()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("alpha mask"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            }),
            materials: Vec::new(),
            layers: HashMap::new(),
            texture_version: u64::MAX,
        }
    }

    /// The alpha test of material `index`, if it has one.
    pub fn material(&self, index: usize) -> Option<MaterialMask> {
        self.materials.get(index).copied().flatten()
    }

    /// True if some material is alpha-tested (the masked raster paths have work).
    pub fn any(&self) -> bool {
        self.materials.iter().any(Option::is_some)
    }

    /// Rebuilds the masks if the scene's masked materials or textures changed, and reassigns
    /// materials to layers.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let wanted: Vec<Option<(MaskKey, &Image)>> = scene.materials.iter().map(|m| mask_source(scene, m)).collect();
        let mut keys: Vec<MaskKey> = wanted.iter().flatten().map(|(k, _)| *k).collect();
        keys.sort_unstable();
        keys.dedup();
        let mut known: Vec<MaskKey> = self.layers.keys().copied().collect();
        known.sort_unstable();
        if keys != known || scene.texture_version != self.texture_version {
            let set = build_masks(&wanted);
            if !set.texels.is_empty() {
                self.view = upload(device, queue, set.size, &set.texels);
                log::info!("built {} alpha masks ({}² with coverage-preserving mips)", set.texels.len(), set.size);
            }
            self.layers = set.layers;
            self.texture_version = scene.texture_version;
        }
        self.materials = assign(&wanted, &self.layers);
    }
}

/// CPU-side masks: the texels of every layer and each requested mask's layer.
struct MaskSet {
    size: u32,
    /// Every requested mask with its layer (`None`: it cuts nothing out and renders without
    /// alpha test, on the opaque fast path).
    layers: HashMap<MaskKey, Option<u32>>,
    /// Per layer, its mip chain.
    texels: Vec<Vec<Vec<u8>>>,
}

fn build_masks(wanted: &[Option<(MaskKey, &Image)>]) -> MaskSet {
    let mut sources: Vec<(MaskKey, &Image)> = wanted.iter().flatten().copied().collect();
    sources.sort_unstable_by_key(|(k, _)| *k);
    sources.dedup_by_key(|(k, _)| *k);
    let size = sources
        .iter()
        .map(|(_, image)| image.width.max(image.height).next_power_of_two())
        .max()
        .unwrap_or(1)
        .clamp(MIN_MASK_SIZE, MAX_MASK_SIZE);
    let mut set = MaskSet { size, layers: HashMap::new(), texels: Vec::new() };
    for (key, image) in sources {
        let cutoff = key.1 as f32 / 255.0;
        let alpha = resample_alpha(image, size);
        let layer = (coverage(&alpha, 1.0, cutoff) < 1.0).then(|| {
            set.texels.push(coverage_preserving_mips(alpha, size, cutoff));
            set.texels.len() as u32 - 1
        });
        set.layers.insert(key, layer);
    }
    set
}

fn assign(wanted: &[Option<(MaskKey, &Image)>], layers: &HashMap<MaskKey, Option<u32>>) -> Vec<Option<MaterialMask>> {
    wanted
        .iter()
        .map(|w| w.and_then(|(key, _)| layers.get(&key).copied().flatten().map(|layer| MaterialMask { layer, cutoff: key.1 })))
        .collect()
}

/// Uploads mask layers (`size`², full mip chains) into a new R8 array.
fn upload(device: &wgpu::Device, queue: &wgpu::Queue, size: u32, layers: &[Vec<Vec<u8>>]) -> wgpu::TextureView {
    let texture = device.create_texture(&texture_descriptor(size, layers.len() as u32));
    for (layer, mips) in layers.iter().enumerate() {
        for (mip, texels) in mips.iter().enumerate() {
            let edge = (size >> mip).max(1);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                texels,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(edge), rows_per_image: Some(edge) },
                wgpu::Extent3d { width: edge, height: edge, depth_or_array_layers: 1 },
            );
        }
    }
    texture.create_view(&view_descriptor())
}

fn texture_descriptor(size: u32, layers: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("alpha masks"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers },
        mip_level_count: mip_count(size, size),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

fn view_descriptor() -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() }
}

/// The mask a material asks for: its texture's base colour image and quantised cutoff. Only
/// mesh-UV mapped image textures carry alpha; other alpha-tested materials render opaque.
/// Whether some material of `scene` is alpha-tested (without building the masks).
pub(crate) fn scene_has_masks(scene: &Scene) -> bool {
    scene.materials.iter().any(|m| mask_source(scene, m).is_some())
}

fn mask_source<'a>(scene: &'a Scene, m: &Material) -> Option<(MaskKey, &'a Image)> {
    let cutoff = (m.alpha_cutoff?.clamp(0.0, 1.0) * 255.0).round() as u8;
    let texture = m.texture?;
    if cutoff == 0 || m.mapping != TextureMapping::MeshUv {
        return None;
    }
    match scene.textures.get(texture.0 as usize)? {
        TextureSource::Images(images) => Some(((texture.0, cutoff), images.base_color.as_deref()?)),
        TextureSource::Procedural(_) => None,
    }
}

/// The image's alpha resampled to `size`² (wrapping, like the repeat sampler): bilinear when
/// magnifying, a box of bilinear taps over each target texel's footprint when minifying.
fn resample_alpha(image: &Image, size: u32) -> Vec<f32> {
    let (w, h) = (image.width as usize, image.height as usize);
    let n = size as usize;
    if w * h == 0 || image.pixels.len() != w * h * 4 {
        return vec![1.0; n * n];
    }
    let alpha = |x: usize, y: usize| image.pixels[(y * w + x) * 4 + 3] as f32 / 255.0;
    if (w, h) == (n, n) {
        return (0..n * n).map(|i| alpha(i % n, i / n)).collect();
    }
    let taps = w.max(h).div_ceil(n).max(1);
    let mut out = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0;
            for ty in 0..taps {
                for tx in 0..taps {
                    let u = (x as f32 + (tx as f32 + 0.5) / taps as f32) / n as f32;
                    let v = (y as f32 + (ty as f32 + 0.5) / taps as f32) / n as f32;
                    sum += sample_alpha(image, u, v);
                }
            }
            out.push(sum / (taps * taps) as f32);
        }
    }
    out
}

/// The image's alpha at `(u, v)`: bilinear and wrapping, like the repeat sampler.
fn sample_alpha(image: &Image, u: f32, v: f32) -> f32 {
    let (w, h) = (image.width as usize, image.height as usize);
    if w * h == 0 || image.pixels.len() != w * h * 4 {
        return 1.0;
    }
    let alpha = |x: usize, y: usize| image.pixels[(y * w + x) * 4 + 3] as f32 / 255.0;
    let x = u * w as f32 - 0.5;
    let y = v * h as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let wrap = |i: f32, n: usize| (i as isize).rem_euclid(n as isize) as usize;
    let (xa, xb, ya, yb) = (wrap(x0, w), wrap(x0 + 1.0, w), wrap(y0, h), wrap(y0 + 1.0, h));
    let top = alpha(xa, ya) * (1.0 - fx) + alpha(xb, ya) * fx;
    let bottom = alpha(xa, yb) * (1.0 - fx) + alpha(xb, yb) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Fraction of the surface of an alpha-masked mesh that is solid: the area-weighted share of
/// its triangles (object-space `positions`, texture coordinates `uvs` before tiling, `indices`) where `material`'s alpha test
/// passes. `None` for materials that are not cut out (no alpha test, or one the rasterisers
/// don't apply, see `mask_source`).
///
/// Lumen's distance fields are built from geometry alone, so a foliage card or a grille would
/// block light like a solid sheet; this is how much of it blocks light on average (UE5 gives
/// two-sided foliage distance fields a coverage for the same reason).
pub(crate) fn surface_coverage(scene: &Scene, material: &Material, positions: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]) -> Option<f32> {
    let ((_, cutoff), image) = mask_source(scene, material)?;
    let cutoff = cutoff as f32 / 255.0;
    let texels = image.width as f32 * image.height as f32;
    let (mut solid, mut total) = (0.0f64, 0.0f64);
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(positions[i as usize]));
        let [ua, ub, uc] = [t[0], t[1], t[2]].map(|i| Vec2::from(uvs[i as usize]) * material.uv_scale);
        let area = (b - a).cross(c - a).length() as f64;
        if area <= 0.0 {
            continue;
        }
        // About one sample per 2x2 texels the triangle covers, on the R2 low-discrepancy
        // sequence mapped uniformly into the triangle (square-root warp).
        let covered = (ub - ua).perp_dot(uc - ua).abs() * 0.5 * texels;
        let samples = (covered / 4.0).clamp(4.0, 4096.0) as u32;
        let passed = (0..samples)
            .filter(|&k| {
                let (r1, r2) = ((0.5 + k as f32 * 0.754_877_7).fract(), (0.5 + k as f32 * 0.569_840_3).fract());
                let s = r1.sqrt();
                let uv = ua * (1.0 - s) + ub * (s * (1.0 - r2)) + uc * (s * r2);
                sample_alpha(image, uv.x, uv.y) >= cutoff
            })
            .count();
        solid += area * passed as f64 / samples as f64;
        total += area;
    }
    Some(if total > 0.0 { (solid / total) as f32 } else { 1.0 })
}

/// Fraction of texels that pass the alpha test after scaling alpha by `scale`.
fn coverage(alpha: &[f32], scale: f32, cutoff: f32) -> f32 {
    alpha.iter().filter(|&&a| a * scale >= cutoff).count() as f32 / alpha.len().max(1) as f32
}

/// Full mip chain of a square, power-of-two alpha level (`size`²) as R8 texels. Every level is
/// box-filtered from the unscaled previous one, then scaled to pass the alpha test over the
/// same fraction of texels as level 0.
fn coverage_preserving_mips(level0: Vec<f32>, size: u32, cutoff: f32) -> Vec<Vec<u8>> {
    let target = coverage(&level0, 1.0, cutoff);
    let quantise = |alpha: &[f32], scale: f32| -> Vec<u8> { alpha.iter().map(|a| ((a * scale).clamp(0.0, 1.0) * 255.0).round() as u8).collect() };
    let mut mips = vec![quantise(&level0, 1.0)];
    let mut level = level0;
    let mut edge = size as usize;
    while edge > 1 {
        let half = edge / 2;
        level = (0..half * half)
            .map(|i| {
                let o = (i / half * 2) * edge + i % half * 2;
                (level[o] + level[o + 1] + level[o + edge] + level[o + edge + 1]) * 0.25
            })
            .collect();
        edge = half;
        // Coverage grows monotonically with the scale, up to all non-zero texels passing at
        // `cutoff / smallest`: bisect, then take the closer bound.
        let smallest = level.iter().copied().filter(|&a| a > 0.0).fold(1.0f32, f32::min);
        let (mut lo, mut hi) = (0.0f32, (cutoff / smallest).max(1.0));
        for _ in 0..32 {
            let mid = 0.5 * (lo + hi);
            if coverage(&level, mid, cutoff) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let error = |scale| (coverage(&level, scale, cutoff) - target).abs();
        let scale = if error(lo) < error(hi) { lo } else { hi };
        mips.push(quantise(&level, scale));
    }
    mips
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::scene::{ImageTexture, TextureDesc, TexturePattern};

    /// A white RGBA image whose alpha is `f(x, y)`.
    fn image(size: u32, f: impl Fn(u32, u32) -> u8) -> Image {
        let pixels = (0..size * size).flat_map(|i| [255, 255, 255, f(i % size, i / size)]).collect();
        Image { width: size, height: size, pixels }
    }

    #[test]
    fn thin_blades_keep_their_coverage_in_coarse_mips() {
        // Soft-edged blades, 10 texels wide every 32: plain box filtering lowers their peak
        // alpha to 0.375 by mip 3, below the cutoff, so distant foliage would vanish.
        let blades = image(128, |x, _| ((1.0 - (x as f32 % 32.0 - 16.0).abs() / 5.0).max(0.0) * 255.0).round() as u8);
        let alpha = resample_alpha(&blades, 128);
        let target = coverage(&alpha, 1.0, 0.5);
        assert_eq!(target, 5.0 / 32.0);
        let mips = coverage_preserving_mips(alpha, 128, 0.5);
        assert_eq!(mips.len(), 8);
        for (level, texels) in mips.iter().enumerate().take(4) {
            // Quantised to whole texels of a 128 >> level wide pattern with period 32 >> level.
            let passing = texels.iter().filter(|&&a| a >= 128).count() as f32 / texels.len() as f32;
            assert!((passing - target).abs() <= 0.1, "mip {level}: coverage {passing}, want {target}");
        }
        assert!(mips[3].iter().any(|&a| a >= 128), "mip 3 keeps the blades");
    }

    #[test]
    fn resampling_preserves_the_mean_and_wraps() {
        let half = image(8, |x, _| if x < 4 { 255 } else { 0 });
        let up = resample_alpha(&half, 16);
        let mean = up.iter().sum::<f32>() / up.len() as f32;
        assert!((mean - 0.5).abs() < 1e-3);
        // Wrapping: the left edge blends with the transparent right edge.
        assert!(up[0] < 1.0 && up[0] > 0.5);
        let down = resample_alpha(&image(32, |x, _| if x % 2 == 0 { 255 } else { 0 }), 16);
        assert!(down.iter().all(|a| (a - 0.5).abs() < 1e-3));
    }

    #[test]
    fn surface_coverage_weighs_the_cut_out_area() {
        let mut scene = Scene::new();
        // Left half opaque, right half cut out.
        let half = scene.add_image_texture(ImageTexture { base_color: Some(Arc::new(image(64, |x, _| if x < 32 { 255 } else { 0 }))), ..Default::default() });
        let masked = Material { texture: Some(half), alpha_cutoff: Some(0.5), ..Default::default() };
        // A unit quad over the whole image, and a second quad (four times the area) mapped to
        // the opaque half only.
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 2.0, 1.0]];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.05, 0.0], [0.45, 0.0], [0.45, 1.0], [0.05, 1.0]];
        let whole = [0, 1, 2, 0, 2, 3];
        let coverage = |indices: &[u32], m: &Material| surface_coverage(&scene, m, &positions, &uvs, indices);
        let c = coverage(&whole, &masked).unwrap();
        assert!((c - 0.5).abs() < 0.03, "{c}");
        assert!((coverage(&[4, 5, 6, 4, 6, 7], &masked).unwrap() - 1.0).abs() < 1e-6);
        let both = coverage(&[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], &masked).unwrap();
        assert!((both - (0.5 + 4.0) / 5.0).abs() < 0.03, "area-weighted: {both}");
        // Two repeats per UV unit still see half the image cut out.
        let tiled = coverage(&whole, &Material { uv_scale: 2.0, ..masked }).unwrap();
        assert!((tiled - 0.5).abs() < 0.03, "{tiled}");
        // Not alpha-tested: no coverage.
        assert_eq!(coverage(&whole, &Material { alpha_cutoff: None, ..masked }), None);
    }

    #[test]
    fn only_cut_out_image_masks_get_layers() {
        let mut scene = Scene::new();
        let half = |x: u32, _| if x < 8 { 255 } else { 0 };
        let cut = scene.add_image_texture(ImageTexture { base_color: Some(Arc::new(image(16, half))), ..Default::default() });
        let solid = scene.add_image_texture(ImageTexture { base_color: Some(Arc::new(image(16, |_, _| 255))), ..Default::default() });
        let procedural = scene.add_texture(TextureDesc::new(TexturePattern::Leaf, glam::Vec3::ONE, glam::Vec3::ONE));
        let masked = |texture| Material { texture: Some(texture), alpha_cutoff: Some(0.5), ..Default::default() };
        let materials = [
            masked(cut),
            masked(solid),
            masked(procedural),
            Material { texture: Some(cut), ..Default::default() },
            Material { mapping: TextureMapping::WorldTriplanar, ..masked(cut) },
            Material { alpha_cutoff: Some(0.25), ..masked(cut) },
            masked(cut),
        ];
        let wanted: Vec<_> = materials.iter().map(|m| mask_source(&scene, m)).collect();
        let keys: Vec<_> = wanted.iter().map(|w| w.map(|(k, _)| k)).collect();
        assert_eq!(keys, [Some((0, 128)), Some((1, 128)), None, None, None, Some((0, 64)), Some((0, 128))]);

        // The opaque image needs no layer; the two cutoffs of the cut-out image get one each,
        // shared by the materials that use them.
        let set = build_masks(&wanted);
        assert_eq!((set.size, set.texels.len()), (16, 2));
        let masks = assign(&wanted, &set.layers);
        let layer = |i: usize| masks[i].map(|m| (m.layer, m.cutoff));
        assert_eq!([layer(0), layer(1), layer(2), layer(5), layer(6)], [Some((1, 128)), None, None, Some((0, 64)), Some((1, 128))]);
    }
}
