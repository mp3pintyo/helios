//! Substrate-lite: the extended material model (UE 5.7's Substrate, reduced to one slab).
//!
//! A [`Slab`] describes a material's interface and medium beyond the metallic-roughness base:
//! the dielectric's F0 / F90 from an index of refraction and a specular colour
//! (`KHR_materials_ior` / `_specular`), anisotropic highlights, a clear coat with its own
//! roughness and normal, a sheen (velvet, cloth), subsurface scattering (skin, wax, marble) or
//! thin diffuse transmission (leaves, paper) and unlit emission. UE's Substrate stacks and mixes
//! any number of slabs per pixel; one slab with these features covers glTF's PBR extensions
//! and UE's legacy shading models.
//!
//! **Cost only where used.** Materials without slab features keep the plain G-buffer and the
//! plain resolve and lighting shaders, bit for bit. Materials with them resolve in extended
//! pipelines over tiles of their own, which also write one 16-byte slab texel per pixel; the
//! lighting passes read it only where the G-buffer flags the pixel (see `substrate.wgsl`,
//! `substrate_gpu.rs`). All lighting paths — the deferred sun and clustered lights, MegaLights,
//! image-based light — evaluate the same lobes from `brdf.wgsl`.

use glam::Vec3;

use crate::scene::TextureHandle;

/// Feature bits (`SLAB_*` in substrate.wgsl).
pub mod features {
    pub const CLEARCOAT: u32 = 1;
    pub const SHEEN: u32 = 2;
    pub const SUBSURFACE: u32 = 4;
    pub const TRANSMISSION: u32 = 8;
    pub const ANISOTROPY: u32 = 16;
    /// F0 / F90 differ from the default dielectric (IOR 1.5, full specular).
    pub const SPECULAR: u32 = 32;
    pub const UNLIT: u32 = 64;
}

/// A material's slab (see the module docs). [`Slab::default`] has no features: a plain
/// metallic-roughness material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slab {
    /// Index of refraction of the dielectric (glTF `KHR_materials_ior`): F0 = ((ior − 1) /
    /// (ior + 1))², 0.04 at the default 1.5.
    pub ior: f32,
    /// Strength of the dielectric's specular reflection (`KHR_materials_specular`): scales F0
    /// and F90.
    pub specular: f32,
    /// Tint of the dielectric's F0 (linear).
    pub specular_color: Vec3,
    /// Clear coat on top (0–1): a second, colourless GGX interface (car paint, lacquer).
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    /// Sheen colour (linear; black: none): retro-reflective fuzz on velvet and cloth.
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,
    /// Share of the diffuse light scattered beneath the surface (0–1): diffused on screen with
    /// the profile below.
    pub subsurface: f32,
    /// How far each channel scatters relative to the others (linear; skin: red furthest).
    pub subsurface_color: Vec3,
    /// Distance light scatters beneath the surface, in metres (the widest channel's).
    pub subsurface_radius: f32,
    /// Thin diffuse transmission (0–1): light from behind passes through (leaves, paper,
    /// lamp shades).
    pub transmission: f32,
    pub transmission_color: Vec3,
    /// Stretch of highlights along the surface (−1–1, `KHR_materials_anisotropy`).
    pub anisotropy: f32,
    /// Direction of the stretch: radians from the texture's u direction.
    pub anisotropy_rotation: f32,
    /// Shows the base colour unlit (`KHR_materials_unlit`).
    pub unlit: bool,
    /// A texture layer multiplying the coat and sheen factors: base colour = sheen colour
    /// (rgb) and sheen roughness (alpha), normal = clear-coat normal, roughness = clear-coat
    /// roughness, metallic = clear coat. Sampled from the 512² array layer with the main
    /// texture's coordinates.
    pub coat_texture: Option<TextureHandle>,
    pub coat_normal_strength: f32,
    /// A texture layer multiplying the specular factors: base colour = specular colour (rgb)
    /// and specular (alpha).
    pub specular_texture: Option<TextureHandle>,
}

impl Default for Slab {
    fn default() -> Self {
        Self {
            ior: 1.5,
            specular: 1.0,
            specular_color: Vec3::ONE,
            clearcoat: 0.0,
            clearcoat_roughness: 0.03,
            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.5,
            subsurface: 0.0,
            subsurface_color: Vec3::new(1.0, 0.45, 0.25),
            subsurface_radius: 0.01,
            transmission: 0.0,
            transmission_color: Vec3::ONE,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            unlit: false,
            coat_texture: None,
            coat_normal_strength: 1.0,
            specular_texture: None,
        }
    }
}

impl Slab {
    /// A clear coat of `amount` and `roughness` (car paint: 1, 0.03).
    pub fn clearcoat(amount: f32, roughness: f32) -> Self {
        Self { clearcoat: amount, clearcoat_roughness: roughness, ..Default::default() }
    }

    /// A sheen of `color` and `roughness` (velvet: deep colour, 0.3–0.6).
    pub fn sheen(color: Vec3, roughness: f32) -> Self {
        Self { sheen_color: color, sheen_roughness: roughness, ..Default::default() }
    }

    /// Subsurface scattering of `amount` over `radius` metres with per-channel `color`.
    pub fn subsurface(amount: f32, color: Vec3, radius: f32) -> Self {
        Self { subsurface: amount, subsurface_color: color, subsurface_radius: radius, ..Default::default() }
    }

    /// Thin diffuse transmission of `amount`, tinted by `color`.
    pub fn thin(amount: f32, color: Vec3) -> Self {
        Self { transmission: amount, transmission_color: color, ..Default::default() }
    }

    /// The dielectric F0 (IOR × specular colour × specular).
    pub fn f0(&self) -> Vec3 {
        let r = (self.ior - 1.0) / (self.ior + 1.0);
        (self.specular_color * (r * r)).min(Vec3::ONE) * self.specular
    }

    /// The features (bits of [`features`]) this slab needs.
    pub fn features(&self) -> u32 {
        use features::*;
        let mut bits = 0;
        let on = |b: bool, bit: u32| if b { bit } else { 0 };
        bits |= on(self.clearcoat > 0.0, CLEARCOAT);
        bits |= on(self.sheen_color.max_element() > 0.0, SHEEN);
        bits |= on(self.subsurface > 0.0 && self.subsurface_radius > 0.0, SUBSURFACE);
        bits |= on(self.transmission > 0.0, TRANSMISSION);
        bits |= on(self.anisotropy != 0.0, ANISOTROPY);
        let default_f0 = (self.f0() - Vec3::splat(0.04)).abs().max_element() > 1e-4;
        bits |= on(default_f0 || self.specular != 1.0 || self.specular_texture.is_some(), SPECULAR);
        bits |= on(self.unlit, UNLIT);
        bits
    }
}

/// [`Slab`] as the shaders read it (`MaterialSlab` in substrate.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuSlab {
    pub f0: [f32; 3],
    pub f90: f32,
    pub sheen_color: [f32; 3],
    pub sheen_roughness: f32,
    pub subsurface_color: [f32; 3],
    pub subsurface_radius: f32,
    pub transmission_color: [f32; 3],
    pub transmission: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub features: u32,
    pub subsurface: f32,
    pub coat_texture: u32,
    pub specular_texture: u32,
    /// First slot of the material's graph parameters.
    pub params: u32,
    pub coat_normal_strength: f32,
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuSlab>() == 112);

impl GpuSlab {
    /// The slab's record with `features` (the slab's own, plus those a material graph adds).
    pub fn new(slab: &Slab, features: u32, params: u32) -> Self {
        let layer = |t: Option<TextureHandle>| t.map_or(u32::MAX, |t| t.0);
        // The subsurface colour is relative: its widest channel spans the radius.
        let scatter = slab.subsurface_color.max(Vec3::splat(1e-3));
        Self {
            f0: slab.f0().to_array(),
            f90: slab.specular.clamp(0.0, 1.0),
            sheen_color: slab.sheen_color.to_array(),
            sheen_roughness: slab.sheen_roughness.clamp(0.0, 1.0),
            subsurface_color: (scatter / scatter.max_element()).to_array(),
            subsurface_radius: slab.subsurface_radius.max(0.0),
            transmission_color: slab.transmission_color.to_array(),
            transmission: slab.transmission.clamp(0.0, 1.0),
            clearcoat: slab.clearcoat.clamp(0.0, 1.0),
            clearcoat_roughness: slab.clearcoat_roughness.clamp(0.0, 1.0),
            anisotropy: slab.anisotropy.clamp(-1.0, 1.0),
            anisotropy_rotation: slab.anisotropy_rotation,
            features,
            subsurface: slab.subsurface.clamp(0.0, 1.0),
            coat_texture: layer(slab.coat_texture),
            specular_texture: layer(slab.specular_texture),
            params,
            coat_normal_strength: slab.coat_normal_strength,
            _pad: [0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_slabs_have_no_features() {
        assert_eq!(Slab::default().features(), 0);
        assert!((Slab::default().f0() - Vec3::splat(0.04)).abs().max_element() < 1e-6);
    }

    #[test]
    fn features_follow_the_parameters() {
        use features::*;
        assert_eq!(Slab::clearcoat(1.0, 0.03).features(), CLEARCOAT);
        assert_eq!(Slab::sheen(Vec3::new(0.5, 0.0, 0.1), 0.4).features(), SHEEN);
        assert_eq!(Slab::subsurface(1.0, Vec3::ONE, 0.01).features(), SUBSURFACE);
        assert_eq!(Slab::thin(0.5, Vec3::ONE).features(), TRANSMISSION);
        assert_eq!(Slab { ior: 1.33, ..Default::default() }.features(), SPECULAR);
        assert_eq!(Slab { specular: 0.5, ..Default::default() }.features(), SPECULAR);
        assert_eq!(Slab { unlit: true, anisotropy: 0.5, ..Default::default() }.features(), UNLIT | ANISOTROPY);
    }

    #[test]
    fn f0_follows_the_index_of_refraction_and_specular() {
        let water = Slab { ior: 1.33, ..Default::default() };
        assert!((water.f0().x - 0.02).abs() < 1e-3);
        let tinted = Slab { specular_color: Vec3::new(1.0, 0.5, 0.0), specular: 0.5, ..Default::default() };
        assert!((tinted.f0() - Vec3::new(0.02, 0.01, 0.0)).abs().max_element() < 1e-6);
        let gpu = GpuSlab::new(&tinted, tinted.features(), 0);
        assert_eq!(gpu.f90, 0.5);
        assert_eq!(gpu.coat_texture, u32::MAX);
    }
}
