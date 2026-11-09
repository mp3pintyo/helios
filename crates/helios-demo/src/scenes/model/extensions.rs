//! glTF's PBR extensions → Substrate slabs: clear coat, sheen, IOR, specular, anisotropy and
//! unlit become [`Slab`] factors, and their textures two optional texture layers per material
//! (see [`Slab::coat_texture`] and [`Slab::specular_texture`]), whose channels are rearranged on
//! the CPU into what the layer packing expects. They are sampled with the main texture's
//! coordinates.

use std::collections::HashMap;
use std::sync::Arc;

use helios_asset::TextureRef;
use helios_render::{Image, ImageTexture, Scene, Slab, TextureHandle};

use super::Model;

/// Where a channel of a combined image comes from: (image index, channel).
type Pick = Option<(usize, usize)>;

/// The slab texture layers created so far, shared between materials using the same images.
#[derive(Default)]
pub struct SlabLayers {
    /// By kind (0: coat and sheen, 1: specular) and source channels.
    layers: HashMap<(u8, [Pick; 12]), TextureHandle>,
}

impl SlabLayers {
    /// The material's slab (default: no features).
    pub fn slab(&mut self, scene: &mut Scene, model: &Model, m: &helios_asset::Material) -> Slab {
        let e = &m.extensions;
        let image = |t: Option<TextureRef>, channel: usize| t.map(|t| (t.image, channel));
        // Coat and sheen: base colour = sheen colour (rgb), its alpha replaced by the
        // "occlusion" image's red = sheen roughness (the sheen roughness texture's alpha);
        // normal = clear-coat normal; roughness (G) = clear-coat roughness (G); metallic (B) =
        // clear coat (R).
        let coat = [
            image(e.sheen_color_texture, 0),
            image(e.sheen_color_texture, 1),
            image(e.sheen_color_texture, 2),
            None,
            image(e.clearcoat_normal_texture, 0),
            image(e.clearcoat_normal_texture, 1),
            image(e.clearcoat_normal_texture, 2),
            None,
            None,
            image(e.clearcoat_roughness_texture, 1),
            image(e.clearcoat_texture, 0),
            image(e.sheen_roughness_texture, 3),
        ];
        let coat_texture = coat.iter().any(Option::is_some).then(|| {
            *self.layers.entry((0, coat)).or_insert_with(|| {
                scene.add_image_texture(ImageTexture {
                    base_color: Some(combine(model, [coat[0], coat[1], coat[2], None])),
                    normal: e.clearcoat_normal_texture.map(|t| model.images[t.image].clone()),
                    metallic_roughness: Some(combine(model, [None, coat[9], coat[10], None])),
                    occlusion: Some(combine(model, [coat[11], None, None, None])),
                })
            })
        });
        // Specular: base colour = specular colour (rgb) and specular (alpha).
        let mut specular = [None; 12];
        specular[..4].copy_from_slice(&[image(e.specular_color_texture, 0), image(e.specular_color_texture, 1), image(e.specular_color_texture, 2), image(e.specular_texture, 3)]);
        let specular_texture = specular.iter().any(Option::is_some).then(|| {
            *self.layers.entry((1, specular)).or_insert_with(|| {
                scene.add_image_texture(ImageTexture { base_color: Some(combine(model, [specular[0], specular[1], specular[2], specular[3]])), ..Default::default() })
            })
        });
        Slab {
            ior: e.ior,
            specular: e.specular,
            specular_color: e.specular_color,
            clearcoat: e.clearcoat,
            clearcoat_roughness: e.clearcoat_roughness,
            sheen_color: e.sheen_color,
            sheen_roughness: e.sheen_roughness,
            anisotropy: e.anisotropy,
            anisotropy_rotation: e.anisotropy_rotation,
            unlit: e.unlit,
            coat_texture,
            coat_normal_strength: e.clearcoat_normal_scale,
            specular_texture,
            ..Default::default()
        }
    }
}

/// An image whose channel `c` is `picks[c]` (255 where `None`), at the largest source's size,
/// sampled nearest.
fn combine(model: &Model, picks: [Pick; 4]) -> Arc<Image> {
    let sources: Vec<&Image> = picks.iter().flatten().map(|&(i, _)| model.images[i].as_ref()).collect();
    let width = sources.iter().map(|i| i.width).max().unwrap_or(1);
    let height = sources.iter().map(|i| i.height).max().unwrap_or(1);
    let mut pixels = vec![255u8; (width * height * 4) as usize];
    for (c, pick) in picks.iter().enumerate() {
        let Some((index, channel)) = *pick else { continue };
        let source = &model.images[index];
        for y in 0..height {
            let sy = (y * source.height / height) as usize;
            for x in 0..width {
                let sx = (x * source.width / width) as usize;
                pixels[((y * width + x) * 4) as usize + c] = source.pixels[(sy * source.width as usize + sx) * 4 + channel];
            }
        }
    }
    Arc::new(Image { width, height, pixels })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(images: Vec<Image>) -> Model {
        let mut model = Model::new("test", helios_asset::Asset::default());
        model.images = images.into_iter().map(Arc::new).collect();
        model
    }

    #[test]
    fn channels_combine_at_the_largest_size() {
        let small = Image { width: 1, height: 1, pixels: vec![10, 20, 30, 40] };
        let large = Image { width: 2, height: 2, pixels: (0..16).collect() };
        let m = model(vec![small, large]);
        let image = combine(&m, [Some((0, 3)), None, Some((1, 1)), None]);
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(&image.pixels[..8], &[40, 255, 1, 255, 40, 255, 5, 255]);
    }

    #[test]
    fn extensions_become_slabs_and_layers() {
        let mut scene = Scene::new();
        let m = model(vec![Image { width: 1, height: 1, pixels: vec![200, 100, 50, 25] }]);
        let mut layers = SlabLayers::default();
        let plain = layers.slab(&mut scene, &m, &helios_asset::Material::default());
        assert_eq!(plain.features(), 0);
        let mut coated = helios_asset::Material::default();
        coated.extensions.clearcoat = 1.0;
        coated.extensions.clearcoat_texture = Some(TextureRef::new(0));
        coated.extensions.ior = 1.33;
        let slab = layers.slab(&mut scene, &m, &coated);
        assert!(slab.coat_texture.is_some() && slab.specular_texture.is_none());
        assert_eq!(slab.features(), helios_render::slab_features::CLEARCOAT | helios_render::slab_features::SPECULAR);
        // Shared with an identical material.
        assert_eq!(layers.slab(&mut scene, &m, &coated).coat_texture, slab.coat_texture);
    }
}
