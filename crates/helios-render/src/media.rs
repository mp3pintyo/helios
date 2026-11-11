//! Bindings shared by the passes that light participating media outside the froxel fog —
//! the volumes' lighting cache and march (`volumes.rs`) and the particles (`particles.rs`):
//! the light sources and fog of `shaders/media_common.wgsl`, at the binding numbers it declares.

use wgpu::BindingResource::{Sampler, TextureView};
use wgpu::{BindingResource, BindingType};

use crate::passes::shading::LightingInputs;
use crate::util::bind;

/// Layout entries of every lighting input (sun, shadows, sky, local lights, Lumen, fog).
pub(crate) fn lighting_entries() -> Vec<(u32, BindingType)> {
    vec![
        (0, bind::uniform()),
        (1, bind::uniform()),
        (2, bind::uniform()),
        (3, bind::storage(true)),
        (4, bind::texture_depth()),
        (5, bind::sampler_comparison()),
        (6, bind::storage(true)),
        (7, bind::storage(true)),
        (8, bind::storage(true)),
        (9, bind::uniform()),
        (10, bind::texture(true)),
        (11, bind::sampler()),
        (12, bind::texture(true)),
        (13, bind::uniform()),
        (14, bind::texture_3d(true)),
        (15, bind::texture_3d(true)),
        (16, bind::storage(true)),
        (17, bind::sampler()),
        (18, bind::texture_3d(true)),
        (19, bind::texture(true)),
        (60, bind::uniform()),
        // Sunlight through glass (glass_shadow.wgsl).
        (90, bind::uniform()),
        (91, bind::texture(true)),
    ]
}

/// The resources of [`lighting_entries`].
pub(crate) fn lighting_resources<'a>(inputs: &'a LightingInputs) -> Vec<(u32, BindingResource<'a>)> {
    let (sdf, voxel_lighting) = inputs.lumen.debug_views();
    vec![
        (0, inputs.frame.as_entire_binding()),
        (1, inputs.fog.uniform.as_entire_binding()),
        (2, inputs.vsm.uniform.as_entire_binding()),
        (3, inputs.vsm.cache.entries.as_entire_binding()),
        (4, TextureView(&inputs.vsm.atlas)),
        (5, Sampler(&inputs.samplers.shadow)),
        (6, inputs.sky.sh.as_entire_binding()),
        (7, inputs.lights.lights.as_entire_binding()),
        (8, inputs.lights.grid.as_entire_binding()),
        (9, inputs.clouds.uniform.as_entire_binding()),
        (10, TextureView(&inputs.clouds.shadow)),
        (11, Sampler(&inputs.samplers.linear)),
        (12, TextureView(&inputs.sky.transmittance)),
        (13, inputs.lumen.uniform().as_entire_binding()),
        (14, TextureView(sdf)),
        (15, TextureView(voxel_lighting)),
        (16, inputs.lumen.radiance_probes().as_entire_binding()),
        (17, Sampler(&inputs.samplers.linear)),
        (18, TextureView(inputs.fog.integrated())),
        (19, TextureView(&inputs.sky.sky_view)),
        (60, inputs.vsm.local.uniform.as_entire_binding()),
        (90, inputs.glass.uniform.as_entire_binding()),
        (91, TextureView(inputs.glass.view())),
    ]
}

/// Layout entries of the fog in front of a medium only (frame, fog, aerial perspective).
pub(crate) fn fog_entries() -> Vec<(u32, BindingType)> {
    vec![(0, bind::uniform()), (1, bind::uniform()), (11, bind::sampler()), (12, bind::texture(true)), (18, bind::texture_3d(true)), (19, bind::texture(true))]
}

/// The resources of [`fog_entries`].
pub(crate) fn fog_resources<'a>(inputs: &'a LightingInputs) -> Vec<(u32, BindingResource<'a>)> {
    vec![
        (0, inputs.frame.as_entire_binding()),
        (1, inputs.fog.uniform.as_entire_binding()),
        (11, Sampler(&inputs.samplers.linear)),
        (12, TextureView(&inputs.sky.transmittance)),
        (18, TextureView(inputs.fog.integrated())),
        (19, TextureView(&inputs.sky.sky_view)),
    ]
}
