//! GPU mirror of a [`Scene`]: always-resident cluster/page records of every mesh, the
//! streamed geometry pool with its page table, instances, materials with their alpha masks and
//! the persistent cluster visibility bitset used by two-pass occlusion culling.

use glam::{Affine2, Mat3, Mat4, Vec2, Vec3, Vec4};
use helios_geometry::{GpuCluster, GpuPage, INVALID_PAGE};

use crate::alpha_mask::{AlphaMasks, MaterialMask};
use crate::assembly::Assemblies;
use crate::gpu_types::{instance_flags, material_flags, GpuInstance, GpuMaterial, GpuUvTransform};
use crate::motion::SceneChanges;
use crate::plant_pose::PlantPose;
use crate::plant_sim::{Candidate, PlantSim, SIM_BONES, SIM_DEVIATION};
use crate::plant_wind::PlantTables;
use crate::scene::{Instance, Material, Mobility, Scene, TextureHandle, TextureSource, UvTransform};
use crate::skin::{instance_skin_flags, SkinPalettes};
use crate::streaming::{StreamingStats, Streamer};
use crate::tessellation;
use crate::util::{storage_buffer, storage_buffer_init};
use crate::vsm_motion::ShadowMove;
use crate::wind::WindUniform;

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshRange {
    pub page_offset: u32,
    pub page_count: u32,
    pub cluster_offset: u32,
    pub cluster_count: u32,
    pub bounds: Vec4,
}

/// Materials live in a uniform buffer (keeps material passes within 8 storage bindings).
pub const MAX_MATERIALS: usize = 256;

pub struct GpuScene {
    pub clusters: wgpu::Buffer,
    pub pages: wgpu::Buffer,
    /// Global page -> geometry pool word offset (or `NOT_RESIDENT`).
    pub page_table: wgpu::Buffer,
    /// Streamed vertex and triangle data of resident pages.
    pub geometry: wgpu::Buffer,
    /// `[0, page_count)`: per page, the last frame in which a resident parent wanted to refine
    /// into it (streaming feedback). After that: one bit per (instance, cluster), "visible last
    /// frame" (occlusion culling). One buffer keeps the culling pass within 8 storage bindings.
    pub atomics: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    /// Some material samples a texture with the second UV set, a wrap mode other than repeat or
    /// an occlusion layer of its own: the G-buffer resolve needs its `UV_EXTENSIONS` permutation.
    pub uv_extensions: bool,
    /// The plain resolve slots (shading models) the materials use, a bit each: the resolve
    /// creates pipelines for these only.
    pub resolve_kinds: u32,
    /// Per material its plain resolve slot (shading model).
    material_kinds: Vec<u32>,
    /// The materials of the instances in the wind (plants).
    plant_materials: Vec<u32>,
    /// Counts uploads of `materials` (extended materials patch their resolve slots after each).
    pub material_uploads: u64,
    /// [`WindUniform`], written every frame ([`GpuScene::write_wind`]).
    pub wind: wgpu::Buffer,
    /// The tessellation uniform: a per-frame header ([`GpuScene::write_tessellation_view`]) and the
    /// materials' displacement (`tessellation.rs`).
    pub tessellation: wgpu::Buffer,
    /// FNV-1a of the uploaded displacement table: a change reshapes displaced surfaces (cached
    /// shadow pages of them are stale).
    pub displacement_table_hash: u64,
    /// Tessellate displaced materials (the renderer's setting; instances carry it as a flag).
    pub tessellation_enabled: bool,
    /// Alpha masks of alpha-tested materials, sampled by every rasteriser.
    pub masks: AlphaMasks,
    /// Joint palettes of skinned instances, written every frame.
    pub skins: SkinPalettes,
    /// Assembly tables and the per-frame expansion into part instances.
    pub assemblies: Assemblies,
    /// Bone tables of plants (`plant_wind.rs`).
    pub plants: PlantTables,
    /// The dynamic wind's simulated tier: promoted plants near the camera (`plant_sim.rs`).
    pub plant_sim: PlantSim,
    /// The dynamic wind's pose pass: plant skeletons posed once per instance and frame
    /// (`plant_pose.rs`).
    pub plant_pose: PlantPose,
    /// Plants bend their bones in the wind (`RenderSettings::wind_bones`; the instances carry it).
    pub wind_bones_enabled: bool,
    wind_bones_uploaded: bool,
    pub instance_count: u32,
    pub page_count: u32,
    pub meshes: Vec<MeshRange>,
    /// World bounds of all instances.
    bounds: (glam::Vec3, glam::Vec3),
    streamer: Streamer,
    /// CPU copy of the instance records: moves rewrite single records.
    records: Vec<GpuInstance>,
    /// Shadow casters moved by the last [`GpuScene::sync`].
    shadow_moves: Vec<ShadowMove>,
    /// Some shadow-casting instance has a wind response.
    wind_shadows: bool,
    /// Some instance has a wind response; the wind blows (strength > 0, set by write_wind).
    windy: bool,
    wind_blowing: std::sync::atomic::AtomicBool,
    /// Some mesh has voxel LODs (the voxel rasterisers run).
    voxels: bool,
    /// Some instance is displaced (the tessellator runs), and the setting its flags were made with.
    displaced: bool,
    /// Some instance's material displaces it, whatever the setting (the browser compiles
    /// tessellation in the background while it is off: `pipelines.rs`).
    displaceable: bool,
    /// Some shadow caster's displacement is animated (its shadow pages refresh every frame).
    animated_shadows: bool,
    tessellation_uploaded: bool,
    /// Scene versions last uploaded: geometry, materials, instances, textures.
    versions: (u64, u64, u64, u64),
    /// Which graphs drive displacement, as last uploaded (instances carry the displaced flag).
    displacement_graphs_version: u64,
}

impl GpuScene {
    pub fn new(device: &wgpu::Device, pool_bytes: u64) -> Self {
        let empty = |label| storage_buffer(device, label, 16, wgpu::BufferUsages::empty());
        let pool_words = (pool_bytes / 4).min(u32::MAX as u64 - 1) as u32;
        Self {
            clusters: empty("clusters"),
            pages: empty("pages"),
            page_table: empty("page table"),
            geometry: storage_buffer(device, "geometry pool", pool_words as u64 * 4, wgpu::BufferUsages::empty()),
            atomics: storage_buffer(device, "culling atomics", 16, wgpu::BufferUsages::COPY_SRC),
            instances: empty("instances"),
            materials: crate::util::uniform_buffer(device, "materials", (MAX_MATERIALS * std::mem::size_of::<GpuMaterial>()) as u64),
            uv_extensions: false,
            resolve_kinds: 0,
            material_kinds: Vec::new(),
            plant_materials: Vec::new(),
            material_uploads: 0,
            wind: crate::util::uniform_buffer(device, "wind", std::mem::size_of::<WindUniform>() as u64),
            // Lumen's displaced variants copy its material table (lumen_displaced.rs).
            tessellation: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tessellation"),
                size: tessellation::UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            displacement_table_hash: 0,
            tessellation_enabled: true,
            masks: AlphaMasks::new(device),
            skins: SkinPalettes::new(device),
            assemblies: Assemblies::new(device),
            plants: PlantTables::new(device),
            plant_sim: PlantSim::new(device),
            plant_pose: PlantPose::new(device),
            wind_bones_enabled: true,
            wind_bones_uploaded: true,
            instance_count: 0,
            page_count: 0,
            meshes: Vec::new(),
            bounds: (glam::Vec3::ZERO, glam::Vec3::ZERO),
            streamer: Streamer::new(pool_words),
            records: Vec::new(),
            shadow_moves: Vec::new(),
            wind_shadows: false,
            windy: false,
            wind_blowing: std::sync::atomic::AtomicBool::new(false),
            voxels: false,
            displaced: false,
            displaceable: false,
            animated_shadows: false,
            tessellation_uploaded: true,
            versions: (u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            displacement_graphs_version: u64::MAX,
        }
    }

    pub fn bounds(&self) -> (glam::Vec3, glam::Vec3) {
        self.bounds
    }

    pub fn streaming_stats(&self) -> StreamingStats {
        self.streamer.stats()
    }

    /// Whether the wind moves any shadow-casting instance (whose cached shadow pages it then
    /// invalidates).
    pub fn wind_moves_shadows(&self) -> bool {
        self.wind_shadows
    }

    /// Whether some instance deforms this frame: the wind moves it, its material displaces it or it is
    /// skinned. Culling takes its lean specialisation otherwise (DEFORMING in cull.wgsl).
    pub fn deforms(&self) -> bool {
        (self.windy && self.wind_blowing.load(std::sync::atomic::Ordering::Relaxed)) || self.displaced || self.skins.skinned_instances
    }

    /// The plain resolve slots (shading models) of the materials of plants (instances in the wind),
    /// a bit each: only their resolve pipelines carry the plant code (`PLANTS` in plant_wind.wgsl).
    pub fn plant_kinds(&self) -> u32 {
        self.plant_materials.iter().filter_map(|&m| self.material_kinds.get(m as usize)).fold(0, |kinds, &k| kinds | (1 << k.min(31)))
    }

    /// Whether the passes that place geometry carry the plant code (`PLANTS` in plant_wind.wgsl):
    /// the bones are on and some plant has them — or will have them: the browser's first frames
    /// draw the scene before its final bake ([`crate::expect_plant_bones`]).
    pub fn plant_code(&self) -> bool {
        self.wind_bones_enabled && (self.plant_pose.any() || crate::plant_pose::plant_bones_expected())
    }

    /// Whether the wind blows this frame (`write_wind`).
    pub fn wind_blowing(&self) -> bool {
        self.wind_blowing.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether some mesh has voxel LODs.
    pub fn has_voxels(&self) -> bool {
        self.voxels
    }

    /// Whether some instance is displaced and tessellated.
    pub fn has_displacement(&self) -> bool {
        self.displaced
    }

    /// Whether some instance would be displaced with tessellation on.
    pub fn has_displaceable(&self) -> bool {
        self.displaceable
    }

    /// Whether some shadow caster is displaced by a graph that reads the time.
    pub fn animated_displacement_shadows(&self) -> bool {
        self.animated_shadows
    }

    /// Writes the tessellation uniform's per-frame header.
    pub fn write_tessellation_view(&self, queue: &wgpu::Queue, view: &tessellation::TessView) {
        queue.write_buffer(&self.tessellation, 0, bytemuck::bytes_of(view));
    }

    /// Instances culling visits: the scene's and the assembly instances' culling records after
    /// them (their part slots are reached through the records).
    pub fn cull_instance_count(&self) -> u32 {
        self.instance_count + self.assemblies.record_count()
    }

    pub fn write_wind(&self, queue: &wgpu::Queue, wind: &WindUniform) {
        queue.write_buffer(&self.wind, 0, bytemuck::bytes_of(wind));
        self.wind_blowing.store(wind.direction.w > 0.0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Promotes and demotes plants for the dynamic wind's simulation (`plant_sim.rs`) around
    /// `camera`, rewriting the records whose slot changed, and writes the simulation's inputs.
    pub(crate) fn update_plant_sim(&mut self, queue: &wgpu::Queue, scene: &Scene, camera: Vec3, time: f32, dt: f32) {
        if self.records.len() != scene.instances.len() {
            return;
        }
        let changed = self.plant_sim.update(camera, time);
        for index in changed {
            let i = index as usize;
            let record = self.record(scene, i, self.records[i].prev_world, self.records[i].visibility_offset);
            self.records[i] = record;
            queue.write_buffer(&self.instances, (i * std::mem::size_of::<GpuInstance>()) as u64, bytemuck::bytes_of(&self.records[i]));
        }
        self.plant_sim.write(queue, camera, time, dt);
    }

    /// Shadow casters moved by the last [`GpuScene::sync`], with their old and new bounds.
    pub(crate) fn shadow_moves(&self) -> &[ShadowMove] {
        &self.shadow_moves
    }

    /// Uploads whatever changed since the previous call and streams pages. Moved instances
    /// (`changes`, with every instance's previous-frame transform in `previous`) rewrite only
    /// their own records.
    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        changes: &SceneChanges,
        previous: &[Mat4],
        frame: u32,
    ) -> SyncResult {
        let geometry_changed = self.versions.0 != scene.geometry_version;
        if geometry_changed {
            self.upload_geometry(device, queue, scene);
        }
        // Masks depend on materials and their textures; instances carry their material's
        // alpha-masked and displaced flags for culling (a graph gaining or losing its
        // Displacement output changes the latter).
        let materials_changed = geometry_changed
            || self.versions.1 != scene.material_version
            || self.versions.3 != scene.texture_version
            || self.displacement_graphs_version != scene.graphs.displacement_version;
        if materials_changed {
            self.masks.sync(device, queue, scene);
            self.upload_materials(queue, scene);
        }
        // Palettes first: shadow moves of skinned instances cover their largest reach.
        self.skins.sync(device, queue, scene);
        // Switching tessellation changes the instances' displaced flags.
        let instances_changed = self.versions.2 != scene.instance_version
            || self.tessellation_enabled != self.tessellation_uploaded
            || self.wind_bones_enabled != self.wind_bones_uploaded;
        self.shadow_moves.clear();
        let mut bounds_grew = false;
        if materials_changed || instances_changed || changes.structural {
            // Last frame's transforms: `previous` already holds this frame's for the moved.
            let mut prev_world = previous.to_vec();
            for m in &changes.moved {
                prev_world[m.index as usize] = m.previous;
            }
            let old = std::mem::take(&mut self.records);
            self.upload_instances(device, queue, scene, &prev_world, geometry_changed);
            // Moves still count when materials changed in the same frame (without a structural
            // change, cached shadows are only invalidated under them).
            if old.len() == self.records.len() {
                for m in &changes.moved {
                    let i = m.index as usize;
                    self.shadow_moves.extend(shadow_move(scene, i, &old[i], &self.records[i], self.skin_reach(&scene.instances[i])));
                }
            }
        } else {
            bounds_grew = self.update_instances(queue, scene, changes);
        }
        self.versions = (scene.geometry_version, scene.material_version, scene.instance_version, scene.texture_version);
        self.displacement_graphs_version = scene.graphs.displacement_version;

        self.streamer.update(queue, &self.geometry, &scene.meshes, frame);
        self.streamer.upload_page_table(queue, &self.page_table);
        SyncResult { geometry_changed, bounds_grew }
    }

    /// Rewrites the records of the instances that moved (previous transform: last frame's) or
    /// came to rest (previous transform: the current one). Returns true if the scene bounds grew.
    fn update_instances(&mut self, queue: &wgpu::Queue, scene: &Scene, changes: &SceneChanges) -> bool {
        let rest = changes.settled.iter().map(|&i| (i, scene.instances[i as usize].transform));
        let mut updates: Vec<(u32, Mat4)> = changes.moved.iter().map(|m| (m.index, m.previous)).chain(rest).collect();
        updates.sort_unstable_by_key(|u| u.0);
        let (mut lo, mut hi) = self.bounds;
        for &(index, prev_world) in &updates {
            let i = index as usize;
            let instance = &scene.instances[i];
            let record = self.record(scene, i, prev_world, self.records[i].visibility_offset);
            self.shadow_moves.extend(shadow_move(scene, i, &self.records[i], &record, self.skin_reach(instance)));
            lo = lo.min(record.bounds.truncate() - record.bounds.w);
            hi = hi.max(record.bounds.truncate() + record.bounds.w);
            self.records[i] = record;
        }
        // One write per run of consecutive records.
        let mut k = 0;
        while k < updates.len() {
            let first = updates[k].0 as usize;
            let mut end = first + 1;
            while k + 1 < updates.len() && updates[k + 1].0 as usize == end {
                k += 1;
                end += 1;
            }
            let offset = (first * std::mem::size_of::<GpuInstance>()) as u64;
            queue.write_buffer(&self.instances, offset, bytemuck::cast_slice(&self.records[first..end]));
            k += 1;
        }
        // The shadow clipmap's depth range spans the scene bounds: they only grow between
        // structural changes, and growing invalidates the cached shadow pages.
        let grew = (lo, hi) != self.bounds;
        self.bounds = (lo, hi);
        grew
    }

    /// Records the feedback read-back (call at the end of the frame's command encoding).
    pub fn record_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u32) -> bool {
        self.streamer.record_feedback(encoder, &self.atomics, frame)
    }

    pub fn map_feedback(&self) {
        self.streamer.map_feedback();
    }

    fn upload_geometry(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let mut clusters: Vec<GpuCluster> = Vec::new();
        let mut pages: Vec<GpuPage> = Vec::new();
        self.meshes.clear();

        for mesh in &scene.meshes {
            let page_base = pages.len() as u32;
            let cluster_base = clusters.len() as u32;
            let global = |p: u32| if p == INVALID_PAGE { INVALID_PAGE } else { p + page_base };
            clusters.extend(mesh.clusters.iter().map(|c| GpuCluster { page: global(c.page), child_page: global(c.child_page), ..*c }));
            pages.extend(mesh.pages.iter().map(|p| GpuPage { cluster_offset: p.cluster_offset + cluster_base, ..*p }));
            self.meshes.push(MeshRange {
                page_offset: page_base,
                page_count: mesh.pages.len() as u32,
                cluster_offset: cluster_base,
                cluster_count: mesh.clusters.len() as u32,
                bounds: mesh.bounds.to_vec4(),
            });
        }

        let payload: usize = scene.meshes.iter().map(|m| m.payload_bytes()).sum();
        let streamed = scene.page_locations.iter().filter(|l| l.is_some()).count();
        log::info!(
            "uploading geometry: {} meshes ({streamed} streamed from page sources), {} pages, {} clusters, {:.1} MiB streamable ({:.1} MiB pool)",
            scene.meshes.len(),
            pages.len(),
            clusters.len(),
            payload as f64 / 1048576.0,
            self.geometry.size() as f64 / 1048576.0
        );

        self.page_count = pages.len() as u32;
        self.voxels = scene.meshes.iter().any(|m| m.has_voxels());
        let bone_tables = self.plants.upload_geometry(scene);
        self.assemblies.upload_geometry(device, queue, scene, &self.meshes, &bone_tables);
        self.clusters = storage_buffer_init(device, queue, "clusters", bytemuck::cast_slice(&clusters));
        self.pages = storage_buffer_init(device, queue, "pages", bytemuck::cast_slice(&pages));
        self.page_table = storage_buffer(device, "page table", pages.len() as u64 * 4, wgpu::BufferUsages::empty());
        if !self.streamer.reset(device, queue, &self.geometry, &scene.meshes, &scene.page_locations) {
            log::error!("geometry pool too small for the root pages; increase RendererConfig::geometry_pool_bytes");
        }
    }

    fn upload_materials(&mut self, queue: &wgpu::Queue, scene: &Scene) {
        let mut materials: Vec<GpuMaterial> =
            scene.materials.iter().enumerate().map(|(i, m)| gpu_material(scene, m, self.masks.material(i))).collect();
        if materials.len() > MAX_MATERIALS {
            log::warn!("{} materials exceed the limit of {MAX_MATERIALS}; instances of the rest render with the wrong material", materials.len());
        }
        materials.truncate(MAX_MATERIALS);
        (self.uv_extensions, self.resolve_kinds) = resolve_needs(&materials);
        self.material_kinds = materials.iter().map(|m| m.resolve).collect();
        materials.resize(MAX_MATERIALS, bytemuck::Zeroable::zeroed());
        queue.write_buffer(&self.materials, 0, bytemuck::cast_slice(&materials));
        self.material_uploads += 1;
        let header = std::mem::size_of::<tessellation::TessView>() as u64;
        let table = tessellation::material_table(scene);
        let bytes: &[u8] = bytemuck::cast_slice(&table);
        self.displacement_table_hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        queue.write_buffer(&self.tessellation, header, bytes);
    }

    /// How far (object space) the skin of `instance` may have moved any vertex from its rest pose
    /// (0 if not skinned).
    fn skin_reach(&self, instance: &Instance) -> f32 {
        instance.skin.map_or(0.0, |skin| self.skins.largest_reach(skin))
    }

    /// The GPU record of `instance` (moved from `prev_world` since the previous frame).
    fn record(&mut self, scene: &Scene, index: usize, prev_world: Mat4, visibility_offset: u32) -> GpuInstance {
        let instance = &scene.instances[index];
        let mesh = self.meshes[instance.mesh.0 as usize];
        let world = instance.transform;
        let linear = Mat3::from_mat4(world);
        let max_scale = linear.x_axis.length().max(linear.y_axis.length()).max(linear.z_axis.length());
        // Inverse transpose, rescaled so that uniform scale leaves normals unit length.
        let normal = linear.inverse().transpose() * max_scale;
        let center = world.transform_point3(mesh.bounds.truncate());
        // The plant origin of the wind is the instance's own position.
        let origin = world.w_axis;
        let mut flags = gpu_instance_flags(scene, instance, self.masks.material(instance.material.0 as usize).is_some())
            | instance_skin_flags(scene, instance);
        if self.assemblies.is_assembly(instance.mesh.0) {
            flags |= instance_flags::ASSEMBLY;
        }
        if self.tessellation_enabled && displaced(scene, instance) && flags & (instance_flags::HIDDEN | instance_flags::ALPHA_MASKED) == 0 {
            flags |= instance_flags::DISPLACED;
            if tessellation::displacement_animated(scene, &scene.materials[instance.material.0 as usize]) {
                flags |= instance_flags::ANIMATED;
            }
        }
        let response = instance_wind(scene, instance);
        let (mut plant, mut wind) = match self.plants.instance(scene, instance, response, self.wind_bones_enabled) {
            Some((plant, wind)) => (plant, wind),
            None => ([0; 4], response),
        };
        if let Some(slot) = self.plant_sim.slot(index as u32).filter(|_| plant[0] != 0) {
            // Promoted: the simulation may deviate each bone by SIM_DEVIATION times its
            // parametric bound, so every layer of the plant's bound grows by as much.
            plant[2] = slot + 1;
            wind = (wind.truncate() * (1.0 + SIM_DEVIATION)).extend(wind.w);
        }
        GpuInstance {
            world,
            prev_world,
            normal_matrix: [normal.x_axis.extend(origin.x), normal.y_axis.extend(origin.y), normal.z_axis.extend(origin.z)],
            bounds: center.extend(mesh.bounds.w * max_scale),
            page_offset: mesh.page_offset,
            page_count: mesh.page_count,
            cluster_offset: mesh.cluster_offset,
            cluster_count: mesh.cluster_count,
            material: instance.material.0,
            visibility_offset,
            max_scale,
            flags,
            wind,
            plant,
        }
    }

    fn upload_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, prev_world: &[Mat4], layout_changed: bool) {
        let count = scene.instances.len();
        // Visibility bits start after the page feedback words.
        let mut visibility_bits = self.page_count * 32;
        let mut wind_shadows = false;
        let mut windy = false;
        self.plants.reset_headers();
        // Instance indices may have changed: promotion starts over (next frame).
        self.plant_sim.set_candidates(Vec::new());
        let mut instances: Vec<GpuInstance> = Vec::with_capacity(count);
        for (index, (instance, &prev_world)) in scene.instances.iter().zip(prev_world).enumerate() {
            let gpu = self.record(scene, index, prev_world, visibility_bits);
            wind_shadows |= instance.cast_shadow && gpu.wind != Vec4::ZERO;
            windy |= gpu.wind != Vec4::ZERO;
            visibility_bits += gpu.cluster_count;
            instances.push(gpu);
        }
        self.plants.upload(device, queue);
        // Plants the simulation may promote: small ones (a slot holds SIM_BONES bones), not
        // assembly parts.
        let candidates = instances
            .iter()
            .enumerate()
            .filter(|(i, gpu)| {
                let mesh = &scene.meshes[scene.instances[*i].mesh.0 as usize];
                gpu.plant[0] != 0 && gpu.flags & instance_flags::ASSEMBLY == 0 && mesh.bones.as_ref().is_some_and(|b| b.bones.len() as u32 <= SIM_BONES)
            })
            .map(|(i, gpu)| Candidate { index: i as u32, center: gpu.bounds.truncate(), radius: gpu.bounds.w, bones: scene.meshes[scene.instances[i].mesh.0 as usize].bones.as_ref().map_or(0, |b| b.bones.len() as u32) })
            .collect();
        self.plant_sim.set_candidates(candidates);

        let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for i in &instances {
            lo = lo.min(i.bounds.truncate() - i.bounds.w);
            hi = hi.max(i.bounds.truncate() + i.bounds.w);
        }
        self.bounds = if instances.is_empty() { (glam::Vec3::ZERO, glam::Vec3::ONE) } else { (lo, hi) };
        let visibility_bits = self.assemblies.upload_instances(device, queue, scene, visibility_bits);
        // The pose pass visits the plants with bones and the assembly part slots.
        let plants: Vec<u32> = instances.iter().enumerate().filter(|(_, gpu)| gpu.plant[0] != 0).map(|(i, _)| i as u32).collect();
        // The materials in the wind: the plants' shading models (their bones may arrive later in
        // the browser, whose first frames draw the scene's meshes before their final bake).
        self.plant_materials = scene.instances.iter().filter(|i| instance_wind(scene, i) != Vec4::ZERO).map(|i| i.material.0).collect();
        self.plant_materials.sort_unstable();
        self.plant_materials.dedup();
        self.plant_pose.set_instances(device, queue, &plants, count as u32 + self.assemblies.record_count(), self.assemblies.part_capacity());
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        // Assemblies' culling records and part instances are written after the scene's by the GPU.
        let extra = self.assemblies.record_count() + self.assemblies.part_capacity();
        let size = bytes.len() as u64 + extra as u64 * std::mem::size_of::<GpuInstance>() as u64;
        if self.instances.size() < size.max(16) || self.instance_count as usize != count {
            self.instances = storage_buffer(device, "instances", size, wgpu::BufferUsages::empty());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.instances, 0, bytes);
        }
        if layout_changed || self.instance_count as usize != count {
            // Instance layout changed: start occlusion culling from scratch.
            let words = (visibility_bits as u64).div_ceil(32);
            self.atomics = storage_buffer(device, "culling atomics", words * 4, wgpu::BufferUsages::COPY_SRC);
        }

        self.instance_count = count as u32;
        self.wind_shadows = wind_shadows;
        self.windy = windy;
        self.displaced = instances.iter().any(|i| i.flags & instance_flags::DISPLACED != 0);
        self.displaceable = displaceable(scene);
        let animated_caster = instance_flags::ANIMATED | instance_flags::CAST_SHADOW;
        self.animated_shadows = instances.iter().any(|i| i.flags & animated_caster == animated_caster);
        self.tessellation_uploaded = self.tessellation_enabled;
        self.wind_bones_uploaded = self.wind_bones_enabled;
        self.records = instances;
    }
}

/// What [`GpuScene::sync`] changed.
pub(crate) struct SyncResult {
    /// Geometry was rebuilt.
    pub geometry_changed: bool,
    /// A moved instance extended the scene bounds.
    pub bounds_grew: bool,
}

/// The shadow move of instance `index` from record `old` to `new`, if it casts shadows and moved.
/// `reach` (object space) pads the rest-pose bounds: a skinned instance's footprints cover its
/// skin's largest reach, as cached pages may show any pose since the scene was loaded.
fn shadow_move(scene: &Scene, index: usize, old: &GpuInstance, new: &GpuInstance, reach: f32) -> Option<ShadowMove> {
    let instance = &scene.instances[index];
    let mut aabb = scene.meshes[instance.mesh.0 as usize].aabb;
    aabb.min -= Vec3::splat(reach);
    aabb.max += Vec3::splat(reach);
    let pad = |record: &GpuInstance| record.bounds + Vec4::W * (reach * record.max_scale);
    // Hidden instances cast no shadow (their records carry no shadow flag).
    (instance.cast_shadow && instance.visible && old.world != new.world).then(|| ShadowMove {
        instance: index as u32,
        aabb,
        old_world: old.world,
        new_world: new.world,
        old: pad(old),
        new: pad(new),
        movable: instance.mobility == Mobility::Movable,
    })
}

fn gpu_instance_flags(scene: &Scene, instance: &Instance, alpha_masked: bool) -> u32 {
    if !instance.visible {
        // No shadow flag either: shadow invalidation passes skip it like the culling does.
        return instance_flags::HIDDEN;
    }
    let mut flags = if instance.cast_shadow { instance_flags::CAST_SHADOW } else { 0 };
    if instance.mobility == Mobility::Movable {
        flags |= instance_flags::MOVABLE;
    }
    if alpha_masked {
        flags |= instance_flags::ALPHA_MASKED;
    }
    if scene.materials.get(instance.material.0 as usize).is_some_and(|m| m.kind.is_translucent()) {
        flags |= instance_flags::TRANSLUCENT;
    }
    flags
}

/// Whether some shown instance's material displaces it (tessellated once tessellation is on).
/// Hidden instances don't count: a page that hides the displaced ones (the Opus55 site empties
/// the gallery but its floor) needs none of the tessellator's 20 pipelines, 7 s of compiles in
/// Firefox, and their group is held (`Renderer::held_groups`).
pub(crate) fn displaceable(scene: &Scene) -> bool {
    scene.instances.iter().any(|i| i.visible && displaced(scene, i))
}

/// Whether the instance's material displaces it (skinned meshes are not tessellated).
fn displaced(scene: &Scene, instance: &Instance) -> bool {
    instance.skin.is_none() && scene.materials.get(instance.material.0 as usize).is_some_and(|m| tessellation::displacement_source(scene, m).is_some())
}

/// The instance's wind response; translucent instances are drawn by the forward pass, which
/// doesn't animate them, so they stay static everywhere.
fn instance_wind(scene: &Scene, instance: &Instance) -> Vec4 {
    match scene.materials.get(instance.material.0 as usize) {
        Some(m) if !m.kind.is_translucent() && !m.wind.is_static() => m.wind.to_vec4(),
        _ => Vec4::ZERO,
    }
}

fn gpu_material(scene: &Scene, m: &Material, mask: Option<MaterialMask>) -> GpuMaterial {
    let layer = |t: Option<TextureHandle>| t.map_or(255, |t| t.0.min(254));
    let from_images = m.texture.is_some_and(|t| matches!(scene.textures.get(t.0 as usize), Some(TextureSource::Images(_))));
    let alpha_test = mask.map_or(0, |mask| {
        (mask.layer + 1) << material_flags::ALPHA_MASK_SHIFT | (mask.cutoff as u32) << material_flags::ALPHA_CUTOFF_SHIFT
    });
    let has_occlusion = |t: TextureHandle| matches!(scene.textures.get(t.0 as usize), Some(TextureSource::Images(i)) if i.occlusion.is_some());
    let occlusion = m.occlusion_texture.filter(|&t| has_occlusion(t) && m.occlusion_strength > 0.0);
    let occlusion_in_texture = occlusion.is_some() && occlusion == m.texture && m.occlusion_uv == m.uv;
    // The material's tiling applies to every mesh-UV mapping.
    let tiled = |t: &UvTransform| gpu_uv_transform(&UvTransform { matrix: Affine2::from_scale(Vec2::splat(m.uv_scale)) * t.matrix, ..*t });
    GpuMaterial {
        base_color: m.base_color.extend(m.opacity),
        emissive: m.emissive_color.extend(m.emissive),
        roughness: m.roughness,
        metallic: m.metallic,
        kind: m.kind as u32,
        flags: if from_images { material_flags::IMAGE_TEXTURE } else { 0 }
            | if occlusion_in_texture { material_flags::OCCLUSION_IN_TEXTURE } else { 0 }
            | alpha_test,
        texture: m.texture.map_or(u32::MAX, |t| t.0),
        uv_scale: m.uv_scale,
        normal_strength: m.normal_strength,
        mapping: m.mapping as u32,
        detail_scale: m.detail_scale,
        detail_strength: m.detail_strength,
        layers: m.layers.iter().enumerate().fold(0, |acc, (i, &t)| acc | layer(t) << (i * 8)),
        emissive_texture: m.emissive_texture.map_or(u32::MAX, |t| t.0),
        uv: [tiled(&m.uv), tiled(&m.emissive_uv), tiled(&m.occlusion_uv)],
        occlusion_texture: occlusion.map_or(u32::MAX, |t| t.0),
        occlusion_strength: m.occlusion_strength.clamp(0.0, 1.0),
        // Extended materials are patched in by `substrate_gpu.rs`. Translucent kinds never reach
        // the visibility buffer; the clamp keeps them off the extended slots.
        resolve: (m.kind as u32).min(crate::passes::material::SKY_SLOT),
        _pad: 0,
    }
}

/// Whether the materials need the resolve's `UV_EXTENSIONS` permutation, and the shading models
/// they resolve with (a bit per model; see [`crate::passes::material`]).
fn resolve_needs(materials: &[GpuMaterial]) -> (bool, u32) {
    (materials.iter().any(needs_uv_extensions), materials.iter().fold(0, |kinds, m| kinds | (1 << m.resolve.min(31))))
}

/// [`resolve_needs`] of `scene`'s materials, without uploading them (a scene about to be
/// switched to: [`crate::Renderer::prefetch_next_scene`]).
pub(crate) fn scene_resolve_needs(scene: &Scene) -> (bool, u32) {
    let materials: Vec<GpuMaterial> = scene.materials.iter().take(MAX_MATERIALS).map(|m| gpu_material(scene, m, None)).collect();
    resolve_needs(&materials)
}

/// True if the material uses a texture coordinate feature of the resolve's `UV_EXTENSIONS`
/// permutation (see `passes/material.rs`). Texture transforms themselves are always applied.
fn needs_uv_extensions(m: &GpuMaterial) -> bool {
    let separate_occlusion = m.occlusion_texture != u32::MAX && m.flags & material_flags::OCCLUSION_IN_TEXTURE == 0;
    separate_occlusion || m.uv.iter().any(|t| t.uv_set != 0 || t.wrap != 0)
}

fn gpu_uv_transform(t: &UvTransform) -> GpuUvTransform {
    let (m, o) = (t.matrix.matrix2, t.matrix.translation);
    GpuUvTransform {
        u: [m.x_axis.x, m.y_axis.x, o.x],
        uv_set: t.set.min(1),
        v: [m.x_axis.y, m.y_axis.y, o.y],
        wrap: t.wrap[0] as u32 | (t.wrap[1] as u32) << 4,
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::scene::{ImageTexture, TextureDesc, TexturePattern};

    #[test]
    fn rare_texture_coordinates_select_the_extended_resolve() {
        use crate::scene::{UvTransform, Wrap};
        let mut scene = Scene::new();
        let orm = Some(std::sync::Arc::new(crate::scene::Image { width: 1, height: 1, pixels: vec![255; 4] }));
        let layer = scene.add_image_texture(ImageTexture { metallic_roughness: orm.clone(), occlusion: orm, ..Default::default() });
        let needs = |m: Material| needs_uv_extensions(&gpu_material(&scene, &m, None));
        let textured = Material { texture: Some(layer), ..Default::default() };
        // Transforms and occlusion in the material's own texture are part of the common path.
        let moved = UvTransform { matrix: glam::Affine2::from_translation(glam::Vec2::X), ..UvTransform::IDENTITY };
        assert!(!needs(Material { uv: moved, occlusion_texture: Some(layer), occlusion_uv: moved, ..textured }));
        assert!(!needs(Material::default()));
        // The second UV set, clamping and an occlusion layer sampled with other coordinates are not.
        assert!(needs(Material { uv: UvTransform { set: 1, ..UvTransform::IDENTITY }, ..textured }));
        assert!(needs(Material { emissive_uv: UvTransform { wrap: [Wrap::ClampToEdge, Wrap::Repeat], ..UvTransform::IDENTITY }, ..textured }));
        assert!(needs(Material { occlusion_texture: Some(layer), occlusion_uv: moved, ..textured }));
    }

    #[test]
    fn image_textures_flag_their_materials() {
        let mut scene = Scene::new();
        let procedural = scene.add_texture(TextureDesc::new(TexturePattern::Rock, Vec3::ONE, Vec3::ONE));
        let images = scene.add_image_texture(ImageTexture::default());
        let gpu = |texture| gpu_material(&scene, &Material { texture: Some(texture), ..Default::default() }, None);
        assert_eq!(gpu(procedural).flags & material_flags::IMAGE_TEXTURE, 0);
        assert_eq!(gpu(images).flags & material_flags::IMAGE_TEXTURE, material_flags::IMAGE_TEXTURE);

        let glowing = Material { emissive_color: Vec3::new(1.0, 0.5, 0.0), emissive_texture: Some(images), emissive: 2.0, ..Default::default() };
        let g = gpu_material(&scene, &glowing, None);
        assert_eq!(g.emissive, glam::Vec4::new(1.0, 0.5, 0.0, 2.0));
        assert_eq!((g.texture, g.emissive_texture, g.layers), (u32::MAX, images.0, u32::MAX));
    }

    #[test]
    fn alpha_tests_pack_above_the_material_flags() {
        let mut scene = Scene::new();
        let images = scene.add_image_texture(ImageTexture::default());
        let material = Material { texture: Some(images), alpha_cutoff: Some(0.5), ..Default::default() };
        let g = gpu_material(&scene, &material, Some(MaterialMask { layer: 2, cutoff: 128 }));
        assert_eq!(g.flags, material_flags::IMAGE_TEXTURE | 3 << 16 | 128 << 8);
        assert_eq!(gpu_material(&scene, &material, None).flags, material_flags::IMAGE_TEXTURE);
    }

    #[test]
    fn translucent_materials_flag_their_instances() {
        use crate::scene::MaterialKind;
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let opaque = scene.add_material(Material::default());
        let water = scene.add_material(Material { kind: MaterialKind::Water, ..Default::default() });
        let glass = scene.add_material(Material { kind: MaterialKind::Glass, opacity: 0.5, ..Default::default() });
        let rock = scene.add_instance(mesh, opaque, Mat4::IDENTITY);
        let lake = scene.add_instance(mesh, water, Mat4::IDENTITY);
        let pane = scene.add_instance(mesh, glass, Mat4::IDENTITY);
        scene.set_cast_shadow(lake, false);
        let flags = |h: crate::scene::InstanceHandle, masked| gpu_instance_flags(&scene, &scene.instances[h.0 as usize], masked);
        assert_eq!(flags(rock, false), instance_flags::CAST_SHADOW);
        assert_eq!(flags(rock, true), instance_flags::CAST_SHADOW | instance_flags::ALPHA_MASKED);
        assert_eq!(flags(lake, false), instance_flags::TRANSLUCENT);
        assert_eq!(flags(pane, false), instance_flags::CAST_SHADOW | instance_flags::TRANSLUCENT);
        assert_eq!(gpu_material(&scene, &scene.materials[glass.0 as usize], None).base_color.w, 0.5);
    }

    #[test]
    fn instances_take_their_material_wind_response_unless_translucent() {
        use crate::scene::MaterialKind;
        use crate::wind::WindResponse;
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let rock = scene.add_material(Material::default());
        let grass = scene.add_material(Material { kind: MaterialKind::Foliage, wind: WindResponse::GRASS, ..Default::default() });
        let glass = scene.add_material(Material { kind: MaterialKind::Glass, wind: WindResponse::GRASS, ..Default::default() });
        let handles = [rock, grass, glass].map(|m| scene.add_instance(mesh, m, Mat4::IDENTITY));
        let wind = |i: usize| instance_wind(&scene, &scene.instances[handles[i].0 as usize]);
        assert_eq!(wind(0), Vec4::ZERO);
        assert_eq!(wind(1), WindResponse::GRASS.to_vec4());
        assert_eq!(wind(2), Vec4::ZERO);
    }

    #[test]
    fn moved_shadow_casters_report_both_poses_and_their_mobility() {
        let mut scene = Scene::new();
        let mesh = scene.add_mesh(helios_geometry::VirtualMesh::default());
        let material = scene.add_material(Material::default());
        let caster = scene.add_instance(mesh, material, Mat4::IDENTITY);
        let flame = scene.add_instance(mesh, material, Mat4::IDENTITY);
        scene.set_cast_shadow(flame, false);
        scene.set_mobility(caster, Mobility::Movable);
        let record = |world: Mat4| GpuInstance { world, bounds: Vec4::new(0.0, 0.0, 0.0, 1.0), ..bytemuck::Zeroable::zeroed() };
        // A turn about the bounds' centre leaves the bounding sphere as it was, but still moves
        // the caster.
        let (old, new) = (record(Mat4::IDENTITY), record(Mat4::from_rotation_y(0.3)));
        let m = shadow_move(&scene, caster.0 as usize, &old, &new, 0.0).expect("a move");
        assert_eq!((m.old_world, m.new_world, m.movable), (old.world, new.world, true));
        assert!(shadow_move(&scene, caster.0 as usize, &old, &old, 0.0).is_none(), "at rest");
        assert!(shadow_move(&scene, flame.0 as usize, &old, &new, 0.0).is_none(), "casts no shadow");
        assert_eq!(gpu_instance_flags(&scene, &scene.instances[caster.0 as usize], false), instance_flags::CAST_SHADOW | instance_flags::MOVABLE);
        // A skin's reach pads both poses' boxes and spheres.
        let scaled = |world: Mat4| GpuInstance { max_scale: 2.0, ..record(world) };
        let m = shadow_move(&scene, caster.0 as usize, &scaled(Mat4::IDENTITY), &scaled(Mat4::from_rotation_y(0.3)), 0.5).expect("a move");
        let aabb = scene.meshes[mesh.0 as usize].aabb;
        assert_eq!((m.aabb.min, m.aabb.max), (aabb.min - 0.5, aabb.max + 0.5));
        assert_eq!((m.old.w, m.new.w), (2.0, 2.0));
    }
}
