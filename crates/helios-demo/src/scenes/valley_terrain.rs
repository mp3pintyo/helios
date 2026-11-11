//! The valley's mesh terrain (`helios_geometry::mesh_terrain`): a rectangle north of the lake
//! where the landscape becomes a mesh — a cliff band with an overhanging shelf and a cave the
//! river springs from, a natural arch on the hillside to the east, and the river's channel down
//! to the lake. The landscape heightfield leaves the rectangle out and the mesh terrain meets
//! it seamlessly; everything else in the valley stays where it was.

use std::sync::{Arc, RwLock};

use glam::{Mat4, Vec2, Vec3};
use helios_geometry::mesh_terrain::{ChunkConfig, LandscapeSeam, MeshTerrain, Modifier, TerrainChunk, TerrainCollider};
use helios_geometry::{build_virtual_mesh, BuildConfig, Mesh, MeshSdf, SdfMode, SimplifyOptions, Vertex};
use helios_pcg::{Graph, Node, Op, SpawnEntry, SpawnPart, TransformParams, TriangleSurface, Volume};
use helios_render::{Material, MaterialHandle, MaterialKind, MeshHandle, Scene, TextureDesc, TextureMapping, TexturePattern};

use super::{add_baked_with, add_built};

/// The rectangle the landscape leaves out (on the grid lines of both detail levels' grids:
/// multiples of 900 / 384 m).
pub const HOLE_MIN: Vec2 = Vec2::new(-28.125, -60.9375);
pub const HOLE_MAX: Vec2 = Vec2::new(46.875, 14.0625);
/// The meshed region: the hole plus a band where the mesh terrain dips under the landscape,
/// rounded to whole chunks.
const REGION_MIN: Vec2 = Vec2::new(-30.5, -63.5);
const REGION_MAX: Vec2 = Vec2::new(49.5, 16.5);
const SEAM_DIP: f32 = 0.15;

/// Cliff band (mesa) facing the initial view, with its shelf and the cave.
const MESA: Vec3 = Vec3::new(-6.0, 6.5, -48.0);
const MESA_YAW: f32 = -0.21;
/// Free-standing fin with the arch, on the hillside east of the sculpture.
const ARCH: Vec3 = Vec3::new(34.0, 4.2, -22.0);
const ARCH_YAW: f32 = -0.828;

/// The river: (x, bed height, z) from inside the cave to the lake.
pub const RIVER_PATH: [Vec3; 8] = [
    Vec3::new(-11.0, 0.4, -48.0),
    Vec3::new(-12.0, 0.0, -42.5),
    Vec3::new(-10.0, -0.9, -34.0),
    Vec3::new(-8.0, -1.7, -24.0),
    Vec3::new(-7.5, -1.95, -14.0),
    Vec3::new(-8.5, -2.3, -4.0),
    Vec3::new(-9.5, -2.45, 4.0),
    Vec3::new(-8.5, -2.9, 12.0),
];
pub const RIVER_WIDTH: f32 = 3.0;
const RIVER_BANK: f32 = 2.2;

/// A point given in a frame turned by `yaw` about +Y around `center` (the modifiers' frame).
fn turned(center: Vec3, yaw: f32, local: Vec3) -> Vec3 {
    let (s, c) = yaw.sin_cos();
    center + Vec3::new(c * local.x + s * local.z, local.y, -s * local.x + c * local.z)
}

/// The terrain over the valley's landscape (`ground`), for a landscape grid of `resolution`
/// cells.
pub fn mesh_terrain(ground: fn(f32, f32) -> f32, size: f32, resolution: u32) -> MeshTerrain {
    let rock = |center, half_extents, yaw, taper, rounding, blend, roughness, seed| Modifier::Rock { center, half_extents, yaw, taper, rounding, blend, roughness, seed };
    let modifiers = vec![
        // The river's channel first: the rock and the cave are carved on top of it.
        Modifier::Channel { path: RIVER_PATH.to_vec(), width: RIVER_WIDTH, bank: RIVER_BANK },
        // The cliff band, its faces leaning out a little (negative taper).
        rock(MESA, Vec3::new(13.5, 6.5, 6.0), MESA_YAW, -0.12, 3.0, 1.8, 1.5, 21),
        // A rock shelf near its top, overhanging the meadow by ~5 m.
        rock(turned(MESA, MESA_YAW, Vec3::new(4.0, 5.3, 8.5)), Vec3::new(6.0, 1.4, 3.8), MESA_YAW, 0.0, 1.2, 1.5, 0.8, 22),
        // The cave through the cliff, a little above the meadow at its mouth.
        Modifier::Tunnel {
            a: turned(MESA, MESA_YAW, Vec3::new(-5.0, -3.5, 10.0)),
            b: turned(MESA, MESA_YAW, Vec3::new(-5.0, -0.8, -10.0)),
            radius: 2.4,
            blend: 1.0,
            roughness: 0.7,
        },
        // The arch: a fin narrowing upwards, pierced near its foot.
        rock(ARCH, Vec3::new(8.0, 5.0, 1.8), ARCH_YAW, 0.25, 1.5, 1.5, 0.9, 23),
        Modifier::Tunnel { a: turned(ARCH, ARCH_YAW, Vec3::new(0.0, -1.2, -4.0)), b: turned(ARCH, ARCH_YAW, Vec3::new(0.0, -1.2, 4.0)), radius: 3.2, blend: 1.0, roughness: 0.6 },
    ];
    MeshTerrain {
        base: Arc::new(move |x, z| ground(x, z)),
        modifiers,
        region_min: REGION_MIN,
        region_max: REGION_MAX,
        seam: Some(LandscapeSeam { size, resolution, hole_min: HOLE_MIN, hole_max: HOLE_MAX, dip: SEAM_DIP, band: size / 384.0 }),
    }
}

/// The landscape mesh with the hole left out, baked with a heightfield distance field (and so
/// the runtime virtual texture's heightfield) that covers the hole with the mesh terrain's
/// ground — the channel included, the rock left to the chunks' own distance fields.
pub fn landscape(scene: &mut Scene, terrain: &MeshTerrain, size: f32, resolution: u32) -> MeshHandle {
    let ground = terrain.base.clone();
    let (x0, x1, z0, z1) = terrain.seam.expect("the valley's terrain replaces a landscape rectangle").hole_cells();
    let terrain = terrain.clone();
    add_built(scene, "terrain", 2 * (resolution as u64) * (resolution as u64), move || {
        // `primitives::terrain`'s grid, with the valley's heights.
        let n = resolution + 1;
        let vertices: Vec<Vertex> = (0..n * n)
            .map(|i| {
                let uv = Vec2::new((i % n) as f32, (i / n) as f32) / resolution as f32;
                let xz = (uv - 0.5) * size;
                Vertex::new(Vec3::new(xz.x, ground(xz.x, xz.y), xz.y), Vec3::Y, uv)
            })
            .collect();
        let full = Mesh::new(vertices, Vec::new());
        // The distance field's surface: the mesh terrain's ground inside its region.
        let mut filled = full.clone();
        filled.indices = (0..resolution)
            .flat_map(|z| (0..resolution).map(move |x| z * n + x))
            .flat_map(|i| [i, i + n, i + 1, i + 1, i + n, i + n + 1])
            .collect();
        for v in &mut filled.vertices {
            let [x, _, z] = v.position;
            if x >= REGION_MIN.x && x <= REGION_MAX.x && z >= REGION_MIN.y && z <= REGION_MAX.y {
                v.position[1] = terrain.height(x, z);
            }
        }
        let mut holed = full.clone();
        holed.indices = (0..resolution)
            .flat_map(|z| (0..resolution).map(move |x| (x, z)))
            .filter(|&(x, z)| !(x0..x1).contains(&x) || !(z0..z1).contains(&z))
            .flat_map(|(x, z)| {
                let i = z * n + x;
                [i, i + n, i + 1, i + 1, i + n, i + n + 1]
            })
            .collect();
        // Leave out the hole's vertices too.
        let mut remap = vec![u32::MAX; holed.vertices.len()];
        let mut vertices = Vec::with_capacity(holed.vertices.len());
        for i in holed.indices.iter_mut() {
            if remap[*i as usize] == u32::MAX {
                remap[*i as usize] = vertices.len() as u32;
                vertices.push(holed.vertices[*i as usize]);
            }
            *i = remap[*i as usize];
        }
        holed.vertices = vertices;
        holed.compute_normals();
        let start = web_time::Instant::now();
        let mut vm = build_virtual_mesh(&holed, &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let built = start.elapsed();
        let triangles: Vec<[Vec3; 3]> = filled.triangles().map(|t| t.map(|i| filled.vertices[i as usize].position())).collect();
        vm.sdf = MeshSdf::bake_heightfield(&triangles, 512);
        log::info!("landscape: virtual mesh {:.0} ms, heightfield {:.0} ms", built.as_secs_f64() * 1e3, (start.elapsed() - built).as_secs_f64() * 1e3);
        vm
    })
}

/// The collider of the last built valley terrain (camera floor, placement).
static COLLIDER: RwLock<Option<Arc<TerrainCollider>>> = RwLock::new(None);

pub fn collider() -> Option<Arc<TerrainCollider>> {
    COLLIDER.read().ok().and_then(|c| c.clone())
}

/// Whether (x, z) lies in the meshed rectangle.
pub fn in_hole(x: f32, z: f32) -> bool {
    x > HOLE_MIN.x && x < HOLE_MAX.x && z > HOLE_MIN.y && z < HOLE_MAX.y
}

/// The ground under `p`: in the mesh terrain the highest surface at or below it (so the floor
/// under the arch, or the cliff's top above the meadow), elsewhere `fallback`.
pub fn ground_below(p: Vec3, fallback: f32) -> f32 {
    if in_hole(p.x, p.z) {
        if let Some(h) = collider().and_then(|c| c.ground_below(p)) {
            return h;
        }
    }
    fallback
}

/// The top surface at (x, z) (the cliff's top, not the meadow under its shelf).
pub fn top(x: f32, z: f32, fallback: f32) -> f32 {
    if in_hole(x, z) {
        if let Some(h) = collider().and_then(|c| c.top(x, z)) {
            return h;
        }
    }
    fallback
}

/// Meshes the terrain into chunks, adds them with the mesh terrain's material, scatters loose
/// rock on them (PCG on the generated triangles: it lands on the cliff top and ledges and in the
/// meadow, never inside the rock or under the shelf) and publishes the collider.
pub fn add(scene: &mut Scene, terrain: &MeshTerrain, material: MaterialHandle, rocks: &[MeshHandle], rock_materials: &[MaterialHandle]) -> Vec<TerrainChunk> {
    let start = web_time::Instant::now();
    let chunks = terrain.build_chunks(&ChunkConfig::default());
    let meshed = start.elapsed();
    let triangles: usize = chunks.iter().map(|c| c.mesh.triangle_count()).sum();
    let fine = chunks.iter().filter(|c| c.cell < 0.4).count();
    log::info!(
        "mesh terrain: {} chunks ({fine} fine), {triangles} triangles ({} skirt) meshed in {:.0} ms",
        chunks.len(),
        chunks.iter().map(|c| c.skirt_triangles).sum::<usize>(),
        meshed.as_secs_f64() * 1e3
    );
    let bounds = terrain.volume_bounds();
    for chunk in &chunks {
        // Chunks holding rock get a distance field of their own; the rest of the ground is in
        // the landscape's heightfield field.
        let aabb = chunk.mesh.aabb();
        let rocky = bounds.iter().any(|b| b.min.x <= aabb.max.x && b.max.x >= aabb.min.x && b.min.z <= aabb.max.z && b.max.z >= aabb.min.z);
        let config = BuildConfig {
            sdf: if rocky { SdfMode::Volume { resolution: 40 } } else { SdfMode::None },
            // Chunk borders stay put at every level of detail (neighbours share them exactly),
            // on one absolute quantisation grid.
            simplify: SimplifyOptions { lock_borders: true, ..Default::default() },
            position_precision: Some(8),
            ..Default::default()
        };
        let mesh = add_baked_with(scene, "mesh terrain", &chunk.mesh, config);
        scene.add_instance(mesh, material, Mat4::IDENTITY);
    }
    let collider = Arc::new(TerrainCollider::new(chunks.iter().map(|c| &c.mesh), 2.0));
    if let Ok(mut c) = COLLIDER.write() {
        *c = Some(collider.clone());
    }
    scatter_scree(scene, &collider, rocks, rock_materials);
    log::info!("mesh terrain: built in {:.0} ms (collider {} triangles)", start.elapsed().as_secs_f64() * 1e3, collider.triangle_count());
    chunks
}

/// Loose rocks on the cliff: a PCG graph sampling the generated triangles around the cliff and
/// the arch (UE's PCG on a mesh terrain), kept on gentle slopes (ledges, the cliff top, the
/// meadow at the foot - never a wall or the underside of the shelf), pruned and transformed.
fn scatter_scree(scene: &mut Scene, collider: &TerrainCollider, rocks: &[MeshHandle], materials: &[MaterialHandle]) {
    let surface = TriangleSurface { triangles: collider.triangles().to_vec() };
    let entries = (0..rocks.len())
        .map(|i| SpawnEntry { weight: 1.0, parts: vec![SpawnPart { mesh: format!("rock{i}"), material: format!("rock{i}") }] })
        .collect();
    let graph = Graph {
        seed: 77,
        nodes: vec![
            Node::named("sample terrain", Op::SurfaceSampler { points_per_m2: 0.035, jitter: 1.0 }),
            Node::named("ledges", Op::SlopeFilter { min_deg: 0.0, max_deg: 32.0 }),
            Node::named("prune", Op::SelfPruning { radius: 2.5 }),
            Node::new(Op::Transform(TransformParams { scale_min: 0.25, scale_max: 0.8, scale_jitter: 0.3, random_yaw: true, tilt_deg: 15.0, align_to_normal: 0.6, offset_min: -0.25, offset_max: -0.1 })),
            Node::named("scree", Op::Spawner { entries }),
        ],
    };
    let mut spawned = 0;
    for (center, half) in [(MESA, Vec3::new(22.0, 12.0, 14.0)), (ARCH, Vec3::new(12.0, 8.0, 9.0))] {
        let volume = Volume { center: center.into(), half_extents: half.into(), yaw_deg: 0.0 };
        let output = match helios_pcg::evaluate(&graph, &volume, &surface) {
            Ok(output) => output,
            Err(e) => {
                log::warn!("mesh terrain scree: {e}");
                continue;
            }
        };
        for instance in &output.instances {
            let at = instance.transform.w_axis.truncate();
            // Not in the river.
            if river_distance(Vec2::new(at.x, at.z)).0 < RIVER_WIDTH {
                continue;
            }
            let part = &output.parts[instance.part as usize];
            let i: usize = part.mesh.trim_start_matches("rock").parse().unwrap_or(0);
            scene.add_instance(rocks[i % rocks.len()], materials[i % materials.len()], instance.transform);
            spawned += 1;
        }
    }
    log::info!("mesh terrain: {spawned} scree rocks scattered by PCG on its triangles");
}

/// Distance in (x, z) to the river's centre line and the bed height there.
pub fn river_distance(p: Vec2) -> (f32, f32) {
    let mut best = (f32::MAX, 0.0);
    for w in RIVER_PATH.windows(2) {
        let (a, b) = (Vec2::new(w[0].x, w[0].z), Vec2::new(w[1].x, w[1].z));
        let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
        let d = (p - (a + (b - a) * t)).length();
        if d < best.0 {
            best = (d, w[0].y + (w[1].y - w[0].y) * t);
        }
    }
    best
}

/// The landscape material made for meshes: no runtime virtual texture (its top-down pages
/// can't hold cliffs or overhangs), every layer projected triplanar, and cliff rock (bedding,
/// fracture planes, chips; in the landscape's rock colours) in place of the landscape's cracked
/// rock, which tiles into a paving pattern on walls this large.
pub fn material(scene: &mut Scene, landscape: &Material) -> MaterialHandle {
    let cliff = scene.add_texture(TextureDesc::new(TexturePattern::Cliff, Vec3::new(0.33, 0.3, 0.26), Vec3::new(0.15, 0.14, 0.13)).roughness(0.78, 0.92).bump(1.5).seed(31));
    let mut layers = landscape.layers;
    layers[2] = Some(cliff);
    let material = scene.add_material(Material { kind: MaterialKind::Terrain, mapping: TextureMapping::WorldTriplanar, layers, ..*landscape });
    scene.set_material_name(material, "mesh terrain");
    material
}

/// Whether a valley boulder at (x, z) with this footprint would sit in the reshaped ground (the
/// cliff, the arch, the river), where the landscape height it was placed by no longer holds.
pub fn displaces(terrain: &MeshTerrain, x: f32, z: f32, footprint: f32) -> bool {
    if !in_hole(x, z) {
        return false;
    }
    let p = Vec3::new(x, (terrain.base)(x, z), z);
    terrain.volume_bounds().iter().any(|b| p.x > b.min.x - footprint && p.x < b.max.x + footprint && p.z > b.min.z - footprint && p.z < b.max.z + footprint)
        || river_distance(Vec2::new(x, z)).0 < RIVER_WIDTH * 0.5 + RIVER_BANK + footprint
}

/// The river's simulation grid: from inside the cave to the lake.
const RIVER_ORIGIN: Vec2 = Vec2::new(-19.5, -51.0);
const RIVER_CELL: f32 = 0.25;
const RIVER_CELLS: [u32; 2] = [72, 264];
/// The lake the river runs into (valley.rs): its level and its sheet's rectangle.
const LAKE_LEVEL: f32 = -2.2;
const LAKE_RECT: (Vec2, Vec2) = (Vec2::new(-23.2, 3.8), Vec2::new(13.2, 40.2));
/// Boulders standing in the stream: (x, z, size).
const STREAM_BOULDERS: [(f32, f32, f32); 4] = [(-8.9, -28.0, 0.75), (-7.2, -17.5, 0.6), (-8.4, -7.0, 0.8), (-9.4, 1.5, 0.55)];

/// The river: a shallow-water simulation over the channel the mesh terrain carved, its bed
/// read from the terrain's density field (the cave's floor under its roof, rock walls as high
/// cells), boulders in the stream raising the bed, a spring in the cave and the lake as outlet.
pub fn add_river(scene: &mut Scene, terrain: &MeshTerrain, ground: fn(f32, f32) -> f32, rocks: &[MeshHandle], rock_materials: &[MaterialHandle]) {
    let start = web_time::Instant::now();
    let volumes = terrain.volume_bounds();
    let bed_at = |p: Vec2| {
        let h = terrain.height(p.x, p.y);
        if !volumes.iter().any(|b| p.x > b.min.x && p.x < b.max.x && p.y > b.min.z && p.y < b.max.z) {
            return h;
        }
        // Under rock: the floor below the lowest of a few points above the heightfield's ground
        // that lies in the open (the cave's floor under its roof, the ground under the arch), or
        // a wall where all of them are inside the rock.
        [1.0, 2.0, 3.0, 4.0].into_iter().find_map(|above| terrain.surface_below(Vec3::new(p.x, h + above, p.y), above + 3.0, 0.1)).unwrap_or(h + 20.0)
    };
    let mut river = helios_render::River::new(RIVER_ORIGIN, RIVER_CELL, RIVER_CELLS, bed_at);
    // Under walls the dry surface hides at the heightfield's ground, inside the rock; elsewhere
    // just under the bed it lies on - in the cave that is the cave's floor, not the heightfield
    // ground far below it (hiding there made the surface's dry corners plunge through the floor
    // and stand as sheets in front of the step at the cave's mouth).
    river.hide = (0..river.bed.len()).map(|i| {
        let c = river.center(i);
        let h = terrain.height(c.x, c.y);
        if river.bed[i] > h + 10.0 { h } else { river.bed[i] }
    }).collect();
    // Boulders in the stream: instances, and their tops in the bed.
    for (i, &(x, z, size)) in STREAM_BOULDERS.iter().enumerate() {
        let (_, path_bed) = river_distance(Vec2::new(x, z));
        let radii = Vec3::new(1.2, 0.9, 1.0) * size;
        let center = Vec3::new(x, path_bed + radii.y * 0.45, z);
        let rotation = glam::Quat::from_rotation_y(i as f32 * 2.3);
        scene.add_instance(rocks[i % rocks.len()], rock_materials[i % rock_materials.len()], Mat4::from_scale_rotation_translation(radii, rotation, center));
        for (cell, bed) in river.bed.iter_mut().enumerate() {
            let c = RIVER_ORIGIN + (Vec2::new((cell as u32 % RIVER_CELLS[0]) as f32, (cell as u32 / RIVER_CELLS[0]) as f32) + 0.5) * RIVER_CELL;
            let local = rotation.inverse() * Vec3::new(c.x - x, 0.0, c.y - z);
            let r = (local.x / (radii.x * 0.85)).powi(2) + (local.z / (radii.z * 0.85)).powi(2);
            if r < 1.0 {
                *bed = bed.max(center.y + radii.y * 0.8 * (1.0 - r).sqrt());
            }
        }
    }
    // Start full: the channel's design water surface (half a metre over its bed), and the lake.
    for i in 0..river.bed.len() {
        let c = river.center(i);
        let in_lake = c.x > LAKE_RECT.0.x && c.x < LAKE_RECT.1.x && c.y > LAKE_RECT.0.y && c.y < LAKE_RECT.1.y && ground(c.x, c.y) < LAKE_LEVEL;
        river.outlet[i] = in_lake;
        let (d, path_bed) = river_distance(c);
        let target = if in_lake { LAKE_LEVEL } else if d < RIVER_WIDTH * 0.5 + 0.5 { path_bed + 0.45 } else { f32::MIN };
        river.depth[i] = (target - river.bed[i]).max(0.0);
    }
    river.outlet_level = LAKE_LEVEL;
    river.sources.push(helios_render::RiverSource { position: Vec2::new(RIVER_PATH[0].x, RIVER_PATH[0].z), radius: 1.0, rate: 0.9 });
    let surface = river.surface_mesh();
    let mesh = add_baked_with(scene, "river surface", &surface, BuildConfig { sdf: SdfMode::None, ..Default::default() });
    let water = scene.add_material(Material { kind: MaterialKind::Water, base_color: Vec3::ONE, roughness: 0.03, ..Default::default() });
    scene.set_material_name(water, "river water");
    let instance = scene.add_instance(mesh, water, Mat4::IDENTITY);
    scene.set_cast_shadow(instance, false);
    river.surface = Some(instance);
    scene.rivers.push(river);
    log::info!("river: bed and surface made in {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
}
