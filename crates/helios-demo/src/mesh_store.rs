//! Pre-baked, streamed scene geometry.
//!
//! `helios --bake-hvm DIR` bakes a scene and writes each virtual mesh as a `.hvm` file plus a
//! [`Manifest`] — with entropy-coded pages (`.hvm` version 7, `helios_geometry::page_codec`: page
//! payloads 2.2–2.7 times smaller on disk and over the wire, decoded on load), or with
//! `--hvm-raw` as the GPU format (version 5). `helios --stream-from DIR` (native) and `?stream=URL` (browser) then build the
//! *same* scene — materials, instances, lights, camera all come from the procedural builder — but
//! substitute every baked mesh with its pre-baked counterpart, keeping only its header (records,
//! dependencies, bounds, distance field) and root pages in memory. All other page payloads are
//! read on demand by the streamer: from disk through a [`helios_render::page_source::FileSource`],
//! over HTTP range requests through a `HttpSource`.
//!
//! The scene builders stay unaware of this: they add meshes through `scenes::add_baked`, which
//! asks the [`MeshStore`] made active for the duration of `scenes::build_with`. Meshes are matched
//! by build order and name; a mesh missing from the library (the builder changed since baking) is
//! simply baked.
//!
//! The same seam spreads the browser's scene bake over Web Workers (`scene_bake.rs`): each
//! worker builds the scene with a [`Shard`] store, which bakes only the meshes assigned to it
//! (and adds empty stand-ins for the others), and the page builds it once more with a [`Baked`]
//! store holding every worker's meshes.
//!
//! Imported glTF models (`SceneKind::Model`) are not part of this: their meshes are baked in
//! parallel when the model loads and added directly, and a library names only a procedural scene
//! and detail, not a model file. Their source is a file the user already has to load, so streaming
//! them would need a model-keyed cache — left for when models are baked offline.

use std::cell::RefCell;
use std::sync::Arc;

use helios_render::geometry::VirtualMesh;
use helios_render::{MeshHandle, PageSource, Scene};

use crate::scenes::{Detail, SceneKind};

/// File name of the manifest inside a mesh library.
pub const MANIFEST: &str = "manifest.txt";
const MANIFEST_MAGIC: &str = "helios-hvm-manifest 1";

/// Lists the meshes of a baked scene in build order.
///
/// ```text
/// helios-hvm-manifest 1
/// scene forest
/// detail high
/// mesh 000-terrain.hvm 1835180 terrain
/// ```
///
/// Each `mesh` line gives the file, its header length in bytes (so an HTTP client can fetch the
/// header with a single range request) and the mesh name.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    pub scene: SceneKind,
    pub detail: Detail,
    pub meshes: Vec<ManifestEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ManifestEntry {
    pub file: String,
    pub header_bytes: usize,
    pub name: String,
}

fn detail_name(detail: Detail) -> &'static str {
    detail.pick("medium", "high")
}

impl Manifest {
    pub fn to_text(&self) -> String {
        let mut text = format!("{MANIFEST_MAGIC}\nscene {}\ndetail {}\n", self.scene.name(), detail_name(self.detail));
        for m in &self.meshes {
            text.push_str(&format!("mesh {} {} {}\n", m.file, m.header_bytes, m.name));
        }
        text
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some(MANIFEST_MAGIC) {
            return Err(format!("not a mesh library manifest (expected '{MANIFEST_MAGIC}')"));
        }
        let mut manifest = Manifest { scene: SceneKind::Valley, detail: Detail::High, meshes: Vec::new() };
        for line in lines.map(str::trim).filter(|l| !l.is_empty()) {
            let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
            match key {
                "scene" => {
                    manifest.scene = SceneKind::parse(rest).filter(|&k| k != SceneKind::Model).ok_or_else(|| format!("unknown procedural scene '{rest}'"))?
                }
                "detail" => manifest.detail = if rest == "medium" { Detail::Medium } else { Detail::High },
                "mesh" => {
                    let mut fields = rest.splitn(3, ' ');
                    let (Some(file), Some(header), Some(name)) = (fields.next(), fields.next(), fields.next()) else {
                        return Err(format!("bad manifest line '{line}'"));
                    };
                    let header_bytes = header.parse().map_err(|_| format!("bad header length in '{line}'"))?;
                    manifest.meshes.push(ManifestEntry { file: file.into(), header_bytes, name: name.into() });
                }
                _ => return Err(format!("bad manifest line '{line}'")),
            }
        }
        Ok(manifest)
    }
}

/// File name of the `index`-th mesh: `007-nave-wall-bay.hvm`.
#[cfg(any(test, not(target_arch = "wasm32")))]
fn file_name(index: usize, name: &str) -> String {
    let slug: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    format!("{index:03}-{slug}.hvm")
}

/// Pre-baked meshes (headers and root pages) plus the source their other pages stream from.
pub struct Library {
    pub manifest: Manifest,
    meshes: Vec<Option<VirtualMesh>>,
    source: Arc<dyn PageSource>,
    next: usize,
}

impl Library {
    pub fn new(manifest: Manifest, meshes: Vec<VirtualMesh>, source: Arc<dyn PageSource>) -> Self {
        Self { manifest, meshes: meshes.into_iter().map(Some).collect(), source, next: 0 }
    }

    /// CPU bytes held by the library's meshes.
    pub fn cpu_bytes(&self) -> usize {
        self.meshes.iter().flatten().map(VirtualMesh::cpu_bytes).sum()
    }
}

/// Where `scenes::add_baked` gets its meshes from.
pub enum MeshStore {
    /// Bake every mesh (the default).
    Bake,
    /// Bake every mesh and write it to a directory.
    #[cfg(not(target_arch = "wasm32"))]
    Record(native::Recorder),
    /// Take pre-baked, streamed meshes from a library.
    Stream(Library),
    /// Bake only this shard's share of the meshes (a scene-bake worker).
    Shard(Shard),
    /// Take meshes baked elsewhere (by the scene-bake workers); bake the ones missing.
    Baked(Baked),
}

/// One worker's share of a scene bake. Every worker runs the same builder and sees the same
/// meshes in the same order, so each can assign them alike without talking to the others: a
/// mesh goes to the least loaded worker so far (its source triangles are the load; ties to the
/// lowest index), a [`unit`] of meshes to one worker. Meshes of other workers become empty
/// stand-ins, which the builder never reads (but within a unit).
pub struct Shard {
    index: usize,
    loads: Vec<u64>,
    next: u32,
    /// The worker the open unit's meshes go to (`None` until its first mesh), and its depth.
    unit: (Option<usize>, u32),
    /// This shard's meshes: build index, name and the baked mesh.
    pub baked: Vec<(u32, String, VirtualMesh)>,
}

impl Shard {
    /// Worker `index` of `count`.
    pub fn new(index: u32, count: u32) -> Self {
        let count = count.max(1) as usize;
        Self { index: (index as usize).min(count - 1), loads: vec![0; count], next: 0, unit: (None, 0), baked: Vec::new() }
    }

    /// A store that bakes nothing: the scene with empty stand-ins for all its meshes (its
    /// materials, textures and instances are real).
    pub fn stand_in() -> Self {
        Self { index: usize::MAX, ..Self::new(0, 1) }
    }

    /// The worker mesh `cost` goes to (and charges it).
    fn assign(&mut self, cost: u64) -> usize {
        let least = |loads: &[u64]| (0..loads.len()).min_by_key(|&w| (loads[w], w)).unwrap_or(0);
        let worker = match self.unit {
            (Some(worker), _) => worker,
            (None, 0) => least(&self.loads),
            (None, depth) => {
                let worker = least(&self.loads);
                self.unit = (Some(worker), depth);
                worker
            }
        };
        // A mesh costs at least something, so the empty ones spread too.
        self.loads[worker] += cost.max(1);
        worker
    }
}

/// Meshes baked elsewhere, by build index (with the name they were built under).
pub struct Baked {
    meshes: Vec<Option<(String, VirtualMesh)>>,
    next: usize,
    /// Meshes that had to be baked here (missing, or under another name).
    pub baked_here: u32,
}

impl Baked {
    pub fn new(meshes: impl IntoIterator<Item = (u32, String, VirtualMesh)>) -> Self {
        let mut slots: Vec<Option<(String, VirtualMesh)>> = Vec::new();
        for (index, name, mesh) in meshes {
            let index = index as usize;
            if slots.len() <= index {
                slots.resize_with(index + 1, || None);
            }
            slots[index] = Some((name, mesh));
        }
        Self { meshes: slots, next: 0, baked_here: 0 }
    }
}

/// Runs `build` with the meshes it adds kept together: a scene-bake worker bakes all or none of
/// them (for a mesh whose bake reads meshes added before it, like an assembly's base).
pub(crate) fn unit<R>(build: impl FnOnce() -> R) -> R {
    let enter = |delta: i32| {
        ACTIVE.with(|active| {
            if let MeshStore::Shard(shard) = &mut *active.borrow_mut() {
                let depth = shard.unit.1.saturating_add_signed(delta);
                shard.unit = (if depth == 0 { None } else { shard.unit.0 }, depth);
            }
        })
    };
    enter(1);
    let result = build();
    enter(-1);
    result
}

thread_local! {
    static ACTIVE: RefCell<MeshStore> = const { RefCell::new(MeshStore::Bake) };
}

/// Runs `build` with `store` active; returns its result and the store.
pub(crate) fn scoped<R>(store: MeshStore, build: impl FnOnce() -> R) -> (R, MeshStore) {
    let previous = ACTIVE.with(|active| active.replace(store));
    let result = build();
    (result, ACTIVE.with(|active| active.replace(previous)))
}

/// Adds the mesh called `name` to `scene`, from the active store; `bake` bakes it. `cost`
/// estimates the bake (source triangles) for spreading a scene's meshes over workers.
pub(crate) fn add(scene: &mut Scene, name: &str, cost: u64, bake: impl FnOnce() -> VirtualMesh) -> MeshHandle {
    let handle = add_unnamed(scene, name, cost, bake);
    crate::scenes::name_mesh(scene, handle, name);
    handle
}

fn add_unnamed(scene: &mut Scene, name: &str, cost: u64, bake: impl FnOnce() -> VirtualMesh) -> MeshHandle {
    ACTIVE.with(|active| match &mut *active.borrow_mut() {
        MeshStore::Bake => scene.add_mesh(bake()),
        #[cfg(not(target_arch = "wasm32"))]
        MeshStore::Record(recorder) => recorder.add(scene, name, bake()),
        MeshStore::Stream(library) => {
            let index = library.next;
            library.next += 1;
            let entry = library.manifest.meshes.get(index).filter(|e| e.name == name);
            match entry.and(library.meshes.get_mut(index).and_then(Option::take)) {
                Some(mesh) => scene.add_streamed_mesh(mesh, library.source.clone(), index as u32),
                None => {
                    log::warn!("mesh #{index} '{name}' is not in the mesh library (scene changed since it was baked?); baking it");
                    scene.add_mesh(bake())
                }
            }
        }
        MeshStore::Shard(shard) => {
            let index = shard.next;
            shard.next += 1;
            if shard.assign(cost) != shard.index {
                return scene.add_mesh(VirtualMesh::default());
            }
            let mesh = bake();
            // Within a unit a later bake may read this mesh from the scene: it gets a copy.
            let copy = if shard.unit.1 > 0 { mesh.clone() } else { VirtualMesh::default() };
            shard.baked.push((index, name.to_string(), mesh));
            scene.add_mesh(copy)
        }
        MeshStore::Baked(baked) => {
            let index = baked.next;
            baked.next += 1;
            match baked.meshes.get_mut(index).and_then(Option::take) {
                Some((baked_name, mesh)) if baked_name == name => scene.add_mesh(mesh),
                other => {
                    if other.is_some() {
                        log::warn!("baked mesh #{index} is not '{name}' (the builders differ?); baking it here");
                    }
                    baked.baked_here += 1;
                    scene.add_mesh(bake())
                }
            }
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{bake_to, open};

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use helios_render::geometry::VirtualMesh;
    use helios_render::page_source::FileSource;
    use helios_render::{MeshHandle, Scene};
    use web_time::Instant;

    use super::{file_name, Library, Manifest, ManifestEntry, MeshStore, MANIFEST};
    use crate::scenes::{self, Detail, SceneKind};

    /// Writes baked meshes into a directory.
    pub struct Recorder {
        dir: PathBuf,
        manifest: Manifest,
        /// Entropy-code the pages (version 7).
        coded: bool,
        bytes: usize,
        /// Bytes of the page payloads as the GPU holds them.
        payload: usize,
        error: Option<String>,
    }

    impl Recorder {
        pub(super) fn add(&mut self, scene: &mut Scene, name: &str, mesh: VirtualMesh) -> MeshHandle {
            let file = file_name(self.manifest.meshes.len(), name);
            let bytes = if self.coded { mesh.to_coded_bytes() } else { mesh.to_bytes() };
            if let Err(e) = std::fs::write(self.dir.join(&file), &bytes) {
                self.error.get_or_insert(format!("writing {file}: {e}"));
            }
            self.bytes += bytes.len();
            self.payload += mesh.payload_bytes();
            let header_bytes = VirtualMesh::header_len(&bytes).expect("a .hvm just written");
            self.manifest.meshes.push(ManifestEntry { file, header_bytes, name: name.into() });
            scene.add_mesh(mesh)
        }
    }

    /// Bakes scene `kind` and writes its meshes as `.hvm` files plus a manifest into `dir`, with
    /// entropy-coded pages if `coded`.
    pub fn bake_to(dir: &Path, kind: SceneKind, detail: Detail, coded: bool) -> Result<(), String> {
        if kind == SceneKind::Model {
            return Err("only the procedural scenes can be baked to a mesh library (imported models bake at load)".into());
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
        let recorder = Recorder { dir: dir.into(), manifest: Manifest { scene: kind, detail, meshes: Vec::new() }, coded, bytes: 0, payload: 0, error: None };
        let (_, store) = scenes::build_with(kind, detail, None, MeshStore::Record(recorder));
        let MeshStore::Record(recorder) = store else { unreachable!("the store is handed back unchanged") };
        if let Some(error) = recorder.error {
            return Err(error);
        }
        std::fs::write(dir.join(MANIFEST), recorder.manifest.to_text()).map_err(|e| format!("writing the manifest: {e}"))?;
        log::info!(
            "wrote {} meshes ({:.1} MiB; page payloads {:.1} MiB in the GPU format) to {}",
            recorder.manifest.meshes.len(),
            recorder.bytes as f64 / 1048576.0,
            recorder.payload as f64 / 1048576.0,
            dir.display()
        );
        Ok(())
    }

    /// Opens the mesh library in `dir`: reads every header and root tail now, and streams the
    /// other pages through a [`FileSource`] (returned too, for its I/O statistics). Every read
    /// takes at least `latency`, and with a `bandwidth` (bytes per second) the reads share a link
    /// that fast (to emulate slow storage or a network).
    pub fn open(dir: &Path, latency: Duration, bandwidth: Option<f64>) -> Result<(Library, Arc<FileSource>), String> {
        let start = Instant::now();
        let text = std::fs::read_to_string(dir.join(MANIFEST)).map_err(|e| format!("reading {}: {e}", dir.join(MANIFEST).display()))?;
        let manifest = Manifest::parse(&text)?;
        let mut meshes = Vec::new();
        let mut paths = Vec::new();
        let mut loaded = 0usize;
        for entry in &manifest.meshes {
            let path = dir.join(&entry.file);
            let (mesh, bytes) = load_header_and_roots(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            meshes.push(mesh);
            paths.push(path);
            loaded += bytes;
        }
        let source = Arc::new(FileSource::with_link(&paths, 2, latency, bandwidth).map_err(|e| format!("opening the mesh files: {e}"))?);
        let library = Library::new(manifest, meshes, source.clone());
        log::info!(
            "mesh library {}: {} meshes, read {:.1} MiB of headers and root pages in {:.0} ms ({:.1} MiB in RAM)",
            dir.display(),
            paths.len(),
            loaded as f64 / 1048576.0,
            start.elapsed().as_secs_f64() * 1000.0,
            library.cpu_bytes() as f64 / 1048576.0
        );
        Ok((library, source))
    }

    /// Reads a `.hvm`'s header and root pages; returns the mesh and the bytes read.
    fn load_header_and_roots(path: &Path) -> Result<(VirtualMesh, usize), String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut header = vec![0u8; VirtualMesh::PREFIX_BYTES];
        file.read_exact(&mut header).map_err(|e| e.to_string())?;
        header.resize(VirtualMesh::header_len(&header)?, 0);
        file.read_exact(&mut header[VirtualMesh::PREFIX_BYTES..]).map_err(|e| e.to_string())?;
        let mut mesh = VirtualMesh::from_header_bytes(&header)?;
        let tail = mesh.root_tail_start();
        let range = mesh.tail_byte_range(tail);
        let mut roots = vec![0u8; (range.end - range.start) as usize];
        file.seek(SeekFrom::Start(range.start)).and_then(|_| file.read_exact(&mut roots)).map_err(|e| e.to_string())?;
        mesh.set_tail_payload(tail, &roots)?;
        Ok((mesh, header.len() + roots.len()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use helios_render::geometry::{build_virtual_mesh, primitives, BuildConfig, SdfMode};

        #[test]
        fn header_and_roots_load_from_a_file() {
            let vm = build_virtual_mesh(&primitives::rock(2, 4), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
            let dir = std::env::temp_dir().join(format!("helios-mesh-store-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("rock.hvm");
            for coded in [false, true] {
                std::fs::write(&path, if coded { vm.to_coded_bytes() } else { vm.to_bytes() }).unwrap();
                let (mesh, bytes) = load_header_and_roots(&path).unwrap();
                assert_eq!(mesh.clusters, vm.clusters);
                assert_eq!(mesh.is_coded(), coded);
                assert!(bytes < vm.header_bytes() + vm.payload_bytes());
                for p in mesh.root_tail_start()..vm.pages.len() {
                    assert_eq!(mesh.page_payload(p), vm.page_payload(p));
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::fetch_library;

#[cfg(target_arch = "wasm32")]
mod web {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::Poll;

    use helios_render::geometry::VirtualMesh;
    use helios_render::page_source::{fetch, Fetched, HttpSource};

    use super::{Library, Manifest, MANIFEST};

    /// Awaits all `futures` concurrently (the fetches inside run in parallel).
    async fn join_all<T>(futures: Vec<Pin<Box<dyn Future<Output = T>>>>) -> Vec<T> {
        let mut futures = futures;
        let mut results: Vec<Option<T>> = futures.iter().map(|_| None).collect();
        std::future::poll_fn(|cx| {
            let mut pending = false;
            for (future, result) in futures.iter_mut().zip(results.iter_mut()) {
                if result.is_none() {
                    match future.as_mut().poll(cx) {
                        Poll::Ready(value) => *result = Some(value),
                        Poll::Pending => pending = true,
                    }
                }
            }
            if pending {
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        })
        .await;
        results.into_iter().flatten().collect()
    }

    /// Fetches the header and root pages of `url` with two range requests. A server that
    /// ignores `Range` sends the whole file instead, which then simply stays in memory.
    async fn fetch_mesh(url: String, header_bytes: usize) -> Result<VirtualMesh, String> {
        let header = match fetch(&url, Some(0..header_bytes as u64)).await? {
            Fetched::Range(header) => header,
            Fetched::Whole(file) => return VirtualMesh::from_bytes(&file).map_err(|e| format!("{url}: {e}")),
        };
        let mut mesh = VirtualMesh::from_header_bytes(&header).map_err(|e| format!("{url}: {e}"))?;
        let tail = mesh.root_tail_start();
        let range = mesh.tail_byte_range(tail);
        match fetch(&url, Some(range.clone())).await? {
            Fetched::Range(roots) => mesh.set_tail_payload(tail, &roots).map_err(|e| format!("{url}: {e}"))?,
            Fetched::Whole(file) => return VirtualMesh::from_bytes(&file).map_err(|e| format!("{url}: {e}")),
        }
        Ok(mesh)
    }

    /// Loads the mesh library at `base` (a URL prefix holding the manifest and `.hvm` files).
    pub async fn fetch_library(base: &str) -> Result<Library, String> {
        let base = base.trim_end_matches('/');
        let manifest = match fetch(&format!("{base}/{MANIFEST}"), None).await? {
            Fetched::Range(bytes) | Fetched::Whole(bytes) => String::from_utf8(bytes).map_err(|_| "manifest is not UTF-8".to_string())?,
        };
        let manifest = Manifest::parse(&manifest)?;
        let urls: Vec<String> = manifest.meshes.iter().map(|m| format!("{base}/{}", m.file)).collect();
        let fetches = urls
            .iter()
            .zip(&manifest.meshes)
            .map(|(url, entry)| Box::pin(fetch_mesh(url.clone(), entry.header_bytes)) as Pin<Box<dyn Future<Output = _>>>)
            .collect();
        let meshes = join_all(fetches).await.into_iter().collect::<Result<Vec<_>, _>>()?;
        log::info!("mesh library {base}: {} meshes", meshes.len());
        // Scenes hold sources in an `Arc` for native thread safety; the browser has one thread.
        #[allow(clippy::arc_with_non_send_sync)]
        let source = Arc::new(HttpSource::new(urls));
        Ok(Library::new(manifest, meshes, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips() {
        let manifest = Manifest {
            scene: SceneKind::Cathedral,
            detail: Detail::Medium,
            meshes: vec![
                ManifestEntry { file: file_name(0, "terrain"), header_bytes: 1234, name: "terrain".into() },
                ManifestEntry { file: file_name(1, "nave wall bay"), header_bytes: 99, name: "nave wall bay".into() },
            ],
        };
        assert_eq!(manifest.meshes[1].file, "001-nave-wall-bay.hvm");
        assert_eq!(Manifest::parse(&manifest.to_text()).unwrap(), manifest);
        assert!(Manifest::parse("hello").is_err());
        assert!(Manifest::parse(&format!("{MANIFEST_MAGIC}\nmesh a.hvm x name")).is_err());
    }
}
