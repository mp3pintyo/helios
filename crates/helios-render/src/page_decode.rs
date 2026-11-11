//! Decoding of entropy-coded page payloads (`.hvm` version 7, `helios_geometry::page_codec`) off
//! the render thread.
//!
//! The streamer hands every completed read of coded pages to a [`Decoders`] as one
//! [`DecodeJob`] and collects the payload words a later frame:
//!
//! * **natively** a few decode threads take the jobs (the I/O threads keep reading meanwhile);
//! * **in the browser** the bake workers do (`lumen_offload.rs`: the page's pool of Web Workers,
//!   each running a second instance of the wasm module), as [`run_decode_task`]. Without workers
//!   (`?bake-workers=0`, or none could start) or when one fails, the render thread decodes, but
//!   only for a couple of milliseconds per frame, so a burst of arrivals never stalls a frame;
//! * **inline** (tests): decoded at once, ready in the same update.

use std::ops::Range;
use std::sync::Arc;

use helios_geometry::page_codec::decode_page;

/// The stored pages of one completed read: global page ids and their bytes in `buffer`.
pub(crate) struct DecodeJob {
    pub ticket: u64,
    pub buffer: Arc<Vec<u8>>,
    pub pages: Vec<(u32, Range<usize>)>,
}

/// A decoded [`DecodeJob`]: every page's payload words, or why it did not decode.
pub(crate) struct Decoded {
    pub ticket: u64,
    pub pages: Vec<(u32, Result<Vec<u32>, String>)>,
}

impl DecodeJob {
    fn decode(&self) -> Decoded {
        Decoded { ticket: self.ticket, pages: self.pages.iter().map(|(p, range)| (*p, decode_page(&self.buffer[range.clone()]))).collect() }
    }
}

/// Where decode jobs run.
pub(crate) enum Decoders {
    /// Decoded in [`Decoders::submit`], returned by the next poll.
    #[cfg_attr(not(test), allow(dead_code))]
    Inline(Vec<Decoded>),
    #[cfg(not(target_arch = "wasm32"))]
    Threads(Option<threads::Pool>),
    #[cfg(target_arch = "wasm32")]
    Workers(workers::Queue),
}

impl Default for Decoders {
    /// Decode threads natively (started with the first job), the bake workers in the browser.
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        return Decoders::Threads(None);
        #[cfg(target_arch = "wasm32")]
        return Decoders::Workers(workers::Queue::default());
    }
}

impl Decoders {
    pub fn submit(&mut self, job: DecodeJob) {
        match self {
            Decoders::Inline(done) => done.push(job.decode()),
            #[cfg(not(target_arch = "wasm32"))]
            Decoders::Threads(pool) => pool.get_or_insert_with(threads::Pool::new).submit(job),
            #[cfg(target_arch = "wasm32")]
            Decoders::Workers(queue) => queue.submit(job),
        }
    }

    /// Appends the jobs decoded since the previous call.
    pub fn poll(&mut self, done: &mut Vec<Decoded>) {
        match self {
            Decoders::Inline(ready) => done.append(ready),
            #[cfg(not(target_arch = "wasm32"))]
            Decoders::Threads(pool) => {
                if let Some(pool) = pool {
                    pool.poll(done);
                }
            }
            #[cfg(target_arch = "wasm32")]
            Decoders::Workers(queue) => queue.poll(done),
        }
    }
}

/// The first word of a decode task for a bake worker (the scene bake's start with `HSBT`,
/// Lumen's with 0 or 1).
const TASK_MAGIC: u32 = u32::from_le_bytes(*b"HPDT");
const OUTPUT_MAGIC: u32 = u32::from_le_bytes(*b"HPDO");

/// Whether `input` is a page decode task.
pub fn is_decode_task(input: &[u8]) -> bool {
    input.get(..4).is_some_and(|m| m == TASK_MAGIC.to_le_bytes())
}

/// A task for a worker: the stored pages of `job`, each as its length and bytes.
#[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
fn encode_task(job: &DecodeJob) -> Vec<u8> {
    let bytes: usize = job.pages.iter().map(|(_, r)| r.len() + 4).sum();
    let mut out = Vec::with_capacity(8 + bytes);
    out.extend_from_slice(&TASK_MAGIC.to_le_bytes());
    out.extend_from_slice(&(job.pages.len() as u32).to_le_bytes());
    for (_, range) in &job.pages {
        out.extend_from_slice(&(range.len() as u32).to_le_bytes());
        out.extend_from_slice(&job.buffer[range.clone()]);
    }
    out
}

/// A worker's entry for a decode task: every page's payload words (a page that does not decode
/// is returned empty, which fails it). Empty output for input that is not a decode task.
pub fn run_decode_task(input: &[u8]) -> Vec<u8> {
    let mut r = Reader(input);
    let Some(count) = r.u32().filter(|&m| m == TASK_MAGIC).and_then(|_| r.u32()) else { return Vec::new() };
    let mut out = Vec::new();
    out.extend_from_slice(&OUTPUT_MAGIC.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    for _ in 0..count {
        let Some(page) = r.u32().and_then(|len| r.take(len as usize)) else { return Vec::new() };
        let words = decode_page(page).unwrap_or_default();
        out.extend_from_slice(&(words.len() as u32).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&words));
    }
    out
}

/// The payload words of a worker's output for `job`; `None` if it does not fit the job.
#[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
fn decode_output(job: &DecodeJob, output: &[u8]) -> Option<Decoded> {
    let mut r = Reader(output);
    if r.u32()? != OUTPUT_MAGIC || r.u32()? as usize != job.pages.len() {
        return None;
    }
    let mut pages = Vec::with_capacity(job.pages.len());
    for (p, _) in &job.pages {
        let words = r.u32()? as usize;
        let bytes = r.take(words * 4)?;
        let result = if words == 0 { Err("the page did not decode".to_string()) } else { Ok(bytemuck::pod_collect_to_vec(bytes)) };
        pages.push((*p, result));
    }
    r.0.is_empty().then_some(Decoded { ticket: job.ticket, pages })
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        (self.0.len() >= n).then(|| {
            let (head, tail) = self.0.split_at(n);
            self.0 = tail;
            head
        })
    }

    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod threads {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::{Arc, Mutex};

    use super::{DecodeJob, Decoded};

    /// Decode threads: as many as leave the render thread and the I/O threads a core, at most 4.
    pub struct Pool {
        jobs: Sender<DecodeJob>,
        done: Mutex<Receiver<Decoded>>,
        /// Jobs decoded here because no thread could start.
        here: Vec<Decoded>,
    }

    impl Pool {
        pub fn new() -> Self {
            let threads = std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(2).clamp(1, 4);
            let (jobs, queue) = channel::<DecodeJob>();
            let (finished, done) = channel();
            let queue = Arc::new(Mutex::new(queue));
            for i in 0..threads {
                let (queue, finished) = (queue.clone(), finished.clone());
                let spawned = std::thread::Builder::new().name(format!("helios-decode-{i}")).spawn(move || loop {
                    // The lock is only held while waiting for the next job.
                    let Ok(job) = queue.lock().map_err(drop).and_then(|q| q.recv().map_err(drop)) else { return };
                    if finished.send(job.decode()).is_err() {
                        return;
                    }
                });
                if spawned.is_err() {
                    log::warn!("could not start a page decode thread");
                }
            }
            Self { jobs, done: Mutex::new(done), here: Vec::new() }
        }

        pub fn submit(&mut self, job: DecodeJob) {
            // Fails only if no thread could start: then the job is decoded here.
            if let Err(job) = self.jobs.send(job) {
                self.here.push(job.0.decode());
            }
        }

        pub fn poll(&mut self, done: &mut Vec<Decoded>) {
            done.append(&mut self.here);
            if let Ok(receiver) = self.done.lock() {
                done.extend(receiver.try_iter());
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod workers {
    use std::collections::VecDeque;

    use web_time::{Duration, Instant};

    use super::{decode_output, encode_task, DecodeJob, Decoded};
    use crate::lumen_offload::{submit_bytes, BakePoll, Offloaded};

    /// Render-thread decoding per frame where no worker takes the jobs.
    const FRAME_BUDGET: Duration = Duration::from_millis(2);

    #[derive(Default)]
    pub struct Queue {
        running: Vec<(Offloaded, DecodeJob)>,
        /// Jobs decoded on the render thread, with the pages done so far.
        local: VecDeque<(DecodeJob, Vec<(u32, Result<Vec<u32>, String>)>)>,
    }

    impl Queue {
        pub fn submit(&mut self, job: DecodeJob) {
            match submit_bytes(&encode_task(&job)) {
                Some(task) => self.running.push((task, job)),
                None => self.local.push_back((job, Vec::new())),
            }
        }

        pub fn poll(&mut self, done: &mut Vec<Decoded>) {
            let mut i = 0;
            while i < self.running.len() {
                match self.running[i].0.poll() {
                    BakePoll::Pending => i += 1,
                    BakePoll::Done(output) => {
                        let (_, job) = self.running.swap_remove(i);
                        match decode_output(&job, &output) {
                            Some(decoded) => done.push(decoded),
                            None => self.local.push_back((job, Vec::new())),
                        }
                    }
                    BakePoll::Failed => {
                        let (_, job) = self.running.swap_remove(i);
                        self.local.push_back((job, Vec::new()));
                    }
                }
            }
            // Render-thread fallback: at least one page per frame, then while within budget.
            let start = Instant::now();
            while let Some((job, pages)) = self.local.front_mut() {
                let (p, range) = job.pages[pages.len()].clone();
                pages.push((p, helios_geometry::page_codec::decode_page(&job.buffer[range])));
                if pages.len() == job.pages.len() {
                    let (job, pages) = self.local.pop_front().expect("front exists");
                    done.push(Decoded { ticket: job.ticket, pages });
                }
                if start.elapsed() >= FRAME_BUDGET {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helios_geometry::page_codec::encode_page;
    use helios_geometry::{build_virtual_mesh, primitives, BuildConfig, SdfMode};

    #[test]
    fn worker_tasks_round_trip() {
        let vm = build_virtual_mesh(&primitives::rock(2, 4), &BuildConfig { sdf: SdfMode::None, ..Default::default() });
        let mut buffer = Vec::new();
        let mut pages = Vec::new();
        for p in 0..vm.pages.len() {
            let start = buffer.len();
            buffer.extend_from_slice(&encode_page(vm.page_payload(p), vm.page_clusters(p)));
            pages.push((p as u32 + 10, start..buffer.len()));
        }
        let job = DecodeJob { ticket: 7, buffer: Arc::new(buffer), pages };
        let task = encode_task(&job);
        assert!(is_decode_task(&task));
        let decoded = decode_output(&job, &run_decode_task(&task)).expect("the output fits the job");
        assert_eq!(decoded.ticket, 7);
        for (p, words) in decoded.pages {
            assert_eq!(words.unwrap(), vm.page_payload(p as usize - 10));
        }
        assert!(run_decode_task(b"HSBT....").is_empty());

        // Threads decode the same.
        let mut decoders = Decoders::default();
        decoders.submit(DecodeJob { ticket: 8, buffer: job.buffer.clone(), pages: job.pages.clone() });
        let mut done = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while done.is_empty() && std::time::Instant::now() < deadline {
            decoders.poll(&mut done);
            std::thread::yield_now();
        }
        assert_eq!(done.len(), 1);
        assert!(done[0].pages.iter().all(|(p, w)| w.as_deref() == Ok(vm.page_payload(*p as usize - 10))));
    }
}
