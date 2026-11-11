//! Native page source: a small pool of I/O threads doing positioned reads.

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{PageRead, PageSource, ReadCompletion};

/// Cumulative I/O statistics of a [`FileSource`].
#[derive(Clone, Copy, Debug, Default)]
pub struct IoStats {
    pub reads: u64,
    pub failed: u64,
    pub bytes: u64,
    /// Sum over reads of the time from [`PageSource::start`] to the bytes being ready.
    pub total_latency: Duration,
    pub max_latency: Duration,
    /// Sum over reads of the time spent inside the read call itself.
    pub read_time: Duration,
}

impl IoStats {
    pub fn mean_latency(&self) -> Duration {
        self.total_latency / self.reads.max(1) as u32
    }

    /// Bytes per second of read-call time (per I/O thread).
    pub fn throughput(&self) -> f64 {
        self.bytes as f64 / self.read_time.as_secs_f64().max(1e-9)
    }
}

#[derive(Default)]
struct Counters {
    reads: AtomicU64,
    failed: AtomicU64,
    bytes: AtomicU64,
    latency_us: AtomicU64,
    max_latency_us: AtomicU64,
    read_us: AtomicU64,
}

struct Job {
    read: PageRead,
    queued: Instant,
}

/// A completed read and when it may be handed out (later than now with emulated latency).
type Finished = (Instant, ReadCompletion);

/// Completed reads: the channel from the I/O threads plus reads held back until their time.
struct Done {
    receiver: Receiver<Finished>,
    held: Vec<Finished>,
}

/// Reads byte ranges of a fixed set of files on background threads. Completed reads are
/// handed back through a channel, so [`PageSource::poll`] never waits for the disk.
pub struct FileSource {
    jobs: Sender<Job>,
    done: Mutex<Done>,
    counters: Arc<Counters>,
}

impl FileSource {
    /// Opens `paths` (file `i` of a [`PageRead`] is `paths[i]`) and starts `threads` I/O threads.
    pub fn open(paths: &[PathBuf], threads: usize) -> std::io::Result<Self> {
        Self::with_latency(paths, threads, Duration::ZERO)
    }

    /// Like [`FileSource::open`], but no read completes earlier than `latency` after it was
    /// started (independently of the thread count), emulating a slow disk or network to
    /// exercise the streamer's asynchronous paths.
    pub fn with_latency(paths: &[PathBuf], threads: usize, latency: Duration) -> std::io::Result<Self> {
        Self::with_link(paths, threads, latency, None)
    }

    /// Like [`FileSource::with_latency`], and reads also share a link of `bandwidth` bytes per
    /// second (a network connection): each one's bytes cross it after the latency, one read after
    /// another.
    pub fn with_link(paths: &[PathBuf], threads: usize, latency: Duration, bandwidth: Option<f64>) -> std::io::Result<Self> {
        let link = Arc::new(Link { bandwidth: bandwidth.filter(|&b| b > 0.0), free: Mutex::new(Instant::now()) });
        let files: Arc<Vec<File>> = Arc::new(paths.iter().map(File::open).collect::<Result<_, _>>()?);
        let (jobs, job_queue) = channel::<Job>();
        let (finished, receiver) = channel();
        let job_queue = Arc::new(Mutex::new(job_queue));
        let counters = Arc::new(Counters::default());
        for i in 0..threads.max(1) {
            let (files, job_queue, finished, counters, link) = (files.clone(), job_queue.clone(), finished.clone(), counters.clone(), link.clone());
            std::thread::Builder::new()
                .name(format!("helios-io-{i}"))
                .spawn(move || io_thread(&files, &job_queue, &finished, &counters, latency, &link))?;
        }
        Ok(Self { jobs, done: Mutex::new(Done { receiver, held: Vec::new() }), counters })
    }

    pub fn stats(&self) -> IoStats {
        let c = &self.counters;
        let us = |a: &AtomicU64| Duration::from_micros(a.load(Ordering::Relaxed));
        IoStats {
            reads: c.reads.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            bytes: c.bytes.load(Ordering::Relaxed),
            total_latency: us(&c.latency_us),
            max_latency: us(&c.max_latency_us),
            read_time: us(&c.read_us),
        }
    }
}

/// An emulated network link shared by a source's reads.
struct Link {
    /// Bytes per second (`None`: unlimited).
    bandwidth: Option<f64>,
    /// When the link is free again.
    free: Mutex<Instant>,
}

impl Link {
    /// When `bytes` requested at `queued` have crossed the link (after `latency`).
    fn arrival(&self, queued: Instant, latency: Duration, bytes: usize) -> Instant {
        let Some(bandwidth) = self.bandwidth else { return queued + latency };
        let Ok(mut free) = self.free.lock() else { return queued + latency };
        let start = (queued + latency).max(*free);
        *free = start + Duration::from_secs_f64(bytes as f64 / bandwidth);
        *free
    }
}

/// Serves jobs until the [`FileSource`] (the job sender) is dropped.
fn io_thread(files: &[File], job_queue: &Mutex<Receiver<Job>>, finished: &Sender<Finished>, counters: &Counters, latency: Duration, link: &Link) {
    loop {
        // The lock is only held while waiting for the next job, never during a read.
        let Ok(job) = job_queue.lock().map_err(drop).and_then(|queue| queue.recv().map_err(drop)) else { return };
        let start = Instant::now();
        let result = read_range(files, &job.read);
        let now = Instant::now();
        let ready = link.arrival(job.queued, latency, result.as_ref().map_or(0, Vec::len)).max(now);
        let latency_us = (ready - job.queued).as_micros() as u64;
        counters.reads.fetch_add(1, Ordering::Relaxed);
        counters.latency_us.fetch_add(latency_us, Ordering::Relaxed);
        counters.max_latency_us.fetch_max(latency_us, Ordering::Relaxed);
        counters.read_us.fetch_add((now - start).as_micros() as u64, Ordering::Relaxed);
        match &result {
            Ok(bytes) => counters.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed),
            Err(_) => counters.failed.fetch_add(1, Ordering::Relaxed),
        };
        if finished.send((ready, ReadCompletion { ticket: job.read.ticket, result })).is_err() {
            return;
        }
    }
}

fn read_range(files: &[File], read: &PageRead) -> Result<Vec<u8>, String> {
    let file = files.get(read.file as usize).ok_or_else(|| format!("no file {}", read.file))?;
    let mut buffer = vec![0u8; (read.range.end - read.range.start) as usize];
    read_exact_at(file, &mut buffer, read.range.start).map_err(|e| format!("reading file {} at {}: {e}", read.file, read.range.start))?;
    Ok(buffer)
}

/// Positioned read: threads share one handle per file without seeking it.
#[cfg(unix)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buffer, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buffer.len() {
        match std::os::windows::fs::FileExt::seek_read(file, &mut buffer[done..], offset + done as u64)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => done += n,
        }
    }
    Ok(())
}

impl PageSource for FileSource {
    fn start(&self, read: PageRead) {
        // Sending only fails if every I/O thread is gone (they exit on their own only once the
        // source is dropped); the pages then simply stay non-resident.
        let _ = self.jobs.send(Job { read, queued: Instant::now() });
    }

    fn poll(&self, completed: &mut Vec<ReadCompletion>) {
        // Only the render thread polls, so the lock is never contended.
        let Ok(mut done) = self.done.lock() else { return };
        let done = &mut *done;
        done.held.extend(done.receiver.try_iter());
        let now = Instant::now();
        let mut i = 0;
        while i < done.held.len() {
            if done.held[i].0 <= now {
                completed.push(done.held.swap_remove(i).1);
            } else {
                i += 1;
            }
        }
    }

    fn max_in_flight(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ranges_on_background_threads() {
        let dir = std::env::temp_dir().join(format!("helios-file-source-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
        std::fs::write(&path, &data).unwrap();

        let source = FileSource::open(&[path], 2).unwrap();
        let ranges = [0..100u64, 5000..9000, 9990..10_000, 9990..10_001];
        for (ticket, range) in ranges.iter().enumerate() {
            source.start(PageRead { ticket: ticket as u64, file: 0, range: range.clone() });
        }
        source.start(PageRead { ticket: 99, file: 3, range: 0..4 });
        let mut done = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while done.len() < ranges.len() + 1 && Instant::now() < deadline {
            source.poll(&mut done);
            std::thread::yield_now();
        }
        assert_eq!(done.len(), ranges.len() + 1);
        for completion in done {
            match completion.ticket {
                3 | 99 => assert!(completion.result.is_err(), "reads past the end / of missing files fail"),
                t => {
                    let r = ranges[t as usize].clone();
                    assert_eq!(completion.result.unwrap(), data[r.start as usize..r.end as usize]);
                }
            }
        }
        assert_eq!(source.stats().failed, 2);

        // Emulated latency holds completions back without blocking `poll`.
        let slow = FileSource::with_latency(&[dir.join("data.bin")], 1, Duration::from_millis(50)).unwrap();
        let start = Instant::now();
        slow.start(PageRead { ticket: 7, file: 0, range: 10..20 });
        let mut done = Vec::new();
        while done.is_empty() {
            slow.poll(&mut done);
            assert!(start.elapsed() < Duration::from_secs(10));
            std::thread::yield_now();
        }
        assert!(start.elapsed() >= Duration::from_millis(50));
        assert_eq!(done[0].result.as_deref(), Ok(&data[10..20]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
