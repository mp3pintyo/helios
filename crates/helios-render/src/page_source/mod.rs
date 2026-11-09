//! Asynchronous sources of streamed page payloads.
//!
//! A mesh added with [`crate::Scene::add_streamed_mesh`] keeps only its header (cluster and page
//! records, dependencies, bounds, distance field) and root pages in memory. The streamer fetches
//! every other page payload from a [`PageSource`] as a byte range of the mesh's `.hvm` file
//! ([`helios_geometry::VirtualMesh::page_byte_range`]): from disk with [`FileSource`]
//! (background threads doing positioned reads) or over HTTP with `HttpSource` (`fetch` with a
//! `Range` header, in the browser). Reads never block the render thread: they are started with
//! [`PageSource::start`] and collected with [`PageSource::poll`] in a later frame.

use std::ops::Range;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
mod file;
#[cfg(target_arch = "wasm32")]
mod http;

#[cfg(not(target_arch = "wasm32"))]
pub use file::{FileSource, IoStats};
#[cfg(target_arch = "wasm32")]
pub use http::{fetch, Fetched, HttpSource};

/// One asynchronous read: bytes `range` of file `file` of a source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageRead {
    /// Identifies the read in its [`ReadCompletion`].
    pub ticket: u64,
    pub file: u32,
    pub range: Range<u64>,
}

/// The outcome of a [`PageRead`]: exactly `range.len()` bytes, or why they could not be read.
#[derive(Debug)]
pub struct ReadCompletion {
    pub ticket: u64,
    pub result: Result<Vec<u8>, String>,
}

/// `Send + Sync` natively (so scenes and renderers holding sources stay `Send`); no bound in the
/// browser, where `fetch` and its futures are single-threaded.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> MaybeSendSync for T {}
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSendSync for T {}

/// An asynchronous byte-range reader over a set of files.
///
/// Implementations must never block in [`PageSource::start`] or [`PageSource::poll`]: both are
/// called on the render thread. A failed read is not fatal — the pages simply stay
/// non-resident (their coarser parents keep being drawn) and are requested again later.
pub trait PageSource: MaybeSendSync {
    /// Starts reading; the result is delivered by a later [`PageSource::poll`].
    fn start(&self, read: PageRead);

    /// Appends the reads that completed since the previous call.
    fn poll(&self, completed: &mut Vec<ReadCompletion>);

    /// Reads the streamer keeps outstanding at once. Bounding it keeps requests from piling up
    /// behind a slow device, so newly needed pages are not stuck behind stale ones.
    fn max_in_flight(&self) -> usize {
        8
    }

    /// Unrequested bytes worth reading (and discarding) to merge two nearby ranges into one
    /// read. Larger for high-latency sources, where a request costs more than bandwidth.
    fn max_gap_bytes(&self) -> u64 {
        0
    }
}

/// Where a streamed mesh's page payloads live: file `file` of `source`, laid out as a `.hvm`.
#[derive(Clone)]
pub struct PageLocation {
    pub source: Arc<dyn PageSource>,
    pub file: u32,
}

/// A run of coalesced spans: one read covering `spans[first..first + count]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub range: Range<u64>,
    pub first: usize,
    pub count: usize,
}

/// Merges byte `spans` (sorted by start, non-overlapping) into reads of at most `max_bytes`
/// (a single larger span forms its own read), bridging gaps of up to `max_gap` bytes.
pub(crate) fn coalesce(spans: &[Range<u64>], max_bytes: u64, max_gap: u64) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (i, span) in spans.iter().enumerate() {
        if let Some(run) = runs.last_mut() {
            let gap = span.start.checked_sub(run.range.end);
            if gap.is_some_and(|g| g <= max_gap) && span.end - run.range.start <= max_bytes {
                run.range.end = span.end;
                run.count += 1;
                continue;
            }
        }
        runs.push(Run { range: span.clone(), first: i, count: 1 });
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_merges_adjacent_spans_up_to_the_size_limit() {
        let spans = [0..10, 10..30, 30..35, 40..50, 50..200, 200..210];
        let runs = coalesce(&spans, 64, 0);
        assert_eq!(
            runs,
            vec![
                Run { range: 0..35, first: 0, count: 3 },
                Run { range: 40..50, first: 3, count: 1 },
                // Larger than the limit on its own: still one read.
                Run { range: 50..200, first: 4, count: 1 },
                Run { range: 200..210, first: 5, count: 1 },
            ]
        );
        // Every span is covered by exactly one run, in order.
        let covered: usize = runs.iter().map(|r| r.count).sum();
        assert_eq!(covered, spans.len());
    }

    #[test]
    fn coalesce_bridges_small_gaps_only() {
        let spans = [0..10, 14..20, 100..110];
        assert_eq!(coalesce(&spans, 1 << 20, 4), vec![Run { range: 0..20, first: 0, count: 2 }, Run { range: 100..110, first: 2, count: 1 }]);
        assert_eq!(coalesce(&spans, 1 << 20, 3).len(), 3);
        assert!(coalesce(&[], 64, 0).is_empty());
    }
}
