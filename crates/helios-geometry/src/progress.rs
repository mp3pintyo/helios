//! A sign of life from long bakes.
//!
//! A bake runs to completion in one call: a mesh's DAG (seconds for a large terrain in the
//! browser) or a distance field. Whoever waits for it on another thread — the browser's bake
//! workers are watched by the page, which terminates a worker that stays silent — cannot tell a
//! slow bake from a hung one. The bakes call [`tick`] at regular points (each simplified group,
//! every few thousand triangles of a distance field); a thread that installed a [`set_hook`]
//! hears of it every [`TICKS_PER_CALL`] ticks. Without a hook a tick is a thread-local counter.

use std::cell::Cell;

/// Ticks between calls of the hook (a tick is a group simplification, ~0.05–1 ms).
pub const TICKS_PER_CALL: u32 = 32;

thread_local! {
    static HOOK: Cell<Option<fn()>> = const { Cell::new(None) };
    static TICKS: Cell<u32> = const { Cell::new(0) };
}

/// Installs (or with `None` removes) this thread's progress hook.
pub fn set_hook(hook: Option<fn()>) {
    HOOK.with(|h| h.set(hook));
}

/// Reports progress; calls the hook every [`TICKS_PER_CALL`] calls.
#[inline]
pub fn tick() {
    let Some(hook) = HOOK.with(Cell::get) else { return };
    let ticks = TICKS.with(|t| {
        let n = t.get().wrapping_add(1);
        t.set(n);
        n
    });
    if ticks % TICKS_PER_CALL == 0 {
        hook();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static CALLS: AtomicU32 = AtomicU32::new(0);

    #[test]
    fn a_large_bake_reports_progress() {
        fn hook() {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        set_hook(Some(hook));
        let mesh = crate::primitives::icosphere(5);
        let _ = crate::build_virtual_mesh(&mesh, &crate::BuildConfig::default());
        set_hook(None);
        assert!(CALLS.load(Ordering::Relaxed) > 0, "no progress reported");
    }
}
