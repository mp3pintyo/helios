//! Screen percentage, sub-pixel jitter and dynamic resolution: the CPU half of temporal super
//! resolution (the reconstruction itself is `passes/tsr.rs`).
//!
//! As in Unreal Engine (`r.ScreenPercentage`, `r.DynamicRes.*`), the scene renders at an
//! *internal* resolution, a percentage of the output, and TSR reconstructs the output from
//! jittered internal frames. Internal targets are allocated once for the largest percentage
//! the settings allow and each frame renders into their top-left corner, so a dynamic
//! resolution change reallocates nothing and every history stays valid.

use glam::Vec2;

use crate::camera::halton;

/// Smallest screen percentage the renderer accepts.
pub const MIN_SCREEN_PERCENTAGE: f32 = 25.0;
/// Largest screen percentage (the output resolution; no supersampling).
pub const MAX_SCREEN_PERCENTAGE: f32 = 100.0;

/// Clamps a requested screen percentage to the supported range.
pub fn clamp_percentage(percentage: f32) -> f32 {
    if percentage.is_finite() {
        percentage.clamp(MIN_SCREEN_PERCENTAGE, MAX_SCREEN_PERCENTAGE)
    } else {
        MAX_SCREEN_PERCENTAGE
    }
}

/// Internal render size for an `output` size at `percentage` (per axis, rounded, never 0 and
/// never larger than the output).
pub fn internal_size(output: (u32, u32), percentage: f32) -> (u32, u32) {
    let fraction = clamp_percentage(percentage) / 100.0;
    let axis = |n: u32| ((n as f32 * fraction).round() as u32).clamp(1, n.max(1));
    (axis(output.0), axis(output.1))
}

/// Number of jitter positions before the sequence repeats, for a ratio of internal to output
/// pixels of `scale` per axis. Eight positions cover an output pixel at 100 %; an upscaled
/// output pixel is hit by a sample only `scale²` of the time, so the cycle grows by `1 / scale²`
/// (UE scales `r.TemporalAASamples` the same way) so that every output pixel still sees about
/// eight samples per cycle.
pub fn jitter_phase_count(scale: f32) -> u32 {
    let scale = scale.clamp(0.01, 1.0);
    ((8.0 / (scale * scale)).ceil() as u32).clamp(8, 64)
}

/// Sub-pixel jitter of frame `frame` in internal pixels, each axis in [-0.5, 0.5): the Halton
/// (2, 3) sequence over [`jitter_phase_count`] phases.
pub fn jitter_offset(frame: u32, scale: f32) -> Vec2 {
    let i = frame % jitter_phase_count(scale) + 1;
    Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5)
}

/// Dynamic resolution: the renderer picks the screen percentage that keeps the GPU frame time
/// under a budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicResolution {
    /// GPU time per frame to stay under, in milliseconds (measured with timestamp queries;
    /// without them the percentage stays at `max_percentage`).
    pub gpu_budget_ms: f32,
    pub min_percentage: f32,
    pub max_percentage: f32,
}

impl DynamicResolution {
    /// A budget of `gpu_budget_ms` between 50 and 100 %.
    pub fn with_budget(gpu_budget_ms: f32) -> Self {
        Self { gpu_budget_ms, min_percentage: 50.0, max_percentage: 100.0 }
    }

    /// The bounds, clamped and ordered.
    fn bounds(&self) -> (f32, f32) {
        let max = clamp_percentage(self.max_percentage);
        (clamp_percentage(self.min_percentage).min(max), max)
    }
}

/// Picks the screen percentage for dynamic resolution from measured GPU frame times.
///
/// GPU time is modelled as `fixed + k · percentage²` (pixel work scales with the area). The
/// controller averages a window of frames, holds while the average lies in a dead band below
/// the budget, and otherwise steps towards `TARGET` of the budget assuming all of the time
/// scales with the area — which under-corrects (part of the time is fixed), so it approaches
/// from one side instead of oscillating. After a change it discards the frames still in
/// flight at the old resolution. Decreases may be large and quick, increases are limited, as
/// a missed budget costs more than a slightly soft image.
#[derive(Clone, Debug)]
pub struct ResolutionController {
    percentage: f32,
    /// Sum and count of the frame times of the current window.
    window_ms: f32,
    window_frames: usize,
    /// Samples still to ignore after a change (measured at the previous resolution).
    skip: u32,
    last_sample: Option<u64>,
}

impl ResolutionController {
    /// Frames averaged per decision.
    const WINDOW: usize = 6;
    /// Frames ignored after a change: GPU timestamps arrive a few frames late.
    const LATENCY: u32 = 3;
    /// Fraction of the budget the controller steers to.
    const TARGET: f32 = 0.9;
    /// Below this fraction of the budget the resolution goes up (above 1.0 it goes down).
    const LOW: f32 = 0.8;
    const MAX_STEP_DOWN: f32 = 0.75;
    const MAX_STEP_UP: f32 = 1.1;

    pub fn new(percentage: f32) -> Self {
        Self { percentage: clamp_percentage(percentage), window_ms: 0.0, window_frames: 0, skip: 0, last_sample: None }
    }

    #[cfg(test)]
    pub fn percentage(&self) -> f32 {
        self.percentage
    }

    /// Feeds the GPU time of a completed frame, `(sample id, milliseconds)`, measured at
    /// [`Self::percentage`]; the same id is only counted once. Returns the percentage to render
    /// the next frame at.
    pub fn update(&mut self, sample: Option<(u64, f32)>, settings: &DynamicResolution) -> f32 {
        let (min, max) = settings.bounds();
        self.percentage = self.percentage.clamp(min, max);
        let Some((id, ms)) = sample.filter(|&(id, ms)| Some(id) != self.last_sample && ms.is_finite() && ms > 0.0) else {
            return self.percentage;
        };
        self.last_sample = Some(id);
        if self.skip > 0 {
            self.skip -= 1;
            return self.percentage;
        }
        self.window_ms += ms;
        self.window_frames += 1;
        if self.window_frames < Self::WINDOW {
            return self.percentage;
        }
        let average = self.window_ms / self.window_frames as f32;
        (self.window_ms, self.window_frames) = (0.0, 0);

        let budget = settings.gpu_budget_ms.max(0.1);
        if average > budget * Self::LOW && average <= budget {
            return self.percentage;
        }
        let step = (budget * Self::TARGET / average).sqrt().clamp(Self::MAX_STEP_DOWN, Self::MAX_STEP_UP);
        // Whole percent steps: sub-percent changes are invisible and only churn.
        let next = (self.percentage * step).round().clamp(min, max);
        if next != self.percentage {
            self.percentage = next;
            self.skip = Self::LATENCY;
        }
        self.percentage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_size_rounds_per_axis_and_stays_in_range() {
        assert_eq!(internal_size((2560, 1440), 100.0), (2560, 1440));
        assert_eq!(internal_size((2560, 1440), 50.0), (1280, 720));
        assert_eq!(internal_size((2560, 1440), 67.0), (1715, 965));
        assert_eq!(internal_size((1, 1), 25.0), (1, 1));
        // Out-of-range requests are clamped, not trusted.
        assert_eq!(internal_size((1600, 900), 400.0), (1600, 900));
        assert_eq!(internal_size((1600, 900), 1.0), (400, 225));
        assert_eq!(internal_size((1600, 900), f32::NAN), (1600, 900));
    }

    #[test]
    fn jitter_cycle_grows_with_the_upscale_ratio() {
        assert_eq!(jitter_phase_count(1.0), 8);
        assert_eq!(jitter_phase_count(0.5), 32);
        assert_eq!(jitter_phase_count(2.0 / 3.0), 18);
        assert_eq!(jitter_phase_count(0.25), 64);
    }

    #[test]
    fn jitter_matches_halton_and_covers_the_pixel() {
        // At 100 % the sequence is the classic 8-tap Halton (2, 3).
        for frame in 0..16 {
            let i = frame % 8 + 1;
            assert_eq!(jitter_offset(frame, 1.0), Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5));
        }
        for scale in [1.0, 0.67, 0.5] {
            let n = jitter_phase_count(scale);
            let offsets: Vec<Vec2> = (0..n).map(|f| jitter_offset(f, scale)).collect();
            assert!(offsets.iter().all(|o| o.cmpge(Vec2::splat(-0.5)).all() && o.cmplt(Vec2::splat(0.5)).all()));
            // Unbiased: the cycle averages to the pixel centre.
            let mean = offsets.iter().sum::<Vec2>() / n as f32;
            assert!(mean.abs().max_element() < 0.07, "scale {scale}: mean {mean}");
            // Every quadrant of the pixel is visited about equally often.
            for quadrant in 0..4 {
                let (sx, sy) = (quadrant & 1 == 1, quadrant & 2 == 2);
                let count = offsets.iter().filter(|o| (o.x >= 0.0) == sx && (o.y >= 0.0) == sy).count() as f32;
                assert!((count / n as f32 - 0.25).abs() < 0.1, "scale {scale}: quadrant {quadrant} has {count} of {n}");
            }
        }
    }

    /// GPU time of a frame at `percentage`: fixed work plus work proportional to the area.
    fn model(fixed: f32, at_full: f32, percentage: f32) -> f32 {
        fixed + at_full * (percentage / 100.0).powi(2)
    }

    /// Runs the controller for `frames` frames; returns the percentages it chose.
    fn run(controller: &mut ResolutionController, settings: &DynamicResolution, frames: u64, gpu: impl Fn(u64, f32) -> f32) -> Vec<f32> {
        (0..frames)
            .map(|frame| {
                let ms = gpu(frame, controller.percentage());
                controller.update(Some((frame, ms)), settings)
            })
            .collect()
    }

    #[test]
    fn converges_under_the_budget_and_then_holds() {
        let settings = DynamicResolution::with_budget(8.0);
        let mut controller = ResolutionController::new(100.0);
        // 2 ms fixed + 10 ms of pixel work at 100 %: 12 ms, over budget.
        let history = run(&mut controller, &settings, 300, |_, p| model(2.0, 10.0, p));
        let settled = &history[150..];
        assert!(settled.iter().all(|&p| p == settled[0]), "keeps changing: {settled:?}");
        let ms = model(2.0, 10.0, settled[0]);
        assert!(ms <= 8.0 && ms > 8.0 * 0.8, "settled at {} % = {ms} ms", settled[0]);
        // Approaches from above without undershooting below the dead band on the way.
        assert!(history.iter().all(|&p| model(2.0, 10.0, p) > 8.0 * 0.75 || p == 100.0));
    }

    #[test]
    fn respects_bounds() {
        let settings = DynamicResolution { gpu_budget_ms: 5.0, min_percentage: 60.0, max_percentage: 90.0 };
        let mut controller = ResolutionController::new(100.0);
        assert_eq!(controller.update(None, &settings), 90.0);
        let heavy = run(&mut controller, &settings, 200, |_, p| model(4.0, 40.0, p));
        assert_eq!(*heavy.last().unwrap(), 60.0);
        assert!(heavy.iter().all(|&p| (60.0..=90.0).contains(&p)));
        let light = run(&mut controller, &settings, 400, |_, p| model(0.5, 1.0, p));
        assert_eq!(*light.last().unwrap(), 90.0);
    }

    #[test]
    fn noise_and_repeated_samples_do_not_cause_churn() {
        let settings = DynamicResolution::with_budget(10.0);
        let mut controller = ResolutionController::new(70.0);
        // ±6 % pseudo-random noise around 9 ms (inside the dead band).
        let noisy = run(&mut controller, &settings, 600, |frame, _| {
            let hash = (frame.wrapping_mul(2654435761) % 1000) as f32 / 1000.0;
            9.0 * (1.0 + (hash - 0.5) * 0.12)
        });
        assert!(noisy.iter().all(|&p| p == 70.0));
        // The same sample fed again (no new timestamps) is not counted twice.
        let mut controller = ResolutionController::new(100.0);
        for _ in 0..100 {
            controller.update(Some((7, 50.0)), &settings);
        }
        assert_eq!(controller.percentage(), 100.0);
    }

    #[test]
    fn reacts_to_a_load_change() {
        let settings = DynamicResolution::with_budget(8.0);
        let mut controller = ResolutionController::new(100.0);
        // Light scene, then the camera turns to something three times as expensive.
        let history = run(&mut controller, &settings, 400, |frame, p| model(1.0, if frame < 100 { 5.0 } else { 15.0 }, p));
        assert_eq!(history[99], 100.0);
        let after = *history.last().unwrap();
        assert!(after < 75.0 && model(1.0, 15.0, after) <= 8.0, "{after}");
        // Within a third of a second at 60 Hz it is back under budget.
        let first_ok = history[100..].iter().position(|&p| model(1.0, 15.0, p) <= 8.0).unwrap();
        assert!(first_ok < 20, "took {first_ok} frames");
    }
}
