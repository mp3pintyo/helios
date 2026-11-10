//! Cached shadow maps for local lights that move (UE re-renders a movable light's virtual shadow
//! map whenever it moves; this keeps most of the cache).
//!
//! A light's shadow map is a view from one point, and moving that point makes every cached page
//! stale — the whole cube map re-rendered each frame (0.35 ms per chandelier light in the
//! cathedral). But a light is not a point: it has a source radius, and its shadow map may be
//! rendered from *any* point inside the source — the penumbra a soft-shadow filter draws spans
//! every such point's shadow. So a moving light's map keeps the **origin** it was rendered from
//! while the light stays within a small fraction of its source radius of it ([`TOLERANCE`]):
//!
//! * the pages' static casters stay cached (their static layer), and the dynamic ones — the
//!   fixture swinging with the light — are redrawn at their current pose from the held origin, as
//!   they are for a light at rest (`vsm_motion.rs`), so a fixture never shadows itself with a
//!   stale pose;
//! * the receivers correct the parallax between the origin and the light: an occluder at a
//!   fraction 1 / t of the receiver's distance casts its shadow (t − 1) · (light − origin) away
//!   from where the held map has it, so the soft-shadow filter, which knows t from its blocker
//!   search, moves its taps by as much (`vsm_local_sample.wgsl`). Shadows follow the light
//!   smoothly instead of standing still between two renders; what is left is what the origin
//!   cannot see (a sliver of width (t − 1) · offset next to silhouettes, a fraction
//!   `TOLERANCE / 2` of the penumbra there);
//! * when the light leaves the tolerance, its map is **re-anchored**: rendered again in full
//!   from a new origin placed ahead of the light on its path ([`LEAD`]), so that the light
//!   crosses the whole tolerance sphere before the next render.
//!
//! Re-anchoring costs a full render of the light's requested pages, so it is budgeted per frame
//! ([`AnchorSettings::budget`]); the lights furthest out of tolerance go first. A moving light
//! only gets a shadow map if the budget can keep up with it — its recent top speed tells how
//! often it will need a new origin, and the lights that hold maps must not need more renders
//! together than the budget allows — and one that outruns it all the same (beyond
//! [`HARD_LIMIT`] tolerances from its origin) gives its map up. Such lights keep distance-field
//! shadows until they rest, as do lights beyond the slot budget: a sharp, fast light is not
//! flipped between the two kinds of shadow. At rest a light is re-anchored once more, at its
//! own position.

use glam::Vec3;

use crate::scene::Light;

/// A shadow map's origin may be this fraction of the source radius away from the light: the
/// parallax-corrected shadow is then off by at most half of it, relative to the penumbra.
pub(crate) const TOLERANCE: f32 = 0.1;
/// Smallest tolerance (m): sharp lights' shadows are still filtered over a few texels.
const MIN_TOLERANCE: f32 = 0.01;
/// A new origin is placed this fraction of the tolerance ahead of the light, along its motion.
const LEAD: f32 = 0.75;
/// Beyond this many tolerances from its origin a light waiting for the budget gives its shadow
/// map up.
const HARD_LIMIT: f32 = 2.0;
/// Per frame, a light's top speed decays by this factor towards its current speed: a pendulum
/// is still judged by its speed at the bottom while it turns (half-life: two seconds).
const SPEED_DECAY: f32 = 0.995;

/// How far `light` may be from the origin of its shadow map.
pub(crate) fn tolerance(light: &Light) -> f32 {
    (light.source_radius * TOLERANCE).max(MIN_TOLERANCE)
}

/// How lights that move are shadowed (`RenderSettings::moving_light_shadows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MovingLightShadows {
    /// Distance-field soft shadows while they move, shadow maps at rest.
    DistanceFields,
    /// Shadow maps held at an origin near the light and re-anchored within a budget (this
    /// module); distance fields for lights the budget cannot follow.
    #[default]
    Cached,
    /// Shadow maps re-rendered in full every frame (the reference; 0.35 ms per light and frame
    /// in the cathedral).
    EveryFrame,
}

/// Per-frame parameters of [`Anchors::update`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnchorSettings {
    pub mode: MovingLightShadows,
    /// Shadow maps of moving lights rendered in full per frame (re-anchored or new).
    pub budget: usize,
}

/// What one light wants this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightState {
    pub position: Vec3,
    /// Moved within the last few frames (`vsm_local.rs`).
    pub moving: bool,
    /// Its move since the last frame.
    pub step: Vec3,
    pub tolerance: f32,
    /// Its influence on the screen; 0: wants no shadow map (casts no shadows, out of view).
    pub influence: f32,
}

/// The origins moving lights' shadow maps are rendered from (pure CPU logic).
#[derive(Debug, Default)]
pub(crate) struct Anchors {
    /// Per light: the origin of its shadow map, if it has one.
    origins: Vec<Option<Vec3>>,
    /// Per light: gave its shadow map up; no new one until it rests.
    dropped: Vec<bool>,
    /// Per light: its recent top speed (distance per frame).
    speeds: Vec<f32>,
}

impl Anchors {
    /// Per light: the origin to render its shadow map from this frame, or `None` for a light
    /// that gets no shadow map (not wanted, or moving beyond what the mode and budget allow).
    /// Follow with [`Anchors::keep`] once the slots are assigned.
    pub fn update(&mut self, lights: &[LightState], settings: AnchorSettings) -> Vec<Option<Vec3>> {
        if self.origins.len() != lights.len() {
            self.origins = vec![None; lights.len()];
            self.dropped = vec![false; lights.len()];
            self.speeds = vec![0.0; lights.len()];
        }
        // Renders per frame a light's map needs at its recent top speed: one per trip through
        // its tolerance sphere.
        let rate = |speed: f32, light: &LightState| speed / ((1.0 + LEAD) * light.tolerance);
        // Lights whose map must be rendered anew, with their urgency (distance from the origin
        // in tolerances, then influence: new maps and maps of lights that came to rest wait
        // behind lights out of tolerance), and the renders per frame the held maps need.
        let mut renders: Vec<(usize, f32)> = Vec::new();
        let mut load = 0.0;
        for (i, light) in lights.iter().enumerate() {
            self.speeds[i] = light.step.length().max(self.speeds[i] * SPEED_DECAY);
            if !light.moving {
                self.dropped[i] = false;
                self.speeds[i] = 0.0;
            }
            let cached = settings.mode == MovingLightShadows::Cached;
            if light.influence <= 0.0 || self.dropped[i] || (light.moving && settings.mode == MovingLightShadows::DistanceFields) {
                self.origins[i] = None;
                continue;
            }
            match self.origins[i] {
                // Static lights and the reference mode: always at the light, never budgeted.
                _ if !cached => self.origins[i] = Some(light.position),
                None if !light.moving => self.origins[i] = Some(light.position),
                None => renders.push((i, 0.0)),
                Some(origin) => {
                    load += rate(self.speeds[i], light);
                    let error = origin.distance(light.position) / light.tolerance;
                    if error > 1.0 || (!light.moving && error > 0.0) {
                        renders.push((i, error));
                    }
                }
            }
        }
        renders.sort_by(|a, b| b.1.total_cmp(&a.1).then(lights[b.0].influence.total_cmp(&lights[a.0].influence)).then(a.0.cmp(&b.0)));
        let mut left = settings.budget;
        for &(i, error) in &renders {
            let light = &lights[i];
            let new = self.origins[i].is_none();
            let sustained = load + rate(self.speeds[i], light) <= settings.budget as f32;
            if left > 0 && (!new || sustained) {
                left -= 1;
                load += if new { rate(self.speeds[i], light) } else { 0.0 };
                let ahead = if light.moving { light.step.normalize_or_zero() * (light.tolerance * LEAD) } else { Vec3::ZERO };
                self.origins[i] = Some(light.position + ahead);
            } else if !new && error > HARD_LIMIT {
                self.origins[i] = None;
                self.dropped[i] = true;
            }
        }
        self.origins.clone()
    }

    /// Forgets the origins of lights that got no slot (`has_slot` per light).
    pub fn keep(&mut self, has_slot: impl Fn(usize) -> bool) {
        for (i, origin) in self.origins.iter_mut().enumerate() {
            if !has_slot(i) {
                *origin = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHED: AnchorSettings = AnchorSettings { mode: MovingLightShadows::Cached, budget: 1 };

    fn light(x: f32, moving: bool) -> LightState {
        LightState { position: Vec3::new(x, 0.0, 0.0), moving, step: if moving { Vec3::X * 0.01 } else { Vec3::ZERO }, tolerance: 0.1, influence: 1.0 }
    }

    #[test]
    fn origin_is_held_within_the_tolerance_and_leads_the_light() {
        let mut anchors = Anchors::default();
        // A new moving light: rendered from a point ahead of it.
        let first = anchors.update(&[light(0.0, true)], CACHED)[0].unwrap();
        assert!((first.x - 0.1 * LEAD).abs() < 1e-6);
        // It passes through the origin and on to the far side of the tolerance sphere.
        let mut x = 0.0;
        let mut renders = 0;
        let mut origin = first;
        for _ in 0..100 {
            x += 0.01;
            let now = anchors.update(&[light(x, true)], CACHED)[0].unwrap();
            assert!((now.x - x).abs() <= 0.1 + 1e-5, "within tolerance after the update");
            renders += (now != origin) as usize;
            origin = now;
        }
        // 1 m of travel at 0.175 m per origin.
        assert!((5..=6).contains(&renders), "{renders}");
    }

    #[test]
    fn lights_at_rest_are_exact_and_free() {
        let mut anchors = Anchors::default();
        let none = AnchorSettings { budget: 0, ..CACHED };
        // Static lights never wait for the budget.
        assert_eq!(anchors.update(&[light(1.0, false)], none), vec![Some(Vec3::X)]);
        // A moving light that comes to rest is re-anchored at its position, within the budget.
        let mut anchors = Anchors::default();
        let held = anchors.update(&[light(0.0, true)], CACHED)[0].unwrap();
        assert_eq!(anchors.update(&[light(0.0, false)], none), vec![Some(held)]);
        assert_eq!(anchors.update(&[light(0.0, false)], CACHED), vec![Some(Vec3::ZERO)]);
        assert_eq!(anchors.update(&[light(0.0, false)], none), vec![Some(Vec3::ZERO)]);
    }

    #[test]
    fn budget_serves_the_light_furthest_out_and_drops_runaways() {
        let mut anchors = Anchors::default();
        let two = AnchorSettings { budget: 2, ..CACHED };
        let start = anchors.update(&[light(0.0, true), light(5.0, true)], two);
        let (a, b) = (start[0].unwrap(), start[1].unwrap());
        // Both leave their tolerance; the second is further out and gets the frame's render.
        let lights = [light(a.x + 0.12, true), light(b.x + 0.15, true)];
        let next = anchors.update(&lights, CACHED);
        assert_eq!(next[0], Some(a));
        assert_ne!(next[1], Some(b));
        // The first runs beyond the hard limit while the budget is spent elsewhere: it loses its
        // map, and gets no new one until it has rested.
        let none = AnchorSettings { budget: 0, ..CACHED };
        let far = [light(a.x + 0.25, true), lights[1]];
        assert_eq!(anchors.update(&far, none)[0], None);
        assert_eq!(anchors.update(&far, two)[0], None);
        let rested = [light(a.x + 0.25, false), lights[1]];
        assert_eq!(anchors.update(&rested, two)[0], Some(rested[0].position));
    }

    #[test]
    fn new_moving_lights_wait_for_the_budget() {
        let mut anchors = Anchors::default();
        let lights = [light(0.0, true), light(1.0, true)];
        let first = anchors.update(&lights, CACHED);
        assert_eq!(first.iter().flatten().count(), 1);
        anchors.keep(|i| first[i].is_some());
        assert_eq!(anchors.update(&lights, CACHED).iter().flatten().count(), 2);
    }

    #[test]
    fn only_lights_the_budget_can_follow_get_maps() {
        // Sharp, fast lights: each needs a new origin every 3.5 frames (0.01 a frame through a
        // tolerance of 0.02), so one render per frame sustains three of them. The others keep
        // distance fields instead of taking a map only to lose it; the most influential go first.
        let fast = |i: usize| LightState { tolerance: 0.02, influence: 1.0 + i as f32, ..light(i as f32, true) };
        let mut lights: Vec<LightState> = (0..8).map(fast).collect();
        let mut anchors = Anchors::default();
        let mut mapped = Vec::new();
        for _ in 0..60 {
            for l in &mut lights {
                l.position += l.step;
            }
            let origins = anchors.update(&lights, CACHED);
            anchors.keep(|i| origins[i].is_some());
            mapped = (0..8).filter(|&i| origins[i].is_some()).collect();
            for &i in &mapped {
                assert!(origins[i].unwrap().distance(lights[i].position) <= 0.02 * HARD_LIMIT + 1e-5);
            }
        }
        assert_eq!(mapped, vec![5, 6, 7]);
        assert!(anchors.dropped.iter().all(|&d| !d), "nobody took a map it could not keep");
    }

    #[test]
    fn modes_without_caching() {
        let lights = [light(0.0, true), light(1.0, false)];
        let mut anchors = Anchors::default();
        let fields = AnchorSettings { mode: MovingLightShadows::DistanceFields, budget: 1 };
        assert_eq!(anchors.update(&lights, fields), vec![None, Some(Vec3::X)]);
        let every = AnchorSettings { mode: MovingLightShadows::EveryFrame, budget: 0 };
        assert_eq!(anchors.update(&lights, every), vec![Some(Vec3::ZERO), Some(Vec3::X)]);
        // Unwanted lights (out of view, no shadows) have no origin.
        let hidden = LightState { influence: 0.0, ..lights[1] };
        assert_eq!(anchors.update(&[lights[0], hidden], every)[1], None);
        // A slot that went to another light: the origin is forgotten.
        anchors.keep(|i| i == 1);
        assert_eq!(anchors.origins[0], None);
    }
}
