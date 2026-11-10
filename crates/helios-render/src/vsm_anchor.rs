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
//! * when the light leaves the tolerance, its map is **re-anchored**: rendered again from a new
//!   origin placed ahead of the light on its path ([`lead`]), so that the light crosses the
//!   whole tolerance sphere before the next render.
//!
//! **Every face of a cube map has an origin of its own**, and faces are re-anchored one by one.
//! A re-anchor renders the pages the view requests of the face — its cost, which the GPU counts
//! per face every frame (`vsm.wgsl`, `dirty_info`) — so the budget is a number of pages per
//! frame ([`AnchorSettings::budget`]), not of lights: the faces furthest out of tolerance go
//! first, a face nobody looks at costs its one fallback page, and a light whose six faces fall
//! due together is spread over the next frames instead of waiting for a frame with room for all
//! of it. A light moving faster than its tolerance sphere each frame is re-rendered every frame
//! from where it is (no lead: the reference's image) as long as the budget pays for its pages.
//!
//! A moving light only gets a shadow map if the budget can keep up with it — its recent top
//! speed tells how often each face will need a new origin, and the faces of the lights that hold
//! maps must not need more pages per frame together than the budget allows — at its tolerance,
//! or else at twice or four times it ([`MAX_RELAX`]): the parallax correction keeps a map held
//! further off far closer to the reference than distance fields. A held map whose faces fall
//! behind all the same (beyond [`HARD_LIMIT`] tolerances from a face's origin, plus
//! [`LAG_FRAMES`] of the light's motion) has its tolerance doubled in the same way, and only
//! beyond four times it gives its map up. Lights without a map keep distance-field shadows
//! until they rest, as do lights beyond the slot budget: a light is not flipped between the two
//! kinds of shadow. At rest a light is re-anchored once more, at its own position.

use glam::Vec3;

use crate::scene::Light;

/// Faces of a light's shadow map at most (a cube; spot lights have one).
pub(crate) const FACES: usize = 6;
/// A shadow map's origin may be this fraction of the source radius away from the light: the
/// parallax-corrected shadow is then off by at most half of it, relative to the penumbra.
pub(crate) const TOLERANCE: f32 = 0.1;
/// Smallest tolerance (m): sharp lights' shadows are still filtered over a few texels.
const MIN_TOLERANCE: f32 = 0.01;
/// A new origin is placed up to this fraction of the tolerance ahead of the light, along its
/// motion.
const LEAD: f32 = 0.75;
/// Beyond this many tolerances from a face's origin — plus [`LAG_FRAMES`] of its motion — a
/// light waiting for the budget gives its shadow map up.
const HARD_LIMIT: f32 = 2.0;
/// Frames a face of a light that crosses its tolerance sphere every frame may wait for the
/// budget (a frame whose faces cost more than counted) before the light gives its map up.
const LAG_FRAMES: f32 = 2.0;
/// A light the budget cannot follow at its tolerance gets up to this many times the tolerance
/// (doubling it each time), before it keeps distance-field shadows: the parallax correction
/// keeps such a map far closer to the reference than distance fields (the carried lantern at four
/// times its 1 cm: 43 dB against the every-frame reference, distance fields 34 dB).
const MAX_RELAX: f32 = 4.0;
/// Per frame, a light's top speed decays by this factor towards its current speed: a pendulum
/// is still judged by its speed at the bottom while it turns (half-life: two seconds).
const SPEED_DECAY: f32 = 0.995;
/// Pages a face is assumed to cost before the GPU has counted them (a light never mapped).
pub(crate) const DEFAULT_FACE_PAGES: f32 = 8.0;

/// How far `light` may be from the origin of its shadow map.
pub(crate) fn tolerance(light: &Light) -> f32 {
    (light.source_radius * TOLERANCE).max(MIN_TOLERANCE)
}

/// How far ahead of a light moving `step` per frame a new origin goes, so that the light stays
/// within `tolerance` of it for as many frames as possible and is never further from it than
/// needed meanwhile: with k frames until the next render (the most a lead of up to
/// [`LEAD`] tolerances allows), the error over them is max(lead, k · step − lead), smallest at
/// half the way, and k · step − lead must not exceed the tolerance. A light faster than
/// (1 + LEAD) tolerances a frame is rendered every frame: from where it is.
pub(crate) fn lead(step: f32, tolerance: f32) -> f32 {
    if step <= 0.0 {
        return 0.0;
    }
    let k = ((1.0 + LEAD) * tolerance / step).floor();
    (k * step * 0.5).max(k * step - tolerance).min(LEAD * tolerance)
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
    /// Pages of moving lights' faces rendered from new origins per frame (on average: a face
    /// larger than the budget is rendered alone in a frame).
    pub budget: f32,
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
    /// Faces of its shadow map (6: a cube, 1: a spot light).
    pub faces: usize,
    /// Per face: the pages a render from a new origin costs (those the view requests).
    pub pages: [f32; FACES],
}

impl LightState {
    fn cost(&self) -> f32 {
        self.pages[..self.faces].iter().sum()
    }
}

/// Per light: the origin of each face of its shadow map.
pub(crate) type Origins = [Vec3; FACES];

/// What the anchors did in the latest frame (the HUD's `moving lights` line).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnchorStats {
    /// Moving lights with a shadow map.
    pub mapped: u32,
    /// Moving lights that want one but keep distance-field shadows (beyond the budget).
    pub fields: u32,
    /// Of the moving lights with maps, those held at a coarser tolerance (the budget could not
    /// follow them at the usual one).
    pub relaxed: u32,
    /// Faces rendered from a new origin this frame, and their pages (as counted when the face
    /// was last rendered).
    pub faces: u32,
    pub pages: f32,
    /// Pages per frame the held maps need at their lights' recent top speeds.
    pub load: f32,
}

/// The origins moving lights' shadow maps are rendered from (pure CPU logic).
#[derive(Debug, Default)]
pub(crate) struct Anchors {
    /// Per light: the origins of its shadow map's faces, if it has one.
    origins: Vec<Option<Origins>>,
    /// Per light: gave its shadow map up; no new one until it rests.
    dropped: Vec<bool>,
    /// Per light: its recent top speed (distance per frame).
    speeds: Vec<f32>,
    /// Per light: moving, and wants a shadow map (for the statistics).
    wanting: Vec<bool>,
    /// Per light: its tolerance is this many times the usual one ([`MAX_RELAX`]), until it
    /// rests.
    relax: Vec<f32>,
    pub stats: AnchorStats,
}

/// A render the budget may pay for: a light's new map (all faces) or one face of a held map.
#[derive(Clone, Copy, Debug)]
struct Render {
    light: usize,
    face: Option<usize>,
    /// Distance from the origin in tolerances (0 for new maps).
    error: f32,
}

impl Anchors {
    /// Per light: the origins of its shadow map's faces this frame, or `None` for a light that
    /// gets no shadow map (not wanted, or moving beyond what the mode and budget allow). Follow
    /// with [`Anchors::keep`] once the slots are assigned.
    pub fn update(&mut self, lights: &[LightState], settings: AnchorSettings) -> Vec<Option<Origins>> {
        if self.origins.len() != lights.len() {
            self.origins = vec![None; lights.len()];
            self.dropped = vec![false; lights.len()];
            self.speeds = vec![0.0; lights.len()];
            self.relax = vec![1.0; lights.len()];
        }
        let cached = settings.mode == MovingLightShadows::Cached;
        // Renders per frame a face needs at a light's recent top speed: one per trip through
        // its tolerance sphere, at most one a frame.
        let rate = |speed: f32, tolerance: f32| (speed / ((1.0 + LEAD) * tolerance)).min(1.0);
        // Renders wanted this frame, and the pages per frame the held maps need.
        let mut renders: Vec<Render> = Vec::new();
        let mut load = 0.0;
        let mut stats = AnchorStats::default();
        for (i, light) in lights.iter().enumerate() {
            self.speeds[i] = light.step.length().max(self.speeds[i] * SPEED_DECAY);
            if !light.moving {
                self.dropped[i] = false;
                self.speeds[i] = 0.0;
            }
            let unwanted = light.influence <= 0.0 || self.dropped[i] || (light.moving && settings.mode == MovingLightShadows::DistanceFields);
            if unwanted {
                self.origins[i] = None;
            }
            if self.origins[i].is_none() || !light.moving {
                self.relax[i] = 1.0;
            }
            if unwanted {
                continue;
            }
            let tolerance = light.tolerance * self.relax[i];
            match &self.origins[i] {
                // Static lights and the reference mode: always at the light, never budgeted.
                _ if !cached => self.origins[i] = Some([light.position; FACES]),
                None if !light.moving => self.origins[i] = Some([light.position; FACES]),
                None => renders.push(Render { light: i, face: None, error: 0.0 }),
                Some(origins) => {
                    load += rate(self.speeds[i], tolerance) * light.cost();
                    for (face, origin) in origins[..light.faces].iter().enumerate() {
                        let error = origin.distance(light.position) / tolerance;
                        if error > 1.0 || (!light.moving && error > 0.0) {
                            renders.push(Render { light: i, face: Some(face), error });
                        }
                    }
                }
            }
        }
        // Furthest out of tolerance first; new maps and maps of lights that came to rest wait
        // behind them; then the most influential.
        renders.sort_by(|a, b| b.error.total_cmp(&a.error).then(lights[b.light].influence.total_cmp(&lights[a.light].influence)).then(a.light.cmp(&b.light)).then(a.face.cmp(&b.face)));
        let mut spent = 0.0;
        let mut relaxed_now = vec![false; lights.len()];
        for render in &renders {
            let i = render.light;
            let light = &lights[i];
            // A render larger than the whole budget goes alone, so that it is not starved.
            let fits = |cost: f32| (spent == 0.0 && settings.budget > 0.0) || spent + cost <= settings.budget;
            match render.face {
                None => {
                    // The tightest tolerance whose renders the budget sustains, up to MAX_RELAX
                    // times the usual one.
                    let cost = light.cost();
                    let relaxed = std::iter::successors(Some(1.0f32), |r| (*r < MAX_RELAX).then_some(r * 2.0));
                    let sustained = relaxed.map(|r| (r, rate(self.speeds[i], light.tolerance * r) * cost)).find(|&(_, need)| load + need <= settings.budget);
                    if let Some((relax, need)) = sustained.filter(|_| fits(cost)) {
                        spent += cost;
                        load += need;
                        self.relax[i] = relax;
                        self.origins[i] = Some([anchor_point(light, light.tolerance * relax); FACES]);
                        stats.faces += light.faces as u32;
                    }
                }
                Some(face) => {
                    let tolerance = light.tolerance * self.relax[i];
                    let Some(origins) = &mut self.origins[i] else { continue };
                    let cost = light.pages[face];
                    if fits(cost) {
                        spent += cost;
                        origins[face] = anchor_point(light, tolerance);
                        stats.faces += 1;
                    } else if render.error > HARD_LIMIT + LAG_FRAMES * light.step.length() / tolerance {
                        // Falling behind: a coarser tolerance needs fewer renders (once a
                        // frame); beyond the coarsest the light gives its map up until it rests.
                        if relaxed_now[i] {
                            continue;
                        }
                        if self.relax[i] < MAX_RELAX {
                            self.relax[i] *= 2.0;
                            relaxed_now[i] = true;
                        } else {
                            self.origins[i] = None;
                            self.dropped[i] = true;
                        }
                    }
                }
            }
        }
        self.wanting = lights.iter().map(|l| l.moving && l.influence > 0.0).collect();
        stats.pages = spent;
        stats.load = load;
        self.stats = stats;
        self.origins.clone()
    }

    /// Forgets the origins of lights that got no slot (`has_slot` per light). `wanting`: per
    /// light, moving and wanting a map before the slot competition turned it away (the
    /// statistics count it on distance fields; empty: those [`Anchors::update`] saw).
    pub fn keep(&mut self, has_slot: impl Fn(usize) -> bool, wanting: &[bool]) {
        for (i, origin) in self.origins.iter_mut().enumerate() {
            if !has_slot(i) {
                *origin = None;
            }
        }
        if wanting.len() == self.wanting.len() {
            self.wanting.copy_from_slice(wanting);
        }
        let wanting = || self.wanting.iter().zip(&self.origins).zip(&self.relax).filter(|((w, _), _)| **w);
        self.stats.mapped = wanting().filter(|((_, o), _)| o.is_some()).count() as u32;
        self.stats.fields = wanting().filter(|((_, o), _)| o.is_none()).count() as u32;
        self.stats.relaxed = wanting().filter(|((_, o), &r)| o.is_some() && r > 1.0).count() as u32;
    }
}

/// Where a face of `light` is rendered from: at the light, or ahead of it on its path when it
/// moves ([`lead`] for `tolerance`).
fn anchor_point(light: &LightState, tolerance: f32) -> Vec3 {
    if !light.moving {
        return light.position;
    }
    light.position + light.step.normalize_or_zero() * lead(light.step.length(), tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One page per face: the budget counts faces.
    const CACHED: AnchorSettings = AnchorSettings { mode: MovingLightShadows::Cached, budget: 6.0 };

    fn light(x: f32, moving: bool) -> LightState {
        LightState {
            position: Vec3::new(x, 0.0, 0.0),
            moving,
            step: if moving { Vec3::X * 0.01 } else { Vec3::ZERO },
            tolerance: 0.1,
            influence: 1.0,
            faces: FACES,
            pages: [1.0; FACES],
        }
    }

    #[test]
    fn origin_is_held_within_the_tolerance_and_leads_the_light() {
        let mut anchors = Anchors::default();
        // A new moving light: rendered from a point ahead of it.
        let first = anchors.update(&[light(0.0, true)], CACHED)[0].unwrap();
        assert!(first.iter().all(|o| (o.x - 0.1 * LEAD).abs() < 1e-6));
        // It passes through the origin and on to the far side of the tolerance sphere.
        let mut x = 0.0;
        let mut renders = 0;
        let mut origin = first[0];
        for _ in 0..100 {
            x += 0.01;
            let now = anchors.update(&[light(x, true)], CACHED)[0].unwrap();
            assert!(now.iter().all(|o| (o.x - x).abs() <= 0.1 + 1e-5), "within tolerance after the update");
            renders += (now[0] != origin) as usize;
            origin = now[0];
        }
        // 1 m of travel at 0.175 m per origin.
        assert!((5..=6).contains(&renders), "{renders}");
    }

    #[test]
    fn lead_keeps_the_error_within_the_tolerance() {
        for step in [0.001f32, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.5] {
            let tolerance = 0.1;
            let lead = lead(step, tolerance);
            assert!((0.0..=LEAD * tolerance + 1e-6).contains(&lead), "{step}: {lead}");
            // Frames until the light is out of tolerance, and the largest error before.
            let frames = ((tolerance + lead) / step).floor() as usize + 1;
            let worst = (0..frames).map(|k| (k as f32 * step - lead).abs()).fold(0.0, f32::max);
            assert!(worst <= tolerance + 1e-5, "{step}: {worst}");
            if step > (1.0 + LEAD) * tolerance {
                assert_eq!(lead, 0.0, "rendered every frame: from where the light is");
            }
        }
        // A slow light gets the full lead.
        assert!((lead(0.001, 0.1) - LEAD * 0.1).abs() < 1e-6);
    }

    #[test]
    fn lights_at_rest_are_exact_and_free() {
        let mut anchors = Anchors::default();
        let none = AnchorSettings { budget: 0.0, ..CACHED };
        // Static lights never wait for the budget.
        assert_eq!(anchors.update(&[light(1.0, false)], none), vec![Some([Vec3::X; FACES])]);
        // A moving light that comes to rest is re-anchored at its position, within the budget
        // (a render larger than the budget goes alone: here one face a frame).
        let mut anchors = Anchors::default();
        let held = anchors.update(&[light(0.0, true)], CACHED)[0].unwrap();
        assert_eq!(anchors.update(&[light(0.0, false)], none)[0], Some(held));
        let tiny = AnchorSettings { budget: 0.5, ..CACHED };
        let rested = anchors.update(&[light(0.0, false)], tiny)[0].unwrap();
        assert_eq!(rested.iter().filter(|&&o| o == Vec3::ZERO).count(), 1);
        assert_eq!(rested.iter().filter(|&&o| o == held[0]).count(), 5);
        assert_eq!(anchors.update(&[light(0.0, false)], CACHED), vec![Some([Vec3::ZERO; FACES])]);
    }

    #[test]
    fn faces_are_re_anchored_within_the_page_budget() {
        // Two lights, both out of tolerance; the second further. Its faces go first, then the
        // first light's, three faces per frame.
        let mut anchors = Anchors::default();
        let budget = AnchorSettings { budget: 3.0, ..CACHED };
        let lights = [light(0.0, false), light(5.0, false)];
        let start = anchors.update(&lights, budget);
        let moved = [LightState { position: Vec3::new(0.12, 0.0, 0.0), ..light(0.0, true) }, LightState { position: Vec3::new(5.15, 0.0, 0.0), ..light(5.0, true) }];
        let renders = |now: &[Option<Origins>], before: &[Option<Origins>], i: usize| (0..FACES).filter(|&f| now[i].unwrap()[f] != before[i].unwrap()[f]).count();
        let first = anchors.update(&moved, budget);
        assert_eq!((renders(&first, &start, 0), renders(&first, &start, 1)), (0, 3));
        let second = anchors.update(&moved, budget);
        assert_eq!((renders(&second, &first, 0), renders(&second, &first, 1)), (0, 3));
        let third = anchors.update(&moved, budget);
        assert_eq!((renders(&third, &second, 0), renders(&third, &second, 1)), (3, 0));
        assert_eq!(anchors.stats.faces, 3);
        // Expensive faces cost more of the budget: a face of six pages goes alone.
        let mut heavy = moved;
        heavy[0].pages = [6.0; FACES];
        heavy[0].position.x += 0.06;
        let fourth = anchors.update(&heavy, budget);
        assert_eq!(renders(&fourth, &third, 0), 1);
    }

    #[test]
    fn runaways_are_dropped_until_they_rest() {
        let mut anchors = Anchors::default();
        let start = anchors.update(&[light(0.0, true), light(5.0, true)], AnchorSettings { budget: 12.0, ..CACHED });
        let a = start[0].unwrap()[0];
        // Both leave their tolerance; the second is further out and gets the frame's budget.
        let none = AnchorSettings { budget: 1e-3, ..CACHED };
        let lights = [light(a.x + 0.12, true), light(start[1].unwrap()[0].x + 0.15, true)];
        let next = anchors.update(&lights, none);
        assert_eq!(next[0], start[0]);
        // The first runs far beyond the hard limit while the budget is spent elsewhere: at twice,
        // then four times the tolerance it is still too far, so it loses its map, and gets no
        // new one until it has rested.
        let far = [light(a.x + 1.0, true), LightState { position: lights[1].position + Vec3::X * 3.0, ..lights[1] }];
        assert!(anchors.update(&far, none)[0].is_some());
        assert!(anchors.update(&far, none)[0].is_some());
        assert_eq!(anchors.update(&far, none)[0], None);
        assert_eq!(anchors.update(&far, CACHED)[0], None);
        let rested = [light(a.x + 1.0, false), far[1]];
        assert_eq!(anchors.update(&rested, CACHED)[0], Some([rested[0].position; FACES]));
    }

    #[test]
    fn new_moving_lights_wait_for_the_budget() {
        let mut anchors = Anchors::default();
        let lights = [light(0.0, true), light(1.0, true)];
        let first = anchors.update(&lights, CACHED);
        assert_eq!(first.iter().flatten().count(), 1);
        anchors.keep(|i| first[i].is_some(), &[]);
        assert_eq!(anchors.update(&lights, CACHED).iter().flatten().count(), 2);
    }

    #[test]
    fn only_lights_the_budget_can_follow_get_maps() {
        // Sharp, fast spot lights (one face of one page): each needs a new origin every 3.5
        // frames (0.01 a frame through a tolerance of 0.02), so a page per frame sustains three
        // of them. The others keep distance fields instead of taking a map only to lose it;
        // the most influential go first.
        let fast = |i: usize| LightState { tolerance: 0.02, influence: 1.0 + i as f32, faces: 1, ..light(i as f32, true) };
        let mut lights: Vec<LightState> = (0..8).map(fast).collect();
        let mut anchors = Anchors::default();
        // (Room for the next two at twice and four times the tolerance.)
        let one = AnchorSettings { budget: 1.1, ..CACHED };
        let mut mapped = Vec::new();
        for _ in 0..60 {
            for l in &mut lights {
                l.position += l.step;
            }
            let origins = anchors.update(&lights, one);
            anchors.keep(|i| origins[i].is_some(), &[]);
            mapped = (0..8).filter(|&i| origins[i].is_some()).collect();
            for &i in &mapped {
                assert!(origins[i].unwrap()[0].distance(lights[i].position) <= 0.02 * anchors.relax[i] * HARD_LIMIT + 1e-5);
            }
        }
        assert_eq!(mapped, vec![3, 4, 5, 6, 7]);
        assert_eq!(anchors.relax[3..].to_vec(), vec![4.0, 2.0, 1.0, 1.0, 1.0]);
        assert!(anchors.dropped.iter().all(|&d| !d), "nobody took a map it could not keep");
        assert_eq!((anchors.stats.mapped, anchors.stats.fields, anchors.stats.relaxed), (5, 3, 2));
    }

    #[test]
    fn fast_lights_are_rendered_every_frame_from_where_they_are() {
        // Four tolerances a frame: every face, every frame, no lead. Six pages a frame.
        let mut anchors = Anchors::default();
        let mut l = LightState { step: Vec3::X * 0.4, ..light(0.0, true) };
        for _ in 0..10 {
            l.position += l.step;
            let origins = anchors.update(&[l], CACHED)[0].expect("the budget pays for it");
            assert!(origins.iter().all(|&o| o == l.position));
        }
        // A frame short of budget: its faces wait a frame (parallax-corrected meanwhile)
        // instead of the light dropping its map.
        l.position += l.step;
        let short = anchors.update(&[l], AnchorSettings { budget: 1.0, ..CACHED })[0].expect("waits");
        assert_eq!(short.iter().filter(|&&o| o == l.position).count(), 1);
        l.position += l.step;
        assert!(anchors.update(&[l], CACHED)[0].expect("caught up").iter().all(|&o| o == l.position));
        // With room for fewer pages it is held at a coarser tolerance: four times it, a render
        // every 1.75 frames (3.4 pages a frame)...
        let mut anchors = Anchors::default();
        assert!(anchors.update(&[l], AnchorSettings { budget: 5.0, ..CACHED })[0].is_some());
        assert_eq!(anchors.relax[0], 4.0);
        // ... and with less still it keeps distance fields.
        let mut anchors = Anchors::default();
        assert_eq!(anchors.update(&[l], AnchorSettings { budget: 3.0, ..CACHED })[0], None);
    }

    #[test]
    fn lights_falling_behind_are_relaxed_before_they_are_dropped() {
        // One face a frame, and a jump of 2.5 tolerances: the faces left waiting are beyond the
        // limit, so the tolerance doubles (once) instead of the map being dropped.
        let mut anchors = Anchors::default();
        let one = AnchorSettings { budget: 1.0, ..CACHED };
        let start = light(0.0, false);
        anchors.update(&[start], one);
        let jumped = LightState { position: Vec3::X * 0.25, ..light(0.0, true) };
        let origins = anchors.update(&[jumped], one)[0].expect("relaxed, not dropped");
        assert_eq!(origins.iter().filter(|o| o.x > 0.2).count(), 1);
        assert_eq!(anchors.relax[0], 2.0);
        // The others follow, a face a frame.
        for _ in 0..5 {
            anchors.update(&[jumped], one)[0].expect("kept");
        }
        assert_eq!(anchors.relax[0], 2.0);
        assert!(anchors.update(&[jumped], one)[0].unwrap().iter().all(|o| o.x > 0.2));
        // Beyond four times the tolerance it is dropped until it rests.
        let far = LightState { position: Vec3::X * 2.0, ..jumped };
        anchors.update(&[far], one);
        assert_eq!(anchors.relax[0], 4.0);
        assert_eq!(anchors.update(&[far], one)[0], None);
        assert_eq!(anchors.update(&[LightState { moving: false, ..far }], one)[0], Some([far.position; FACES]));
    }

    #[test]
    fn modes_without_caching() {
        let lights = [light(0.0, true), light(1.0, false)];
        let mut anchors = Anchors::default();
        let fields = AnchorSettings { mode: MovingLightShadows::DistanceFields, budget: 6.0 };
        assert_eq!(anchors.update(&lights, fields), vec![None, Some([Vec3::X; FACES])]);
        let every = AnchorSettings { mode: MovingLightShadows::EveryFrame, budget: 0.0 };
        assert_eq!(anchors.update(&lights, every), vec![Some([Vec3::ZERO; FACES]), Some([Vec3::X; FACES])]);
        // Unwanted lights (out of view, no shadows) have no origin.
        let hidden = LightState { influence: 0.0, ..lights[1] };
        assert_eq!(anchors.update(&[lights[0], hidden], every)[1], None);
        // A slot that went to another light: the origin is forgotten.
        anchors.keep(|i| i == 1, &[]);
        assert_eq!(anchors.origins[0], None);
    }
}
