//! Bounding volumes.

use glam::{Vec3, Vec4};

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self { min: Vec3::splat(f32::MAX), max: Vec3::splat(f32::MIN) };

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        points.into_iter().fold(Self::EMPTY, |b, p| Self { min: b.min.min(p), max: b.max.max(p) })
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }
}

/// Bounding sphere. A negative radius denotes the empty sphere.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    pub const EMPTY: Self = Self { center: Vec3::ZERO, radius: -1.0 };

    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Ritter's approximate minimal bounding sphere (within ~5–20% of optimal).
    pub fn from_points(points: &[Vec3]) -> Self {
        let Some(&first) = points.first() else { return Self::EMPTY };

        // Pick the most separated pair among the axis-extreme points.
        let mut extremes = [first; 6];
        for &p in points {
            for axis in 0..3 {
                if p[axis] < extremes[axis * 2][axis] {
                    extremes[axis * 2] = p;
                }
                if p[axis] > extremes[axis * 2 + 1][axis] {
                    extremes[axis * 2 + 1] = p;
                }
            }
        }
        let (mut a, mut b) = (extremes[0], extremes[1]);
        for axis in 1..3 {
            let (c, d) = (extremes[axis * 2], extremes[axis * 2 + 1]);
            if c.distance_squared(d) > a.distance_squared(b) {
                (a, b) = (c, d);
            }
        }

        let mut center = (a + b) * 0.5;
        let mut radius = a.distance(b) * 0.5;
        for &p in points {
            let d = p.distance(center);
            if d > radius {
                let new_radius = (radius + d) * 0.5;
                center += (p - center) * ((new_radius - radius) / d);
                radius = new_radius;
            }
        }
        Self { center, radius: radius * (1.0 + 1e-5) + 1e-7 }
    }

    /// Smallest sphere enclosing both spheres.
    pub fn merge(self, other: Self) -> Self {
        if self.radius < 0.0 {
            return other;
        }
        if other.radius < 0.0 {
            return self;
        }
        let d = other.center - self.center;
        let dist = d.length();
        if dist + other.radius <= self.radius {
            return self;
        }
        if dist + self.radius <= other.radius {
            return other;
        }
        let radius = (dist + self.radius + other.radius) * 0.5;
        let center = self.center + d * ((radius - self.radius) / dist);
        Self { center, radius: radius * (1.0 + 1e-6) }
    }

    pub fn merge_all(spheres: impl IntoIterator<Item = Sphere>) -> Self {
        spheres.into_iter().fold(Self::EMPTY, Self::merge)
    }

    pub fn contains_sphere(&self, other: &Sphere, tolerance: f32) -> bool {
        self.center.distance(other.center) + other.radius <= self.radius + tolerance
    }

    pub fn to_vec4(self) -> Vec4 {
        self.center.extend(self.radius)
    }
}
