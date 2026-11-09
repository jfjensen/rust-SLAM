// Unit B, part 1: the similarity transform (the course's slam_04_c/d and
// concatenate_transform in slam_b_library).
//
// A similarity moves a point p to   scale * R(angle) * p + translation.
// Given matching point pairs (left[i] should land on right[i]), the best
// one in the least-squares sense has a closed form:
//
//   1. centre both lists on their centres of mass, lc and rc
//   2. cs = sum(r . l)          ss = sum(r x l) (the 2D cross product)
//      the rotation is the direction of (cs, ss)
//   3. scale = sqrt(sum |r|^2 / sum |l|^2)   (or 1, with fix_scale)
//   4. translation = rc - scale * R * lc
//
// The rotation is kept as (cos, sin), like the course's tuple, rather than
// an angle: that's what the formula gives, and what applying it needs.

use crate::robot::{Pose, wrap_angle};
use bevy::math::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub scale: f32,
    /// (cos, sin) of the rotation angle: a unit vector.
    pub rotation: Vec2,
    pub translation: Vec2,
}

impl Similarity {
    pub const IDENTITY: Similarity = Similarity {
        scale: 1.0,
        rotation: Vec2::X,
        translation: Vec2::ZERO,
    };

    /// The course's apply_transform.
    pub fn apply(&self, p: Vec2) -> Vec2 {
        // Vec2::rotate multiplies as complex numbers: rotation by (cos, sin).
        self.scale * self.rotation.rotate(p) + self.translation
    }

    pub fn angle(&self) -> f32 {
        self.rotation.to_angle()
    }

    /// `self` after `first`: the course's concatenate_transform(self, first).
    pub fn after(&self, first: &Similarity) -> Similarity {
        Similarity {
            scale: self.scale * first.scale,
            rotation: self.rotation.rotate(first.rotation),
            translation: self.apply(first.translation),
        }
    }

    /// The course's correct_pose: move the position with the transform,
    /// and turn the heading by its angle.
    pub fn correct_pose(&self, pose: Pose) -> Pose {
        let p = self.apply(pose.position());
        Pose::new(p.x, p.y, wrap_angle(pose.heading + self.angle()))
    }

    /// The course's estimate_transform. None with fewer than two pairs,
    /// or when the points are all in one place (no direction to align).
    pub fn estimate(left: &[Vec2], right: &[Vec2], fix_scale: bool) -> Option<Similarity> {
        assert_eq!(left.len(), right.len(), "point pairs");
        if left.len() < 2 {
            return None;
        }
        let lc = centre(left);
        let rc = centre(right);
        let (mut cs, mut ss, mut rr, mut ll) = (0.0, 0.0, 0.0, 0.0);
        for (&l, &r) in left.iter().zip(right) {
            let (l, r) = (l - lc, r - rc);
            cs += r.dot(l);
            ss += l.perp_dot(r);
            rr += r.length_squared();
            ll += l.length_squared();
        }
        if rr == 0.0 || ll == 0.0 {
            return None;
        }
        // NOTE: the repo's solution also gives up when cs or ss is exactly
        // 0, but ss = 0 just means "no rotation needed". Only both being 0
        // leaves the angle undefined.
        let rotation = Vec2::new(cs, ss).try_normalize()?;
        let scale = if fix_scale { 1.0 } else { (rr / ll).sqrt() };
        Some(Similarity {
            scale,
            rotation,
            translation: rc - scale * rotation.rotate(lc),
        })
    }
}

/// Centre of mass (the course's compute_center). Zero for no points.
pub fn centre(points: &[Vec2]) -> Vec2 {
    if points.is_empty() {
        return Vec2::ZERO;
    }
    points.iter().sum::<Vec2>() / points.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec2, b: Vec2) -> bool {
        a.distance(b) < 1e-2
    }

    /// Move some points by a known transform: estimate finds it again.
    #[test]
    fn estimate_recovers_a_known_transform() {
        let known = Similarity {
            scale: 1.3,
            rotation: Vec2::from_angle(0.4),
            translation: Vec2::new(120.0, -45.0),
        };
        let left = [
            Vec2::new(100.0, 200.0),
            Vec2::new(900.0, 150.0),
            Vec2::new(400.0, 1200.0),
            Vec2::new(1500.0, 1700.0),
        ];
        let right: Vec<Vec2> = left.iter().map(|&p| known.apply(p)).collect();
        let found = Similarity::estimate(&left, &right, false).unwrap();
        assert!((found.scale - 1.3).abs() < 1e-4, "{found:?}");
        assert!((found.angle() - 0.4).abs() < 1e-4, "{found:?}");
        assert!(close(found.translation, known.translation), "{found:?}");

        // fix_scale: the scale stays 1, the rotation is still found.
        let rigid = Similarity {
            scale: 1.0,
            ..known
        };
        let right: Vec<Vec2> = left.iter().map(|&p| rigid.apply(p)).collect();
        let found = Similarity::estimate(&left, &right, true).unwrap();
        assert!(close(found.translation, rigid.translation), "{found:?}");
    }

    /// A pure shift has ss = 0 exactly. The repo's version returns None
    /// for it; this one finds the shift.
    #[test]
    fn a_pure_shift_is_found() {
        let left = [Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)];
        let right = [Vec2::new(10.0, 5.0), Vec2::new(110.0, 5.0)];
        let found = Similarity::estimate(&left, &right, true).unwrap();
        assert_eq!(found.rotation, Vec2::X);
        assert!(close(found.translation, Vec2::new(10.0, 5.0)));
    }

    #[test]
    fn too_few_or_degenerate_pairs_give_none() {
        let p = Vec2::new(5.0, 5.0);
        assert_eq!(Similarity::estimate(&[p], &[p], true), None);
        assert_eq!(Similarity::estimate(&[p, p], &[p, p], true), None);
    }

    /// Applying `a.after(&b)` is applying b, then a.
    #[test]
    fn after_is_apply_twice() {
        let a = Similarity {
            scale: 0.9,
            rotation: Vec2::from_angle(-1.1),
            translation: Vec2::new(3.0, 4.0),
        };
        let b = Similarity {
            scale: 1.2,
            rotation: Vec2::from_angle(0.3),
            translation: Vec2::new(-50.0, 20.0),
        };
        let p = Vec2::new(70.0, -30.0);
        assert!(close(a.after(&b).apply(p), a.apply(b.apply(p))));
        assert!(close(Similarity::IDENTITY.after(&a).apply(p), a.apply(p)));
    }

    /// Correcting a pose moves it the way its scan points move: a point
    /// seen at the same local position lands where the transform puts it.
    #[test]
    fn correct_pose_moves_the_scan_with_it() {
        let t = Similarity {
            scale: 1.0,
            rotation: Vec2::from_angle(0.2),
            translation: Vec2::new(30.0, -10.0),
        };
        let pose = Pose::new(800.0, 600.0, 1.0);
        let local = Vec2::new(400.0, 50.0);
        let moved = t.correct_pose(pose);
        assert!(close(moved.to_world(local), t.apply(pose.to_world(local))));
    }
}
