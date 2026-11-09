// Part 3: the arena. A walled square with cylinders (the landmarks) in it,
// and the one question the LiDAR asks: how far until a ray hits something?
//
// The cylinder positions are the course's robot_arena_landmarks.txt. The
// diameter is ours: 90 mm cylinders show up on 4 or more beams even
// from the far corner, so the course's detector finds them.

use crate::logfile::Landmark;
use bevy::math::Vec2;

pub const ARENA_SIZE: f32 = 2000.0;
pub const CYLINDER_DIAMETER: f32 = 90.0;

/// The course's landmark positions.
pub const COURSE_LANDMARKS: [(f32, f32); 6] = [
    (1291.0, 1881.0),
    (482.0, 682.0),
    (1191.0, 747.0),
    (1693.0, 1043.0),
    (383.0, 1458.0),
    (1805.0, 190.0),
];

#[derive(Clone, Debug, PartialEq)]
pub struct Arena {
    /// Walls along x = 0, x = size.x, y = 0, y = size.y.
    pub size: Vec2,
    pub landmarks: Vec<Landmark>,
}

impl Arena {
    pub fn course() -> Arena {
        Arena::with_landmarks(
            COURSE_LANDMARKS
                .iter()
                .map(|&(x, y)| Landmark {
                    position: Vec2::new(x, y),
                    diameter: CYLINDER_DIAMETER,
                })
                .collect(),
        )
    }

    /// The course's arena with these landmarks (read from a file).
    pub fn with_landmarks(landmarks: Vec<Landmark>) -> Arena {
        Arena {
            size: Vec2::splat(ARENA_SIZE),
            landmarks,
        }
    }

    /// Distance along a ray (dir: unit vector) from a point inside the
    /// arena to the first wall or cylinder.
    pub fn raycast(&self, origin: Vec2, dir: Vec2) -> f32 {
        let mut best = f32::INFINITY;
        // Walls: where does the ray reach x = 0 or x = size.x (and y)?
        for (o, d, high) in [
            (origin.x, dir.x, self.size.x),
            (origin.y, dir.y, self.size.y),
        ] {
            if d > 0.0 {
                best = best.min((high - o) / d);
            } else if d < 0.0 {
                best = best.min(-o / d);
            }
        }
        for l in &self.landmarks {
            if let Some(t) = ray_circle(origin, dir, l.position, l.diameter / 2.0) {
                best = best.min(t);
            }
        }
        best
    }

    /// Would a round robot of this radius, centred here, touch anything?
    pub fn collides(&self, centre: Vec2, radius: f32) -> bool {
        let inside = centre.cmpge(Vec2::splat(radius)).all()
            && centre.cmple(self.size - Vec2::splat(radius)).all();
        !inside
            || self
                .landmarks
                .iter()
                .any(|l| centre.distance(l.position) < radius + l.diameter / 2.0)
    }
}

/// First positive distance along the ray to the circle, if it hits.
fn ray_circle(origin: Vec2, dir: Vec2, centre: Vec2, radius: f32) -> Option<f32> {
    // |origin + t dir - centre|^2 = r^2, with |dir| = 1:
    // t^2 + 2 b t + c = 0
    let oc = origin - centre;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_stop_at_the_walls() {
        let arena = Arena::with_landmarks(vec![]);
        let o = Vec2::new(500.0, 1500.0);
        assert!((arena.raycast(o, Vec2::X) - 1500.0).abs() < 1e-3);
        assert!((arena.raycast(o, Vec2::NEG_Y) - 1500.0).abs() < 1e-3);
        assert!((arena.raycast(o, Vec2::Y) - 500.0).abs() < 1e-3);
        // Diagonal to the top-left corner region: hits x = 0 first.
        let d = Vec2::new(-1.0, 1.0).normalize();
        assert!((arena.raycast(o, d) - 500.0 * 2f32.sqrt()).abs() < 1e-2);
    }

    #[test]
    fn rays_stop_at_the_front_of_a_cylinder() {
        let arena = Arena::with_landmarks(vec![Landmark {
            position: Vec2::new(1000.0, 1000.0),
            diameter: 100.0,
        }]);
        let t = arena.raycast(Vec2::new(500.0, 1000.0), Vec2::X);
        assert!((t - 450.0).abs() < 1e-3, "{t}");
        // A ray that passes beside it goes on to the wall.
        let t = arena.raycast(Vec2::new(500.0, 1100.0), Vec2::X);
        assert!((t - 1500.0).abs() < 1e-3, "{t}");
    }

    #[test]
    fn the_start_is_free_and_walls_collide() {
        let arena = Arena::course();
        let centre = crate::robot::START_POSE.shifted(-crate::robot::SCANNER_DISPLACEMENT);
        assert!(!arena.collides(centre.position(), 80.0));
        assert!(arena.collides(Vec2::new(50.0, 1000.0), 80.0));
        assert!(arena.collides(Vec2::new(1291.0, 1800.0), 80.0));
    }
}
