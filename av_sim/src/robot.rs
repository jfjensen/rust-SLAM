// Part 1: the robot and its scanner, as the course describes them.
//
// All lengths in millimetres, angles in radians, x to the right and y up
// (the same as Bevy's 2D world, and as the course's plots).
//
// The course's robot is a LEGO Mindstorms car with two driven wheels and
// a LiDAR mounted 30 mm in front of the axle. Every pose the course
// computes is the SCANNER's pose, not the axle centre's: that's where the
// measurements come from.

use bevy::math::Vec2;
use std::f32::consts::TAU;

/// Millimetres per encoder tick, as calibrated for the LEGO robot.
pub const TICKS_TO_MM: f32 = 0.349;
/// Distance between the two wheels (the wheel gauge).
pub const ROBOT_WIDTH: f32 = 150.0;
/// The scanner sits this far in front of the axle centre.
pub const SCANNER_DISPLACEMENT: f32 = 30.0;

/// The LiDAR: 660 beams, counterclockwise, index 330 straight ahead.
/// (The original LEGO robot has its scanner turned 4 degrees to the
/// right, a mounting angle of -0.0698 rad. Ours is mounted straight.)
pub const BEAM_COUNT: usize = 660;
pub const BEAM_STEP: f32 = 0.006_135_923;

/// Where the course's robot starts (scanner pose).
pub const START_POSE: Pose = Pose {
    x: 1850.0,
    y: 1897.0,
    heading: 213.0 / 180.0 * std::f32::consts::PI,
};

/// A position and a heading (0 = along +x, counterclockwise positive).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Pose {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
}

impl Pose {
    pub const fn new(x: f32, y: f32, heading: f32) -> Pose {
        Pose { x, y, heading }
    }

    pub fn position(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Unit vector in the heading direction.
    pub fn forward(&self) -> Vec2 {
        Vec2::from_angle(self.heading)
    }

    /// The same robot, `distance` further forward (negative: back).
    /// Turns a centre pose into a scanner pose, and back.
    pub fn shifted(&self, distance: f32) -> Pose {
        let p = self.position() + self.forward() * distance;
        Pose::new(p.x, p.y, self.heading)
    }

    /// A point in this pose's own frame (x ahead, y to the left) in world
    /// coordinates. The original `scanner_to_world`.
    pub fn to_world(&self, local: Vec2) -> Vec2 {
        self.position() + self.forward().rotate(local)
    }
}

/// Wrap an angle into [0, 2 pi), as the course does with `% (2 * pi)`.
pub fn wrap_angle(a: f32) -> f32 {
    a.rem_euclid(TAU)
}

/// The course's `beam_index_to_angle`: beam index (may be fractional,
/// for the average ray of a cylinder) to an angle in the scanner frame.
pub fn beam_index_to_angle(index: f32) -> f32 {
    (index - 330.0) * BEAM_STEP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beam_330_looks_straight_ahead() {
        assert!(beam_index_to_angle(330.0).abs() < 1e-6);
        // The full fan is about 232 degrees.
        let fan = beam_index_to_angle(659.0) - beam_index_to_angle(0.0);
        assert!(
            (fan.to_degrees() - 231.7).abs() < 0.1,
            "{}",
            fan.to_degrees()
        );
    }

    #[test]
    fn to_world_turns_and_moves() {
        let pose = Pose::new(100.0, 200.0, std::f32::consts::FRAC_PI_2);
        // 10 mm ahead of a robot facing +y is 10 mm up.
        let p = pose.to_world(Vec2::new(10.0, 0.0));
        assert!((p - Vec2::new(100.0, 210.0)).length() < 1e-4);
        // 10 mm to its left is 10 mm towards -x.
        let p = pose.to_world(Vec2::new(0.0, 10.0));
        assert!((p - Vec2::new(90.0, 200.0)).length() < 1e-4);
    }

    #[test]
    fn shifting_there_and_back_is_the_same_pose() {
        let pose = Pose::new(5.0, -3.0, 2.0);
        let back = pose
            .shifted(SCANNER_DISPLACEMENT)
            .shifted(-SCANNER_DISPLACEMENT);
        assert!((back.position() - pose.position()).length() < 1e-4);
    }
}
