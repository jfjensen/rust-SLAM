// Unit A, part 1: the motion model (the course's slam_02_b filter_step).
//
// From two encoder increments to a new scanner pose. Both wheels moved the
// same: drive straight. Otherwise the robot drove along an arc:
//
//     l, r   = wheel travel in mm (ticks * ticks_to_mm)
//     alpha  = (r - l) / width          the turn, in radians
//     R      = l / alpha                the left wheel's arc radius
//     c      = the arc's centre, R + width/2 to the robot's left
//
// The model is for the axle centre, but the pose is the scanner's: step
// back to the centre first, and forward to the scanner at the end.

use crate::core::{AvSet, RobotLog};
use crate::robot::{Pose, ROBOT_WIDTH, SCANNER_DISPLACEMENT, START_POSE, TICKS_TO_MM, wrap_angle};
use bevy::math::Vec2;
use bevy::prelude::*;

/// The robot as the motion model believes it to be.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct MotionParams {
    pub ticks_to_mm: f32,
    pub robot_width: f32,
    pub scanner_displacement: f32,
    /// The measured start pose (scanner).
    pub start: Pose,
}

impl Default for MotionParams {
    fn default() -> MotionParams {
        MotionParams {
            ticks_to_mm: TICKS_TO_MM,
            robot_width: ROBOT_WIDTH,
            scanner_displacement: SCANNER_DISPLACEMENT,
            start: START_POSE,
        }
    }
}

/// One step of the motion model: old scanner pose + tick increments
/// (left, right) -> new scanner pose.
pub fn filter_step(old: Pose, ticks: (i64, i64), p: &MotionParams) -> Pose {
    let centre = old.shifted(-p.scanner_displacement);
    let l = ticks.0 as f32 * p.ticks_to_mm;
    let r = ticks.1 as f32 * p.ticks_to_mm;

    let moved = if ticks.0 == ticks.1 {
        // Straight on.
        let pos = centre.position() + centre.forward() * l;
        Pose::new(pos.x, pos.y, centre.heading)
    } else {
        let alpha = (r - l) / p.robot_width;
        let radius = l / alpha + p.robot_width / 2.0;
        // The vector from the arc's centre to the robot is "to the right"
        // of the heading: (sin, -cos).
        let right = |heading: f32| Vec2::new(heading.sin(), -heading.cos());
        let c = centre.position() - right(centre.heading) * radius;
        let heading = centre.heading + alpha;
        let pos = c + right(heading) * radius;
        Pose::new(pos.x, pos.y, wrap_angle(heading))
    };
    // NOTE: forward along the NEW heading. (The repo's solution adds the
    // displacement along the old heading in the turning case: a small
    // error, up to 30 mm * alpha per step.)
    moved.shifted(p.scanner_displacement)
}

/// The whole trajectory from the start pose: poses_from_ticks.txt.
pub fn poses_from_ticks(increments: &[(i64, i64)], p: &MotionParams) -> Vec<Pose> {
    let mut pose = p.start;
    increments
        .iter()
        .map(|&inc| {
            pose = filter_step(pose, inc, p);
            pose
        })
        .collect()
}

// --- The plugin ----------------------------------------------------------------
//
// Reads RobotLog, writes DeadReckoning: one pose per motor record. It only
// computes the new steps each frame (the live robot adds one at a time),
// and starts over when a different log is loaded (the generation changed).

#[derive(Resource, Default, Debug)]
pub struct DeadReckoning {
    pub poses: Vec<Pose>,
    generation: u64,
}

#[derive(Default)]
pub struct MotionModelPlugin;

impl Plugin for MotionModelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionParams>()
            .init_resource::<DeadReckoning>()
            .add_systems(Update, update_dead_reckoning.in_set(AvSet::Algorithms));
    }
}

fn update_dead_reckoning(
    log: Res<RobotLog>,
    params: Res<MotionParams>,
    mut dr: ResMut<DeadReckoning>,
) {
    if dr.generation != log.generation || params.is_changed() {
        dr.poses.clear();
        dr.generation = log.generation;
    }
    let motors = &log.log.motors;
    let from = dr.poses.len();
    if from >= motors.len() {
        return;
    }
    for i in from..motors.len() {
        let old = dr.poses.last().copied().unwrap_or(params.start);
        dr.poses
            .push(filter_step(old, log.log.motor_increment(i), &params));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SynthConfig, generate};

    #[test]
    fn equal_ticks_drive_straight() {
        let p = MotionParams::default();
        let start = Pose::new(0.0, 0.0, 0.0);
        let end = filter_step(start, (100, 100), &p);
        assert!((end.x - 34.9).abs() < 1e-3 && end.y.abs() < 1e-4);
    }

    #[test]
    fn a_full_circle_comes_back() {
        // Right wheel only: the robot pivots around its left wheel.
        let p = MotionParams {
            scanner_displacement: 0.0,
            ..MotionParams::default()
        };
        let start = Pose::new(0.0, 0.0, 0.0);
        // A quarter turn: r = width * pi / 2 mm.
        let ticks = (ROBOT_WIDTH * std::f32::consts::FRAC_PI_2 / TICKS_TO_MM).round() as i64;
        let end = filter_step(start, (0, ticks), &p);
        // The left wheel at (0, 75) stays put; the centre swings around it.
        assert!(
            (end.position() - Vec2::new(75.0, 75.0)).length() < 0.5,
            "{end:?}"
        );
        assert!((end.heading - std::f32::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn scanner_displacement_is_on_the_new_heading() {
        // Turn on the spot: the centre doesn't move, so the scanner must
        // stay 30 mm from it, along the new heading.
        let p = MotionParams::default();
        let start = Pose::new(SCANNER_DISPLACEMENT, 0.0, 0.0);
        let end = filter_step(start, (-200, 200), &p);
        let centre = end.shifted(-SCANNER_DISPLACEMENT);
        assert!(centre.position().length() < 1e-3, "{centre:?}");
    }

    /// On a perfect robot the motion model matches the simulator's own
    /// (sub-step) integration: two independent calculations agree.
    #[test]
    fn matches_the_simulator_when_calibration_is_perfect() {
        let log = generate(SynthConfig::PERFECT, 1, 278);
        let poses = poses_from_ticks(&log.motor_increments(), &MotionParams::default());
        let worst = poses
            .iter()
            .zip(&log.reference)
            .map(|(p, r)| p.position().distance(Vec2::new(r.x as f32, r.y as f32)))
            .fold(0.0, f32::max);
        // Ticks are whole numbers and references whole mm: about 1 mm
        // (measured: 1.0, see the measure test in unit_a/mod.rs).
        assert!(worst < 3.0, "worst {worst}");
    }

    /// On the realistic robot it drifts: that's what units B to G fix.
    #[test]
    fn drifts_when_calibration_is_off() {
        let log = generate(SynthConfig::REALISTIC, 1, 278);
        let poses = poses_from_ticks(&log.motor_increments(), &MotionParams::default());
        let last = poses.last().unwrap().position();
        let r = log.reference.last().unwrap();
        let error = last.distance(Vec2::new(r.x as f32, r.y as f32));
        // Measured: 114 mm for seed 1 (100 to 210 for seeds 1 to 3).
        assert!(error > 50.0, "error {error}");
    }
}
