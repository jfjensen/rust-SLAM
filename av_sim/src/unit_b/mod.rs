// Unit B of the course: correct the drifting pose with what the scanner
// sees, against a known map.
//
//   similarity.rs  the similarity transform: estimate, apply, chain
//   landmarks.rs   pair detected cylinders with the map's, correct the pose
//   walls.rs       ICP: pull the scan points onto the walls, correct the pose
//
// Two corrections side by side, each its own plugin, each predicting from
// its own last corrected pose. They use Unit A's motion model and cylinder
// detector as plain functions, so they don't need Unit A's plugins.

pub mod landmarks;
pub mod similarity;
pub mod walls;

use crate::logfile::LogFile;
use crate::robot::Pose;
use crate::unit_a::cylinders::DetectorParams;
use crate::unit_a::motion::MotionParams;
use crate::world::ARENA_SIZE;
use bevy::app::plugin_group;
use bevy::math::Vec2;
use landmarks::{LandmarkInputs, LandmarkParams, LandmarkStep, landmark_step};
use walls::{WallParams, WallStep, wall_step};

plugin_group! {
    /// Unit B's algorithms.
    pub struct UnitBPlugins {
        landmarks:::LandmarkCorrectionPlugin,
        walls:::WallCorrectionPlugin,
    }
}

// --- Whole runs, without Bevy (for tests and measuring) ---------------------------

/// The landmark correction over a whole log, with the default parameters.
pub fn landmark_trajectory(log: &LogFile) -> Vec<LandmarkStep> {
    let motion = MotionParams::default();
    let landmarks: Vec<Vec2> = log.landmarks.iter().map(|l| l.position).collect();
    let inputs = LandmarkInputs {
        motion: &motion,
        detector: &DetectorParams::default(),
        params: &LandmarkParams::default(),
        landmarks: &landmarks,
    };
    let mut pose = motion.start;
    (0..log.motors.len().min(log.scans.len()))
        .map(|i| {
            let step = landmark_step(pose, log.motor_increment(i), &log.scans[i].ranges, &inputs);
            pose = step.pose;
            step
        })
        .collect()
}

/// The wall correction over a whole log, with the default parameters.
pub fn wall_trajectory(log: &LogFile) -> Vec<WallStep> {
    let motion = MotionParams::default();
    let params = WallParams::default();
    let mut pose = motion.start;
    (0..log.motors.len().min(log.scans.len()))
        .map(|i| {
            let step = wall_step(
                pose,
                log.motor_increment(i),
                &log.scans[i].ranges,
                Vec2::splat(ARENA_SIZE),
                &motion,
                &params,
            );
            pose = step.pose;
            step
        })
        .collect()
}

/// Mean and worst distance (mm) from the reference positions.
pub fn error_stats(poses: &[Pose], log: &LogFile) -> (f32, f32) {
    let errors: Vec<f32> = poses
        .iter()
        .zip(&log.reference)
        .map(|(p, r)| p.position().distance(Vec2::new(r.x as f32, r.y as f32)))
        .collect();
    let mean = errors.iter().sum::<f32>() / errors.len().max(1) as f32;
    (mean, errors.iter().copied().fold(0.0, f32::max))
}

/// The numbers behind the test thresholds:
///     cargo test unit_b::measure -- --ignored --nocapture
#[cfg(test)]
mod measure {
    use super::*;
    use crate::synth::{DEFAULT_STEPS, SynthConfig, generate};
    use crate::unit_a::motion::poses_from_ticks;

    #[test]
    #[ignore]
    fn measure() {
        for seed in 1..=5 {
            let log = generate(SynthConfig::REALISTIC, seed, DEFAULT_STEPS);
            let dr = poses_from_ticks(&log.motor_increments(), &MotionParams::default());
            let lm = landmark_trajectory(&log);
            let wall = wall_trajectory(&log);
            let lm_poses: Vec<Pose> = lm.iter().map(|s| s.pose).collect();
            let wall_poses: Vec<Pose> = wall.iter().map(|s| s.pose).collect();
            let show = |(mean, worst): (f32, f32)| format!("mean {mean:5.1} worst {worst:5.1}");
            println!(
                "seed {seed}: dead reckoning {} | landmarks {} ({} of {} steps corrected) | walls {}",
                show(error_stats(&dr, &log)),
                show(error_stats(&lm_poses, &log)),
                lm.iter().filter(|s| s.transform.is_some()).count(),
                lm.len(),
                show(error_stats(&wall_poses, &log)),
            );
        }
    }
}
