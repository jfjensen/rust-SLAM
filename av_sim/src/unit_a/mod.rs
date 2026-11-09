// Unit A of the course: where am I from the wheels alone, and what do I see?
//
//   motion.rs     the motion model: ticks -> poses (dead reckoning)
//   cylinders.rs  the cylinder detector: scan -> landmarks in the robot frame
//
// Each is a pure function (tested on its own) plus a plugin that runs it
// on the RobotLog. Add both with UnitAPlugins.

pub mod cylinders;
pub mod motion;

// NEW: plugin_group! makes a PluginGroup (like DefaultPlugins) from a list
// of `module:::Plugin` entries. app.add_plugins(UnitAPlugins) adds them all.
use bevy::app::plugin_group;

plugin_group! {
    /// Unit A's algorithms.
    pub struct UnitAPlugins {
        motion:::MotionModelPlugin,
        cylinders:::CylinderDetectionPlugin,
    }
}

/// The numbers behind the test thresholds:
///     cargo test measure -- --ignored --nocapture
#[cfg(test)]
mod measure {
    use super::cylinders::{DetectorParams, detect};
    use super::motion::{MotionParams, poses_from_ticks};
    use crate::synth::{DEFAULT_STEPS, SynthConfig, generate};
    use bevy::math::Vec2;

    #[test]
    #[ignore]
    fn measure() {
        for (name, config) in [
            ("perfect", SynthConfig::PERFECT),
            ("realistic", SynthConfig::REALISTIC),
        ] {
            for seed in 1..=3 {
                let log = generate(config, seed, DEFAULT_STEPS);
                let poses = poses_from_ticks(&log.motor_increments(), &MotionParams::default());
                let errors: Vec<f32> = poses
                    .iter()
                    .zip(&log.reference)
                    .map(|(p, r)| p.position().distance(Vec2::new(r.x as f32, r.y as f32)))
                    .collect();
                let worst = errors.iter().copied().fold(0.0, f32::max);
                let p = DetectorParams::default();
                let det: Vec<usize> = log
                    .scans
                    .iter()
                    .map(|s| detect(&s.ranges, &p).len())
                    .collect();
                println!(
                    "{name:9} seed {seed}: dead reckoning error last {:6.1} mm, worst {:6.1} mm; {} cylinders in {} scans",
                    errors.last().unwrap(),
                    worst,
                    det.iter().sum::<usize>(),
                    det.len()
                );
            }
        }
    }
}
