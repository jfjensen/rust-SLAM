// Unit F of the course: EKF SLAM, localising AND mapping at once.
//
//   slam.rs   the filter: a state that grows with every new landmark
//   here      one SLAM step on one log record (with the course's data
//             association: nearest landmark in the map, or a new one),
//             the plugin, whole runs, and how good the map is
//
// The filter never sees the arena's landmarks (the log's L records). The
// app draws them, and map_errors() compares, but that's all.

pub mod slam;

use crate::core::{AvSet, RobotLog};
use crate::logfile::LogFile;
use crate::robot::{Pose, START_POSE, wrap_angle};
use crate::unit_a::cylinders::{DetectorParams, detect};
use crate::unit_a::motion::MotionParams;
use crate::unit_d::ekf::{EkfParams, ErrorEllipse, wrap_pi};
use bevy::app::plugin_group;
use bevy::math::{DMat3, DVec2, DVec3};
use bevy::prelude::*;
use slam::Slam;

plugin_group! {
    /// Unit F's filter.
    pub struct UnitFPlugins {
        :SlamPlugin,
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SlamSetup {
    pub filter: EkfParams,
    /// Where the map's frame starts. The covariance there is zero: this
    /// pose DEFINES the frame.
    pub start: Pose,
    /// A detection further than this from every landmark in the map is a
    /// new landmark.
    pub max_cylinder_distance: f32,
}

impl Default for SlamSetup {
    /// The course's slam_09_c: much larger measurement noise than Unit D
    /// (600 mm, 45 degrees), and a wider association radius (500 mm).
    fn default() -> SlamSetup {
        SlamSetup {
            filter: EkfParams {
                measurement_distance_stddev: 600.0,
                measurement_angle_stddev: 45.0_f64.to_radians(),
                ..EkfParams::default()
            },
            start: START_POSE,
            max_cylinder_distance: 500.0,
        }
    }
}

impl SlamSetup {
    /// The scanner noise our synthetic scanner really has (as av04's T).
    pub fn sharp_scanner() -> SlamSetup {
        SlamSetup {
            filter: EkfParams {
                measurement_distance_stddev: 20.0,
                measurement_angle_stddev: 1.0_f64.to_radians(),
                ..SlamSetup::default().filter
            },
            ..SlamSetup::default()
        }
    }

    pub fn start_filter(&self) -> Slam {
        let centre = self.start.shifted(-self.filter.scanner_displacement as f32);
        Slam::new(
            DVec3::new(
                centre.x as f64,
                centre.y as f64,
                wrap_pi(centre.heading as f64),
            ),
            DMat3::ZERO,
        )
    }
}

/// A centre state to the scanner pose (f32, heading in [0, 2 pi)).
pub fn scanner_pose(state: DVec3, p: &EkfParams) -> Pose {
    Pose::new(state.x as f32, state.y as f32, wrap_angle(state.z as f32))
        .shifted(p.scanner_displacement as f32)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlamObservation {
    /// The detection in the world, placed with the predicted pose.
    pub seen_at: Vec2,
    /// The map landmark it went with.
    pub landmark: usize,
    /// It wasn't near any landmark: it started a new one.
    pub new: bool,
}

/// What one step left behind, for drawing (not the whole covariance).
#[derive(Clone, Debug, PartialEq)]
pub struct SlamStep {
    pub pose: Pose,
    pub ellipse: ErrorEllipse,
    pub heading_stddev: f64,
    /// The map after this step, and each landmark's error ellipse.
    pub landmarks: Vec<Vec2>,
    pub landmark_ellipses: Vec<ErrorEllipse>,
    pub observations: Vec<SlamObservation>,
}

pub struct SlamInputs<'a> {
    pub motion: &'a MotionParams,
    pub detector: &'a DetectorParams,
    pub setup: &'a SlamSetup,
}

/// The course's loop body: predict, then for every detection either
/// correct with its landmark or add a new one and correct with that.
pub fn slam_step(
    slam: &mut Slam,
    ticks: (i64, i64),
    scan: &[u32],
    inputs: &SlamInputs,
) -> SlamStep {
    let p = &inputs.setup.filter;
    let mm = inputs.motion.ticks_to_mm as f64;
    slam.predict(DVec2::new(ticks.0 as f64 * mm, ticks.1 as f64 * mm), p);

    // Association first, with the predicted pose and the map as it is
    // now (the course's get_observations), then the corrections.
    let scanner = scanner_pose(slam.robot(), p);
    let max2 = (inputs.setup.max_cylinder_distance as f64).powi(2);
    let found: Vec<(DVec2, Vec2, Option<usize>)> = detect(scan, inputs.detector)
        .iter()
        .map(|c| {
            let world = scanner.to_world(c.local);
            let w = DVec2::new(world.x as f64, world.y as f64);
            let nearest = (0..slam.landmark_count())
                .map(|i| (i, slam.landmark(i).distance_squared(w)))
                .filter(|&(_, d2)| d2 < max2)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i);
            let measurement = DVec2::new(c.local.length() as f64, c.local.to_angle() as f64);
            (measurement, world, nearest)
        })
        .collect();
    let observations = found
        .into_iter()
        .map(|(measurement, world, nearest)| {
            let (landmark, new) = match nearest {
                Some(i) => (i, false),
                None => (
                    slam.add_landmark(DVec2::new(world.x as f64, world.y as f64)),
                    true,
                ),
            };
            slam.correct(measurement, landmark, p);
            SlamObservation {
                seen_at: world,
                landmark,
                new,
            }
        })
        .collect();

    let n = slam.landmark_count();
    SlamStep {
        pose: scanner_pose(slam.robot(), p),
        ellipse: slam.robot_ellipse(),
        heading_stddev: slam.heading_stddev(),
        landmarks: (0..n)
            .map(|i| {
                let l = slam.landmark(i);
                Vec2::new(l.x as f32, l.y as f32)
            })
            .collect(),
        landmark_ellipses: (0..n).map(|i| slam.landmark_ellipse(i)).collect(),
        observations,
    }
}

/// How good a map is: for each true landmark, the distance to the nearest
/// map landmark; and how many map landmarks are nobody's nearest (extra:
/// spurious detections, or one cylinder mapped twice).
pub fn map_errors(map: &[Vec2], truth: &[Vec2]) -> (Vec<f32>, usize) {
    let nearest = |t: &Vec2| {
        map.iter()
            .enumerate()
            .map(|(i, m)| (i, m.distance(*t)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    };
    let mut used = vec![false; map.len()];
    let errors = truth
        .iter()
        .filter_map(|t| {
            nearest(t).map(|(i, d)| {
                used[i] = true;
                d
            })
        })
        .collect();
    (errors, used.iter().filter(|u| !**u).count())
}

// --- The plugin ----------------------------------------------------------------

#[derive(Resource, Debug, Default)]
pub struct SlamMapping {
    pub steps: Vec<SlamStep>,
    /// The filter after the last step (the next one continues from it).
    slam: Option<Slam>,
    generation: u64,
    setup: Option<SlamSetup>,
}

impl SlamMapping {
    pub fn poses(&self) -> impl Iterator<Item = Pose> + '_ {
        self.steps.iter().map(|s| s.pose)
    }
}

#[derive(Default)]
pub struct SlamPlugin;

impl Plugin for SlamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionParams>()
            .init_resource::<DetectorParams>()
            .init_resource::<SlamSetup>()
            .init_resource::<SlamMapping>()
            .add_systems(Update, update_slam.in_set(AvSet::Algorithms));
    }
}

fn update_slam(
    log: Res<RobotLog>,
    motion: Res<MotionParams>,
    detector: Res<DetectorParams>,
    setup: Res<SlamSetup>,
    mut out: ResMut<SlamMapping>,
) {
    if out.generation != log.generation || out.setup != Some(*setup) || out.steps.len() > log.len()
    {
        out.steps.clear();
        out.slam = None;
        out.generation = log.generation;
        out.setup = Some(*setup);
    }
    let inputs = SlamInputs {
        motion: &motion,
        detector: &detector,
        setup: &setup,
    };
    let out = &mut *out;
    let slam = out.slam.get_or_insert_with(|| setup.start_filter());
    let records = log.log.motors.len().min(log.log.scans.len());
    for i in out.steps.len()..records {
        let step = slam_step(
            slam,
            log.log.motor_increment(i),
            &log.log.scans[i].ranges,
            &inputs,
        );
        out.steps.push(step);
    }
}

// --- Whole runs, without Bevy (for tests and measuring) ---------------------------

pub fn slam_trajectory(log: &LogFile, setup: &SlamSetup) -> Vec<SlamStep> {
    let inputs = SlamInputs {
        motion: &MotionParams::default(),
        detector: &DetectorParams::default(),
        setup,
    };
    let mut slam = setup.start_filter();
    (0..log.motors.len().min(log.scans.len()))
        .map(|i| {
            slam_step(
                &mut slam,
                log.motor_increment(i),
                &log.scans[i].ranges,
                &inputs,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{DEFAULT_STEPS, SynthConfig, generate};
    use crate::unit_b::error_stats;

    fn truth_landmarks(log: &LogFile) -> Vec<Vec2> {
        log.landmarks.iter().map(|l| l.position).collect()
    }

    /// Over a whole run the robot stays near the truth, and the map has
    /// every arena landmark, each close to where it really is.
    #[test]
    fn it_maps_the_arena_and_follows_the_truth() {
        let log = generate(SynthConfig::REALISTIC, 1, DEFAULT_STEPS);
        for setup in [SlamSetup::default(), SlamSetup::sharp_scanner()] {
            let steps = slam_trajectory(&log, &setup);
            let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
            let (mean, worst) = error_stats(&poses, &log);
            assert!(mean < 30.0 && worst < 80.0, "mean {mean} worst {worst}");
            let last = steps.last().unwrap();
            let (errors, _extra) = map_errors(&last.landmarks, &truth_landmarks(&log));
            assert!(errors.iter().all(|&e| e < 60.0), "{errors:?}");
        }
    }

    /// The covariance stays symmetric with positive variances over a whole
    /// run with the sharp scanner (where the course's update form blew
    /// up: see the NOTE at the end of Slam::correct).
    #[test]
    fn covariance_stays_healthy() {
        let log = generate(SynthConfig::REALISTIC, 1, DEFAULT_STEPS);
        let setup = SlamSetup::sharp_scanner();
        let inputs = SlamInputs {
            motion: &MotionParams::default(),
            detector: &DetectorParams::default(),
            setup: &setup,
        };
        let mut slam = setup.start_filter();
        for i in 0..DEFAULT_STEPS {
            slam_step(
                &mut slam,
                log.motor_increment(i),
                &log.scans[i].ranges,
                &inputs,
            );
            let c = &slam.covariance;
            assert!(c.asymmetry() < 1e-6, "step {i}: {}", c.asymmetry());
            // Step 0 starts from zero covariance; after that, all positive.
            if i > 0 {
                assert!((0..c.size()).all(|k| c.get(k, k) > 0.0), "step {i}");
            }
        }
        assert_eq!(slam.landmark_count(), 6);
    }

    #[test]
    fn map_errors_counts_extras() {
        let truth = [Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)];
        let map = [
            Vec2::new(3.0, 4.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(500.0, 500.0),
        ];
        let (errors, extra) = map_errors(&map, &truth);
        assert_eq!(errors, vec![5.0, 0.0]);
        assert_eq!(extra, 1);
    }

    /// Robot error, and map error at the end, over seeds 1-5. Run with
    ///   cargo test unit_f::tests::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        for seed in 1..=5 {
            let log = generate(SynthConfig::REALISTIC, seed, DEFAULT_STEPS);
            for (name, setup) in [
                ("course", SlamSetup::default()),
                ("sharp ", SlamSetup::sharp_scanner()),
            ] {
                let steps = slam_trajectory(&log, &setup);
                let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
                let (mean, worst) = error_stats(&poses, &log);
                let last = steps.last().unwrap();
                let (errors, extra) = map_errors(&last.landmarks, &truth_landmarks(&log));
                let map_mean = errors.iter().sum::<f32>() / errors.len() as f32;
                let map_worst = errors.iter().copied().fold(0.0, f32::max);
                println!(
                    "seed {seed} {name}: robot mean {mean:5.1} worst {worst:5.1} mm | map {} landmarks ({extra} extra), mean {map_mean:5.1} worst {map_worst:5.1} mm",
                    last.landmarks.len()
                );
            }
        }
    }
}
