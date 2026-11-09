// Unit D of the course: localisation with the extended Kalman filter.
//
//   ekf.rs   the filter itself: motion and measurement models, their
//            Jacobians, predict, correct, the error ellipse
//   here     one filter step on one log record (ticks + scan), the plugin,
//            whole runs for tests
//
// Unlike Unit B's corrections, the filter knows how sure it is. It
// doesn't jump to where the landmarks say: it weighs them against its own
// prediction, landmark by landmark, by their covariances. And with no
// landmark in view it still carries on, only less sure.
//
// The filter's state is the robot's CENTRE (as in the course). Everything
// it hands out (EkfStep::pose) is the SCANNER pose, like the rest of
// av_sim and like the course's F records.

pub mod ekf;

use crate::core::{AvSet, RobotLog};
use crate::logfile::LogFile;
use crate::robot::{Pose, START_POSE, wrap_angle};
use crate::unit_a::cylinders::{DetectorParams, detect};
use crate::unit_a::motion::MotionParams;
use crate::unit_b::landmarks::find_cylinder_pairs;
use bevy::app::plugin_group;
use bevy::math::{DMat3, DVec2, DVec3};
use bevy::prelude::*;
use ekf::{Ekf, EkfParams, ErrorEllipse, error_ellipse};

plugin_group! {
    /// Unit D's filter.
    pub struct UnitDPlugins {
        :EkfPlugin,
    }
}

/// How sure the filter is of the start pose, and how far a detection may
/// be from a landmark to count as seeing it (the course's slam_07_f).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct EkfSetup {
    pub filter: EkfParams,
    pub start: Pose,
    pub start_position_stddev: f64,
    pub start_heading_stddev: f64,
    pub max_cylinder_distance: f32,
}

impl Default for EkfSetup {
    fn default() -> EkfSetup {
        EkfSetup {
            filter: EkfParams::default(),
            start: START_POSE,
            start_position_stddev: 100.0,
            start_heading_stddev: 10.0_f64.to_radians(),
            max_cylinder_distance: 300.0,
        }
    }
}

impl EkfSetup {
    /// The course's settings, but with the scanner noise our synthetic
    /// scanner really has (20 mm, 1 degree; the course's 200 mm and 15
    /// degrees were for the real LEGO robot). The control noise stays:
    /// lowering it makes the filter worse AND over-confident, because the
    /// model's errors aren't noise but bias (ticks_to_mm 0.349 vs 0.352,
    /// width 150 vs 154): the large control noise is what covers them.
    pub fn sharp_scanner() -> EkfSetup {
        EkfSetup {
            filter: EkfParams {
                measurement_distance_stddev: 20.0,
                measurement_angle_stddev: 1.0_f64.to_radians(),
                ..EkfParams::default()
            },
            ..EkfSetup::default()
        }
    }

    /// The filter before the first step: the start pose (a scanner pose)
    /// moved back to the centre.
    pub fn start_filter(&self) -> Ekf {
        let centre = self.start.shifted(-self.filter.scanner_displacement as f32);
        let s = self.start_position_stddev;
        Ekf {
            state: DVec3::new(
                centre.x as f64,
                centre.y as f64,
                ekf::wrap_pi(centre.heading as f64),
            ),
            covariance: DMat3::from_diagonal(DVec3::new(
                s * s,
                s * s,
                self.start_heading_stddev.powi(2),
            )),
        }
    }
}

/// The scanner pose for a filter state (f32, heading in [0, 2 pi) as in
/// the rest of av_sim).
pub fn scanner_pose(ekf: &Ekf, p: &EkfParams) -> Pose {
    Pose::new(
        ekf.state.x as f32,
        ekf.state.y as f32,
        wrap_angle(ekf.state.z as f32),
    )
    .shifted(p.scanner_displacement as f32)
}

/// One landmark the filter corrected with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    /// (range, bearing) from the scanner, as detected.
    pub measurement: DVec2,
    /// Index into the log's landmarks.
    pub landmark: usize,
    /// Where the detection was in the world, placed with the PREDICTED
    /// pose (that's what it was matched with).
    pub seen_at: Vec2,
    /// Measurement minus expected measurement, when it was applied.
    pub innovation: DVec2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EkfStep {
    /// After predict, before any correction.
    pub predicted: Ekf,
    /// The scanner pose of `predicted`.
    pub predicted_pose: Pose,
    pub observations: Vec<Observation>,
    /// After all the corrections.
    pub ekf: Ekf,
    /// The scanner pose of `ekf`.
    pub pose: Pose,
}

impl EkfStep {
    pub fn ellipse(&self) -> ErrorEllipse {
        error_ellipse(&self.ekf.covariance)
    }

    pub fn predicted_ellipse(&self) -> ErrorEllipse {
        error_ellipse(&self.predicted.covariance)
    }

    /// The heading's standard deviation, radians.
    pub fn heading_stddev(&self) -> f64 {
        self.ekf.covariance.z_axis.z.max(0.0).sqrt()
    }
}

pub struct EkfInputs<'a> {
    pub motion: &'a MotionParams,
    pub detector: &'a DetectorParams,
    pub setup: &'a EkfSetup,
    pub landmarks: &'a [Vec2],
}

/// The course's filter loop body: predict with the ticks, then correct
/// with every detected cylinder that is near a landmark.
pub fn ekf_step(previous: Ekf, ticks: (i64, i64), scan: &[u32], inputs: &EkfInputs) -> EkfStep {
    let p = &inputs.setup.filter;
    let mm = inputs.motion.ticks_to_mm as f64;
    let mut ekf = previous;
    ekf.predict(DVec2::new(ticks.0 as f64 * mm, ticks.1 as f64 * mm), p);
    let predicted = ekf;

    let scanner = scanner_pose(&predicted, p);
    let cylinders = detect(scan, inputs.detector);
    let world: Vec<Vec2> = cylinders
        .iter()
        .map(|c| scanner.to_world(c.local))
        .collect();
    let pairs = find_cylinder_pairs(&world, inputs.landmarks, inputs.setup.max_cylinder_distance);
    let observations = pairs
        .into_iter()
        .map(|(c, l)| {
            let local = cylinders[c].local;
            let measurement = DVec2::new(local.length() as f64, local.to_angle() as f64);
            let landmark = inputs.landmarks[l];
            let innovation = ekf.correct(
                measurement,
                DVec2::new(landmark.x as f64, landmark.y as f64),
                p,
            );
            Observation {
                measurement,
                landmark: l,
                seen_at: world[c],
                innovation,
            }
        })
        .collect();
    EkfStep {
        predicted,
        predicted_pose: scanner,
        observations,
        ekf,
        pose: scanner_pose(&ekf, p),
    }
}

// --- The plugin ----------------------------------------------------------------

#[derive(Resource, Debug, Default)]
pub struct EkfLocalization {
    pub steps: Vec<EkfStep>,
    generation: u64,
    setup: Option<EkfSetup>,
}

impl EkfLocalization {
    pub fn poses(&self) -> impl Iterator<Item = Pose> + '_ {
        self.steps.iter().map(|s| s.pose)
    }
}

#[derive(Default)]
pub struct EkfPlugin;

impl Plugin for EkfPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionParams>()
            .init_resource::<DetectorParams>()
            .init_resource::<EkfSetup>()
            .init_resource::<EkfLocalization>()
            .add_systems(Update, update_ekf.in_set(AvSet::Algorithms));
    }
}

/// Start over for a new log or new settings; then filter the new records.
fn update_ekf(
    log: Res<RobotLog>,
    motion: Res<MotionParams>,
    detector: Res<DetectorParams>,
    setup: Res<EkfSetup>,
    mut out: ResMut<EkfLocalization>,
) {
    if out.generation != log.generation || out.setup != Some(*setup) || out.steps.len() > log.len()
    {
        out.steps.clear();
        out.generation = log.generation;
        out.setup = Some(*setup);
    }
    let landmarks: Vec<Vec2> = log.arena.landmarks.iter().map(|l| l.position).collect();
    let inputs = EkfInputs {
        motion: &motion,
        detector: &detector,
        setup: &setup,
        landmarks: &landmarks,
    };
    let records = log.log.motors.len().min(log.log.scans.len());
    for i in out.steps.len()..records {
        let previous = out
            .steps
            .last()
            .map_or_else(|| setup.start_filter(), |s| s.ekf);
        let step = ekf_step(
            previous,
            log.log.motor_increment(i),
            &log.log.scans[i].ranges,
            &inputs,
        );
        out.steps.push(step);
    }
}

// --- Whole runs, without Bevy (for tests and measuring) ---------------------------

pub fn ekf_trajectory(log: &LogFile, setup: &EkfSetup) -> Vec<EkfStep> {
    let landmarks: Vec<Vec2> = log.landmarks.iter().map(|l| l.position).collect();
    let inputs = EkfInputs {
        motion: &MotionParams::default(),
        detector: &DetectorParams::default(),
        setup,
        landmarks: &landmarks,
    };
    let mut ekf = setup.start_filter();
    (0..log.motors.len().min(log.scans.len()))
        .map(|i| {
            let step = ekf_step(ekf, log.motor_increment(i), &log.scans[i].ranges, &inputs);
            ekf = step.ekf;
            step
        })
        .collect()
}

/// How far the true (scanner) position is from the estimate, in the
/// filter's own standard deviations (the Mahalanobis distance, squared).
/// For a filter that is honest about itself, this is under 5.99 about 95%
/// of the time (chi-square, two degrees of freedom).
pub fn surprise(step: &EkfStep, truth: Vec2) -> f64 {
    let d = DVec2::new(
        (truth.x - step.pose.x) as f64,
        (truth.y - step.pose.y) as f64,
    );
    let c = step.ekf.covariance;
    let cov = bevy::math::DMat2::from_cols(c.x_axis.truncate(), c.y_axis.truncate());
    d.dot(cov.inverse() * d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{DEFAULT_STEPS, SynthConfig, generate};
    use crate::unit_b::error_stats;

    /// On a synthetic run the filter stays close to the truth, and the
    /// truth stays inside its 95% ellipse nearly always.
    #[test]
    fn the_filter_follows_the_truth() {
        let log = generate(SynthConfig::REALISTIC, 1, DEFAULT_STEPS);
        let steps = ekf_trajectory(&log, &EkfSetup::default());
        assert_eq!(steps.len(), DEFAULT_STEPS);
        let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
        let (mean, worst) = error_stats(&poses, &log);
        assert!(mean < 15.0 && worst < 60.0, "mean {mean} worst {worst}");
        let inside = steps
            .iter()
            .zip(&log.reference)
            .filter(|(s, r)| surprise(s, Vec2::new(r.x as f32, r.y as f32)) < 5.99)
            .count();
        assert!(inside as f64 > 0.9 * steps.len() as f64, "{inside}");
        // It sees landmarks on most steps.
        let seeing = steps.iter().filter(|s| !s.observations.is_empty()).count();
        assert!(seeing > steps.len() / 2, "{seeing}");
    }

    /// With no landmarks it only predicts: it drifts like dead reckoning,
    /// and its ellipse grows step by step.
    #[test]
    fn without_landmarks_the_ellipse_only_grows() {
        let mut log = generate(SynthConfig::REALISTIC, 1, 60);
        log.landmarks.clear();
        let steps = ekf_trajectory(&log, &EkfSetup::default());
        assert!(steps.iter().all(|s| s.observations.is_empty()));
        let areas: Vec<f64> = steps
            .iter()
            .map(|s| s.ekf.covariance.determinant())
            .collect();
        assert!(areas.windows(2).all(|w| w[1] >= w[0] * 0.999), "{areas:?}");
    }

    /// Mean and worst distance from the truth, and how often the truth is
    /// inside the 95% ellipse, over seeds 1-5. Run with
    ///   cargo test unit_d::tests::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        let tuned = EkfSetup::sharp_scanner();
        for seed in 1..=5 {
            let log = generate(SynthConfig::REALISTIC, seed, DEFAULT_STEPS);
            for (name, setup) in [("course", EkfSetup::default()), ("sharp ", tuned)] {
                let steps = ekf_trajectory(&log, &setup);
                let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
                let (mean, worst) = error_stats(&poses, &log);
                let inside = steps
                    .iter()
                    .zip(&log.reference)
                    .filter(|(s, r)| surprise(s, Vec2::new(r.x as f32, r.y as f32)) < 5.99)
                    .count();
                let major: f64 =
                    steps.iter().map(|s| s.ellipse().major).sum::<f64>() / steps.len() as f64;
                println!(
                    "seed {seed} {name}: mean {mean:5.1} worst {worst:5.1} mm, inside 95% ellipse {:4.1}%, mean major axis {major:5.1} mm",
                    100.0 * inside as f64 / steps.len() as f64
                );
            }
        }
    }
}
