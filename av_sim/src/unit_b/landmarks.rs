// Unit B, part 2: correcting the pose with the landmarks (the course's
// slam_04_a to 04_d).
//
// Every step:
//   1. predict: the motion model moves the LAST CORRECTED pose by the ticks
//   2. detect the cylinders in the scan (Unit A), and place them in the
//      world with the predicted pose
//   3. pair each with the nearest map landmark, if it's close enough
//   4. estimate the similarity (scale fixed to 1) that moves the detected
//      cylinders onto their landmarks
//   5. correct the pose with it; the next step predicts from there
//
// With fewer than two pairs there is no transform, and the pose stays as
// predicted: it drifts until two cylinders are in view again.

use super::similarity::Similarity;
use crate::core::{AvSet, RobotLog};
use crate::robot::Pose;
use crate::unit_a::cylinders::{DetectorParams, detect};
use crate::unit_a::motion::{MotionParams, filter_step};
use bevy::math::Vec2;
use bevy::prelude::*;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct LandmarkParams {
    /// A detection further than this from every landmark is left unpaired.
    /// The course uses 300 in 04_c and 400 in 04_d.
    pub max_cylinder_distance: f32,
}

impl Default for LandmarkParams {
    fn default() -> LandmarkParams {
        LandmarkParams {
            max_cylinder_distance: 400.0,
        }
    }
}

/// For each cylinder, the nearest reference cylinder if it is closer than
/// max_radius: (index in cylinders, index in reference).
pub fn find_cylinder_pairs(
    cylinders: &[Vec2],
    reference: &[Vec2],
    max_radius: f32,
) -> Vec<(usize, usize)> {
    cylinders
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            reference
                .iter()
                .enumerate()
                .map(|(j, r)| (j, c.distance(*r)))
                .filter(|&(_, d)| d < max_radius)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(j, _)| (i, j))
        })
        .collect()
}

/// What one step of the landmark correction computed (the course writes
/// F, D C and W C records from this).
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkStep {
    /// The motion model's prediction, before correcting.
    pub predicted: Pose,
    /// The detected cylinders in the world, placed with `predicted`.
    pub cylinders: Vec<Vec2>,
    /// (index in cylinders, index in the map's landmarks).
    pub pairs: Vec<(usize, usize)>,
    pub transform: Option<Similarity>,
    /// The corrected pose (= predicted when there is no transform).
    pub pose: Pose,
}

pub struct LandmarkInputs<'a> {
    pub motion: &'a MotionParams,
    pub detector: &'a DetectorParams,
    pub params: &'a LandmarkParams,
    pub landmarks: &'a [Vec2],
}

pub fn landmark_step(
    previous: Pose,
    ticks: (i64, i64),
    scan: &[u32],
    inputs: &LandmarkInputs,
) -> LandmarkStep {
    let predicted = filter_step(previous, ticks, inputs.motion);
    let cylinders: Vec<Vec2> = detect(scan, inputs.detector)
        .iter()
        .map(|c| predicted.to_world(c.local))
        .collect();
    let pairs = find_cylinder_pairs(
        &cylinders,
        inputs.landmarks,
        inputs.params.max_cylinder_distance,
    );
    let left: Vec<Vec2> = pairs.iter().map(|&(i, _)| cylinders[i]).collect();
    let right: Vec<Vec2> = pairs.iter().map(|&(_, j)| inputs.landmarks[j]).collect();
    let transform = Similarity::estimate(&left, &right, true);
    let pose = transform.map_or(predicted, |t| t.correct_pose(predicted));
    LandmarkStep {
        predicted,
        cylinders,
        pairs,
        transform,
        pose,
    }
}

// --- The plugin ----------------------------------------------------------------
//
// The same pattern as Unit A's plugins, with one difference: each step
// starts from the previous step's CORRECTED pose, so the steps can only be
// computed in order. (Unit A's dead reckoning could be too, but there the
// previous pose is just the previous prediction.)

#[derive(Resource, Default, Debug)]
pub struct LandmarkCorrection {
    pub steps: Vec<LandmarkStep>,
    generation: u64,
}

impl LandmarkCorrection {
    pub fn poses(&self) -> impl Iterator<Item = Pose> + '_ {
        self.steps.iter().map(|s| s.pose)
    }
}

#[derive(Default)]
pub struct LandmarkCorrectionPlugin;

impl Plugin for LandmarkCorrectionPlugin {
    fn build(&self, app: &mut App) {
        // init_resource leaves a resource alone if it's already there, so
        // this plugin and Unit A's can both ask for MotionParams.
        app.init_resource::<MotionParams>()
            .init_resource::<DetectorParams>()
            .init_resource::<LandmarkParams>()
            .init_resource::<LandmarkCorrection>()
            .add_systems(Update, update_landmark_correction.in_set(AvSet::Algorithms));
    }
}

fn update_landmark_correction(
    log: Res<RobotLog>,
    motion: Res<MotionParams>,
    detector: Res<DetectorParams>,
    params: Res<LandmarkParams>,
    mut out: ResMut<LandmarkCorrection>,
) {
    if out.generation != log.generation
        || motion.is_changed()
        || detector.is_changed()
        || params.is_changed()
    {
        out.steps.clear();
        out.generation = log.generation;
    }
    let n = log.log.motors.len().min(log.log.scans.len());
    if out.steps.len() >= n {
        return;
    }
    let landmarks: Vec<Vec2> = log.arena.landmarks.iter().map(|l| l.position).collect();
    let inputs = LandmarkInputs {
        motion: &motion,
        detector: &detector,
        params: &params,
        landmarks: &landmarks,
    };
    for i in out.steps.len()..n {
        let previous = out.steps.last().map_or(motion.start, |s| s.pose);
        let step = landmark_step(
            previous,
            log.log.motor_increment(i),
            &log.log.scans[i].ranges,
            &inputs,
        );
        out.steps.push(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SynthConfig, generate};
    use crate::unit_b::{error_stats, landmark_trajectory};
    use crate::world::COURSE_LANDMARKS;

    #[test]
    fn pairs_take_the_nearest_landmark_within_reach() {
        let reference = [Vec2::new(0.0, 0.0), Vec2::new(500.0, 0.0)];
        let cylinders = [
            Vec2::new(300.0, 0.0),  // nearer the second
            Vec2::new(50.0, 50.0),  // the first
            Vec2::new(0.0, 900.0),  // too far from both
            Vec2::new(240.0, 10.0), // the first, just
        ];
        assert_eq!(
            find_cylinder_pairs(&cylinders, &reference, 400.0),
            vec![(0, 1), (1, 0), (3, 0)]
        );
        // Index 0 is a real landmark index (the repo's `closest_cyl == []`
        // test has to tell 0 from "none").
        assert_eq!(
            find_cylinder_pairs(&[Vec2::ZERO], &reference, 10.0),
            vec![(0, 0)]
        );
    }

    /// The point of Unit B: on the realistic robot, correcting with the
    /// landmarks keeps the pose near the truth, where dead reckoning drifts.
    #[test]
    fn landmarks_keep_the_pose_near_the_truth() {
        let log = generate(SynthConfig::REALISTIC, 1, 278);
        let steps = landmark_trajectory(&log);
        assert_eq!(steps.len(), 278);
        let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
        let (mean, worst) = error_stats(&poses, &log);
        // Measured (see the measure test in unit_b/mod.rs): mean 1.6,
        // worst 5.5 mm, where dead reckoning is 114 mm off at the end.
        assert!(mean < 5.0 && worst < 20.0, "mean {mean}, worst {worst}");
        // Most steps see two or more cylinders and get a transform.
        let corrected = steps.iter().filter(|s| s.transform.is_some()).count();
        // (Measured: 230 of 278.)
        assert!(corrected > 200, "{corrected} corrected steps");
        // ... and the pairs point at real landmarks.
        assert!(
            steps
                .iter()
                .flat_map(|s| &s.pairs)
                .all(|&(_, j)| j < COURSE_LANDMARKS.len())
        );
    }
}
