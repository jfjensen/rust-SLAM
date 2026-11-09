// Unit B, part 3: correcting the pose with the walls, by ICP (the course's
// slam_05_a to 05_c).
//
// The landmarks are only six points. The walls are seen in almost every
// scan, but a wall point has no partner of its own: any point along the
// wall would do. ICP (iterative closest point) gets around that:
//
//   1. take every 10th scan point, in the world with the predicted pose
//   2. pair each point within eps of a wall with its closest point ON the
//      wall (drop it straight onto the wall)
//   3. estimate the similarity for those pairs, and move the points
//   4. repeat 2-3: with the points moved, the pairs are better
//
// The transforms add up (`after`), and the total corrects the pose.

use super::similarity::Similarity;
use crate::core::{AvSet, RobotLog};
use crate::robot::{Pose, beam_index_to_angle};
use crate::unit_a::motion::{MotionParams, filter_step};
use bevy::math::Vec2;
use bevy::prelude::*;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WallParams {
    /// Take every n-th beam.
    pub sampling: usize,
    /// Points further than this from every wall take no part.
    pub eps: f32,
    pub iterations: usize,
    /// Ranges at or below this are "no echo" (as the cylinder detector's).
    pub min_valid_distance: f32,
}

impl Default for WallParams {
    fn default() -> WallParams {
        // The course's values.
        WallParams {
            sampling: 10,
            eps: 150.0,
            iterations: 40,
            min_valid_distance: 20.0,
        }
    }
}

/// Every `sampling`-th beam as a point in the scanner's frame.
///
/// NOTE: the course's get_subsampled_points keeps the no-echo beams too
/// (range 0: a point AT the scanner). Near a wall those get paired with
/// it and pull the pose. Here they are left out.
pub fn subsample(scan: &[u32], p: &WallParams) -> Vec<Vec2> {
    scan.iter()
        .enumerate()
        .step_by(p.sampling.max(1))
        .filter(|&(_, &r)| r as f32 > p.min_valid_distance)
        .map(|(i, &r)| Vec2::from_angle(beam_index_to_angle(i as f32)) * r as f32)
        .collect()
}

/// For each point within eps of a wall (x = 0, x = size.x, y = 0 or
/// y = size.y, tried in that order), the point and its foot on that wall.
/// Returns (points, feet): the left and right lists for estimate.
pub fn wall_pairs(points: &[Vec2], size: Vec2, eps: f32) -> (Vec<Vec2>, Vec<Vec2>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &p in points {
        let foot = if p.x.abs() < eps {
            Vec2::new(0.0, p.y)
        } else if (p.x - size.x).abs() < eps {
            Vec2::new(size.x, p.y)
        } else if p.y.abs() < eps {
            Vec2::new(p.x, 0.0)
        } else if (p.y - size.y).abs() < eps {
            Vec2::new(p.x, size.y)
        } else {
            continue;
        };
        left.push(p);
        right.push(foot);
    }
    (left, right)
}

/// The course's get_icp_transform: the transform that moves the points
/// onto the walls.
pub fn icp(points: &[Vec2], size: Vec2, p: &WallParams) -> Similarity {
    let mut total = Similarity::IDENTITY;
    for _ in 0..p.iterations {
        let moved: Vec<Vec2> = points.iter().map(|&q| total.apply(q)).collect();
        let (left, right) = wall_pairs(&moved, size, p.eps);
        // NOTE: the repo's solution sets the total back to the identity
        // when an estimate fails, which throws away the iterations so far.
        // Here we stop and keep them.
        let Some(step) = Similarity::estimate(&left, &right, true) else {
            break;
        };
        total = step.after(&total);
    }
    total
}

#[derive(Clone, Debug, PartialEq)]
pub struct WallStep {
    pub predicted: Pose,
    /// The subsampled scan points in the world, placed with `predicted`.
    pub points: Vec<Vec2>,
    /// What ICP found (the identity when no points are near a wall).
    pub transform: Similarity,
    pub pose: Pose,
}

pub fn wall_step(
    previous: Pose,
    ticks: (i64, i64),
    scan: &[u32],
    size: Vec2,
    motion: &MotionParams,
    params: &WallParams,
) -> WallStep {
    let predicted = filter_step(previous, ticks, motion);
    let points: Vec<Vec2> = subsample(scan, params)
        .into_iter()
        .map(|q| predicted.to_world(q))
        .collect();
    let transform = icp(&points, size, params);
    WallStep {
        predicted,
        points,
        transform,
        pose: transform.correct_pose(predicted),
    }
}

// --- The plugin ----------------------------------------------------------------

#[derive(Resource, Default, Debug)]
pub struct WallCorrection {
    pub steps: Vec<WallStep>,
    generation: u64,
}

#[derive(Default)]
pub struct WallCorrectionPlugin;

impl Plugin for WallCorrectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionParams>()
            .init_resource::<WallParams>()
            .init_resource::<WallCorrection>()
            .add_systems(Update, update_wall_correction.in_set(AvSet::Algorithms));
    }
}

fn update_wall_correction(
    log: Res<RobotLog>,
    motion: Res<MotionParams>,
    params: Res<WallParams>,
    mut out: ResMut<WallCorrection>,
) {
    if out.generation != log.generation || motion.is_changed() || params.is_changed() {
        out.steps.clear();
        out.generation = log.generation;
    }
    let n = log.log.motors.len().min(log.log.scans.len());
    for i in out.steps.len()..n {
        let previous = out.steps.last().map_or(motion.start, |s| s.pose);
        let step = wall_step(
            previous,
            log.log.motor_increment(i),
            &log.log.scans[i].ranges,
            log.arena.size,
            &motion,
            &params,
        );
        out.steps.push(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SynthConfig, generate};
    use crate::unit_b::{error_stats, wall_trajectory};

    const SIZE: Vec2 = Vec2::splat(2000.0);

    #[test]
    fn points_drop_onto_the_nearest_wall() {
        let points = [
            Vec2::new(40.0, 700.0),    // left wall
            Vec2::new(1000.0, 1930.0), // top wall
            Vec2::new(1000.0, 1000.0), // the middle: no wall
            Vec2::new(2100.0, 300.0),  // outside, near the right wall
        ];
        let (left, right) = wall_pairs(&points, SIZE, 150.0);
        assert_eq!(left, vec![points[0], points[1], points[3]]);
        assert_eq!(
            right,
            vec![
                Vec2::new(0.0, 700.0),
                Vec2::new(1000.0, 2000.0),
                Vec2::new(2000.0, 300.0)
            ]
        );
    }

    #[test]
    fn no_echo_beams_are_left_out() {
        let mut scan = vec![500; 660];
        scan[10] = 0;
        let p = WallParams::default();
        let points = subsample(&scan, &p);
        // Beams 0, 10, ..., 650 is 66 beams; beam 10 has no echo.
        assert_eq!(points.len(), 65);
        assert!(points.iter().all(|q| (q.length() - 500.0).abs() < 1e-3));
    }

    /// Points along two walls, shifted and turned a little: ICP brings
    /// them back, which one estimate alone can't (its pairs are off).
    #[test]
    fn icp_undoes_a_small_misalignment() {
        let mut truth = Vec::new();
        for i in 1..20 {
            let t = i as f32 * 100.0;
            truth.push(Vec2::new(t, 0.0)); // bottom wall
            truth.push(Vec2::new(0.0, t)); // left wall
        }
        let error = Similarity {
            scale: 1.0,
            rotation: Vec2::from_angle(0.03),
            translation: Vec2::new(40.0, -25.0),
        };
        let seen: Vec<Vec2> = truth.iter().map(|&q| error.apply(q)).collect();
        let p = WallParams::default();
        let worst = |t: &Similarity| {
            seen.iter()
                .zip(&truth)
                .map(|(&s, &q)| t.apply(s).distance(q))
                .fold(0.0, f32::max)
        };
        let one = Similarity::estimate(
            &wall_pairs(&seen, SIZE, p.eps).0,
            &wall_pairs(&seen, SIZE, p.eps).1,
            true,
        )
        .unwrap();
        let fixed = icp(&seen, SIZE, &p);
        assert!(worst(&fixed) < 1.0, "icp leaves {}", worst(&fixed));
        assert!(
            worst(&one) > 5.0 * worst(&fixed),
            "one step {}",
            worst(&one)
        );
    }

    #[test]
    fn walls_keep_the_pose_near_the_truth() {
        let log = generate(SynthConfig::REALISTIC, 1, 278);
        let steps = wall_trajectory(&log);
        let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
        let (mean, worst) = error_stats(&poses, &log);
        // Measured (see the measure test in unit_b/mod.rs): mean 8.7,
        // worst 39 mm. Less exact than the landmarks: the walls only say
        // how far the robot is from each, and a corner is often out of view.
        assert!(mean < 20.0 && worst < 80.0, "mean {mean}, worst {worst}");
    }
}
