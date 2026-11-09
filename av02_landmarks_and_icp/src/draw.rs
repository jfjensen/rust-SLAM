// Part 3: drawing, all with gizmos (as in av01). Only reads: the log, the
// algorithms' results, the playhead.
//
//   3a  the arena: walls, landmarks
//   3b  the paths: truth (green), dead reckoning (orange), corrected by
//       the landmarks (magenta) and by the walls (blue)
//   3c  the current step in detail (V):
//         landmarks  detections placed with the PREDICTED pose (cyan), a
//                    line to the landmark each is paired with (yellow), and
//                    where the transform moves them (magenta rings)
//         walls      scan points placed with the predicted pose (grey),
//                    after ICP (blue), and a line from each point near a
//                    wall to its foot on the wall (yellow)
//   3d  the plots: distance from the truth, and size of each correction

use crate::layout::*;
use crate::{Detail, Layers};
use av_sim::core::{AvSet, Playhead, RobotLog};
use av_sim::robot::Pose;
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_b::landmarks::LandmarkCorrection;
use av_sim::unit_b::walls::{WallCorrection, WallParams, wall_pairs};
use bevy::prelude::*;

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (draw_arena, draw_paths, draw_detail, draw_plots).in_set(AvSet::Present),
        );
    }
}

/// Robot size on screen (it's 150 mm wide, too small to see well).
const ROBOT_SCREEN_RADIUS: f32 = 9.0;
const CROSS: f32 = 5.0;

// --- 3a ------------------------------------------------------------------------

fn draw_arena(log: Res<RobotLog>, mut gizmos: Gizmos) {
    let arena = &log.arena;
    let a = world_to_screen(Vec2::ZERO);
    let b = world_to_screen(arena.size);
    gizmos.rect_2d((a + b) / 2.0, b - a, FRAME);
    for l in &arena.landmarks {
        gizmos.circle_2d(
            world_to_screen(l.position),
            l.diameter / 2.0 * ARENA_SCALE,
            LANDMARK,
        );
    }
}

// --- 3b ------------------------------------------------------------------------

/// The truth's positions, as floats.
fn truth(log: &RobotLog) -> impl Iterator<Item = Vec2> + '_ {
    log.log
        .reference
        .iter()
        .map(|r| Vec2::new(r.x as f32, r.y as f32))
}

fn draw_paths(
    log: Res<RobotLog>,
    dr: Res<DeadReckoning>,
    by_landmarks: Res<LandmarkCorrection>,
    by_walls: Res<WallCorrection>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let i = playhead.index;
    let upto = i + 1;
    let mut path = |poses: &[Pose], colour: Color| {
        let shown = &poses[..upto.min(poses.len())];
        gizmos.linestrip_2d(shown.iter().map(|p| world_to_screen(p.position())), colour);
        if let Some(pose) = poses.get(i) {
            draw_robot(&mut gizmos, pose, colour);
        }
    };
    if layers.dead_reckoning {
        path(&dr.poses, BELIEF);
    }
    let landmark_poses: Vec<Pose> = by_landmarks.poses().collect();
    path(&landmark_poses, BY_LANDMARKS);
    let wall_poses: Vec<Pose> = by_walls.steps.iter().map(|s| s.pose).collect();
    path(&wall_poses, BY_WALLS);
    // The truth last, on top: the landmark path is often right under it.
    gizmos.linestrip_2d(truth(&log).take(upto).map(world_to_screen), TRUTH);
    if let Some(p) = truth(&log).nth(i) {
        // P records have no heading: a circle only.
        gizmos.circle_2d(world_to_screen(p), ROBOT_SCREEN_RADIUS, TRUTH);
    }
}

/// A triangle pointing along the heading.
fn draw_robot(gizmos: &mut Gizmos, pose: &Pose, colour: Color) {
    let c = world_to_screen(pose.position());
    let f = pose.forward();
    let side = f.perp();
    let r = ROBOT_SCREEN_RADIUS;
    let nose = c + f * r * 1.4;
    let left = c - f * r + side * r * 0.8;
    let right = c - f * r - side * r * 0.8;
    gizmos.linestrip_2d([nose, left, right, nose], colour);
}

// --- 3c ------------------------------------------------------------------------

fn draw_detail(
    log: Res<RobotLog>,
    by_landmarks: Res<LandmarkCorrection>,
    by_walls: Res<WallCorrection>,
    wall_params: Res<WallParams>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let i = playhead.index;
    match layers.detail {
        Detail::Landmarks => {
            let Some(step) = by_landmarks.steps.get(i) else {
                return;
            };
            for &c in &step.cylinders {
                cross(&mut gizmos, world_to_screen(c), DETECTION);
            }
            for &(c, l) in &step.pairs {
                let landmark = log.arena.landmarks[l].position;
                gizmos.line_2d(
                    world_to_screen(step.cylinders[c]),
                    world_to_screen(landmark),
                    PAIR,
                );
                if let Some(t) = step.transform {
                    gizmos.circle_2d(
                        world_to_screen(t.apply(step.cylinders[c])),
                        4.0,
                        BY_LANDMARKS,
                    );
                }
            }
        }
        Detail::Walls => {
            let Some(step) = by_walls.steps.get(i) else {
                return;
            };
            for &p in &step.points {
                gizmos.rect_2d(world_to_screen(p), Vec2::splat(2.0), RAW_POINT);
            }
            let moved: Vec<Vec2> = step
                .points
                .iter()
                .map(|&p| step.transform.apply(p))
                .collect();
            for &p in &moved {
                gizmos.rect_2d(world_to_screen(p), Vec2::splat(3.0), BY_WALLS);
            }
            let (near, feet) = wall_pairs(&moved, log.arena.size, wall_params.eps);
            for (p, foot) in near.into_iter().zip(feet) {
                gizmos.line_2d(world_to_screen(p), world_to_screen(foot), PAIR);
            }
        }
        Detail::Off => {}
    }
}

fn cross(gizmos: &mut Gizmos, p: Vec2, colour: Color) {
    gizmos.line_2d(p - Vec2::splat(CROSS), p + Vec2::splat(CROSS), colour);
    gizmos.line_2d(
        p + Vec2::new(-CROSS, CROSS),
        p + Vec2::new(CROSS, -CROSS),
        colour,
    );
}

// --- 3d ------------------------------------------------------------------------

fn draw_plots(
    log: Res<RobotLog>,
    dr: Res<DeadReckoning>,
    by_landmarks: Res<LandmarkCorrection>,
    by_walls: Res<WallCorrection>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let steps = log.len();
    let errors_plot = ERROR_PLOT.with_steps(steps);
    let fixes_plot = CORRECTION_PLOT.with_steps(steps);
    for (plot, grid) in [(&errors_plot, ERROR_GRID), (&fixes_plot, CORRECTION_GRID)] {
        gizmos.rect_2d(plot.centre(), plot.size, FRAME);
        let mut y = grid;
        while y < plot.y_range.1 {
            gizmos.line_2d(
                plot.to_screen(0.0, y),
                plot.to_screen(plot.x_range.1, y),
                GRID,
            );
            y += grid;
        }
        // The playhead.
        let x = playhead.index as f32;
        gizmos.line_2d(
            plot.to_screen(x, plot.y_range.0),
            plot.to_screen(x, plot.y_range.1),
            FRAME,
        );
    }

    // Distance from the truth, step by step.
    let truth: Vec<Vec2> = truth(&log).collect();
    let mut errors = |poses: &mut dyn Iterator<Item = Pose>, colour: Color| {
        gizmos.linestrip_2d(
            poses
                .zip(&truth)
                .enumerate()
                .map(|(i, (p, t))| errors_plot.to_screen(i as f32, p.position().distance(*t))),
            colour,
        );
    };
    if layers.dead_reckoning {
        errors(&mut dr.poses.iter().copied(), BELIEF);
    }
    errors(&mut by_landmarks.poses(), BY_LANDMARKS);
    errors(&mut by_walls.steps.iter().map(|s| s.pose), BY_WALLS);

    // How far each step's correction moved the pose: predicted -> corrected.
    let moved = |predicted: Pose, pose: Pose| predicted.position().distance(pose.position());
    gizmos.linestrip_2d(
        by_landmarks
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| fixes_plot.to_screen(i as f32, moved(s.predicted, s.pose))),
        BY_LANDMARKS,
    );
    gizmos.linestrip_2d(
        by_walls
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| fixes_plot.to_screen(i as f32, moved(s.predicted, s.pose))),
        BY_WALLS,
    );
}
