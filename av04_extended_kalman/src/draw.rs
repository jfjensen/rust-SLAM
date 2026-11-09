// Part 3: drawing, all with gizmos (as in av01 and av02). Only reads: the
// log, the algorithms' results, the playhead.
//
//   3a  the arena: walls, landmarks
//   3b  the paths: truth (green), dead reckoning (orange), Kalman (purple)
//   3c  the filter at the current step (V):
//         the 95% ellipse after predicting (blue) and after correcting
//         (purple), the heading's 95% wedge, the detections placed with
//         the predicted pose (cyan) and a line from the robot to each
//         landmark it corrected with (yellow); with Trail, also the
//         corrected ellipse every TRAIL_EVERY steps
//   3d  the plots: distance from the truth, and the ellipse's size

use crate::layout::*;
use crate::{Detail, Layers};
use av_sim::core::{AvSet, Playhead, RobotLog};
use av_sim::robot::Pose;
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_d::ekf::ErrorEllipse;
use av_sim::unit_d::{EkfLocalization, EkfStep};
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
/// Trail: an ellipse every this many steps.
const TRAIL_EVERY: usize = 10;
/// Length of the heading wedge's sides, pixels.
const WEDGE: f32 = 40.0;

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
    kalman: Res<EkfLocalization>,
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
    let poses: Vec<Pose> = kalman.poses().collect();
    path(&poses, KALMAN);
    // The truth last, on top: the filter's path is often right under it.
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

/// The 95% ellipse around a (world) centre.
fn ellipse(gizmos: &mut Gizmos, centre: Vec2, e: &ErrorEllipse, colour: Color) {
    let half = Vec2::new(e.major as f32, e.minor as f32) * ELLIPSE_SIGMAS * ARENA_SCALE;
    gizmos.ellipse_2d(
        Isometry2d::new(world_to_screen(centre), Rot2::radians(e.angle as f32)),
        half,
        colour,
    );
}

fn draw_detail(
    log: Res<RobotLog>,
    kalman: Res<EkfLocalization>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    if layers.detail == Detail::Off {
        return;
    }
    let i = playhead.index;
    if layers.detail == Detail::Trail {
        for step in kalman.steps.iter().take(i).step_by(TRAIL_EVERY) {
            ellipse(&mut gizmos, step.pose.position(), &step.ellipse(), TRAIL);
        }
    }
    let Some(step) = kalman.steps.get(i) else {
        return;
    };
    draw_step(&mut gizmos, &log, step);
}

fn draw_step(gizmos: &mut Gizmos, log: &RobotLog, step: &EkfStep) {
    // Each ellipse around its own pose. (The ellipse is the centre's
    // uncertainty; it's drawn around the scanner, as the course's viewer
    // does: the two differ only by the heading's share, a few mm.)
    ellipse(
        gizmos,
        step.predicted_pose.position(),
        &step.predicted_ellipse(),
        PREDICTED,
    );
    ellipse(gizmos, step.pose.position(), &step.ellipse(), KALMAN);

    // The heading wedge: two sigma either side.
    let c = world_to_screen(step.pose.position());
    let spread = HEADING_SIGMAS * step.heading_stddev() as f32;
    for side in [-spread, spread] {
        let dir = Vec2::from_angle(step.pose.heading + side);
        gizmos.line_2d(c, c + dir * WEDGE, KALMAN);
    }

    for o in &step.observations {
        let landmark = log.arena.landmarks[o.landmark].position;
        cross(gizmos, world_to_screen(o.seen_at), DETECTION);
        gizmos.line_2d(c, world_to_screen(landmark), PAIR);
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
    kalman: Res<EkfLocalization>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let steps = log.len();
    let errors_plot = ERROR_PLOT.with_steps(steps);
    let sizes_plot = ELLIPSE_PLOT.with_steps(steps);
    for (plot, grid) in [(&errors_plot, ERROR_GRID), (&sizes_plot, ELLIPSE_GRID)] {
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
    errors(&mut kalman.poses(), KALMAN);

    // The ellipse's long half axis (one sigma), predicted and corrected.
    let mut sizes = |of: &dyn Fn(&EkfStep) -> ErrorEllipse, colour: Color| {
        gizmos.linestrip_2d(
            kalman
                .steps
                .iter()
                .enumerate()
                .map(|(i, s)| sizes_plot.to_screen(i as f32, of(s).major as f32)),
            colour,
        );
    };
    sizes(&|s| s.predicted_ellipse(), PREDICTED);
    sizes(&|s| s.ellipse(), KALMAN);
}
