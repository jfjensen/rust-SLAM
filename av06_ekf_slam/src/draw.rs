// Part 3: drawing, all with gizmos (as in av01..av05). Only reads: the
// log, the algorithms' results, the playhead.
//
//   3a  the arena: walls, the REAL landmarks (grey; SLAM doesn't get them)
//   3b  the paths: truth (green), dead reckoning (orange), Kalman with
//       the map given (blue), SLAM (purple)
//   3c  SLAM's map and this step (V):
//         Step  each map landmark as a cross with its 95% ellipse, the
//               robot's 95% ellipse and heading wedge, a yellow line to
//               each landmark corrected with, a cyan ring where a
//               detection started a new landmark
//         Map   the crosses only
//   3d  the plots: distance from the truth, the map's error

use crate::layout::*;
use crate::{Detail, Layers};
use av_sim::core::{AvSet, Playhead, RobotLog};
use av_sim::robot::Pose;
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_d::EkfLocalization;
use av_sim::unit_d::ekf::ErrorEllipse;
use av_sim::unit_f::{SlamMapping, SlamStep};
use bevy::prelude::*;

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (draw_arena, draw_paths, draw_map, draw_plots).in_set(AvSet::Present),
        );
    }
}

/// Robot size on screen (it's 150 mm wide, too small to see well).
const ROBOT_SCREEN_RADIUS: f32 = 9.0;
const CROSS: f32 = 6.0;
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
    slam: Res<SlamMapping>,
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
    if layers.kalman {
        let poses: Vec<Pose> = kalman.poses().collect();
        path(&poses, KALMAN);
    }
    let poses: Vec<Pose> = slam.poses().collect();
    path(&poses, SLAM);
    // The truth last, on top: the filters' paths are often right under it.
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

fn cross(gizmos: &mut Gizmos, p: Vec2, colour: Color) {
    gizmos.line_2d(p - Vec2::X * CROSS, p + Vec2::X * CROSS, colour);
    gizmos.line_2d(p - Vec2::Y * CROSS, p + Vec2::Y * CROSS, colour);
}

fn draw_map(
    slam: Res<SlamMapping>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    if layers.detail == Detail::Off {
        return;
    }
    let Some(step) = slam.steps.get(playhead.index) else {
        return;
    };
    for &l in &step.landmarks {
        cross(&mut gizmos, world_to_screen(l), SLAM);
    }
    if layers.detail == Detail::Step {
        draw_step(&mut gizmos, step);
    }
}

fn draw_step(gizmos: &mut Gizmos, step: &SlamStep) {
    for (l, e) in step.landmarks.iter().zip(&step.landmark_ellipses) {
        if e.major <= MAX_DRAWN_SIGMA {
            ellipse(gizmos, *l, e, SLAM);
        }
    }
    ellipse(gizmos, step.pose.position(), &step.ellipse, SLAM);
    let c = world_to_screen(step.pose.position());
    let spread = HEADING_SIGMAS * step.heading_stddev as f32;
    for side in [-spread, spread] {
        let dir = Vec2::from_angle(step.pose.heading + side);
        gizmos.line_2d(c, c + dir * WEDGE, SLAM);
    }
    for o in &step.observations {
        gizmos.line_2d(c, world_to_screen(step.landmarks[o.landmark]), PAIR);
        if o.new {
            gizmos.circle_2d(world_to_screen(o.seen_at), 10.0, NEW_LANDMARK);
        }
    }
}

// --- 3d ------------------------------------------------------------------------

/// The map's error at one step: the mean distance from each map landmark
/// to the nearest real one (0 for an empty map).
pub fn map_error(step: &SlamStep, real: &[Vec2]) -> f32 {
    if step.landmarks.is_empty() || real.is_empty() {
        return 0.0;
    }
    let total: f32 = step
        .landmarks
        .iter()
        .map(|m| real.iter().map(|r| r.distance(*m)).fold(f32::MAX, f32::min))
        .sum();
    total / step.landmarks.len() as f32
}

fn draw_plots(
    log: Res<RobotLog>,
    dr: Res<DeadReckoning>,
    kalman: Res<EkfLocalization>,
    slam: Res<SlamMapping>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let steps = log.len();
    let errors_plot = ERROR_PLOT.with_steps(steps);
    let map_plot = MAP_PLOT.with_steps(steps);
    for (plot, grid) in [(&errors_plot, ERROR_GRID), (&map_plot, MAP_GRID)] {
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
    if layers.kalman {
        errors(&mut kalman.poses(), KALMAN);
    }
    errors(&mut slam.poses(), SLAM);

    // The map's error.
    let real: Vec<Vec2> = log.arena.landmarks.iter().map(|l| l.position).collect();
    gizmos.linestrip_2d(
        slam.steps
            .iter()
            .enumerate()
            .map(|(i, s)| map_plot.to_screen(i as f32, map_error(s, &real))),
        SLAM,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_error_is_the_mean_nearest_distance() {
        let step = SlamStep {
            pose: Pose::default(),
            ellipse: ErrorEllipse {
                angle: 0.0,
                major: 0.0,
                minor: 0.0,
            },
            heading_stddev: 0.0,
            landmarks: vec![Vec2::new(3.0, 4.0), Vec2::new(100.0, 1.0)],
            landmark_ellipses: Vec::new(),
            observations: Vec::new(),
        };
        let real = [Vec2::ZERO, Vec2::new(100.0, 0.0)];
        assert_eq!(map_error(&step, &real), 3.0);
    }
}
