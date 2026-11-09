// Part 3: drawing, all with gizmos (as in av01..av04). Only reads: the
// log, the algorithms' results, the playhead.
//
//   3a  the arena: walls, landmarks
//   3b  the paths: truth (green), dead reckoning (orange), Kalman (blue),
//       particle filter mean (pink)
//   3c  the particles at the current step (V):
//         Cloud    after resampling: a short tick per particle, pointing
//                  its way, and the cloud's 95% ellipse and heading wedge
//         Weights  before resampling, yellow, as bright as its weight
//                  compared with the heaviest particle
//   3d  the plots: distance from the truth, effective particle count

use crate::layout::*;
use crate::{Detail, Layers};
use av_sim::core::{AvSet, Playhead, RobotLog};
use av_sim::robot::Pose;
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_d::EkfLocalization;
use av_sim::unit_d::ekf::ErrorEllipse;
use av_sim::unit_e::{ParticleLocalization, ParticleStep};
use bevy::math::DVec3;
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
    pf: Res<ParticleLocalization>,
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
    let poses: Vec<Pose> = pf.poses().collect();
    path(&poses, PARTICLES);
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

/// A particle: a short tick from its centre, its way.
fn tick(gizmos: &mut Gizmos, p: DVec3, colour: Color) {
    let at = world_to_screen(Vec2::new(p.x as f32, p.y as f32));
    gizmos.line_2d(
        at,
        at + Vec2::from_angle(p.z as f32) * PARTICLE_TICK,
        colour,
    );
}

fn draw_detail(
    pf: Res<ParticleLocalization>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let Some(step) = pf.steps.get(playhead.index) else {
        return;
    };
    match layers.detail {
        Detail::Cloud => draw_cloud(&mut gizmos, step),
        Detail::Weights => draw_weights(&mut gizmos, step),
        Detail::Off => {}
    }
}

fn draw_cloud(gizmos: &mut Gizmos, step: &ParticleStep) {
    for &p in &step.particles {
        tick(gizmos, p, PARTICLE);
    }
    ellipse(gizmos, step.pose.position(), &step.ellipse, PARTICLES);
    let c = world_to_screen(step.pose.position());
    let spread = HEADING_SIGMAS * step.heading_stddev as f32;
    for side in [-spread, spread] {
        let dir = Vec2::from_angle(step.pose.heading + side);
        gizmos.line_2d(c, c + dir * WEDGE, PARTICLES);
    }
}

fn draw_weights(gizmos: &mut Gizmos, step: &ParticleStep) {
    let top = step.weights.iter().copied().fold(0.0, f64::max);
    if top <= 0.0 {
        return;
    }
    for (&p, &w) in step.predicted.iter().zip(&step.weights) {
        // A floor of 0.08, so the particles about to die are still seen.
        let alpha = 0.08 + 0.92 * (w / top) as f32;
        tick(gizmos, p, WEIGHED.with_alpha(alpha));
    }
}

// --- 3d ------------------------------------------------------------------------

fn draw_plots(
    log: Res<RobotLog>,
    dr: Res<DeadReckoning>,
    kalman: Res<EkfLocalization>,
    pf: Res<ParticleLocalization>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let steps = log.len();
    let errors_plot = ERROR_PLOT.with_steps(steps);
    let count_plot = COUNT_PLOT.with_steps(steps);
    for (plot, grid) in [(&errors_plot, ERROR_GRID), (&count_plot, COUNT_GRID)] {
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
    errors(&mut pf.poses(), PARTICLES);

    // The effective count, as % of all particles.
    gizmos.linestrip_2d(
        pf.steps.iter().enumerate().map(|(i, s)| {
            let share = 100.0 * s.effective / s.particles.len().max(1) as f64;
            count_plot.to_screen(i as f32, share as f32)
        }),
        PARTICLES,
    );
}
