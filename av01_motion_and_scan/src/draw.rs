// Part 3: drawing, all with gizmos (redrawn every frame, nothing to spawn
// or clean up). Only reads: the log, the algorithms' results, the playhead.
//
//   3a  the arena: walls, landmarks
//   3b  the robot: true path (green), dead-reckoning path (orange)
//   3c  what it sees: scan points and detected cylinders, placed with the
//       DEAD-RECKONING pose. As it drifts, the scan slides off the walls
//       and the cylinders off the landmarks: that gap is what units B to G
//       are about.
//   3d  the scan plot and its derivative, with the detector's edges

use crate::Layers;
use crate::layout::*;
use av_sim::core::{AvSet, Playhead, RobotLog};
use av_sim::robot::{Pose, beam_index_to_angle};
use av_sim::unit_a::cylinders::{DetectedCylinders, DetectorParams, compute_derivative};
use av_sim::unit_a::motion::DeadReckoning;
use bevy::prelude::*;

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (draw_arena, draw_paths, draw_seen, draw_plots).in_set(AvSet::Present),
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

fn draw_paths(
    log: Res<RobotLog>,
    dr: Res<DeadReckoning>,
    playhead: Res<Playhead>,
    mut gizmos: Gizmos,
) {
    let upto = playhead.index + 1;
    let truth = &log.log.reference;
    if !truth.is_empty() {
        let points = truth[..upto.min(truth.len())]
            .iter()
            .map(|r| world_to_screen(Vec2::new(r.x as f32, r.y as f32)));
        gizmos.linestrip_2d(points, TRUTH);
        if let Some(r) = truth.get(playhead.index) {
            // P records have no heading: a circle only.
            let p = world_to_screen(Vec2::new(r.x as f32, r.y as f32));
            gizmos.circle_2d(p, ROBOT_SCREEN_RADIUS, TRUTH);
        }
    }
    if !dr.poses.is_empty() {
        let poses = &dr.poses[..upto.min(dr.poses.len())];
        gizmos.linestrip_2d(poses.iter().map(|p| world_to_screen(p.position())), BELIEF);
        if let Some(pose) = dr.poses.get(playhead.index) {
            draw_robot(&mut gizmos, pose, BELIEF);
        }
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

fn draw_seen(
    log: Res<RobotLog>,
    dr: Res<DeadReckoning>,
    found: Res<DetectedCylinders>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    params: Res<DetectorParams>,
    mut gizmos: Gizmos,
) {
    let i = playhead.index;
    let Some(pose) = dr.poses.get(i) else {
        return;
    };
    if layers.scan
        && let Some(scan) = log.log.scans.get(i)
    {
        for (beam, &range) in scan.ranges.iter().enumerate() {
            if range as f32 <= params.min_valid_distance {
                continue;
            }
            let local = Vec2::from_angle(beam_index_to_angle(beam as f32)) * range as f32;
            let p = world_to_screen(pose.to_world(local));
            gizmos.rect_2d(p, Vec2::splat(2.0), SCAN_POINT);
        }
    }
    if layers.cylinders
        && let Some(cylinders) = found.per_scan.get(i)
    {
        let from = world_to_screen(pose.position());
        for c in cylinders {
            let p = world_to_screen(pose.to_world(c.local));
            gizmos.line_2d(from, p, DETECTION.with_alpha(0.3));
            cross(&mut gizmos, p, DETECTION);
        }
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
    found: Res<DetectedCylinders>,
    playhead: Res<Playhead>,
    params: Res<DetectorParams>,
    mut gizmos: Gizmos,
) {
    for plot in [&RANGE_PLOT, &DERIVATIVE_PLOT] {
        gizmos.rect_2d(plot.centre(), plot.size, FRAME);
    }
    let Some(scan) = log.log.scans.get(playhead.index) else {
        return;
    };
    let ranges = &scan.ranges;

    // Ranges, and the beams without an echo as red ticks at the bottom.
    gizmos.linestrip_2d(
        ranges
            .iter()
            .enumerate()
            .map(|(i, &r)| RANGE_PLOT.to_screen(i as f32, r as f32)),
        SCAN_POINT,
    );
    for (i, &r) in ranges.iter().enumerate() {
        if r as f32 <= params.min_valid_distance {
            let p = RANGE_PLOT.to_screen(i as f32, 0.0);
            gizmos.line_2d(p, p + Vec2::Y * 8.0, NO_ECHO);
        }
    }

    // The derivative, and the edge threshold above and below zero.
    let derivative = compute_derivative(ranges, params.min_valid_distance);
    gizmos.linestrip_2d(
        derivative
            .iter()
            .enumerate()
            .map(|(i, &d)| DERIVATIVE_PLOT.to_screen(i as f32, d)),
        DERIVATIVE,
    );
    for y in [params.depth_jump, 0.0, -params.depth_jump] {
        let colour = if y == 0.0 { FRAME } else { THRESHOLD };
        gizmos.line_2d(
            DERIVATIVE_PLOT.to_screen(0.0, y),
            DERIVATIVE_PLOT.to_screen(660.0, y),
            colour,
        );
    }

    // Each cylinder: a line through both plots at its average beam, and a
    // dot at its average depth.
    if let Some(cylinders) = found.per_scan.get(playhead.index) {
        for c in cylinders {
            for plot in [&RANGE_PLOT, &DERIVATIVE_PLOT] {
                gizmos.line_2d(
                    plot.to_screen(c.ray, plot.y_range.0),
                    plot.to_screen(c.ray, plot.y_range.1),
                    DETECTION.with_alpha(0.5),
                );
            }
            gizmos.circle_2d(RANGE_PLOT.to_screen(c.ray, c.depth), 4.0, DETECTION);
        }
    }
}
