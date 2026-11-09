// Part 3: drawing, all with gizmos (as in av01 and av02). Only reads: the
// track, the two filters' results, the playhead.
//
//   3a  the track: the true robot (green), the measurement (yellow), each
//       filter's estimate with a bar one sigma either side
//   3b  the beliefs of the current step, the course's slam_06_f picture:
//       before (grey), prediction (blue), measurement (green), correction
//       (red). Histogram filter: light stepped lines, one step per cm.
//       Kalman filter: full smooth curves.
//   3c  the errors: estimate minus truth, step by step, with the Kalman
//       filter's two-sigma band, and the steps without a measurement shaded

use crate::Layers;
use crate::layout::*;
use av_sim::core::{AvSet, Playhead};
use av_sim::unit_c::distribution::Distribution;
use av_sim::unit_c::kalman::Density;
use av_sim::unit_c::track::TRACK_LENGTH;
use av_sim::unit_c::{HistogramFilter, HistogramParams, KalmanFilter, Track};
use bevy::prelude::*;

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (draw_track, draw_beliefs, draw_errors).in_set(AvSet::Present),
        );
    }
}

const ROBOT_RADIUS: f32 = 8.0;
/// Where the estimates sit under the track line.
const HISTOGRAM_ROW: f32 = -14.0;
const KALMAN_ROW: f32 = -24.0;

// --- 3a ------------------------------------------------------------------------

fn draw_track(
    track: Res<Track>,
    hist: Res<HistogramFilter>,
    kalman: Res<KalmanFilter>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let y = TRACK_ORIGIN.y;
    gizmos.line_2d(
        Vec2::new(track_x(0.0), y),
        Vec2::new(track_x(TRACK_LENGTH), y),
        FRAME,
    );
    let mut cm = 0.0;
    while cm <= TRACK_LENGTH {
        let x = track_x(cm);
        gizmos.line_2d(Vec2::new(x, y - 4.0), Vec2::new(x, y + 4.0), FRAME);
        cm += TRACK_TICK as f64;
    }
    let i = playhead.index;
    let Some(step) = track.log.steps.get(i) else {
        return;
    };
    // The robot, sitting on the track.
    gizmos.circle_2d(
        Vec2::new(track_x(step.truth), y + ROBOT_RADIUS + 2.0),
        ROBOT_RADIUS,
        TRUTH,
    );
    if let Some(z) = step.measured {
        let x = track_x(z);
        gizmos.line_2d(Vec2::new(x, y - 8.0), Vec2::new(x, y + 26.0), MEASURED);
    }
    // Each estimate: a dot at the mean, a bar one sigma either side.
    let mut estimate = |row: f32, mean: f64, sigma: f64, colour: Color| {
        let r = y + row;
        gizmos.line_2d(
            Vec2::new(track_x(mean - sigma), r),
            Vec2::new(track_x(mean + sigma), r),
            colour,
        );
        gizmos.circle_2d(Vec2::new(track_x(mean), r), 3.0, colour);
    };
    if let (true, Some(h)) = (layers.view.histogram(), hist.steps.get(i)) {
        let c = &h.correction;
        estimate(
            HISTOGRAM_ROW,
            c.mean(),
            c.variance().sqrt(),
            HIST_CORRECTION,
        );
    }
    if let (true, Some(k)) = (layers.view.kalman(), kalman.steps.get(i)) {
        let c = k.correction;
        estimate(KALMAN_ROW, c.mu, c.sigma(), KALMAN_CORRECTION);
    }
}

// --- 3b ------------------------------------------------------------------------

fn draw_beliefs(
    track: Res<Track>,
    hist: Res<HistogramFilter>,
    params: Res<HistogramParams>,
    kalman: Res<KalmanFilter>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let plot = &BELIEF_PLOT;
    gizmos.rect_2d(plot.centre(), plot.size, FRAME);
    let mut p = BELIEF_GRID;
    while p < plot.y_range.1 {
        gizmos.line_2d(
            plot.to_screen(plot.x_range.0, p),
            plot.to_screen(plot.x_range.1, p),
            GRID,
        );
        p += BELIEF_GRID;
    }
    let i = playhead.index;
    let Some(step) = track.log.steps.get(i) else {
        return;
    };
    let model = &track.log.model;
    gizmos.line_2d(
        plot.to_screen(step.truth as f32, plot.y_range.0),
        plot.to_screen(step.truth as f32, plot.y_range.1),
        TRUTH,
    );

    if let (true, Some(h)) = (layers.view.histogram(), hist.steps.get(i)) {
        let mut stepped = |d: &Distribution, colour: Color| {
            gizmos.linestrip_2d(stepped_outline(d), colour);
        };
        if layers.before {
            stepped(&hist.belief_before(i, model, params.shape), BEFORE);
        }
        stepped(&h.prediction, HIST_PREDICTION);
        // No measurement: the correction IS the prediction. Drawing it
        // on top would hide the blue, which says "only predicted".
        if let Some(m) = &h.measurement {
            stepped(m, HIST_MEASUREMENT);
            stepped(&h.correction, HIST_CORRECTION);
        }
    }
    if let (true, Some(k)) = (layers.view.kalman(), kalman.steps.get(i)) {
        let mut curve = |d: Density, colour: Color| {
            gizmos.linestrip_2d(smooth_curve(d), colour);
        };
        // Grey "before" only once: the histogram's, if it's shown.
        if layers.before && !layers.view.histogram() {
            curve(kalman.belief_before(i, model), BEFORE);
        }
        curve(k.prediction, KALMAN_PREDICTION);
        if let Some(m) = k.measurement {
            curve(m, KALMAN_MEASUREMENT);
            curve(k.correction, KALMAN_CORRECTION);
        }
    }
}

/// A histogram as the course plots it (linestyle='steps'): flat across
/// each cm cell, straight down to 0 at the ends. Cells off the track are
/// left out.
fn stepped_outline(d: &Distribution) -> Vec<Vec2> {
    let plot = &BELIEF_PLOT;
    let on_track: Vec<(f32, f32)> = d
        .cells()
        .filter(|&(x, _)| (0..=TRACK_LENGTH as i64).contains(&x))
        .map(|(x, p)| (x as f32, p as f32))
        .collect();
    let (Some(first), Some(last)) = (on_track.first(), on_track.last()) else {
        return Vec::new();
    };
    let mut points = vec![plot.to_screen(first.0 - 0.5, 0.0)];
    for &(x, p) in &on_track {
        points.push(plot.to_screen(x - 0.5, p));
        points.push(plot.to_screen(x + 0.5, p));
    }
    points.push(plot.to_screen(last.0 + 0.5, 0.0));
    points
}

/// A gaussian's density, once per cm, out to 5 sigma (on the track only).
/// Density per cm is probability per cell, so it's on the histogram's scale.
fn smooth_curve(d: Density) -> Vec<Vec2> {
    let reach = 5.0 * d.sigma();
    let from = (d.mu - reach).max(0.0).floor() as i64;
    let to = (d.mu + reach).min(TRACK_LENGTH).ceil() as i64;
    (from..=to)
        .map(|x| BELIEF_PLOT.to_screen(x as f32, d.pdf(x as f64) as f32))
        .collect()
}

// --- 3c ------------------------------------------------------------------------

fn draw_errors(
    track: Res<Track>,
    hist: Res<HistogramFilter>,
    kalman: Res<KalmanFilter>,
    playhead: Res<Playhead>,
    layers: Res<Layers>,
    mut gizmos: Gizmos,
) {
    let plot = ERROR_PLOT.with_steps(track.len().saturating_sub(1));
    gizmos.rect_2d(plot.centre(), plot.size, FRAME);
    let mut y = plot.y_range.0 + ERROR_GRID;
    while y < plot.y_range.1 {
        let colour = if y == 0.0 { FRAME } else { GRID };
        gizmos.line_2d(
            plot.to_screen(0.0, y),
            plot.to_screen(plot.x_range.1, y),
            colour,
        );
        y += ERROR_GRID;
    }
    let steps = &track.log.steps;
    // No sensor: shade the step.
    for (i, s) in steps.iter().enumerate() {
        if s.measured.is_none() {
            let x = i as f32;
            gizmos.line_2d(
                plot.to_screen(x, plot.y_range.0),
                plot.to_screen(x, plot.y_range.1),
                NO_SENSOR,
            );
        }
    }
    // The sensor's own error: a small cross per measurement.
    for (i, s) in steps.iter().enumerate() {
        if let Some(z) = s.measured {
            let c = plot.to_screen(i as f32, (z - s.truth) as f32);
            gizmos.line_2d(c - Vec2::new(3.0, 0.0), c + Vec2::new(3.0, 0.0), MEASURED);
            gizmos.line_2d(c - Vec2::new(0.0, 3.0), c + Vec2::new(0.0, 3.0), MEASURED);
        }
    }
    if layers.view.kalman() {
        // The two-sigma band: the filter says the truth is inside ~95% of
        // the time. The red line should leave it about once in 20 steps.
        for sign in [-2.0, 2.0] {
            gizmos.linestrip_2d(
                kalman
                    .steps
                    .iter()
                    .enumerate()
                    .map(|(i, k)| plot.to_screen(i as f32, (sign * k.correction.sigma()) as f32)),
                BEFORE,
            );
        }
        gizmos.linestrip_2d(
            kalman
                .steps
                .iter()
                .zip(steps)
                .enumerate()
                .map(|(i, (k, s))| plot.to_screen(i as f32, (k.correction.mu - s.truth) as f32)),
            KALMAN_CORRECTION,
        );
    }
    if layers.view.histogram() {
        gizmos.linestrip_2d(
            hist.steps.iter().zip(steps).enumerate().map(|(i, (h, s))| {
                plot.to_screen(i as f32, (h.correction.mean() - s.truth) as f32)
            }),
            HIST_CORRECTION,
        );
    }
    let x = playhead.index as f32;
    gizmos.line_2d(
        plot.to_screen(x, plot.y_range.0),
        plot.to_screen(x, plot.y_range.1),
        FRAME,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlines_stay_on_the_track() {
        // A belief hanging off the start of the track: only cells 0.. drawn.
        let d = Distribution::gaussian(3, 4.0);
        let points = stepped_outline(&d);
        let left = BELIEF_PLOT.to_screen(-0.5, 0.0);
        assert_eq!(points[0], left);
        assert!(points.iter().all(|p| p.x >= left.x));
        // Two points per cell, plus the two ends.
        let cells = d.cells().filter(|&(x, _)| x >= 0).count();
        assert_eq!(points.len(), 2 * cells + 2);

        let curve = smooth_curve(Density::new(TRACK_LENGTH - 2.0, 3.0));
        assert!(curve.last().unwrap().x <= track_x(TRACK_LENGTH));
    }
}
