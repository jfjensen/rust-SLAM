// Part 2: where everything goes on screen, and its colours.
//
// The window is 1280 x 720 and the 2D camera sits at (0, 0), so the screen
// runs from x = -640 to 640 and y = -360 to 360. Top to bottom:
//
//   the track     the robot's world, 0 to 500 cm, seen from the side
//   beliefs       the distributions of the current step, over the same
//                 x axis as the track, so a peak sits under the robot
//   errors        step by step: how far each estimate is from the truth
//                 (left), and the keys (right)

use av_sim::unit_c::track::TRACK_LENGTH;
use bevy::prelude::*;

pub const WINDOW_SIZE: UVec2 = UVec2::new(1280, 720);

/// The track's 0 cm on screen, and pixels per cm.
pub const TRACK_ORIGIN: Vec2 = Vec2::new(-600.0, 225.0);
pub const TRACK_SCALE: f32 = 2.4;
/// Tick marks along the track every this many cm.
pub const TRACK_TICK: f32 = 50.0;

/// A position on the track (cm) to a screen x.
pub fn track_x(cm: f64) -> f32 {
    TRACK_ORIGIN.x + cm as f32 * TRACK_SCALE
}

/// A rectangle on screen showing x_range by y_range.
pub struct Plot {
    pub min: Vec2,
    pub size: Vec2,
    pub x_range: (f32, f32),
    pub y_range: (f32, f32),
}

impl Plot {
    /// A data point on screen; y is clamped to the plot.
    pub fn to_screen(&self, x: f32, y: f32) -> Vec2 {
        let fx = (x - self.x_range.0) / (self.x_range.1 - self.x_range.0);
        let fy = ((y - self.y_range.0) / (self.y_range.1 - self.y_range.0)).clamp(0.0, 1.0);
        self.min + self.size * Vec2::new(fx, fy)
    }

    /// The same plot showing steps 0 to `steps` (a live run grows).
    pub fn with_steps(&self, steps: usize) -> Plot {
        Plot {
            x_range: (0.0, steps.max(1) as f32),
            ..*self
        }
    }

    pub fn centre(&self) -> Vec2 {
        self.min + self.size / 2.0
    }

    /// The UI position (from the window's top-left) of the plot's top-left
    /// corner, for its caption.
    pub fn caption_at(&self) -> Vec2 {
        let top_left = Vec2::new(self.min.x, self.min.y + self.size.y);
        Vec2::new(
            top_left.x + WINDOW_SIZE.x as f32 / 2.0,
            WINDOW_SIZE.y as f32 / 2.0 - top_left.y,
        )
    }
}

/// Probability per cm cell, under the track (same x as the track).
pub const BELIEF_PLOT: Plot = Plot {
    min: Vec2::new(TRACK_ORIGIN.x, -70.0),
    size: Vec2::new(TRACK_LENGTH as f32 * TRACK_SCALE, 240.0),
    x_range: (0.0, TRACK_LENGTH as f32),
    y_range: (0.0, 0.06),
};
/// Grid lines in the belief plot every this much probability.
pub const BELIEF_GRID: f32 = 0.01;

/// Estimate minus truth (cm) by step. The x range is set per frame.
pub const ERROR_PLOT: Plot = Plot {
    min: Vec2::new(TRACK_ORIGIN.x, -340.0),
    size: Vec2::new(800.0, 220.0),
    x_range: (0.0, 60.0),
    y_range: (-50.0, 50.0),
};
pub const ERROR_GRID: f32 = 10.0;

/// The key help, right of the error plot (UI position).
pub const HELP_AT: Vec2 = Vec2::new(860.0, 500.0);

// --- Colours (the course's slam_06_e/f: prediction blue, measurement green,
// correction red; light for the histogram, full for Kalman) -------------------

pub const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.11);
pub const FRAME: Color = Color::srgb(0.35, 0.35, 0.42);
pub const GRID: Color = Color::srgba(0.35, 0.35, 0.42, 0.35);
pub const TRUTH: Color = Color::srgb(0.35, 0.85, 0.4);
/// The belief before this step.
pub const BEFORE: Color = Color::srgba(0.7, 0.7, 0.7, 0.6);
pub const HIST_PREDICTION: Color = Color::srgba(0.6, 0.65, 1.0, 0.75);
pub const HIST_MEASUREMENT: Color = Color::srgba(0.6, 1.0, 0.6, 0.75);
pub const HIST_CORRECTION: Color = Color::srgba(1.0, 0.6, 0.65, 0.9);
pub const KALMAN_PREDICTION: Color = Color::srgb(0.25, 0.4, 1.0);
pub const KALMAN_MEASUREMENT: Color = Color::srgb(0.2, 0.8, 0.2);
pub const KALMAN_CORRECTION: Color = Color::srgb(1.0, 0.2, 0.25);
/// The sensor's readings (track and error plot).
pub const MEASURED: Color = Color::srgb(0.95, 0.9, 0.3);
/// Steps without a measurement, in the error plot.
pub const NO_SENSOR: Color = Color::srgba(0.5, 0.5, 0.6, 0.18);

// --- Layout checks, at compile time (as in av01) ----------------------------------

const HALF_W: f32 = WINDOW_SIZE.x as f32 / 2.0;
const HALF_H: f32 = WINDOW_SIZE.y as f32 / 2.0;

const fn on_screen(p: Vec2) -> bool {
    p.x >= -HALF_W && p.x <= HALF_W && p.y >= -HALF_H && p.y <= HALF_H
}

const fn top(plot: &Plot) -> f32 {
    plot.min.y + plot.size.y
}

const _: () = {
    // The track fits, with room above it for the header (five lines,
    // about 100 px) and the robot.
    assert!(on_screen(TRACK_ORIGIN));
    assert!(TRACK_ORIGIN.x + TRACK_LENGTH as f32 * TRACK_SCALE <= HALF_W);
    assert!(HALF_H - TRACK_ORIGIN.y >= 125.0);
    // The belief plot is under the track (room for the estimate bars and
    // its caption), the error plot under that.
    assert!(top(&BELIEF_PLOT) + 50.0 <= TRACK_ORIGIN.y);
    assert!(top(&ERROR_PLOT) + 30.0 <= BELIEF_PLOT.min.y);
    assert!(on_screen(ERROR_PLOT.min));
    // The help is right of the error plot and below the belief plot.
    assert!(HELP_AT.x >= ERROR_PLOT.min.x + ERROR_PLOT.size.x + HALF_W);
    assert!(HELP_AT.y >= HALF_H - BELIEF_PLOT.min.y);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_and_belief_plot_share_their_x() {
        for cm in [0.0, 137.0, TRACK_LENGTH] {
            assert!((BELIEF_PLOT.to_screen(cm as f32, 0.0).x - track_x(cm)).abs() < 1e-3);
        }
        assert_eq!(BELIEF_PLOT.caption_at(), Vec2::new(40.0, 190.0));
        // A live run of 10 steps fills the error plot's width.
        let live = ERROR_PLOT.with_steps(10);
        assert_eq!(live.to_screen(10.0, 50.0), ERROR_PLOT.min + ERROR_PLOT.size);
        assert!(ERROR_PLOT.with_steps(0).to_screen(0.0, 0.0).is_finite());
    }
}
