// Part 2: where everything goes on screen, and its colours.
//
// The window is 1280 x 720 and the 2D camera sits at (0, 0), so the screen
// runs from x = -640 to 640 and y = -360 to 360. Left: the arena, seen from
// above. Right: the current scan as a plot, like the lower half of the
// course's logfile_viewer.py, and its derivative under it.

use bevy::prelude::*;

pub const WINDOW_SIZE: UVec2 = UVec2::new(1280, 720);

/// The arena's (0, 0) corner on screen, and pixels per millimetre.
pub const ARENA_ORIGIN: Vec2 = Vec2::new(-620.0, -320.0);
pub const ARENA_SCALE: f32 = 0.3;

pub fn world_to_screen(p: Vec2) -> Vec2 {
    ARENA_ORIGIN + p * ARENA_SCALE
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

/// The key help sits in the gap between the two plots (UI position).
pub const HELP_AT: Vec2 = Vec2::new(670.0, 344.0);
pub const HELP_HEIGHT: f32 = 72.0;

/// Ranges, beam by beam.
pub const RANGE_PLOT: Plot = Plot {
    min: Vec2::new(30.0, 20.0),
    size: Vec2::new(590.0, 250.0),
    x_range: (0.0, 660.0),
    y_range: (0.0, 3000.0),
};

/// The derivative of the ranges.
pub const DERIVATIVE_PLOT: Plot = Plot {
    min: Vec2::new(30.0, -320.0),
    size: Vec2::new(590.0, 250.0),
    x_range: (0.0, 660.0),
    y_range: (-600.0, 600.0),
};

// --- Colours -------------------------------------------------------------------

pub const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.11);
pub const FRAME: Color = Color::srgb(0.35, 0.35, 0.42);
pub const LANDMARK: Color = Color::srgb(0.75, 0.75, 0.8);
/// The truth (reference positions).
pub const TRUTH: Color = Color::srgb(0.35, 0.85, 0.4);
/// What the motion model believes (dead reckoning).
pub const BELIEF: Color = Color::srgb(1.0, 0.6, 0.15);
pub const SCAN_POINT: Color = Color::srgba(0.4, 0.7, 1.0, 0.7);
pub const DETECTION: Color = Color::srgb(0.2, 0.95, 0.95);
pub const DERIVATIVE: Color = Color::srgb(0.85, 0.5, 0.95);
pub const THRESHOLD: Color = Color::srgba(1.0, 0.3, 0.3, 0.6);
pub const NO_ECHO: Color = Color::srgb(1.0, 0.25, 0.25);

// --- Layout checks, at COMPILE time ---------------------------------------------
//
// NEW: everything here is a constant, so `const { assert!(..) }` checks it
// while compiling: move a plot into the arena and the build fails, before
// any test runs. (Clippy suggested it: assert! on constants in a #[test]
// is a check that can only ever give one answer.)

const HALF_W: f32 = WINDOW_SIZE.x as f32 / 2.0;
const HALF_H: f32 = WINDOW_SIZE.y as f32 / 2.0;
/// The arena's far corner on screen.
const ARENA_TOP_RIGHT: Vec2 = Vec2::new(
    ARENA_ORIGIN.x + av_sim::world::ARENA_SIZE * ARENA_SCALE,
    ARENA_ORIGIN.y + av_sim::world::ARENA_SIZE * ARENA_SCALE,
);

const fn on_screen(p: Vec2) -> bool {
    p.x >= -HALF_W && p.x <= HALF_W && p.y >= -HALF_H && p.y <= HALF_H
}

const fn plot_on_screen(plot: &Plot) -> bool {
    on_screen(plot.min)
        && on_screen(Vec2::new(
            plot.min.x + plot.size.x,
            plot.min.y + plot.size.y,
        ))
}

const _: () = {
    // The arena fits, left of both plots, with room above for the header.
    assert!(on_screen(ARENA_ORIGIN) && on_screen(ARENA_TOP_RIGHT));
    assert!(ARENA_TOP_RIGHT.x < RANGE_PLOT.min.x);
    assert!(ARENA_TOP_RIGHT.x < DERIVATIVE_PLOT.min.x);
    assert!(HALF_H - ARENA_TOP_RIGHT.y >= 50.0);
    // The plots fit, the derivative under the ranges.
    assert!(plot_on_screen(&RANGE_PLOT) && plot_on_screen(&DERIVATIVE_PLOT));
    assert!(DERIVATIVE_PLOT.min.y + DERIVATIVE_PLOT.size.y < RANGE_PLOT.min.y);
    // The help sits between them (UI y runs downwards from the top).
    assert!(HELP_AT.y >= HALF_H - RANGE_PLOT.min.y);
    assert!(HELP_AT.y + HELP_HEIGHT <= HALF_H - (DERIVATIVE_PLOT.min.y + DERIVATIVE_PLOT.size.y));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_corners_and_clamping() {
        assert_eq!(RANGE_PLOT.to_screen(0.0, 0.0), RANGE_PLOT.min);
        assert_eq!(
            RANGE_PLOT.to_screen(660.0, 3000.0),
            RANGE_PLOT.min + RANGE_PLOT.size
        );
        // Beyond the range: stuck to the top edge.
        assert_eq!(
            RANGE_PLOT.to_screen(0.0, 9000.0).y,
            RANGE_PLOT.min.y + RANGE_PLOT.size.y
        );
        // The derivative plot's zero is in its middle.
        assert_eq!(
            DERIVATIVE_PLOT.to_screen(330.0, 0.0),
            DERIVATIVE_PLOT.centre()
        );
        // The caption corner is the plot's top-left, in UI coordinates.
        assert_eq!(RANGE_PLOT.caption_at(), Vec2::new(670.0, 90.0));
    }
}
