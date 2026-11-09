// Part 2: where everything goes on screen, and its colours.
//
// The window is 1280 x 720 and the 2D camera sits at (0, 0), so the screen
// runs from x = -640 to 640 and y = -360 to 360. Left: the arena, seen from
// above (as in av01). Right: how far each pose is from the truth, step by
// step, and under it how big each step's correction was.

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

/// The key help sits in the gap between the two plots (UI position).
pub const HELP_AT: Vec2 = Vec2::new(670.0, 344.0);
pub const HELP_HEIGHT: f32 = 72.0;

/// Distance from the truth (mm) by step. The x range is set per frame
/// (with_steps); 278 is the length of a generated run.
pub const ERROR_PLOT: Plot = Plot {
    min: Vec2::new(30.0, 20.0),
    size: Vec2::new(590.0, 250.0),
    x_range: (0.0, 278.0),
    y_range: (0.0, 200.0),
};

/// How far each step's correction moved the pose (mm).
pub const CORRECTION_PLOT: Plot = Plot {
    min: Vec2::new(30.0, -320.0),
    size: Vec2::new(590.0, 250.0),
    x_range: (0.0, 278.0),
    y_range: (0.0, 40.0),
};

/// Grid lines every this many mm in the two plots.
pub const ERROR_GRID: f32 = 50.0;
pub const CORRECTION_GRID: f32 = 10.0;

// --- Colours -------------------------------------------------------------------

pub const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.11);
pub const FRAME: Color = Color::srgb(0.35, 0.35, 0.42);
pub const GRID: Color = Color::srgba(0.35, 0.35, 0.42, 0.35);
pub const LANDMARK: Color = Color::srgb(0.75, 0.75, 0.8);
/// The truth (reference positions).
pub const TRUTH: Color = Color::srgb(0.35, 0.85, 0.4);
/// Dead reckoning (Unit A).
pub const BELIEF: Color = Color::srgb(1.0, 0.6, 0.15);
/// Corrected with the landmarks.
pub const BY_LANDMARKS: Color = Color::srgb(1.0, 0.35, 0.8);
/// Corrected with the walls (ICP).
pub const BY_WALLS: Color = Color::srgb(0.35, 0.6, 1.0);
/// Detections and scan points before correcting.
pub const DETECTION: Color = Color::srgb(0.2, 0.95, 0.95);
pub const RAW_POINT: Color = Color::srgba(0.7, 0.7, 0.7, 0.5);
/// A pair: detection to landmark, point to wall.
pub const PAIR: Color = Color::srgba(1.0, 0.9, 0.3, 0.6);

// --- Layout checks, at compile time (as in av01) ----------------------------------

const HALF_W: f32 = WINDOW_SIZE.x as f32 / 2.0;
const HALF_H: f32 = WINDOW_SIZE.y as f32 / 2.0;
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
    assert!(on_screen(ARENA_ORIGIN) && on_screen(ARENA_TOP_RIGHT));
    assert!(ARENA_TOP_RIGHT.x < ERROR_PLOT.min.x);
    assert!(ARENA_TOP_RIGHT.x < CORRECTION_PLOT.min.x);
    assert!(HALF_H - ARENA_TOP_RIGHT.y >= 50.0);
    assert!(plot_on_screen(&ERROR_PLOT) && plot_on_screen(&CORRECTION_PLOT));
    assert!(CORRECTION_PLOT.min.y + CORRECTION_PLOT.size.y < ERROR_PLOT.min.y);
    assert!(HELP_AT.y >= HALF_H - ERROR_PLOT.min.y);
    assert!(HELP_AT.y + HELP_HEIGHT <= HALF_H - (CORRECTION_PLOT.min.y + CORRECTION_PLOT.size.y));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plots_map_steps_and_millimetres() {
        assert_eq!(ERROR_PLOT.to_screen(0.0, 0.0), ERROR_PLOT.min);
        // A live run of 100 steps fills the plot's width.
        let live = ERROR_PLOT.with_steps(100);
        assert_eq!(
            live.to_screen(100.0, 200.0),
            ERROR_PLOT.min + ERROR_PLOT.size
        );
        // Off the scale: stuck to the top edge.
        assert_eq!(
            ERROR_PLOT.to_screen(0.0, 900.0).y,
            ERROR_PLOT.min.y + ERROR_PLOT.size.y
        );
        assert_eq!(ERROR_PLOT.caption_at(), Vec2::new(670.0, 90.0));
        // An empty log doesn't divide by zero.
        assert!(ERROR_PLOT.with_steps(0).to_screen(0.0, 0.0).is_finite());
    }
}
