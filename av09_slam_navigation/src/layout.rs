// Part 2: where everything goes on screen, and its colours.
//
// The window is 1280 x 720 and the 2D camera sits at (0, 0), so the screen
// runs from x = -640 to 640 and y = -360 to 360. Left: the arena from
// above, 0.33 px per mm. Right: the text.

use bevy::prelude::*;

pub const WINDOW_SIZE: UVec2 = UVec2::new(1280, 720);

/// The arena's (0, 0) corner on screen, and pixels per millimetre.
pub const ARENA_ORIGIN: Vec2 = Vec2::new(-625.0, -340.0);
pub const ARENA_SCALE: f32 = 0.33;
pub const ARENA_PX: f32 = av_sim::world::ARENA_SIZE * ARENA_SCALE;

pub fn world_to_screen(p: Vec2) -> Vec2 {
    ARENA_ORIGIN + p * ARENA_SCALE
}

pub fn screen_to_world(p: Vec2) -> Vec2 {
    (p - ARENA_ORIGIN) / ARENA_SCALE
}

/// Is this screen point over the arena?
pub fn over_arena(p: Vec2) -> bool {
    let w = screen_to_world(p);
    w.cmpge(Vec2::ZERO).all() && w.cmple(Vec2::splat(av_sim::world::ARENA_SIZE)).all()
}

/// The text panel's top-left corner (UI position, from the window's
/// top-left).
pub const PANEL_AT: Vec2 = Vec2::new(705.0, 44.0);

/// Ellipses show this many sigmas: 2.45 is the 95% ellipse in 2D.
pub const ELLIPSE_SIGMAS: f32 = 2.45;
/// Ellipses with a long half axis (1 sigma) over this many mm
/// aren't drawn (as in av06: they'd cover the screen).
pub const MAX_DRAWN_SIGMA: f64 = 150.0;
/// The true path drawn: the last this many steps (a minute).
pub const TRAIL_STEPS: usize = 300;

// --- Colours -------------------------------------------------------------------

pub const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.11);
pub const FRAME: Color = Color::srgb(0.35, 0.35, 0.42);
/// The real cylinders (the robot isn't told where they are).
pub const LANDMARK: Color = Color::srgb(0.75, 0.75, 0.8);
pub const TRUTH: Color = Color::srgb(0.35, 0.85, 0.4);
pub const SLAM: Color = Color::srgb(0.8, 0.45, 1.0);
pub const BELIEF: Color = Color::srgb(1.0, 0.6, 0.15);
pub const ROUTE: Color = Color::srgb(1.0, 0.9, 0.3);
pub const TARGET: Color = Color::srgb(1.0, 1.0, 1.0);
pub const GOAL: Color = Color::srgb(1.0, 0.35, 0.55);
/// A blocked cell of the planning map (RGBA, see-through).
pub const BLOCKED_RGBA: [u8; 4] = [190, 60, 50, 90];
pub const FREE_RGBA: [u8; 4] = [0, 0, 0, 0];

// --- Layout checks, at compile time (as in av01) ----------------------------------

const HALF_W: f32 = WINDOW_SIZE.x as f32 / 2.0;
const HALF_H: f32 = WINDOW_SIZE.y as f32 / 2.0;

const _: () = {
    assert!(ARENA_ORIGIN.x >= -HALF_W && ARENA_ORIGIN.y >= -HALF_H);
    assert!(ARENA_ORIGIN.y + ARENA_PX <= HALF_H);
    // The panel starts right of the arena.
    assert!(PANEL_AT.x - HALF_W > ARENA_ORIGIN.x + ARENA_PX);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_and_world_agree() {
        let p = Vec2::new(1234.0, 567.0);
        assert!((screen_to_world(world_to_screen(p)) - p).length() < 1e-3);
        assert!(over_arena(world_to_screen(p)));
        assert!(!over_arena(Vec2::new(200.0, 0.0)));
    }
}
