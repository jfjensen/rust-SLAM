// Part 2: where everything goes on screen, and its colours.
//
// The window is 1280 x 720 and the 2D camera sits at (0, 0). Left: the
// map, 800 x 600 pixels, the same size on both pages (the grid page has
// 200 x 150 cells of 4 pixels, the car page 100 x 75 cells of 8, as in
// the course). Right: the text.

use bevy::prelude::*;

pub const WINDOW_SIZE: UVec2 = UVec2::new(1280, 720);

/// The map's bottom-left corner on screen, and its size in pixels.
pub const MAP_MIN: Vec2 = Vec2::new(-630.0, -330.0);
pub const MAP_SIZE: Vec2 = Vec2::new(800.0, 600.0);

/// Grid page: cells and pixels per cell (the course's pp_01).
pub const GRID_CELLS: (usize, usize) = (200, 150);
/// Car page: cells and pixels per cell (the course's pp_02).
pub const CAR_CELLS: (usize, usize) = (100, 75);

/// Pixels per cell for a grid of this width (the map is always 800 wide).
pub fn cell_px(width: usize) -> f32 {
    MAP_SIZE.x / width as f32
}

/// The screen centre of cell (x, y) (or of any point in cell units).
pub fn cell_to_screen(p: Vec2, width: usize) -> Vec2 {
    MAP_MIN + p * cell_px(width)
}

/// The cell under a screen point, if it's on the map.
pub fn screen_to_cell(p: Vec2, width: usize, height: usize) -> Option<(i32, i32)> {
    let c = (p - MAP_MIN) / cell_px(width);
    let (x, y) = (c.x.floor() as i32, c.y.floor() as i32);
    (x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height).then_some((x, y))
}

/// A screen point in cell units (not rounded; may be off the map).
pub fn screen_to_cell_units(p: Vec2, width: usize) -> Vec2 {
    (p - MAP_MIN) / cell_px(width)
}

/// The text panel, right of the map (UI position from the top-left).
pub const PANEL_AT: Vec2 = Vec2::new(830.0, 30.0);

// --- Colours (the course's: red walls, blue potential, green visited) ----------

pub const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.11);
pub const FRAME: Color = Color::srgb(0.35, 0.35, 0.42);
/// Texture colours, RGBA bytes.
pub const FREE_RGBA: [u8; 4] = [22, 22, 30, 255];
pub const WALL_RGBA: [u8; 4] = [200, 70, 60, 255];
pub const FRONT_RGBA: [u8; 4] = [240, 220, 80, 255];
/// The path, start and goal (gizmos).
pub const PATH: Color = Color::WHITE;
pub const START: Color = Color::srgb(0.35, 0.95, 0.45);
pub const GOAL: Color = Color::srgb(1.0, 0.45, 0.85);
/// The cell being expanded now.
pub const CURRENT: Color = Color::srgb(1.0, 1.0, 0.6);

/// A free cell's colour: green for visited (brighter = costlier, as the
/// course shades by cost), blue for the potential function's cost.
pub fn cell_rgba(visited: Option<f32>, potential: u8) -> [u8; 4] {
    let mut c = FREE_RGBA;
    if potential > 0 {
        c[2] = c[2].saturating_add((potential as f32 * 0.6) as u8);
    }
    if let Some(v) = visited {
        c[1] = c[1].saturating_add((40.0 + 160.0 * v) as u8);
    }
    c
}

// --- Layout checks, at compile time (as in av01) ----------------------------------

const HALF_W: f32 = WINDOW_SIZE.x as f32 / 2.0;
const HALF_H: f32 = WINDOW_SIZE.y as f32 / 2.0;

const _: () = {
    // The map is on screen, with room above it for the title line.
    assert!(MAP_MIN.x >= -HALF_W && MAP_MIN.y >= -HALF_H);
    assert!(MAP_MIN.x + MAP_SIZE.x <= HALF_W);
    assert!(HALF_H - (MAP_MIN.y + MAP_SIZE.y) >= 30.0);
    // The panel starts right of the map.
    assert!(PANEL_AT.x >= MAP_MIN.x + MAP_SIZE.x + HALF_W);
    // Both pages fill the map exactly, with square cells.
    assert!(GRID_CELLS.0 as f32 / GRID_CELLS.1 as f32 == MAP_SIZE.x / MAP_SIZE.y);
    assert!(CAR_CELLS.0 as f32 / CAR_CELLS.1 as f32 == MAP_SIZE.x / MAP_SIZE.y);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_and_screen_agree() {
        let (w, h) = GRID_CELLS;
        assert_eq!(cell_px(w), 4.0);
        assert_eq!(cell_px(CAR_CELLS.0), 8.0);
        // The middle of cell (10, 20) maps back to it.
        let p = cell_to_screen(Vec2::new(10.5, 20.5), w);
        assert_eq!(screen_to_cell(p, w, h), Some((10, 20)));
        assert_eq!(screen_to_cell(MAP_MIN - Vec2::ONE, w, h), None);
        assert_eq!(screen_to_cell(MAP_MIN + MAP_SIZE, w, h), None);
    }
}
