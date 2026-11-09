// Unit PP, part 1: the world the planners search, a grid of cells (the
// course's world_obstacles array and common.py).
//
// Each cell holds a cost from 0 to 255:
//   255      an obstacle: never entered
//   1..254   free, but with a price (the potential function: near walls)
//   0        free
//
// The planners only read costs. Where they come from is up to the grid:
// painted walls alone, or walls plus the potential function.
//
// Cell (x, y): x to the right, y UP, both from 0. Cell i = x + y * width.

use crate::rng::Rng;
use crate::world::Arena;

/// A cell that can't be entered.
pub const BLOCKED: u8 = 255;

/// The course's brush: a (2N + 1) x (2N + 1) box, N = 2.
pub const BRUSH: i32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    /// True for a wall cell.
    pub walls: Vec<bool>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Grid {
        Grid {
            width,
            height,
            walls: vec![false; width * height],
        }
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    pub fn index(&self, x: i32, y: i32) -> usize {
        x as usize + y as usize * self.width
    }

    pub fn is_wall(&self, x: i32, y: i32) -> bool {
        self.contains(x, y) && self.walls[self.index(x, y)]
    }

    /// Paint (or erase) the brush box around a cell, clipped to the grid
    /// (the course's set_obstacle).
    pub fn paint(&mut self, x: i32, y: i32, on: bool) {
        for dy in -BRUSH..=BRUSH {
            for dx in -BRUSH..=BRUSH {
                if self.contains(x + dx, y + dy) {
                    let i = self.index(x + dx, y + dy);
                    self.walls[i] = on;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.walls.fill(false);
    }

    /// `count` brush boxes at random places.
    pub fn random(width: usize, height: usize, count: usize, seed: u64) -> Grid {
        let mut grid = Grid::new(width, height);
        let mut rng = Rng::new(seed);
        for _ in 0..count {
            let x = (rng.unit() * width as f64) as i32;
            let y = (rng.unit() * height as f64) as i32;
            grid.paint(x, y, true);
        }
        grid
    }

    /// The SLAM arena (Units A-G) on a width x height grid: square cells,
    /// as many as fit the arena along the shorter side; columns or rows
    /// left over are walls (outside the arena). Walls and cylinders are
    /// grown by `clearance` mm (the robot's half-width), so that a path
    /// for a POINT is a path for the robot's centre.
    pub fn arena(width: usize, height: usize, arena: &Arena, clearance: f32) -> Grid {
        let cells = width.min(height);
        let mm = arena.size.x / cells as f32;
        let mut grid = Grid::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let wall = if x >= cells || y >= cells {
                    true
                } else {
                    let p = bevy::math::Vec2::new((x as f32 + 0.5) * mm, (y as f32 + 0.5) * mm);
                    let near_wall = p.x < clearance
                        || p.y < clearance
                        || p.x > arena.size.x - clearance
                        || p.y > arena.size.y - clearance;
                    let near_cylinder = arena
                        .landmarks
                        .iter()
                        .any(|l| l.position.distance(p) < l.diameter / 2.0 + clearance);
                    near_wall || near_cylinder
                };
                grid.walls[x + y * width] = wall;
            }
        }
        grid
    }

    /// The arena preset's millimetres per cell (see `arena`).
    pub fn arena_cell_mm(width: usize, height: usize, arena: &Arena) -> f32 {
        arena.size.x / width.min(height) as f32
    }

    /// The costs the planners read: walls only.
    pub fn plain_costs(&self) -> Vec<u8> {
        self.walls
            .iter()
            .map(|&w| if w { BLOCKED } else { 0 })
            .collect()
    }

    /// The costs with the potential function (the course's pp_01_f
    /// apply_distance_transform): 255 at walls, falling by 16 per cell of
    /// distance, 0 from 16 cells away. With no walls at all: all 0.
    pub fn potential_costs(&self) -> Vec<u8> {
        if !self.walls.contains(&true) {
            return vec![0; self.walls.len()];
        }
        distance_transform(&self.walls, self.width, self.height)
            .into_iter()
            .map(|d| (255.0 - (16.0 * d).min(255.0)) as u8)
            .collect()
    }
}

/// Large enough to stand for "no wall in this row/column yet".
const FAR: f64 = 1e20;

/// The exact Euclidean distance, in cells, from every cell to the nearest
/// wall cell (0 on walls). The course uses scipy's distance_transform_edt;
/// this is the two-pass method of Felzenszwalb and Huttenlocher (2012):
/// squared distances along each column, then along each row, each pass a
/// lower envelope of parabolas. O(cells).
pub fn distance_transform(walls: &[bool], width: usize, height: usize) -> Vec<f64> {
    let mut d2: Vec<f64> = walls.iter().map(|&w| if w { 0.0 } else { FAR }).collect();
    // Columns.
    let mut column = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            column[y] = d2[x + y * width];
        }
        let out = envelope_1d(&column);
        for y in 0..height {
            d2[x + y * width] = out[y];
        }
    }
    // Rows.
    for y in 0..height {
        let row = &mut d2[y * width..(y + 1) * width];
        let out = envelope_1d(row);
        row.copy_from_slice(&out);
    }
    d2.into_iter().map(f64::sqrt).collect()
}

/// One line of squared distances: out[q] = min over p of (q - p)^2 + f[p].
/// Each p gives a parabola; walk them left to right keeping the ones that
/// are lowest somewhere (v), and where each takes over from the last (z).
fn envelope_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0.0; n + 1];
    let mut k = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    // Where the parabolas from q and p cross.
    let crossing = |q: usize, p: usize| {
        ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64))
    };
    for q in 1..n {
        let mut s = crossing(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = crossing(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    let mut out = vec![0.0; n];
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        *o = (q as f64 - p as f64).powi(2) + f[p];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast transform against measuring every distance.
    #[test]
    fn distance_transform_matches_brute_force() {
        let grid = Grid::random(37, 23, 12, 4);
        let fast = distance_transform(&grid.walls, grid.width, grid.height);
        let walls: Vec<(i32, i32)> = (0..grid.height as i32)
            .flat_map(|y| (0..grid.width as i32).map(move |x| (x, y)))
            .filter(|&(x, y)| grid.is_wall(x, y))
            .collect();
        for y in 0..grid.height as i32 {
            for x in 0..grid.width as i32 {
                let slow = walls
                    .iter()
                    .map(|&(wx, wy)| (((wx - x).pow(2) + (wy - y).pow(2)) as f64).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let got = fast[grid.index(x, y)];
                assert!((got - slow).abs() < 1e-9, "({x}, {y}): {got} vs {slow}");
            }
        }
    }

    #[test]
    fn potential_falls_off_from_the_walls() {
        let mut grid = Grid::new(40, 5);
        for y in 0..5 {
            let i = grid.index(0, y);
            grid.walls[i] = true;
        }
        let c = grid.potential_costs();
        assert_eq!(c[grid.index(0, 2)], 255);
        assert_eq!(c[grid.index(1, 2)], 255 - 16);
        assert_eq!(c[grid.index(10, 2)], 255 - 160);
        assert_eq!(c[grid.index(16, 2)], 0);
        assert_eq!(Grid::new(4, 4).potential_costs(), vec![0; 16]);
    }

    #[test]
    fn the_brush_is_five_by_five_and_clipped() {
        let mut grid = Grid::new(10, 10);
        grid.paint(5, 5, true);
        assert_eq!(grid.walls.iter().filter(|&&w| w).count(), 25);
        grid.paint(0, 0, true);
        assert_eq!(grid.walls.iter().filter(|&&w| w).count(), 25 + 9);
        grid.paint(5, 5, false);
        assert_eq!(grid.walls.iter().filter(|&&w| w).count(), 9);
    }

    /// The arena preset: walls round the edge, a blob at each cylinder,
    /// free space between, and the leftover columns walled off.
    #[test]
    fn the_arena_preset() {
        let arena = Arena::course();
        let grid = Grid::arena(200, 150, &arena, 75.0);
        let mm = Grid::arena_cell_mm(200, 150, &arena);
        let cell = |x: f32, y: f32| ((x / mm) as i32, (y / mm) as i32);
        for l in &arena.landmarks {
            let (x, y) = cell(l.position.x, l.position.y);
            assert!(grid.is_wall(x, y), "{:?}", l.position);
        }
        let (x, y) = cell(1000.0, 1300.0);
        assert!(!grid.is_wall(x, y));
        assert!(grid.is_wall(0, 75) && grid.is_wall(160, 75));
    }
}
