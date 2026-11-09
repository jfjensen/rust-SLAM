// Unit NAV, part 1: the planning map, built from what the robot KNOWS.
//
// A grid (unit_pp's Grid, so the planners from av08 run on it) of the
// room, with each landmark of the SLAM map drawn in as a blocked disc:
//
//   radius = cylinder + robot + margin + 2 sigma of the landmark's position
//
// The first three make a path for a POINT a path for the robot's centre
// (av08's arena preset does the same). The last one is new: a landmark the
// filter isn't sure about is a bigger obstacle, and shrinks as SLAM gets
// surer. The sigma is capped: a landmark seen once has a sigma of metres.
//
// The room's outline is given (a known room, unknown furniture): EKF SLAM's
// map has point landmarks, no walls.

use crate::unit_d::ekf::ErrorEllipse;
use crate::unit_pp::grid::Grid;
use bevy::math::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapSetup {
    /// Millimetres per cell.
    pub cell_mm: f32,
    /// The room: walls along x = 0, x = room.x, y = 0, y = room.y.
    pub room: Vec2,
    pub robot_radius: f32,
    pub cylinder_radius: f32,
    /// Extra room for steering errors.
    pub margin: f32,
    /// How many sigmas of a landmark's uncertainty are added.
    pub sigmas: f32,
    /// A landmark's sigma counts at most this much (mm).
    pub max_sigma: f32,
}

impl Default for MapSetup {
    fn default() -> MapSetup {
        MapSetup {
            cell_mm: 20.0,
            room: Vec2::splat(crate::world::ARENA_SIZE),
            robot_radius: crate::synth::ROBOT_RADIUS,
            cylinder_radius: crate::world::CYLINDER_DIAMETER / 2.0,
            margin: 40.0,
            sigmas: 2.0,
            max_sigma: 150.0,
        }
    }
}

impl MapSetup {
    /// The blocked disc's radius round a landmark with this ellipse.
    pub fn disc_radius(&self, e: &ErrorEllipse) -> f32 {
        let sigma = (e.major as f32).clamp(0.0, self.max_sigma);
        self.cylinder_radius + self.robot_radius + self.margin + self.sigmas * sigma
    }

    /// How far from the walls the robot's centre must stay.
    pub fn wall_clearance(&self) -> f32 {
        self.robot_radius + self.margin
    }
}

/// A cell: (x, y), x to the right, y up (as in unit_pp).
pub type Cell = (i32, i32);

#[derive(Clone, Debug, PartialEq)]
pub struct PlanMap {
    pub grid: Grid,
    pub cell_mm: f32,
    /// Each landmark's disc: centre and radius (mm), for drawing.
    pub discs: Vec<(Vec2, f32)>,
}

impl PlanMap {
    pub fn build(landmarks: &[Vec2], ellipses: &[ErrorEllipse], s: &MapSetup) -> PlanMap {
        let width = (s.room.x / s.cell_mm).ceil() as usize;
        let height = (s.room.y / s.cell_mm).ceil() as usize;
        let discs: Vec<(Vec2, f32)> = landmarks
            .iter()
            .zip(ellipses)
            .map(|(&p, e)| (p, s.disc_radius(e)))
            .collect();
        let mut grid = Grid::new(width, height);
        let edge = s.wall_clearance();
        for y in 0..height {
            for x in 0..width {
                let p = Vec2::new((x as f32 + 0.5) * s.cell_mm, (y as f32 + 0.5) * s.cell_mm);
                let near_wall =
                    p.x < edge || p.y < edge || p.x > s.room.x - edge || p.y > s.room.y - edge;
                let in_disc = discs.iter().any(|&(c, r)| c.distance(p) < r);
                grid.walls[x + y * width] = near_wall || in_disc;
            }
        }
        PlanMap {
            grid,
            cell_mm: s.cell_mm,
            discs,
        }
    }

    pub fn to_cell(&self, p: Vec2) -> Cell {
        let c = (p / self.cell_mm).floor();
        (c.x as i32, c.y as i32)
    }

    /// A cell's centre in mm.
    pub fn to_world(&self, (x, y): Cell) -> Vec2 {
        Vec2::new(x as f32 + 0.5, y as f32 + 0.5) * self.cell_mm
    }

    /// Outside the grid counts as blocked.
    pub fn blocked(&self, (x, y): Cell) -> bool {
        !self.grid.contains(x, y) || self.grid.is_wall(x, y)
    }

    pub fn blocked_at(&self, p: Vec2) -> bool {
        self.blocked(self.to_cell(p))
    }

    /// The free cell nearest to this one (square rings outward, the
    /// nearest by distance within the first ring that has any), at most
    /// `max_rings` away.
    pub fn nearest_free(&self, from: Cell, max_rings: i32) -> Option<Cell> {
        if !self.blocked(from) {
            return Some(from);
        }
        for r in 1..=max_rings {
            let ring = (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)));
            let best = ring
                .filter(|&(dx, dy)| dx.abs() == r || dy.abs() == r)
                .map(|(dx, dy)| (from.0 + dx, from.1 + dy))
                .filter(|&c| !self.blocked(c))
                .min_by_key(|&c| (c.0 - from.0).pow(2) + (c.1 - from.1).pow(2));
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// Every cell the planner could reach from `from` (8 neighbours, as
    /// unit_pp's search moves): a flood fill. Indexed like grid.walls.
    pub fn reachable(&self, from: Cell) -> Vec<bool> {
        let mut seen = vec![false; self.grid.walls.len()];
        if self.blocked(from) {
            return seen;
        }
        let mut todo = vec![from];
        seen[self.grid.index(from.0, from.1)] = true;
        while let Some((x, y)) = todo.pop() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let c = (x + dx, y + dy);
                    if !self.blocked(c) && !seen[self.grid.index(c.0, c.1)] {
                        seen[self.grid.index(c.0, c.1)] = true;
                        todo.push(c);
                    }
                }
            }
        }
        seen
    }

    /// Can the robot's centre go in a straight line from a to b? Checked
    /// every half cell.
    pub fn line_free(&self, a: Vec2, b: Vec2) -> bool {
        let n = (a.distance(b) / (self.cell_mm / 2.0)).ceil().max(1.0) as usize;
        (0..=n).all(|i| !self.blocked_at(a.lerp(b, i as f32 / n as f32)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ellipse(sigma: f64) -> ErrorEllipse {
        ErrorEllipse {
            angle: 0.0,
            major: sigma,
            minor: sigma,
        }
    }

    #[test]
    fn uncertain_landmarks_block_more() {
        let s = MapSetup::default();
        let at = Vec2::new(1000.0, 1000.0);
        let sure = PlanMap::build(&[at], &[ellipse(5.0)], &s);
        let unsure = PlanMap::build(&[at], &[ellipse(80.0)], &s);
        let count = |m: &PlanMap| m.grid.walls.iter().filter(|w| **w).count();
        assert!(count(&unsure) > count(&sure));
        // 155 mm from the centre: inside the sure disc (145 + 10) ...
        let p = at + Vec2::new(150.0, 0.0);
        assert!(sure.blocked_at(p));
        // ... and 200 mm from it is outside the sure one, inside the unsure.
        let p = at + Vec2::new(200.0, 0.0);
        assert!(!sure.blocked_at(p) && unsure.blocked_at(p));
        // A huge sigma counts as the cap.
        let huge = PlanMap::build(&[at], &[ellipse(1e5)], &s);
        assert_eq!(huge.discs[0].1, s.disc_radius(&ellipse(150.0)));
    }

    #[test]
    fn the_walls_are_grown_and_the_middle_is_free() {
        let m = PlanMap::build(&[], &[], &MapSetup::default());
        assert_eq!((m.grid.width, m.grid.height), (100, 100));
        assert!(m.blocked_at(Vec2::new(50.0, 1000.0)));
        assert!(m.blocked_at(Vec2::new(1000.0, 1950.0)));
        assert!(!m.blocked_at(Vec2::new(150.0, 1000.0)));
        assert!(m.blocked((-1, 5)) && m.blocked((100, 5)));
        // 120 mm (robot 80 + margin 40): 6 cells along each wall.
        assert_eq!(
            m.grid.walls.iter().filter(|w| **w).count(),
            100 * 100 - 88 * 88
        );
    }

    #[test]
    fn nearest_free_and_lines() {
        let s = MapSetup::default();
        let at = Vec2::new(1000.0, 1000.0);
        let m = PlanMap::build(&[at], &[ellipse(0.0)], &s);
        let free = m.nearest_free(m.to_cell(at), 20).unwrap();
        assert!(!m.blocked(free));
        // The disc is 145 mm: the nearest free cell is just outside it.
        let d = m.to_world(free).distance(at);
        assert!((145.0..175.0).contains(&d), "{d}");
        assert!(!m.line_free(Vec2::new(500.0, 1000.0), Vec2::new(1500.0, 1000.0)));
        assert!(m.line_free(Vec2::new(500.0, 1300.0), Vec2::new(1500.0, 1300.0)));
    }

    #[test]
    fn reachable_stops_at_a_wall_across_the_room() {
        // Three sure landmarks in a row across the middle, close enough
        // for their discs (145 mm) to overlap with the walls' 100 mm.
        let row: Vec<Vec2> = [300.0, 700.0, 1000.0, 1300.0, 1700.0]
            .iter()
            .map(|&x| Vec2::new(x, 1000.0))
            .collect();
        let m = PlanMap::build(&row, &[ellipse(40.0); 5], &MapSetup::default());
        let below = m.reachable(m.to_cell(Vec2::new(1000.0, 400.0)));
        let above = m.to_cell(Vec2::new(1000.0, 1600.0));
        let under = m.to_cell(Vec2::new(200.0, 300.0));
        assert!(!below[m.grid.index(above.0, above.1)]);
        assert!(below[m.grid.index(under.0, under.1)]);
    }
}
