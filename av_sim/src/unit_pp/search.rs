// Unit PP, part 2: searching the grid (the course's pp_01_a..f).
//
//   Dijkstra   take the cheapest cell in the front (a heap), mark it
//              visited, put its unvisited neighbours in the front with
//              their cost so far; stop at the goal. The front spreads like
//              a ring of water: every cell nearer than the goal is visited.
//   A*         the same, but the front is ordered by cost so far PLUS the
//              straight-line distance still to go. It never overestimates,
//              so the path is still the cheapest, and the front leans
//              towards the goal: far fewer cells visited.
//   potential  either one, with each cell's cost / 64 added to entering
//              it (grid.rs): the path keeps away from walls when it can.
//
// 8 neighbours: 1 straight, sqrt(2) diagonal (as in the course; diagonal
// steps may squeeze between two wall cells that touch at a corner).
//
// The search is a struct with step() (expand ONE cell) so the app can
// animate it, and run() (to the end).

use super::grid::{BLOCKED, Grid};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Dijkstra,
    AStar,
}

/// The course's movements: (dx, dy, cost).
const MOVES: [(i32, i32, f64); 8] = [
    (1, 0, 1.0),
    (0, 1, 1.0),
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f64::consts::SQRT_2),
    (-1, 1, std::f64::consts::SQRT_2),
    (-1, -1, std::f64::consts::SQRT_2),
    (1, -1, std::f64::consts::SQRT_2),
];

/// The course's divider: a cell's cost / 64 is added to entering it.
pub const POTENTIAL_DIVIDER: f64 = 64.0;

/// One entry in the front.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Entry {
    /// What the heap orders by: cost so far (+ estimate to go, for A*).
    priority: f64,
    cost: f64,
    cell: (i32, i32),
    from: Option<(i32, i32)>,
}

// BinaryHeap is a MAX-heap: reverse the order so the cheapest comes out.
// (Python's heapq is a min-heap, so the course doesn't need this.)
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.total_cmp(&self.priority)
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Searching,
    Found,
    /// The front ran empty: the goal can't be reached.
    NoPath,
}

#[derive(Clone, Debug)]
pub struct Search {
    pub algorithm: Algorithm,
    width: usize,
    height: usize,
    costs: Vec<u8>,
    goal: (i32, i32),
    front: BinaryHeap<Entry>,
    /// The cost each visited cell was reached with (None: not visited).
    pub visited: Vec<Option<f64>>,
    /// Where each visited cell was reached from (the course's came_from).
    came_from: Vec<Option<(i32, i32)>>,
    pub expanded: usize,
    pub status: Status,
    /// The cell expanded last (for drawing).
    pub current: Option<(i32, i32)>,
}

impl Search {
    /// `costs`: grid.plain_costs() or grid.potential_costs().
    pub fn new(
        grid: &Grid,
        costs: Vec<u8>,
        start: (i32, i32),
        goal: (i32, i32),
        algorithm: Algorithm,
    ) -> Search {
        let n = grid.width * grid.height;
        let mut search = Search {
            algorithm,
            width: grid.width,
            height: grid.height,
            costs,
            goal,
            front: BinaryHeap::new(),
            visited: vec![None; n],
            came_from: vec![None; n],
            expanded: 0,
            status: Status::Searching,
            current: None,
        };
        let blocked = |c: (i32, i32)| !search.inside(c) || search.costs[search.index(c)] == BLOCKED;
        if blocked(start) || blocked(goal) {
            search.status = Status::NoPath;
        } else {
            search.front.push(Entry {
                priority: search.estimate(start),
                cost: 0.0,
                cell: start,
                from: None,
            });
        }
        search
    }

    fn inside(&self, (x, y): (i32, i32)) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    fn index(&self, (x, y): (i32, i32)) -> usize {
        x as usize + y as usize * self.width
    }

    /// A*'s estimate of the cost still to go: the straight-line distance.
    /// Never more than the real cost (every step costs at least its
    /// length), so A* still finds the cheapest path. 0 for Dijkstra.
    fn estimate(&self, (x, y): (i32, i32)) -> f64 {
        match self.algorithm {
            Algorithm::Dijkstra => 0.0,
            Algorithm::AStar => {
                (((x - self.goal.0).pow(2) + (y - self.goal.1).pow(2)) as f64).sqrt()
            }
        }
    }

    /// Expand one cell. Returns the status after it.
    pub fn step(&mut self) -> Status {
        if self.status != Status::Searching {
            return self.status;
        }
        // Take the cheapest; skip cells already visited (they can be in the
        // front several times, from different neighbours).
        let entry = loop {
            match self.front.pop() {
                None => {
                    self.status = Status::NoPath;
                    return self.status;
                }
                Some(e) if self.visited[self.index(e.cell)].is_some() => continue,
                Some(e) => break e,
            }
        };
        let i = self.index(entry.cell);
        self.visited[i] = Some(entry.cost);
        self.came_from[i] = entry.from;
        self.expanded += 1;
        self.current = Some(entry.cell);
        if entry.cell == self.goal {
            self.status = Status::Found;
            return self.status;
        }
        for (dx, dy, step) in MOVES {
            let next = (entry.cell.0 + dx, entry.cell.1 + dy);
            if !self.inside(next) {
                continue;
            }
            let j = self.index(next);
            if self.visited[j].is_some() || self.costs[j] == BLOCKED {
                continue;
            }
            let cost = entry.cost + step + self.costs[j] as f64 / POTENTIAL_DIVIDER;
            self.front.push(Entry {
                priority: cost + self.estimate(next),
                cost,
                cell: next,
                from: Some(entry.cell),
            });
        }
        self.status
    }

    /// Search to the end.
    pub fn run(&mut self) -> Status {
        while self.step() == Status::Searching {}
        self.status
    }

    /// The path, start to goal, once found (the course's unwinding of
    /// came_from); empty otherwise.
    pub fn path(&self) -> Vec<(i32, i32)> {
        if self.status != Status::Found {
            return Vec::new();
        }
        let mut path = vec![self.goal];
        let mut at = self.goal;
        while let Some(prev) = self.came_from[self.index(at)] {
            path.push(prev);
            at = prev;
        }
        path.reverse();
        path
    }

    /// The path's cost (as searched: with the potential, if it was used).
    pub fn path_cost(&self) -> Option<f64> {
        (self.status == Status::Found).then(|| self.visited[self.index(self.goal)].unwrap_or(0.0))
    }

    /// The cells in the front now (for drawing), each once.
    pub fn front_cells(&self) -> Vec<(i32, i32)> {
        let mut cells: Vec<(i32, i32)> = self
            .front
            .iter()
            .map(|e| e.cell)
            .filter(|&c| self.visited[self.index(c)].is_none())
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// The largest cost of any visited cell (for shading).
    pub fn max_visited(&self) -> f64 {
        self.visited.iter().flatten().copied().fold(0.0, f64::max)
    }
}

/// A path's length in cells (1 per straight step, sqrt(2) per diagonal).
pub fn path_length(path: &[(i32, i32)]) -> f64 {
    path.windows(2)
        .map(|w| {
            let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            ((dx * dx + dy * dy) as f64).sqrt()
        })
        .sum()
}

/// The smallest distance (cells) from any cell of the path to a wall.
pub fn clearance(path: &[(i32, i32)], grid: &Grid) -> f64 {
    let d = super::grid::distance_transform(&grid.walls, grid.width, grid.height);
    path.iter()
        .map(|&c| d[grid.index(c.0, c.1)])
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(grid: &Grid, costs: Vec<u8>, algorithm: Algorithm) -> Search {
        let mut s = Search::new(grid, costs, (5, 5), (190, 140), algorithm);
        s.run();
        s
    }

    /// On random maps, A* finds a path exactly as cheap as Dijkstra's and
    /// never visits more cells.
    #[test]
    fn a_star_is_as_good_and_does_less() {
        for seed in 1..=5 {
            let mut grid = Grid::random(200, 150, 250, seed);
            grid.paint(5, 5, false);
            grid.paint(190, 140, false);
            let d = solve(&grid, grid.plain_costs(), Algorithm::Dijkstra);
            let a = solve(&grid, grid.plain_costs(), Algorithm::AStar);
            assert_eq!(d.status, Status::Found, "seed {seed}");
            let (dc, ac) = (d.path_cost().unwrap(), a.path_cost().unwrap());
            assert!((dc - ac).abs() < 1e-9, "seed {seed}: {dc} vs {ac}");
            assert!(a.expanded <= d.expanded, "seed {seed}");
            // The path really is one: neighbours, no walls.
            let p = a.path();
            assert!(
                p.windows(2)
                    .all(|w| (w[0].0 - w[1].0).abs() <= 1 && (w[0].1 - w[1].1).abs() <= 1)
            );
            assert!(p.iter().all(|&(x, y)| !grid.is_wall(x, y)));
            assert!((path_length(&p) - ac).abs() < 1e-9);
        }
    }

    /// With the potential function, the path keeps further from walls.
    /// Start and goal are both BELOW a gap in a wall, so the shortest path
    /// cuts round the gap's lower corner; the potential pushes it off.
    #[test]
    fn the_potential_keeps_away_from_walls() {
        let mut grid = Grid::new(200, 150);
        for y in 0..150 {
            if !(60..110).contains(&y) {
                grid.paint(100, y, true);
            }
        }
        let run = |costs: Vec<u8>| {
            let mut s = Search::new(&grid, costs, (5, 20), (190, 20), Algorithm::AStar);
            assert_eq!(s.run(), Status::Found);
            s.path()
        };
        let plain = run(grid.plain_costs());
        let with = run(grid.potential_costs());
        let (c0, c1) = (clearance(&plain, &grid), clearance(&with, &grid));
        assert!(c0 < 2.0, "the plain path hugs the corner: {c0}");
        assert!(c1 > c0 + 3.0, "{c0} -> {c1}");
        // ... and pays for it in length.
        assert!(path_length(&with) > path_length(&plain));
    }

    #[test]
    fn walled_in_means_no_path() {
        let mut grid = Grid::new(60, 60);
        for x in 0..60 {
            grid.paint(x, 30, true);
        }
        let mut s = Search::new(
            &grid,
            grid.plain_costs(),
            (10, 10),
            (10, 50),
            Algorithm::AStar,
        );
        assert_eq!(s.run(), Status::NoPath);
        assert!(s.path().is_empty());
        // Start on a wall: no search at all.
        let s = Search::new(
            &grid,
            grid.plain_costs(),
            (10, 30),
            (10, 50),
            Algorithm::AStar,
        );
        assert_eq!(s.status, Status::NoPath);
    }

    /// Stepping by hand ends exactly where run() does.
    #[test]
    fn stepping_is_running() {
        let mut grid = Grid::random(80, 60, 40, 2);
        grid.paint(1, 1, false);
        grid.paint(75, 55, false);
        let mut a = Search::new(
            &grid,
            grid.plain_costs(),
            (1, 1),
            (75, 55),
            Algorithm::Dijkstra,
        );
        let mut b = a.clone();
        a.run();
        let mut steps = 0;
        while b.step() == Status::Searching {
            steps += 1;
        }
        assert_eq!(a.path(), b.path());
        assert_eq!(a.expanded, steps + 1);
    }
}
