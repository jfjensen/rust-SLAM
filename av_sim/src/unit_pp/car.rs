// Unit PP, part 3: a car (the course's pp_02_a/b).
//
// A grid search moves a POINT to any neighbour. A car can't turn on the
// spot: from a pose (x, y, heading) it can only drive an arc. The course
// gives it three (or six) moves, each 5 cells long:
//
//   curvature  1/10 (left, radius 10), 0 (straight), -1/10 (right)
//   length     +5 forward (and, if allowed, -5 backward)
//
// A* then searches over POSES. Poses are real numbers, so "visited" means
// "a pose in the same cell and the same 10-degree heading slice" (the
// course's pose_index). The goal is reached when within 2 cells and 15
// degrees (states_close). The estimate to go is the straight-line distance:
// a car can't do better than a straight line.
//
// NOTE, two bugs in the repo's pp_02_b, fixed here:
//   - it stores expanded states under `pose` but reads them back under
//     `pose_index(pose)`: rebuilding the path would fail on a missing key
//   - it never checks whether a pose_index was expanded already, so the
//     same few states are expanded over and over until the 500,000 limit
//     (CarSearch::closed_set = false shows that)
// NOTE, and one difference: the course checks only each arc's END cell for
// an obstacle; an arc is 5 cells long and can cut a corner of a 5 x 5
// obstacle box. Here points every half cell along the arc are checked.

use super::grid::Grid;
use crate::unit_d::ekf::wrap_pi;
use bevy::math::{DVec2, DVec3};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// (curvature, length): forward left, straight, right, then backward.
pub const MOVES: [(f64, f64); 6] = [
    (0.1, 5.0),
    (0.0, 5.0),
    (-0.1, 5.0),
    (0.1, -5.0),
    (0.0, -5.0),
    (-0.1, -5.0),
];

/// The course's states_close tolerances.
pub const GOAL_DISTANCE: f64 = 2.0;
pub const GOAL_ANGLE_DEGREES: f64 = 15.0;
/// The course's pose_index rasters.
const POSITION_RASTER: f64 = 1.0;
const HEADING_RASTER_DEGREES: f64 = 10.0;
/// Collision check spacing along an arc, cells.
const CHECK_EVERY: f64 = 0.5;
/// The course's timeout.
pub const NODE_LIMIT: usize = 500_000;

/// The end of an arc from `pose` (the course's CurveSegment.end_pose).
pub fn end_pose(pose: DVec3, curvature: f64, length: f64) -> DVec3 {
    let (x, y, theta) = (pose.x, pose.y, pose.z);
    if curvature == 0.0 {
        return DVec3::new(x + length * theta.cos(), y + length * theta.sin(), theta);
    }
    let (tx, ty) = (theta.cos(), theta.sin());
    let radius = 1.0 / curvature;
    // The circle's centre, to the left for positive curvature.
    let (xc, yc) = (x - radius * ty, y + radius * tx);
    let angle = length / radius;
    let (cosa, sina) = (angle.cos(), angle.sin());
    DVec3::new(
        xc + radius * (cosa * ty + sina * tx),
        yc + radius * (sina * ty - cosa * tx),
        wrap_pi(theta + angle),
    )
}

/// Points along an arc, `every` apart, start included, end included.
pub fn arc_points(pose: DVec3, curvature: f64, length: f64, every: f64) -> Vec<DVec2> {
    let n = (length.abs() / every).ceil().max(1.0) as usize;
    (0..=n)
        .map(|k| end_pose(pose, curvature, length * k as f64 / n as f64).truncate())
        .collect()
}

/// The course's pose_index: cell and 10-degree heading slice.
pub fn pose_index(pose: DVec3) -> (i32, i32, i32) {
    (
        (pose.x / POSITION_RASTER).floor() as i32,
        (pose.y / POSITION_RASTER).floor() as i32,
        (pose.z / HEADING_RASTER_DEGREES.to_radians()).floor() as i32,
    )
}

/// The course's states_close.
pub fn states_close(p: DVec3, q: DVec3) -> bool {
    wrap_pi(p.z - q.z).abs() < GOAL_ANGLE_DEGREES.to_radians()
        && p.truncate().distance(q.truncate()) <= GOAL_DISTANCE
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Entry {
    priority: f64,
    cost: f64,
    pose: DVec3,
    /// The pose this came from, and by which move.
    from: Option<(DVec3, usize)>,
}

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

pub use super::search::Status;

/// A pose_index (cell and heading slice).
type StateKey = (i32, i32, i32);
/// An expanded state: its pose, and the pose and move it came from.
type Expanded = (DVec3, Option<(DVec3, usize)>);

/// One arc of the found path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub start: DVec3,
    pub curvature: f64,
    pub length: f64,
}

#[derive(Clone, Debug)]
pub struct CarSearch {
    grid: Grid,
    goal: DVec3,
    /// 3: forward only. 6: forward and backward.
    pub moves: usize,
    /// Skip a pose whose pose_index was expanded already (the fix).
    pub closed_set: bool,
    front: BinaryHeap<Entry>,
    /// Expanded states: pose_index -> (the pose, where it came from).
    expanded_states: HashMap<StateKey, Expanded>,
    /// The cost each cell was first expanded with (for shading).
    pub visited: Vec<Option<f64>>,
    pub expanded: usize,
    pub status: Status,
    /// The pose that reached the goal.
    reached: Option<DVec3>,
    pub current: Option<DVec3>,
}

impl CarSearch {
    pub fn new(grid: &Grid, start: DVec3, goal: DVec3, backward: bool) -> CarSearch {
        let mut s = CarSearch {
            grid: grid.clone(),
            goal,
            moves: if backward { 6 } else { 3 },
            closed_set: true,
            front: BinaryHeap::new(),
            expanded_states: HashMap::new(),
            visited: vec![None; grid.width * grid.height],
            expanded: 0,
            status: Status::Searching,
            reached: None,
            current: None,
        };
        if !s.free(start.truncate()) || !s.free(goal.truncate()) {
            s.status = Status::NoPath;
        } else {
            s.front.push(Entry {
                priority: start.truncate().distance(goal.truncate()),
                cost: 0.0,
                pose: start,
                from: None,
            });
        }
        s
    }

    fn free(&self, p: DVec2) -> bool {
        let (x, y) = (p.x.floor() as i32, p.y.floor() as i32);
        self.grid.contains(x, y) && !self.grid.is_wall(x, y)
    }

    pub fn step(&mut self) -> Status {
        if self.status != Status::Searching {
            return self.status;
        }
        if self.expanded >= NODE_LIMIT {
            self.status = Status::NoPath;
            return self.status;
        }
        let entry = loop {
            match self.front.pop() {
                None => {
                    self.status = Status::NoPath;
                    return self.status;
                }
                Some(e)
                    if self.closed_set
                        && self.expanded_states.contains_key(&pose_index(e.pose)) =>
                {
                    continue;
                }
                Some(e) => break e,
            }
        };
        self.expanded_states
            .insert(pose_index(entry.pose), (entry.pose, entry.from));
        let cell = self.grid.index(entry.pose.x as i32, entry.pose.y as i32);
        self.visited[cell].get_or_insert(entry.cost);
        self.expanded += 1;
        self.current = Some(entry.pose);
        if states_close(entry.pose, self.goal) {
            self.reached = Some(entry.pose);
            self.status = Status::Found;
            return self.status;
        }
        for (m, &(curvature, length)) in MOVES.iter().enumerate().take(self.moves) {
            let next = end_pose(entry.pose, curvature, length);
            if self.closed_set && self.expanded_states.contains_key(&pose_index(next)) {
                continue;
            }
            if !arc_points(entry.pose, curvature, length, CHECK_EVERY)
                .iter()
                .all(|&p| self.free(p))
            {
                continue;
            }
            let cost = entry.cost + length.abs();
            self.front.push(Entry {
                priority: cost + next.truncate().distance(self.goal.truncate()),
                cost,
                pose: next,
                from: Some((entry.pose, m)),
            });
        }
        self.status
    }

    pub fn run(&mut self) -> Status {
        while self.step() == Status::Searching {}
        self.status
    }

    /// The arcs from start to goal, once found.
    pub fn path(&self) -> Vec<Segment> {
        let Some(mut at) = self.reached else {
            return Vec::new();
        };
        let mut segments = Vec::new();
        while let Some(&(_, Some((prev, m)))) = self.expanded_states.get(&pose_index(at)) {
            let (curvature, length) = MOVES[m];
            segments.push(Segment {
                start: prev,
                curvature,
                length,
            });
            at = prev;
        }
        segments.reverse();
        segments
    }

    pub fn path_length(&self) -> f64 {
        self.path().iter().map(|s| s.length.abs()).sum()
    }

    pub fn max_visited(&self) -> f64 {
        self.visited.iter().flatten().copied().fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn arcs_turn_the_right_way() {
        let start = DVec3::new(10.0, 10.0, 0.0);
        // A quarter circle to the left, radius 10: ends 10 ahead, 10 left,
        // facing up.
        let q = end_pose(start, 0.1, 10.0 * FRAC_PI_2);
        assert!(
            (q - DVec3::new(20.0, 20.0, FRAC_PI_2)).length() < 1e-9,
            "{q}"
        );
        // Backward along a straight line.
        let b = end_pose(start, 0.0, -5.0);
        assert!((b - DVec3::new(5.0, 10.0, 0.0)).length() < 1e-12);
        let pts = arc_points(start, -0.1, 5.0, 0.5);
        assert_eq!(pts.len(), 11);
        assert!(
            pts.iter().all(|p| p.y <= 10.0 + 1e-12),
            "right turn goes down"
        );
    }

    /// Every point of the found path is free, and it ends near the goal.
    #[test]
    fn the_path_is_drivable_and_arrives() {
        let mut grid = Grid::new(100, 75);
        for y in 0..50 {
            grid.paint(50, y, true);
        }
        let start = DVec3::new(10.0, 10.0, 0.0);
        let goal = DVec3::new(85.0, 10.0, -FRAC_PI_2);
        let mut s = CarSearch::new(&grid, start, goal, false);
        assert_eq!(s.run(), Status::Found);
        let path = s.path();
        assert!(!path.is_empty());
        for seg in &path {
            for p in arc_points(seg.start, seg.curvature, seg.length, 0.25) {
                assert!(!grid.is_wall(p.x.floor() as i32, p.y.floor() as i32), "{p}");
            }
        }
        let last = path.last().unwrap();
        let end = end_pose(last.start, last.curvature, last.length);
        assert!(states_close(end, goal), "{end}");
        assert!((path[0].start - start).length() < 1e-12);
    }

    /// A dead end: the car faces into a pocket too narrow to turn in.
    /// Forward only can't get out; with reversing it can.
    #[test]
    fn reversing_gets_out_of_a_dead_end() {
        let mut grid = Grid::new(100, 75);
        for x in 20..60 {
            grid.paint(x, 30, true);
            grid.paint(x, 44, true);
        }
        for y in 30..=44 {
            grid.paint(60, y, true);
        }
        let start = DVec3::new(50.0, 37.0, 0.0);
        let goal = DVec3::new(10.0, 37.0, std::f64::consts::PI - 1e-9);
        let mut forward = CarSearch::new(&grid, start, goal, false);
        assert_eq!(forward.run(), Status::NoPath);
        let mut both = CarSearch::new(&grid, start, goal, true);
        assert_eq!(both.run(), Status::Found);
        assert!(both.path().iter().any(|s| s.length < 0.0));
    }

    /// Without the closed set (the repo's version), the same search
    /// expands far more states: they're found again and again.
    #[test]
    fn the_closed_set_saves_work() {
        let grid = Grid::random(100, 75, 30, 3);
        let mut grid = grid;
        grid.paint(10, 10, false);
        grid.paint(85, 60, false);
        let start = DVec3::new(10.5, 10.5, 0.0);
        let goal = DVec3::new(85.5, 60.5, 0.0);
        let mut with = CarSearch::new(&grid, start, goal, false);
        assert_eq!(with.run(), Status::Found);
        let mut without = CarSearch::new(&grid, start, goal, false);
        without.closed_set = false;
        // Not all the way to 500,000 in a test: 10 times as many is enough.
        let limit = with.expanded * 10;
        while without.expanded < limit && without.step() == Status::Searching {}
        assert!(
            without.expanded >= limit,
            "{} vs {}",
            without.expanded,
            with.expanded
        );
    }
}
