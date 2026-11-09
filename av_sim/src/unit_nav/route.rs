// Unit NAV, part 2: a route on the planning map, and following it.
//
//   plan()     av08's A* with the potential function, from the robot to
//              the goal, then the cell path shortened by line of sight
//   pursue()   pure pursuit: steer at a point a fixed distance further
//              along the route, but not round a corner
//
// Why shorten the path? A* on a grid moves in 8 directions only, so its
// path zigzags (and the potential function bends it away from the discs).
// A robot following every cell would wobble. String pulling keeps the
// first point and jumps to the furthest later point it can see in a
// straight line, again and again. The discs already include the robot's
// size and a margin, so a straight line that misses them is safe; the
// potential's extra distance from the discs is given up where a straight
// line fits.

use super::planmap::{Cell, PlanMap};
use crate::robot::{Pose, ROBOT_WIDTH};
use crate::unit_pp::search::{Algorithm, Search, Status};
use bevy::math::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The robot is inside a blocked area and no free cell is near.
    StartBlocked,
    /// The goal is inside a blocked area (a disc, or too near a wall).
    GoalBlocked,
    /// No way round.
    NoPath,
}

impl PlanError {
    pub fn text(self) -> &'static str {
        match self {
            PlanError::StartBlocked => "the robot is boxed in",
            PlanError::GoalBlocked => "the goal is inside an obstacle",
            PlanError::NoPath => "no way to the goal",
        }
    }
}

/// How far (cells) to look for a free cell when the robot is inside a
/// disc: a landmark's uncertainty can grow round it.
const ESCAPE_RINGS: i32 = 15;

/// A route from `from` to `to` (mm), start and goal included.
pub fn plan(map: &PlanMap, from: Vec2, to: Vec2) -> Result<Vec<Vec2>, PlanError> {
    let goal = map.to_cell(to);
    if map.blocked(goal) {
        return Err(PlanError::GoalBlocked);
    }
    let start = map
        .nearest_free(map.to_cell(from), ESCAPE_RINGS)
        .ok_or(PlanError::StartBlocked)?;
    let mut search = Search::new(
        &map.grid,
        map.grid.potential_costs(),
        start,
        goal,
        Algorithm::AStar,
    );
    if search.run() != Status::Found {
        return Err(PlanError::NoPath);
    }
    let cells: Vec<Cell> = search.path();
    let mut points: Vec<Vec2> = cells.iter().map(|&c| map.to_world(c)).collect();
    // The real start and goal instead of their cells' centres. (When the
    // robot was inside a disc, the first leg takes it out to the free cell.)
    if map.to_cell(from) == start {
        points[0] = from;
    } else {
        points.insert(0, from);
    }
    let last = points.len() - 1;
    points[last] = to;
    Ok(shorten(&points, map))
}

/// String pulling: from each kept point, jump to the furthest point it
/// can see. The next point is always allowed (the first leg may start
/// inside a disc).
pub fn shorten(points: &[Vec2], map: &PlanMap) -> Vec<Vec2> {
    let Some(&first) = points.first() else {
        return Vec::new();
    };
    let mut out = vec![first];
    let mut i = 0;
    while i + 1 < points.len() {
        let mut j = i + 1;
        for k in (i + 2..points.len()).rev() {
            if map.line_free(points[i], points[k]) {
                j = k;
                break;
            }
        }
        out.push(points[j]);
        i = j;
    }
    out
}

pub fn route_length(points: &[Vec2]) -> f32 {
    points.windows(2).map(|w| w[0].distance(w[1])).sum()
}

// --- Following ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FollowSetup {
    /// Pure pursuit's look-ahead distance, mm.
    pub lookahead: f32,
    pub max_speed: f32,
    /// rad/s of turn per rad of heading error, and the most.
    pub turn_gain: f32,
    pub max_turn_rate: f32,
    /// The goal is reached this close (mm).
    pub arrive: f32,
    /// Slow down from this far from the goal.
    pub slow_down: f32,
    /// Go on to the next leg this close to a corner (mm).
    pub corner: f32,
}

impl Default for FollowSetup {
    fn default() -> FollowSetup {
        FollowSetup {
            lookahead: 250.0,
            max_speed: crate::synth::MAX_SPEED,
            turn_gain: 1.5,
            max_turn_rate: 1.2,
            arrive: 60.0,
            slow_down: 300.0,
            corner: 50.0,
        }
    }
}

/// What pure pursuit says to do this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steer {
    pub left: f32,
    pub right: f32,
    /// The point steered at.
    pub target: Vec2,
    pub arrived: bool,
}

/// The closest point to `p` on the segment a-b, as a fraction along it.
fn along(a: Vec2, b: Vec2, p: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 == 0.0 {
        0.0
    } else {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    }
}

/// Pure pursuit for a robot whose axle centre is at `centre`. `progress`
/// is the leg the robot is on: it only moves forward.
pub fn pursue(centre: Pose, route: &[Vec2], progress: &mut usize, s: &FollowSetup) -> Steer {
    let p = centre.position();
    let Some(&goal) = route.last() else {
        return Steer {
            left: 0.0,
            right: 0.0,
            target: p,
            arrived: true,
        };
    };
    if p.distance(goal) < s.arrive {
        return Steer {
            left: 0.0,
            right: 0.0,
            target: goal,
            arrived: true,
        };
    }
    // Which leg? The next one once the robot is near its corner.
    let legs = route.len().saturating_sub(1);
    while *progress + 1 < legs && p.distance(route[*progress + 1]) < s.corner {
        *progress += 1;
    }
    // Look `lookahead` ahead along this leg, but not past its end: a robot
    // that can turn on the spot needn't cut corners (cutting them is how
    // pure pursuit clips the obstacle a corner goes round).
    let target = match (route.get(*progress), route.get(*progress + 1)) {
        (Some(&a), Some(&b)) => {
            let at = a.lerp(b, along(a, b, p));
            let left = at.distance(b);
            if left > s.lookahead {
                at + (b - at) / left * s.lookahead
            } else {
                b
            }
        }
        _ => goal,
    };
    // Steer at it, as av_sim's autopilot steers at its waypoints.
    let error = centre.forward().angle_to(target - p);
    let turn = (s.turn_gain * error).clamp(-s.max_turn_rate, s.max_turn_rate);
    let slow = (p.distance(goal) / s.slow_down).clamp(0.2, 1.0);
    let forward = s.max_speed * error.cos().max(0.0).powi(2) * slow;
    let half = turn * ROBOT_WIDTH / 2.0;
    Steer {
        left: forward - half,
        right: forward + half,
        target,
        arrived: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{STEP_MS, drive};
    use crate::unit_d::ekf::ErrorEllipse;
    use crate::unit_nav::planmap::MapSetup;

    fn sure(n: usize) -> Vec<ErrorEllipse> {
        vec![
            ErrorEllipse {
                angle: 0.0,
                major: 5.0,
                minor: 5.0
            };
            n
        ]
    }

    #[test]
    fn the_route_goes_round_and_is_short() {
        let landmarks = [Vec2::new(1000.0, 1000.0), Vec2::new(1000.0, 1350.0)];
        let map = PlanMap::build(&landmarks, &sure(2), &MapSetup::default());
        let from = Vec2::new(500.0, 1100.0);
        let to = Vec2::new(1500.0, 1150.0);
        let route = plan(&map, from, to).unwrap();
        assert_eq!(route[0], from);
        assert_eq!(*route.last().unwrap(), to);
        for w in route.windows(2) {
            assert!(map.line_free(w[0], w[1]), "{w:?}");
        }
        // A handful of corners, not a cell per point, and not much
        // longer than the straight line.
        assert!(route.len() <= 5, "{route:?}");
        let straight = from.distance(to);
        let length = route_length(&route);
        assert!(length > straight && length < 1.6 * straight, "{length}");
    }

    #[test]
    fn plan_errors() {
        let map = PlanMap::build(&[Vec2::new(1000.0, 1000.0)], &sure(1), &MapSetup::default());
        let free = Vec2::new(500.0, 500.0);
        assert_eq!(
            plan(&map, free, Vec2::new(1000.0, 1050.0)),
            Err(PlanError::GoalBlocked)
        );
        // From inside the disc: out to the nearest free cell first.
        let route = plan(&map, Vec2::new(1000.0, 1050.0), free).unwrap();
        assert!(map.blocked_at(route[0]) && !map.blocked_at(route[1]));
    }

    /// A perfect robot (the simulator's own drive(), no noise) follows an
    /// L-shaped route to its end.
    #[test]
    fn pursuit_follows_an_l() {
        let route = [
            Vec2::new(300.0, 300.0),
            Vec2::new(1500.0, 300.0),
            Vec2::new(1500.0, 1500.0),
        ];
        let s = FollowSetup::default();
        let mut centre = Pose::new(300.0, 300.0, 0.0);
        let mut progress = 0;
        let dt = STEP_MS as f32 / 1000.0;
        let mut worst: f32 = 0.0;
        for step in 0..400 {
            let steer = pursue(centre, &route, &mut progress, &s);
            if steer.arrived {
                assert!(worst < 120.0, "strayed {worst} mm from the route");
                // 2400 mm at the top speed (32 mm a step) is 75 steps.
                assert!(step >= 75, "faster than the top speed: {step} steps");
                return;
            }
            centre = drive(centre, steer.left * dt, steer.right * dt, ROBOT_WIDTH);
            let p = centre.position();
            let off = (p.y - 300.0).abs().min((p.x - 1500.0).abs());
            worst = worst.max(off);
        }
        panic!("didn't arrive: at {centre:?}");
    }
}
