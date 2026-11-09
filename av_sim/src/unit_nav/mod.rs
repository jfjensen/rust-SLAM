// Unit NAV (not in the course): Units F and PP together. The robot drives
// to a goal using only what EKF SLAM tells it: its pose, and the map of
// the cylinders seen so far.
//
//   planmap.rs  the planning grid, from the SLAM map (discs grown by each
//               landmark's uncertainty)
//   route.rs    A* on it (unit_pp), string pulling, pure pursuit
//   here        the navigator: when to replan, and why; random goals
//               ("wander"); the plugin, which writes the live robot's
//               DriveCommand
//
// Like the other units' plugins, the navigator never sees the truth: not
// the true pose, not the arena's cylinders. It runs after the algorithms,
// on each new SLAM step, and its wheel speeds drive the NEXT live step.

pub mod planmap;
pub mod route;

use crate::core::{AvSet, RobotLog};
use crate::rng::Rng;
use crate::robot::{Pose, SCANNER_DISPLACEMENT};
use crate::source::{DriveCommand, LiveSim};
use crate::unit_a::motion::DeadReckoning;
use crate::unit_f::{SlamMapping, SlamStep};
use bevy::prelude::*;
use planmap::{MapSetup, PlanMap};
use route::{FollowSetup, PlanError, Steer, plan, pursue};

/// Which pose the navigator steers by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerBy {
    Slam,
    /// Unit A's dead reckoning: it drifts, and the robot doesn't know.
    DeadReckoning,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct NavSetup {
    pub map: MapSetup,
    pub follow: FollowSetup,
    pub steer_by: SteerBy,
    /// Pick a random goal whenever there is none (or it was reached).
    pub wander: bool,
}

impl Default for NavSetup {
    fn default() -> NavSetup {
        NavSetup {
            map: MapSetup::default(),
            follow: FollowSetup::default(),
            steer_by: SteerBy::Slam,
            wander: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavStatus {
    /// No goal.
    Idle,
    Driving,
    Arrived,
    /// The last plan failed; it's tried again every RETRY steps (the map
    /// changes as the robot learns).
    Failed(PlanError),
    /// It stopped moving while driving (it ran into something: the
    /// simulator stops the robot at an obstacle, and the encoders don't
    /// count): reversing a little, then planning again.
    BackingUp,
    /// It backed up MAX_BACKUPS times and is still stuck: gave the goal up.
    Stuck,
}

/// Why the route was planned again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Replan {
    NewGoal,
    /// The map has a landmark more than when the route was planned.
    NewLandmark,
    /// The rest of the route now crosses a blocked cell: a landmark moved,
    /// or its disc grew.
    Blocked,
    /// A failed plan, tried again.
    Retry,
    /// It backed up after running into something.
    BackedUp,
}

impl Replan {
    pub fn text(self) -> &'static str {
        match self {
            Replan::NewGoal => "new goal",
            Replan::NewLandmark => "new landmark in the map",
            Replan::Blocked => "the route is blocked now",
            Replan::Retry => "trying again",
            Replan::BackedUp => "backed up after a bump",
        }
    }
}

/// Steps between tries after a failed plan.
const RETRY: usize = 10;
/// Steps without moving (STILL_MM, STILL_RAD) while driving before it
/// backs up. Turning on the spot counts as moving.
const STILL_STEPS: usize = 5;
const STILL_MM: f32 = 5.0;
const STILL_RAD: f32 = 0.05;
/// Backing up: this many steps at this speed (about 100 mm), at most
/// MAX_BACKUPS times per goal.
const BACKUP_STEPS: usize = 6;
const BACKUP_SPEED: f32 = 80.0;
const MAX_BACKUPS: usize = 3;
/// Random goals are at least this far from the robot.
const MIN_WANDER: f32 = 600.0;

#[derive(Resource, Debug)]
pub struct Navigator {
    /// Set by the app (a click), or by wander.
    pub goal: Option<Vec2>,
    pub status: NavStatus,
    /// The route being followed (axle centre positions, mm).
    pub route: Vec<Vec2>,
    pub progress: usize,
    pub steer: Option<Steer>,
    /// The planning map of the last step.
    pub map: Option<PlanMap>,
    /// The pose it steers by (axle centre).
    pub centre: Option<Pose>,
    pub last_replan: Option<(Replan, usize)>,
    pub replans: usize,
    pub goals_reached: usize,
    pub goals_given_up: usize,
    /// Times it backed up after running into something (all goals).
    pub backups_total: usize,
    /// Which legs were free when the route was planned.
    legs_free: Vec<bool>,
    /// What the route was planned for.
    planned: Option<(Vec2, usize)>,
    failed_at: usize,
    /// The pose last time it moved, and the step.
    still: (Pose, usize),
    backups: usize,
    backing_until: usize,
    steps_seen: usize,
    generation: u64,
    rng: Rng,
}

impl Default for Navigator {
    fn default() -> Navigator {
        Navigator {
            goal: None,
            status: NavStatus::Idle,
            route: Vec::new(),
            progress: 0,
            steer: None,
            map: None,
            centre: None,
            last_replan: None,
            replans: 0,
            goals_reached: 0,
            goals_given_up: 0,
            backups_total: 0,
            legs_free: Vec::new(),
            planned: None,
            failed_at: 0,
            still: (Pose::default(), 0),
            backups: 0,
            backing_until: 0,
            steps_seen: 0,
            generation: u64::MAX,
            rng: Rng::new(1),
        }
    }
}

impl Navigator {
    /// A goal from the app (a click). Replanned on the next step.
    pub fn set_goal(&mut self, goal: Vec2) {
        self.goal = Some(goal);
        self.status = NavStatus::Driving;
        self.planned = None;
        self.backups = 0;
    }

    /// Has a leg still ahead that was free when planned become blocked?
    /// (Legs that weren't free then don't count: the way out of a disc,
    /// or a diagonal step through two cells' corner. Flagging those would
    /// replan on every step.)
    fn route_blocked(&self, map: &PlanMap) -> bool {
        let r = &self.route;
        (self.progress..r.len().saturating_sub(1))
            .any(|k| self.legs_free[k] && !map.line_free(r[k], r[k + 1]))
    }

    /// A random point the robot can reach, MIN_WANDER from it.
    fn random_goal(&mut self, map: &PlanMap, at: Vec2, room: Vec2) -> Option<Vec2> {
        let start = map.nearest_free(map.to_cell(at), 15)?;
        let reachable = map.reachable(start);
        (0..500).find_map(|_| {
            let p = Vec2::new(self.rng.unit() as f32, self.rng.unit() as f32) * room;
            let c = map.to_cell(p);
            let ok = !map.blocked(c)
                && reachable[map.grid.index(c.0, c.1)]
                && p.distance(at) > MIN_WANDER;
            ok.then_some(p)
        })
    }

    /// One navigation step: the SLAM step just computed (`step` is its
    /// index), the pose to steer by (scanner pose). Returns the wheel
    /// speeds for the next live step.
    pub fn update(
        &mut self,
        slam: &SlamStep,
        step: usize,
        scanner: Pose,
        setup: &NavSetup,
    ) -> DriveCommand {
        let centre = scanner.shifted(-SCANNER_DISPLACEMENT);
        let at = centre.position();
        self.centre = Some(centre);
        let map = PlanMap::build(&slam.landmarks, &slam.landmark_ellipses, &setup.map);
        let landmarks = slam.landmarks.len();

        // A goal to go to? (Wander picks one when there's nothing to do.)
        // Wandering, a failed plan isn't tried again: pick another goal.
        let finished = |s: NavStatus| {
            matches!(s, NavStatus::Idle | NavStatus::Arrived | NavStatus::Stuck)
                || (setup.wander && matches!(s, NavStatus::Failed(_)))
        };
        if setup.wander
            && (self.goal.is_none() || finished(self.status))
            && let Some(g) = self.random_goal(&map, at, setup.map.room)
        {
            self.set_goal(g);
        }
        let Some(goal) = self.goal.filter(|_| !finished(self.status)) else {
            self.map = Some(map);
            self.steer = None;
            return DriveCommand::default();
        };

        // Backing up after a bump? Straight back, then plan again.
        let reverse = DriveCommand {
            left: -BACKUP_SPEED,
            right: -BACKUP_SPEED,
        };
        let mut backed_up = false;
        if self.status == NavStatus::BackingUp {
            if step < self.backing_until {
                self.map = Some(map);
                self.steer = None;
                return reverse;
            }
            self.status = NavStatus::Driving;
            backed_up = true;
        }

        // Plan again? And why.
        let reason = match (self.planned, self.status) {
            _ if backed_up => Some(Replan::BackedUp),
            (None, _) => Some(Replan::NewGoal),
            (Some((g, _)), _) if g != goal => Some(Replan::NewGoal),
            (_, NavStatus::Failed(_)) => (step >= self.failed_at + RETRY).then_some(Replan::Retry),
            (Some((_, n)), _) if landmarks > n => Some(Replan::NewLandmark),
            _ if self.route_blocked(&map) => Some(Replan::Blocked),
            _ => None,
        };
        if let Some(reason) = reason {
            self.replans += 1;
            self.last_replan = Some((reason, step));
            self.planned = Some((goal, landmarks));
            self.progress = 0;
            // Only a new goal restarts the "not moving" clock: replanning
            // while pinned against a wall mustn't stop it backing up.
            if reason == Replan::NewGoal {
                self.still = (centre, step);
            }
            match plan(&map, at, goal) {
                Ok(route) => {
                    self.legs_free = route
                        .windows(2)
                        .map(|w| map.line_free(w[0], w[1]))
                        .collect();
                    self.route = route;
                    self.status = NavStatus::Driving;
                }
                Err(e) => {
                    self.route.clear();
                    self.status = NavStatus::Failed(e);
                    self.failed_at = step;
                }
            }
        }
        self.map = Some(map);
        if self.status != NavStatus::Driving {
            self.steer = None;
            return DriveCommand::default();
        }

        // Not moving? (Ran into something.) Back up, or give up.
        let (last, since) = self.still;
        let turned = crate::unit_d::ekf::wrap_pi((centre.heading - last.heading) as f64).abs();
        if at.distance(last.position()) > STILL_MM || turned > STILL_RAD as f64 {
            self.still = (centre, step);
        } else if step > since + STILL_STEPS {
            self.steer = None;
            self.still = (centre, step);
            if self.backups < MAX_BACKUPS {
                self.backups += 1;
                self.backups_total += 1;
                self.status = NavStatus::BackingUp;
                self.backing_until = step + BACKUP_STEPS;
                return reverse;
            }
            self.status = NavStatus::Stuck;
            self.goals_given_up += 1;
            return DriveCommand::default();
        }

        let steer = pursue(centre, &self.route, &mut self.progress, &setup.follow);
        self.steer = Some(steer);
        if steer.arrived {
            self.status = NavStatus::Arrived;
            self.goals_reached += 1;
            self.backups = 0;
        }
        DriveCommand {
            left: steer.left,
            right: steer.right,
        }
    }
}

// --- The plugin ----------------------------------------------------------------

#[derive(Default)]
pub struct NavPlugin;

impl Plugin for NavPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NavSetup>()
            .init_resource::<Navigator>()
            .add_systems(Startup, take_the_wheel)
            .add_systems(
                Update,
                navigate.after(AvSet::Algorithms).before(AvSet::Present),
            );
    }
}

/// The live robot's autopilot off: the navigator drives.
fn take_the_wheel(mut live: ResMut<LiveSim>) {
    live.autopilot = false;
}

fn navigate(
    log: Res<RobotLog>,
    slam: Res<SlamMapping>,
    dr: Res<DeadReckoning>,
    setup: Res<NavSetup>,
    mut nav: ResMut<Navigator>,
    mut drive: ResMut<DriveCommand>,
) {
    if nav.generation != log.generation {
        // A new run: start over (the same random goals for the same seed).
        let wander_seed = log.generation;
        *nav = Navigator {
            generation: log.generation,
            rng: Rng::new(wander_seed),
            ..Navigator::default()
        };
    }
    let n = slam.steps.len();
    if n == nav.steps_seen || n == 0 {
        return;
    }
    nav.steps_seen = n;
    let step = &slam.steps[n - 1];
    let scanner = match setup.steer_by {
        SteerBy::Slam => step.pose,
        SteerBy::DeadReckoning => dr.poses.get(n - 1).copied().unwrap_or(step.pose),
    };
    let command = nav.update(step, n - 1, scanner, &setup);
    drive.set_if_neq(command);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AvCorePlugin;
    use crate::source::{DataSourcePlugin, Mode};
    use crate::synth::STEP_MS;
    use crate::unit_a::UnitAPlugins;
    use crate::unit_f::{SlamSetup, UnitFPlugins};
    use bevy::state::app::StatesPlugin;
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    fn nav_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin, AvCorePlugin, DataSourcePlugin))
            .add_plugins((UnitAPlugins, UnitFPlugins, NavPlugin))
            .insert_resource(SlamSetup::sharp_scanner())
            .insert_state(Mode::Live)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                STEP_MS,
            )));
        app.finish();
        app.cleanup();
        app
    }

    /// The whole loop: wander for 1500 steps (5 minutes of robot time).
    /// Goals get reached, the true robot never bumps, and SLAM's pose
    /// stays near the truth.
    #[test]
    fn it_wanders_without_bumping() {
        let mut app = nav_app();
        let mut bumps = 0;
        let mut worst: f32 = 0.0;
        for _ in 0..1500 {
            app.update();
            let w = app.world();
            bumps += w.resource::<LiveSim>().sim.bumped as usize;
            let truth = w.resource::<LiveSim>().sim.scanner_pose();
            let Some(step) = w.resource::<SlamMapping>().steps.last() else {
                continue;
            };
            worst = worst.max(step.pose.position().distance(truth.position()));
        }
        let w = app.world();
        let nav = w.resource::<Navigator>();
        println!(
            "reached {} given up {} replans {} bumps {bumps} worst {worst:.0}",
            nav.goals_reached, nav.goals_given_up, nav.replans
        );
        assert!(!w.resource::<LiveSim>().autopilot);
        assert!(nav.goals_reached >= 5, "{}", nav.goals_reached);
        assert_eq!(bumps, 0);
        assert_eq!(nav.goals_given_up, 0);
        assert!(worst < 60.0, "{worst}");
    }

    /// Goals reached, replans, bumps and SLAM's worst error over 3000
    /// steps, seeds 1-5, steering by SLAM (both noise settings) and by
    /// dead reckoning. Run with
    ///   cargo test --release unit_nav::tests::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        for (name, slam_setup, steer_by) in [
            ("SLAM, sharp ", SlamSetup::sharp_scanner(), SteerBy::Slam),
            ("SLAM, course", SlamSetup::default(), SteerBy::Slam),
            (
                "dead reckon.",
                SlamSetup::sharp_scanner(),
                SteerBy::DeadReckoning,
            ),
        ] {
            for seed in 1..=5u64 {
                let mut app = nav_app();
                app.insert_resource(slam_setup);
                app.world_mut().resource_mut::<NavSetup>().steer_by = steer_by;
                // Live runs take the next seed: start from seed - 1.
                app.world_mut().resource_mut::<LiveSim>().seed = seed - 1;
                let mut bumps = 0;
                let mut worst: f32 = 0.0;
                let mut failed_steps = 0;
                for _ in 0..3000 {
                    app.update();
                    let w = app.world();
                    let live = w.resource::<LiveSim>();
                    bumps += live.sim.bumped as usize;
                    failed_steps +=
                        matches!(w.resource::<Navigator>().status, NavStatus::Failed(_)) as usize;
                    if let Some(step) = w.resource::<SlamMapping>().steps.last() {
                        worst = worst.max(
                            step.pose
                                .position()
                                .distance(live.sim.scanner_pose().position()),
                        );
                    }
                }
                let w = app.world();
                let nav = w.resource::<Navigator>();
                println!(
                    "{name} seed {seed}: {} steps, reached {:2}, given up {}, replans {:3}, frames failed {failed_steps:4}, bumps {bumps:3}, SLAM worst {worst:5.0} mm",
                    w.resource::<SlamMapping>().steps.len(),
                    nav.goals_reached,
                    nav.goals_given_up,
                    nav.replans
                );
            }
        }
    }
}
