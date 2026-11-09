// Part 6: where the RobotLog comes from. Two modes (a Bevy state):
//
//   Replay  read the course-format files (made by `generate`, or saved
//           from a live run) and step through them, like the course's
//           logfile_viewer.py.
//   Live    the synthetic robot drives now, one record every 0.2 s,
//           steered by the autopilot or by the app's keys (DriveCommand).
//
// The algorithms can't tell the difference: in both modes they only see
// the RobotLog grow or change.

use crate::core::{AvSet, Playhead, RobotLog, Status};
use crate::logfile::{LogFile, LogPaths};
use crate::synth::{self, Autopilot, STEP_MS, Simulator, SynthConfig};
use crate::world::Arena;
use bevy::prelude::*;
use std::time::Duration;

#[derive(States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    #[default]
    Replay,
    Live,
}

/// Which files Replay reads. If they don't exist yet, they are generated
/// (with this seed) and written first.
#[derive(Resource, Debug, Clone)]
pub struct ReplaySource {
    pub prefix: String,
    pub seed: u64,
}

pub const DEFAULT_PREFIX: &str = "data/generated";
pub const LIVE_PREFIX: &str = "data/live";

impl Default for ReplaySource {
    fn default() -> ReplaySource {
        ReplaySource {
            prefix: DEFAULT_PREFIX.into(),
            seed: 1,
        }
    }
}

/// Wheel speeds (left, right) in mm/s for the live robot, when the
/// autopilot is off. The app writes it from the keys.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct DriveCommand {
    pub left: f32,
    pub right: f32,
}

#[derive(Resource, Debug)]
pub struct LiveSim {
    pub sim: Simulator,
    pub pilot: Autopilot,
    pub autopilot: bool,
    pub paused: bool,
    /// Each live run gets the next seed.
    pub seed: u64,
    timer: Timer,
}

impl Default for LiveSim {
    fn default() -> LiveSim {
        LiveSim {
            sim: Simulator::new(SynthConfig::REALISTIC, Arena::course(), 0),
            pilot: Autopilot::default(),
            autopilot: true,
            paused: false,
            seed: 0,
            timer: step_timer(),
        }
    }
}

fn step_timer() -> Timer {
    Timer::new(Duration::from_millis(STEP_MS), TimerMode::Repeating)
}

pub struct DataSourcePlugin;

impl Plugin for DataSourcePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<Mode>()
            .init_resource::<ReplaySource>()
            .init_resource::<DriveCommand>()
            .init_resource::<LiveSim>()
            .add_systems(OnEnter(Mode::Replay), enter_replay)
            .add_systems(OnEnter(Mode::Live), enter_live)
            .add_systems(
                Update,
                (
                    step_replay.run_if(in_state(Mode::Replay)),
                    step_live.run_if(in_state(Mode::Live)),
                )
                    .in_set(AvSet::Simulate),
            );
    }
}

/// Read the files, generating them first if needed.
pub fn load_or_generate(source: &ReplaySource) -> Result<(LogFile, bool), String> {
    let paths = LogPaths::with_prefix(&source.prefix);
    if std::path::Path::new(&paths.motors).exists() {
        return paths.read().map(|log| (log, false));
    }
    let log = synth::generate(SynthConfig::REALISTIC, source.seed, synth::DEFAULT_STEPS);
    paths
        .write(&log)
        .map_err(|e| format!("writing {}: {e}", paths.motors))?;
    Ok((log, true))
}

fn enter_replay(
    source: Res<ReplaySource>,
    mut log: ResMut<RobotLog>,
    mut playhead: ResMut<Playhead>,
    mut status: ResMut<Status>,
) {
    match load_or_generate(&source) {
        Ok((new, generated)) => {
            status.0 = format!(
                "{} {}_*.txt ({} steps)",
                if generated { "generated" } else { "loaded" },
                source.prefix,
                new.len()
            );
            log.replace(new);
        }
        Err(e) => {
            status.0 = format!("could not read the log: {e}");
            log.replace(LogFile::default());
        }
    }
    *playhead = Playhead {
        index: 0,
        playing: true,
        timer: step_timer(),
    };
}

fn enter_live(
    mut live: ResMut<LiveSim>,
    mut log: ResMut<RobotLog>,
    mut playhead: ResMut<Playhead>,
    mut status: ResMut<Status>,
) {
    let seed = live.restart(&mut log);
    playhead.index = 0;
    status.0 = format!("live run, seed {seed}");
}

impl LiveSim {
    /// A new live run with the next seed, starting a new log (the
    /// algorithms start over). Keeps the autopilot setting. Returns the
    /// seed. Entering Live mode does this; av09 also calls it for a new
    /// run without leaving Live.
    pub fn restart(&mut self, log: &mut RobotLog) -> u64 {
        let seed = self.seed + 1;
        *self = LiveSim {
            sim: Simulator::new(SynthConfig::REALISTIC, Arena::course(), seed),
            seed,
            autopilot: self.autopilot,
            ..LiveSim::default()
        };
        let mut new = LogFile {
            landmarks: self.sim.arena.landmarks.clone(),
            ..LogFile::default()
        };
        synth::push(&mut new, self.sim.records());
        log.replace(new);
        seed
    }
}

fn step_replay(time: Res<Time>, log: Res<RobotLog>, mut playhead: ResMut<Playhead>) {
    if !playhead.playing {
        return;
    }
    playhead.timer.tick(time.delta());
    let steps = playhead.timer.times_finished_this_tick() as usize;
    let last = log.len().saturating_sub(1);
    playhead.index = (playhead.index + steps).min(last);
    if playhead.index == last {
        playhead.playing = false;
    }
}

/// At most this many steps in one frame (after a long freeze).
const MAX_STEPS_PER_FRAME: u32 = 5;

fn step_live(
    time: Res<Time>,
    drive: Res<DriveCommand>,
    mut live: ResMut<LiveSim>,
    mut log: ResMut<RobotLog>,
    mut playhead: ResMut<Playhead>,
) {
    if live.paused {
        return;
    }
    live.timer.tick(time.delta());
    let steps = live
        .timer
        .times_finished_this_tick()
        .min(MAX_STEPS_PER_FRAME);
    for _ in 0..steps {
        let live = &mut *live;
        let (l, r) = if live.autopilot {
            live.pilot.command(live.sim.centre())
        } else {
            (drive.left, drive.right)
        };
        live.sim.advance(l, r);
        synth::push(&mut log.log, live.sim.records());
    }
    playhead.index = log.len().saturating_sub(1);
}

/// Save the current log in the course format.
pub fn save_log(log: &LogFile, prefix: &str) -> Result<String, String> {
    let paths = LogPaths::with_prefix(prefix);
    paths
        .write(log)
        .map(|()| format!("saved {} steps to {prefix}_*.txt", log.len()))
        .map_err(|e| format!("saving to {prefix}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AvCorePlugin;
    use crate::unit_a::UnitAPlugins;
    use crate::unit_a::cylinders::DetectedCylinders;
    use crate::unit_a::motion::DeadReckoning;
    use bevy::state::app::StatesPlugin;
    use bevy::time::TimeUpdateStrategy;

    /// A live app without a window, one 0.2 s step per frame.
    fn live_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin, AvCorePlugin, DataSourcePlugin))
            .add_plugins(UnitAPlugins)
            .insert_state(Mode::Live)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                STEP_MS,
            )));
        app.finish();
        app.cleanup();
        app
    }

    /// Live mode adds one record per 0.2 s, and the algorithms keep up in
    /// the same frame (they run after Simulate).
    #[test]
    fn live_steps_reach_the_algorithms_in_the_same_frame() {
        let mut app = live_app();
        // The first update starts the clock (no time has passed yet).
        app.update();
        for _ in 0..10 {
            app.update();
        }
        let w = app.world();
        let n = w.resource::<RobotLog>().len();
        assert!(n >= 10, "{n} steps");
        assert_eq!(w.resource::<DeadReckoning>().poses.len(), n);
        assert_eq!(w.resource::<DetectedCylinders>().per_scan.len(), n);
        assert_eq!(w.resource::<Playhead>().index, n - 1);
    }

    /// Switching mode loads a different log: the algorithms start over
    /// instead of appending to the old results.
    #[test]
    fn a_new_log_restarts_the_algorithms() {
        let mut app = live_app();
        for _ in 0..6 {
            app.update();
        }
        let dir = std::env::temp_dir().join(format!("av_sim_test_{}", std::process::id()));
        let prefix = dir.join("run").to_string_lossy().into_owned();
        app.insert_resource(ReplaySource {
            prefix: prefix.clone(),
            seed: 9,
        });
        app.world_mut()
            .resource_mut::<NextState<Mode>>()
            .set(Mode::Replay);
        app.update();
        let w = app.world();
        assert_eq!(w.resource::<RobotLog>().len(), synth::DEFAULT_STEPS);
        assert_eq!(
            w.resource::<DeadReckoning>().poses.len(),
            synth::DEFAULT_STEPS
        );
        assert!(w.resource::<Status>().0.starts_with("generated"));
        // ... and the generated files are there in the course format.
        let back = LogPaths::with_prefix(&prefix).read().unwrap();
        assert_eq!(back.motors, w.resource::<RobotLog>().log.motors);
        let _ = std::fs::remove_dir_all(dir);
    }
}
