// Unit C's data source: the same two modes as ../source.rs (and the same
// Mode state), for the 1D track.
//
//   Replay  read data/track.txt (generated on the first run) and step
//           through it.
//   Live    the track robot drives now, one step every 0.5 s, on the
//           autopilot or by the app's TrackCommand; the sensor can be
//           switched off.

use super::Track;
use super::track::{DEFAULT_STEPS, TrackLog, TrackModel, TrackSim, generate};
use crate::core::{AvSet, Playhead, Status};
use crate::source::Mode;
use bevy::prelude::*;
use std::time::Duration;

pub const DEFAULT_TRACK: &str = "data/track.txt";
pub const LIVE_TRACK: &str = "data/live_track.txt";
/// A filter step every this many ms, in both modes. Slower than Units A
/// and B: there's more to look at in each step.
pub const TRACK_STEP_MS: u64 = 500;

#[derive(Resource, Debug, Clone)]
pub struct TrackFile {
    pub path: String,
    pub seed: u64,
}

impl Default for TrackFile {
    fn default() -> TrackFile {
        TrackFile {
            path: DEFAULT_TRACK.into(),
            seed: 1,
        }
    }
}

/// How far to move each live step when the autopilot is off (cm). The
/// app sets it from the keys.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct TrackCommand(pub f64);

#[derive(Resource, Debug)]
pub struct LiveTrack {
    pub sim: TrackSim,
    pub autopilot: bool,
    pub paused: bool,
    /// The sensor: off makes the steps prediction only.
    pub measuring: bool,
    pub seed: u64,
    timer: Timer,
}

impl Default for LiveTrack {
    fn default() -> LiveTrack {
        LiveTrack {
            sim: TrackSim::new(TrackModel::default(), 0),
            autopilot: true,
            paused: false,
            measuring: true,
            seed: 0,
            timer: step_timer(),
        }
    }
}

fn step_timer() -> Timer {
    Timer::new(Duration::from_millis(TRACK_STEP_MS), TimerMode::Repeating)
}

pub struct TrackSourcePlugin;

impl Plugin for TrackSourcePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<Mode>()
            .init_resource::<Track>()
            .init_resource::<TrackFile>()
            .init_resource::<TrackCommand>()
            .init_resource::<LiveTrack>()
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

/// Read the file, generating it first if it isn't there.
pub fn load_or_generate(file: &TrackFile) -> Result<(TrackLog, bool), String> {
    if std::path::Path::new(&file.path).exists() {
        return TrackLog::read(&file.path).map(|log| (log, false));
    }
    let log = generate(file.seed, DEFAULT_STEPS);
    log.write(&file.path)?;
    Ok((log, true))
}

fn enter_replay(
    file: Res<TrackFile>,
    mut track: ResMut<Track>,
    mut playhead: ResMut<Playhead>,
    mut status: ResMut<Status>,
) {
    match load_or_generate(&file) {
        Ok((log, generated)) => {
            status.0 = format!(
                "{} {} ({} steps)",
                if generated { "generated" } else { "loaded" },
                file.path,
                log.len()
            );
            track.replace(log);
        }
        Err(e) => {
            status.0 = format!("could not read the track: {e}");
            track.replace(TrackLog::default());
        }
    }
    *playhead = Playhead {
        index: 0,
        playing: true,
        timer: step_timer(),
    };
}

fn enter_live(
    mut live: ResMut<LiveTrack>,
    mut track: ResMut<Track>,
    mut playhead: ResMut<Playhead>,
    mut status: ResMut<Status>,
) {
    let seed = live.seed + 1;
    let autopilot = live.autopilot;
    *live = LiveTrack {
        sim: TrackSim::new(TrackModel::default(), seed),
        seed,
        autopilot,
        ..LiveTrack::default()
    };
    track.replace(TrackLog {
        model: TrackModel::default(),
        steps: Vec::new(),
    });
    playhead.index = 0;
    status.0 = format!("live run, seed {seed}");
}

fn step_replay(time: Res<Time>, track: Res<Track>, mut playhead: ResMut<Playhead>) {
    if !playhead.playing {
        return;
    }
    playhead.timer.tick(time.delta());
    let steps = playhead.timer.times_finished_this_tick() as usize;
    let last = track.len().saturating_sub(1);
    playhead.index = (playhead.index + steps).min(last);
    if playhead.index == last {
        playhead.playing = false;
    }
}

/// At most this many steps in one frame (after a long freeze).
const MAX_STEPS_PER_FRAME: u32 = 3;

fn step_live(
    time: Res<Time>,
    command: Res<TrackCommand>,
    mut live: ResMut<LiveTrack>,
    mut track: ResMut<Track>,
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
        let control = if live.autopilot {
            live.sim.autopilot()
        } else {
            command.0
        };
        let step = live.sim.step(control, live.measuring);
        track.log.steps.push(step);
    }
    playhead.index = track.len().saturating_sub(1);
}

/// Save the live run.
pub fn save_track(log: &TrackLog, path: &str) -> Result<String, String> {
    log.write(path)
        .map(|()| format!("saved {} steps to {path}", log.len()))
        .map_err(|e| format!("saving to {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AvCorePlugin;
    use crate::unit_c::{HistogramFilter, HistogramParams, KalmanFilter, Shape, UnitCPlugins};
    use bevy::state::app::StatesPlugin;
    use bevy::time::TimeUpdateStrategy;

    fn live_app() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            StatesPlugin,
            AvCorePlugin,
            TrackSourcePlugin,
        ))
        .add_plugins(UnitCPlugins)
        .insert_state(Mode::Live)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            TRACK_STEP_MS,
        )));
        app.finish();
        app.cleanup();
        app
    }

    /// Live steps reach both filters in the same frame; changing the shape
    /// starts the histogram filter over, not the Kalman filter.
    #[test]
    fn live_steps_reach_both_filters() {
        let mut app = live_app();
        // Virtual time caps a frame at 250 ms (Time<Virtual>'s max_delta),
        // so a filter step takes two of these frames.
        for _ in 0..14 {
            app.update();
        }
        let n = app.world().resource::<Track>().len();
        assert!(n >= 6, "{n} steps");
        assert_eq!(app.world().resource::<HistogramFilter>().steps.len(), n);
        assert_eq!(app.world().resource::<KalmanFilter>().steps.len(), n);
        assert_eq!(app.world().resource::<Playhead>().index, n - 1);

        app.world_mut().resource_mut::<LiveTrack>().paused = true;
        let before = app.world().resource::<HistogramFilter>().steps[n - 1]
            .correction
            .clone();
        app.world_mut().resource_mut::<HistogramParams>().shape = Shape::Triangle;
        app.update();
        let after = &app.world().resource::<HistogramFilter>().steps[n - 1].correction;
        assert_ne!(after, &before);
    }

    /// Replay generates the file the first time, and reads it after.
    #[test]
    fn replay_generates_then_reads() {
        let dir = std::env::temp_dir().join(format!("av_sim_track_{}", std::process::id()));
        let path = dir.join("track.txt").to_string_lossy().into_owned();
        let file = TrackFile {
            path: path.clone(),
            seed: 4,
        };
        let (first, generated) = load_or_generate(&file).unwrap();
        assert!(generated);
        let (second, generated) = load_or_generate(&file).unwrap();
        assert!(!generated);
        assert_eq!(second.len(), first.len());
        assert_eq!(second.model, first.model);
        let _ = std::fs::remove_dir_all(dir);
    }
}
