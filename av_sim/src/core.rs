// Part 5: what every app and algorithm plugin shares.
//
//   RobotLog   the log: replayed from files, or growing in live mode
//   Playhead   which step the viewer shows
//   Status     one line for the user ("loaded ...", "saved ...")
//   AvSet      the frame order: Input -> Simulate -> Algorithms -> Present
//
// The algorithm plugins only READ the RobotLog, like the course's scripts
// only read the log files. They never see the true pose.

use crate::logfile::LogFile;
use crate::world::Arena;
use bevy::prelude::*;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AvSet {
    /// Keys and mouse (the app's).
    Input,
    /// Load a log, or drive the live robot one step.
    Simulate,
    /// The algorithms (unit_a, later units).
    Algorithms,
    /// Drawing and text (the app's).
    Present,
}

#[derive(Resource, Debug)]
pub struct RobotLog {
    pub log: LogFile,
    /// The arena the landmarks describe (for drawing, and for the live
    /// simulator; algorithms in later units get the landmarks from here).
    pub arena: Arena,
    /// Goes up every time a different log is loaded, so the algorithm
    /// plugins know to start over. Appending a live step doesn't change it.
    pub generation: u64,
}

impl Default for RobotLog {
    fn default() -> RobotLog {
        RobotLog {
            log: LogFile::default(),
            arena: Arena::course(),
            generation: 0,
        }
    }
}

impl RobotLog {
    /// Swap in a whole new log.
    pub fn replace(&mut self, log: LogFile) {
        self.arena = if log.landmarks.is_empty() {
            Arena::course()
        } else {
            Arena::with_landmarks(log.landmarks.clone())
        };
        self.log = log;
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.log.len()
    }

    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }
}

#[derive(Resource, Debug, Default)]
pub struct Playhead {
    pub index: usize,
    /// Replay: step forward on its own.
    pub playing: bool,
    pub timer: Timer,
}

#[derive(Resource, Debug, Default)]
pub struct Status(pub String);

pub struct AvCorePlugin;

impl Plugin for AvCorePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RobotLog>()
            .init_resource::<Playhead>()
            .init_resource::<Status>()
            .configure_sets(
                Update,
                (
                    AvSet::Input,
                    AvSet::Simulate,
                    AvSet::Algorithms,
                    AvSet::Present,
                )
                    .chain(),
            );
    }
}
