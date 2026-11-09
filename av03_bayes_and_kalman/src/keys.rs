// Part 4: the keys. Letters by what they type (typed.rs, any layout);
// Tab, Space, arrows, Home and End by position.
//
//   both modes   Tab  Replay <-> Live
//                V    curves: both filters -> histogram -> Kalman
//                G    histogram filter: gaussians <-> triangles
//                B    the belief before the step (grey) on/off
//   Replay       Space play/pause   Left/Right one step (Shift: 10)
//                Home/End           L  switch between track.txt and live_track.txt
//   Live         Space pause        A  autopilot on/off
//                Left/Right held: drive (autopilot off)
//                M  sensor on/off   R  save the run as data/live_track.txt

use crate::Layers;
use crate::typed::Typed;
use av_sim::core::{AvSet, Playhead, Status};
use av_sim::source::Mode;
use av_sim::unit_c::source::{
    DEFAULT_TRACK, LIVE_TRACK, LiveTrack, TrackCommand, TrackFile, save_track,
};
use av_sim::unit_c::track::SPEED;
use av_sim::unit_c::{HistogramParams, Shape, Track};
use bevy::prelude::*;
use std::path::Path;

pub struct KeysPlugin;

impl Plugin for KeysPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                switch_mode,
                view_keys,
                replay_keys.run_if(in_state(Mode::Replay)),
                live_keys.run_if(in_state(Mode::Live)),
            )
                .in_set(AvSet::Input),
        );
    }
}

/// Steps for Shift+arrow in replay.
const BIG_STEP: usize = 10;

fn switch_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mode: Res<State<Mode>>,
    mut next: ResMut<NextState<Mode>>,
) {
    if keys.just_pressed(KeyCode::Tab) {
        next.set(match mode.get() {
            Mode::Replay => Mode::Live,
            Mode::Live => Mode::Replay,
        });
    }
}

fn view_keys(typed: Res<Typed>, mut layers: ResMut<Layers>, mut params: ResMut<HistogramParams>) {
    if typed.pressed('v') {
        layers.view = layers.view.next();
    }
    if typed.pressed('b') {
        layers.before = !layers.before;
    }
    if typed.pressed('g') {
        // The histogram filter sees the change and starts over.
        params.shape = match params.shape {
            Shape::Gaussian => Shape::Triangle,
            Shape::Triangle => Shape::Gaussian,
        };
    }
}

fn replay_keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    track: Res<Track>,
    mut playhead: ResMut<Playhead>,
    mut file: ResMut<TrackFile>,
    mut next: ResMut<NextState<Mode>>,
    mut status: ResMut<Status>,
) {
    let last = track.len().saturating_sub(1);
    let step = if crate::shift_held(&keys) {
        BIG_STEP
    } else {
        1
    };
    if keys.just_pressed(KeyCode::Space) {
        if playhead.index >= last {
            playhead.index = 0;
        }
        playhead.playing = !playhead.playing;
    }
    let mut moved = None;
    if keys.just_pressed(KeyCode::ArrowRight) {
        moved = Some((playhead.index + step).min(last));
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        moved = Some(playhead.index.saturating_sub(step));
    }
    if keys.just_pressed(KeyCode::Home) {
        moved = Some(0);
    }
    if keys.just_pressed(KeyCode::End) {
        moved = Some(last);
    }
    if let Some(index) = moved {
        playhead.index = index;
        playhead.playing = false;
    }
    if typed.pressed('l') {
        let other = if file.path == DEFAULT_TRACK {
            LIVE_TRACK
        } else {
            DEFAULT_TRACK
        };
        if other == LIVE_TRACK && !Path::new(LIVE_TRACK).exists() {
            status.0 = "no live run saved yet: Tab to Live, drive, R to save".into();
        } else {
            file.path = other.into();
            // set() re-runs OnEnter(Replay): that loads the other file.
            next.set(Mode::Replay);
        }
    }
}

fn live_keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    track: Res<Track>,
    mut live: ResMut<LiveTrack>,
    mut command: ResMut<TrackCommand>,
    mut status: ResMut<Status>,
) {
    if keys.just_pressed(KeyCode::Space) {
        live.paused = !live.paused;
    }
    if typed.pressed('a') {
        live.autopilot = !live.autopilot;
    }
    if typed.pressed('m') {
        live.measuring = !live.measuring;
    }
    if typed.pressed('r') {
        status.0 = save_track(&track.log, LIVE_TRACK).unwrap_or_else(|e| e);
    }
    let new = TrackCommand(drive_from_keys(
        keys.pressed(KeyCode::ArrowLeft),
        keys.pressed(KeyCode::ArrowRight),
    ));
    command.set_if_neq(new);
}

/// Left/right arrow to the next step's control (cm). Both or neither: stand.
pub fn drive_from_keys(left: bool, right: bool) -> f64 {
    match (left, right) {
        (true, false) => -SPEED,
        (false, true) => SPEED,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_to_controls() {
        assert_eq!(drive_from_keys(false, true), SPEED);
        assert_eq!(drive_from_keys(true, false), -SPEED);
        assert_eq!(drive_from_keys(true, true), 0.0);
        assert_eq!(drive_from_keys(false, false), 0.0);
    }
}
