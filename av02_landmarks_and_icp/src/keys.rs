// Part 4: the keys. Letters by what they type (typed.rs, any layout);
// Tab, Space, arrows, Home and End by position.
//
//   both modes   Tab  Replay <-> Live
//                V    detail: landmarks -> walls -> off
//                D    dead-reckoning path on/off
//                E    export apply_transform.txt and icp_wall_transform.txt
//   Replay       Space play/pause   Left/Right one step (Shift: 10)
//                Home/End           L  switch between data/generated and data/live
//   Live         Space pause        A  autopilot on/off
//                arrows drive (autopilot off)
//                R  save the run as data/live_*.txt (course format)

use crate::Layers;
use crate::typed::Typed;
use av_sim::core::{AvSet, Playhead, RobotLog, Status};
use av_sim::logfile::{points_line, pose_line};
use av_sim::source::{
    DEFAULT_PREFIX, DriveCommand, LIVE_PREFIX, LiveSim, Mode, ReplaySource, save_log,
};
use av_sim::synth::MAX_SPEED;
use av_sim::unit_a::cylinders::DetectedCylinders;
use av_sim::unit_b::landmarks::LandmarkCorrection;
use av_sim::unit_b::walls::WallCorrection;
use bevy::prelude::*;
use std::path::Path;

pub struct KeysPlugin;

impl Plugin for KeysPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                switch_mode,
                layer_keys,
                export_key,
                replay_keys.run_if(in_state(Mode::Replay)),
                live_keys.run_if(in_state(Mode::Live)),
            )
                .in_set(AvSet::Input),
        );
    }
}

/// Steps for Shift+arrow in replay.
const BIG_STEP: usize = 10;
/// Turning speed of each wheel when steering with the arrows, mm/s.
const STEER: f32 = 0.5 * MAX_SPEED;

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

fn layer_keys(typed: Res<Typed>, mut layers: ResMut<Layers>) {
    if typed.pressed('v') {
        layers.detail = layers.detail.next();
    }
    if typed.pressed('d') {
        layers.dead_reckoning = !layers.dead_reckoning;
    }
}

/// E: write what the corrections computed, in the course's formats, next
/// to the log files. Per step, as slam_04_d and slam_05_c write them:
///   apply_transform.txt     F (corrected pose), D C (detections, scanner
///                           frame), W C (detections moved by the transform)
///   icp_wall_transform.txt  F (corrected pose), W C (scan points moved
///                           by the ICP transform)
fn export_key(
    typed: Res<Typed>,
    source: Res<ReplaySource>,
    found: Res<DetectedCylinders>,
    by_landmarks: Res<LandmarkCorrection>,
    by_walls: Res<WallCorrection>,
    mut status: ResMut<Status>,
) {
    if !typed.pressed('e') {
        return;
    }
    let dir = Path::new(&source.prefix)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let mut landmarks = String::new();
    for (i, step) in by_landmarks.steps.iter().enumerate() {
        let local: Vec<Vec2> = found
            .per_scan
            .get(i)
            .map(|scan| scan.iter().map(|c| c.local).collect())
            .unwrap_or_default();
        let moved: Vec<Vec2> = match step.transform {
            Some(t) => step
                .pairs
                .iter()
                .map(|&(c, _)| t.apply(step.cylinders[c]))
                .collect(),
            None => Vec::new(),
        };
        landmarks += &pose_line(&step.pose);
        landmarks += &points_line("D C", &local);
        landmarks += &points_line("W C", &moved);
    }
    let mut walls = String::new();
    for step in &by_walls.steps {
        let moved: Vec<Vec2> = step
            .points
            .iter()
            .map(|&p| step.transform.apply(p))
            .collect();
        walls += &pose_line(&step.pose);
        walls += &points_line("W C", &moved);
    }
    let landmarks_path = dir.join("apply_transform.txt");
    let walls_path = dir.join("icp_wall_transform.txt");
    let result = std::fs::create_dir_all(&dir)
        .and_then(|()| std::fs::write(&landmarks_path, landmarks))
        .and_then(|()| std::fs::write(&walls_path, walls));
    status.0 = match result {
        Ok(()) => format!(
            "exported {} steps to {} and {}",
            by_landmarks.steps.len(),
            landmarks_path.display(),
            walls_path.display()
        ),
        Err(e) => format!("export failed: {e}"),
    };
}

fn replay_keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    log: Res<RobotLog>,
    mut playhead: ResMut<Playhead>,
    mut source: ResMut<ReplaySource>,
    mut next: ResMut<NextState<Mode>>,
    mut status: ResMut<Status>,
) {
    let last = log.len().saturating_sub(1);
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
        let other = if source.prefix == DEFAULT_PREFIX {
            LIVE_PREFIX
        } else {
            DEFAULT_PREFIX
        };
        if other == LIVE_PREFIX && !Path::new(&format!("{LIVE_PREFIX}_motors.txt")).exists() {
            status.0 = "no live run saved yet: Tab to Live, drive, R to save".into();
        } else {
            source.prefix = other.into();
            // set() re-runs OnEnter(Replay) even though we're already in
            // Replay: that's what loads the other files.
            next.set(Mode::Replay);
        }
    }
}

fn live_keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    log: Res<RobotLog>,
    mut live: ResMut<LiveSim>,
    mut drive: ResMut<DriveCommand>,
    mut status: ResMut<Status>,
) {
    if keys.just_pressed(KeyCode::Space) {
        live.paused = !live.paused;
    }
    if typed.pressed('a') {
        live.autopilot = !live.autopilot;
    }
    if typed.pressed('r') {
        status.0 = save_log(&log.log, LIVE_PREFIX).unwrap_or_else(|e| e);
    }
    let new = drive_from_keys(
        keys.pressed(KeyCode::ArrowUp),
        keys.pressed(KeyCode::ArrowDown),
        keys.pressed(KeyCode::ArrowLeft),
        keys.pressed(KeyCode::ArrowRight),
    );
    drive.set_if_neq(new);
}

/// Arrow keys to wheel speeds: up/down drive, left/right turn (on the spot
/// when neither up nor down is held).
pub fn drive_from_keys(up: bool, down: bool, left: bool, right: bool) -> DriveCommand {
    let forward = match (up, down) {
        (true, false) => MAX_SPEED,
        (false, true) => -MAX_SPEED / 2.0,
        _ => 0.0,
    };
    let turn = match (left, right) {
        (true, false) => STEER,
        (false, true) => -STEER,
        _ => 0.0,
    };
    // Keep within the motor's top speed.
    let scale = MAX_SPEED / (forward.abs() + turn.abs()).max(MAX_SPEED);
    DriveCommand {
        left: (forward - turn) * scale,
        right: (forward + turn) * scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_to_wheels() {
        let up = drive_from_keys(true, false, false, false);
        assert_eq!((up.left, up.right), (MAX_SPEED, MAX_SPEED));
        // Left alone: turn on the spot, counterclockwise (right wheel forward).
        let left = drive_from_keys(false, false, true, false);
        assert!(left.right > 0.0 && left.left == -left.right);
        // Up and left: the right wheel faster, neither above top speed.
        let both = drive_from_keys(true, false, true, false);
        assert!(both.right > both.left && both.right <= MAX_SPEED + 1e-3);
        // Up and down cancel.
        assert_eq!(
            drive_from_keys(true, true, false, false),
            DriveCommand::default()
        );
    }
}
