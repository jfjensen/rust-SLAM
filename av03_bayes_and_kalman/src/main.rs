// Autonomous vehicles 03: the Bayes filter in 1D (Bevy 0.20, av_sim).
// Unit C of Claus Brenner's SLAM course, on a synthetic track.
//
// Run with:     cargo run                         replay data/track.txt
//                                                 (generated on the first run)
//               cargo run -- live                 drive the robot live
//               cargo run -- replay data/live_track.txt
//                                                 replay a saved live run
//               cargo run -- generate [seed] [steps]
//                                                 write data/track.txt, no window
// Test with:    cargo test   (and in ../av_sim: cargo test)
// Measure:      in ../av_sim: cargo test unit_c::tests::measure -- --ignored --nocapture
//
// WHAT YOU SEE. A robot on a straight track. It is told to move 20 cm a
// step, but the wheels slip, and its position sensor is noisy (and off for
// a stretch in the middle of a generated run). It can't know where it is,
// only how likely each place is: a probability distribution, the belief.
// Each step, two things happen to the belief:
//
//   predict   move it by the control. It gets wider (blue).
//   correct   multiply it by what the sensor says (green). It gets
//             narrower, and moves towards the measurement (red).
//
// The histogram filter does this with a list of probabilities, one per cm
// (light, stepped lines). The Kalman filter does it with two numbers, a
// mean and a variance (full lines). With gaussians they agree; G switches
// the histogram filter to triangles, as the course's scripts can.
//
// Where the code is:
//   av_sim (library)   frame order, Playhead, the Mode state (see av01)
//   av_sim/unit_c      NEW: distributions, histogram and Kalman filters,
//                      the track robot and its Replay/Live source
//   Part 1   this file    command line, app setup
//   Part 2   layout.rs    screen layout and colours
//   Part 3   draw.rs      gizmo drawing
//   Part 4   keys.rs      keys
//   Part 5   hud.rs       text
//   helpers  typed.rs     keys by letter, any layout

mod draw;
mod hud;
mod keys;
mod layout;
mod typed;

use av_sim::core::AvCorePlugin;
use av_sim::source::Mode;
use av_sim::unit_c::UnitCPlugins;
use av_sim::unit_c::source::{DEFAULT_TRACK, TrackFile, TrackSourcePlugin};
use av_sim::unit_c::track::{DEFAULT_STEPS, generate};
use bevy::prelude::*;
use bevy::window::WindowResolution;

/// Which filter's curves the belief plot shows (V cycles through them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Both,
    Histogram,
    Kalman,
}

impl View {
    pub fn next(self) -> View {
        match self {
            View::Both => View::Histogram,
            View::Histogram => View::Kalman,
            View::Kalman => View::Both,
        }
    }

    pub fn histogram(self) -> bool {
        self != View::Kalman
    }

    pub fn kalman(self) -> bool {
        self != View::Histogram
    }
}

/// What to draw.
#[derive(Resource)]
pub struct Layers {
    pub view: View,
    /// The belief before this step (grey).
    pub before: bool,
}

impl Default for Layers {
    fn default() -> Layers {
        Layers {
            view: View::Both,
            before: true,
        }
    }
}

// --- Part 1a: the command line -----------------------------------------------------

#[derive(Debug, PartialEq)]
enum Command {
    Generate { seed: u64, steps: usize },
    Run { mode: Mode, path: String },
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    let number = |i: usize, default: u64| -> Result<u64, String> {
        args.get(i).map_or(Ok(default), |s| {
            s.parse().map_err(|_| format!("'{s}' is not a number"))
        })
    };
    match args.first().map(String::as_str) {
        None => Ok(Command::Run {
            mode: Mode::Replay,
            path: DEFAULT_TRACK.into(),
        }),
        Some("live") => Ok(Command::Run {
            mode: Mode::Live,
            path: DEFAULT_TRACK.into(),
        }),
        Some("replay") => Ok(Command::Run {
            mode: Mode::Replay,
            path: args.get(1).cloned().unwrap_or_else(|| DEFAULT_TRACK.into()),
        }),
        Some("generate") => Ok(Command::Generate {
            seed: number(1, 1)?,
            steps: number(2, DEFAULT_STEPS as u64)? as usize,
        }),
        Some(other) => Err(format!(
            "unknown command '{other}': use live, replay [file] or generate [seed] [steps]"
        )),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, path) = match parse_args(&args) {
        Ok(Command::Run { mode, path }) => (mode, path),
        Ok(Command::Generate { seed, steps }) => {
            match generate(seed, steps).write(DEFAULT_TRACK) {
                Ok(()) => println!("wrote {steps} steps (seed {seed}) to {DEFAULT_TRACK}"),
                Err(e) => eprintln!("could not write {e}"),
            }
            return;
        }
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    // --- Part 1b: the app ----------------------------------------------------------
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "AV 03: the Bayes filter in 1D, histogram and Kalman (Unit C)".into(),
                    resolution: WindowResolution::new(layout::WINDOW_SIZE.x, layout::WINDOW_SIZE.y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(layout::BACKGROUND))
        // The library: shared resources and frame order, the track as the
        // data source (instead of Units A and B's robot logs), Unit C's two
        // filters.
        .add_plugins((AvCorePlugin, TrackSourcePlugin, UnitCPlugins))
        .insert_resource(TrackFile { path, seed: 1 })
        .insert_state(mode)
        // This app: keys, drawing, text.
        .init_resource::<Layers>()
        .add_plugins((
            typed::TypedPlugin,
            keys::KeysPlugin,
            draw::DrawPlugin,
            hud::HudPlugin,
        ))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
        })
        .run();
}

pub fn shift_held(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
}

#[cfg(test)]
mod tests {
    use super::*;
    use av_sim::core::Playhead;
    use av_sim::unit_c::{HistogramFilter, HistogramParams, KalmanFilter, Shape, Track};
    use bevy::input::keyboard::{Key, KeyboardFocusLost, KeyboardInput};
    use bevy::state::app::StatesPlugin;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn command_line() {
        assert_eq!(
            parse_args(&args("")),
            Ok(Command::Run {
                mode: Mode::Replay,
                path: DEFAULT_TRACK.into()
            })
        );
        assert_eq!(
            parse_args(&args("replay data/live_track.txt")),
            Ok(Command::Run {
                mode: Mode::Replay,
                path: "data/live_track.txt".into()
            })
        );
        assert_eq!(
            parse_args(&args("generate 7")),
            Ok(Command::Generate {
                seed: 7,
                steps: DEFAULT_STEPS
            })
        );
        assert!(parse_args(&args("generate x")).is_err());
        assert!(parse_args(&args("fly")).is_err());
    }

    /// Type a letter: typed.rs reads KeyboardInput messages.
    fn type_letter(app: &mut App, code: KeyCode, letter: &str) {
        // No InputPlugin here to clear last frame's just_pressed arrows.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release_all();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut().write_message(KeyboardInput {
            key_code: code,
            logical_key: Key::Character(letter.into()),
            state: bevy::input::ButtonState::Pressed,
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        });
        app.update();
    }

    /// The whole app minus window and rendering: the keys step the replay,
    /// V cycles the view, G switches the histogram filter's shape (and it
    /// starts over, in the same frame).
    #[test]
    fn keys_step_the_replay_and_switch_the_shape() {
        let dir = std::env::temp_dir().join(format!("av03_test_{}", std::process::id()));
        let path = dir.join("track.txt").to_string_lossy().into_owned();
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            StatesPlugin,
            AvCorePlugin,
            TrackSourcePlugin,
        ))
        .add_plugins(UnitCPlugins)
        .insert_resource(TrackFile { path, seed: 2 })
        .init_resource::<Layers>()
        .init_resource::<ButtonInput<KeyCode>>()
        .add_plugins((typed::TypedPlugin, keys::KeysPlugin));
        app.add_message::<KeyboardInput>()
            .add_message::<KeyboardFocusLost>();
        app.update();
        assert_eq!(app.world().resource::<Track>().len(), DEFAULT_STEPS);
        assert_eq!(
            app.world().resource::<KalmanFilter>().steps.len(),
            DEFAULT_STEPS
        );

        let press = |app: &mut App, key: KeyCode| {
            let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            input.release_all();
            input.clear();
            input.press(key);
            app.update();
        };
        press(&mut app, KeyCode::End);
        assert_eq!(app.world().resource::<Playhead>().index, DEFAULT_STEPS - 1);
        press(&mut app, KeyCode::ArrowLeft);
        let ph = app.world().resource::<Playhead>();
        assert_eq!(ph.index, DEFAULT_STEPS - 2);
        assert!(!ph.playing);

        type_letter(&mut app, KeyCode::KeyV, "v");
        assert_eq!(app.world().resource::<Layers>().view, View::Histogram);

        let gaussian_end = app.world().resource::<HistogramFilter>().steps[DEFAULT_STEPS - 1]
            .correction
            .clone();
        type_letter(&mut app, KeyCode::KeyG, "g");
        let w = app.world();
        assert_eq!(w.resource::<HistogramParams>().shape, Shape::Triangle);
        let steps = &w.resource::<HistogramFilter>().steps;
        assert_eq!(steps.len(), DEFAULT_STEPS);
        assert_ne!(steps[DEFAULT_STEPS - 1].correction, gaussian_end);
        let _ = std::fs::remove_dir_all(dir);
    }
}
