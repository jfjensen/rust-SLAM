// Autonomous vehicles 04: the extended Kalman filter (Bevy 0.20, av_sim).
// Unit D of Claus Brenner's SLAM course, on a synthetic robot.
//
// Run with:     cargo run                       replay data/generated_*.txt
//                                               (generated on the first run)
//               cargo run -- live               drive the robot live
//               cargo run -- replay data/live   replay a saved live run
//               cargo run -- generate [seed] [steps]
//                                               write data/generated_*.txt, no window
// Test with:    cargo test   (and in ../av_sim: cargo test)
// Measure:      in ../av_sim: cargo test unit_d::tests::measure -- --ignored --nocapture
//
// WHAT YOU SEE. Left, the arena: the true path (green), dead reckoning
// (orange, Unit A), and the Kalman filter's path (purple). Around the
// robot, the filter's 95% error ellipse: where it believes the robot is,
// 19 times in 20. Blue: after predicting (bigger), purple: after
// correcting with the landmarks it saw (yellow lines). The wedge in front
// is the heading's 95% range. V shows the ellipses along the whole path.
// T switches between the course's noise settings and ones that match the
// synthetic scanner. Right, how far each path is from the truth, and how
// big the ellipse is before and after each correction.
//
// Where the code is:
//   av_sim (library)   robot, log format, arena, synthetic robot, Replay
//                      and Live modes (see av01)
//   av_sim/unit_a      motion model and cylinder detector (av01)
//   av_sim/unit_d      NEW: the extended Kalman filter
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
use av_sim::logfile::LogPaths;
use av_sim::source::{DEFAULT_PREFIX, DataSourcePlugin, Mode, ReplaySource};
use av_sim::synth::{DEFAULT_STEPS, SynthConfig, generate};
use av_sim::unit_a::UnitAPlugins;
use av_sim::unit_d::UnitDPlugins;
use bevy::prelude::*;
use bevy::window::WindowResolution;

/// Which ellipses the arena shows (V cycles through them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// This step: predicted and corrected ellipse, the landmarks seen.
    Step,
    /// Also the corrected ellipse every few steps along the path.
    Trail,
    Off,
}

impl Detail {
    pub fn next(self) -> Detail {
        match self {
            Detail::Step => Detail::Trail,
            Detail::Trail => Detail::Off,
            Detail::Off => Detail::Step,
        }
    }
}

/// What to draw in the arena.
#[derive(Resource)]
pub struct Layers {
    pub dead_reckoning: bool,
    pub detail: Detail,
}

impl Default for Layers {
    fn default() -> Layers {
        Layers {
            dead_reckoning: true,
            detail: Detail::Step,
        }
    }
}

// --- Part 1a: the command line -----------------------------------------------------

#[derive(Debug, PartialEq)]
enum Command {
    Generate { seed: u64, steps: usize },
    Run { mode: Mode, prefix: String },
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
            prefix: DEFAULT_PREFIX.into(),
        }),
        Some("live") => Ok(Command::Run {
            mode: Mode::Live,
            prefix: DEFAULT_PREFIX.into(),
        }),
        Some("replay") => Ok(Command::Run {
            mode: Mode::Replay,
            prefix: args
                .get(1)
                .cloned()
                .unwrap_or_else(|| DEFAULT_PREFIX.into()),
        }),
        Some("generate") => Ok(Command::Generate {
            seed: number(1, 1)?,
            steps: number(2, DEFAULT_STEPS as u64)? as usize,
        }),
        Some(other) => Err(format!(
            "unknown command '{other}': use live, replay [prefix] or generate [seed] [steps]"
        )),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, prefix) = match parse_args(&args) {
        Ok(Command::Run { mode, prefix }) => (mode, prefix),
        Ok(Command::Generate { seed, steps }) => {
            let log = generate(SynthConfig::REALISTIC, seed, steps);
            let paths = LogPaths::with_prefix(DEFAULT_PREFIX);
            match paths.write(&log) {
                Ok(()) => println!("wrote {steps} steps (seed {seed}) to {DEFAULT_PREFIX}_*.txt"),
                Err(e) => eprintln!("could not write {DEFAULT_PREFIX}_*.txt: {e}"),
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
                    title: "AV 04: localisation with the extended Kalman filter (Unit D)".into(),
                    resolution: WindowResolution::new(layout::WINDOW_SIZE.x, layout::WINDOW_SIZE.y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(layout::BACKGROUND))
        // The library: shared resources and frame order, the data source,
        // Unit A for dead reckoning, Unit D's filter.
        .add_plugins((AvCorePlugin, DataSourcePlugin))
        .add_plugins((UnitAPlugins, UnitDPlugins))
        .insert_resource(ReplaySource { prefix, seed: 1 })
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
    use av_sim::core::{Playhead, RobotLog};
    use av_sim::unit_d::{EkfLocalization, EkfSetup};
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
                prefix: DEFAULT_PREFIX.into()
            })
        );
        assert_eq!(
            parse_args(&args("replay data/live")),
            Ok(Command::Run {
                mode: Mode::Replay,
                prefix: "data/live".into()
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
        // No InputPlugin here to clear last frame's just_pressed keys.
        let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        input.release_all();
        input.clear();
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
    /// the filter has a step per record, V cycles the detail, T switches
    /// the settings and the filter starts over with them (same frame).
    #[test]
    fn keys_step_the_replay_and_switch_the_settings() {
        let dir = std::env::temp_dir().join(format!("av04_test_{}", std::process::id()));
        let prefix = dir.join("generated").to_string_lossy().into_owned();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin, AvCorePlugin, DataSourcePlugin))
            .add_plugins((UnitAPlugins, UnitDPlugins))
            .insert_resource(ReplaySource { prefix, seed: 2 })
            .init_resource::<Layers>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_plugins((typed::TypedPlugin, keys::KeysPlugin));
        app.add_message::<KeyboardInput>()
            .add_message::<KeyboardFocusLost>();
        app.update();
        assert_eq!(app.world().resource::<RobotLog>().len(), DEFAULT_STEPS);
        assert_eq!(
            app.world().resource::<EkfLocalization>().steps.len(),
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
        assert_eq!(app.world().resource::<Layers>().detail, Detail::Trail);

        let course_major = app.world().resource::<EkfLocalization>().steps[10]
            .ellipse()
            .major;
        type_letter(&mut app, KeyCode::KeyT, "t");
        let w = app.world();
        assert_eq!(*w.resource::<EkfSetup>(), EkfSetup::sharp_scanner());
        let steps = &w.resource::<EkfLocalization>().steps;
        assert_eq!(steps.len(), DEFAULT_STEPS);
        assert!(steps[10].ellipse().major < course_major);
        let _ = std::fs::remove_dir_all(dir);
    }
}
