// Autonomous vehicles 01: motion model and LiDAR scan (Bevy 0.20, av_sim).
// Unit A of Claus Brenner's SLAM course, on a synthetic robot.
//
// Run with:     cargo run                       replay data/generated_*.txt
//                                               (generated on the first run)
//               cargo run -- live               drive the robot live
//               cargo run -- replay data/live   replay a saved live run
//               cargo run -- generate [seed] [steps]
//                                               write data/generated_*.txt, no window
// Test with:    cargo test   (and in ../av_sim: cargo test)
// Measure:      in ../av_sim: cargo test measure -- --ignored --nocapture
//
// WHAT YOU SEE. Left, the arena from above: walls, the six cylinders, the
// true path (green) and the path the motion model computes from the wheel
// ticks alone (orange). The scan and the detected cylinders are placed
// with the orange pose, so as dead reckoning drifts, the scan slides off
// the walls. Right, the current scan as the course's viewer shows it,
// and its derivative with the detector's edge threshold.
//
// Where the code is:
//   av_sim (library)   Parts 1-6: robot, log format, arena, synthetic
//                      robot, shared resources, Replay and Live modes
//   av_sim/unit_a      the course's Unit A algorithms, as plugins
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
use bevy::prelude::*;
use bevy::window::WindowResolution;

/// What to draw in the arena.
#[derive(Resource)]
pub struct Layers {
    pub scan: bool,
    pub cylinders: bool,
}

impl Default for Layers {
    fn default() -> Layers {
        Layers {
            scan: true,
            cylinders: true,
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
                    title: "AV 01: motion model and LiDAR scan (Unit A)".into(),
                    resolution: WindowResolution::new(layout::WINDOW_SIZE.x, layout::WINDOW_SIZE.y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(layout::BACKGROUND))
        // The library: shared resources and frame order, the data source,
        // and the course's Unit A algorithms.
        .add_plugins((AvCorePlugin, DataSourcePlugin))
        .add_plugins(UnitAPlugins)
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
    use av_sim::unit_a::motion::DeadReckoning;
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

    /// The whole app minus window and rendering: the keys pause the replay
    /// and step it, in the same frame (Input runs before Simulate).
    #[test]
    fn keys_step_the_replay() {
        let dir = std::env::temp_dir().join(format!("av01_test_{}", std::process::id()));
        let prefix = dir.join("generated").to_string_lossy().into_owned();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin, AvCorePlugin, DataSourcePlugin))
            .add_plugins(UnitAPlugins)
            .insert_resource(ReplaySource { prefix, seed: 2 })
            .init_resource::<Layers>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_plugins((typed::TypedPlugin, keys::KeysPlugin));
        // typed.rs reads these messages; the window plugins would add them.
        app.add_message::<bevy::input::keyboard::KeyboardInput>()
            .add_message::<bevy::input::keyboard::KeyboardFocusLost>();
        app.update();
        assert_eq!(app.world().resource::<RobotLog>().len(), DEFAULT_STEPS);

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
        // Dead reckoning has a pose for every step.
        assert_eq!(
            app.world().resource::<DeadReckoning>().poses.len(),
            DEFAULT_STEPS
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
