// Autonomous vehicles 05: the particle filter (Bevy 0.20, av_sim).
// Unit E of Claus Brenner's SLAM course, on a synthetic robot.
//
// Run with:     cargo run                       replay data/generated_*.txt
//                                               (generated on the first run)
//               cargo run -- live               drive the robot live
//               cargo run -- replay data/live   replay a saved live run
//               cargo run -- generate [seed] [steps]
//                                               write data/generated_*.txt, no window
// Test with:    cargo test   (and in ../av_sim: cargo test)
// Measure:      in ../av_sim: cargo test unit_e::tests::measure -- --ignored --nocapture
//
// WHAT YOU SEE. Left, the arena: the true path (green), dead reckoning
// (orange, Unit A), Unit D's Kalman filter (blue) and the particle
// filter's mean (pink). The pink ticks are the particles: 500 guesses of
// where the robot is and which way it faces. Each step they all move with
// their own random wheel noise, then are weighed by how well the scanner's
// view fits them, then drawn again in proportion to their weights. V shows
// the weighing (the cloud before resampling, bright = heavy). U starts
// them anywhere in the arena: the robot doesn't know where it is, and the
// cloud has to find it. N changes the count, S the random seed. Right,
// how far each path is from the truth, and how many particles really
// counted each step.
//
// Where the code is:
//   av_sim (library)   robot, log format, arena, synthetic robot, Replay
//                      and Live modes (see av01)
//   av_sim/unit_a      motion model and cylinder detector (av01)
//   av_sim/unit_d      the extended Kalman filter (av04), for comparison
//   av_sim/unit_e      NEW: the particle filter
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
use av_sim::unit_e::UnitEPlugins;
use bevy::prelude::*;
use bevy::window::WindowResolution;

/// How the particles are shown (V cycles through them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// After resampling: what the filter carries to the next step.
    Cloud,
    /// Before resampling, bright by weight: what the scanner's view did.
    Weights,
    Off,
}

impl Detail {
    pub fn next(self) -> Detail {
        match self {
            Detail::Cloud => Detail::Weights,
            Detail::Weights => Detail::Off,
            Detail::Off => Detail::Cloud,
        }
    }
}

/// What to draw in the arena.
#[derive(Resource)]
pub struct Layers {
    pub dead_reckoning: bool,
    pub kalman: bool,
    pub detail: Detail,
}

impl Default for Layers {
    fn default() -> Layers {
        Layers {
            dead_reckoning: true,
            kalman: true,
            detail: Detail::Cloud,
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
                    title: "AV 05: localisation with a particle filter (Unit E)".into(),
                    resolution: WindowResolution::new(layout::WINDOW_SIZE.x, layout::WINDOW_SIZE.y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(layout::BACKGROUND))
        // The library: shared resources and frame order, the data source,
        // Unit A for dead reckoning, Unit D's Kalman filter to compare
        // with, Unit E's particle filter.
        .add_plugins((AvCorePlugin, DataSourcePlugin))
        .add_plugins((UnitAPlugins, UnitDPlugins, UnitEPlugins))
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
    use av_sim::unit_e::{ParticleLocalization, ParticleSetup, Start};
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
    /// the filter has a step per record, V cycles the detail, U switches
    /// the start and the filter starts over with it (same frame).
    #[test]
    fn keys_step_the_replay_and_switch_the_start() {
        let dir = std::env::temp_dir().join(format!("av05_test_{}", std::process::id()));
        let prefix = dir.join("generated").to_string_lossy().into_owned();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin, AvCorePlugin, DataSourcePlugin))
            .add_plugins((UnitAPlugins, UnitDPlugins, UnitEPlugins))
            .insert_resource(ReplaySource { prefix, seed: 2 })
            .init_resource::<Layers>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_plugins((typed::TypedPlugin, keys::KeysPlugin));
        app.add_message::<KeyboardInput>()
            .add_message::<KeyboardFocusLost>();
        app.update();
        assert_eq!(app.world().resource::<RobotLog>().len(), DEFAULT_STEPS);
        assert_eq!(
            app.world().resource::<ParticleLocalization>().steps.len(),
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
        assert_eq!(app.world().resource::<Layers>().detail, Detail::Weights);

        let known_major = app.world().resource::<ParticleLocalization>().steps[0]
            .ellipse
            .major;
        type_letter(&mut app, KeyCode::KeyU, "u");
        let w = app.world();
        assert_eq!(w.resource::<ParticleSetup>().start, Start::Anywhere);
        let steps = &w.resource::<ParticleLocalization>().steps;
        assert_eq!(steps.len(), DEFAULT_STEPS);
        // Spread over the arena at the start, not around the start pose.
        assert!(steps[0].ellipse.major > 3.0 * known_major);
        let _ = std::fs::remove_dir_all(dir);
    }
}
