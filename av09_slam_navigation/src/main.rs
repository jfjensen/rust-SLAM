// Autonomous vehicles 09: navigating on a SLAM map (Bevy 0.20, av_sim).
// Units F (EKF SLAM, av06) and PP (path planning, av08) of Claus
// Brenner's SLAM course, together. Not a course script: our own.
//
// Run with:     cargo run
// Test with:    cargo test   (and in ../av_sim: cargo test)
// Measure:      in ../av_sim: cargo test --release unit_nav::tests::measure -- --ignored --nocapture
//
// WHAT YOU SEE. The live robot (green: the truth) drives to a goal (pink)
// using only what EKF SLAM tells it: its pose (purple) and its map of the
// cylinders seen so far (purple crosses with 95% ellipses). It isn't told
// where the real cylinders (grey) are. Every step it builds a planning map
// from the SLAM map (red shading): each landmark becomes a disc as big as
// the cylinder, plus the robot, plus a margin, plus 2 sigmas of how unsure
// SLAM is about it. A* (av08, with the potential function) plans a route
// through it (yellow), shortened to straight legs, and pure pursuit steers
// at a point 250 mm further on (white), never past the next corner. It
// plans again when the goal changes, a new landmark appears, or a disc
// grows across the route. If it stops moving (it ran into something), it
// backs up and plans again.
//
// Click in the arena for a goal. W wanders: a random goal each time one is
// reached. T switches SLAM between the sharp scanner and the course's 600
// mm noise: the filter gets less sure, the discs grow to their cap, and
// parts of the room are cut off. D steers by dead reckoning instead of
// SLAM: watch the bump count.
//
// Where the code is:
//   av_sim/unit_f      EKF SLAM (av06)
//   av_sim/unit_pp     grid and A* (av08)
//   av_sim/unit_nav    NEW: planning map, route, pure pursuit, replanning
//   Part 1   this file    app setup, mouse, keys, bump count
//   Part 2   layout.rs    screen layout and colours
//   Part 3   draw.rs      drawing
//   Part 4   hud.rs       text
//   helpers  typed.rs     keys by letter, any layout

mod draw;
mod hud;
mod layout;
mod typed;

use av_sim::core::{AvCorePlugin, AvSet, RobotLog, Status};
use av_sim::source::{DataSourcePlugin, LiveSim, Mode};
use av_sim::unit_a::UnitAPlugins;
use av_sim::unit_f::{SlamSetup, UnitFPlugins};
use av_sim::unit_nav::{NavPlugin, NavSetup, Navigator, SteerBy};
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowResolution};
use typed::Typed;

/// What to draw (V cycles).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum View {
    /// Everything: planning map, discs, SLAM's ellipses.
    #[default]
    All,
    /// The route and the robots, no planning map.
    Route,
}

/// Time speeds (+ and -).
pub const SPEEDS: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

#[derive(Resource, Debug, Default)]
pub struct Speed(pub usize);

/// How often the TRUE robot ran into something (the simulator stopped it).
/// The navigator can't know: it counts in the app, from the simulator.
#[derive(Resource, Debug, Default)]
pub struct Bumps {
    pub count: usize,
    seen: usize,
    generation: u64,
}

/// Where the mouse is, in world (screen) coordinates (as in av08).
#[derive(Resource, Debug, Default, PartialEq)]
pub struct Pointer {
    pub at: Option<Vec2>,
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "AV 09: driving on a SLAM map (Units F + PP)".into(),
                    resolution: WindowResolution::new(layout::WINDOW_SIZE.x, layout::WINDOW_SIZE.y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(layout::BACKGROUND))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
        })
        .add_systems(Update, track_pointer.before(AvSet::Input))
        .add_plugins((draw::DrawPlugin, hud::HudPlugin))
        .add_plugins(add_app_logic)
        .run();
}

/// Everything but the window and the drawing (the tests use this too).
fn add_app_logic(app: &mut App) {
    app.add_plugins((AvCorePlugin, DataSourcePlugin))
        .add_plugins((UnitAPlugins, UnitFPlugins, NavPlugin))
        // The sharp scanner at the start: with the course's noise the
        // discs are at their cap (T shows it).
        .insert_resource(SlamSetup::sharp_scanner())
        .insert_state(Mode::Live)
        .init_resource::<View>()
        .init_resource::<Speed>()
        .init_resource::<Bumps>()
        .init_resource::<Pointer>()
        .add_plugins(typed::TypedPlugin)
        .add_systems(Update, (click, keys).in_set(AvSet::Input))
        .add_systems(Update, count_bumps.after(AvSet::Simulate));
}

// --- Part 1a: the mouse ----------------------------------------------------------

fn track_pointer(
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut pointer: ResMut<Pointer>,
) {
    let (camera, camera_transform) = *camera;
    let at = window
        .cursor_position()
        .and_then(|c| camera.viewport_to_world_2d(camera_transform, c).ok());
    pointer.set_if_neq(Pointer { at });
}

/// A left click in the arena: the goal. Wander goes off (you're steering).
fn click(
    buttons: Res<ButtonInput<MouseButton>>,
    pointer: Res<Pointer>,
    mut nav: ResMut<Navigator>,
    mut setup: ResMut<NavSetup>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    if let Some(p) = pointer.at.filter(|&p| layout::over_arena(p)) {
        nav.set_goal(layout::screen_to_world(p));
        setup.wander = false;
    }
}

// --- Part 1b: keys -----------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    mut setup: ResMut<NavSetup>,
    mut slam: ResMut<SlamSetup>,
    mut view: ResMut<View>,
    mut speed: ResMut<Speed>,
    mut time: ResMut<Time<Virtual>>,
    mut live: ResMut<LiveSim>,
    mut log: ResMut<RobotLog>,
    mut status: ResMut<Status>,
) {
    if typed.pressed('w') {
        setup.wander = !setup.wander;
    }
    if typed.pressed('t') {
        *slam = if *slam == SlamSetup::sharp_scanner() {
            SlamSetup::default()
        } else {
            SlamSetup::sharp_scanner()
        };
    }
    if typed.pressed('d') {
        setup.steer_by = match setup.steer_by {
            SteerBy::Slam => SteerBy::DeadReckoning,
            SteerBy::DeadReckoning => SteerBy::Slam,
        };
    }
    if typed.pressed('v') {
        *view = match *view {
            View::All => View::Route,
            View::Route => View::All,
        };
    }
    if keys.just_pressed(KeyCode::Space) {
        live.paused = !live.paused;
    }
    let faster = typed.pressed('+') || keys.just_pressed(KeyCode::NumpadAdd);
    let slower = typed.pressed('-') || keys.just_pressed(KeyCode::NumpadSubtract);
    if faster || slower {
        speed.0 = if faster {
            (speed.0 + 1).min(SPEEDS.len() - 1)
        } else {
            speed.0.saturating_sub(1)
        };
        time.set_relative_speed(SPEEDS[speed.0]);
    }
    if typed.pressed('n') {
        let seed = live.restart(&mut log);
        status.0 = format!("live run, seed {seed}");
    }
}

/// Count the steps where the true robot bumped (once per new log record).
fn count_bumps(log: Res<RobotLog>, live: Res<LiveSim>, mut bumps: ResMut<Bumps>) {
    if bumps.generation != log.generation {
        *bumps = Bumps {
            generation: log.generation,
            ..Bumps::default()
        };
    }
    if log.len() != bumps.seen {
        bumps.seen = log.len();
        bumps.count += live.sim.bumped as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use av_sim::synth::STEP_MS;
    use av_sim::unit_f::SlamMapping;
    use av_sim::unit_nav::NavStatus;
    use bevy::input::keyboard::{Key, KeyboardFocusLost, KeyboardInput};
    use bevy::state::app::StatesPlugin;
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    /// The app minus window and drawing, one live step per update.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                STEP_MS,
            )))
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_message::<KeyboardInput>()
            .add_message::<KeyboardFocusLost>();
        add_app_logic(&mut app);
        app.finish();
        app.cleanup();
        app
    }

    fn type_letter(app: &mut App, code: KeyCode, letter: &str) {
        app.world_mut().write_message(KeyboardInput {
            key_code: code,
            logical_key: Key::Character(letter.into()),
            state: bevy::input::ButtonState::Pressed,
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        });
        app.update();
        // Let go, so the next press is a new one.
        app.world_mut().write_message(KeyboardInput {
            key_code: code,
            logical_key: Key::Character(letter.into()),
            state: bevy::input::ButtonState::Released,
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        });
    }

    /// A click in the arena sets the goal and stops wandering; the robot
    /// gets there.
    #[test]
    fn a_click_sets_the_goal_and_the_robot_gets_there() {
        let mut app = app();
        for _ in 0..5 {
            app.update();
        }
        let goal = Vec2::new(700.0, 400.0);
        app.world_mut().resource_mut::<Pointer>().at = Some(layout::world_to_screen(goal));
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        {
            let w = app.world();
            assert!(!w.resource::<NavSetup>().wander);
            let set = w.resource::<Navigator>().goal.unwrap();
            assert!(set.distance(goal) < 0.1, "{set}");
        }
        // No InputPlugin here to clear just_pressed: do it by hand.
        let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
        buttons.release(MouseButton::Left);
        buttons.clear();
        for _ in 0..400 {
            app.update();
            if app.world().resource::<Navigator>().status == NavStatus::Arrived {
                break;
            }
        }
        let w = app.world();
        assert_eq!(w.resource::<Navigator>().status, NavStatus::Arrived);
        let truth = w.resource::<LiveSim>().sim.centre().position();
        assert!(truth.distance(goal) < 100.0, "{truth}");
        assert_eq!(w.resource::<Bumps>().count, 0);
    }

    /// The letters switch the settings; T restarts SLAM with the course's
    /// noise; N starts a new run.
    #[test]
    fn keys() {
        let mut app = app();
        for _ in 0..20 {
            app.update();
        }
        type_letter(&mut app, KeyCode::KeyW, "w");
        type_letter(&mut app, KeyCode::KeyD, "d");
        type_letter(&mut app, KeyCode::KeyT, "t");
        type_letter(&mut app, KeyCode::KeyV, "v");
        {
            let w = app.world();
            let setup = w.resource::<NavSetup>();
            assert!(!setup.wander);
            assert_eq!(setup.steer_by, SteerBy::DeadReckoning);
            assert_eq!(*w.resource::<SlamSetup>(), SlamSetup::default());
            assert_eq!(*w.resource::<View>(), View::Route);
            assert!(w.resource::<SlamMapping>().steps.len() >= 20);
        }
        let generation = app.world().resource::<RobotLog>().generation;
        type_letter(&mut app, KeyCode::KeyN, "n");
        app.update();
        let w = app.world();
        assert_eq!(w.resource::<RobotLog>().generation, generation + 1);
        assert!(w.resource::<RobotLog>().len() <= 3);
        assert_eq!(w.resource::<LiveSim>().seed, 2);
    }
}
