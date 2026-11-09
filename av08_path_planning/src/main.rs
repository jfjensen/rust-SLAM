// Autonomous vehicles 08: path planning (Bevy 0.20, av_sim).
// Unit PP of Claus Brenner's SLAM course.
//
// Run with:     cargo run
// Test with:    cargo test   (and in ../av_sim: cargo test)
// Measure:      in ../av_sim: cargo test --release unit_pp::tests::measure -- --ignored --nocapture
//
// WHAT YOU SEE. Two pages (Tab):
//
//   GRID  a 200 x 150 grid. Paint walls (red) with the left mouse button,
//         erase with the right; Shift + left places the start, Shift +
//         right the goal. The search spreads out from the start (green,
//         brighter = costlier to reach; yellow = the front, the cells
//         waiting to be looked at) until it reaches the goal, then the
//         path is drawn. 1 Dijkstra: a ring of water. 2 A*: a beam towards
//         the goal, same path, far fewer cells. 3 A* with the potential
//         function (blue): the path keeps away from walls, for a price.
//   CAR   a car that can't turn on the spot: it drives arcs, 5 cells
//         long, turning radius 10. A* now searches over (x, y, heading).
//         Shift + left-drag places the start AND points it; Shift +
//         right-drag the goal. B allows reversing.
//
// Both pages: the search is animated (Space play/pause, Right one step,
// + and - speed, Enter to the end, I instant: solve at once after every
// edit, as the course does). M loads the SLAM arena (av01-av07) with its
// walls and cylinders grown by the robot's half-width; R random walls.
//
// Where the code is:
//   av_sim/unit_pp     NEW: grid, distance transform, Dijkstra and A*, the car
//   Part 1   this file      pages, animation, the mouse pointer, app setup
//   Part 2   layout.rs      screen layout and colours
//   Part 3   grid_page.rs   the grid page: editing, stepping, drawing
//   Part 4   car_page.rs    the car page: the same for the car
//   Part 5   hud.rs         text
//   helpers  typed.rs       keys by letter, any layout

mod car_page;
mod grid_page;
mod hud;
mod layout;
mod typed;

use av_sim::core::{AvCorePlugin, AvSet};
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowResolution};
use typed::Typed;

// --- Part 1a: pages and the animation --------------------------------------------

#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    #[default]
    Grid,
    Car,
}

/// Cells (or car states) expanded per frame at each speed.
pub const SPEEDS: [usize; 7] = [1, 4, 16, 64, 256, 1024, 4096];

/// How the search on the current page is stepped.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct Animation {
    pub playing: bool,
    /// Index into SPEEDS.
    pub speed: usize,
    /// Solve at once after every edit (the course's behaviour).
    pub instant: bool,
    /// One step, then pause (Right).
    step_once: bool,
    /// Run to the end this frame (Enter).
    finish: bool,
}

impl Default for Animation {
    fn default() -> Animation {
        Animation {
            playing: true,
            speed: 3,
            instant: false,
            step_once: false,
            finish: false,
        }
    }
}

impl Animation {
    /// How many steps to take this frame; consumes Right and Enter.
    pub fn take_steps(&mut self) -> usize {
        let n = if self.finish || self.instant {
            usize::MAX
        } else if self.playing {
            SPEEDS[self.speed]
        } else if self.step_once {
            1
        } else {
            0
        };
        self.finish = false;
        self.step_once = false;
        n
    }

    /// After an edit: play the new search from the start.
    pub fn restart(&mut self) {
        self.playing = true;
    }
}

// --- Part 1b: the mouse ------------------------------------------------------------

/// Where the cursor is, in world (= screen, the camera doesn't move)
/// coordinates. Filled from the window; the
/// pages read only this, so a test can set it directly.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct Pointer {
    pub at: Option<Vec2>,
}

/// What the mouse is doing to the map this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseAction {
    Paint,
    Erase,
    /// Shift + left: press places the start, drag points it.
    Start {
        pressed: bool,
    },
    /// Shift + right: the same for the goal.
    Goal {
        pressed: bool,
    },
}

pub fn mouse_action(buttons: &ButtonInput<MouseButton>, shift: bool) -> Option<MouseAction> {
    let left = buttons.pressed(MouseButton::Left);
    let right = buttons.pressed(MouseButton::Right);
    match (shift, left, right) {
        (false, true, _) => Some(MouseAction::Paint),
        (false, false, true) => Some(MouseAction::Erase),
        (true, true, _) => Some(MouseAction::Start {
            pressed: buttons.just_pressed(MouseButton::Left),
        }),
        (true, false, true) => Some(MouseAction::Goal {
            pressed: buttons.just_pressed(MouseButton::Right),
        }),
        _ => None,
    }
}

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

pub fn shift_held(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
}

// --- Part 1c: keys for both pages ---------------------------------------------------

fn shared_keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    mut page: ResMut<Page>,
    mut anim: ResMut<Animation>,
) {
    if keys.just_pressed(KeyCode::Tab) {
        *page = match *page {
            Page::Grid => Page::Car,
            Page::Car => Page::Grid,
        };
    }
    if keys.just_pressed(KeyCode::Space) {
        anim.playing = !anim.playing;
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        anim.playing = false;
        anim.step_once = true;
    }
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
        anim.finish = true;
    }
    // + and -: typed, or the numpad's (on AZERTY '+' needs Shift).
    if typed.pressed('+') || keys.just_pressed(KeyCode::NumpadAdd) {
        anim.speed = (anim.speed + 1).min(SPEEDS.len() - 1);
    }
    if typed.pressed('-') || keys.just_pressed(KeyCode::NumpadSubtract) {
        anim.speed = anim.speed.saturating_sub(1);
    }
    if typed.pressed('i') {
        anim.instant = !anim.instant;
    }
}

// --- Part 1d: the app ----------------------------------------------------------------

/// Everything but the window: shared by main() and the tests.
pub fn add_app_logic(app: &mut App) {
    app.add_plugins(AvCorePlugin)
        .init_resource::<Page>()
        .init_resource::<Animation>()
        .init_resource::<Pointer>()
        .add_plugins(typed::TypedPlugin)
        .add_systems(Update, shared_keys.in_set(AvSet::Input))
        .add_plugins((grid_page::GridPagePlugin, car_page::CarPagePlugin));
}

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "AV 08: path planning, Dijkstra, A* and a car (Unit PP)".into(),
                    resolution: WindowResolution::new(layout::WINDOW_SIZE.x, layout::WINDOW_SIZE.y)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            })
            // One texture pixel per cell, shown 4 or 8 pixels wide: keep
            // them sharp.
            .set(ImagePlugin::default_nearest()),
    )
    .insert_resource(ClearColor(layout::BACKGROUND));
    add_app_logic(&mut app);
    app.add_plugins((
        grid_page::GridDrawPlugin,
        car_page::CarDrawPlugin,
        hud::HudPlugin,
    ))
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn(Camera2d);
    })
    // Before the pages read it (they edit in AvSet::Input).
    .add_systems(Update, track_pointer.before(AvSet::Input))
    .run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use av_sim::unit_pp::search::Status;
    use bevy::input::keyboard::{Key, KeyboardFocusLost, KeyboardInput};

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            bevy::image::ImagePlugin::default(),
        ));
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_message::<KeyboardInput>()
            .add_message::<KeyboardFocusLost>();
        add_app_logic(&mut app);
        // App::run() calls these; a test calling update() must do it
        // itself (plugins add some resources only in finish()).
        app.finish();
        app.cleanup();
        app.update();
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
    }

    fn press_key(app: &mut App, key: KeyCode) {
        let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        input.release_all();
        input.clear();
        input.press(key);
        app.update();
        let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        input.release_all();
        input.clear();
    }

    #[test]
    fn mouse_buttons_to_actions() {
        let mut b = ButtonInput::<MouseButton>::default();
        assert_eq!(mouse_action(&b, false), None);
        b.press(MouseButton::Left);
        assert_eq!(mouse_action(&b, false), Some(MouseAction::Paint));
        assert_eq!(
            mouse_action(&b, true),
            Some(MouseAction::Start { pressed: true })
        );
        b.clear();
        assert_eq!(
            mouse_action(&b, true),
            Some(MouseAction::Start { pressed: false })
        );
        b.release_all();
        b.press(MouseButton::Right);
        assert_eq!(mouse_action(&b, false), Some(MouseAction::Erase));
    }

    /// The app without a window: Enter finishes the search; painting with
    /// the pointer adds walls and restarts it; 2 picks A*; Tab goes to
    /// the car page, which finds its path too.
    #[test]
    fn pointer_and_keys_drive_both_pages() {
        let mut app = app();
        press_key(&mut app, KeyCode::Enter);
        let page = app.world().resource::<grid_page::GridPage>();
        assert_eq!(page.search.status, Status::Found);
        let walls_before = page.grid.walls.iter().filter(|w| **w).count();

        // Paint at the middle of the map.
        let middle = layout::MAP_MIN + layout::MAP_SIZE / 2.0;
        app.world_mut().resource_mut::<Pointer>().at = Some(middle);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .release_all();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .clear();
        let page = app.world().resource::<grid_page::GridPage>();
        assert!(page.grid.walls.iter().filter(|w| **w).count() > walls_before);
        assert!(page.grid.is_wall(100, 75));

        // The digit row goes by position (any layout).
        press_key(&mut app, KeyCode::Digit2);
        assert_eq!(
            app.world().resource::<grid_page::GridPage>().planner,
            grid_page::Planner::AStar
        );

        press_key(&mut app, KeyCode::Tab);
        assert_eq!(*app.world().resource::<Page>(), Page::Car);
        press_key(&mut app, KeyCode::Enter);
        let car = app.world().resource::<car_page::CarPage>();
        assert_eq!(car.search.status, Status::Found);
        assert!(!car.search.path().is_empty());
        // B by what it types: reversing on, and a new search.
        let version = car.version;
        type_letter(&mut app, KeyCode::KeyB, "b");
        let car = app.world().resource::<car_page::CarPage>();
        assert!(car.backward);
        assert_eq!(car.version, version + 1);
    }
}
