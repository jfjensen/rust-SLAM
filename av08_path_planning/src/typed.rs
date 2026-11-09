// Keys by what they TYPE, so they work on any keyboard layout.
// (The same file in each app.)
//
// `KeyCode::KeyA` is a POSITION on the keyboard: where A is on an English
// (QWERTY) keyboard. On a French (AZERTY) one that key types Q. That's
// right for movement (WASD and ZQSD are the same four keys), but wrong for
// "press A to switch the animation on". Every KeyboardInput message also
// carries the `logical_key`: what the key means in the user's layout. This
// file collects those.
//
// Why not Bevy's ButtonInput<Key>? It goes by the logical key for releases
// too: press Shift, press C (logical "C"), let go of Shift, let go of C
// (logical "c"), and "C" stays pressed for ever, so the next Shift+C is
// never "just pressed". Here a held key is let go by its POSITION, which
// doesn't change with Shift.
//
// Punctuation: some of it needs AltGr or Shift on other layouts ([ and ]
// on AZERTY), so the pages accept the typed character OR the key at the
// English position: `typed.pressed('[') || keys.just_pressed(BracketLeft)`.
// Space, Tab, Enter, the arrows and the digit row stay KeyCodes: they're
// in the same place everywhere.

use bevy::input::keyboard::{Key, KeyboardFocusLost, KeyboardInput};
use bevy::input::{ButtonState, InputSystems};
use bevy::prelude::*;

pub struct TypedPlugin;

impl Plugin for TypedPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Typed>()
            .add_systems(PreUpdate, read_typed.after(InputSystems));
    }
}

#[derive(Resource, Default)]
pub struct Typed {
    /// Characters typed this frame, in lower case.
    pressed: Vec<char>,
    /// Keys held down: where they are, and what they typed when pressed.
    held: Vec<(KeyCode, char)>,
}

impl Typed {
    /// Was this character (letters in lower case) typed this frame, with
    /// or without Shift?
    pub fn pressed(&self, c: char) -> bool {
        self.pressed.contains(&c)
    }

    /// Is the key that typed this character still held down? (Not every
    /// app holds keys, hence the allow.)
    #[allow(dead_code)]
    pub fn held(&self, c: char) -> bool {
        self.held.iter().any(|&(_, h)| h == c)
    }
}

fn read_typed(
    mut reader: MessageReader<KeyboardInput>,
    mut focus_lost: MessageReader<KeyboardFocusLost>,
    mut typed: ResMut<Typed>,
) {
    typed.pressed.clear();
    // Switching to another window: its key releases never arrive here.
    if focus_lost.read().count() > 0 {
        typed.held.clear();
    }
    for input in reader.read() {
        match input.state {
            ButtonState::Pressed => {
                // Holding a key down repeats it; only the first press counts.
                if input.repeat {
                    continue;
                }
                if let Key::Character(text) = &input.logical_key {
                    for c in text.chars().flat_map(char::to_lowercase) {
                        typed.pressed.push(c);
                        typed.held.push((input.key_code, c));
                    }
                }
            }
            ButtonState::Released => typed.held.retain(|&(key, _)| key != input.key_code),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(app: &mut App, key_code: KeyCode, text: &str, state: ButtonState) {
        app.world_mut().write_message(KeyboardInput {
            key_code,
            logical_key: Key::Character(text.into()),
            state,
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        });
        app.update();
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_message::<KeyboardInput>()
            .add_message::<KeyboardFocusLost>()
            .add_systems(Update, read_typed)
            .init_resource::<Typed>();
        app
    }

    /// AZERTY: the key where QWERTY has Q types "a". It must count as A,
    /// not as Q.
    #[test]
    fn letters_go_by_what_they_type() {
        let mut app = app();
        press(&mut app, KeyCode::KeyQ, "a", ButtonState::Pressed);
        let typed = app.world().resource::<Typed>();
        assert!(typed.pressed('a') && !typed.pressed('q'));
    }

    /// Shift+C, let go of Shift first: C still counts as let go, and the
    /// next Shift+C is a new press.
    #[test]
    fn shift_released_first_does_not_stick() {
        let mut app = app();
        press(&mut app, KeyCode::KeyC, "C", ButtonState::Pressed);
        assert!(app.world().resource::<Typed>().held('c'));
        press(&mut app, KeyCode::KeyC, "c", ButtonState::Released);
        assert!(!app.world().resource::<Typed>().held('c'));
        press(&mut app, KeyCode::KeyC, "C", ButtonState::Pressed);
        assert!(app.world().resource::<Typed>().pressed('c'));
    }
}
