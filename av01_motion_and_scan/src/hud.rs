// Part 5: the text. A header (mode, step, status), an info line like the
// course's LegoLogfile.info(), the keys, and the plot captions.

use crate::layout::{DERIVATIVE_PLOT, HELP_AT, Plot, RANGE_PLOT};
use av_sim::core::{AvSet, Playhead, RobotLog, Status};
use av_sim::robot::Pose;
use av_sim::source::{LiveSim, Mode, ReplaySource};
use av_sim::unit_a::cylinders::DetectedCylinders;
use av_sim::unit_a::motion::DeadReckoning;
use bevy::prelude::*;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_text)
            .add_systems(Update, (update_header, update_help).in_set(AvSet::Present));
    }
}

#[derive(Component)]
struct Header;

#[derive(Component)]
struct Help;

const TEXT: Color = Color::srgb(0.95, 0.92, 0.7);
const PANEL: Color = Color::srgba(0.05, 0.05, 0.08, 0.75);

fn spawn_text(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(15.0),
        TextColor(TEXT),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(6.0),
            left: Val::Px(8.0),
            padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(PANEL),
        Header,
    ));
    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(13.0),
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(HELP_AT.x),
            top: Val::Px(HELP_AT.y),
            padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(PANEL),
        Help,
    ));
    for (plot, caption) in [
        (&RANGE_PLOT, "scan: range (mm) by beam, cylinders in cyan"),
        (&DERIVATIVE_PLOT, "derivative; red lines: edge threshold"),
    ] {
        spawn_caption(&mut commands, plot, caption);
    }
}

fn spawn_caption(commands: &mut Commands, plot: &Plot, caption: &str) {
    let at = plot.caption_at();
    commands.spawn((
        Text::new(caption),
        TextFont::from_font_size(12.0),
        TextColor(Color::srgb(0.7, 0.7, 0.75)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(at.x + 4.0),
            top: Val::Px(at.y + 2.0),
            ..default()
        },
    ));
}

#[allow(clippy::too_many_arguments)]
fn update_header(
    mode: Res<State<Mode>>,
    source: Res<ReplaySource>,
    live: Res<LiveSim>,
    log: Res<RobotLog>,
    playhead: Res<Playhead>,
    dr: Res<DeadReckoning>,
    found: Res<DetectedCylinders>,
    status: Res<Status>,
    mut header: Single<&mut Text, With<Header>>,
) {
    let i = playhead.index;
    let what = match mode.get() {
        Mode::Replay => format!(
            "REPLAY {}   step {}/{}{}",
            source.prefix,
            i + 1,
            log.len(),
            if playhead.playing { "" } else { "   (paused)" }
        ),
        Mode::Live => format!(
            "LIVE   step {}   {}{}{}",
            i + 1,
            if live.autopilot {
                "autopilot"
            } else {
                "arrow keys"
            },
            if live.paused { "   (paused)" } else { "" },
            if live.sim.bumped { "   BUMP" } else { "" },
        ),
    };
    let text = format!(
        "AV 01 - Unit A: motion model and cylinders   {what}\n{}\n{}",
        info(
            &log,
            i,
            dr.poses.get(i),
            found.per_scan.get(i).map(Vec::len)
        ),
        status.0
    );
    if header.0 != text {
        header.0 = text;
    }
}

/// The course's info() line for step i: reference position, motor
/// increments, dead-reckoning pose, its error, cylinders found.
fn info(log: &RobotLog, i: usize, pose: Option<&Pose>, cylinders: Option<usize>) -> String {
    let mut s = String::new();
    let truth = log.log.reference.get(i);
    if let Some(r) = truth {
        s += &format!("true {:4} {:4}", r.x, r.y);
    }
    let motors = &log.log.motors;
    if i < motors.len() {
        let (l, r) = match i {
            0 => (0, 0),
            _ => (
                motors[i].left - motors[i - 1].left,
                motors[i].right - motors[i - 1].right,
            ),
        };
        s += &format!(" | motor {l:4} {r:4}");
    }
    if let Some(p) = pose {
        s += &format!(
            " | dead reckoning {:6.1} {:6.1} {:5.1} deg",
            p.x,
            p.y,
            p.heading.to_degrees()
        );
        if let Some(r) = truth {
            let error = p.position().distance(Vec2::new(r.x as f32, r.y as f32));
            s += &format!(" | off by {error:5.1} mm");
        }
    }
    if let Some(n) = cylinders {
        s += &format!(" | {n} cylinders");
    }
    s
}

fn update_help(mode: Res<State<Mode>>, mut help: Single<&mut Text, With<Help>>) {
    if !mode.is_changed() {
        return;
    }
    let keys = match mode.get() {
        Mode::Replay => {
            "Space play/pause   Left/Right step (Shift: 10)   Home/End\n\
             L  data/generated <-> data/live"
        }
        Mode::Live => {
            "Space pause   A autopilot on/off   arrows drive\n\
             R  save run to data/live_*.txt"
        }
    };
    help.0 = format!(
        "{keys}\nS scan points   C cylinders   E export F/D files   Tab replay/live\n\
         green: truth   orange: dead reckoning   cyan: detections"
    );
}
