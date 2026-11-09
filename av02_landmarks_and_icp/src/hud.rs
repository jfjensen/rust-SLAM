// Part 5: the text. A header (mode, step, status), a line with how far
// each path is from the truth, the keys, and the plot captions.

use crate::Layers;
use crate::layout::{CORRECTION_PLOT, ERROR_PLOT, HELP_AT, Plot};
use av_sim::core::{AvSet, Playhead, RobotLog, Status};
use av_sim::robot::Pose;
use av_sim::source::{LiveSim, Mode, ReplaySource};
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_b::landmarks::LandmarkCorrection;
use av_sim::unit_b::walls::WallCorrection;
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
        (
            &ERROR_PLOT,
            "distance from the truth (mm) by step, grid 50 mm",
        ),
        (
            &CORRECTION_PLOT,
            "how far each step's correction moved the pose (mm), grid 10 mm",
        ),
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
    by_landmarks: Res<LandmarkCorrection>,
    by_walls: Res<WallCorrection>,
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
    let landmark_step = by_landmarks.steps.get(i);
    let pairs = landmark_step.map_or(String::new(), |s| {
        format!(
            " ({} of {} cylinders paired{})",
            s.pairs.len(),
            s.cylinders.len(),
            if s.transform.is_some() {
                ""
            } else {
                ", no fix"
            }
        )
    });
    let text = format!(
        "AV 02 - Unit B: landmarks and walls   {what}\n\
         off by:  dead reckoning {}   landmarks {}{pairs}   walls {}\n{}",
        off_by(&log, i, dr.poses.get(i)),
        off_by(&log, i, landmark_step.map(|s| &s.pose)),
        off_by(&log, i, by_walls.steps.get(i).map(|s| &s.pose)),
        status.0
    );
    if header.0 != text {
        header.0 = text;
    }
}

/// "12.3 mm": the pose's distance from the reference position of step i.
fn off_by(log: &RobotLog, i: usize, pose: Option<&Pose>) -> String {
    match (pose, log.log.reference.get(i)) {
        (Some(p), Some(r)) => format!(
            "{:5.1} mm",
            p.position().distance(Vec2::new(r.x as f32, r.y as f32))
        ),
        _ => "  -  ".into(),
    }
}

fn update_help(
    mode: Res<State<Mode>>,
    layers: Res<Layers>,
    mut help: Single<&mut Text, With<Help>>,
) {
    if !mode.is_changed() && !layers.is_changed() {
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
        "{keys}\nV detail: {:?}   D dead reckoning   E export   Tab replay/live\n\
         green truth   orange dead reckoning   magenta landmarks   blue walls",
        layers.detail
    );
}
