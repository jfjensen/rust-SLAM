// Part 5: the text. A header (mode, step, status), a line with how far
// each path is from the truth, the keys, and the plot captions.

use crate::Layers;
use crate::layout::{COUNT_PLOT, ERROR_PLOT, HELP_AT, Plot};
use av_sim::core::{AvSet, Playhead, RobotLog, Status};
use av_sim::robot::Pose;
use av_sim::source::{LiveSim, Mode, ReplaySource};
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_d::EkfLocalization;
use av_sim::unit_e::{ParticleLocalization, ParticleSetup, Start};
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
            "mm from the truth, grid 10: orange dead reckoning, blue Kalman, pink particles",
        ),
        (
            &COUNT_PLOT,
            "particles that really counted (effective count, % of all), grid 20%",
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
    kalman: Res<EkfLocalization>,
    pf: Res<ParticleLocalization>,
    setup: Res<ParticleSetup>,
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
    let step = pf.steps.get(i);
    let cloud = step.map_or(String::new(), |s| {
        format!(
            "   ellipse {:.0} x {:.0} mm, heading +/- {:.1} deg (1 sigma)\n\
             {} cylinders seen, {:.0} of {} particles really counted{}",
            s.ellipse.major,
            s.ellipse.minor,
            s.heading_stddev.to_degrees(),
            s.sightings,
            s.effective,
            s.particles.len(),
            if s.updated {
                ""
            } else {
                "   (robot stood still: no update)"
            }
        )
    });
    let start = match setup.start {
        Start::Known => "start known",
        Start::Anywhere => "start ANYWHERE",
    };
    let text = format!(
        "AV 05 - Unit E: particle filter   {what}   {} particles, {start}, seed {}\n\
         off by:  dead reckoning {}   Kalman {}   particles {}{cloud}\n{}",
        setup.count,
        setup.seed,
        off_by(&log, i, dr.poses.get(i)),
        off_by(&log, i, kalman.steps.get(i).map(|s| &s.pose)),
        off_by(&log, i, step.map(|s| &s.pose)),
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
            "{:6.1} mm",
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
        "{keys}\nV particles: {:?}   D dead reckoning   K Kalman   E export\n\
         U start known/anywhere   N count   S seed   Tab replay/live",
        layers.detail
    );
}
