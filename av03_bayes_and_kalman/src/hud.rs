// Part 5: the text. A header (mode, step, what happened this step, each
// filter's estimate), the keys, and the plot captions.

use crate::Layers;
use crate::layout::{BELIEF_PLOT, ERROR_PLOT, HELP_AT, Plot};
use av_sim::core::{AvSet, Playhead, Status};
use av_sim::source::Mode;
use av_sim::unit_c::source::{LiveTrack, TrackFile};
use av_sim::unit_c::{HistogramFilter, HistogramParams, KalmanFilter, Track};
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
            &BELIEF_PLOT,
            "belief: probability per cm. grey before, blue prediction, green measurement, red correction. \
             light = histogram, full = Kalman",
        ),
        (
            &ERROR_PLOT,
            "estimate - truth (cm) by step, grid 10 cm. grey: Kalman 2 sigma, yellow: sensor, shaded: no sensor",
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
            top: Val::Px(at.y - 18.0),
            ..default()
        },
    ));
}

#[allow(clippy::too_many_arguments)]
fn update_header(
    mode: Res<State<Mode>>,
    file: Res<TrackFile>,
    live: Res<LiveTrack>,
    track: Res<Track>,
    playhead: Res<Playhead>,
    hist: Res<HistogramFilter>,
    params: Res<HistogramParams>,
    kalman: Res<KalmanFilter>,
    status: Res<Status>,
    mut header: Single<&mut Text, With<Header>>,
) {
    let i = playhead.index;
    let what = match mode.get() {
        Mode::Replay => format!(
            "REPLAY {}   step {}/{}{}",
            file.path,
            i + 1,
            track.len(),
            if playhead.playing { "" } else { "   (paused)" }
        ),
        Mode::Live => format!(
            "LIVE   step {}   {}   sensor {}{}",
            i + 1,
            if live.autopilot {
                "autopilot"
            } else {
                "arrow keys"
            },
            if live.measuring { "on" } else { "OFF" },
            if live.paused { "   (paused)" } else { "" },
        ),
    };
    let this_step = match track.log.steps.get(i) {
        Some(s) => format!(
            "told to move {:+.0} cm, sensor says {}, truly at {:.1} cm",
            s.control,
            s.measured
                .map_or("nothing".into(), |z| format!("{z:.1} cm")),
            s.truth
        ),
        None => "no steps yet".into(),
    };
    let truth = track.log.steps.get(i).map(|s| s.truth);
    let estimate = |mean: f64, sigma: f64| {
        let off = truth.map_or(String::new(), |t| format!(", off by {:+.1}", mean - t));
        format!("{mean:6.1} cm, sigma {sigma:4.1}{off}")
    };
    let histogram = hist.steps.get(i).map_or("-".into(), |h| {
        let c = &h.correction;
        format!(
            "{}{}",
            estimate(c.mean(), c.variance().sqrt()),
            if h.disagreed {
                "  MEASUREMENT IGNORED: no overlap"
            } else {
                ""
            }
        )
    });
    let kalman = kalman.steps.get(i).map_or("-".into(), |k| {
        let gain = if k.measurement.is_some() {
            format!(", gain {:.2}", k.gain)
        } else {
            String::new()
        };
        format!("{}{gain}", estimate(k.correction.mu, k.correction.sigma()))
    });
    let text = format!(
        "AV 03 - Unit C: the Bayes filter in 1D   {what}\n\
         this step: {this_step}\n\
         histogram ({:?}s): {histogram}\n\
         Kalman: {kalman}\n{}",
        params.shape, status.0
    );
    if header.0 != text {
        header.0 = text;
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
            "Space play/pause\n\
             Left/Right step (Shift: 10)\n\
             Home/End   L track <-> live run"
        }
        Mode::Live => {
            "Space pause   A autopilot\n\
             Left/Right hold to drive\n\
             M sensor on/off   R save run"
        }
    };
    help.0 = format!(
        "{keys}\nV curves: {:?}\nG gaussians/triangles\nB before: {}\nTab replay/live",
        layers.view,
        if layers.before { "on" } else { "off" }
    );
}
