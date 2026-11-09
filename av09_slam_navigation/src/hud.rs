// Part 4: the text. A title line above the arena; right of it, a panel:
// the navigator, how good SLAM is (against the truth, which only the app
// knows), and the keys.

use crate::layout::PANEL_AT;
use crate::{Bumps, SPEEDS, Speed};
use av_sim::core::{AvSet, RobotLog, Status};
use av_sim::source::LiveSim;
use av_sim::unit_f::{SlamMapping, SlamSetup, map_errors};
use av_sim::unit_nav::route::route_length;
use av_sim::unit_nav::{NavSetup, NavStatus, Navigator, SteerBy};
use bevy::prelude::*;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_text)
            .add_systems(Update, (update_title, update_panel).in_set(AvSet::Present));
    }
}

#[derive(Component)]
struct Title;

#[derive(Component)]
struct Panel;

const TEXT: Color = Color::srgb(0.95, 0.92, 0.7);

fn spawn_text(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(16.0),
        TextColor(TEXT),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(10.0),
            ..default()
        },
        Title,
    ));
    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(14.0),
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(PANEL_AT.x),
            top: Val::Px(PANEL_AT.y),
            ..default()
        },
        Panel,
    ));
}

fn set_text(text: &mut Text, new: String) {
    if text.0 != new {
        text.0 = new;
    }
}

fn update_title(
    status: Res<Status>,
    live: Res<LiveSim>,
    speed: Res<Speed>,
    log: Res<RobotLog>,
    mut title: Single<&mut Text, With<Title>>,
) {
    let running = if live.paused { "PAUSED" } else { "running" };
    let seconds = log.len() as f32 * av_sim::synth::STEP_MS as f32 / 1000.0;
    set_text(
        &mut title,
        format!(
            "AV 09 - driving on a SLAM map (Units F + PP)   {}   {running} x{}   {seconds:.0} s",
            status.0, SPEEDS[speed.0]
        ),
    );
}

fn status_text(status: NavStatus) -> String {
    match status {
        NavStatus::Idle => "no goal (click in the arena, or W)".into(),
        NavStatus::Driving => "driving".into(),
        NavStatus::Arrived => "arrived".into(),
        NavStatus::Failed(e) => format!("NO ROUTE: {}", e.text()),
        NavStatus::BackingUp => "BACKING UP: ran into something".into(),
        NavStatus::Stuck => "STUCK: gave the goal up".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn update_panel(
    nav: Res<Navigator>,
    setup: Res<NavSetup>,
    slam_setup: Res<SlamSetup>,
    slam: Res<SlamMapping>,
    log: Res<RobotLog>,
    bumps: Res<Bumps>,
    mut panel: Single<&mut Text, With<Panel>>,
) {
    let mut t = format!("GOAL   {}\n", status_text(nav.status));
    if nav.status == NavStatus::Driving && !nav.route.is_empty() {
        t += &format!(
            "       route {:.0} mm in {} straight legs\n",
            route_length(&nav.route),
            nav.route.len() - 1
        );
    } else {
        t += "\n";
    }
    let last = nav.last_replan.map_or("-".to_string(), |(r, step)| {
        format!("{} (step {step})", r.text())
    });
    t += &format!(
        "       planned {} times, last: {last}\n\
         \x20      goals reached {}, given up {}, backed up {} times\n\
         \x20      the TRUE robot bumped: {} steps\n\n",
        nav.replans, nav.goals_reached, nav.goals_given_up, nav.backups_total, bumps.count
    );

    // How good SLAM is: only the app can say (it has the truth).
    let errors: Vec<f32> = slam
        .steps
        .iter()
        .zip(&log.log.reference)
        .map(|(s, r)| {
            s.pose
                .position()
                .distance(Vec2::new(r.x as f32, r.y as f32))
        })
        .collect();
    let now = errors.last().copied().unwrap_or(0.0);
    let mean = errors.iter().sum::<f32>() / errors.len().max(1) as f32;
    let scanner = if *slam_setup == SlamSetup::sharp_scanner() {
        "the sharp scanner (20 mm, 1 deg)"
    } else {
        "the course's noise (600 mm, 45 deg)"
    };
    t += &format!(
        "SLAM   with {scanner}\n\
         \x20      pose error now {now:.0} mm, mean {mean:.0} mm\n"
    );
    if let Some(step) = slam.steps.last() {
        let truth: Vec<Vec2> = log.arena.landmarks.iter().map(|l| l.position).collect();
        let (map_err, extra) = map_errors(&step.landmarks, &truth);
        let seen = map_err.len().min(step.landmarks.len());
        let map_mean = map_err.iter().sum::<f32>() / map_err.len().max(1) as f32;
        t += &format!(
            "       map {} landmarks ({extra} extra), mean error {map_mean:.0} mm\n",
            step.landmarks.len()
        );
        if seen > 0
            && let Some(map) = &nav.map
        {
            let (lo, hi) = map
                .discs
                .iter()
                .fold((f32::MAX, 0.0f32), |(lo, hi), d| (lo.min(d.1), hi.max(d.1)));
            t += &format!("       discs {lo:.0} to {hi:.0} mm (cap {:.0})\n", {
                let m = &setup.map;
                m.cylinder_radius + m.robot_radius + m.margin + m.sigmas * m.max_sigma
            });
        }
    }
    t += &format!(
        "\nsteering by {}\n\n",
        match setup.steer_by {
            SteerBy::Slam => "SLAM's pose",
            SteerBy::DeadReckoning => "DEAD RECKONING (orange): it drifts",
        }
    );
    t += &format!(
        "click   a goal (wander off)\n\
         W       wander, random goals: {}\n\
         T       sharp scanner / course noise\n\
         D       steer by SLAM / dead reckoning\n\
         V       planning map and ellipses on/off\n\
         Space   pause   + - speed   N new run\n\n\
         grey: real cylinders (not told)\n\
         purple: SLAM's pose and map (95%)\n\
         red: where the robot's centre may not go\n\
         yellow: the route   white: steered at\n\
         green: the true robot (80 mm bumper)",
        if setup.wander { "on" } else { "off" }
    );
    set_text(&mut panel, t);
}
