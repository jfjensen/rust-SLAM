// Part 5: the text. A title line above the map; right of it, a panel with
// the search's state, the comparison of the three planners (grid page),
// and the keys.

use crate::car_page::CarPage;
use crate::grid_page::GridPage;
use crate::layout::PANEL_AT;
use crate::{Animation, Page, SPEEDS};
use av_sim::core::AvSet;
use av_sim::unit_pp::search::Status;
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

fn status_text(status: Status) -> &'static str {
    match status {
        Status::Searching => "searching",
        Status::Found => "found",
        Status::NoPath => "NO PATH",
    }
}

fn update_title(
    page: Res<Page>,
    grid: Res<GridPage>,
    car: Res<CarPage>,
    anim: Res<Animation>,
    mut title: Single<&mut Text, With<Title>>,
) {
    let what = match *page {
        Page::Grid => format!(
            "GRID page   {}   {}",
            grid.planner.name(),
            status_text(grid.search.status)
        ),
        Page::Car => format!(
            "CAR page   {}   {}",
            if car.backward {
                "forward and backward"
            } else {
                "forward only"
            },
            status_text(car.search.status)
        ),
    };
    let animation = if anim.instant {
        "instant".to_string()
    } else {
        format!(
            "{}, {} per frame",
            if anim.playing { "playing" } else { "paused" },
            SPEEDS[anim.speed]
        )
    };
    let text = format!("AV 08 - Unit PP: path planning   {what}   ({animation})");
    if title.0 != text {
        title.0 = text;
    }
}

fn update_panel(
    page: Res<Page>,
    grid: Res<GridPage>,
    car: Res<CarPage>,
    mut panel: Single<&mut Text, With<Panel>>,
) {
    let text = match *page {
        Page::Grid => {
            let s = &grid.search;
            let path = s.path();
            let mut t = format!(
                "cells expanded   {}\nin the front     {}\n",
                s.expanded,
                s.front_cells().len()
            );
            if !path.is_empty() {
                t += &format!(
                    "path length      {:.1} cells\npath cost        {:.1}\n",
                    av_sim::unit_pp::search::path_length(&path),
                    s.path_cost().unwrap_or(0.0)
                );
            }
            t += "\nsame map, to the end:\n";
            for (planner, expanded, length) in grid.comparison {
                let length = length.map_or("no path".to_string(), |l| format!("length {l:.1}"));
                t += &format!("  {:<15}{expanded:>6} cells, {length}\n", planner.name());
            }
            t += &format!(
                "\nmouse   left paint, right erase\n        Shift+left start, Shift+right goal\n\
                 1 2 3   Dijkstra, A*, A* + potential\n\
                 V       visited cells: {}\n\
                 P       potential (blue, with 3): {}\n",
                on_off(grid.show_visited),
                on_off(grid.show_potential)
            );
            t
        }
        Page::Car => {
            let s = &car.search;
            let mut t = format!("states expanded  {}\n", s.expanded);
            if s.status == Status::Found {
                let path = s.path();
                let back = path.iter().filter(|p| p.length < 0.0).count();
                t += &format!(
                    "path             {} arcs, {:.0} cells\n                 {back} reversing\n",
                    path.len(),
                    s.path_length()
                );
            }
            t += &format!(
                "\nmouse   left paint, right erase\n        Shift+left drag: start and its heading\n\
                 \x20       Shift+right drag: goal and its heading\n\
                 B       reversing: {}\n\
                 V       visited cells: {}\n",
                on_off(car.backward),
                on_off(car.show_visited)
            );
            t
        }
    };
    let text = text
        + "C clear   R random walls   M SLAM arena\n\n\
           Space play/pause   Right one step\n\
           + / - speed   Enter to the end\n\
           I instant (solve after every edit)\n\
           Tab grid <-> car";
    if panel.0 != text {
        panel.0 = text;
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}
