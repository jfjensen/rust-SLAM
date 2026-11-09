// Part 3: drawing. Only reads: the log, SLAM, the navigator.
//
//   3a  the planning map: a texture (100 x 100 cells; as in av08), red
//       where the robot's centre may not go, under everything else
//   3b  the arena, the real cylinders, SLAM's map with its ellipses
//       (not the big ones: gizmos aren't clipped to the arena)
//   3c  the route, the pursuit point, the goal, the robots and paths

use crate::View;
use crate::layout::*;
use av_sim::core::{AvSet, RobotLog};
use av_sim::robot::Pose;
use av_sim::source::LiveSim;
use av_sim::unit_a::motion::DeadReckoning;
use av_sim::unit_d::ekf::ErrorEllipse;
use av_sim::unit_f::SlamMapping;
use av_sim::unit_nav::{NavSetup, Navigator, SteerBy};
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_map_sprite).add_systems(
            Update,
            (redraw_map, draw_arena, draw_navigation).in_set(AvSet::Present),
        );
    }
}

const ROBOT_SCREEN_RADIUS: f32 = 9.0;
const CROSS: f32 = 6.0;

// --- 3a ------------------------------------------------------------------------

/// The sprite showing the planning map, and what it shows now.
#[derive(Component, Default)]
struct MapSprite {
    drawn: Option<(Vec<bool>, View)>,
}

fn spawn_map_sprite(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &FREE_RGBA,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    commands.spawn((
        Sprite {
            image: images.add(image),
            custom_size: Some(Vec2::splat(ARENA_PX)),
            ..default()
        },
        Transform::from_translation((ARENA_ORIGIN + Vec2::splat(ARENA_PX / 2.0)).extend(-1.0)),
        MapSprite::default(),
    ));
}

/// A new texture when the blocked cells change (not every frame).
fn redraw_map(
    nav: Res<Navigator>,
    view: Res<View>,
    sprite: Single<(&Sprite, &mut MapSprite)>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(map) = &nav.map else {
        return;
    };
    let (sprite, mut drawn) = sprite.into_inner();
    let grid = &map.grid;
    if drawn
        .drawn
        .as_ref()
        .is_some_and(|(walls, v)| *walls == grid.walls && *v == *view)
    {
        return;
    }
    let (w, h) = (grid.width, grid.height);
    let mut bytes = Vec::with_capacity(w * h * 4);
    // Image rows go DOWN from the top; cell y goes UP.
    for row in 0..h {
        let y = h - 1 - row;
        for x in 0..w {
            let on = *view == View::All && grid.walls[x + y * w];
            bytes.extend(if on { BLOCKED_RGBA } else { FREE_RGBA });
        }
    }
    let image = Image::new(
        Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    if let Some(mut slot) = images.get_mut(&sprite.image) {
        *slot = image;
    }
    drawn.drawn = Some((grid.walls.clone(), *view));
}

// --- 3b ------------------------------------------------------------------------

fn ellipse(gizmos: &mut Gizmos, centre: Vec2, e: &ErrorEllipse, colour: Color) {
    let half = Vec2::new(e.major as f32, e.minor as f32) * ELLIPSE_SIGMAS * ARENA_SCALE;
    gizmos.ellipse_2d(
        Isometry2d::new(world_to_screen(centre), Rot2::radians(e.angle as f32)),
        half,
        colour,
    );
}

fn cross(gizmos: &mut Gizmos, p: Vec2, colour: Color) {
    let c = world_to_screen(p);
    gizmos.line_2d(c - Vec2::splat(CROSS), c + Vec2::splat(CROSS), colour);
    gizmos.line_2d(
        c + Vec2::new(-CROSS, CROSS),
        c + Vec2::new(CROSS, -CROSS),
        colour,
    );
}

fn draw_arena(log: Res<RobotLog>, slam: Res<SlamMapping>, view: Res<View>, mut gizmos: Gizmos) {
    let arena = &log.arena;
    gizmos.rect_2d(
        world_to_screen(arena.size / 2.0),
        arena.size * ARENA_SCALE,
        FRAME,
    );
    for l in &arena.landmarks {
        gizmos.circle_2d(
            world_to_screen(l.position),
            l.diameter / 2.0 * ARENA_SCALE,
            LANDMARK,
        );
    }
    let Some(step) = slam.steps.last() else {
        return;
    };
    for (p, e) in step.landmarks.iter().zip(&step.landmark_ellipses) {
        cross(&mut gizmos, *p, SLAM);
        if *view == View::All && e.major <= MAX_DRAWN_SIGMA {
            ellipse(&mut gizmos, *p, e, SLAM);
        }
    }
}

// --- 3c ------------------------------------------------------------------------

/// A triangle pointing along the heading.
fn draw_robot(gizmos: &mut Gizmos, pose: &Pose, colour: Color) {
    let c = world_to_screen(pose.position());
    let f = pose.forward();
    let side = f.perp();
    let r = ROBOT_SCREEN_RADIUS;
    let nose = c + f * r * 1.4;
    let left = c - f * r + side * r * 0.8;
    let right = c - f * r - side * r * 0.8;
    gizmos.linestrip_2d([nose, left, right, nose], colour);
}

#[allow(clippy::too_many_arguments)]
fn draw_navigation(
    log: Res<RobotLog>,
    slam: Res<SlamMapping>,
    dr: Res<DeadReckoning>,
    nav: Res<Navigator>,
    setup: Res<NavSetup>,
    view: Res<View>,
    live: Res<LiveSim>,
    mut gizmos: Gizmos,
) {
    // The route from where the robot is now, the point it steers at, the goal.
    if let (Some(centre), false) = (nav.centre, nav.route.is_empty()) {
        let rest = nav.route.iter().skip(nav.progress + 1);
        let points = std::iter::once(centre.position()).chain(rest.copied());
        gizmos.linestrip_2d(points.map(world_to_screen), ROUTE);
        if *view == View::All {
            for p in &nav.route {
                gizmos.circle_2d(world_to_screen(*p), 2.5, ROUTE);
            }
        }
    }
    if let Some(steer) = nav.steer {
        gizmos.circle_2d(world_to_screen(steer.target), 4.0, TARGET);
    }
    if let Some(goal) = nav.goal {
        let g = world_to_screen(goal);
        gizmos.circle_2d(g, 10.0, GOAL);
        gizmos.circle_2d(g, 4.0, GOAL);
    }

    // The last minute of each path, and the robots.
    let n = slam.steps.len();
    let from = n.saturating_sub(TRAIL_STEPS);
    let slam_poses: Vec<Vec2> = slam.steps[from..]
        .iter()
        .map(|s| world_to_screen(s.pose.position()))
        .collect();
    gizmos.linestrip_2d(slam_poses, SLAM);
    if let Some(s) = slam.steps.last() {
        draw_robot(&mut gizmos, &s.pose, SLAM);
        if *view == View::All && s.ellipse.major <= MAX_DRAWN_SIGMA {
            ellipse(&mut gizmos, s.pose.position(), &s.ellipse, SLAM);
        }
    }
    if setup.steer_by == SteerBy::DeadReckoning {
        let poses = &dr.poses[dr.poses.len().saturating_sub(TRAIL_STEPS)..];
        gizmos.linestrip_2d(poses.iter().map(|p| world_to_screen(p.position())), BELIEF);
        if let Some(p) = poses.last() {
            draw_robot(&mut gizmos, p, BELIEF);
        }
    }
    // The truth last, on top.
    let reference = &log.log.reference;
    let truth = reference[reference.len().saturating_sub(TRAIL_STEPS)..]
        .iter()
        .map(|r| world_to_screen(Vec2::new(r.x as f32, r.y as f32)));
    gizmos.linestrip_2d(truth, TRUTH);
    // The robot's real size: 80 mm round the axle centre (the
    // simulator's bumper).
    gizmos.circle_2d(
        world_to_screen(live.sim.centre().position()),
        av_sim::synth::ROBOT_RADIUS * ARENA_SCALE,
        TRUTH,
    );
}
