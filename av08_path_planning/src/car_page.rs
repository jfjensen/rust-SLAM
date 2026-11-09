// Part 4: the car page (the course's pp_02). The same shape as the grid
// page (grid_page.rs), but the start and goal are POSES, and the search
// runs over (x, y, heading).
//
//   4a  the page's state and how a search is (re)started
//   4b  editing: mouse and keys (Shift + drag points the start or goal)
//   4c  stepping the search
//   4d  drawing: cells in a texture; the arcs, the car and the poses
//       with gizmos

use crate::layout::*;
use crate::typed::Typed;
use crate::{Animation, MouseAction, Page, Pointer, mouse_action, shift_held};
use av_sim::core::AvSet;
use av_sim::unit_pp::car::{CarSearch, Segment, arc_points};
use av_sim::unit_pp::grid::Grid;
use av_sim::unit_pp::search::Status;
use av_sim::world::Arena;
use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

// --- 4a ------------------------------------------------------------------------

/// Random walls: this many brush boxes (fewer than the grid page: the
/// cells are twice as big, and the car needs room to turn).
const RANDOM_BOXES: usize = 40;
const ROBOT_CLEARANCE: f32 = av_sim::robot::ROBOT_WIDTH / 2.0;
/// The car marker drawn along the path: length and width in cells.
/// NOTE: only a marker. The planner treats the car as a POINT (as the
/// course does): a car body drawn to scale would overlap the walls it
/// passes close to. The arena preset (M) is the honest case: there the
/// walls are grown by the robot's half-width.
const CAR_SIZE: Vec2 = Vec2::new(2.0, 1.0);

#[derive(Resource)]
pub struct CarPage {
    pub grid: Grid,
    pub start: DVec3,
    pub goal: DVec3,
    /// Allow reversing (the course's Forward / FW+BW button).
    pub backward: bool,
    pub search: CarSearch,
    pub show_visited: bool,
    pub seed: u64,
    pub image: Handle<Image>,
    pub version: u64,
    drawn: Option<(u64, usize, bool)>,
}

impl CarPage {
    pub fn restart(&mut self) {
        self.search = CarSearch::new(&self.grid, self.start, self.goal, self.backward);
        self.version += 1;
    }

    fn randomise(&mut self) {
        let (w, h) = CAR_CELLS;
        self.grid = Grid::random(w, h, RANDOM_BOXES, self.seed);
        // Room around the start and goal.
        for p in [self.start, self.goal] {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    self.grid
                        .paint(p.x as i32 + 4 * dx, p.y as i32 + 4 * dy, false);
                }
            }
        }
    }

    fn load_arena(&mut self) {
        let (w, h) = CAR_CELLS;
        let arena = Arena::course();
        self.grid = Grid::arena(w, h, &arena, ROBOT_CLEARANCE);
        let mm = Grid::arena_cell_mm(w, h, &arena) as f64;
        self.start = DVec3::new(250.0 / mm, 250.0 / mm, 0.0);
        self.goal = DVec3::new(1750.0 / mm, 1750.0 / mm, 0.0);
    }
}

impl FromWorld for CarPage {
    fn from_world(world: &mut World) -> CarPage {
        let (w, h) = CAR_CELLS;
        let image = Image::new_fill(
            Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &FREE_RGBA,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let grid = Grid::new(w, h);
        let start = DVec3::new(10.5, 10.5, 0.0);
        let goal = DVec3::new(85.5, 60.5, 0.0);
        let mut page = CarPage {
            search: CarSearch::new(&grid, start, goal, false),
            grid,
            start,
            goal,
            backward: false,
            show_visited: true,
            seed: 1,
            image: world.resource_mut::<Assets<Image>>().add(image),
            version: 0,
            drawn: None,
        };
        page.randomise();
        page.restart();
        page
    }
}

#[derive(Component)]
struct CarMap;

pub struct CarPagePlugin;

impl Plugin for CarPagePlugin {
    fn build(&self, app: &mut App) {
        let on_page = |page: Res<Page>| *page == Page::Car;
        app.init_resource::<CarPage>().add_systems(
            Update,
            (
                (edit_with_mouse, car_keys).chain().in_set(AvSet::Input),
                step_search.in_set(AvSet::Simulate),
            )
                .run_if(on_page),
        );
    }
}

/// The drawing: the texture on a sprite, and the gizmos. Separate from
/// the logic so the headless tests can leave it out (gizmos need the
/// rendering plugins).
pub struct CarDrawPlugin;

impl Plugin for CarDrawPlugin {
    fn build(&self, app: &mut App) {
        let on_page = |page: Res<Page>| *page == Page::Car;
        app.add_systems(Startup, spawn_map)
            .add_systems(
                Update,
                (show_map, redraw_texture, draw_overlay)
                    .in_set(AvSet::Present)
                    .run_if(on_page),
            )
            .add_systems(
                Update,
                hide_map.run_if(move |page: Res<Page>| *page != Page::Car),
            );
    }
}

fn spawn_map(mut commands: Commands, page: Res<CarPage>) {
    commands.spawn((
        Sprite {
            image: page.image.clone(),
            custom_size: Some(MAP_SIZE),
            ..default()
        },
        Transform::from_translation((MAP_MIN + MAP_SIZE / 2.0).extend(0.0)),
        Visibility::Hidden,
        CarMap,
    ));
}

fn show_map(mut map: Query<&mut Visibility, With<CarMap>>) {
    for mut v in &mut map {
        v.set_if_neq(Visibility::Inherited);
    }
}

fn hide_map(mut map: Query<&mut Visibility, With<CarMap>>) {
    for mut v in &mut map {
        v.set_if_neq(Visibility::Hidden);
    }
}

// --- 4b ------------------------------------------------------------------------

/// A drag shorter than this (cells) doesn't change the heading: the first
/// pixel of a press has no direction yet.
const MIN_DRAG: f64 = 0.5;

fn edit_with_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    pointer: Res<Pointer>,
    mut page: ResMut<CarPage>,
    mut anim: ResMut<Animation>,
) {
    let (w, h) = CAR_CELLS;
    let (Some(action), Some(at)) = (mouse_action(&buttons, shift_held(&keys)), pointer.at) else {
        return;
    };
    let units = screen_to_cell_units(at, w);
    let at = DVec3::new(units.x as f64, units.y as f64, 0.0);
    // Press: move the pose here, keep its heading. Drag: point it here.
    let place_or_point = |pose: &mut DVec3, pressed: bool| {
        let old = *pose;
        if pressed {
            *pose = DVec3::new(at.x, at.y, pose.z);
        } else if at.truncate().distance(pose.truncate()) > MIN_DRAG {
            let d = at.truncate() - pose.truncate();
            pose.z = d.y.atan2(d.x);
        }
        *pose != old
    };
    let changed = match action {
        MouseAction::Paint | MouseAction::Erase => {
            match screen_to_cell(pointer.at.unwrap_or_default(), w, h) {
                Some((x, y)) => {
                    let before = page.grid.walls.clone();
                    page.grid.paint(x, y, action == MouseAction::Paint);
                    page.grid.walls != before
                }
                None => false,
            }
        }
        MouseAction::Start { pressed } => place_or_point(&mut page.start, pressed),
        MouseAction::Goal { pressed } => place_or_point(&mut page.goal, pressed),
    };
    if changed {
        page.restart();
        anim.restart();
    }
}

fn car_keys(typed: Res<Typed>, mut page: ResMut<CarPage>, mut anim: ResMut<Animation>) {
    let mut changed = false;
    if typed.pressed('b') {
        page.backward = !page.backward;
        changed = true;
    }
    if typed.pressed('v') {
        page.show_visited = !page.show_visited;
    }
    if typed.pressed('c') {
        page.grid.clear();
        changed = true;
    }
    if typed.pressed('r') {
        page.seed += 1;
        page.randomise();
        changed = true;
    }
    if typed.pressed('m') {
        page.load_arena();
        changed = true;
    }
    if changed {
        page.restart();
        anim.restart();
    }
}

// --- 4c ------------------------------------------------------------------------

fn step_search(mut page: ResMut<CarPage>, mut anim: ResMut<Animation>) {
    let n = anim.take_steps();
    let search = &mut page.search;
    for _ in 0..n {
        if search.step() != Status::Searching {
            break;
        }
    }
}

// --- 4d ------------------------------------------------------------------------

fn redraw_texture(mut page: ResMut<CarPage>, mut images: ResMut<Assets<Image>>) {
    let wanted = (page.version, page.search.expanded, page.show_visited);
    if page.drawn == Some(wanted) {
        return;
    }
    let (w, h) = (page.grid.width, page.grid.height);
    let max = page.search.max_visited().max(1e-9);
    let mut bytes = Vec::with_capacity(w * h * 4);
    for row in 0..h {
        let y = h - 1 - row;
        for x in 0..w {
            let i = x + y * w;
            let color = if page.grid.walls[i] {
                WALL_RGBA
            } else {
                let visited = page
                    .show_visited
                    .then(|| page.search.visited[i].map(|c| (c / max) as f32))
                    .flatten();
                cell_rgba(visited, 0)
            };
            bytes.extend(color);
        }
    }
    if let Some(mut image) = images.get_mut(&page.image) {
        image.data = Some(bytes);
    }
    page.drawn = Some(wanted);
}

/// A pose as an arrow: a circle at the position, a line along the heading.
fn draw_pose(gizmos: &mut Gizmos, pose: DVec3, width: usize, colour: Color) {
    let at = cell_to_screen(Vec2::new(pose.x as f32, pose.y as f32), width);
    let dir = Vec2::from_angle(pose.z as f32);
    gizmos.circle_2d(at, 6.0, colour);
    gizmos.line_2d(at, at + dir * 24.0, colour);
}

/// The car's outline, centred on a pose.
fn draw_car(gizmos: &mut Gizmos, pose: DVec3, width: usize, colour: Color) {
    let at = cell_to_screen(Vec2::new(pose.x as f32, pose.y as f32), width);
    gizmos.rect_2d(
        Isometry2d::new(at, Rot2::radians(pose.z as f32)),
        CAR_SIZE * cell_px(width),
        colour,
    );
}

fn draw_overlay(page: Res<CarPage>, mut gizmos: Gizmos) {
    let w = page.grid.width;
    gizmos.rect_2d(MAP_MIN + MAP_SIZE / 2.0, MAP_SIZE, FRAME);
    let path: Vec<Segment> = page.search.path();
    let points = path
        .iter()
        .flat_map(|s| arc_points(s.start, s.curvature, s.length, 0.5))
        .map(|p| cell_to_screen(Vec2::new(p.x as f32, p.y as f32), w));
    gizmos.linestrip_2d(points, PATH);
    // The car every other arc; reversing arcs in the goal's colour.
    for (i, s) in path.iter().enumerate() {
        if i % 2 == 0 {
            let colour = if s.length < 0.0 {
                GOAL
            } else {
                PATH.with_alpha(0.5)
            };
            draw_car(&mut gizmos, s.start, w, colour);
        }
    }
    if page.search.status == Status::Searching
        && let Some(p) = page.search.current
    {
        draw_pose(&mut gizmos, p, w, CURRENT);
    }
    draw_pose(&mut gizmos, page.start, w, START);
    draw_pose(&mut gizmos, page.goal, w, GOAL);
}
