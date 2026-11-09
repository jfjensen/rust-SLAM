// Part 3: the grid page (the course's pp_01). Its state, the mouse and
// keys that edit it, the search stepped a few cells per frame, and the
// drawing: a texture for the cells (30,000 cells are too
// many for gizmos), gizmos for the path, start and goal.
//
//   3a  the page's state and how a search is (re)started
//   3b  editing: mouse and keys
//   3c  stepping the search
//   3d  drawing

use crate::layout::*;
use crate::typed::Typed;
use crate::{Animation, MouseAction, Page, Pointer, mouse_action, shift_held};
use av_sim::core::AvSet;
use av_sim::unit_pp::grid::Grid;
use av_sim::unit_pp::search::{Algorithm, Search, Status};
use av_sim::world::Arena;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

// --- 3a ------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planner {
    Dijkstra,
    AStar,
    /// A* with the potential function's costs (pp_01_f).
    AStarPotential,
}

impl Planner {
    pub const ALL: [Planner; 3] = [Planner::Dijkstra, Planner::AStar, Planner::AStarPotential];

    pub fn name(self) -> &'static str {
        match self {
            Planner::Dijkstra => "Dijkstra",
            Planner::AStar => "A*",
            Planner::AStarPotential => "A* + potential",
        }
    }

    fn algorithm(self) -> Algorithm {
        match self {
            Planner::Dijkstra => Algorithm::Dijkstra,
            _ => Algorithm::AStar,
        }
    }
}

/// Random walls: this many brush boxes.
const RANDOM_BOXES: usize = 250;
/// The robot's half-width, mm: the arena preset grows walls by this much.
const ROBOT_CLEARANCE: f32 = av_sim::robot::ROBOT_WIDTH / 2.0;

#[derive(Resource)]
pub struct GridPage {
    pub grid: Grid,
    pub start: (i32, i32),
    pub goal: (i32, i32),
    pub planner: Planner,
    pub search: Search,
    /// The potential function's costs for the current walls (computed once
    /// per edit; also drawn).
    pub potential: Vec<u8>,
    pub show_visited: bool,
    pub show_potential: bool,
    pub seed: u64,
    /// For the HUD: cells expanded and path length, per planner, on the
    /// current map (each run to the end once per edit).
    pub comparison: [(Planner, usize, Option<f64>); 3],
    pub image: Handle<Image>,
    /// Goes up with every edit: the texture compares it.
    pub version: u64,
    drawn: Option<(u64, usize, bool, bool)>,
}

impl GridPage {
    fn costs(&self) -> Vec<u8> {
        match self.planner {
            Planner::AStarPotential => self.potential.clone(),
            _ => self.grid.plain_costs(),
        }
    }

    /// After any change: new costs, a new search, the comparison.
    pub fn restart(&mut self) {
        self.potential = self.grid.potential_costs();
        self.search = Search::new(
            &self.grid,
            self.costs(),
            self.start,
            self.goal,
            self.planner.algorithm(),
        );
        for (i, planner) in Planner::ALL.into_iter().enumerate() {
            let costs = match planner {
                Planner::AStarPotential => self.potential.clone(),
                _ => self.grid.plain_costs(),
            };
            let mut s = Search::new(
                &self.grid,
                costs,
                self.start,
                self.goal,
                planner.algorithm(),
            );
            s.run();
            let length = (s.status == Status::Found)
                .then(|| av_sim::unit_pp::search::path_length(&s.path()));
            self.comparison[i] = (planner, s.expanded, length);
        }
        self.version += 1;
    }

    /// Random walls, with the start and goal kept free.
    fn randomise(&mut self) {
        let (w, h) = GRID_CELLS;
        self.grid = Grid::random(w, h, RANDOM_BOXES, self.seed);
        self.grid.paint(self.start.0, self.start.1, false);
        self.grid.paint(self.goal.0, self.goal.1, false);
    }

    /// The SLAM arena, with the start and goal moved to two free corners.
    fn load_arena(&mut self) {
        let (w, h) = GRID_CELLS;
        let arena = Arena::course();
        self.grid = Grid::arena(w, h, &arena, ROBOT_CLEARANCE);
        let mm = Grid::arena_cell_mm(w, h, &arena);
        let cell = |x: f32, y: f32| ((x / mm) as i32, (y / mm) as i32);
        self.start = cell(250.0, 250.0);
        self.goal = cell(1750.0, 1750.0);
    }
}

impl FromWorld for GridPage {
    fn from_world(world: &mut World) -> GridPage {
        let (w, h) = GRID_CELLS;
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
        let mut page = GridPage {
            search: Search::new(
                &grid,
                grid.plain_costs(),
                (0, 0),
                (0, 0),
                Algorithm::Dijkstra,
            ),
            potential: grid.plain_costs(),
            grid,
            start: (10, 10),
            goal: (190, 140),
            planner: Planner::Dijkstra,
            show_visited: true,
            show_potential: true,
            seed: 1,
            comparison: Planner::ALL.map(|p| (p, 0, None)),
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
struct GridMap;

pub struct GridPagePlugin;

impl Plugin for GridPagePlugin {
    fn build(&self, app: &mut App) {
        let on_page = |page: Res<Page>| *page == Page::Grid;
        app.init_resource::<GridPage>().add_systems(
            Update,
            (
                (edit_with_mouse, grid_keys).chain().in_set(AvSet::Input),
                step_search.in_set(AvSet::Simulate),
            )
                .run_if(on_page),
        );
    }
}

/// The drawing: the texture on a sprite, and the gizmos. Separate from
/// the logic so the headless tests can leave it out (gizmos need the
/// rendering plugins).
pub struct GridDrawPlugin;

impl Plugin for GridDrawPlugin {
    fn build(&self, app: &mut App) {
        let on_page = |page: Res<Page>| *page == Page::Grid;
        app.add_systems(Startup, spawn_map)
            .add_systems(
                Update,
                (show_map, redraw_texture, draw_overlay)
                    .in_set(AvSet::Present)
                    .run_if(on_page),
            )
            .add_systems(
                Update,
                hide_map.run_if(move |page: Res<Page>| *page != Page::Grid),
            );
    }
}

fn spawn_map(mut commands: Commands, page: Res<GridPage>) {
    commands.spawn((
        Sprite {
            image: page.image.clone(),
            custom_size: Some(MAP_SIZE),
            ..default()
        },
        Transform::from_translation((MAP_MIN + MAP_SIZE / 2.0).extend(0.0)),
        GridMap,
    ));
}

fn show_map(mut map: Query<&mut Visibility, With<GridMap>>) {
    for mut v in &mut map {
        v.set_if_neq(Visibility::Inherited);
    }
}

fn hide_map(mut map: Query<&mut Visibility, With<GridMap>>) {
    for mut v in &mut map {
        v.set_if_neq(Visibility::Hidden);
    }
}

// --- 3b ------------------------------------------------------------------------

fn edit_with_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    pointer: Res<Pointer>,
    mut page: ResMut<GridPage>,
    mut anim: ResMut<Animation>,
) {
    let (w, h) = GRID_CELLS;
    let (Some(action), Some((x, y))) = (
        mouse_action(&buttons, shift_held(&keys)),
        pointer.at.and_then(|p| screen_to_cell(p, w, h)),
    ) else {
        return;
    };
    let changed = match action {
        MouseAction::Paint | MouseAction::Erase => {
            let on = action == MouseAction::Paint;
            let before = page.grid.walls.clone();
            page.grid.paint(x, y, on);
            page.grid.walls != before
        }
        // The grid page has no headings: press or drag both just move it.
        MouseAction::Start { .. } => std::mem::replace(&mut page.start, (x, y)) != (x, y),
        MouseAction::Goal { .. } => std::mem::replace(&mut page.goal, (x, y)) != (x, y),
    };
    if changed {
        page.restart();
        anim.restart();
    }
}

fn grid_keys(
    keys: Res<ButtonInput<KeyCode>>,
    typed: Res<Typed>,
    mut page: ResMut<GridPage>,
    mut anim: ResMut<Animation>,
) {
    let mut changed = false;
    // The digit row by position (the same on every layout).
    for (key, planner) in [
        (KeyCode::Digit1, Planner::Dijkstra),
        (KeyCode::Digit2, Planner::AStar),
        (KeyCode::Digit3, Planner::AStarPotential),
    ] {
        if keys.just_pressed(key) && page.planner != planner {
            page.planner = planner;
            changed = true;
        }
    }
    if typed.pressed('v') {
        page.show_visited = !page.show_visited;
    }
    if typed.pressed('p') {
        page.show_potential = !page.show_potential;
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

// --- 3c ------------------------------------------------------------------------

fn step_search(mut page: ResMut<GridPage>, mut anim: ResMut<Animation>) {
    let n = anim.take_steps();
    let search = &mut page.search;
    for _ in 0..n {
        if search.step() != Status::Searching {
            break;
        }
    }
}

// --- 3d ------------------------------------------------------------------------

fn redraw_texture(mut page: ResMut<GridPage>, mut images: ResMut<Assets<Image>>) {
    let wanted = (
        page.version,
        page.search.expanded,
        page.show_visited,
        page.show_potential,
    );
    if page.drawn == Some(wanted) {
        return;
    }
    let (w, h) = (page.grid.width, page.grid.height);
    let max = page.search.max_visited().max(1e-9);
    let mut front = vec![false; w * h];
    for (x, y) in page.search.front_cells() {
        front[page.grid.index(x, y)] = true;
    }
    let mut bytes = Vec::with_capacity(w * h * 4);
    // Image rows go DOWN from the top; cell y goes UP.
    for row in 0..h {
        let y = h - 1 - row;
        for x in 0..w {
            let i = x + y * w;
            let color = if page.grid.walls[i] {
                WALL_RGBA
            } else if page.show_visited && front[i] {
                FRONT_RGBA
            } else {
                let visited = page
                    .show_visited
                    .then(|| page.search.visited[i].map(|c| (c / max) as f32))
                    .flatten();
                // Blue only where it's used: the third planner.
                let potential = if page.show_potential && page.planner == Planner::AStarPotential {
                    page.potential[i]
                } else {
                    0
                };
                cell_rgba(visited, potential)
            };
            bytes.extend(color);
        }
    }
    if let Some(mut image) = images.get_mut(&page.image) {
        image.data = Some(bytes);
    }
    page.drawn = Some(wanted);
}

fn draw_overlay(page: Res<GridPage>, mut gizmos: Gizmos) {
    let w = page.grid.width;
    gizmos.rect_2d(MAP_MIN + MAP_SIZE / 2.0, MAP_SIZE, FRAME);
    let centre = |(x, y): (i32, i32)| cell_to_screen(Vec2::new(x as f32 + 0.5, y as f32 + 0.5), w);
    let path = page.search.path();
    if !path.is_empty() {
        gizmos.linestrip_2d(path.iter().map(|&c| centre(c)), PATH);
    }
    if page.search.status == Status::Searching
        && let Some(c) = page.search.current
    {
        gizmos.circle_2d(centre(c), 4.0, CURRENT);
    }
    gizmos.circle_2d(centre(page.start), 7.0, START);
    gizmos.circle_2d(centre(page.goal), 7.0, GOAL);
}
