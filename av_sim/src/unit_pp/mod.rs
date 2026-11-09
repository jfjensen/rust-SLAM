// Unit PP of the course: path planning. Not SLAM: the map is known (here:
// painted, random, or the SLAM arena), and the question is how to get
// from a start to a goal.
//
//   grid.rs    the cells, the brush, the arena preset, the potential
//              function (an exact distance transform)
//   search.rs  Dijkstra and A* on the grid (pp_01)
//   car.rs     A* over a car's poses, driving in arcs (pp_02)
//
// No plugins: the searches are plain structs with step() and run(), and
// the app decides when to step them (animation). Nothing here reads the
// robot logs.

pub mod car;
pub mod grid;
pub mod search;

#[cfg(test)]
mod tests {
    use super::car::CarSearch;
    use super::grid::Grid;
    use super::search::{Algorithm, Search, Status, clearance, path_length};
    use bevy::math::DVec3;

    /// Cells expanded and path length per algorithm, over seeds 1-5, and
    /// the car's states expanded with and without the closed set. Run with
    ///   cargo test --release unit_pp::tests::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        for seed in 1..=5 {
            let mut grid = Grid::random(200, 150, 250, seed);
            grid.paint(5, 5, false);
            grid.paint(190, 140, false);
            let mut line = format!("seed {seed}:");
            for (name, algorithm, potential) in [
                ("Dijkstra", Algorithm::Dijkstra, false),
                ("A*", Algorithm::AStar, false),
                ("A*+potential", Algorithm::AStar, true),
            ] {
                let costs = if potential {
                    grid.potential_costs()
                } else {
                    grid.plain_costs()
                };
                let mut s = Search::new(&grid, costs, (5, 5), (190, 140), algorithm);
                s.run();
                let p = s.path();
                // The closest point is set by the narrowest gap on any
                // route; the mean shows whether the path keeps its distance.
                let d = super::grid::distance_transform(&grid.walls, grid.width, grid.height);
                let mean = p.iter().map(|&(x, y)| d[grid.index(x, y)]).sum::<f64>()
                    / p.len().max(1) as f64;
                line += &format!(
                    "  {name} {:5} cells, length {:5.1}, clearance min {:3.1} mean {:4.1} |",
                    s.expanded,
                    path_length(&p),
                    clearance(&p, &grid),
                    mean
                );
            }
            println!("{line}");

            let mut grid = Grid::random(100, 75, 40, seed);
            grid.paint(10, 10, false);
            grid.paint(85, 60, false);
            let start = DVec3::new(10.5, 10.5, 0.0);
            let goal = DVec3::new(85.5, 60.5, 0.0);
            let mut with = CarSearch::new(&grid, start, goal, false);
            let status = with.run();
            let mut without = CarSearch::new(&grid, start, goal, false);
            without.closed_set = false;
            let status_without = without.run();
            println!(
                "        car: closed set {:?} after {} states, path {:.0} | without {:?} after {} states",
                status,
                with.expanded,
                with.path_length(),
                status_without,
                without.expanded
            );
            assert!(status != Status::Searching);
        }
    }
}
