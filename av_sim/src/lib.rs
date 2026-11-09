// av_sim: the shared library for the autonomous-vehicle series, after
// Claus Brenner's SLAM course (reference code: github.com/jfjensen/SLAM).
// Synthetic data instead of the real LEGO robot's, in the same file format.
//
// Read in this order:
//   Part 1   robot.rs      the robot, its scanner, poses
//   Part 2   logfile.rs    the course's log format (M S P L F D records)
//   Part 3   world.rs      the arena and ray casting
//   Part 4   synth.rs      the synthetic robot, autopilot, a whole run
//            rng.rs        seeded random numbers
//   Part 5   core.rs       RobotLog, Playhead, AvSet (frame order)
//   Part 6   source.rs     Replay and Live modes
//   Unit A   unit_a/       motion model, cylinder detector (+ plugins)
//   Unit B   unit_b/       similarity transform, landmark and wall (ICP)
//                          corrections (+ plugins)
//   Unit C   unit_c/       1D distributions, histogram and Kalman filters,
//                          a 1D track with its own Replay/Live (+ plugins)
//   Unit D   unit_d/       the extended Kalman filter for the pose (+ plugin)
//   Unit E   unit_e/       the particle filter for the pose (+ plugin)
//   Unit F   unit_f/       EKF SLAM: the pose and the map together (+ plugin)
//   Unit G   unit_g/       FastSLAM: particles, each with its own map (+ plugin)
//   Unit PP  unit_pp/      path planning: Dijkstra, A*, potential function,
//                          a car's state space (no plugins)
//   Unit NAV unit_nav/     not in the course: F and PP together, driving
//                          to a goal on the SLAM map (+ plugin)
//
// An app adds AvCorePlugin, DataSourcePlugin, the units' plugins, and its
// own drawing (see av01_motion_and_scan).

pub mod core;
pub mod logfile;
pub mod rng;
pub mod robot;
pub mod source;
pub mod synth;
pub mod unit_a;
pub mod unit_b;
pub mod unit_c;
pub mod unit_d;
pub mod unit_e;
pub mod unit_f;
pub mod unit_g;
pub mod unit_nav;
pub mod unit_pp;
pub mod world;
