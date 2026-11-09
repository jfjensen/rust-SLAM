# rust-SLAM

The algorithms from Claus Brenner's SLAM course, written in Rust, with a [Bevy](https://bevyengine.org) app for each unit so you can watch them work.

The original course code is in Python, at [github.com/jfjensen/SLAM](https://github.com/jfjensen/SLAM). This repo does the same things on a synthetic robot: a simulated two-wheeled robot with a LiDAR drives around an arena with six cylinders, and writes its log files in the course's format. Because the simulator knows where the robot really is, every app can show the true path next to the estimated ones, and tell you how far off each estimate is.

## What is in here

| Folder | Unit | What it does |
|---|---|---|
| `av_sim` | | The library: the log file format, the arena, the simulated robot and LiDAR, and the algorithms, one module and one Bevy plugin per unit |
| `av01_motion_and_scan` | A | Motion model (dead reckoning from the wheel ticks) and cylinder detection in the LiDAR scan |
| `av02_landmarks_and_icp` | B | Correcting the pose with the landmarks (similarity transform), and with the walls (ICP) |
| `av03_bayes_and_kalman` | C | The Bayes filter in 1D: the histogram filter and the Kalman filter |
| `av04_extended_kalman` | D | Localisation with the extended Kalman filter |
| `av05_particle_filter` | E | Localisation with a particle filter, also from an unknown start |
| `av06_ekf_slam` | F | EKF SLAM: localising and building the map at the same time |
| `av07_fastslam` | G | FastSLAM: a particle filter where each particle has its own map |
| `av08_path_planning` | PP | Dijkstra, A*, A* with a potential function, and A* for a car that can't turn on the spot |
| `av09_slam_navigation` | F + PP | Not from the course: the robot drives to a goal, planning on its own EKF SLAM map |

## Running it

You need Rust (edition 2024, so 1.85 or newer). Run each app from its own folder, because the apps read and write their log files in `data/` relative to where they run:

```bash
cd av01_motion_and_scan
cargo run
```

The first build compiles Bevy, which takes a while. After that, rebuilds are quick.

The apps av01 to av07 take a few commands:

```bash
cargo run                         # replay data/generated_*.txt
cargo run -- live                 # drive the robot live
cargo run -- replay data/live     # replay a saved live run
cargo run -- generate [seed] [steps]   # write new generated_*.txt, no window
```

av03 uses `data/track.txt` instead, and av08 and av09 are always live. The comment at the top of each app's `main.rs` explains what is on the screen and which keys do what.

## Tests

```bash
cargo test --workspace
```

Each unit also has a few ignored "measure" tests, which run the algorithms over many seeds and print how well they do. For example:

```bash
cd av_sim
cargo test --release unit_f::tests::measure -- --ignored --nocapture
```

## Build notes

- The `dev` feature (on by default) switches on Bevy's dynamic linking, for fast rebuilds. For a stand-alone executable, build with `cargo build --release --no-default-features`.
- `.cargo/config.toml` uses the LLD linker on Windows, as the Bevy setup guide suggests. On other systems it is not used.

## Credits

The algorithms, the log file format and the arena follow Claus Brenner's SLAM course. The Python version is at [github.com/jfjensen/SLAM](https://github.com/jfjensen/SLAM).

## License

MIT, see [LICENSE](LICENSE).
