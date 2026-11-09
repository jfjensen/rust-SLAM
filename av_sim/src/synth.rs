// Part 4: the synthetic robot. It drives through the arena and writes what
// the course's real robot wrote: encoder ticks, LiDAR scans and a reference
// (true) position.
//
// Why dead reckoning drifts, here as on the real robot:
//
//   1. Calibration. The "true" ticks-to-mm and wheel gauge are a little
//      off from the course's 0.349 and 150 that the motion model uses.
//   2. Slip. Each step the wheels move the robot a little more or less
//      than the encoders counted.
//
// The true motion is integrated in small sub-steps (not with the course's
// arc formula), so the motion model in unit_a is checked against an
// independent calculation, not against itself.

use crate::logfile::{LogFile, MotorRecord, ReferenceRecord, ScanRecord};
use crate::rng::Rng;
use crate::robot::{BEAM_COUNT, Pose, SCANNER_DISPLACEMENT, START_POSE, beam_index_to_angle};
use crate::world::Arena;
use bevy::math::Vec2;

/// Time between records, like the course's log (about 0.2 to 0.3 s).
pub const STEP_MS: u64 = 200;
/// Steps in a generated run: the course's log has 278.
pub const DEFAULT_STEPS: usize = 278;
/// The robot is round for collisions, this big.
pub const ROBOT_RADIUS: f32 = 80.0;
/// Top wheel speed, mm/s.
pub const MAX_SPEED: f32 = 160.0;

const SUB_STEPS: usize = 20;

/// How the synthetic world differs from the robot's model of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthConfig {
    /// The true mm per tick (the model uses robot::TICKS_TO_MM).
    pub true_ticks_to_mm: f32,
    /// The true wheel gauge (the model uses robot::ROBOT_WIDTH).
    pub true_width: f32,
    /// Slip: relative noise on each wheel's travel per step.
    pub slip: f64,
    /// LiDAR range noise, mm.
    pub range_noise: f64,
    /// Chance that a beam returns no echo (written as 0).
    pub dropout: f64,
}

impl SynthConfig {
    /// A realistic robot: calibration a little off, some slip and noise.
    pub const REALISTIC: SynthConfig = SynthConfig {
        true_ticks_to_mm: 0.352,
        true_width: 154.0,
        slip: 0.02,
        range_noise: 4.0,
        dropout: 0.005,
    };

    /// A perfect robot: matches the model exactly, no noise.
    pub const PERFECT: SynthConfig = SynthConfig {
        true_ticks_to_mm: crate::robot::TICKS_TO_MM,
        true_width: crate::robot::ROBOT_WIDTH,
        slip: 0.0,
        range_noise: 0.0,
        dropout: 0.0,
    };
}

/// One step's worth of records.
#[derive(Clone, Debug)]
pub struct Records {
    pub motor: MotorRecord,
    pub scan: ScanRecord,
    pub reference: ReferenceRecord,
}

// --- Part 4a: the simulator ----------------------------------------------------

#[derive(Clone, Debug)]
pub struct Simulator {
    pub config: SynthConfig,
    pub arena: Arena,
    rng: Rng,
    /// The true pose of the axle centre.
    centre: Pose,
    /// Encoder counts, kept as fractions; the log gets whole ticks.
    ticks: (f64, f64),
    time_ms: u64,
    /// Did the last step stop at an obstacle?
    pub bumped: bool,
}

impl Simulator {
    /// Start where the course's robot starts. The encoders start at the
    /// course's first values, so the numbers look familiar.
    pub fn new(config: SynthConfig, arena: Arena, seed: u64) -> Simulator {
        Simulator {
            config,
            arena,
            rng: Rng::new(seed),
            centre: START_POSE.shifted(-SCANNER_DISPLACEMENT),
            ticks: (20795.0, 16067.0),
            time_ms: 0,
            bumped: false,
        }
    }

    /// The true scanner pose.
    pub fn scanner_pose(&self) -> Pose {
        self.centre.shifted(SCANNER_DISPLACEMENT)
    }

    /// The records for the robot as it is now.
    pub fn records(&mut self) -> Records {
        let scanner = self.scanner_pose();
        Records {
            motor: MotorRecord {
                time_ms: self.time_ms,
                left: self.ticks.0.floor() as i64,
                right: self.ticks.1.floor() as i64,
            },
            scan: ScanRecord {
                time_ms: self.time_ms,
                ranges: self.scan(scanner),
            },
            reference: ReferenceRecord {
                time_ms: self.time_ms,
                x: scanner.x.round() as i32,
                y: scanner.y.round() as i32,
            },
        }
    }

    /// One LiDAR sweep from this scanner pose, in mm. 0 is "no echo".
    fn scan(&mut self, scanner: Pose) -> Vec<u32> {
        (0..BEAM_COUNT)
            .map(|i| {
                let dir = Vec2::from_angle(scanner.heading + beam_index_to_angle(i as f32));
                let d = self.arena.raycast(scanner.position(), dir) as f64
                    + self.rng.gaussian(self.config.range_noise);
                if self.rng.chance(self.config.dropout) || !d.is_finite() {
                    0
                } else {
                    d.round().max(0.0) as u32
                }
            })
            .collect()
    }

    /// Drive for one step with these wheel speeds (mm/s, as the robot
    /// believes them). If the robot would hit something it stays where it
    /// is, and the encoders don't count (a bumper stops the motors).
    pub fn advance(&mut self, left_speed: f32, right_speed: f32) {
        let dt = STEP_MS as f64 / 1000.0;
        let c = self.config;
        // What the encoders count.
        let enc_l = left_speed as f64 * dt;
        let enc_r = right_speed as f64 * dt;
        // What the wheels really do: the encoder's mm are true_ticks_to_mm
        // per tick, not 0.349, and each wheel slips a little.
        let model = crate::robot::TICKS_TO_MM as f64;
        let ticks_l = enc_l / model;
        let ticks_r = enc_r / model;
        let true_l = ticks_l * c.true_ticks_to_mm as f64 * (1.0 + self.rng.gaussian(c.slip));
        let true_r = ticks_r * c.true_ticks_to_mm as f64 * (1.0 + self.rng.gaussian(c.slip));
        let moved = drive(self.centre, true_l as f32, true_r as f32, c.true_width);
        self.time_ms += STEP_MS;
        self.bumped = self.arena.collides(moved.position(), ROBOT_RADIUS);
        if !self.bumped {
            self.centre = moved;
            self.ticks.0 += ticks_l;
            self.ticks.1 += ticks_r;
        }
    }
}

/// The truth: integrate the wheel motion in small sub-steps (unicycle
/// model: forward speed is the mean of the wheels, turn rate their
/// difference over the gauge). Not the course's closed-form arc.
pub fn drive(centre: Pose, left: f32, right: f32, width: f32) -> Pose {
    let mut p = centre;
    let n = SUB_STEPS as f32;
    for _ in 0..SUB_STEPS {
        let forward = (left + right) / 2.0 / n;
        let turn = (right - left) / width / n;
        // Midpoint heading: second-order accurate.
        let mid = p.heading + turn / 2.0;
        p.x += forward * mid.cos();
        p.y += forward * mid.sin();
        p.heading += turn;
    }
    p.heading = crate::robot::wrap_angle(p.heading);
    p
}

// --- Part 4b: the autopilot ----------------------------------------------------
//
// Drives a loop of waypoints around the cylinders: turn towards the next
// one, slow down while the heading is far off, take the next waypoint when
// close. Uses the TRUE pose: it's part of the world, not of the robot's
// algorithms (which only see the log).

pub const COURSE_WAYPOINTS: [(f32, f32); 6] = [
    (1500.0, 1600.0),
    (900.0, 1150.0),
    (750.0, 400.0),
    (1450.0, 420.0),
    (1480.0, 1350.0),
    (850.0, 1750.0),
];
const WAYPOINT_REACHED: f32 = 120.0;
const TURN_GAIN: f32 = 1.5;
const MAX_TURN_RATE: f32 = 1.2; // rad/s

#[derive(Clone, Debug)]
pub struct Autopilot {
    pub waypoints: Vec<Vec2>,
    pub next: usize,
}

impl Default for Autopilot {
    fn default() -> Autopilot {
        Autopilot {
            waypoints: COURSE_WAYPOINTS
                .iter()
                .map(|&(x, y)| Vec2::new(x, y))
                .collect(),
            next: 0,
        }
    }
}

impl Autopilot {
    /// Wheel speeds (left, right) in mm/s for a robot at this centre pose.
    pub fn command(&mut self, centre: Pose) -> (f32, f32) {
        let target = self.waypoints[self.next];
        if centre.position().distance(target) < WAYPOINT_REACHED {
            self.next = (self.next + 1) % self.waypoints.len();
        }
        let to = self.waypoints[self.next] - centre.position();
        let error = centre.forward().angle_to(to);
        let turn = (TURN_GAIN * error).clamp(-MAX_TURN_RATE, MAX_TURN_RATE);
        let forward = MAX_SPEED * error.cos().max(0.0).powi(2);
        let half = turn * crate::robot::ROBOT_WIDTH / 2.0;
        (forward - half, forward + half)
    }
}

impl Simulator {
    /// The true centre pose (for the autopilot).
    pub fn centre(&self) -> Pose {
        self.centre
    }
}

// --- Part 4c: a whole run ------------------------------------------------------

/// Drive the autopilot for `steps` records and return the log, landmarks
/// included: what `cargo run -- generate` writes to disk.
pub fn generate(config: SynthConfig, seed: u64, steps: usize) -> LogFile {
    let mut sim = Simulator::new(config, Arena::course(), seed);
    let mut pilot = Autopilot::default();
    let mut log = LogFile {
        landmarks: sim.arena.landmarks.clone(),
        ..LogFile::default()
    };
    for step in 0..steps {
        if step > 0 {
            let (l, r) = pilot.command(sim.centre());
            sim.advance(l, r);
        }
        push(&mut log, sim.records());
    }
    log
}

/// Add one step's records to a log.
pub fn push(log: &mut LogFile, r: Records) {
    log.motors.push(r.motor);
    log.scans.push(r.scan);
    log.reference.push(r.reference);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_has_every_record_for_every_step() {
        let log = generate(SynthConfig::REALISTIC, 1, 50);
        assert_eq!(log.motors.len(), 50);
        assert_eq!(log.scans.len(), 50);
        assert_eq!(log.reference.len(), 50);
        assert!(log.scans.iter().all(|s| s.ranges.len() == BEAM_COUNT));
        assert_eq!(log.landmarks.len(), 6);
    }

    #[test]
    fn same_seed_same_log() {
        assert_eq!(
            generate(SynthConfig::REALISTIC, 3, 20),
            generate(SynthConfig::REALISTIC, 3, 20)
        );
    }

    /// The autopilot goes round the arena without bumping, for many seeds,
    /// and visits every waypoint.
    #[test]
    fn autopilot_never_bumps() {
        for seed in 0..20 {
            let mut sim = Simulator::new(SynthConfig::REALISTIC, Arena::course(), seed);
            let mut pilot = Autopilot::default();
            let mut visited = 0;
            let mut last = pilot.next;
            for step in 0..2 * DEFAULT_STEPS {
                let (l, r) = pilot.command(sim.centre());
                sim.advance(l, r);
                assert!(!sim.bumped, "seed {seed} bumped at step {step}");
                if pilot.next != last {
                    visited += 1;
                    last = pilot.next;
                }
            }
            assert!(visited >= COURSE_WAYPOINTS.len(), "seed {seed}: {visited}");
        }
    }

    #[test]
    fn straight_and_on_the_spot() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let p = drive(start, 100.0, 100.0, 150.0);
        assert!((p.position() - Vec2::new(100.0, 0.0)).length() < 1e-3);
        // Wheels in opposite directions: turn on the spot by 2 * 75 / 150 rad.
        let p = drive(start, -75.0, 75.0, 150.0);
        assert!(p.position().length() < 1e-3);
        assert!((p.heading - 1.0).abs() < 1e-4);
    }

    #[test]
    fn scans_see_the_nearby_wall() {
        // At the start the robot faces down-left (213 degrees) close to the
        // top wall: beam 0 points about 97 degrees, almost straight up.
        let mut sim = Simulator::new(SynthConfig::PERFECT, Arena::course(), 0);
        let r = sim.records();
        let up = r.scan.ranges[0] as f32;
        let expected =
            (2000.0 - START_POSE.y) / (START_POSE.heading + beam_index_to_angle(0.0)).sin();
        assert!((up - expected).abs() <= 1.0, "{up} vs {expected}");
    }
}
