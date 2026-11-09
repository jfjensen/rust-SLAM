// Unit E of the course: localisation with a particle filter.
//
//   particles.rs  predict, weigh, resample, mean and spread
//   here          one filter step on one log record, the plugin, whole
//                 runs for tests
//
// Two ways to start (ParticleSetup::start):
//   Known     particles around the measured start pose (slam_08_a..c)
//   Anywhere  particles all over the arena, any heading (slam_08_d): the
//             robot doesn't know where it is. Unit D's Kalman filter
//             can't do this: it is one gaussian, one blob.
//
// Like Unit D, the particles are robot CENTRES; EkfStep-style, the pose
// handed out is the scanner's.

pub mod particles;

use crate::core::{AvSet, RobotLog};
use crate::logfile::LogFile;
use crate::rng::Rng;
use crate::robot::{Pose, START_POSE, wrap_angle};
use crate::unit_a::cylinders::{DetectorParams, detect};
use crate::unit_a::motion::MotionParams;
use crate::unit_d::ekf::{EkfParams, ErrorEllipse, wrap_pi};
use crate::world::ARENA_SIZE;
use bevy::app::plugin_group;
use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use particles::{Particle, Sighting};

plugin_group! {
    /// Unit E's filter.
    pub struct UnitEPlugins {
        :ParticleFilterPlugin,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Start {
    /// Around the measured start pose.
    #[default]
    Known,
    /// Anywhere in the arena.
    Anywhere,
}

/// The filter's settings (the course's slam_08_d, except the count when
/// the start is known: 08_a..c use 50-300).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ParticleSetup {
    /// Motion and measurement noise: the same numbers as Unit D's filter.
    pub filter: EkfParams,
    pub count: usize,
    pub start: Start,
    pub start_pose: Pose,
    /// Spread of the Known start: x, y (mm), heading (rad).
    pub start_sigma: DVec3,
    /// Seeds the particles' random numbers: same seed, same run.
    pub seed: u64,
}

impl Default for ParticleSetup {
    fn default() -> ParticleSetup {
        ParticleSetup {
            filter: EkfParams::default(),
            count: 500,
            start: Start::Known,
            start_pose: START_POSE,
            start_sigma: DVec3::new(100.0, 100.0, 10.0_f64.to_radians()),
            seed: 1,
        }
    }
}

impl ParticleSetup {
    pub fn initial_particles(&self, rng: &mut Rng) -> Vec<Particle> {
        match self.start {
            Start::Known => {
                let c = self
                    .start_pose
                    .shifted(-self.filter.scanner_displacement as f32);
                let centre = DVec3::new(c.x as f64, c.y as f64, wrap_pi(c.heading as f64));
                particles::around(centre, self.start_sigma, self.count, rng)
            }
            Start::Anywhere => particles::anywhere(ARENA_SIZE as f64, self.count, rng),
        }
    }
}

/// A centre state to the scanner pose (f32, heading in [0, 2 pi)).
pub fn scanner_pose(state: DVec3, p: &EkfParams) -> Pose {
    Pose::new(state.x as f32, state.y as f32, wrap_angle(state.z as f32))
        .shifted(p.scanner_displacement as f32)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParticleStep {
    /// After predicting, before resampling...
    pub predicted: Vec<Particle>,
    /// ... and how much each of those counted (they add up to 1).
    pub weights: Vec<f64>,
    /// After resampling: what the next step starts from.
    pub particles: Vec<Particle>,
    /// The mean of `particles`, as a scanner pose.
    pub pose: Pose,
    pub ellipse: ErrorEllipse,
    pub heading_stddev: f64,
    /// 1 / sum(w^2) of the weights: how many particles really counted.
    pub effective: f64,
    /// Cylinders the scanner saw this step.
    pub sightings: usize,
    /// The best particle's log weight (0 = it fits every sighting exactly;
    /// see particles::log_weight). 0 when not updated.
    pub best_log_weight: f64,
    /// False when the robot didn't move: then the course skips predict and
    /// correct (see particle_step).
    pub updated: bool,
}

/// The course's filter loop body (slam_08_d).
///
/// Only when the robot moved: resampling over and over with nothing new
/// (no motion noise to spread the copies) would keep copying the luckiest
/// particles until the cloud is a handful of points.
pub fn particle_step(
    previous: &[Particle],
    ticks: (i64, i64),
    scan: &[u32],
    inputs: &ParticleInputs,
    rng: &mut Rng,
) -> ParticleStep {
    let p = &inputs.setup.filter;
    let mm = inputs.motion.ticks_to_mm as f64;
    let control = DVec2::new(ticks.0 as f64 * mm, ticks.1 as f64 * mm);
    let cylinders = detect(scan, inputs.detector);
    let sightings: Vec<Sighting> = cylinders
        .iter()
        .map(|c| Sighting {
            measurement: DVec2::new(c.local.length() as f64, c.local.to_angle() as f64),
            local: DVec2::new(c.local.x as f64, c.local.y as f64),
        })
        .collect();

    let mut predicted = previous.to_vec();
    let updated = control != DVec2::ZERO;
    let (weights, after, best_log_weight) = if updated {
        particles::predict(&mut predicted, control, p, rng);
        let log_weights: Vec<f64> = predicted
            .iter()
            .map(|&q| particles::log_weight(q, &sightings, inputs.landmarks, p))
            .collect();
        let best = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights = particles::normalise(&log_weights);
        let after = particles::resample(&predicted, &weights, rng);
        (weights, after, best)
    } else {
        let n = predicted.len().max(1) as f64;
        (vec![1.0 / n; predicted.len()], predicted.clone(), 0.0)
    };
    let centre = particles::mean(&after);
    let (ellipse, heading_stddev) = particles::spread(&after, centre);
    ParticleStep {
        effective: particles::effective_count(&weights),
        predicted,
        weights,
        pose: scanner_pose(centre, p),
        particles: after,
        ellipse,
        heading_stddev,
        sightings: sightings.len(),
        best_log_weight,
        updated,
    }
}

pub struct ParticleInputs<'a> {
    pub motion: &'a MotionParams,
    pub detector: &'a DetectorParams,
    pub setup: &'a ParticleSetup,
    pub landmarks: &'a [DVec2],
}

// --- The plugin ----------------------------------------------------------------

#[derive(Resource, Debug, Default)]
pub struct ParticleLocalization {
    pub steps: Vec<ParticleStep>,
    /// The random numbers carry on from step to step (and are restarted
    /// from the seed with the filter), so a run is the same however many
    /// frames it was computed in.
    rng: Option<Rng>,
    generation: u64,
    setup: Option<ParticleSetup>,
}

impl ParticleLocalization {
    pub fn poses(&self) -> impl Iterator<Item = Pose> + '_ {
        self.steps.iter().map(|s| s.pose)
    }
}

#[derive(Default)]
pub struct ParticleFilterPlugin;

impl Plugin for ParticleFilterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionParams>()
            .init_resource::<DetectorParams>()
            .init_resource::<ParticleSetup>()
            .init_resource::<ParticleLocalization>()
            .add_systems(Update, update_particles.in_set(AvSet::Algorithms));
    }
}

fn landmarks_of(log: &RobotLog) -> Vec<DVec2> {
    log.arena
        .landmarks
        .iter()
        .map(|l| DVec2::new(l.position.x as f64, l.position.y as f64))
        .collect()
}

fn update_particles(
    log: Res<RobotLog>,
    motion: Res<MotionParams>,
    detector: Res<DetectorParams>,
    setup: Res<ParticleSetup>,
    mut out: ResMut<ParticleLocalization>,
) {
    if out.generation != log.generation || out.setup != Some(*setup) || out.steps.len() > log.len()
    {
        out.steps.clear();
        out.rng = None;
        out.generation = log.generation;
        out.setup = Some(*setup);
    }
    let landmarks = landmarks_of(&log);
    let inputs = ParticleInputs {
        motion: &motion,
        detector: &detector,
        setup: &setup,
        landmarks: &landmarks,
    };
    let out = &mut *out;
    let rng = out.rng.get_or_insert_with(|| Rng::new(setup.seed));
    let records = log.log.motors.len().min(log.log.scans.len());
    for i in out.steps.len()..records {
        let step = match out.steps.last() {
            Some(last) => particle_step(
                &last.particles,
                log.log.motor_increment(i),
                &log.log.scans[i].ranges,
                &inputs,
                rng,
            ),
            None => {
                let start = setup.initial_particles(rng);
                particle_step(
                    &start,
                    log.log.motor_increment(i),
                    &log.log.scans[i].ranges,
                    &inputs,
                    rng,
                )
            }
        };
        out.steps.push(step);
    }
}

// --- Whole runs, without Bevy (for tests and measuring) ---------------------------

pub fn particle_trajectory(log: &LogFile, setup: &ParticleSetup) -> Vec<ParticleStep> {
    let landmarks: Vec<DVec2> = log
        .landmarks
        .iter()
        .map(|l| DVec2::new(l.position.x as f64, l.position.y as f64))
        .collect();
    let inputs = ParticleInputs {
        motion: &MotionParams::default(),
        detector: &DetectorParams::default(),
        setup,
        landmarks: &landmarks,
    };
    let mut rng = Rng::new(setup.seed);
    let mut cloud = setup.initial_particles(&mut rng);
    (0..log.motors.len().min(log.scans.len()))
        .map(|i| {
            let step = particle_step(
                &cloud,
                log.motor_increment(i),
                &log.scans[i].ranges,
                &inputs,
                &mut rng,
            );
            cloud = step.particles.clone();
            step
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{DEFAULT_STEPS, SynthConfig, generate};
    use crate::unit_b::error_stats;

    /// Distance from the truth from step `from` on.
    fn errors_after(steps: &[ParticleStep], log: &LogFile, from: usize) -> (f32, f32) {
        let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
        let mut tail = log.clone();
        tail.reference = tail.reference.split_off(from);
        error_stats(&poses[from..], &tail)
    }

    #[test]
    fn from_a_known_start_it_follows_the_truth() {
        let log = generate(SynthConfig::REALISTIC, 1, DEFAULT_STEPS);
        let steps = particle_trajectory(&log, &ParticleSetup::default());
        let (mean, worst) = errors_after(&steps, &log, 0);
        assert!(mean < 20.0 && worst < 80.0, "mean {mean} worst {worst}");
    }

    /// The kidnapped robot: no idea where it is at the start, found within
    /// a few dozen steps, then followed.
    #[test]
    fn from_anywhere_it_finds_the_robot() {
        let log = generate(SynthConfig::REALISTIC, 1, DEFAULT_STEPS);
        let setup = ParticleSetup {
            start: Start::Anywhere,
            ..ParticleSetup::default()
        };
        let steps = particle_trajectory(&log, &setup);
        assert!(steps[0].ellipse.major > 300.0, "{:?}", steps[0].ellipse);
        let (mean, worst) = errors_after(&steps, &log, 60);
        assert!(mean < 25.0 && worst < 100.0, "mean {mean} worst {worst}");
    }

    /// The plugin, fed one record per frame, gives the same run as the
    /// whole-run function: the random numbers carry over between frames.
    #[test]
    fn frame_by_frame_is_the_same_run() {
        let log = generate(SynthConfig::REALISTIC, 2, 30);
        let whole = particle_trajectory(&log, &ParticleSetup::default());
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            crate::core::AvCorePlugin,
            ParticleFilterPlugin,
        ));
        let mut partial = log.clone();
        partial.motors.clear();
        partial.scans.clear();
        app.world_mut().resource_mut::<RobotLog>().replace(partial);
        for i in 0..30 {
            let mut r = app.world_mut().resource_mut::<RobotLog>();
            r.log.motors.push(log.motors[i]);
            r.log.scans.push(log.scans[i].clone());
            app.update();
        }
        let steps = &app.world().resource::<ParticleLocalization>().steps;
        assert_eq!(steps.len(), 30);
        assert_eq!(steps[29].particles, whole[29].particles);
    }

    /// How often would the course's plain products (with the pdfs'
    /// constant factors) be 0 for EVERY particle? Then its resampling wheel
    /// copies one particle N times. f64's exp underflows below about -745.
    /// Uniform start; the course's scanner noise and av04's sharp one.
    /// Run with  cargo test unit_e::tests::underflow -- --ignored --nocapture
    #[test]
    #[ignore]
    fn underflow() {
        let sharp = EkfParams {
            measurement_distance_stddev: 20.0,
            measurement_angle_stddev: 1.0_f64.to_radians(),
            ..EkfParams::default()
        };
        for (name, filter) in [("course", EkfParams::default()), ("sharp ", sharp)] {
            let mut zero = 0;
            let mut total = 0;
            for seed in 1..=5 {
                let log = generate(SynthConfig::REALISTIC, seed, DEFAULT_STEPS);
                let setup = ParticleSetup {
                    filter,
                    start: Start::Anywhere,
                    seed,
                    ..ParticleSetup::default()
                };
                let root = (2.0 * std::f64::consts::PI).sqrt();
                let constant = -(filter.measurement_distance_stddev * root).ln()
                    - (filter.measurement_angle_stddev * root).ln();
                for s in particle_trajectory(&log, &setup)
                    .iter()
                    .filter(|s| s.updated)
                {
                    total += 1;
                    if s.best_log_weight + s.sightings as f64 * constant < -745.0 {
                        zero += 1;
                    }
                }
            }
            println!("{name}: every weight 0 on {zero} of {total} steps");
        }
    }

    /// Mean and worst distance from the truth over seeds 1-5, both starts.
    /// Run with  cargo test unit_e::tests::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        for seed in 1..=5 {
            let log = generate(SynthConfig::REALISTIC, seed, DEFAULT_STEPS);
            for (name, start) in [("known   ", Start::Known), ("anywhere", Start::Anywhere)] {
                let setup = ParticleSetup {
                    start,
                    seed,
                    ..ParticleSetup::default()
                };
                let steps = particle_trajectory(&log, &setup);
                let (mean, worst) = errors_after(&steps, &log, 0);
                let (mean60, worst60) = errors_after(&steps, &log, 60);
                let found = steps.iter().zip(&log.reference).position(|(s, r)| {
                    s.pose
                        .position()
                        .distance(Vec2::new(r.x as f32, r.y as f32))
                        < 50.0
                        && s.ellipse.major < 100.0
                });
                println!(
                    "seed {seed} {name}: all steps mean {mean:6.1} worst {worst:6.1} mm | from step 60 mean {mean60:5.1} worst {worst60:5.1} | found at step {found:?}"
                );
            }
        }
    }
}
