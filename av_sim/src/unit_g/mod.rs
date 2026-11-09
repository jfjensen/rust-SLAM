// Unit G of the course: FastSLAM, the last unit.
//
//   fastslam.rs  a particle with its own map of little landmark filters
//   here         one step for the whole particle set, the plugin, whole
//                runs for tests
//
// Like Unit F, the filter never sees the arena's landmarks, and it starts
// at the true start pose (the course starts at an arbitrary one; see
// Unit F) so its map can be drawn over the real one.

pub mod fastslam;

use crate::core::{AvSet, RobotLog};
use crate::logfile::LogFile;
use crate::rng::Rng;
use crate::robot::{Pose, START_POSE};
use crate::unit_a::cylinders::{DetectorParams, detect};
use crate::unit_a::motion::MotionParams;
use crate::unit_d::ekf::{EkfParams, ErrorEllipse, error_ellipse, wrap_pi};
use crate::unit_e::particles::{self as pf, Sighting};
use crate::unit_e::scanner_pose;
use bevy::app::plugin_group;
use bevy::math::{DMat3, DVec2, DVec3};
use bevy::prelude::*;
use fastslam::{FastParticle, Landmark};

plugin_group! {
    /// Unit G's filter.
    pub struct UnitGPlugins {
        :FastSlamPlugin,
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct FastSlamSetup {
    pub filter: EkfParams,
    pub count: usize,
    /// Below this likelihood, a detection starts a new landmark (the
    /// course's minimum_correspondence_likelihood; a density in mm and
    /// radians, so it only means something next to these noise numbers).
    pub min_likelihood: f64,
    pub start: Pose,
    pub seed: u64,
}

impl Default for FastSlamSetup {
    /// The course's slam_10_f: scanner noise 200 mm and 15 degrees (as in
    /// Unit D), 200 particles, threshold 0.001.
    fn default() -> FastSlamSetup {
        FastSlamSetup {
            filter: EkfParams::default(),
            count: 200,
            min_likelihood: 0.001,
            start: START_POSE,
            seed: 1,
        }
    }
}

impl FastSlamSetup {
    /// The scanner noise our synthetic scanner really has (as av04's T).
    pub fn sharp_scanner() -> FastSlamSetup {
        FastSlamSetup {
            filter: EkfParams {
                measurement_distance_stddev: 20.0,
                measurement_angle_stddev: 1.0_f64.to_radians(),
                ..EkfParams::default()
            },
            ..FastSlamSetup::default()
        }
    }

    /// Every particle starts at the start pose, certain, with no map.
    pub fn initial_particles(&self) -> Vec<FastParticle> {
        let c = self.start.shifted(-self.filter.scanner_displacement as f32);
        let pose = DVec3::new(c.x as f64, c.y as f64, wrap_pi(c.heading as f64));
        vec![FastParticle::new(pose); self.count]
    }
}

/// One step, kept for drawing. Not the particles' maps: only the map of
/// the particle nearest the mean (as the course writes it).
#[derive(Clone, Debug, PartialEq)]
pub struct FastSlamStep {
    /// Every particle's pose, after resampling.
    pub poses: Vec<DVec3>,
    /// The mean pose, as a scanner pose.
    pub pose: Pose,
    pub ellipse: ErrorEllipse,
    pub heading_stddev: f64,
    /// The map of the particle nearest the mean.
    pub map: Vec<Landmark>,
    /// How many particles really counted (1 / sum(w^2)).
    pub effective: f64,
    pub sightings: usize,
    /// Detections that started a new landmark, per particle (average).
    pub new_per_particle: f64,
    /// Landmarks per particle, smallest and largest: different particles
    /// can believe in different numbers of landmarks.
    pub map_sizes: (usize, usize),
}

impl FastSlamStep {
    pub fn landmark_positions(&self) -> Vec<Vec2> {
        self.map
            .iter()
            .map(|l| Vec2::new(l.position.x as f32, l.position.y as f32))
            .collect()
    }

    pub fn landmark_ellipse(l: &Landmark) -> ErrorEllipse {
        let c = l.covariance;
        error_ellipse(&DMat3::from_cols(
            c.col(0).extend(0.0),
            c.col(1).extend(0.0),
            DVec3::ZERO,
        ))
    }
}

pub struct FastSlamInputs<'a> {
    pub motion: &'a MotionParams,
    pub detector: &'a DetectorParams,
    pub setup: &'a FastSlamSetup,
}

/// The course's loop body: predict, correct every particle, resample.
pub fn fastslam_step(
    particles: &mut Vec<FastParticle>,
    ticks: (i64, i64),
    scan: &[u32],
    inputs: &FastSlamInputs,
    rng: &mut Rng,
) -> FastSlamStep {
    let p = &inputs.setup.filter;
    let mm = inputs.motion.ticks_to_mm as f64;
    let control = DVec2::new(ticks.0 as f64 * mm, ticks.1 as f64 * mm);

    // Predict: Unit E's noisy motion, on each particle's pose.
    let mut poses: Vec<DVec3> = particles.iter().map(|q| q.pose).collect();
    pf::predict(&mut poses, control, p, rng);
    for (q, pose) in particles.iter_mut().zip(&poses) {
        q.pose = *pose;
    }

    // Correct each particle's map, and weigh it.
    let sightings: Vec<Sighting> = detect(scan, inputs.detector)
        .iter()
        .map(|c| Sighting {
            measurement: DVec2::new(c.local.length() as f64, c.local.to_angle() as f64),
            local: DVec2::new(c.local.x as f64, c.local.y as f64),
        })
        .collect();
    let min_ll = inputs.setup.min_likelihood.ln();
    let (log_weights, new): (Vec<f64>, Vec<usize>) = particles
        .iter_mut()
        .map(|q| q.correct(&sightings, min_ll, p))
        .unzip();
    // NOTE: the course multiplies the likelihoods; adding logs can't
    // underflow (see Unit E).
    let weights = pf::normalise(&log_weights);
    *particles = pf::resample(particles, &weights, rng);

    let poses: Vec<DVec3> = particles.iter().map(|q| q.pose).collect();
    let centre = pf::mean(&poses);
    let (ellipse, heading_stddev) = pf::spread(&poses, centre);
    let nearest = (0..particles.len())
        .min_by(|&a, &b| {
            let da = poses[a].truncate().distance(centre.truncate());
            let db = poses[b].truncate().distance(centre.truncate());
            da.total_cmp(&db)
        })
        .unwrap_or(0);
    let sizes = particles.iter().map(|q| q.landmarks.len());
    FastSlamStep {
        pose: scanner_pose(centre, p),
        ellipse,
        heading_stddev,
        map: particles
            .get(nearest)
            .map(|q| q.landmarks.clone())
            .unwrap_or_default(),
        effective: pf::effective_count(&weights),
        sightings: sightings.len(),
        new_per_particle: new.iter().sum::<usize>() as f64 / new.len().max(1) as f64,
        map_sizes: (sizes.clone().min().unwrap_or(0), sizes.max().unwrap_or(0)),
        poses,
    }
}

// --- The plugin ----------------------------------------------------------------

#[derive(Resource, Debug, Default)]
pub struct FastSlamMapping {
    pub steps: Vec<FastSlamStep>,
    /// The particles and the random numbers carry on from step to step.
    particles: Vec<FastParticle>,
    rng: Option<Rng>,
    generation: u64,
    setup: Option<FastSlamSetup>,
}

impl FastSlamMapping {
    pub fn poses(&self) -> impl Iterator<Item = Pose> + '_ {
        self.steps.iter().map(|s| s.pose)
    }
}

#[derive(Default)]
pub struct FastSlamPlugin;

impl Plugin for FastSlamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionParams>()
            .init_resource::<DetectorParams>()
            .init_resource::<FastSlamSetup>()
            .init_resource::<FastSlamMapping>()
            .add_systems(Update, update_fastslam.in_set(AvSet::Algorithms));
    }
}

fn update_fastslam(
    log: Res<RobotLog>,
    motion: Res<MotionParams>,
    detector: Res<DetectorParams>,
    setup: Res<FastSlamSetup>,
    mut out: ResMut<FastSlamMapping>,
) {
    if out.generation != log.generation || out.setup != Some(*setup) || out.steps.len() > log.len()
    {
        out.steps.clear();
        out.rng = None;
        out.generation = log.generation;
        out.setup = Some(*setup);
    }
    let inputs = FastSlamInputs {
        motion: &motion,
        detector: &detector,
        setup: &setup,
    };
    let out = &mut *out;
    if out.rng.is_none() {
        out.rng = Some(Rng::new(setup.seed));
        out.particles = setup.initial_particles();
    }
    let rng = out.rng.as_mut().expect("set above");
    let records = log.log.motors.len().min(log.log.scans.len());
    for i in out.steps.len()..records {
        let step = fastslam_step(
            &mut out.particles,
            log.log.motor_increment(i),
            &log.log.scans[i].ranges,
            &inputs,
            rng,
        );
        out.steps.push(step);
    }
}

// --- Whole runs, without Bevy (for tests and measuring) ---------------------------

pub fn fastslam_trajectory(log: &LogFile, setup: &FastSlamSetup) -> Vec<FastSlamStep> {
    let inputs = FastSlamInputs {
        motion: &MotionParams::default(),
        detector: &DetectorParams::default(),
        setup,
    };
    let mut rng = Rng::new(setup.seed);
    let mut particles = setup.initial_particles();
    (0..log.motors.len().min(log.scans.len()))
        .map(|i| {
            fastslam_step(
                &mut particles,
                log.motor_increment(i),
                &log.scans[i].ranges,
                &inputs,
                &mut rng,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{DEFAULT_STEPS, SynthConfig, generate};
    use crate::unit_b::error_stats;

    fn truth_landmarks(log: &LogFile) -> Vec<Vec2> {
        log.landmarks.iter().map(|l| l.position).collect()
    }

    /// For each landmark in the map, the distance to the nearest real one:
    /// (mean, worst).
    fn map_landmark_errors(step: &FastSlamStep, real: &[Vec2]) -> (f32, f32) {
        let d: Vec<f32> = step
            .landmark_positions()
            .iter()
            .map(|m| real.iter().map(|r| r.distance(*m)).fold(f32::MAX, f32::min))
            .collect();
        (
            d.iter().sum::<f32>() / d.len().max(1) as f32,
            d.iter().copied().fold(0.0, f32::max),
        )
    }

    /// Over a whole run the robot stays near the truth, and every landmark
    /// in the map is near a real one. Not every real one is in the map:
    /// the cylinder at (1291, 1881) stands 119 mm in front of the top wall,
    /// the detector finds it on 7 of the 175 steps it's in view (seed 1;
    /// no depth jump against the wall), and the course's counter rule (-1
    /// whenever in view) deletes it like a spurious one. EKF SLAM (Unit F)
    /// has no counter and keeps it.
    #[test]
    fn it_maps_the_arena_and_follows_the_truth() {
        let log = generate(SynthConfig::REALISTIC, 1, DEFAULT_STEPS);
        let steps = fastslam_trajectory(&log, &FastSlamSetup::sharp_scanner());
        let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
        let (mean, worst) = error_stats(&poses, &log);
        assert!(mean < 20.0 && worst < 60.0, "mean {mean} worst {worst}");
        let last = steps.last().unwrap();
        assert!((5..=6).contains(&last.map.len()), "{}", last.map.len());
        let (_, worst_map) = map_landmark_errors(last, &truth_landmarks(&log));
        assert!(worst_map < 60.0, "{worst_map}");
    }

    /// The plugin, fed one record per frame, gives the same run as the
    /// whole-run function.
    #[test]
    fn frame_by_frame_is_the_same_run() {
        let log = generate(SynthConfig::REALISTIC, 2, 20);
        let setup = FastSlamSetup {
            count: 30,
            ..FastSlamSetup::default()
        };
        let whole = fastslam_trajectory(&log, &setup);
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, crate::core::AvCorePlugin, FastSlamPlugin))
            .insert_resource(setup);
        let mut partial = log.clone();
        partial.motors.clear();
        partial.scans.clear();
        app.world_mut().resource_mut::<RobotLog>().replace(partial);
        for i in 0..20 {
            let mut r = app.world_mut().resource_mut::<RobotLog>();
            r.log.motors.push(log.motors[i]);
            r.log.scans.push(log.scans[i].clone());
            app.update();
        }
        let steps = &app.world().resource::<FastSlamMapping>().steps;
        assert_eq!(steps.len(), 20);
        assert_eq!(steps[19], whole[19]);
    }

    /// Robot error, and each map landmark's distance from the nearest real
    /// one, over seeds 1-5. Run with
    ///   cargo test --release unit_g::tests::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        let course_1000 = FastSlamSetup {
            count: 1000,
            ..FastSlamSetup::default()
        };
        // The course's scanner noise, but wheel noise closer to the
        // synthetic robot's (it slips 2%; the course's 0.35 / 0.6 were for
        // the real LEGO robot).
        let course_calm = FastSlamSetup {
            filter: EkfParams {
                control_motion_factor: 0.1,
                control_turn_factor: 0.2,
                ..EkfParams::default()
            },
            ..FastSlamSetup::default()
        };
        for seed in 1..=5 {
            let log = generate(SynthConfig::REALISTIC, seed, DEFAULT_STEPS);
            for (name, setup) in [
                ("course     ", FastSlamSetup::default()),
                ("course 1000", course_1000),
                ("course calm", course_calm),
                ("sharp      ", FastSlamSetup::sharp_scanner()),
            ] {
                let setup = FastSlamSetup { seed, ..setup };
                let steps = fastslam_trajectory(&log, &setup);
                let poses: Vec<Pose> = steps.iter().map(|s| s.pose).collect();
                let (mean, worst) = error_stats(&poses, &log);
                let last = steps.last().unwrap();
                let (map_mean, map_worst) = map_landmark_errors(last, &truth_landmarks(&log));
                // Of all detections, how many started a new landmark
                // instead of matching one (averaged over particles).
                let seen: usize = steps.iter().map(|s| s.sightings).sum();
                let new: f64 = steps.iter().map(|s| s.new_per_particle).sum();
                println!(
                    "seed {seed} {name}: robot mean {mean:5.1} worst {worst:5.1} mm | map {} landmarks, from the nearest real one: mean {map_mean:5.1} worst {map_worst:5.1} mm | {:4.1}% of detections made a new landmark",
                    last.map.len(),
                    100.0 * new / seen as f64
                );
            }
        }
    }
}
