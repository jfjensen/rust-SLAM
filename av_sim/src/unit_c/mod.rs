// Unit C of the course: what the robot believes about where it is, as a
// probability distribution, in 1D.
//
//   distribution.rs  discrete distributions, convolve (move), multiply
//                    (measure), the histogram filter step
//   kalman.rs        the same with gaussians only: the 1D Kalman filter
//   track.rs         the 1D world: a robot on a straight track, the run
//                    as a file (B R Q S records)
//   source.rs        Replay and Live for the track (like ../source.rs)
//
// The two filters run side by side on the same controls and measurements,
// each in its own plugin. Like the earlier units, they only read the Track
// resource, never the truth.

pub mod distribution;
pub mod kalman;
pub mod source;
pub mod track;

use bevy::app::plugin_group;
use bevy::prelude::*;
use distribution::{Distribution, HistogramStep, histogram_step};
use kalman::{Density, KalmanStep, kalman_step};
use track::{TrackLog, TrackModel, TrackStep};

plugin_group! {
    /// Unit C's filters.
    pub struct UnitCPlugins {
        :HistogramFilterPlugin,
        :KalmanFilterPlugin,
    }
}

/// The run the filters read: replayed from a file, or growing live.
#[derive(Resource, Debug, Default)]
pub struct Track {
    pub log: TrackLog,
    /// Goes up when a different run is loaded (see RobotLog).
    pub generation: u64,
}

impl Track {
    pub fn replace(&mut self, log: TrackLog) {
        self.log = log;
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.log.len()
    }

    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }
}

// --- The histogram filter ---------------------------------------------------------

/// The shape the histogram filter gives the start, the controls and the
/// measurements (the course's `Dist = Distribution.triangle or
/// Distribution.gaussian`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shape {
    #[default]
    Gaussian,
    /// NOTE: as in the course, the sigma is used as the half width. A
    /// triangle that wide is narrower than the gaussian (its sigma is
    /// about 0.4 of the half width), so the filter trusts everything more.
    Triangle,
}

impl Shape {
    /// A distribution of this shape around `centre`, rounded to a cell.
    pub fn make(self, centre: f64, sigma: f64) -> Distribution {
        let c = centre.round() as i64;
        match self {
            Shape::Gaussian => Distribution::gaussian(c, sigma),
            Shape::Triangle => Distribution::triangle(c, sigma.round() as i64),
        }
    }
}

#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HistogramParams {
    pub shape: Shape,
}

#[derive(Resource, Debug, Default)]
pub struct HistogramFilter {
    /// One per track step.
    pub steps: Vec<HistogramStep>,
    generation: u64,
    shape: Shape,
}

impl HistogramFilter {
    /// The belief before step i: the start, or step i-1's correction.
    pub fn belief_before(&self, i: usize, model: &TrackModel, shape: Shape) -> Distribution {
        match i.checked_sub(1).and_then(|j| self.steps.get(j)) {
            Some(step) => step.correction.clone(),
            None => shape.make(model.start, model.start_sigma),
        }
    }
}

/// One histogram filter step on one track step.
pub fn histogram_track_step(
    belief: &Distribution,
    step: &TrackStep,
    model: &TrackModel,
    shape: Shape,
) -> HistogramStep {
    histogram_step(
        belief,
        &shape.make(step.control, model.control_sigma),
        step.measured
            .map(|z| shape.make(z, model.measurement_sigma)),
    )
}

#[derive(Default)]
pub struct HistogramFilterPlugin;

impl Plugin for HistogramFilterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Track>()
            .init_resource::<HistogramParams>()
            .init_resource::<HistogramFilter>()
            .add_systems(
                Update,
                update_histogram.in_set(crate::core::AvSet::Algorithms),
            );
    }
}

/// Start over when the run or the shape changes; then filter the new steps.
fn update_histogram(
    track: Res<Track>,
    params: Res<HistogramParams>,
    mut out: ResMut<HistogramFilter>,
) {
    if out.generation != track.generation
        || out.shape != params.shape
        || out.steps.len() > track.len()
    {
        out.steps.clear();
        out.generation = track.generation;
        out.shape = params.shape;
    }
    let model = track.log.model;
    for i in out.steps.len()..track.len() {
        let belief = out.belief_before(i, &model, params.shape);
        let step = histogram_track_step(&belief, &track.log.steps[i], &model, params.shape);
        out.steps.push(step);
    }
}

// --- The Kalman filter --------------------------------------------------------------

#[derive(Resource, Debug, Default)]
pub struct KalmanFilter {
    pub steps: Vec<KalmanStep>,
    generation: u64,
}

impl KalmanFilter {
    pub fn belief_before(&self, i: usize, model: &TrackModel) -> Density {
        match i.checked_sub(1).and_then(|j| self.steps.get(j)) {
            Some(step) => step.correction,
            None => Density::new(model.start, model.start_sigma),
        }
    }
}

pub fn kalman_track_step(belief: Density, step: &TrackStep, model: &TrackModel) -> KalmanStep {
    kalman_step(
        belief,
        Density::new(step.control, model.control_sigma),
        step.measured
            .map(|z| Density::new(z, model.measurement_sigma)),
    )
}

#[derive(Default)]
pub struct KalmanFilterPlugin;

impl Plugin for KalmanFilterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Track>()
            .init_resource::<KalmanFilter>()
            .add_systems(Update, update_kalman.in_set(crate::core::AvSet::Algorithms));
    }
}

fn update_kalman(track: Res<Track>, mut out: ResMut<KalmanFilter>) {
    if out.generation != track.generation || out.steps.len() > track.len() {
        out.steps.clear();
        out.generation = track.generation;
    }
    let model = track.log.model;
    for i in out.steps.len()..track.len() {
        let belief = out.belief_before(i, &model);
        let step = kalman_track_step(belief, &track.log.steps[i], &model);
        out.steps.push(step);
    }
}

// --- Whole runs, without Bevy (for tests and measuring) ---------------------------

pub fn histogram_run(log: &TrackLog, shape: Shape) -> Vec<HistogramStep> {
    let mut belief = shape.make(log.model.start, log.model.start_sigma);
    log.steps
        .iter()
        .map(|s| {
            let step = histogram_track_step(&belief, s, &log.model, shape);
            belief = step.correction.clone();
            step
        })
        .collect()
}

pub fn kalman_run(log: &TrackLog) -> Vec<KalmanStep> {
    let mut belief = Density::new(log.model.start, log.model.start_sigma);
    log.steps
        .iter()
        .map(|s| {
            let step = kalman_track_step(belief, s, &log.model);
            belief = step.correction;
            step
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use track::{SENSOR_GAP, generate};

    /// Over generated runs, the truth stays inside the Kalman filter's
    /// belief about as often as a gaussian says: within 2 sigma ~95% of
    /// the time. That's what "the filter is consistent" means.
    #[test]
    fn the_truth_is_inside_two_sigma() {
        let (mut inside, mut total) = (0, 0);
        for seed in 1..=20 {
            let log = generate(seed, track::DEFAULT_STEPS);
            for (k, s) in kalman_run(&log).iter().zip(&log.steps) {
                total += 1;
                if (k.correction.mu - s.truth).abs() < 2.0 * k.correction.sigma() {
                    inside += 1;
                }
            }
        }
        let share = inside as f64 / total as f64;
        assert!(share > 0.9 && share < 0.99, "{share}");
    }

    /// In the sensor gap the belief only grows; after it, one measurement
    /// shrinks it again.
    #[test]
    fn the_gap_widens_the_belief() {
        let log = generate(1, track::DEFAULT_STEPS);
        let k = kalman_run(&log);
        let before = k[SENSOR_GAP.start - 1].correction.sigma2;
        let last_in_gap = k[SENSOR_GAP.end - 1].correction.sigma2;
        let after = k[SENSOR_GAP.end].correction.sigma2;
        let r2 = log.model.control_sigma.powi(2);
        let gap = SENSOR_GAP.len() as f64;
        assert!((last_in_gap - (before + gap * r2)).abs() < 1e-6);
        assert!(after < last_in_gap / 2.0);

        let h = histogram_run(&log, Shape::Gaussian);
        assert!(
            h[SENSOR_GAP.end - 1].correction.variance()
                > 4.0 * h[SENSOR_GAP.start - 1].correction.variance()
        );
    }

    /// With gaussians the two filters agree all the way through a run.
    #[test]
    fn the_filters_agree_on_a_run() {
        let log = generate(2, track::DEFAULT_STEPS);
        for (h, k) in histogram_run(&log, Shape::Gaussian)
            .iter()
            .zip(kalman_run(&log))
        {
            // The histogram rounds the controls and measurements to whole cells.
            assert!(
                (h.correction.mean() - k.correction.mu).abs() < 1.0,
                "{} vs {:?}",
                h.correction.mean(),
                k.correction
            );
        }
    }

    /// Mean distance from the truth: measurements alone, the Kalman
    /// filter, the histogram filter with both shapes. Run with
    ///   cargo test unit_c::measure -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure() {
        for seed in 1..=5 {
            let log = generate(seed, track::DEFAULT_STEPS);
            let mean = |xs: &mut dyn Iterator<Item = f64>| {
                let v: Vec<f64> = xs.collect();
                v.iter().sum::<f64>() / v.len() as f64
            };
            let sensor = mean(
                &mut log
                    .steps
                    .iter()
                    .filter_map(|s| s.measured.map(|z| (z - s.truth).abs())),
            );
            let kalman = mean(
                &mut kalman_run(&log)
                    .iter()
                    .zip(&log.steps)
                    .map(|(k, s)| (k.correction.mu - s.truth).abs()),
            );
            let gauss = mean(
                &mut histogram_run(&log, Shape::Gaussian)
                    .iter()
                    .zip(&log.steps)
                    .map(|(h, s)| (h.correction.mean() - s.truth).abs()),
            );
            let tri = mean(
                &mut histogram_run(&log, Shape::Triangle)
                    .iter()
                    .zip(&log.steps)
                    .map(|(h, s)| (h.correction.mean() - s.truth).abs()),
            );
            let ignored = histogram_run(&log, Shape::Triangle)
                .iter()
                .filter(|h| h.disagreed)
                .count();
            println!(
                "seed {seed}: sensor {sensor:5.1}  kalman {kalman:5.1}  histogram gaussian {gauss:5.1}  triangle {tri:5.1} cm, {ignored} measurements ignored"
            );
        }
    }
}
