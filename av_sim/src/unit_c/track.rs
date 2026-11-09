// Unit C, part 3: the 1D world the filters run in.
//
// Unit C has no log files: the course types its controls and measurements
// into the scripts. Here a synthetic robot drives up and down a straight
// track, so there is a truth to compare with, and the run can be saved
// and replayed like the other units' logs. One cell = one cm.
//
// Each step the robot is told to move by `control` cm, and really moves
// that plus some noise. Then, unless the sensor is off, it measures its
// position, again with noise. The filters see the controls and the
// measurements, never the truth.
//
// The file (data/track.txt), one record per line, like the course's logs:
//
//   B 20 1           start belief: position, sigma (cm)
//   R 6              the filters' control sigma (cm per step)
//   Q 15             the filters' measurement sigma (cm)
//   S 20 41.3 40.2   a step: control, measurement (- for none), truth
//
// Lines starting with # are comments; unknown records are skipped.

use crate::rng::Rng;

/// The track runs from 0 to this many cm.
pub const TRACK_LENGTH: f64 = 500.0;
/// The autopilot turns around this far from either end.
const TURN_MARGIN: f64 = 40.0;
/// Steps in a generated run.
pub const DEFAULT_STEPS: usize = 60;
/// A generated run has no measurements in these steps: a tunnel, say.
pub const SENSOR_GAP: std::ops::Range<usize> = 22..34;

/// What the filters assume (the B, R and Q records).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackModel {
    pub start: f64,
    pub start_sigma: f64,
    pub control_sigma: f64,
    pub measurement_sigma: f64,
}

impl Default for TrackModel {
    fn default() -> TrackModel {
        TrackModel {
            start: 20.0,
            start_sigma: 1.0,
            control_sigma: 6.0,
            measurement_sigma: 15.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackStep {
    /// How far the robot was told to move (cm).
    pub control: f64,
    /// Where the sensor said it was, if it said anything.
    pub measured: Option<f64>,
    /// Where it really was after this step. For drawing only.
    pub truth: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackLog {
    pub model: TrackModel,
    pub steps: Vec<TrackStep>,
}

impl TrackLog {
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn to_text(&self) -> String {
        let m = &self.model;
        let mut s = String::from(
            "# av03 1D track: B start sigma, R control sigma, Q measurement sigma,\n# S control measurement(- = none) truth. Units: cm.\n",
        );
        s += &format!(
            "B {} {}\nR {}\nQ {}\n",
            m.start, m.start_sigma, m.control_sigma, m.measurement_sigma
        );
        for step in &self.steps {
            let z = step.measured.map_or("-".into(), |z| format!("{z:.1}"));
            s += &format!("S {} {z} {:.1}\n", step.control, step.truth);
        }
        s
    }

    pub fn parse(text: &str) -> Result<TrackLog, String> {
        let mut log = TrackLog::default();
        for (n, line) in text.lines().enumerate() {
            let words: Vec<&str> = line.split_whitespace().collect();
            let bad = || format!("line {}: '{line}'", n + 1);
            let num = |i: usize| -> Result<f64, String> {
                words.get(i).and_then(|w| w.parse().ok()).ok_or_else(bad)
            };
            match words.first().copied() {
                Some("B") => {
                    log.model.start = num(1)?;
                    log.model.start_sigma = num(2)?;
                }
                Some("R") => log.model.control_sigma = num(1)?,
                Some("Q") => log.model.measurement_sigma = num(1)?,
                Some("S") => log.steps.push(TrackStep {
                    control: num(1)?,
                    measured: if words.get(2) == Some(&"-") {
                        None
                    } else {
                        Some(num(2)?)
                    },
                    truth: num(3)?,
                }),
                _ => {} // comments, blank lines, records of later units
            }
        }
        Ok(log)
    }

    pub fn read(path: &str) -> Result<TrackLog, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        TrackLog::parse(&text)
    }

    pub fn write(&self, path: &str) -> Result<(), String> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{path}: {e}"))?;
        }
        std::fs::write(path, self.to_text()).map_err(|e| format!("{path}: {e}"))
    }
}

/// The synthetic robot on the track.
#[derive(Clone, Debug)]
pub struct TrackSim {
    pub model: TrackModel,
    /// The true position (cm).
    pub truth: f64,
    /// The autopilot's direction: +1 or -1.
    pub direction: f64,
    rng: Rng,
}

/// How far the autopilot asks to move each step (cm).
pub const SPEED: f64 = 20.0;

impl TrackSim {
    /// The robot starts where the model says, give or take start_sigma.
    pub fn new(model: TrackModel, seed: u64) -> TrackSim {
        let mut rng = Rng::new(seed);
        let truth = model.start + rng.gaussian(model.start_sigma);
        TrackSim {
            model,
            truth,
            direction: 1.0,
            rng,
        }
    }

    /// The autopilot: drive at SPEED, turn around near the ends. It goes
    /// by the truth, as if it could see the end markers.
    pub fn autopilot(&mut self) -> f64 {
        if self.truth > TRACK_LENGTH - TURN_MARGIN {
            self.direction = -1.0;
        } else if self.truth < TURN_MARGIN {
            self.direction = 1.0;
        }
        self.direction * SPEED
    }

    /// Move by `control` (plus noise, stopped by the ends of the track),
    /// then measure if `measure`.
    pub fn step(&mut self, control: f64, measure: bool) -> TrackStep {
        let moved = control + self.rng.gaussian(self.model.control_sigma);
        self.truth = (self.truth + moved).clamp(0.0, TRACK_LENGTH);
        let measured =
            measure.then(|| self.truth + self.rng.gaussian(self.model.measurement_sigma));
        TrackStep {
            control,
            measured,
            truth: self.truth,
        }
    }
}

/// A whole run on the autopilot, with the sensor off in SENSOR_GAP.
pub fn generate(seed: u64, steps: usize) -> TrackLog {
    let model = TrackModel::default();
    let mut sim = TrackSim::new(model, seed);
    let steps = (0..steps)
        .map(|i| {
            let control = sim.autopilot();
            sim.step(control, !SENSOR_GAP.contains(&i))
        })
        .collect();
    TrackLog { model, steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trip() {
        let log = generate(3, 40);
        let back = TrackLog::parse(&log.to_text()).unwrap();
        assert_eq!(back.model, log.model);
        assert_eq!(back.len(), 40);
        for (a, b) in back.steps.iter().zip(&log.steps) {
            assert_eq!(a.control, b.control);
            assert_eq!(a.measured.is_some(), b.measured.is_some());
            assert!((a.truth - b.truth).abs() <= 0.05);
        }
        assert!(TrackLog::parse("S 20 x 3").is_err());
        assert_eq!(TrackLog::parse("# hi\nM 1 2 3\n").unwrap().len(), 0);
    }

    /// The generated run: up and down the track, sensor off in the gap.
    #[test]
    fn generated_runs_turn_around_and_have_a_gap() {
        let log = generate(1, DEFAULT_STEPS);
        assert_eq!(log.len(), DEFAULT_STEPS);
        assert!(
            log.steps
                .iter()
                .all(|s| (0.0..=TRACK_LENGTH).contains(&s.truth))
        );
        assert!(log.steps.iter().any(|s| s.control < 0.0), "never turned");
        for (i, s) in log.steps.iter().enumerate() {
            assert_eq!(s.measured.is_none(), SENSOR_GAP.contains(&i), "step {i}");
        }
        // Same seed, same run.
        assert_eq!(generate(1, 10), generate(1, 10));
    }
}
