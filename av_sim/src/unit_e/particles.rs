// Unit E, part 1: the particle filter (the course's slam_08_a..d).
//
// Unit C's histogram filter kept a probability for every cell. In 3D
// (x, y, heading) that's far too many cells. A particle filter keeps a few
// hundred guesses instead, each a full (x, y, heading) of the robot's
// centre. Where there are many particles, the robot is likely to be.
//
//   predict    move every particle with the motion model g (Unit D's),
//              each with its OWN random wheel noise: the cloud spreads
//   weigh      how well does what the scanner saw fit each particle?
//              The course's probability_of_measurement, multiplied over
//              the cylinders seen
//   resample   draw a new cloud from the old one, each particle picked in
//              proportion to its weight (the resampling wheel): good
//              guesses are copied, bad ones die out
//
// No Jacobians, no gaussians needed: the cloud can take any shape, even
// several blobs at once. That's what lets it start knowing NOTHING
// (particles all over the arena) and still find the robot.

use crate::rng::Rng;
use crate::unit_d::ekf::{EkfParams, ErrorEllipse, error_ellipse, g, h, wrap_pi};
use bevy::math::{DMat3, DVec2, DVec3};
use std::f64::consts::PI;

/// One guess of the robot's centre: (x, y, heading), heading in [-pi, pi).
pub type Particle = DVec3;

/// What one detected cylinder gives the filter: (range, bearing) from the
/// scanner, and its position in the scanner's frame (for matching).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sighting {
    pub measurement: DVec2,
    pub local: DVec2,
}

/// The prediction: move each particle with its own noisy controls (the
/// course's predict). The noise is the same as Unit D's control noise.
pub fn predict(particles: &mut [Particle], control: DVec2, p: &EkfParams, rng: &mut Rng) {
    let (l, r) = (control.x, control.y);
    let turn = p.control_turn_factor * (l - r);
    let sigma_l = ((p.control_motion_factor * l).powi(2) + turn * turn).sqrt();
    let sigma_r = ((p.control_motion_factor * r).powi(2) + turn * turn).sqrt();
    for particle in particles.iter_mut() {
        let noisy = DVec2::new(l + rng.gaussian(sigma_l), r + rng.gaussian(sigma_r));
        *particle = g(*particle, noisy, p.robot_width);
    }
}

/// The scanner's position for a particle.
fn scanner(particle: Particle, d: f64) -> (DVec2, f64) {
    let heading = particle.z;
    (
        DVec2::new(particle.x, particle.y) + DVec2::from_angle(heading) * d,
        heading,
    )
}

/// The course's assign_cylinders: each sighting, placed in the world with
/// this particle's pose, goes with the NEAREST landmark, however far away.
/// (With a particle in the wrong place, the pairs are wrong too, and the
/// measurements don't fit: that's what gives it a low weight.)
pub fn assign(
    particle: Particle,
    sightings: &[Sighting],
    landmarks: &[DVec2],
    d: f64,
) -> Vec<usize> {
    let (at, heading) = scanner(particle, d);
    sightings
        .iter()
        .filter_map(|s| {
            let world = at + DVec2::from_angle(heading).rotate(s.local);
            (0..landmarks.len()).min_by(|&a, &b| {
                world
                    .distance_squared(landmarks[a])
                    .total_cmp(&world.distance_squared(landmarks[b]))
            })
        })
        .collect()
}

/// The log of a particle's weight: the course's probability_of_measurement
/// (a gaussian in range times a gaussian in bearing) multiplied over all
/// sightings, as a sum of logs.
///
/// NOTE: the course multiplies the pdfs themselves. Far from the truth,
/// with several cylinders and a sharp sensor, those products can underflow
/// to exactly 0 for EVERY particle. The resampling wheel then never moves
/// (all weights 0) and picks the same particle N times: the whole cloud
/// collapses onto one guess. Measured (mod.rs, test `underflow`, uniform
/// start, seeds 1-5): never with the course's noise, once in 1385 steps
/// with av04's sharp scanner (20 mm, 1 degree). Rare, but fatal when it
/// happens; logs can't underflow.
/// The pdfs' constant factors are left out: they're the same for every
/// particle, and normalising removes them anyway.
pub fn log_weight(
    particle: Particle,
    sightings: &[Sighting],
    landmarks: &[DVec2],
    p: &EkfParams,
) -> f64 {
    let d = p.scanner_displacement;
    assign(particle, sightings, landmarks, d)
        .into_iter()
        .zip(sightings)
        .map(|(l, s)| {
            let expected = h(particle, landmarks[l], d);
            let dr = (s.measurement.x - expected.x) / p.measurement_distance_stddev;
            let da = wrap_pi(s.measurement.y - expected.y) / p.measurement_angle_stddev;
            -0.5 * (dr * dr + da * da)
        })
        .sum()
}

/// Weights that add up to 1, from log weights (subtract the largest
/// first, so the best particle gets exp(0) = 1 and nothing overflows).
pub fn normalise(log_weights: &[f64]) -> Vec<f64> {
    let top = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = log_weights.iter().map(|&lw| (lw - top).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.into_iter().map(|x| x / sum).collect()
}

/// The resampling wheel (the course's resample): walk round a wheel on
/// which each particle has a slice as wide as its weight, in random steps
/// of up to twice the largest weight. Wide slices get landed on more.
/// Generic over what a particle is: Unit G resamples whole particles
/// (pose + map), cloned.
pub fn resample<T: Clone>(particles: &[T], weights: &[f64], rng: &mut Rng) -> Vec<T> {
    let n = particles.len();
    if n == 0 {
        return Vec::new();
    }
    let top = weights.iter().copied().fold(0.0, f64::max);
    let mut index = ((rng.unit() * n as f64) as usize).min(n - 1);
    let mut offset = 0.0;
    (0..n)
        .map(|_| {
            offset += rng.unit() * 2.0 * top;
            while offset > weights[index] {
                offset -= weights[index];
                index = (index + 1) % n;
            }
            particles[index].clone()
        })
        .collect()
}

/// How many particles really count: 1 / sum(w^2). All equal: n. All the
/// weight on one: 1.
pub fn effective_count(weights: &[f64]) -> f64 {
    1.0 / weights.iter().map(|w| w * w).sum::<f64>()
}

/// The mean pose (the course's get_mean): positions averaged, headings
/// averaged as unit vectors, so 179 and -179 degrees average to 180, not 0.
///
/// NOTE: with two blobs, this is between them, where there's no particle
/// at all. The cloud is the answer; the mean is a summary.
pub fn mean(particles: &[Particle]) -> Particle {
    let n = particles.len().max(1) as f64;
    let (mut x, mut y, mut c, mut s) = (0.0, 0.0, 0.0, 0.0);
    for p in particles {
        x += p.x;
        y += p.y;
        c += p.z.cos();
        s += p.z.sin();
    }
    DVec3::new(x / n, y / n, s.atan2(c))
}

/// The spread around the mean (the course's
/// get_error_ellipse_and_heading_variance): the position's error ellipse
/// and the heading's standard deviation.
pub fn spread(particles: &[Particle], centre: Particle) -> (ErrorEllipse, f64) {
    let n = particles.len();
    if n < 2 {
        return (
            ErrorEllipse {
                angle: 0.0,
                major: 0.0,
                minor: 0.0,
            },
            0.0,
        );
    }
    let (mut sxx, mut sxy, mut syy, mut shh) = (0.0, 0.0, 0.0, 0.0);
    for p in particles {
        let (dx, dy) = (p.x - centre.x, p.y - centre.y);
        let dh = wrap_pi(p.z - centre.z);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        shh += dh * dh;
    }
    let k = 1.0 / (n - 1) as f64;
    let cov = DMat3::from_cols(
        DVec3::new(sxx * k, sxy * k, 0.0),
        DVec3::new(sxy * k, syy * k, 0.0),
        DVec3::Z,
    );
    (error_ellipse(&cov), (shh * k).sqrt())
}

/// Particles around a known pose (slam_08_a..c).
pub fn around(centre: Particle, sigma: DVec3, n: usize, rng: &mut Rng) -> Vec<Particle> {
    (0..n)
        .map(|_| {
            DVec3::new(
                centre.x + rng.gaussian(sigma.x),
                centre.y + rng.gaussian(sigma.y),
                wrap_pi(centre.z + rng.gaussian(sigma.z)),
            )
        })
        .collect()
}

/// Particles anywhere in a square arena, any heading (slam_08_d).
pub fn anywhere(size: f64, n: usize, rng: &mut Rng) -> Vec<Particle> {
    (0..n)
        .map(|_| {
            DVec3::new(
                rng.unit() * size,
                rng.unit() * size,
                (rng.unit() * 2.0 - 1.0) * PI,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wheel picks each particle about as often as its weight says,
    /// and never one with weight 0.
    #[test]
    fn the_wheel_follows_the_weights() {
        let particles: Vec<Particle> = (0..4).map(|i| DVec3::splat(i as f64)).collect();
        // The course's resampling_wheel.py example.
        let weights = [0.3, 0.0, 0.4, 0.3];
        let mut rng = Rng::new(5);
        let mut counts = [0usize; 4];
        for _ in 0..2000 {
            for p in resample(&particles, &weights, &mut rng) {
                counts[p.x as usize] += 1;
            }
        }
        let total = 8000.0;
        assert_eq!(counts[1], 0);
        for (i, &w) in weights.iter().enumerate() {
            let share = counts[i] as f64 / total;
            assert!((share - w).abs() < 0.02, "{i}: {share} vs {w}");
        }
    }

    /// The course's version, with every weight 0 (all underflowed): the
    /// wheel never turns, and every new particle is the same one.
    #[test]
    fn all_zero_weights_collapse_the_course_version() {
        let particles: Vec<Particle> = (0..50).map(|i| DVec3::splat(i as f64)).collect();
        let picked = resample(&particles, &[0.0; 50], &mut Rng::new(1));
        assert!(picked.iter().all(|p| *p == picked[0]));
        // With log weights there's always a best particle with weight 1
        // before normalising, however small the probabilities were.
        let w = normalise(&[-2000.0, -2001.0, -5000.0]);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w[0] > w[1] && w[1] > 0.0 && w[2] == 0.0);
        // Plain products: e^-2000 is 0 in f64.
        assert_eq!((-2000.0_f64).exp(), 0.0);
    }

    #[test]
    fn mean_heading_wraps_and_spread_measures() {
        let ps = [
            DVec3::new(0.0, 0.0, PI - 0.1),
            DVec3::new(10.0, 0.0, -PI + 0.1),
        ];
        let m = mean(&ps);
        assert!((m.z.abs() - PI).abs() < 1e-9, "{m}");
        assert_eq!((m.x, m.y), (5.0, 0.0));
        let (e, heading) = spread(&ps, m);
        // Two points 10 apart: variance 2 * 25 / (2 - 1) = 50 along x.
        assert!((e.major - 50.0_f64.sqrt()).abs() < 1e-9 && e.minor.abs() < 1e-9);
        assert!((heading - 0.02_f64.sqrt()).abs() < 1e-9);
    }

    /// The particle at the truth gets the highest weight.
    #[test]
    fn the_truth_weighs_most() {
        let p = EkfParams::default();
        let landmarks = [
            DVec2::new(500.0, 500.0),
            DVec2::new(1500.0, 800.0),
            DVec2::new(900.0, 1600.0),
        ];
        let truth = DVec3::new(1000.0, 1000.0, 0.3);
        let (at, heading) = scanner(truth, p.scanner_displacement);
        let sightings: Vec<Sighting> = landmarks
            .iter()
            .map(|&l| Sighting {
                measurement: h(truth, l, p.scanner_displacement),
                local: DVec2::from_angle(-heading).rotate(l - at),
            })
            .collect();
        let candidates = [
            truth,
            truth + DVec3::new(80.0, 0.0, 0.0),
            truth + DVec3::new(0.0, 0.0, 0.3),
            DVec3::new(300.0, 1700.0, -2.0),
        ];
        let lw: Vec<f64> = candidates
            .iter()
            .map(|&c| log_weight(c, &sightings, &landmarks, &p))
            .collect();
        assert_eq!(lw[0], 0.0);
        assert!(lw[1..].iter().all(|&w| w < 0.0), "{lw:?}");
    }

    /// Predicting spreads the cloud; the spread grows with distance driven.
    #[test]
    fn predicting_spreads_the_cloud() {
        let p = EkfParams::default();
        let mut rng = Rng::new(3);
        let mut cloud = vec![DVec3::new(1000.0, 1000.0, 0.0); 400];
        predict(&mut cloud, DVec2::new(50.0, 50.0), &p, &mut rng);
        let (short, _) = spread(&cloud, mean(&cloud));
        for _ in 0..4 {
            predict(&mut cloud, DVec2::new(50.0, 50.0), &p, &mut rng);
        }
        let (long, _) = spread(&cloud, mean(&cloud));
        assert!(long.major > 1.5 * short.major, "{short:?} {long:?}");
        // On average it went 250 mm along x.
        assert!((mean(&cloud).x - 1250.0).abs() < 15.0);
    }
}
