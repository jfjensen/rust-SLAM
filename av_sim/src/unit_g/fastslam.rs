// Unit G, part 1: FastSLAM (the course's slam_10_a..f).
//
// Unit F kept ONE robot and ONE big covariance over the robot and every
// landmark: (3 + 2N)^2 numbers, and every correction touches all of them.
// FastSLAM splits it up:
//
//   - a particle filter for the robot's pose (Unit E), and
//   - in EVERY particle, its own map: each landmark a tiny Kalman filter,
//     a 2D position and a 2x2 covariance.
//
// Why that works: if the robot's path were known exactly, the landmarks'
// errors wouldn't be tied to each other any more (in Unit F they are
// tied only THROUGH the robot's uncertainty). Each particle claims to know
// the path, so in each particle the landmarks are independent: N little
// 2x2 filters instead of one big one.
//
// Per step and particle:
//   predict    move the pose with its own random wheel noise (Unit E)
//   per detection:
//     match    the likelihood of it belonging to each of this particle's
//              landmarks (from the landmark's own covariance); the best
//              one if it's likely enough, else a NEW landmark
//     update   that landmark's little Kalman filter
//     weight   times the likelihood (a new landmark: the threshold)
//   counters   +2 when a landmark is seen, -1 when it should have been
//              (in the scanner's field of view); below 0, it's removed.
//              Spurious detections die out.
//   resample   by weight (Unit E's wheel), whole particles with their maps

use crate::robot::{BEAM_COUNT, beam_index_to_angle};
use crate::unit_d::ekf::{EkfParams, g, h, wrap_pi};
use crate::unit_e::particles::Sighting;
use bevy::math::{DMat2, DVec2, DVec3};
use std::f64::consts::TAU;

/// One landmark in one particle's map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmark {
    pub position: DVec2,
    pub covariance: DMat2,
    /// +2 seen, -1 should have been seen; removed below 0.
    pub counter: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FastParticle {
    /// The robot's centre: x, y, heading.
    pub pose: DVec3,
    pub landmarks: Vec<Landmark>,
}

/// The measurement noise as a covariance (the course's Qt).
pub fn measurement_covariance(p: &EkfParams) -> DMat2 {
    DMat2::from_diagonal(DVec2::new(
        p.measurement_distance_stddev.powi(2),
        p.measurement_angle_stddev.powi(2),
    ))
}

/// dh/dlandmark: how (range, bearing) change when the LANDMARK moves (the
/// course's dh_dlandmark; it's -dh/d(x, y) of Unit D's dh_dstate).
pub fn dh_dlandmark(pose: DVec3, landmark: DVec2, d: f64) -> DMat2 {
    let theta = pose.z;
    let dx = landmark.x - (pose.x + d * theta.cos());
    let dy = landmark.y - (pose.y + d * theta.sin());
    let q = dx * dx + dy * dy;
    let sq = q.sqrt();
    // Columns: d/dmx, d/dmy. Rows: range, bearing.
    DMat2::from_cols(DVec2::new(dx / sq, -dy / q), DVec2::new(dy / sq, dx / q))
}

/// The bearings the scanner covers (the course's min_max_bearing).
pub fn field_of_view() -> (f64, f64) {
    (
        beam_index_to_angle(0.0) as f64,
        beam_index_to_angle((BEAM_COUNT - 1) as f64 as f32) as f64,
    )
}

impl FastParticle {
    pub fn new(pose: DVec3) -> FastParticle {
        FastParticle {
            pose,
            landmarks: Vec::new(),
        }
    }

    /// The scanner's world position and heading.
    fn scanner(&self, d: f64) -> (DVec2, f64) {
        (
            self.pose.truncate() + DVec2::from_angle(self.pose.z) * d,
            self.pose.z,
        )
    }

    pub fn move_by(&mut self, control: DVec2, w: f64) {
        self.pose = g(self.pose, control, w);
    }

    /// The landmark's measurement covariance and the Jacobian, for landmark i.
    fn jacobian_and_ql(&self, i: usize, qt: DMat2, d: f64) -> (DMat2, DMat2) {
        let l = &self.landmarks[i];
        let hm = dh_dlandmark(self.pose, l.position, d);
        (hm, hm * l.covariance * hm.transpose() + qt)
    }

    /// The log of the likelihood that `z` came from landmark i (the
    /// course's wl_likelihood_of_correspondence, as a log).
    pub fn log_likelihood(&self, z: DVec2, i: usize, qt: DMat2, d: f64) -> f64 {
        let (_, ql) = self.jacobian_and_ql(i, qt, d);
        let mut dz = z - h(self.pose, self.landmarks[i].position, d);
        dz.y = wrap_pi(dz.y);
        -0.5 * dz.dot(ql.inverse() * dz) - (TAU * ql.determinant().sqrt()).ln()
    }

    /// A new landmark where the detection says, with the measurement's
    /// uncertainty carried into x, y: H^-1 * Qt * H^-T (the course's
    /// initialize_new_landmark).
    ///
    /// NOTE: the course computes H with the SCANNER pose and the scanner
    /// displacement, shifting the scanner forward twice (60 mm instead of
    /// 30). Here H is computed from the robot's pose, like everywhere else.
    pub fn add_landmark(&mut self, sighting: &Sighting, qt: DMat2, d: f64) {
        let (at, heading) = self.scanner(d);
        let position = at + DVec2::from_angle(heading).rotate(sighting.local);
        let h_inv = dh_dlandmark(self.pose, position, d).inverse();
        self.landmarks.push(Landmark {
            position,
            covariance: h_inv * qt * h_inv.transpose(),
            counter: 1,
        });
    }

    /// The landmark's little Kalman filter (the course's update_landmark).
    /// NOTE: covariance as S - K Ql K^T, symmetric by construction (see
    /// Unit F's correct); the course uses (I - K H) S.
    pub fn update_landmark(&mut self, i: usize, z: DVec2, qt: DMat2, d: f64) {
        let (hm, ql) = self.jacobian_and_ql(i, qt, d);
        let l = &mut self.landmarks[i];
        let k = l.covariance * hm.transpose() * ql.inverse();
        let mut dz = z - h(self.pose, l.position, d);
        dz.y = wrap_pi(dz.y);
        l.position += k * dz;
        l.covariance -= k * ql * k.transpose();
    }

    /// One detection (the course's update_particle): match, update or add.
    /// Returns the log weight it contributes.
    pub fn observe(
        &mut self,
        s: &Sighting,
        candidates: usize,
        min_log_likelihood: f64,
        qt: DMat2,
        d: f64,
    ) -> f64 {
        let best = (0..candidates)
            .map(|i| (i, self.log_likelihood(s.measurement, i, qt, d)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((i, ll)) if ll >= min_log_likelihood => {
                self.update_landmark(i, s.measurement, qt, d);
                self.landmarks[i].counter += 2;
                ll
            }
            _ => {
                self.add_landmark(s, qt, d);
                min_log_likelihood
            }
        }
    }

    /// -1 for every landmark the scanner should be seeing now (only the
    /// bearing is checked, as in the course).
    pub fn decrement_visible(&mut self, d: f64) {
        let (lo, hi) = field_of_view();
        let pose = self.pose;
        for l in &mut self.landmarks {
            let bearing = h(pose, l.position, d).y;
            if (lo..=hi).contains(&bearing) {
                l.counter -= 1;
            }
        }
    }

    pub fn remove_spurious(&mut self) {
        self.landmarks.retain(|l| l.counter >= 0);
    }

    /// The whole correction for one particle (the course's
    /// update_and_compute_weights, for one particle): returns its log
    /// weight, and how many detections started a new landmark.
    pub fn correct(
        &mut self,
        sightings: &[Sighting],
        min_log_likelihood: f64,
        p: &EkfParams,
    ) -> (f64, usize) {
        let d = p.scanner_displacement;
        let qt = measurement_covariance(p);
        self.decrement_visible(d);
        // Detections can only match landmarks that were there BEFORE this
        // scan (as in the course): two detections can't both make and
        // then match the same new landmark.
        let candidates = self.landmarks.len();
        let log_weight = sightings
            .iter()
            .map(|s| self.observe(s, candidates, min_log_likelihood, qt, d))
            .sum();
        let new = self.landmarks.len() - candidates;
        self.remove_spurious();
        (log_weight, new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: f64 = 30.0;

    fn sharp() -> EkfParams {
        EkfParams {
            measurement_distance_stddev: 20.0,
            measurement_angle_stddev: 1.0_f64.to_radians(),
            ..EkfParams::default()
        }
    }

    /// A sighting of a world point from a particle's pose.
    fn sighting(p: &FastParticle, world: DVec2) -> Sighting {
        let (at, heading) = p.scanner(D);
        Sighting {
            measurement: h(p.pose, world, D),
            local: DVec2::from_angle(-heading).rotate(world - at),
        }
    }

    #[test]
    fn dh_dlandmark_is_minus_dh_dstate() {
        let pose = DVec3::new(1000.0, 800.0, 0.6);
        let landmark = DVec2::new(1400.0, 1300.0);
        let full = crate::unit_d::ekf::dh_dstate(pose, landmark, D);
        let ours = dh_dlandmark(pose, landmark, D);
        for (row, col) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!((ours.col(col)[row] + full.col(col)[row]).abs() < 1e-12);
        }
    }

    /// A new landmark lands where the detection is, and its covariance is
    /// the measurement's: along the line of sight sigma = range noise,
    /// across it sigma = distance * angle noise.
    #[test]
    fn a_new_landmark_carries_the_measurement_noise() {
        let p = sharp();
        let mut particle = FastParticle::new(DVec3::new(0.0, 0.0, 0.0));
        let world = DVec2::new(1030.0, 0.0); // 1000 mm straight ahead of the scanner
        particle.add_landmark(&sighting(&particle, world), measurement_covariance(&p), D);
        let l = particle.landmarks[0];
        assert!(l.position.distance(world) < 1e-9);
        assert!(
            (l.covariance.col(0).x.sqrt() - 20.0).abs() < 1e-6,
            "{:?}",
            l.covariance
        );
        let across = 1000.0 * 1.0_f64.to_radians();
        assert!(
            (l.covariance.col(1).y.sqrt() - across).abs() < 1e-6,
            "{:?}",
            l.covariance
        );
    }

    /// Seen again: it matches (no new landmark), and gets surer.
    #[test]
    fn seeing_it_again_matches_and_narrows() {
        let p = sharp();
        let qt = measurement_covariance(&p);
        let min = (0.001_f64).ln();
        let mut particle = FastParticle::new(DVec3::new(500.0, 500.0, 0.3));
        let world = DVec2::new(1300.0, 900.0);
        particle.correct(&[sighting(&particle, world)], min, &p);
        let before = particle.landmarks[0].covariance.determinant();
        let (w, new) = particle.correct(&[sighting(&particle, world)], min, &p);
        assert_eq!(particle.landmarks.len(), 1);
        assert_eq!(new, 0);
        assert!(w > min);
        assert!(particle.landmarks[0].covariance.determinant() < before);
        // Born at 1, -1 for being in view, +2 for being seen.
        assert_eq!(particle.landmarks[0].counter, 2);
        // Something far from it: a second landmark.
        particle.correct(&[sighting(&particle, DVec2::new(1500.0, 300.0))], min, &p);
        assert_eq!(particle.landmarks.len(), 2);
        let _ = qt;
    }

    /// A spurious landmark (seen once, then in view but never seen again)
    /// is removed after its counter runs out: 1 at birth, -1 per step.
    #[test]
    fn spurious_landmarks_die_out() {
        let p = sharp();
        let min = (0.001_f64).ln();
        let mut particle = FastParticle::new(DVec3::new(500.0, 500.0, 0.0));
        particle.correct(&[sighting(&particle, DVec2::new(1200.0, 520.0))], min, &p);
        assert_eq!(particle.landmarks.len(), 1);
        particle.correct(&[], min, &p);
        assert_eq!(particle.landmarks.len(), 1, "counter 0: still there");
        particle.correct(&[], min, &p);
        assert!(particle.landmarks.is_empty(), "counter -1: removed");
        // Out of view (behind the robot): it isn't counted down.
        particle.correct(&[sighting(&particle, DVec2::new(1200.0, 520.0))], min, &p);
        particle.pose.z = std::f64::consts::PI;
        for _ in 0..5 {
            particle.correct(&[], min, &p);
        }
        assert_eq!(particle.landmarks.len(), 1);
    }
}
