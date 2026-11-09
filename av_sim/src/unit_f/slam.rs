// Unit F, part 1: EKF SLAM (the course's slam_09_a..c).
//
// Unit D's Kalman filter needed the map. Here the robot builds it: the
// state is the robot (x, y, heading) FOLLOWED BY every landmark found so
// far (x, y), and the covariance covers all of them together:
//
//   state       [ x y heading | x0 y0 | x1 y1 | ... ]      3 + 2N numbers
//   covariance  (3 + 2N) x (3 + 2N), robot block top left
//
//   predict       only the robot moves: G is Unit D's G3 in the top left
//                 and the identity elsewhere, R only touches the robot
//   add landmark  the state grows by (x, y), placed with the current
//                 pose; its variance is huge (1e10, as in the course),
//                 meaning "no idea yet" - the next correction fixes it
//   correct       Unit D's correction with a wider H: one measurement
//                 depends on the robot (H3) and on ONE landmark (-H3's
//                 x and y columns: moving the landmark is like moving the
//                 robot the other way)
//
// The off-diagonal blocks are what makes it SLAM: seeing one landmark
// again also moves the others, because their errors are tied to the
// robot's errors when they were first seen.
//
// SIZES. The matrices grow, so glam's fixed 3x3 isn't enough, and a
// general matrix product would cost (3 + 2N)^3 per step. But H has only
// five non-zero columns, so everything here is written out for that
// shape: a correction costs about (3 + 2N)^2. No matrix crate needed.

use crate::unit_d::ekf::{
    EkfParams, ErrorEllipse, dg_dcontrol, dg_dstate, dh_dstate, error_ellipse, g, h, wrap_pi,
};
use bevy::math::{DMat2, DMat3, DVec2, DVec3};

/// A square matrix of f64, row-major, that can grow.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    n: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(n: usize) -> Matrix {
        Matrix {
            n,
            data: vec![0.0; n * n],
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] = v;
    }

    /// Two more rows and columns, zero except `diagonal` on the new diagonal.
    pub fn grow2(&mut self, diagonal: f64) {
        let old = self.n;
        let mut bigger = Matrix::zeros(old + 2);
        for r in 0..old {
            bigger.data[r * (old + 2)..r * (old + 2) + old]
                .copy_from_slice(&self.data[r * old..(r + 1) * old]);
        }
        bigger.set(old, old, diagonal);
        bigger.set(old + 1, old + 1, diagonal);
        *self = bigger;
    }

    /// The 3x3 block at (i, i).
    pub fn block3(&self, i: usize) -> DMat3 {
        let col = |c: usize| {
            DVec3::new(
                self.get(i, i + c),
                self.get(i + 1, i + c),
                self.get(i + 2, i + c),
            )
        };
        DMat3::from_cols(col(0), col(1), col(2))
    }

    /// The 2x2 block at (i, i), as a 3x3 with a zero third row and column
    /// (for error_ellipse).
    pub fn block2(&self, i: usize) -> DMat3 {
        DMat3::from_cols(
            DVec3::new(self.get(i, i), self.get(i + 1, i), 0.0),
            DVec3::new(self.get(i, i + 1), self.get(i + 1, i + 1), 0.0),
            DVec3::ZERO,
        )
    }

    /// The largest |a[r][c] - a[c][r]|: a covariance must be symmetric.
    pub fn asymmetry(&self) -> f64 {
        let mut worst: f64 = 0.0;
        for r in 0..self.n {
            for c in 0..r {
                worst = worst.max((self.get(r, c) - self.get(c, r)).abs());
            }
        }
        worst
    }
}

/// The course's variance for a landmark seen for the first time (mm^2).
pub const NEW_LANDMARK_VARIANCE: f64 = 1e10;

#[derive(Clone, Debug, PartialEq)]
pub struct Slam {
    /// Robot (3) then landmarks (2 each).
    pub state: Vec<f64>,
    pub covariance: Matrix,
}

impl Slam {
    /// A robot with no map yet. With zero covariance, the start pose
    /// defines the map's frame (the course starts anywhere, e.g. (500, 0,
    /// 45 degrees): SLAM has no absolute frame of its own).
    pub fn new(robot: DVec3, covariance: DMat3) -> Slam {
        let mut c = Matrix::zeros(3);
        for r in 0..3 {
            for k in 0..3 {
                c.set(r, k, covariance.col(k)[r]);
            }
        }
        Slam {
            state: vec![robot.x, robot.y, robot.z],
            covariance: c,
        }
    }

    pub fn robot(&self) -> DVec3 {
        DVec3::new(self.state[0], self.state[1], self.state[2])
    }

    pub fn landmark_count(&self) -> usize {
        (self.state.len() - 3) / 2
    }

    pub fn landmark(&self, i: usize) -> DVec2 {
        DVec2::new(self.state[3 + 2 * i], self.state[4 + 2 * i])
    }

    pub fn robot_ellipse(&self) -> ErrorEllipse {
        error_ellipse(&self.covariance.block3(0))
    }

    pub fn heading_stddev(&self) -> f64 {
        self.covariance.get(2, 2).max(0.0).sqrt()
    }

    pub fn landmark_ellipse(&self, i: usize) -> ErrorEllipse {
        error_ellipse(&self.covariance.block2(3 + 2 * i))
    }

    /// The course's predict, block by block:
    ///   robot-robot        G3 * S_rr * G3^T + R3
    ///   robot-landmarks    G3 * S_rm (and its mirror image)
    ///   landmarks          unchanged
    pub fn predict(&mut self, control: DVec2, p: &EkfParams) {
        let robot = self.robot();
        let (l, r) = (control.x, control.y);
        let turn = p.control_turn_factor * (l - r);
        let var_l = (p.control_motion_factor * l).powi(2) + turn * turn;
        let var_r = (p.control_motion_factor * r).powi(2) + turn * turn;
        let v = dg_dcontrol(robot, control, p.robot_width);
        let r3 = v * DMat3::from_diagonal(DVec3::new(var_l, var_r, 0.0)) * v.transpose();
        let g3 = dg_dstate(robot, control, p.robot_width);

        let rr = g3 * self.covariance.block3(0) * g3.transpose() + r3;
        for row in 0..3 {
            for col in 0..3 {
                self.covariance.set(row, col, rr.col(col)[row]);
            }
        }
        let n = self.covariance.size();
        for col in 3..n {
            let column = DVec3::new(
                self.covariance.get(0, col),
                self.covariance.get(1, col),
                self.covariance.get(2, col),
            );
            let moved = g3 * column;
            for row in 0..3 {
                self.covariance.set(row, col, moved[row]);
                self.covariance.set(col, row, moved[row]);
            }
        }
        let new = g(robot, control, p.robot_width);
        self.state[..3].copy_from_slice(&[new.x, new.y, new.z]);
    }

    /// The course's add_landmark_to_state. Returns the new landmark's index.
    pub fn add_landmark(&mut self, at: DVec2) -> usize {
        self.state.extend([at.x, at.y]);
        self.covariance.grow2(NEW_LANDMARK_VARIANCE);
        self.landmark_count() - 1
    }

    /// The course's correct, for landmark `i`. H is 2 x (3 + 2N) with five
    /// non-zero columns: 0, 1, 2 (robot) and a, a + 1 (the landmark).
    /// Returns the innovation.
    pub fn correct(&mut self, measurement: DVec2, i: usize, p: &EkfParams) -> DVec2 {
        let d = p.scanner_displacement;
        let landmark = self.landmark(i);
        let h3 = dh_dstate(self.robot(), landmark, d);
        let a = 3 + 2 * i;
        // The non-zero columns of H, and H's two rows in those columns.
        let cols = [0, 1, 2, a, a + 1];
        let row = |k: usize| {
            let (x, y, t) = (h3.col(0)[k], h3.col(1)[k], h3.col(2)[k]);
            [x, y, t, -x, -y]
        };
        let hk = [row(0), row(1)];

        // H * S: 2 x n.
        let n = self.covariance.size();
        let mut hs = [vec![0.0; n], vec![0.0; n]];
        for (row, h_row) in hs.iter_mut().zip(hk) {
            for (c, out) in row.iter_mut().enumerate() {
                *out = cols
                    .iter()
                    .zip(h_row)
                    .map(|(&j, hj)| hj * self.covariance.get(j, c))
                    .sum();
            }
        }
        // S = H * S * H^T + Q: 2 x 2.
        let s_entry = |k: usize, l: usize| -> f64 {
            cols.iter().zip(hk[l]).map(|(&j, hj)| hs[k][j] * hj).sum()
        };
        let s = DMat2::from_cols(
            DVec2::new(
                s_entry(0, 0) + p.measurement_distance_stddev.powi(2),
                s_entry(1, 0),
            ),
            DVec2::new(
                s_entry(0, 1),
                s_entry(1, 1) + p.measurement_angle_stddev.powi(2),
            ),
        );
        let s_inv = s.inverse();
        // K = S_cov * H^T * S^-1 = (H * S_cov)^T * S^-1 (the covariance is
        // symmetric): n x 2.
        let k_gain: Vec<DVec2> = (0..n)
            .map(|r| s_inv.transpose() * DVec2::new(hs[0][r], hs[1][r]))
            .collect();

        let mut innovation = measurement - h(self.robot(), landmark, d);
        innovation.y = wrap_pi(innovation.y);
        for (r, k) in k_gain.iter().enumerate() {
            self.state[r] += k.dot(innovation);
        }
        // NOTE: the course doesn't wrap the heading here.
        self.state[2] = wrap_pi(self.state[2]);
        // The course: S_cov = (I - K H) S_cov = S_cov - K * (H * S_cov).
        // Here: S_cov = S_cov - K * S * K^T. The same thing on paper
        // (K * S = S_cov * H^T, so K * S * K^T = S_cov * H^T * K^T =
        // (K * H * S_cov)^T = K * H * S_cov, the covariance being
        // symmetric), but symmetric by construction.
        //
        // NOTE: why it matters. The course's form isn't exactly symmetric
        // after rounding, and the next steps feed that back in. With the
        // course's huge measurement noise the gain is small and it never
        // grows. With a sharp scanner (20 mm, 1 degree), on seed 1, the
        // asymmetry went 1e-6 (step 0), 1e-2 (step 60), 1e3 (step 133);
        // at step 133 a variance went negative and the filter fell apart
        // (100+ phantom landmarks, errors in metres). The test
        // covariance_stays_healthy (mod.rs) runs that case.
        // Only the lower triangle is computed, and mirrored: kr . (S kc)
        // and kc . (S kr) are equal on paper, but not always to the last
        // bit, and with variances of 1e10 that last bit is 1e-6.
        let ks: Vec<DVec2> = k_gain.iter().map(|k| s * *k).collect();
        for (r, kr) in k_gain.iter().enumerate() {
            for (c, kc) in ks.iter().enumerate().take(r + 1) {
                let v = self.covariance.get(r, c) - kr.dot(*kc);
                self.covariance.set(r, c, v);
                self.covariance.set(c, r, v);
            }
        }
        innovation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> EkfParams {
        EkfParams {
            measurement_distance_stddev: 20.0,
            measurement_angle_stddev: 1.0_f64.to_radians(),
            ..EkfParams::default()
        }
    }

    fn start() -> Slam {
        Slam::new(
            DVec3::new(1000.0, 1000.0, 0.5),
            DMat3::from_diagonal(DVec3::new(25.0, 25.0, 0.001)),
        )
    }

    /// With no landmarks, SLAM's predict is Unit D's predict.
    #[test]
    fn predict_matches_unit_d() {
        let p = params();
        let control = DVec2::new(40.0, 55.0);
        let mut slam = start();
        slam.predict(control, &p);
        let mut ekf = crate::unit_d::ekf::Ekf {
            state: DVec3::new(1000.0, 1000.0, 0.5),
            covariance: DMat3::from_diagonal(DVec3::new(25.0, 25.0, 0.001)),
        };
        ekf.predict(control, &p);
        assert!((slam.robot() - ekf.state).length() < 1e-9);
        assert!(
            (slam.covariance.block3(0) - ekf.covariance)
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-9)
        );
    }

    /// With the landmark known exactly (variance 0), SLAM's correct is
    /// Unit D's correct: the full H, written out the short way, agrees.
    #[test]
    fn correct_with_a_certain_landmark_matches_unit_d() {
        let p = params();
        let landmark = DVec2::new(1600.0, 1300.0);
        let mut slam = start();
        let i = slam.add_landmark(landmark);
        slam.covariance.set(3, 3, 0.0);
        slam.covariance.set(4, 4, 0.0);
        let z = DVec2::new(650.0, 0.1);
        slam.correct(z, i, &p);
        let mut ekf = crate::unit_d::ekf::Ekf {
            state: DVec3::new(1000.0, 1000.0, 0.5),
            covariance: DMat3::from_diagonal(DVec3::new(25.0, 25.0, 0.001)),
        };
        ekf.correct(z, landmark, &p);
        assert!(
            (slam.robot() - ekf.state).length() < 1e-9,
            "{} vs {}",
            slam.robot(),
            ekf.state
        );
        assert!(
            (slam.covariance.block3(0) - ekf.covariance)
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-9)
        );
        assert_eq!(slam.landmark(i), landmark);
    }

    /// A new landmark: placed roughly, then the first correction pulls it
    /// to where the measurement says and its huge variance collapses.
    #[test]
    fn a_new_landmark_settles_after_one_correction() {
        let p = params();
        let mut slam = start();
        let truth = DVec2::new(1500.0, 1400.0);
        let z = h(slam.robot(), truth, p.scanner_displacement);
        let i = slam.add_landmark(truth + DVec2::new(40.0, -30.0));
        slam.correct(z, i, &p);
        // Not exactly: one linearised step from 50 mm off leaves a few mm.
        assert!(
            slam.landmark(i).distance(truth) < 5.0,
            "{}",
            slam.landmark(i)
        );
        let e = slam.landmark_ellipse(i);
        // Now about as sure as the robot plus one measurement: tens of mm.
        assert!(e.major < 50.0 && e.major > 5.0, "{e:?}");
        // The robot hardly moved: a brand-new landmark can't correct it.
        assert!((slam.robot() - start().robot()).length() < 1e-3);
        assert!(slam.covariance.asymmetry() < 1e-6);
    }

    /// Loop closure: map a, drive (the estimate drifts), map b for the
    /// first time (so b inherits the drift), see a again. The robot snaps
    /// back, and b, through its correlation with the robot, moves towards
    /// where it really is, without being seen again.
    ///
    /// (Mapping a and b from the same pose and THEN drifting wouldn't move
    /// b: their errors are tied to each other, and the drift says nothing
    /// about them. That was this test's first version, and it was wrong.)
    #[test]
    fn loop_closure_moves_the_other_landmarks() {
        let p = params();
        let mut slam = start();
        let a = DVec2::new(1500.0, 1400.0);
        let b = DVec2::new(600.0, 1500.0);
        let see = |robot: DVec3, l: DVec2| h(robot, l, p.scanner_displacement);
        // The robot really stands still; its estimate drives on.
        let truth = slam.robot();
        let z = see(truth, a);
        let ia = slam.add_landmark(a);
        slam.correct(z, ia, &p);
        for _ in 0..5 {
            slam.predict(DVec2::new(30.0, 30.0), &p);
        }
        // b, first seen now, is placed with the drifted pose.
        let z = see(truth, b);
        let seen_from = slam.robot();
        let placed = {
            let (r, bearing) = (z.x, z.y);
            let scanner =
                seen_from.truncate() + DVec2::from_angle(seen_from.z) * p.scanner_displacement;
            scanner + DVec2::from_angle(seen_from.z + bearing) * r
        };
        let ib = slam.add_landmark(placed);
        slam.correct(z, ib, &p);
        let b_error_before = slam.landmark(ib).distance(b);
        let robot_error_before = (slam.robot() - truth).truncate().length();
        assert!(b_error_before > 50.0, "{b_error_before}");

        // See a again.
        let z = see(truth, a);
        slam.correct(z, ia, &p);
        let robot_error_after = (slam.robot() - truth).truncate().length();
        let b_error_after = slam.landmark(ib).distance(b);
        assert!(
            robot_error_after < robot_error_before,
            "{robot_error_before} -> {robot_error_after}"
        );
        // A good part of the way (150 -> 94 mm here): one landmark only
        // partly corrects the robot, and b follows the robot's correction.
        assert!(
            b_error_after < 0.8 * b_error_before,
            "{b_error_before} -> {b_error_after}"
        );
    }

    #[test]
    fn growing_keeps_the_old_entries() {
        let mut m = Matrix::zeros(3);
        m.set(1, 2, 7.0);
        m.grow2(5.0);
        assert_eq!(m.size(), 5);
        assert_eq!(
            (m.get(1, 2), m.get(3, 3), m.get(4, 4), m.get(3, 4)),
            (7.0, 5.0, 5.0, 0.0)
        );
    }
}
