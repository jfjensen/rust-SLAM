// Unit D, part 1: the extended Kalman filter for the robot's pose (the
// course's slam_07_a..f).
//
// Unit C's Kalman filter in 1D, now for the state (x, y, heading) of the
// robot's CENTRE, with its 3x3 covariance. Two things make it "extended":
//
//   - The motion g and the measurement h are not straight lines (they
//     have sines, cosines, atan2). The filter uses their slopes at the
//     current state instead, the Jacobians: G = dg/dstate, V = dg/dcontrol,
//     H = dh/dstate.
//   - The control noise is in wheel space (left, right), so it's carried
//     into state space with V:  R = V * diag(sigma_l^2, sigma_r^2) * V^T.
//
//   predict   covariance = G * covariance * G^T + R       state = g(state, control)
//   correct   K = covariance * H^T * (H * covariance * H^T + Q)^-1
//             state = state + K * (z - h(state))          covariance = (I - K * H) * covariance
//
// once per observed landmark, one after the other. z = (range, bearing)
// to a cylinder, as the scanner sees it.
//
// MATRICES. glam has 3x3 matrices but no 3x2 or 2x3 ones. So V and H are
// stored as 3x3, padded with zeros (V's third column, H's third row), and
// the measurement gets a dummy third component with variance 1 and
// innovation 0. The padding never mixes with the real numbers: S comes out
// block diagonal, its inverse too, and K's third column is zero. It is the
// course's arithmetic in 3x3 clothes. f64 throughout: the covariance holds
// mm^2 (around 10^4) next to rad^2 (around 10^-3).

use bevy::math::{DMat3, DVec2, DVec3};
use std::f64::consts::{PI, TAU};

/// The course's (a + pi) % (2 pi) - pi: into [-pi, pi).
pub fn wrap_pi(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

/// A 3x3 matrix written row by row (glam builds them column by column).
fn rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> DMat3 {
    DMat3::from_cols(DVec3::from(r0), DVec3::from(r1), DVec3::from(r2)).transpose()
}

/// The filter's constants (the course's slam_07_f, except where noted).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EkfParams {
    pub robot_width: f64,
    pub scanner_displacement: f64,
    /// Wheel noise: sigma = this * distance driven ...
    pub control_motion_factor: f64,
    /// ... plus this * (left - right): wheels slip more when turning.
    pub control_turn_factor: f64,
    pub measurement_distance_stddev: f64,
    pub measurement_angle_stddev: f64,
}

impl Default for EkfParams {
    fn default() -> EkfParams {
        EkfParams {
            // NOTE: av_sim's model robot (Unit A's 150); the course's script uses 155.
            robot_width: crate::robot::ROBOT_WIDTH as f64,
            scanner_displacement: crate::robot::SCANNER_DISPLACEMENT as f64,
            control_motion_factor: 0.35,
            control_turn_factor: 0.6,
            measurement_distance_stddev: 200.0,
            measurement_angle_stddev: 15.0_f64.to_radians(),
        }
    }
}

/// The motion model g: centre state + (left, right) in mm -> new state.
/// Unit A's filter_step, for the centre, without the scanner shift.
pub fn g(state: DVec3, control: DVec2, w: f64) -> DVec3 {
    let (x, y, theta) = (state.x, state.y, state.z);
    let (l, r) = (control.x, control.y);
    if r != l {
        let alpha = (r - l) / w;
        let rad = l / alpha;
        DVec3::new(
            x + (rad + w / 2.0) * ((theta + alpha).sin() - theta.sin()),
            y + (rad + w / 2.0) * (-(theta + alpha).cos() + theta.cos()),
            wrap_pi(theta + alpha),
        )
    } else {
        DVec3::new(x + l * theta.cos(), y + l * theta.sin(), theta)
    }
}

/// G = dg/dstate. Only the heading column is interesting: turning the
/// robot a little swings where the step ends.
pub fn dg_dstate(state: DVec3, control: DVec2, w: f64) -> DMat3 {
    let theta = state.z;
    let (l, r) = (control.x, control.y);
    let (a, b) = if r != l {
        let alpha = (r - l) / w;
        let rad = l / alpha;
        (
            (rad + w / 2.0) * ((theta + alpha).cos() - theta.cos()),
            (rad + w / 2.0) * ((theta + alpha).sin() - theta.sin()),
        )
    } else {
        (-l * theta.sin(), l * theta.cos())
    };
    rows([1.0, 0.0, a], [0.0, 1.0, b], [0.0, 0.0, 1.0])
}

/// V = dg/dcontrol, 3x2 (left, right), padded to 3x3 with a zero column.
pub fn dg_dcontrol(state: DVec3, control: DVec2, w: f64) -> DMat3 {
    let theta = state.z;
    let (l, r) = (control.x, control.y);
    let (left, right) = if r != l {
        let alpha = (r - l) / w;
        let t = theta + alpha;
        let wr = w * r / ((r - l) * (r - l));
        let wl = w * l / ((r - l) * (r - l));
        let r2l = (r + l) / (2.0 * (r - l));
        (
            DVec3::new(
                wr * (t.sin() - theta.sin()) - r2l * t.cos(),
                wr * (-t.cos() + theta.cos()) - r2l * t.sin(),
                -1.0 / w,
            ),
            DVec3::new(
                -wl * (t.sin() - theta.sin()) + r2l * t.cos(),
                -wl * (-t.cos() + theta.cos()) + r2l * t.sin(),
                1.0 / w,
            ),
        )
    } else {
        (
            DVec3::new(
                0.5 * (theta.cos() + l / w * theta.sin()),
                0.5 * (theta.sin() - l / w * theta.cos()),
                -1.0 / w,
            ),
            DVec3::new(
                0.5 * (-l / w * theta.sin() + theta.cos()),
                0.5 * (l / w * theta.cos() + theta.sin()),
                1.0 / w,
            ),
        )
    };
    DMat3::from_cols(left, right, DVec3::ZERO)
}

/// The measurement model h: what (range, bearing) the scanner should see
/// for a landmark, from this centre state.
pub fn h(state: DVec3, landmark: DVec2, d: f64) -> DVec2 {
    let dx = landmark.x - (state.x + d * state.z.cos());
    let dy = landmark.y - (state.y + d * state.z.sin());
    DVec2::new(dx.hypot(dy), wrap_pi(dy.atan2(dx) - state.z))
}

/// H = dh/dstate, 2x3, padded to 3x3 with a zero row.
pub fn dh_dstate(state: DVec3, landmark: DVec2, d: f64) -> DMat3 {
    let theta = state.z;
    let dx = landmark.x - (state.x + d * theta.cos());
    let dy = landmark.y - (state.y + d * theta.sin());
    let q = dx * dx + dy * dy;
    let sq = q.sqrt();
    rows(
        [
            -dx / sq,
            -dy / sq,
            d / sq * (dx * theta.sin() - dy * theta.cos()),
        ],
        [
            dy / q,
            -dx / q,
            -d / q * (dx * theta.cos() + dy * theta.sin()) - 1.0,
        ],
        [0.0, 0.0, 0.0],
    )
}

/// The position's uncertainty as an ellipse: the direction of its long
/// axis, and the standard deviations along it and across it. (The course's
/// get_error_ellipse, with the 2x2 eigenvalues worked out by hand.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorEllipse {
    pub angle: f64,
    pub major: f64,
    pub minor: f64,
}

pub fn error_ellipse(covariance: &DMat3) -> ErrorEllipse {
    let (a, b, c) = (
        covariance.x_axis.x,
        covariance.y_axis.x,
        covariance.y_axis.y,
    );
    let mid = (a + c) / 2.0;
    let spread = (((a - c) / 2.0).powi(2) + b * b).sqrt();
    ErrorEllipse {
        angle: 0.5 * (2.0 * b).atan2(a - c),
        major: (mid + spread).max(0.0).sqrt(),
        minor: (mid - spread).max(0.0).sqrt(),
    }
}

/// The filter's belief: a state and how unsure it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ekf {
    /// (x, y, heading) of the robot's centre; heading in [-pi, pi).
    pub state: DVec3,
    pub covariance: DMat3,
}

impl Ekf {
    pub fn predict(&mut self, control: DVec2, p: &EkfParams) {
        let (l, r) = (control.x, control.y);
        let var_l =
            (p.control_motion_factor * l).powi(2) + (p.control_turn_factor * (l - r)).powi(2);
        let var_r =
            (p.control_motion_factor * r).powi(2) + (p.control_turn_factor * (l - r)).powi(2);
        let control_cov = DMat3::from_diagonal(DVec3::new(var_l, var_r, 0.0));
        let gm = dg_dstate(self.state, control, p.robot_width);
        let v = dg_dcontrol(self.state, control, p.robot_width);
        self.covariance = gm * self.covariance * gm.transpose() + v * control_cov * v.transpose();
        self.state = g(self.state, control, p.robot_width);
    }

    /// One landmark's (range, bearing). Returns the innovation: how far the
    /// measurement was from what the filter expected.
    pub fn correct(&mut self, measurement: DVec2, landmark: DVec2, p: &EkfParams) -> DVec2 {
        let hm = dh_dstate(self.state, landmark, p.scanner_displacement);
        let q = DMat3::from_diagonal(DVec3::new(
            p.measurement_distance_stddev.powi(2),
            p.measurement_angle_stddev.powi(2),
            1.0, // the padding's dummy measurement
        ));
        let s = hm * self.covariance * hm.transpose() + q;
        let k = self.covariance * hm.transpose() * s.inverse();
        let mut innovation = measurement - h(self.state, landmark, p.scanner_displacement);
        innovation.y = wrap_pi(innovation.y);
        self.state += k * innovation.extend(0.0);
        // NOTE: the course leaves the heading as it comes out; wrapping it
        // keeps it in g's range.
        self.state.z = wrap_pi(self.state.z);
        self.covariance = (DMat3::IDENTITY - k * hm) * self.covariance;
        innovation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 150.0;
    const D: f64 = 30.0;

    /// A Jacobian against the slope measured by nudging each input.
    fn numeric(f: impl Fn(DVec3) -> DVec3, at: DVec3) -> DMat3 {
        let e = 1e-6;
        let col = |unit: DVec3| (f(at + unit * e) - f(at - unit * e)) / (2.0 * e);
        DMat3::from_cols(col(DVec3::X), col(DVec3::Y), col(DVec3::Z))
    }

    fn close(a: DMat3, b: DMat3, tol: f64) -> bool {
        (a - b).to_cols_array().iter().all(|v| v.abs() < tol)
    }

    const STATE: DVec3 = DVec3::new(1200.0, 900.0, 0.7);
    const TURNING: DVec2 = DVec2::new(40.0, 55.0);
    const STRAIGHT: DVec2 = DVec2::new(40.0, 40.0);

    /// slam_07_b and _c check their derivatives against numbers; so do we.
    #[test]
    fn motion_jacobians_match_the_slopes() {
        for control in [TURNING, STRAIGHT] {
            let gm = numeric(|s| g(s, control, W), STATE);
            assert!(close(dg_dstate(STATE, control, W), gm, 1e-5), "{control}");
        }
        // V: nudge (left, right); the third input is padding.
        let v = numeric(|c| g(STATE, c.truncate(), W), TURNING.extend(0.0));
        assert!(close(dg_dcontrol(STATE, TURNING, W), v, 1e-4));
        // Not for STRAIGHT: nudging one wheel there takes g's turning branch
        // with a turn of 1e-8 rad and a radius of kilometres, and the sines
        // cancel into noise. Instead: the straight formula is where the
        // turning one goes as right comes to equal left.
        let almost = DVec2::new(STRAIGHT.x, STRAIGHT.y + 1e-3);
        assert!(close(
            dg_dcontrol(STATE, STRAIGHT, W),
            dg_dcontrol(STATE, almost, W),
            1e-4
        ));
    }

    /// slam_07_e: the measurement Jacobian.
    #[test]
    fn measurement_jacobian_matches_the_slopes() {
        let landmark = DVec2::new(1900.0, 1500.0);
        let hm = numeric(|s| h(s, landmark, D).extend(0.0), STATE);
        assert!(close(dh_dstate(STATE, landmark, D), hm, 1e-6));
        // A landmark straight ahead of the scanner: bearing 0.
        let ahead = DVec2::new(STATE.x, STATE.y) + DVec2::from_angle(STATE.z) * (D + 500.0);
        let z = h(STATE, ahead, D);
        assert!((z.x - 500.0).abs() < 1e-9 && z.y.abs() < 1e-9);
    }

    #[test]
    fn ellipse_of_a_tilted_covariance() {
        // sigma 30 along 30 degrees, 10 across it.
        let rot = bevy::math::DMat2::from_angle(30.0_f64.to_radians());
        let c2 = rot * bevy::math::DMat2::from_diagonal(DVec2::new(900.0, 100.0)) * rot.transpose();
        let c = DMat3::from_cols(c2.x_axis.extend(0.0), c2.y_axis.extend(0.0), DVec3::Z);
        let e = error_ellipse(&c);
        assert!((e.angle.to_degrees() - 30.0).abs() < 1e-9, "{e:?}");
        assert!((e.major - 30.0).abs() < 1e-9 && (e.minor - 10.0).abs() < 1e-9);
    }

    /// Predicting only: the uncertainty grows, more when turning.
    /// Correcting: it shrinks, and the state moves towards what was seen.
    #[test]
    fn predict_grows_correct_shrinks() {
        let p = EkfParams::default();
        let start = Ekf {
            state: STATE,
            covariance: DMat3::from_diagonal(DVec3::new(100.0, 100.0, 0.001)),
        };
        let mut straight = start;
        straight.predict(STRAIGHT, &p);
        let mut turning = start;
        turning.predict(TURNING, &p);
        assert!(straight.covariance.determinant() > start.covariance.determinant());
        assert!(turning.covariance.z_axis.z > straight.covariance.z_axis.z);

        // The robot is really 50 mm further along x than it believes.
        let landmark = DVec2::new(2000.0, 1300.0);
        let truth = straight.state + DVec3::new(50.0, 0.0, 0.0);
        let mut ekf = straight;
        ekf.covariance =
            DMat3::from_diagonal(DVec3::new(100.0_f64.powi(2), 100.0_f64.powi(2), 0.01));
        let before = ekf;
        for _ in 0..5 {
            ekf.correct(h(truth, landmark, D), landmark, &p);
        }
        let error = |e: &Ekf| (e.state.truncate() - truth.truncate()).length();
        assert!(
            error(&ekf) < error(&before),
            "{} vs {}",
            error(&ekf),
            error(&before)
        );
        assert!(error_ellipse(&ekf.covariance).major < error_ellipse(&before.covariance).major);
        // Still a covariance: symmetric (nearly) and positive on the diagonal.
        let c = ekf.covariance;
        assert!((c.y_axis.x - c.x_axis.y).abs() < 1e-6 * c.x_axis.x);
        assert!(c.x_axis.x > 0.0 && c.y_axis.y > 0.0 && c.z_axis.z > 0.0);
    }

    #[test]
    fn innovation_wraps_the_bearing() {
        let p = EkfParams::default();
        // Landmark just behind on the left: expected bearing near +pi.
        let state = DVec3::new(0.0, 0.0, 0.0);
        let landmark = DVec2::new(-1000.0, 1.0);
        let expected = h(state, landmark, D);
        assert!(expected.y > 3.1);
        // Measured just past pi, i.e. near -pi: the innovation is small.
        let mut ekf = Ekf {
            state,
            covariance: DMat3::from_diagonal(DVec3::new(100.0, 100.0, 0.01)),
        };
        let measured = DVec2::new(expected.x, wrap_pi(expected.y + 0.01));
        let innovation = ekf.correct(measured, landmark, &p);
        assert!((innovation.y - 0.01).abs() < 1e-9, "{innovation}");
    }
}
