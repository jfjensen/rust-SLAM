// Unit C, part 2: the 1D Kalman filter (the course's slam_06_f).
//
// If the belief, the control and the measurement are all gaussians, the
// histogram filter's results are gaussians too, and two numbers each
// (mean and variance) are enough. No cells, no lists:
//
//   predict   mu  = a * mu + u            sigma2 = a^2 * sigma2 + R
//   correct   K   = c * sigma2 / (c^2 * sigma2 + Q)
//             mu  = mu + K * (z - c * mu)
//             sigma2 = (1 - K * c) * sigma2
//
// u, R: the control and its variance. z, Q: the measurement and its
// variance. Here (as in the course) a = c = 1: the robot moves by u, and
// the sensor measures the position itself.

/// A gaussian: mean and variance (the course's Density).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Density {
    pub mu: f64,
    pub sigma2: f64,
}

impl Density {
    pub fn new(mu: f64, sigma: f64) -> Density {
        Density {
            mu,
            sigma2: sigma * sigma,
        }
    }

    pub fn sigma(&self) -> f64 {
        self.sigma2.sqrt()
    }

    /// The probability density at x (the course's normpdf).
    pub fn pdf(&self, x: f64) -> f64 {
        let d = x - self.mu;
        (-0.5 * d * d / self.sigma2).exp() / (std::f64::consts::TAU * self.sigma2).sqrt()
    }
}

/// The course's a and c.
const A: f64 = 1.0;
const C: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStep {
    pub prediction: Density,
    pub measurement: Option<Density>,
    pub correction: Density,
    /// The Kalman gain: how much of the surprise (z - mu) was believed.
    /// 0 when there was no measurement.
    pub gain: f64,
}

/// The course's kalman_filter_step. With no measurement, the correction is
/// the prediction.
pub fn kalman_step(belief: Density, control: Density, measurement: Option<Density>) -> KalmanStep {
    let prediction = Density {
        mu: A * belief.mu + control.mu,
        sigma2: A * A * belief.sigma2 + control.sigma2,
    };
    let (correction, gain) = match measurement {
        Some(z) => {
            let k = C * prediction.sigma2 / (C * C * prediction.sigma2 + z.sigma2);
            let corrected = Density {
                mu: prediction.mu + k * (z.mu - C * prediction.mu),
                sigma2: (1.0 - k * C) * prediction.sigma2,
            };
            (corrected, k)
        }
        None => (prediction, 0.0),
    };
    KalmanStep {
        prediction,
        measurement,
        correction,
        gain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unit_c::distribution::{Distribution, histogram_step};

    /// slam_06_f's two steps, worked out by hand:
    ///   1. predict 10 + 40 = 50, 1 + 100 = 101. K = 101 / 201.
    ///      correct 50 + K * 10 = 55.025, (1 - K) * 101 = 50.249
    ///   2. predict 125.025, 150.249. K = 150.249 / 550.249.
    ///      correct 125.025 + K * 14.975 = 129.114, (1 - K) * 150.249 = 109.22
    #[test]
    fn the_course_example() {
        let mut belief = Density::new(10.0, 1.0);
        let steps = [((40.0, 10.0), (60.0, 10.0)), ((70.0, 10.0), (140.0, 20.0))];
        let mut out = Vec::new();
        for ((u, r), (z, q)) in steps {
            let step = kalman_step(belief, Density::new(u, r), Some(Density::new(z, q)));
            belief = step.correction;
            out.push(step);
        }
        let close = |a: f64, b: f64| (a - b).abs() < 1e-2;
        assert!(close(out[0].correction.mu, 55.025), "{:?}", out[0]);
        assert!(close(out[0].correction.sigma2, 50.249), "{:?}", out[0]);
        assert!(close(out[1].prediction.mu, 125.025), "{:?}", out[1]);
        assert!(close(out[1].correction.mu, 129.114), "{:?}", out[1]);
        assert!(close(out[1].correction.sigma2, 109.22), "{:?}", out[1]);
    }

    /// The point of slam_06_f: with gaussians, the histogram filter and the
    /// Kalman filter agree (up to the cells being whole numbers).
    #[test]
    fn histogram_and_kalman_agree_on_gaussians() {
        let mut hist = Distribution::gaussian(10, 1.0);
        let mut kal = Density::new(10.0, 1.0);
        let steps = [
            (40, 10.0, Some((60, 10.0))),
            (70, 10.0, None),
            (30, 5.0, Some((150, 20.0))),
        ];
        for (u, r, z) in steps {
            let h = histogram_step(
                &hist,
                &Distribution::gaussian(u, r),
                z.map(|(z, q)| Distribution::gaussian(z, q)),
            );
            let k = kalman_step(
                kal,
                Density::new(u as f64, r),
                z.map(|(z, q)| Density::new(z as f64, q)),
            );
            hist = h.correction;
            kal = k.correction;
            assert!(
                (hist.mean() - kal.mu).abs() < 0.05,
                "{} vs {:?}",
                hist.mean(),
                kal
            );
            assert!(
                (hist.variance() - kal.sigma2).abs() < 0.01 * kal.sigma2,
                "{} vs {:?}",
                hist.variance(),
                kal
            );
        }
    }

    #[test]
    fn no_measurement_only_predicts() {
        let step = kalman_step(Density::new(5.0, 2.0), Density::new(20.0, 3.0), None);
        assert_eq!(step.correction, step.prediction);
        assert_eq!(
            step.prediction,
            Density {
                mu: 25.0,
                sigma2: 13.0
            }
        );
        assert_eq!(step.gain, 0.0);
    }

    #[test]
    fn pdf_integrates_to_one() {
        let d = Density::new(100.0, 7.0);
        let total: f64 = (0..200).map(|x| d.pdf(x as f64)).sum();
        assert!((total - 1.0).abs() < 1e-6, "{total}");
    }
}
