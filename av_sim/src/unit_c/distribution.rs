// Unit C, part 1: discrete distributions (the course's distribution.py and
// slam_06_b/c), and the histogram filter built from them (slam_06_d/e).
//
// A Distribution holds a probability for each whole cell offset, offset+1,
// ... offset+len-1; every other cell is 0. The values add up to 1.
//
//   convolve   "move": every cell's probability is spread out by the
//              control's distribution. The belief gets wider.
//   multiply   "measure": cell by cell, belief times measurement, then
//              normalised again. The belief gets narrower.
//
// The histogram filter step is just those two, one after the other.

#[derive(Clone, Debug, PartialEq)]
pub struct Distribution {
    pub offset: i64,
    pub values: Vec<f64>,
}

/// How far out the course's gaussian goes, in sigmas.
pub const GAUSSIAN_CUT: f64 = 5.0;

impl Distribution {
    pub fn start(&self) -> i64 {
        self.offset
    }

    /// The first cell after the distribution.
    pub fn stop(&self) -> i64 {
        self.offset + self.values.len() as i64
    }

    /// The probability of one cell (0 outside the stored range).
    pub fn value(&self, cell: i64) -> f64 {
        usize::try_from(cell - self.offset)
            .ok()
            .and_then(|i| self.values.get(i))
            .copied()
            .unwrap_or(0.0)
    }

    /// Scale the values so they add up to 1. False (and unchanged) if they
    /// add up to 0: there's nothing to scale.
    pub fn normalize(&mut self) -> bool {
        let sum: f64 = self.values.iter().sum();
        if sum <= 0.0 {
            return false;
        }
        for v in &mut self.values {
            *v /= sum;
        }
        true
    }

    /// Certain: all the probability in one cell.
    pub fn unit_pulse(centre: i64) -> Distribution {
        Distribution {
            offset: centre,
            values: vec![1.0],
        }
    }

    /// Peak at `centre`, falling in straight lines to 0 at centre +/-
    /// half_width (the course's triangle). Half width 1 is a unit pulse.
    pub fn triangle(centre: i64, half_width: i64) -> Distribution {
        let w = half_width.max(1);
        // 1, 2, .. w-1, w, w-1, .. 2, 1
        let values = (-w + 1..w).map(|i| (w - i.abs()) as f64).collect();
        let mut d = Distribution {
            offset: centre - w + 1,
            values,
        };
        d.normalize();
        d
    }

    /// The bell curve around `mu`, cut off at GAUSSIAN_CUT sigmas (the
    /// course's gaussian). A sigma of 0 is a unit pulse.
    pub fn gaussian(mu: i64, sigma: f64) -> Distribution {
        if sigma <= 0.0 {
            return Distribution::unit_pulse(mu);
        }
        let extent = (GAUSSIAN_CUT * sigma).ceil() as i64;
        let values = (-extent..=extent)
            .map(|x| (-0.5 * (x * x) as f64 / (sigma * sigma)).exp())
            .collect();
        let mut d = Distribution {
            offset: mu - extent,
            values,
        };
        d.normalize();
        d
    }

    /// The expected cell.
    pub fn mean(&self) -> f64 {
        self.cells().map(|(x, p)| x as f64 * p).sum()
    }

    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        self.cells()
            .map(|(x, p)| (x as f64 - mean).powi(2) * p)
            .sum()
    }

    /// The highest value (for scaling a plot).
    pub fn peak(&self) -> f64 {
        self.values.iter().copied().fold(0.0, f64::max)
    }

    /// (cell, probability) for every stored cell.
    pub fn cells(&self) -> impl Iterator<Item = (i64, f64)> + '_ {
        (self.offset..).zip(self.values.iter().copied())
    }
}

/// Move `a` by `b`: the course's convolve. Every cell of `a` hands its
/// probability on, spread out the way `b` is spread. The course builds one
/// shifted, scaled copy of `b` per cell of `a` and adds them (sum); adding
/// into one list as we go is the same thing.
pub fn convolve(a: &Distribution, b: &Distribution) -> Distribution {
    let mut values = vec![0.0; a.values.len() + b.values.len() - 1];
    for (i, pa) in a.values.iter().enumerate() {
        for (j, pb) in b.values.iter().enumerate() {
            values[i + j] += pa * pb;
        }
    }
    let mut d = Distribution {
        offset: a.offset + b.offset,
        values,
    };
    d.normalize();
    d
}

/// Combine a belief with a measurement: the course's multiply. None when
/// they don't overlap at all: every product is 0, so there's no
/// distribution left.
///
/// NOTE: the course multiplies over the union of the two ranges and, with
/// no overlap, returns all zeros (normalize does nothing). The filter then
/// carries an empty belief forever. Here the caller decides what to do.
/// Outside the overlap every product is 0, so this only keeps the overlap.
pub fn multiply(a: &Distribution, b: &Distribution) -> Option<Distribution> {
    let start = a.start().max(b.start());
    let stop = a.stop().min(b.stop());
    let mut d = Distribution {
        offset: start,
        values: (start..stop).map(|x| a.value(x) * b.value(x)).collect(),
    };
    d.normalize().then_some(d)
}

/// One step of the histogram filter, kept whole for drawing.
#[derive(Clone, Debug)]
pub struct HistogramStep {
    /// After moving: the belief convolved with the control.
    pub prediction: Distribution,
    /// What the sensor said, if it said anything this step.
    pub measurement: Option<Distribution>,
    /// After measuring: the prediction multiplied by the measurement. The
    /// prediction when there was no measurement, or when the two didn't
    /// overlap (see `disagreed`).
    pub correction: Distribution,
    /// The measurement and the prediction had no cell in common.
    pub disagreed: bool,
}

/// The course's histogram_filter_step: predict, then correct.
pub fn histogram_step(
    belief: &Distribution,
    control: &Distribution,
    measurement: Option<Distribution>,
) -> HistogramStep {
    let prediction = convolve(belief, control);
    let product = measurement.as_ref().map(|m| multiply(&prediction, m));
    let disagreed = matches!(product, Some(None));
    let correction = product.flatten().unwrap_or_else(|| prediction.clone());
    HistogramStep {
        prediction,
        measurement,
        correction,
        disagreed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums_to_one(d: &Distribution) -> bool {
        (d.values.iter().sum::<f64>() - 1.0).abs() < 1e-9
    }

    #[test]
    fn the_course_shapes() {
        // triangle(5, 3): 1 2 3 2 1 over cells 3..=7.
        let t = Distribution::triangle(5, 3);
        assert_eq!(t.offset, 3);
        let total = 9.0;
        assert_eq!(t.values, [1.0, 2.0, 3.0, 2.0, 1.0].map(|v| v / total));
        assert!((t.mean() - 5.0).abs() < 1e-9);
        // A triangle of half width w has variance (w^2 - 1) / 6 in cells.
        assert!((t.variance() - 8.0 / 6.0).abs() < 1e-9);

        let g = Distribution::gaussian(40, 10.0);
        assert_eq!((g.start(), g.stop()), (-10, 91));
        assert!(sums_to_one(&g));
        assert!((g.mean() - 40.0).abs() < 1e-9);
        assert!((g.variance() - 100.0).abs() < 0.1, "{}", g.variance());

        assert_eq!(Distribution::triangle(7, 1), Distribution::unit_pulse(7));
        assert_eq!(Distribution::gaussian(7, 0.0), Distribution::unit_pulse(7));
        assert_eq!(t.value(2), 0.0);
        assert_eq!(t.value(8), 0.0);
    }

    /// Moving adds: the means add up, and so do the variances.
    #[test]
    fn convolving_adds_means_and_variances() {
        let a = Distribution::gaussian(10, 3.0);
        let b = Distribution::triangle(20, 4);
        let c = convolve(&a, &b);
        assert!(sums_to_one(&c));
        assert!((c.mean() - 30.0).abs() < 1e-9);
        assert!((c.variance() - (a.variance() + b.variance())).abs() < 1e-6);
        // A unit pulse moves a distribution without changing its shape.
        let moved = convolve(&b, &Distribution::unit_pulse(5));
        assert_eq!(moved.offset, b.offset + 5);
        assert_eq!(moved.values, b.values);
    }

    /// Two gaussians multiplied: the result is narrower than either, and
    /// between them, nearer the narrower one (the slam_06_c picture).
    #[test]
    fn multiplying_narrows_and_lands_between() {
        let belief = Distribution::gaussian(400, 30.0);
        let measured = Distribution::gaussian(450, 15.0);
        let m = multiply(&belief, &measured).unwrap();
        assert!(sums_to_one(&m));
        assert!(m.variance() < measured.variance());
        assert!(m.mean() > 425.0 && m.mean() < 450.0, "{}", m.mean());
        // With gaussians this is the Kalman filter's answer (kalman.rs):
        // variance 1 / (1/900 + 1/225) = 180, mean 400 + 0.8 * 50 = 440.
        assert!((m.variance() - 180.0).abs() < 0.5, "{}", m.variance());
        assert!((m.mean() - 440.0).abs() < 0.05, "{}", m.mean());
    }

    #[test]
    fn no_overlap_gives_none() {
        let a = Distribution::triangle(100, 10);
        let b = Distribution::triangle(200, 10);
        assert_eq!(multiply(&a, &b), None);
        // The filter step keeps the prediction and says so.
        let step = histogram_step(&a, &Distribution::unit_pulse(0), Some(b));
        assert!(step.disagreed);
        assert_eq!(step.correction, step.prediction);
        // No measurement at all: not a disagreement.
        let step = histogram_step(&a, &Distribution::unit_pulse(0), None);
        assert!(!step.disagreed);
    }
}
