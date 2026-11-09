// A small seeded random number generator (SplitMix64), so the synthetic
// data is the same for the same seed, on every machine, without a crate.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Normal distribution with mean 0 and this standard deviation
    /// (Box-Muller).
    pub fn gaussian(&mut self, sigma: f64) -> f64 {
        let u1 = 1.0 - self.unit(); // (0, 1]: ln(0) would be -inf
        let u2 = self.unit();
        sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// True with this probability.
    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_numbers() {
        let a: Vec<u64> = (0..5)
            .map({
                let mut r = Rng::new(7);
                move |_| r.next_u64()
            })
            .collect();
        let mut r = Rng::new(7);
        assert!(a.iter().all(|&x| x == r.next_u64()));
    }

    #[test]
    fn gaussian_has_the_asked_spread() {
        let mut r = Rng::new(1);
        let n = 20_000;
        let xs: Vec<f64> = (0..n).map(|_| r.gaussian(3.0)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let sd = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((sd - 3.0).abs() < 0.1, "sd {sd}");
    }
}
