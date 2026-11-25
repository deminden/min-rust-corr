//! Beta(a, b) density, CDF and quantile (clean-room, see PROVENANCE.md).

use super::special::{beta_inc_pair, ln_beta};

fn valid(a: f64, b: f64) -> bool {
    a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()
}

/// Density f(x; a, b). NaN for invalid parameters; 0 outside [0, 1].
pub fn pdf(x: f64, a: f64, b: f64) -> f64 {
    if !valid(a, b) || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    if x == 0.0 || x == 1.0 {
        let e = if x == 0.0 { a } else { b };
        return if e < 1.0 {
            f64::INFINITY
        } else if e == 1.0 {
            (-ln_beta(a, b)).exp()
        } else {
            0.0
        };
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - ln_beta(a, b)).exp()
}

/// CDF I_x(a, b). NaN for invalid parameters.
pub fn cdf(x: f64, a: f64, b: f64) -> f64 {
    if !valid(a, b) || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    beta_inc_pair(x, 1.0 - x, a, b).0
}

/// Quantile Q(p; a, b): the x with I_x(a, b) = p. Bracketed Newton with
/// bisection fallback; symmetric parameters use Q(1−p) = 1 − Q(p) so that
/// the lower half is always solved directly. NaN for invalid input.
pub fn quantile(p: f64, a: f64, b: f64) -> f64 {
    if !valid(a, b) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    if a == b && p > 0.5 {
        return 1.0 - quantile_inner(1.0 - p, a, b);
    }
    quantile_inner(p, a, b)
}

fn quantile_inner(p: f64, a: f64, b: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    // Start at the mean, then a few bisection steps to get into the basin.
    let mut x = a / (a + b);
    for _ in 0..8 {
        if cdf(x, a, b) < p {
            lo = x;
        } else {
            hi = x;
        }
        x = 0.5 * (lo + hi);
    }
    for _ in 0..60 {
        let f = cdf(x, a, b) - p;
        if f == 0.0 {
            return x;
        }
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let d = pdf(x, a, b);
        let mut next = if d > 0.0 && d.is_finite() { x - f / d } else { f64::NAN };
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        let step = (next - x).abs();
        x = next;
        if step <= 4.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE) || hi - lo <= f64::EPSILON * x {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_case() {
        assert!((cdf(0.3, 1.0, 1.0) - 0.3).abs() < 1e-15);
        assert!((quantile(0.3, 1.0, 1.0) - 0.3).abs() < 1e-15);
        assert!((pdf(0.3, 1.0, 1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn invalid() {
        assert!(cdf(0.5, -1.0, 1.0).is_nan());
        assert!(quantile(1.5, 2.0, 2.0).is_nan());
        assert_eq!(pdf(1.5, 2.0, 2.0), 0.0);
    }

    #[test]
    fn round_trip() {
        for &(a, b) in &[(6.0, 6.0), (2.0, 5.0), (0.7, 9.0), (50.0, 50.0)] {
            for k in 1..20 {
                let p = k as f64 / 20.0;
                let x = quantile(p, a, b);
                assert!((cdf(x, a, b) - p).abs() < 1e-13, "a={a} b={b} p={p}");
            }
        }
    }
}
