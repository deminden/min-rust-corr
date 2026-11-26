//! erfc via the regularised incomplete gamma function Q(½, x²)
//! (DLMF 8.7.1 power series and 8.9.2 continued fraction).

use super::special::{EPS, MAX_ITER, TINY, ln_gamma};

/// Regularised lower incomplete gamma P(a, z) by its power series.
fn gamma_p_series(a: f64, z: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    for k in 1..MAX_ITER {
        term *= z / (a + k as f64);
        sum += term;
        if term.abs() < sum.abs() * EPS {
            break;
        }
    }
    (a * z.ln() - z - ln_gamma(a)).exp() * sum
}

/// Regularised upper incomplete gamma Q(a, z) by its continued fraction.
fn gamma_q_cf(a: f64, z: f64) -> f64 {
    // Q = e^{-z} z^a / Γ(a) · 1/(z+1−a − 1(1−a)/(z+3−a − 2(2−a)/(...)))
    let b0 = z + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b0;
    let mut h = d;
    for n in 1..MAX_ITER {
        let an = -(n as f64) * (n as f64 - a);
        let bn = b0 + 2.0 * n as f64;
        d = an * d + bn;
        if d.abs() < TINY {
            d = TINY;
        }
        c = bn + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    (a * z.ln() - z - ln_gamma(a)).exp() * h
}

/// Complementary error function for x ≥ 0 (uses erfc(−x) = 2 − erfc(x)).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x == 0.0 {
        return 1.0;
    }
    let z = x * x;
    if z < 1.5 { 1.0 - gamma_p_series(0.5, z) } else { gamma_q_cf(0.5, z) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn erfc_values() {
        assert!(rel(erfc(1.0), 0.157_299_207_050_285_13) < 4e-15);
        assert!(rel(erfc(0.1), 0.887_537_083_981_715_16) < 1e-15);
        assert!(rel(erfc(1.5), 0.033_894_853_524_689_246) < 1e-14);
        assert!((erfc(-1.0) - (2.0 - erfc(1.0))).abs() < 1e-16);
    }
}
