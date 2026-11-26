//! Special functions written from their mathematical definitions:
//! log-gamma (Lanczos below 10, Stirling series above), log-beta and the
//! regularised incomplete beta function (continued fraction DLMF 8.17.22,
//! evaluated by the modified Lentz algorithm). See also [`super::gamma`].

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
pub(crate) const LN_SQRT_PI: f64 = 0.572_364_942_924_700_1;
pub(crate) const TINY: f64 = 1e-300;
pub(crate) const EPS: f64 = 1e-16;
pub(crate) const MAX_ITER: usize = 2000;

/// Stirling remainder δ(x) = lnΓ(x) − [(x−½)ln x − x + ½ln 2π], for x ≥ 10.
/// Coefficients B_{2k} / (2k(2k−1)) (DLMF 5.11.1).
fn stirling_delta(x: f64) -> f64 {
    const C: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360_360.0,
        1.0 / 156.0,
        -3617.0 / 122_400.0,
    ];
    let r = 1.0 / x;
    let r2 = r * r;
    let mut s = 0.0;
    for c in C.iter().rev() {
        s = s * r2 + c;
    }
    s * r
}

/// Lanczos approximation (g = 7, 9 terms; published coefficients) of
/// ln Γ(x) for 0.5 ≤ x < 10.
fn ln_gamma_lanczos(x: f64) -> f64 {
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = x - 1.0;
    let mut a = P[0];
    for (k, p) in P.iter().enumerate().skip(1) {
        a += p / (z + k as f64);
    }
    let t = z + G + 0.5;
    LN_SQRT_2PI + (z + 0.5) * t.ln() - t + a.ln()
}

/// ln Γ(x) for x > 0 (NaN otherwise).
pub fn ln_gamma(x: f64) -> f64 {
    if !(x > 0.0) || x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    if x == 0.5 {
        return LN_SQRT_PI;
    }
    if x < 0.5 {
        return ln_gamma_lanczos(x + 1.0) - x.ln();
    }
    if x < 10.0 {
        return ln_gamma_lanczos(x);
    }
    (x - 0.5) * x.ln() - x + LN_SQRT_2PI + stirling_delta(x)
}

/// ln B(a, b) for a, b > 0. Uses a cancellation-free form when both
/// arguments are ≥ 10.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if !(a > 0.0 && b > 0.0) {
        return f64::NAN;
    }
    if a >= 10.0 && b >= 10.0 {
        let s = a + b;
        LN_SQRT_2PI + (a - 0.5) * (a / s).ln() + (b - 0.5) * (b / s).ln() - 0.5 * s.ln()
            + stirling_delta(a)
            + stirling_delta(b)
            - stirling_delta(s)
    } else {
        ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    }
}

/// Modified Lentz evaluation of 1 / (1 + d1/(1 + d2/(1 + ...))) where
/// `d(m)` returns the m-th partial numerator (m ≥ 1).
fn lentz(d: impl Fn(usize) -> f64) -> f64 {
    let mut f = 1.0;
    let mut c = 1.0;
    let mut dd = 0.0;
    for m in 1..MAX_ITER {
        let num = d(m);
        dd = 1.0 + num * dd;
        if dd.abs() < TINY {
            dd = TINY;
        }
        c = 1.0 + num / c;
        if c.abs() < TINY {
            c = TINY;
        }
        dd = 1.0 / dd;
        let delta = c * dd;
        f *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    1.0 / f
}

/// Continued-fraction part of I_x(a, b) (valid for x < (a+1)/(a+b+2)).
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    lentz(|m| {
        let k = (m / 2) as f64;
        if m % 2 == 1 {
            -(a + k) * (a + b + k) * x / ((a + 2.0 * k) * (a + 2.0 * k + 1.0))
        } else {
            k * (b - k) * x / ((a + 2.0 * k - 1.0) * (a + 2.0 * k))
        }
    })
}

/// Regularised incomplete beta I_x(a, b) given both x and y = 1 − x
/// (passing y separately avoids cancellation when x is close to 1).
/// Returns (I_x(a,b), 1 − I_x(a,b)).
pub fn beta_inc_pair(x: f64, y: f64, a: f64, b: f64) -> (f64, f64) {
    if !(a > 0.0 && b > 0.0) || x.is_nan() || y.is_nan() || x < 0.0 || y < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if y == 0.0 {
        return (1.0, 0.0);
    }
    let ln_front = a * x.ln() + b * y.ln() - ln_beta(a, b);
    if x < (a + 1.0) / (a + b + 2.0) {
        let v = (ln_front.exp() / a) * beta_cf(x, a, b);
        (v, 1.0 - v)
    } else {
        let v = (ln_front.exp() / b) * beta_cf(y, b, a);
        (1.0 - v, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn rel(a: f64, b: f64) -> f64 {
        if b == 0.0 { a.abs() } else { ((a - b) / b).abs() }
    }

    #[test]
    fn lgamma_matches_r() {
        let cases = [
            (0.5, 0.572_364_942_924_700_08),
            (3.0, std::f64::consts::LN_2),
            (10.5, 13.940_625_219_403_763),
            (50.0, 144.565_743_946_344_87),
            (171.3, 708.114_947_038_997_12),
        ];
        for (x, want) in cases {
            assert!(rel(ln_gamma(x), want) < 2e-15, "x={x}");
        }
        assert!(ln_gamma(1.0).abs() < 1e-15);
        assert!((ln_gamma(0.5) * 2.0 - PI.ln()).abs() < 1e-15);
    }

    #[test]
    fn ln_beta_branches_agree() {
        let direct = ln_gamma(12.0) + ln_gamma(15.0) - ln_gamma(27.0);
        assert!((ln_beta(12.0, 15.0) - direct).abs() < 1e-13);
    }
}
