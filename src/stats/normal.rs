//! Standard-normal p-values and quantiles.

use super::gamma::erfc;
use std::f64::consts::{PI, SQRT_2};

/// P(|Z| ≥ |z|) = erfc(|z| / √2). NaN for non-finite `z`.
pub fn two_sided_p(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    erfc(z.abs() / SQRT_2).clamp(0.0, 1.0)
}

/// P(Z ≥ z) = erfc(z / √2) / 2. NaN for non-finite `z`.
pub fn upper_p(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    (0.5 * erfc(z / SQRT_2)).clamp(0.0, 1.0)
}

/// Φ(x), the standard-normal CDF.
pub fn cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// Φ⁻¹(p): Acklam's rational approximation refined by two Halley steps on
/// Φ (computed from erfc), accurate to a few ulp. NaN outside (0, 1).
pub fn quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return if p == 0.0 {
            f64::NEG_INFINITY
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    // 1 − p is exact for p ≥ 0.5, so refine in the lower tail where Φ(x)
    // is computed without cancellation.
    if p > 0.5 {
        return -quantile(1.0 - p);
    }
    let mut x = acklam(p);
    for _ in 0..2 {
        let e = cdf(x) - p;
        let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
        x -= u / (1.0 + 0.5 * x * u);
    }
    x
}

/// Acklam's approximation to Φ⁻¹ (relative error about 1e-9).
fn acklam(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let poly = |c: &[f64], t: f64| c.iter().fold(0.0, |acc, &k| acc * t + k);
    let tail = |q: f64| poly(&C, q) / (poly(&D, q) * q + 1.0);
    const P_LOW: f64 = 0.02425;
    if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        poly(&A, r) * q / (poly(&B, r) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_pnorm() {
        // 2 * pnorm(-z) from R 4.3.3.
        let cases = [
            (0.1, 0.920_344_325_445_942_02),
            (1.0, 0.317_310_507_862_914_09),
            (2.0, 0.045_500_263_896_358_424),
            (3.0, 0.002_699_796_063_260_189_2),
            (5.0, 5.733_031_437_583_878_2e-7),
            (8.0, 1.244_192_114_854_357e-15),
            (12.0, 3.552_964_224_155_358_1e-33),
            (20.0, 5.507_248_237_212_467_5e-89),
            (37.0, 1.145_114_244_504_915_4e-299),
        ];
        for (z, want) in cases {
            let got = two_sided_p(z);
            assert!(((got - want) / want).abs() < 5e-13, "z={z} got={got} want={want}");
            assert!((upper_p(z) - 0.5 * want).abs() <= 5e-13 * want);
        }
    }

    #[test]
    fn quantile_matches_r_qnorm() {
        // qnorm(p) from R 4.6.2.
        let cases = [
            (1e-300, -37.047_096_299_361_201),
            (1e-10, -6.361_340_902_404_055_7),
            (1e-3, -3.090_232_306_167_813_2),
            (0.02, -2.053_748_910_631_822_5),
            (0.1, -1.281_551_565_544_600_8),
            (0.3, -0.524_400_512_708_040_67),
            (0.7, 0.524_400_512_708_040_67),
            (0.975, 1.959_963_984_540_053_6),
            (0.999_999, 4.753_424_308_817_089_1),
        ];
        for (p, want) in cases {
            let got = quantile(p);
            assert!(((got - want) / want).abs() < 1e-14, "p={p} got={got} want={want}");
        }
        assert_eq!(quantile(0.5), 0.0);
        assert!(quantile(1.5).is_nan());
    }
}
