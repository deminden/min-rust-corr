//! Correlation coefficient → two-sided Student-t p-value.

use super::special::beta_inc_pair;

/// Two-sided p-value of a correlation `r` computed from `n` observations,
/// using t = r·√((n−2)/(1−r²)) with n−2 degrees of freedom.
///
/// Evaluated without forming t: P(|T| ≥ t) = I_{1−r²}((n−2)/2, ½).
/// Returns NaN when `r` is not finite or n < 3, and 0 when |r| = 1.
pub fn corr_t_pvalue(r: f64, n: usize) -> f64 {
    if !r.is_finite() || n < 3 {
        return f64::NAN;
    }
    let r = r.clamp(-1.0, 1.0).abs();
    if r >= 1.0 {
        return 0.0;
    }
    let df = (n - 2) as f64;
    let one_minus_r2 = (1.0 - r) * (1.0 + r);
    beta_inc_pair(one_minus_r2, r * r, 0.5 * df, 0.5).0.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_pt() {
        // 2 * pt(-|r| sqrt((n-2)/(1-r^2)), n-2) from R 4.3.3.
        let cases = [
            (0.01, 3, 0.993_633_696_168_253_92),
            (0.3, 10, 0.399_691_468_750_000_17),
            (0.9, 77, 9.134_104_527_613_937_8e-29),
            (0.999_999_99, 5, 1.200_421_761_130_900_4e-12),
        ];
        for (r, n, want) in cases {
            let got = corr_t_pvalue(r, n);
            assert!(((got - want) / want).abs() < 1e-9, "r={r} n={n} got={got}");
        }
        assert_eq!(corr_t_pvalue(1.0, 10), 0.0);
        assert!(corr_t_pvalue(0.5, 2).is_nan());
    }
}
