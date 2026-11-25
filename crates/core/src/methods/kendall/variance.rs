//! Tie-corrected variance of Kendall's S under independence
//! (Kendall, *Rank Correlation Methods*, ch. 4):
//!
//! var(S) = [n(n−1)(2n+5) − Σt(t−1)(2t+5) − Σu(u−1)(2u+5)] / 18
//!        + [Σt(t−1) · Σu(u−1)] / [2n(n−1)]
//!        + [Σt(t−1)(t−2) · Σu(u−1)(u−2)] / [9n(n−1)(n−2)]

use crate::prep::rank::TieSums;
use crate::stats::normal::two_sided_p;

/// Variance of S for `n` observations with tie sums `tx`, `ty`.
pub fn var_s(n: usize, tx: &TieSums, ty: &TieSums) -> f64 {
    let nf = n as f64;
    let v0 = nf * (nf - 1.0) * (2.0 * nf + 5.0);
    let mut v = (v0 - tx.t5 as f64 - ty.t5 as f64) / 18.0;
    let t2x = 2.0 * tx.pairs as f64;
    let t2y = 2.0 * ty.pairs as f64;
    v += t2x * t2y / (2.0 * nf * (nf - 1.0));
    if n > 2 {
        v += (tx.t3 as f64) * (ty.t3 as f64) / (9.0 * nf * (nf - 1.0) * (nf - 2.0));
    }
    v
}

/// z-score S / √var(S) (NaN when the variance is not positive).
pub fn z_score(s: i64, n: usize, tx: &TieSums, ty: &TieSums) -> f64 {
    let v = var_s(n, tx, ty);
    if v > 0.0 { s as f64 / v.sqrt() } else { f64::NAN }
}

/// Two-sided normal p-value of a z-score; NaN when n < 3.
pub fn pvalue(z: f64, n: usize) -> f64 {
    if n < 3 { f64::NAN } else { two_sided_p(z) }
}
