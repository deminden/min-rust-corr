//! Chatterjee's rank correlation ξ (Chatterjee 2021, J. Amer. Statist.
//! Assoc. 116(536), arXiv:1909.10140).
//!
//! Sort the pairs by x (ties in x ordered by observation index, so the
//! result is deterministic; the paper breaks them at random). With
//! r_i = #{j : y_j ≤ y_i} and l_i = #{j : y_j ≥ y_i} in that order,
//!
//! ξ(x, y) = 1 − n Σ_{i<n} |r_{i+1} − r_i| / (2 Σ_i l_i (n − l_i)).
//!
//! ξ measures how well y is a function of x and is not symmetric: output
//! cell (i, j) is ξ(row i, row j). The optional symmetrised variant reports
//! max(ξ(x, y), ξ(y, x)). P-values are one-sided, 1 − Φ(√n ξ / τ̂), with
//! the tie-aware variance estimate τ̂² of the paper (Section 2); the
//! symmetrised variant uses the Bonferroni bound min(1, 2 min(p₁, p₂)).

use crate::engine::{PairKernel, PairStat};
use crate::error::Result;
use crate::options::{ChatterjeeParams, PvalueSpec};
use crate::prep::rank::{for_each_tie_group, sort_order};
use crate::prep::standardize::all_equal;
use crate::stats::normal::upper_p;

/// ξ kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Chatterjee {
    params: ChatterjeeParams,
}

impl Chatterjee {
    /// Create with parameters.
    pub fn new(params: ChatterjeeParams) -> Self {
        Self { params }
    }
}

/// A row prepared for both roles: sort order (as x), max-ranks r (as y),
/// the denominator 2 Σ l(n − l) and the null standard deviation τ̂.
#[derive(Debug, Clone, Default)]
pub struct XiRow {
    perm: Vec<u32>,
    r: Vec<u32>,
    den: f64,
    tau: f64,
}

impl XiRow {
    fn new(x: &[f64]) -> Option<Self> {
        if x.len() < 2 || all_equal(x) {
            return None;
        }
        let n = x.len();
        let mut row = XiRow { perm: Vec::new(), r: vec![0; n], den: 0.0, tau: f64::NAN };
        sort_order(x, &mut row.perm);
        let mut den = 0.0;
        for_each_tie_group(x, &row.perm, |s, e| {
            let l = (n - s) as f64;
            den += (e - s) as f64 * l * (n as f64 - l);
            row.perm[s..e].iter().for_each(|&k| row.r[k as usize] = e as u32);
        });
        row.den = 2.0 * den;
        row.tau = tau_hat(&row.r, &row.perm, den);
        Some(row)
    }
}

/// τ̂ = √(a − 2b + c²) / d with u = sorted r / n (Chatterjee 2021, §2):
/// a = Σ(2n − 2i + 1)u_i² / n², b = Σ(v_i + (n − i)u_i)² / n³,
/// c = Σ(2n − 2i + 1)u_i / n², d = Σ l(n − l) / n³, v_i = Σ_{j≤i} u_j.
fn tau_hat(r: &[u32], perm: &[u32], sum_l: f64) -> f64 {
    let n = r.len() as f64;
    let (mut a, mut b, mut c, mut v) = (0.0, 0.0, 0.0, 0.0);
    for (i, &k) in perm.iter().enumerate() {
        let u = f64::from(r[k as usize]) / n;
        let w = 2.0 * n - 2.0 * (i as f64 + 1.0) + 1.0;
        v += u;
        a += w * u * u;
        c += w * u;
        let t = v + (n - (i as f64 + 1.0)) * u;
        b += t * t;
    }
    let (a, b, c, d) = (a / (n * n), b / (n * n * n), c / (n * n), sum_l / (n * n * n));
    let t2 = a - 2.0 * b + c * c;
    if t2 > 0.0 && d > 0.0 { t2.sqrt() / d } else { f64::NAN }
}

/// ξ(x → y) and its one-sided p-value.
fn xi(x: &XiRow, y: &XiRow, want_p: bool) -> (f64, f64) {
    let n = x.perm.len();
    let mut s: u64 = 0;
    for w in x.perm.windows(2) {
        s += u64::from(y.r[w[1] as usize].abs_diff(y.r[w[0] as usize]));
    }
    let v = 1.0 - n as f64 * s as f64 / y.den;
    let p = if want_p { upper_p((n as f64).sqrt() * v / y.tau) } else { f64::NAN };
    (v, p)
}

impl Chatterjee {
    fn stat(&self, a: &XiRow, b: &XiRow, want_p: bool) -> PairStat {
        let n_eff = a.perm.len();
        let (v, p) = xi(a, b, want_p);
        if !self.params.symmetric {
            return PairStat { value: v, n_eff, aux: p };
        }
        let (w, q) = xi(b, a, want_p);
        PairStat { value: v.max(w), n_eff, aux: (2.0 * p.min(q)).min(1.0) }
    }
}

impl PairKernel for Chatterjee {
    /// Whether p-values are wanted.
    type Shared = bool;
    type Prepared = XiRow;
    type Scratch = ();

    fn shared(&self, _: usize, _: usize, p: &PvalueSpec) -> Result<bool> {
        Ok(p.wanted())
    }

    fn scratch(&self, _: usize) {}

    fn prepare(&self, x: &[f64], _: &bool, _: &mut ()) -> Option<XiRow> {
        XiRow::new(x)
    }

    fn eval(&self, a: &XiRow, b: &XiRow, sh: &bool, _: &mut ()) -> PairStat {
        self.stat(a, b, *sh)
    }

    fn eval_gathered(&self, x: &[f64], y: &[f64], sh: &bool, _: &mut ()) -> PairStat {
        match (XiRow::new(x), XiRow::new(y)) {
            (Some(a), Some(b)) => self.stat(&a, &b, *sh),
            _ => PairStat::nan(x.len()),
        }
    }

    fn pvalue(&self, st: &PairStat, _: &bool) -> f64 {
        st.aux
    }

    fn symmetric(&self) -> bool {
        self.params.symmetric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_function_and_independence() {
        let x: Vec<f64> = (0..200).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| (v - 99.5) * (v - 99.5)).collect();
        let k = Chatterjee::default();
        let st = k.eval_gathered(&x, &y, &true, &mut ());
        // y is a function of x: ξ = 1 − 3/(n+1) for distinct values.
        assert!(st.value > 0.95, "{}", st.value);
        assert!(st.aux < 1e-20);
        let back = k.eval_gathered(&y, &x, &true, &mut ());
        assert!(back.value < st.value, "{} {}", back.value, st.value);
        let diag = k.eval_gathered(&x, &x, &false, &mut ());
        assert!((diag.value - (1.0 - 3.0 / 201.0)).abs() < 1e-15);
    }

    #[test]
    fn tau_without_ties_near_two_fifths() {
        let x: Vec<f64> = (0..2000).map(|i| f64::from((i * 7919) % 2003)).collect();
        let row = XiRow::new(&x).unwrap();
        assert!((row.tau * row.tau - 0.4).abs() < 1e-3);
    }
}
