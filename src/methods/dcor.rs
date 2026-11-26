//! Distance correlation (Székely, Rizzo & Bakirov 2007, Ann. Statist.
//! 35(6)), V-statistic version as in the definition:
//!
//! dCor²(x, y) = Σ_kl A_kl B_kl / √(Σ_kl A_kl² Σ_kl B_kl²)
//!
//! with A, B the double-centred distance matrices a_kl = |x_k − x_l|.
//! Reported value: dCor = √max(dCor², 0) ∈ [0, 1].
//!
//! Every pair costs O(n log n) (Huo & Székely 2016, Technometrics 58(4)):
//! with row sums s_k = Σ_l a_kl and S = Σ_k s_k,
//! Σ A B = Σ a b − (2/n) Σ s t + S T / n², and Σ_kl a_kl b_kl is
//! accumulated in x order with a Fenwick tree keyed by the rank of y.
//! Rows are standardised first (dCor is invariant to affine maps), which
//! keeps the sums well scaled.

use crate::engine::{PairKernel, PairStat};
use crate::error::Result;
use crate::options::PvalueSpec;
use crate::prep::rank::dense_ranks;
use crate::prep::standardize::{all_equal, mean};

/// Distance-correlation kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistanceCorrelation;

/// A row prepared once: standardised values, sort order, dense ranks,
/// distance row sums s, their total S and Σ A².
#[derive(Debug, Clone, Default)]
pub struct DcorRow {
    z: Vec<f64>,
    perm: Vec<u32>,
    rank: Vec<u32>,
    s: Vec<f64>,
    total: f64,
    ss: f64,
}

impl DcorRow {
    fn new(x: &[f64]) -> Option<Self> {
        let n = x.len();
        if n < 2 || all_equal(x) {
            return None;
        }
        let m = mean(x);
        let sd = (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64).sqrt();
        if !(sd > 0.0 && sd.is_finite()) {
            return None;
        }
        let z: Vec<f64> = x.iter().map(|v| (v - m) / sd).collect();
        let (mut perm, mut rank) = (Vec::new(), vec![0; n]);
        dense_ranks(&z, &mut perm, &mut rank);
        // s_k = Σ_l |z_k − z_l| from sorted prefix sums.
        let mut s = vec![0.0; n];
        let total_z: f64 = perm.iter().map(|&k| z[k as usize]).sum();
        let mut before = 0.0;
        for (i, &k) in perm.iter().enumerate() {
            let v = z[k as usize];
            let after = total_z - before - v;
            s[k as usize] = v * i as f64 - before + after - v * (n - 1 - i) as f64;
            before += v;
        }
        let total: f64 = s.iter().sum();
        // Σ a² = 2n Σ z² − 2 (Σ z)²; Σ z = 0 up to rounding.
        let sum_a2 = 2.0 * n as f64 * z.iter().map(|v| v * v).sum::<f64>() - 2.0 * total_z * total_z;
        let ss = centred_sum(sum_a2, &s, &s, total, total, n);
        (ss > 0.0).then_some(DcorRow { z, perm, rank, s, total, ss })
    }
}

/// Σ A B from Σ a b and the row sums.
fn centred_sum(sum_ab: f64, s: &[f64], t: &[f64], sa: f64, tb: f64, n: usize) -> f64 {
    let n = n as f64;
    let st: f64 = s.iter().zip(t).map(|(a, b)| a * b).sum();
    sum_ab - 2.0 / n * st + sa * tb / (n * n)
}

/// Fenwick tree of (count, Σx, Σy, Σxy) keyed by the rank of y.
#[derive(Debug, Default)]
pub struct Fenwick {
    t: Vec<[f64; 4]>,
}

impl Fenwick {
    fn reset(&mut self, n: usize) {
        self.t.clear();
        self.t.resize(n + 1, [0.0; 4]);
    }

    fn add(&mut self, rank: usize, v: [f64; 4]) {
        let mut i = rank + 1;
        while i < self.t.len() {
            self.t[i].iter_mut().zip(v).for_each(|(a, b)| *a += b);
            i += i & i.wrapping_neg();
        }
    }

    /// Sums over entries with rank < `rank`.
    fn below(&self, rank: usize) -> [f64; 4] {
        let (mut acc, mut i) = ([0.0; 4], rank);
        while i > 0 {
            acc.iter_mut().zip(self.t[i]).for_each(|(a, b)| *a += b);
            i -= i & i.wrapping_neg();
        }
        acc
    }
}

/// Σ_{k,l} |x_k − x_l| |y_k − y_l| in O(n log n).
fn sum_ab(a: &DcorRow, b: &DcorRow, f: &mut Fenwick) -> f64 {
    f.reset(a.z.len());
    let mut all = [0.0; 4];
    let mut acc = 0.0;
    for &k in &a.perm {
        let k = k as usize;
        let (x, y, r) = (a.z[k], b.z[k], b.rank[k] as usize);
        // Earlier points have x_j ≤ x; split them by y_j < y or not.
        let lo = f.below(r);
        let term = |c: [f64; 4]| x * y * c[0] - x * c[2] - y * c[1] + c[3];
        let low = term(lo);
        let high = term(all) - low;
        acc += low - high;
        let v = [1.0, x, y, x * y];
        f.add(r, v);
        all.iter_mut().zip(v).for_each(|(s, d)| *s += d);
    }
    2.0 * acc
}

impl DistanceCorrelation {
    fn stat(a: &DcorRow, b: &DcorRow, f: &mut Fenwick) -> PairStat {
        // Canonical argument order: (x, y) and (y, x) give identical bits.
        let key = |r: &DcorRow| (r.ss.to_bits(), r.total.to_bits());
        let swap = key(a).cmp(&key(b)).then_with(|| cmp_slices(&a.z, &b.z)).is_gt();
        let (a, b) = if swap { (b, a) } else { (a, b) };
        let n = a.z.len();
        let ab = centred_sum(sum_ab(a, b, f), &a.s, &b.s, a.total, b.total, n);
        let r2 = ab / (a.ss * b.ss).sqrt();
        PairStat { value: r2.max(0.0).sqrt().min(1.0), n_eff: n, aux: f64::NAN }
    }
}

fn cmp_slices(x: &[f64], y: &[f64]) -> std::cmp::Ordering {
    x.iter().zip(y).map(|(a, b)| a.total_cmp(b)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal)
}

impl PairKernel for DistanceCorrelation {
    type Shared = ();
    type Prepared = DcorRow;
    type Scratch = Fenwick;

    fn shared(&self, _: usize, _: usize, _: &PvalueSpec) -> Result<()> {
        Ok(())
    }

    fn scratch(&self, _: usize) -> Fenwick {
        Fenwick::default()
    }

    fn prepare(&self, x: &[f64], _: &(), _: &mut Fenwick) -> Option<DcorRow> {
        DcorRow::new(x)
    }

    fn eval(&self, a: &DcorRow, b: &DcorRow, _: &(), f: &mut Fenwick) -> PairStat {
        Self::stat(a, b, f)
    }

    fn eval_gathered(&self, x: &[f64], y: &[f64], _: &(), f: &mut Fenwick) -> PairStat {
        match (DcorRow::new(x), DcorRow::new(y)) {
            (Some(a), Some(b)) => Self::stat(&a, &b, f),
            _ => PairStat::nan(x.len()),
        }
    }

    fn pvalue(&self, _: &PairStat, _: &()) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O(n²) definition.
    fn naive(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let centre = |v: &[f64]| {
            let d: Vec<Vec<f64>> = v.iter().map(|a| v.iter().map(|b| (a - b).abs()).collect()).collect();
            let rm: Vec<f64> = d.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
            let gm = rm.iter().sum::<f64>() / n as f64;
            (0..n)
                .map(|k| (0..n).map(|l| d[k][l] - rm[k] - rm[l] + gm).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let (a, b) = (centre(x), centre(y));
        let dot = |p: &Vec<Vec<f64>>, q: &Vec<Vec<f64>>| -> f64 {
            p.iter().flatten().zip(q.iter().flatten()).map(|(u, v)| u * v).sum()
        };
        (dot(&a, &b) / (dot(&a, &a) * dot(&b, &b)).sqrt()).max(0.0).sqrt()
    }

    #[test]
    fn matches_definition_with_ties() {
        let x = [1.0, 3.0, 2.0, 2.0, 8.0, -1.0, 4.0, 4.0, 0.5];
        let y = [2.0, 1.0, 1.0, 5.0, 3.0, 3.0, -2.0, 7.0, 0.0];
        let (a, b) = (DcorRow::new(&x).unwrap(), DcorRow::new(&y).unwrap());
        let got = DistanceCorrelation::stat(&a, &b, &mut Fenwick::default()).value;
        assert!((got - naive(&x, &y)).abs() < 1e-13, "{got} vs {}", naive(&x, &y));
        let same = DistanceCorrelation::stat(&a, &a, &mut Fenwick::default()).value;
        assert!((same - 1.0).abs() < 1e-14);
    }
}
