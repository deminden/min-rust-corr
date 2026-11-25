//! Kendall's tau-b with tie correction.
//!
//! tau_b = S / √((n0 − n1)(n0 − n2)); p-values use the tie-corrected normal
//! approximation without continuity correction (equal to R's
//! `cor.test(method = "kendall", exact = FALSE)`).

pub mod knight;
pub mod variance;

use crate::engine::{PairKernel, PairStat};
use crate::error::Result;
use crate::options::PvalueSpec;
use crate::prep::rank::{TieSums, dense_ranks};
use crate::prep::standardize::all_equal;
use knight::{KnightScratch, kendall_s};

/// Kendall tau-b kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Kendall;

/// A row prepared once: sort permutation, dense ranks and tie sums.
#[derive(Debug, Clone, Default)]
pub struct KendallRow {
    perm: Vec<u32>,
    ranks: Vec<u32>,
    ties: TieSums,
}

impl KendallRow {
    /// Fill from finite values; false when degenerate (n < 2 or constant).
    fn fill(&mut self, x: &[f64], counts: &mut Vec<u32>) -> bool {
        if x.len() < 2 || all_equal(x) {
            return false;
        }
        self.ranks.resize(x.len(), 0);
        dense_ranks(x, &mut self.perm, &mut self.ranks);
        self.ties = TieSums::from_dense(&self.ranks, counts);
        true
    }
}

/// Per-thread scratch.
#[derive(Debug, Default)]
pub struct KendallScratch {
    knight: KnightScratch,
    counts: Vec<u32>,
    gx: KendallRow,
    gy: KendallRow,
}

fn tau_b(a: &KendallRow, b: &KendallRow, s: &mut KnightScratch) -> PairStat {
    let n = a.ranks.len();
    let sv = kendall_s(&a.perm, &a.ranks, &b.ranks, a.ties.pairs, b.ties.pairs, s);
    let n0 = (n as u64) * (n as u64 - 1) / 2;
    let den = ((n0 - a.ties.pairs) as f64 * (n0 - b.ties.pairs) as f64).sqrt();
    let value = if den > 0.0 { sv as f64 / den } else { f64::NAN };
    PairStat { value, n_eff: n, aux: variance::z_score(sv, n, &a.ties, &b.ties) }
}

impl PairKernel for Kendall {
    type Shared = ();
    type Prepared = KendallRow;
    type Scratch = KendallScratch;

    fn shared(&self, _: usize, _: &PvalueSpec) -> Result<()> {
        Ok(())
    }

    fn scratch(&self, _: usize) -> KendallScratch {
        KendallScratch::default()
    }

    fn prepare(&self, x: &[f64], _: &(), s: &mut KendallScratch) -> Option<KendallRow> {
        let mut r = KendallRow::default();
        r.fill(x, &mut s.counts).then_some(r)
    }

    fn eval(&self, a: &KendallRow, b: &KendallRow, _: &(), s: &mut KendallScratch) -> PairStat {
        tau_b(a, b, &mut s.knight)
    }

    fn eval_gathered(&self, x: &[f64], y: &[f64], _: &(), s: &mut KendallScratch) -> PairStat {
        if !s.gx.fill(x, &mut s.counts) || !s.gy.fill(y, &mut s.counts) {
            return PairStat::nan(x.len());
        }
        tau_b(&s.gx, &s.gy, &mut s.knight)
    }

    fn pvalue(&self, st: &PairStat, _: &()) -> f64 {
        variance::pvalue(st.aux, st.n_eff)
    }
}

/// Kendall's S for two finite slices (convenience for tests).
pub fn s_statistic(x: &[f64], y: &[f64]) -> Option<i64> {
    let (mut a, mut b, mut c) = (KendallRow::default(), KendallRow::default(), Vec::new());
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    // Constant rows still have a well-defined S (= 0 contribution).
    a.ranks.resize(x.len(), 0);
    b.ranks.resize(y.len(), 0);
    dense_ranks(x, &mut a.perm, &mut a.ranks);
    dense_ranks(y, &mut b.perm, &mut b.ranks);
    a.ties = TieSums::from_dense(&a.ranks, &mut c);
    b.ties = TieSums::from_dense(&b.ranks, &mut c);
    Some(kendall_s(&a.perm, &a.ranks, &b.ranks, a.ties.pairs, b.ties.pairs, &mut KnightScratch::default()))
}
