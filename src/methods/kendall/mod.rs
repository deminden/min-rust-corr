//! Kendall's tau-b with tie correction.
//!
//! tau_b = S / √((n0 − n1)(n0 − n2)); p-values use the tie-corrected normal
//! approximation without continuity correction (equal to R's
//! `cor.test(method = "kendall", exact = FALSE)`).

pub mod bitset;
pub mod knight;
mod popcount;
pub mod variance;

use crate::engine::{PairKernel, PairStat};
use crate::error::Result;
use crate::options::PvalueSpec;
use crate::prep::rank::{TieSums, dense_ranks};
use crate::prep::standardize::all_equal;
use bitset::SignBits;
use knight::{KnightScratch, kendall_s};

/// Kendall tau-b kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Kendall;

/// Shared state: whether prepared rows carry pair-sign bitsets.
#[derive(Debug, Clone, Copy)]
pub struct KendallShared {
    bitsets: bool,
}

/// A row prepared once: tie sums plus either pair-sign bitsets or the sort
/// permutation and dense ranks used by Knight's algorithm.
#[derive(Debug, Clone, Default)]
pub struct KendallRow {
    n: usize,
    perm: Vec<u32>,
    ranks: Vec<u32>,
    ties: TieSums,
    bits: Option<SignBits>,
}

impl KendallRow {
    /// Fill from finite values; false when degenerate (n < 2 or constant).
    fn fill(&mut self, x: &[f64], counts: &mut Vec<u32>) -> bool {
        if x.len() < 2 || all_equal(x) {
            return false;
        }
        self.rank(x, counts);
        true
    }

    /// Ranks, permutation and tie sums of `x` (no degeneracy check).
    fn rank(&mut self, x: &[f64], counts: &mut Vec<u32>) {
        self.n = x.len();
        self.ranks.resize(x.len(), 0);
        dense_ranks(x, &mut self.perm, &mut self.ranks);
        self.ties = TieSums::from_dense(&self.ranks, counts);
    }

    /// Kendall's S against `other` (same length).
    fn s(&self, other: &KendallRow, s: &mut KnightScratch) -> i64 {
        match (&self.bits, &other.bits) {
            (Some(x), Some(y)) => x.s(y),
            _ => kendall_s(&self.perm, &self.ranks, &other.ranks, self.ties.pairs, other.ties.pairs, s),
        }
    }

    /// Switch to the bitset representation (ranks are no longer needed).
    fn into_bitsets(mut self) -> Self {
        self.bits = Some(SignBits::new(&self.ranks, self.ties.pairs));
        self.perm = Vec::new();
        self.ranks = Vec::new();
        self
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
    let n = a.n;
    let sv = a.s(b, s);
    let n0 = (n as u64) * (n as u64 - 1) / 2;
    let den = ((n0 - a.ties.pairs) as f64 * (n0 - b.ties.pairs) as f64).sqrt();
    let value = if den > 0.0 { sv as f64 / den } else { f64::NAN };
    PairStat { value, n_eff: n, aux: variance::z_score(sv, n, &a.ties, &b.ties) }
}

impl PairKernel for Kendall {
    type Shared = KendallShared;
    type Prepared = KendallRow;
    type Scratch = KendallScratch;

    fn shared(&self, n_obs: usize, n_rows: usize, _: &PvalueSpec) -> Result<KendallShared> {
        Ok(KendallShared { bitsets: bitset::fits(n_obs, n_rows) })
    }

    fn scratch(&self, _: usize) -> KendallScratch {
        KendallScratch::default()
    }

    fn prepare(&self, x: &[f64], sh: &KendallShared, s: &mut KendallScratch) -> Option<KendallRow> {
        let mut r = KendallRow::default();
        if !r.fill(x, &mut s.counts) {
            return None;
        }
        Some(if sh.bitsets { r.into_bitsets() } else { r })
    }

    fn eval(&self, a: &KendallRow, b: &KendallRow, _: &KendallShared, s: &mut KendallScratch) -> PairStat {
        tau_b(a, b, &mut s.knight)
    }

    fn eval_gathered(&self, x: &[f64], y: &[f64], _: &KendallShared, s: &mut KendallScratch) -> PairStat {
        if !s.gx.fill(x, &mut s.counts) || !s.gy.fill(y, &mut s.counts) {
            return PairStat::nan(x.len());
        }
        tau_b(&s.gx, &s.gy, &mut s.knight)
    }

    fn pvalue(&self, st: &PairStat, _: &KendallShared) -> f64 {
        variance::pvalue(st.aux, st.n_eff)
    }
}

/// Kendall's S for two finite slices (convenience for tests).
pub fn s_statistic(x: &[f64], y: &[f64]) -> Option<i64> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    // Constant rows still have a well-defined S, so no degeneracy check.
    let (mut a, mut b, mut c) = (KendallRow::default(), KendallRow::default(), Vec::new());
    a.rank(x, &mut c);
    b.rank(y, &mut c);
    Some(a.s(&b, &mut KnightScratch::default()))
}
