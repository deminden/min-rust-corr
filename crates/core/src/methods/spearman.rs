//! Spearman rank correlation: Pearson correlation of average ranks
//! (ties get the mean of their positions). P-values use the Student-t
//! approximation, equivalent to R's `cor.test(..., exact = FALSE)`.

use super::pearson::Pearson;
use crate::engine::RowTransform;
use crate::prep::RowStatus;
use crate::prep::rank::average_ranks;

/// Spearman row transform: average ranks, then the Pearson transform.
#[derive(Debug, Clone, Copy, Default)]
pub struct Spearman;

/// Scratch for ranking.
#[derive(Debug, Default)]
pub struct SpearmanScratch {
    idx: Vec<u32>,
    ranks: Vec<f64>,
}

impl RowTransform for Spearman {
    type Scratch = SpearmanScratch;

    fn transform(&self, x: &[f64], out: &mut [f64], s: &mut SpearmanScratch) -> RowStatus {
        s.ranks.resize(x.len(), 0.0);
        average_ranks(x, &mut s.idx, &mut s.ranks);
        Pearson.transform(&s.ranks, out, &mut ())
    }
}
