//! Score-based Gram methods: Pearson (raw values), Spearman (average
//! ranks) and Gaussian-rank correlation (normal scores), optionally
//! adjusted for covariates (partial correlation).
//!
//! Each row is mapped to its scores, centred, residualised against the
//! covariate basis (when given) and scaled to unit norm; r is then a dot
//! product. P-values use t = r√(df/(1−r²)) with df = n − 2 − k, where k is
//! the rank of the covariate matrix (k = 0 without covariates).
//!
//! Gaussian-rank scores are Φ⁻¹(R/(n+1)) with average ranks R (Boudt,
//! Cornelissen & Croux 2012, Stat. Comput. 22(2)).

use super::covariates::Basis;
use crate::engine::RowTransform;
use crate::prep::RowStatus;
use crate::prep::rank::average_ranks;
use crate::prep::standardize::{center_unit_norm, scale_unit_norm};
use crate::stats::normal::quantile;
use crate::stats::tdist::corr_t_pvalue;

/// What a row is replaced by before the Pearson computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scores {
    /// The values themselves (Pearson).
    Raw,
    /// Average ranks (Spearman).
    Ranks,
    /// Normal scores Φ⁻¹(rank/(n+1)) (Gaussian rank).
    NormalScores,
}

/// Scratch for ranking.
#[derive(Debug, Default)]
pub struct ScoreScratch {
    idx: Vec<u32>,
    buf: Vec<f64>,
}

impl Scores {
    /// Scores of `x` (borrowed for `Raw`, otherwise written into scratch).
    pub fn apply<'a>(self, x: &'a [f64], s: &'a mut ScoreScratch) -> &'a [f64] {
        if self == Scores::Raw {
            return x;
        }
        s.buf.resize(x.len(), 0.0);
        average_ranks(x, &mut s.idx, &mut s.buf);
        if self == Scores::NormalScores {
            let d = x.len() as f64 + 1.0;
            s.buf.iter_mut().for_each(|r| *r = quantile(*r / d));
        }
        &s.buf
    }
}

/// Gram transform of a score method with optional covariate adjustment.
#[derive(Debug, Clone)]
pub struct Scored {
    scores: Scores,
    adjust: Option<Basis>,
}

impl Scored {
    /// Unadjusted transform.
    pub fn new(scores: Scores) -> Self {
        Self { scores, adjust: None }
    }

    /// Transform residualised against `basis`.
    pub fn adjusted(scores: Scores, basis: Basis) -> Self {
        Self { scores, adjust: Some(basis) }
    }
}

impl RowTransform for Scored {
    type Scratch = ScoreScratch;

    fn transform(&self, x: &[f64], out: &mut [f64], s: &mut ScoreScratch) -> RowStatus {
        let st = center_unit_norm(self.scores.apply(x, s), out);
        match &self.adjust {
            None => st,
            Some(_) if st != RowStatus::Ok => st,
            Some(b) if b.n_obs() != x.len() => {
                out.fill(0.0);
                RowStatus::Degenerate
            }
            Some(b) => {
                let left = b.residualize(out);
                scale_unit_norm(out, left > 1e-12)
            }
        }
    }

    fn pvalue(&self, r: f64, n: usize) -> f64 {
        let k = self.adjust.as_ref().map_or(0, Basis::rank);
        n.checked_sub(k).map_or(f64::NAN, |m| corr_t_pvalue(r, m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_scores_are_symmetric() {
        let mut s = ScoreScratch::default();
        let z = Scores::NormalScores.apply(&[3.0, 1.0, 2.0], &mut s).to_vec();
        assert!((z[0] - 0.674_489_750_196_081_7).abs() < 1e-15);
        assert_eq!(z[2], 0.0);
        assert_eq!(z[1], -z[0]);
    }
}
