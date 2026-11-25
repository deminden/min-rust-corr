//! Single-pair evaluation (no matrix construction).

use crate::engine::gram::dot;
use crate::engine::{PairKernel, RowTransform, clamp_corr};
use crate::error::{CorrError, Result};
use crate::layout::PairResult;
use crate::methods::MethodVisitor;
use crate::options::{NanPolicy, PvalueSpec};
use crate::prep::RowStatus;
use crate::stats::tdist::corr_t_pvalue;

pub struct PairV<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub policy: NanPolicy,
    pub pvalues: PvalueSpec,
}

impl PairV<'_> {
    /// Apply the NaN policy; `None` means the result is NaN (Propagate).
    fn gather(&self) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
        if self.x.len() != self.y.len() {
            return Err(CorrError::ShapeMismatch {
                what: "vector length",
                left: self.x.len(),
                right: self.y.len(),
            });
        }
        if self.x.is_empty() {
            return Err(CorrError::Empty { what: "vectors" });
        }
        let bad = |v: &[f64]| v.iter().position(|a| !a.is_finite());
        match self.policy {
            NanPolicy::Error => {
                if let Some(col) = bad(self.x) {
                    return Err(CorrError::NonFinite { row: 0, col });
                }
                if let Some(col) = bad(self.y) {
                    return Err(CorrError::NonFinite { row: 1, col });
                }
            }
            NanPolicy::Propagate if bad(self.x).is_some() || bad(self.y).is_some() => return Ok(None),
            _ => {}
        }
        Ok(Some(
            self.x
                .iter()
                .zip(self.y)
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .map(|(a, b)| (*a, *b))
                .unzip(),
        ))
    }

    fn nan(&self) -> PairResult {
        PairResult { corr: f64::NAN, pvalue: self.pvalues.wanted().then_some(f64::NAN), n_eff: self.x.len() }
    }
}

impl MethodVisitor for PairV<'_> {
    type Out = PairResult;

    fn gram<T: RowTransform>(self, t: &T) -> Result<PairResult> {
        let Some((gx, gy)) = self.gather()? else { return Ok(self.nan()) };
        let n = gx.len();
        let (mut tx, mut ty, mut s) = (vec![0.0; n], vec![0.0; n], T::Scratch::default());
        let ok = t.transform(&gx, &mut tx, &mut s) == RowStatus::Ok
            && t.transform(&gy, &mut ty, &mut s) == RowStatus::Ok;
        let corr = if ok { clamp_corr(dot(&tx, &ty)) } else { f64::NAN };
        Ok(PairResult { corr, pvalue: self.pvalues.wanted().then(|| corr_t_pvalue(corr, n)), n_eff: n })
    }

    fn pairwise<K: PairKernel>(self, k: &K) -> Result<PairResult> {
        let Some((gx, gy)) = self.gather()? else { return Ok(self.nan()) };
        let sh = k.shared(self.x.len(), &self.pvalues)?;
        let st = k.eval_gathered(&gx, &gy, &sh, &mut k.scratch(self.x.len()));
        let corr = if st.value.is_nan() { st.value } else { clamp_corr(st.value) };
        Ok(PairResult { corr, pvalue: self.pvalues.wanted().then(|| k.pvalue(&st, &sh)), n_eff: gx.len() })
    }
}
