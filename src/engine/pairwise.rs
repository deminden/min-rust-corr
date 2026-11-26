//! Pairwise-kernel engine: rows prepared once in parallel, kernel evaluated
//! per pair with per-thread scratch; rows with missing values take the
//! gathered slow path.

use super::dispatch::{Grid, RowState, row_states};
use super::{Engine, PairKernel, PairStat, clamp_corr};
use crate::error::Result;
use crate::options::NanPolicy;
use crate::prep::standardize::all_equal;
use crate::prep::validity::{gather_common, gather_finite, to_contiguous};
use ndarray::ArrayView2;
use rayon::prelude::*;
use std::ops::Range;

/// One input matrix after preparation.
pub struct PairSide<'a, P> {
    data: ArrayView2<'a, f64>,
    prep: Vec<Option<P>>,
    state: Vec<RowState>,
}

impl<'a, P: Send + Sync> PairSide<'a, P> {
    /// Prepare every row of `data` in parallel.
    pub fn build<K>(k: &K, sh: &K::Shared, data: ArrayView2<'a, f64>, policy: NanPolicy) -> Result<Self>
    where
        K: PairKernel<Prepared = P>,
    {
        let mut state = row_states(data, policy)?;
        let m = data.ncols();
        let prep: Vec<Option<P>> = state
            .par_iter_mut()
            .enumerate()
            .map_init(
                || (k.scratch(m), Vec::new()),
                |(s, buf), (i, st)| {
                    let row = data.row(i);
                    match *st {
                        RowState::Dead => None,
                        RowState::Clean => {
                            let p = k.prepare(to_contiguous(row, buf), sh, s);
                            if p.is_none() {
                                *st = RowState::Dead;
                            }
                            p
                        }
                        RowState::Missing => {
                            gather_finite(row, buf);
                            if buf.len() < 2 || all_equal(buf) {
                                *st = RowState::Dead;
                            }
                            None
                        }
                    }
                },
            )
            .collect();
        Ok(Self { data, prep, state })
    }
}

/// Pairwise engine over one (symmetric) or two (cross) sides.
pub struct PairEngine<'a, K: PairKernel> {
    k: &'a K,
    sh: &'a K::Shared,
    a: PairSide<'a, K::Prepared>,
    b: Option<PairSide<'a, K::Prepared>>,
    n_obs: usize,
}

/// Per-thread scratch of the pairwise engine.
pub struct PairScratch<S> {
    inner: S,
    gx: Vec<f64>,
    gy: Vec<f64>,
}

impl<'a, K: PairKernel> PairEngine<'a, K> {
    /// Build a symmetric (`b = None`) or cross engine.
    pub fn new(
        k: &'a K,
        sh: &'a K::Shared,
        a: PairSide<'a, K::Prepared>,
        b: Option<PairSide<'a, K::Prepared>>,
    ) -> Self {
        let n_obs = a.data.ncols();
        Self { k, sh, a, b, n_obs }
    }

    fn side_b(&self) -> &PairSide<'a, K::Prepared> {
        self.b.as_ref().unwrap_or(&self.a)
    }

    fn pair(&self, i: usize, j: usize, s: &mut PairScratch<K::Scratch>) -> PairStat {
        let b = self.side_b();
        match (&self.a.prep[i], &b.prep[j]) {
            (Some(pa), Some(pb)) => self.k.eval(pa, pb, self.sh, &mut s.inner),
            _ => {
                gather_common(self.a.data.row(i), b.data.row(j), &mut s.gx, &mut s.gy);
                self.k.eval_gathered(&s.gx, &s.gy, self.sh, &mut s.inner)
            }
        }
    }
}

impl<K: PairKernel> Engine for PairEngine<'_, K> {
    type Scratch = PairScratch<K::Scratch>;

    fn scratch(&self) -> Self::Scratch {
        PairScratch { inner: self.k.scratch(self.n_obs), gx: vec![], gy: vec![] }
    }

    fn n_cols(&self) -> usize {
        self.side_b().state.len()
    }

    fn fill(
        &self,
        rows: Range<usize>,
        out: &mut [f64],
        pout: Option<&mut [f64]>,
        upper_only: bool,
        s: &mut Self::Scratch,
    ) {
        let b = self.side_b();
        let want_p = pout.is_some();
        let grid = Grid { sym: self.b.is_none(), a: &self.a.state, b: &b.state, upper: upper_only };
        grid.fill(rows, out, pout, |i, j, _, _| {
            let st = self.pair(i, j, s);
            let r = if st.value.is_nan() { st.value } else { clamp_corr(st.value) };
            (r, if want_p { self.k.pvalue(&st, self.sh) } else { f64::NAN })
        });
    }
}
