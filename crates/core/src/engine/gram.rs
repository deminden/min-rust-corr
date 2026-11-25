//! Gram engine: Z = transform(rows), corr = Z_a · Z_bᵀ computed in row
//! blocks with a matrix-product kernel, plus the per-pair slow path for
//! rows with missing values (which reuses the method's transform).

use super::dispatch::{RowState, row_states};
use super::{Engine, RowTransform, clamp_corr};
use crate::error::Result;
use crate::options::NanPolicy;
use crate::prep::RowStatus;
use crate::prep::validity::{gather_common, gather_finite, to_contiguous};
use crate::stats::tdist::corr_t_pvalue;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, s};
use rayon::prelude::*;
use std::ops::Range;

/// One input matrix after transformation.
pub struct GramSide<'a> {
    data: ArrayView2<'a, f64>,
    z: Array2<f64>,
    state: Vec<RowState>,
}

impl<'a> GramSide<'a> {
    /// Transform every row of `data` in parallel.
    pub fn build<T: RowTransform>(t: &T, data: ArrayView2<'a, f64>, policy: NanPolicy) -> Result<Self> {
        let mut state = row_states(data, policy)?;
        let m = data.ncols();
        let mut z = Array2::<f64>::zeros((data.nrows(), m));
        z.axis_iter_mut(Axis(0)).into_par_iter().zip(state.par_iter_mut()).enumerate().for_each_init(
            || (T::Scratch::default(), Vec::new(), Vec::new()),
            |(s, buf, tmp), (i, (mut zrow, st))| {
                let row = data.row(i);
                let ok = match *st {
                    RowState::Dead => true,
                    RowState::Clean => match zrow.as_slice_mut() {
                        Some(out) => t.transform(to_contiguous(row, buf), out, s) == RowStatus::Ok,
                        None => false,
                    },
                    RowState::Missing => {
                        gather_finite(row, buf);
                        tmp.resize(buf.len(), 0.0);
                        t.transform(buf, tmp, s) == RowStatus::Ok
                    }
                };
                if !ok {
                    *st = RowState::Dead;
                    zrow.fill(0.0);
                }
            },
        );
        Ok(Self { data, z, state })
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.z.nrows()
    }
}

/// Gram engine over one (symmetric) or two (cross) sides.
pub struct GramEngine<'a, T: RowTransform> {
    t: &'a T,
    a: GramSide<'a>,
    b: Option<GramSide<'a>>,
}

/// Per-thread scratch of the Gram engine.
pub struct GramScratch<S> {
    inner: S,
    gx: Vec<f64>,
    gy: Vec<f64>,
    tx: Vec<f64>,
    ty: Vec<f64>,
}

impl<'a, T: RowTransform> GramEngine<'a, T> {
    /// Build a symmetric (`b = None`) or cross engine.
    pub fn new(t: &'a T, a: GramSide<'a>, b: Option<GramSide<'a>>) -> Self {
        Self { t, a, b }
    }

    fn side_b(&self) -> &GramSide<'a> {
        self.b.as_ref().unwrap_or(&self.a)
    }

    /// Correlation and n_eff of a pair where at least one row has
    /// missing values.
    fn slow_pair(&self, i: usize, j: usize, s: &mut GramScratch<T::Scratch>) -> (f64, usize) {
        gather_common(self.a.data.row(i), self.side_b().data.row(j), &mut s.gx, &mut s.gy);
        let n = s.gx.len();
        s.tx.resize(n, 0.0);
        s.ty.resize(n, 0.0);
        let ok = self.t.transform(&s.gx, &mut s.tx, &mut s.inner) == RowStatus::Ok
            && self.t.transform(&s.gy, &mut s.ty, &mut s.inner) == RowStatus::Ok;
        if !ok {
            return (f64::NAN, n);
        }
        (clamp_corr(dot(&s.tx, &s.ty)), n)
    }
}

/// Plain dot product.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

impl<T: RowTransform> Engine for GramEngine<'_, T> {
    type Scratch = GramScratch<T::Scratch>;

    fn scratch(&self) -> Self::Scratch {
        GramScratch { inner: T::Scratch::default(), gx: vec![], gy: vec![], tx: vec![], ty: vec![] }
    }

    fn n_cols(&self) -> usize {
        self.side_b().n_rows()
    }

    fn fill(
        &self,
        rows: Range<usize>,
        out: &mut [f64],
        mut pout: Option<&mut [f64]>,
        upper_only: bool,
        s: &mut Self::Scratch,
    ) {
        let sym = self.b.is_none();
        let b = self.side_b();
        let nb = b.n_rows();
        let m = self.a.data.ncols();
        let upper = upper_only && sym;
        let c0 = if upper { rows.start } else { 0 };
        if let Ok(mut ov) = ArrayViewMut2::from_shape((rows.len(), nb), &mut *out) {
            let lhs = self.a.z.slice(s![rows.clone(), ..]);
            let rhs = b.z.slice(s![c0.., ..]);
            general_mat_mul(1.0, &lhs, &rhs.t(), 0.0, &mut ov.slice_mut(s![.., c0..]));
        }
        for (k, i) in rows.enumerate() {
            let j0 = if upper { i } else { 0 };
            for j in j0..nb {
                let idx = k * nb + j;
                let (sa, sb) = (self.a.state[i], b.state[j]);
                let (r, n) = if sym && i == j {
                    (if sa == RowState::Dead { f64::NAN } else { 1.0 }, 0)
                } else if sa == RowState::Dead || sb == RowState::Dead {
                    (f64::NAN, 0)
                } else if sa == RowState::Missing || sb == RowState::Missing {
                    self.slow_pair(i, j, s)
                } else {
                    (clamp_corr(out[idx]), m)
                };
                out[idx] = r;
                if let Some(p) = pout.as_deref_mut() {
                    p[idx] = corr_t_pvalue(r, n);
                }
            }
        }
    }
}
