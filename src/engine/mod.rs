//! Computation engines.
//!
//! Every method is either a *Gram* method ([`RowTransform`]: each row maps
//! to a unit-norm vector and correlations are dot products) or a
//! *pairwise-kernel* method ([`PairKernel`]: rows are prepared once and a
//! kernel is evaluated per pair). Both engines implement [`Engine`], which
//! fills row-major blocks of the output; [`fill_parallel`] is the single
//! parallel driver used by every layout.

pub mod dispatch;
pub mod gram;
pub mod pairwise;

use crate::error::Result;
use crate::options::PvalueSpec;
use crate::prep::RowStatus;
use rayon::prelude::*;
use std::ops::Range;

/// Rows handled by one parallel task. Fixed (not derived from the thread
/// count) so results are bit-identical for any number of threads.
pub const TASK_ROWS: usize = 16;

/// Gram methods: one row (finite values only) → unit-norm vector.
pub trait RowTransform: Sync {
    /// Per-thread scratch space.
    type Scratch: Default + Send;
    /// Write the transform of `x` into `out` (same length as `x`).
    fn transform(&self, x: &[f64], out: &mut [f64], s: &mut Self::Scratch) -> RowStatus;

    /// Correlation from the dot product of two transformed rows and their
    /// squared norms. The default suits unit-norm transforms (r = dot).
    fn combine(&self, dot: f64, _sq_a: f64, _sq_b: f64) -> f64 {
        dot
    }

    /// P-value of `r` from `n` observations (default: two-sided Student t,
    /// n − 2 degrees of freedom).
    fn pvalue(&self, r: f64, n: usize) -> f64 {
        crate::stats::tdist::corr_t_pvalue(r, n)
    }
}

/// Statistic produced by a pairwise kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairStat {
    /// Correlation estimate.
    pub value: f64,
    /// Observations used.
    pub n_eff: usize,
    /// Method-specific auxiliary value (e.g. Kendall's z-score).
    pub aux: f64,
}

impl PairStat {
    /// A NaN statistic.
    pub fn nan(n_eff: usize) -> Self {
        Self { value: f64::NAN, n_eff, aux: f64::NAN }
    }
}

/// Pairwise methods: prepare once per row, evaluate per pair.
pub trait PairKernel: Sync {
    /// Tables shared by all rows (depend only on the sample size).
    type Shared: Sync;
    /// Per-row prepared data.
    type Prepared: Send + Sync;
    /// Per-thread scratch space.
    type Scratch: Send;
    /// Build shared state for `n_obs` observations and `n_rows` rows to be
    /// prepared in total (lets a kernel size per-row structures to a memory
    /// budget).
    fn shared(&self, n_obs: usize, n_rows: usize, pvalues: &PvalueSpec) -> Result<Self::Shared>;
    /// Allocate scratch for up to `n_obs` observations.
    fn scratch(&self, n_obs: usize) -> Self::Scratch;
    /// Prepare a complete (all finite) row; `None` when degenerate.
    fn prepare(&self, x: &[f64], sh: &Self::Shared, s: &mut Self::Scratch) -> Option<Self::Prepared>;
    /// Evaluate the kernel on two prepared rows.
    fn eval(
        &self,
        a: &Self::Prepared,
        b: &Self::Prepared,
        sh: &Self::Shared,
        s: &mut Self::Scratch,
    ) -> PairStat;
    /// Evaluate on already-gathered pairwise-complete values.
    fn eval_gathered(&self, x: &[f64], y: &[f64], sh: &Self::Shared, s: &mut Self::Scratch) -> PairStat;
    /// P-value of a statistic.
    fn pvalue(&self, st: &PairStat, sh: &Self::Shared) -> f64;

    /// Whether the statistic is symmetric in its two arguments (the
    /// symmetric layouts then compute one triangle and mirror it).
    fn symmetric(&self) -> bool {
        true
    }
}

/// Something that can fill row-major output blocks.
pub trait Engine: Sync {
    /// Per-thread scratch.
    type Scratch: Send;
    /// Allocate scratch.
    fn scratch(&self) -> Self::Scratch;
    /// Number of output columns.
    fn n_cols(&self) -> usize;
    /// Whether cell (i, j) equals cell (j, i) in a single-matrix layout.
    fn symmetric(&self) -> bool {
        true
    }
    /// Fill rows `rows` into `out` (`rows.len() × n_cols`). With
    /// `upper_only`, only entries with column ≥ row need to be written.
    fn fill(
        &self,
        rows: Range<usize>,
        out: &mut [f64],
        pout: Option<&mut [f64]>,
        upper_only: bool,
        s: &mut Self::Scratch,
    );
}

/// Fill `rows` of the output in parallel tasks of [`TASK_ROWS`] rows.
pub fn fill_parallel<E: Engine>(
    e: &E,
    rows: Range<usize>,
    out: &mut [f64],
    pout: Option<&mut [f64]>,
    upper_only: bool,
) {
    let nc = e.n_cols();
    if nc == 0 || rows.is_empty() {
        return;
    }
    let chunk = TASK_ROWS * nc;
    let start = rows.start;
    let task = |s: &mut E::Scratch, k: usize, o: &mut [f64], p: Option<&mut [f64]>| {
        let r0 = start + k * TASK_ROWS;
        let r1 = r0 + o.len() / nc;
        e.fill(r0..r1, o, p, upper_only, s);
    };
    match pout {
        Some(p) => out
            .par_chunks_mut(chunk)
            .zip(p.par_chunks_mut(chunk))
            .enumerate()
            .for_each_init(|| e.scratch(), |s, (k, (o, p))| task(s, k, o, Some(p))),
        None => out
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each_init(|| e.scratch(), |s, (k, o)| task(s, k, o, None)),
    }
}

/// Remove rounding noise at the boundary: values within 4 ulp of ±1 (or
/// overshooting it) become exactly ±1. Larger overshoot is a bug and trips
/// a debug assertion.
#[inline]
pub fn clamp_corr(r: f64) -> f64 {
    debug_assert!(!(r.abs() > 1.0 + 1e-12), "correlation overshoot {r}");
    if r.abs() >= 1.0 - 4.0 * f64::EPSILON { r.signum() } else { r }
}
