//! The [`Correlator`] builder: the public entry point for every layout.

use crate::engine::gram::{GramEngine, GramSide};
use crate::engine::pairwise::{PairEngine, PairSide};
use crate::engine::{PairKernel, RowTransform};
use crate::error::Result;
use crate::layout::{CorrOutput, PackedOutput, PairResult, RowBlock};
use crate::methods::{MethodVisitor, check_pvalues, visit};
use crate::options::{Method, NanPolicy, PvalueSpec};
use ndarray::ArrayView2;

mod exec;
mod pair;

/// Configured correlation computation.
///
/// ```
/// use mincorr::{Correlator, Method, PvalueSpec};
/// use ndarray::array;
/// let x = array![[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0]];
/// let out = Correlator::new(Method::Spearman).pvalues(PvalueSpec::Analytic).full(x.view()).unwrap();
/// assert_eq!(out.corr[[0, 1]], 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Correlator {
    method: Method,
    nan_policy: NanPolicy,
    pvalues: PvalueSpec,
    block_rows: Option<usize>,
}

impl Correlator {
    /// New correlator with default options (pairwise-complete NaN handling,
    /// no p-values).
    pub fn new(method: Method) -> Self {
        Self { method, nan_policy: NanPolicy::default(), pvalues: PvalueSpec::None, block_rows: None }
    }

    /// Set the NaN policy.
    pub fn nan_policy(mut self, p: NanPolicy) -> Self {
        self.nan_policy = p;
        self
    }

    /// Set the p-value mode.
    pub fn pvalues(mut self, p: PvalueSpec) -> Self {
        self.pvalues = p;
        self
    }

    /// Rows per streamed block (default: ~48 MB per block).
    pub fn block_rows(mut self, rows: usize) -> Self {
        self.block_rows = Some(rows);
        self
    }

    /// The configured method.
    pub fn method(&self) -> &Method {
        &self.method
    }

    /// The configured p-value mode.
    pub fn pvalue_spec(&self) -> PvalueSpec {
        self.pvalues
    }

    /// Full symmetric matrix of all row pairs of `data` (rows = variables).
    pub fn full(&self, data: ArrayView2<'_, f64>) -> Result<CorrOutput> {
        self.run(data, None, exec::Full)
    }

    /// Cross matrix between the rows of `a` and the rows of `b`.
    pub fn cross(&self, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<CorrOutput> {
        self.run(a, Some(b), exec::Cross)
    }

    /// Packed upper triangle (diagonal included) of [`Correlator::full`].
    pub fn upper(&self, data: ArrayView2<'_, f64>) -> Result<PackedOutput> {
        self.run(data, None, exec::Upper { block_rows: self.block_rows })
    }

    /// Stream the full matrix in row blocks, in row order. Each block holds
    /// complete rows, so symmetric entries are computed twice; peak memory
    /// is one block.
    pub fn full_blocks(
        &self,
        data: ArrayView2<'_, f64>,
        sink: impl FnMut(RowBlock<'_>) -> Result<()>,
    ) -> Result<()> {
        self.run(data, None, exec::Blocks { sink, block_rows: self.block_rows })
    }

    /// Stream the cross matrix in row blocks of `a`, in row order.
    pub fn cross_blocks(
        &self,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        sink: impl FnMut(RowBlock<'_>) -> Result<()>,
    ) -> Result<()> {
        self.run(a, Some(b), exec::Blocks { sink, block_rows: self.block_rows })
    }

    /// Correlation of two vectors.
    pub fn pair(&self, x: &[f64], y: &[f64]) -> Result<PairResult> {
        check_pvalues(&self.method, &self.pvalues)?;
        visit(&self.method, pair::PairV { x, y, policy: self.nan_policy, pvalues: self.pvalues })
    }

    fn run<U: exec::EngineUser>(
        &self,
        a: ArrayView2<'_, f64>,
        b: Option<ArrayView2<'_, f64>>,
        user: U,
    ) -> Result<U::Out> {
        check_pvalues(&self.method, &self.pvalues)?;
        match b {
            Some(b) => crate::engine::dispatch::check_cross(a, b)?,
            None => crate::engine::dispatch::check_shape(a, "matrix")?,
        }
        visit(&self.method, Build { a, b, policy: self.nan_policy, pvalues: self.pvalues, user })
    }
}

struct Build<'a, U> {
    a: ArrayView2<'a, f64>,
    b: Option<ArrayView2<'a, f64>>,
    policy: NanPolicy,
    pvalues: PvalueSpec,
    user: U,
}

impl<U: exec::EngineUser> MethodVisitor for Build<'_, U> {
    type Out = U::Out;

    fn gram<T: RowTransform>(self, t: &T) -> Result<U::Out> {
        let sa = GramSide::build(t, self.a, self.policy)?;
        let sb = self.b.map(|b| GramSide::build(t, b, self.policy)).transpose()?;
        let e = GramEngine::new(t, sa, sb);
        self.user.run(&e, self.a.nrows(), self.pvalues.wanted())
    }

    fn pairwise<K: PairKernel>(self, k: &K) -> Result<U::Out> {
        let sh = k.shared(self.a.ncols(), &self.pvalues)?;
        let sa = PairSide::build(k, &sh, self.a, self.policy)?;
        let sb = self.b.map(|b| PairSide::build(k, &sh, b, self.policy)).transpose()?;
        let e = PairEngine::new(k, &sh, sa, sb);
        self.user.run(&e, self.a.nrows(), self.pvalues.wanted())
    }
}
