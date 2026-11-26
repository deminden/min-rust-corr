//! The [`Correlator`] builder: the public entry point for every layout.

use crate::engine::gram::{GramEngine, GramSide};
use crate::engine::pairwise::{PairEngine, PairSide};
use crate::engine::{PairKernel, RowTransform};
use crate::error::Result;
use crate::layout::{CorrOutput, PackedOutput, PairResult, RowBlock};
use crate::methods::{Covariates, MethodVisitor, check_pvalues, visit};
use crate::options::{Method, NanPolicy, PvalueSpec};
use ndarray::{Array2, ArrayView1, ArrayView2};

mod exec;
mod pair;
mod prepare;

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
    triangle_cache: Option<usize>,
    covariates: Option<Array2<f64>>,
}

impl Correlator {
    /// New correlator with default options (pairwise-complete NaN handling,
    /// no p-values).
    pub fn new(method: Method) -> Self {
        Self {
            method,
            nan_policy: NanPolicy::default(),
            pvalues: PvalueSpec::None,
            block_rows: None,
            triangle_cache: None,
            covariates: None,
        }
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

    /// Let [`Correlator::full_blocks`] compute the symmetric triangle once in
    /// memory when the whole output (correlations and p-values) needs at most
    /// `bytes`; above that it streams complete rows and evaluates each
    /// symmetric pair twice. Off by default (pure streaming).
    pub fn triangle_cache(mut self, bytes: usize) -> Self {
        self.triangle_cache = Some(bytes);
        self
    }

    /// Adjust for covariates (partial correlation): `cov` has one row per
    /// covariate and one column per observation. Every row is residualised
    /// on the covariates (after the method's rank or normal-score
    /// transform, which is also applied to the covariates) before
    /// correlating; p-values use n − 2 − k degrees of freedom, k = rank of
    /// the covariates. Pearson, Spearman and Gaussian rank only. Covariates
    /// must be finite; data with missing values needs
    /// [`NanPolicy::Propagate`] or [`NanPolicy::Error`].
    pub fn covariates(mut self, cov: Array2<f64>) -> Self {
        self.covariates = Some(cov);
        self
    }

    /// The same configuration with another method.
    pub fn method_replaced(mut self, method: Method) -> Self {
        self.method = method;
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
    /// complete rows; peak memory is one block (see
    /// [`Correlator::triangle_cache`] for the trade-off).
    pub fn full_blocks(
        &self,
        data: ArrayView2<'_, f64>,
        sink: impl FnMut(RowBlock<'_>) -> Result<()>,
    ) -> Result<()> {
        let (block_rows, cache) = (self.block_rows, self.triangle_cache);
        self.run(data, None, exec::Blocks { sink, block_rows, cache, symmetric: true })
    }

    /// Stream the cross matrix in row blocks of `a`, in row order.
    pub fn cross_blocks(
        &self,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        sink: impl FnMut(RowBlock<'_>) -> Result<()>,
    ) -> Result<()> {
        self.run(
            a,
            Some(b),
            exec::Blocks { sink, block_rows: self.block_rows, cache: None, symmetric: false },
        )
    }

    /// Correlation of two vectors.
    pub fn pair(&self, x: &[f64], y: &[f64]) -> Result<PairResult> {
        check_pvalues(&self.method, &self.pvalues)?;
        if x.len() != y.len() {
            return Err(crate::CorrError::ShapeMismatch {
                what: "vector length",
                left: x.len(),
                right: y.len(),
            });
        }
        let pv = pair::PairV { x, y, policy: self.nan_policy, pvalues: self.pvalues };
        let both = ndarray::stack![ndarray::Axis(0), ArrayView1::from(x), ArrayView1::from(y)];
        self.check_covariates(&[both.view()])?;
        let cov = self.cov(x.len());
        match prepare::proportional(&self.method, &[both.view()]).and_then(|t| t.into_iter().next()) {
            Some(t) => {
                let (tx, ty) = (t.row(0).to_vec(), t.row(1).to_vec());
                visit(&self.method, cov, pair::PairV { x: &tx, y: &ty, ..pv })
            }
            None => visit(&self.method, cov, pv),
        }
    }

    fn cov(&self, n_obs: usize) -> Option<Covariates<'_>> {
        self.covariates.as_ref().map(|c| Covariates { data: c.view(), n_obs })
    }

    /// Covariate adjustment cannot drop observations per pair.
    fn check_covariates(&self, parts: &[ArrayView2<'_, f64>]) -> Result<()> {
        let missing = parts.iter().any(|p| crate::prep::validity::first_nonfinite(*p).is_some());
        if self.covariates.is_some() && self.nan_policy == NanPolicy::PairwiseComplete && missing {
            return Err(crate::CorrError::Unsupported(
                "covariates with missing values: use the propagate or error NaN policy".into(),
            ));
        }
        Ok(())
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
        let parts: Vec<_> = std::iter::once(a).chain(b).collect();
        self.check_covariates(&parts)?;
        let (policy, pvalues, cov) = (self.nan_policy, self.pvalues, self.cov(a.ncols()));
        match prepare::proportional(&self.method, &parts) {
            Some(t) => {
                let (a, b) = (t[0].view(), t.get(1).map(|m| m.view()));
                visit(&self.method, cov, Build { a, b, policy, pvalues, user })
            }
            None => visit(&self.method, cov, Build { a, b, policy, pvalues, user }),
        }
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
        let rows = self.a.nrows() + self.b.map_or(0, |b| b.nrows());
        let sh = k.shared(self.a.ncols(), rows, &self.pvalues)?;
        let sa = PairSide::build(k, &sh, self.a, self.policy)?;
        let sb = self.b.map(|b| PairSide::build(k, &sh, b, self.policy)).transpose()?;
        let e = PairEngine::new(k, &sh, sa, sb);
        self.user.run(&e, self.a.nrows(), self.pvalues.wanted())
    }
}
