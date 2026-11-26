//! Method plug-ins and the single place that maps a [`Method`] to an
//! engine.

pub mod bicor;
pub mod chatterjee;
pub mod covariates;
pub mod dcor;
pub mod hellcor;
pub mod kendall;
pub mod rho;
pub mod scores;

use crate::engine::{PairKernel, RowTransform};
use crate::error::{CorrError, Result};
use crate::options::{Method, PvalueSpec};
use ndarray::ArrayView2;
use scores::{Scored, Scores};

/// Receives the concrete method implementation.
pub trait MethodVisitor {
    /// Output of the visit.
    type Out;
    /// Called for Gram methods.
    fn gram<T: RowTransform>(self, t: &T) -> Result<Self::Out>;
    /// Called for pairwise-kernel methods.
    fn pairwise<K: PairKernel>(self, k: &K) -> Result<Self::Out>;
}

/// Covariates (rows = covariates, columns = the `n_obs` observations) for
/// partial correlation.
#[derive(Debug, Clone, Copy)]
pub struct Covariates<'a> {
    /// Covariate matrix.
    pub data: ArrayView2<'a, f64>,
    /// Number of observations of the correlated data.
    pub n_obs: usize,
}

fn scored<V: MethodVisitor>(sc: Scores, cov: Option<Covariates<'_>>, v: V) -> Result<V::Out> {
    match cov {
        None => v.gram(&Scored::new(sc)),
        Some(c) => v.gram(&Scored::adjusted(sc, covariates::Basis::new(c.data, sc, c.n_obs)?)),
    }
}

/// Dispatch `m` to `v` (the only `match` on methods in the crate).
pub fn visit<V: MethodVisitor>(m: &Method, cov: Option<Covariates<'_>>, v: V) -> Result<V::Out> {
    if cov.is_some() && !m.supports_covariates() {
        return Err(CorrError::Unsupported(format!(
            "covariates are supported by pearson, spearman and gaussian_rank, not {}",
            m.name()
        )));
    }
    match m {
        Method::Pearson => scored(Scores::Raw, cov, v),
        Method::Spearman => scored(Scores::Ranks, cov, v),
        Method::GaussianRank => scored(Scores::NormalScores, cov, v),
        Method::Kendall => v.pairwise(&kendall::Kendall),
        Method::Proportionality(p) => {
            p.validate()?;
            v.gram(&rho::Rho)
        }
        Method::Chatterjee(p) => v.pairwise(&chatterjee::Chatterjee::new(*p)),
        Method::DistanceCorrelation => v.pairwise(&dcor::DistanceCorrelation),
        Method::Bicor(p) => {
            p.validate()?;
            v.gram(&bicor::Bicor::new(*p))
        }
        Method::Hellcor(p) => {
            p.validate()?;
            v.pairwise(&hellcor::Hellcor::new(*p))
        }
    }
}

/// Check that the p-value mode is available for the method.
pub fn check_pvalues(m: &Method, p: &PvalueSpec) -> Result<()> {
    let hell = matches!(m, Method::Hellcor(_));
    let none = matches!(m, Method::Proportionality(_) | Method::DistanceCorrelation);
    match p {
        PvalueSpec::None => Ok(()),
        _ if none => Err(CorrError::Unsupported(format!("p-values are not available for {}", m.name()))),
        PvalueSpec::Analytic if hell => Err(CorrError::Unsupported(
            "analytic p-values are not available for hellcor; use Monte Carlo".into(),
        )),
        PvalueSpec::MonteCarlo { .. } if !hell => Err(CorrError::Unsupported(format!(
            "Monte-Carlo p-values are only implemented for hellcor, not {}",
            m.name()
        ))),
        PvalueSpec::MonteCarlo { samples: 0, .. } => {
            Err(CorrError::InvalidParam { name: "mc_samples", reason: "must be > 0".into() })
        }
        _ => Ok(()),
    }
}
