//! Method plug-ins and the single place that maps a [`Method`] to an
//! engine.

pub mod bicor;
pub mod hellcor;
pub mod kendall;
pub mod pearson;
pub mod spearman;

use crate::engine::{PairKernel, RowTransform};
use crate::error::{CorrError, Result};
use crate::options::{Method, PvalueSpec};

/// Receives the concrete method implementation.
pub trait MethodVisitor {
    /// Output of the visit.
    type Out;
    /// Called for Gram methods.
    fn gram<T: RowTransform>(self, t: &T) -> Result<Self::Out>;
    /// Called for pairwise-kernel methods.
    fn pairwise<K: PairKernel>(self, k: &K) -> Result<Self::Out>;
}

/// Dispatch `m` to `v` (the only `match` on methods in the crate).
pub fn visit<V: MethodVisitor>(m: &Method, v: V) -> Result<V::Out> {
    match m {
        Method::Pearson => v.gram(&pearson::Pearson),
        Method::Spearman => v.gram(&spearman::Spearman),
        Method::Kendall => v.pairwise(&kendall::Kendall),
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
    match p {
        PvalueSpec::None => Ok(()),
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
