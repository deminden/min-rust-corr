//! User-facing options: method selection, NaN policy and p-value mode.
//! Method parameters (bicor, hellcor, flat settings) are re-exported here.

mod params;

pub use params::{
    BicorParams, ChatterjeeParams, HellcorCriterion, HellcorParams, MadZeroFallback, MethodSettings,
    ProportionalityParams,
};

use crate::error::{CorrError, Result};
use std::str::FromStr;

/// Correlation method.
#[derive(Debug, Clone, PartialEq)]
pub enum Method {
    /// Pearson product-moment correlation.
    Pearson,
    /// Spearman rank correlation (Pearson on average ranks).
    Spearman,
    /// Kendall tau-b (tie-corrected).
    Kendall,
    /// Biweight midcorrelation.
    Bicor(BicorParams),
    /// Hellinger correlation (Geenens & Lafaye de Micheaux, 2022).
    Hellcor(HellcorParams),
    /// Gaussian-rank correlation: Pearson on normal scores Φ⁻¹(R/(n+1)).
    GaussianRank,
    /// Proportionality ρp = 1 − var(x − y)/(var x + var y) on log-ratios.
    Proportionality(ProportionalityParams),
    /// Chatterjee's rank correlation ξ (asymmetric by default).
    Chatterjee(ChatterjeeParams),
    /// Distance correlation (Székely, Rizzo & Bakirov 2007).
    DistanceCorrelation,
}

impl Method {
    /// Lower-case method name (`"pearson"`, `"spearman"`, ...).
    pub fn name(&self) -> &'static str {
        match self {
            Method::Pearson => "pearson",
            Method::Spearman => "spearman",
            Method::Kendall => "kendall",
            Method::Bicor(_) => "bicor",
            Method::Hellcor(_) => "hellcor",
            Method::GaussianRank => "gaussian_rank",
            Method::Proportionality(_) => "rho",
            Method::Chatterjee(_) => "xi",
            Method::DistanceCorrelation => "dcor",
        }
    }

    /// Parse a method name with default parameters; case-insensitive.
    /// Aliases: `biweight`, `hellinger` (v0.2), `gaussrank`,
    /// `proportionality`, `chatterjee`, `distance`.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pearson" => Ok(Method::Pearson),
            "spearman" => Ok(Method::Spearman),
            "kendall" => Ok(Method::Kendall),
            "bicor" | "biweight" => Ok(Method::Bicor(BicorParams::default())),
            "hellcor" | "hellinger" => Ok(Method::Hellcor(HellcorParams::default())),
            "gaussian_rank" | "gaussian-rank" | "gaussrank" => Ok(Method::GaussianRank),
            "rho" | "proportionality" => Ok(Method::Proportionality(ProportionalityParams::default())),
            "xi" | "chatterjee" => Ok(Method::Chatterjee(ChatterjeeParams::default())),
            "dcor" | "distance" => Ok(Method::DistanceCorrelation),
            other => {
                Err(CorrError::InvalidParam { name: "method", reason: format!("unknown method '{other}'") })
            }
        }
    }

    /// Parse a method name and apply `s` to its parameters.
    pub fn with_settings(name: &str, s: &MethodSettings) -> Result<Self> {
        Ok(match Self::from_name(name)? {
            Method::Bicor(p) => Method::Bicor(BicorParams { max_p_outliers: s.max_p_outliers, ..p }),
            Method::Hellcor(p) => {
                Method::Hellcor(HellcorParams { alpha: s.alpha, kmax: s.kmax, lmax: s.lmax, ..p })
            }
            Method::Proportionality(_) => {
                Method::Proportionality(ProportionalityParams { pseudocount: s.pseudocount, clr: s.clr })
            }
            Method::Chatterjee(_) => Method::Chatterjee(ChatterjeeParams { symmetric: s.xi_symmetric }),
            m => m,
        })
    }

    /// True when the method is evaluated pair by pair (Kendall, Hellcor,
    /// ξ, distance correlation) rather than through a matrix product.
    pub fn is_pairwise(&self) -> bool {
        matches!(
            self,
            Method::Kendall | Method::Hellcor(_) | Method::Chatterjee(_) | Method::DistanceCorrelation
        )
    }

    /// True when cell (i, j) always equals cell (j, i).
    pub fn is_symmetric(&self) -> bool {
        !matches!(self, Method::Chatterjee(p) if !p.symmetric)
    }

    /// True when the method accepts covariates (partial correlation).
    pub fn supports_covariates(&self) -> bool {
        matches!(self, Method::Pearson | Method::Spearman | Method::GaussianRank)
    }
}

/// How non-finite values (NaN, ±inf) are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NanPolicy {
    /// Any non-finite value is an error.
    Error,
    /// A row containing any non-finite value yields NaN for all its pairs.
    Propagate,
    /// Each pair uses the observations finite in both rows (default).
    #[default]
    PairwiseComplete,
}

/// P-value computation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PvalueSpec {
    /// No p-values.
    #[default]
    None,
    /// Analytic (asymptotic) p-values: Student t for Pearson, Spearman,
    /// Gaussian rank and bicor; tie-corrected normal approximation for
    /// Kendall; one-sided normal approximation for ξ. Not available for
    /// ρp and distance correlation.
    Analytic,
    /// Monte-Carlo permutation null (Hellcor only).
    MonteCarlo {
        /// Number of null replicates B; resolution is 1/(B+1).
        samples: usize,
        /// Seed of the random streams.
        seed: u64,
    },
}

impl FromStr for NanPolicy {
    type Err = CorrError;

    /// `pairwise` / `pairwise_complete`, `propagate`, `error` / `raise`.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "pairwise" | "pairwise_complete" | "pairwise-complete" => Ok(NanPolicy::PairwiseComplete),
            "propagate" => Ok(NanPolicy::Propagate),
            "error" | "raise" => Ok(NanPolicy::Error),
            other => Err(CorrError::InvalidParam {
                name: "nan_policy",
                reason: format!("unknown policy '{other}'"),
            }),
        }
    }
}

impl PvalueSpec {
    /// The natural p-value mode for `method`: Monte Carlo for hellcor,
    /// analytic otherwise.
    pub fn auto(method: &Method, samples: usize, seed: u64) -> Self {
        match method {
            Method::Hellcor(_) => PvalueSpec::MonteCarlo { samples, seed },
            _ => PvalueSpec::Analytic,
        }
    }

    /// True unless `None`.
    pub fn wanted(&self) -> bool {
        !matches!(self, PvalueSpec::None)
    }
}
