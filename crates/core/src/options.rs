//! User-facing options: method selection, NaN policy, p-value mode and
//! method parameters.

use crate::error::{CorrError, Result};

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
        }
    }

    /// Parse a method name with default parameters. Accepts the v0.2 aliases
    /// `biweight` and `hellinger`; case-insensitive.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pearson" => Ok(Method::Pearson),
            "spearman" => Ok(Method::Spearman),
            "kendall" => Ok(Method::Kendall),
            "bicor" | "biweight" => Ok(Method::Bicor(BicorParams::default())),
            "hellcor" | "hellinger" => Ok(Method::Hellcor(HellcorParams::default())),
            other => {
                Err(CorrError::InvalidParam { name: "method", reason: format!("unknown method '{other}'") })
            }
        }
    }

    /// True when the method is evaluated pair by pair (Kendall, Hellcor)
    /// rather than through a matrix product.
    pub fn is_pairwise(&self) -> bool {
        matches!(self, Method::Kendall | Method::Hellcor(_))
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
    /// Analytic (asymptotic) p-values: Student t for Pearson, Spearman and
    /// bicor; tie-corrected normal approximation for Kendall.
    Analytic,
    /// Monte-Carlo permutation null (Hellcor only).
    MonteCarlo {
        /// Number of null replicates B; resolution is 1/(B+1).
        samples: usize,
        /// Seed of the random streams.
        seed: u64,
    },
}

impl PvalueSpec {
    /// True unless `None`.
    pub fn wanted(&self) -> bool {
        !matches!(self, PvalueSpec::None)
    }
}

/// What bicor does for a row whose median absolute deviation is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MadZeroFallback {
    /// Use the Pearson transform for that row only (default).
    #[default]
    Pearson,
    /// Treat the row as degenerate (NaN).
    Nan,
}

/// Parameters of the biweight midcorrelation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BicorParams {
    /// Maximum fraction of observations per side of the median that may get
    /// zero weight; `1.0` disables the adjustment.
    pub max_p_outliers: f64,
    /// Behaviour when MAD = 0.
    pub mad_zero_fallback: MadZeroFallback,
}

impl Default for BicorParams {
    fn default() -> Self {
        Self { max_p_outliers: 1.0, mad_zero_fallback: MadZeroFallback::Pearson }
    }
}

impl BicorParams {
    /// Validate ranges.
    pub fn validate(&self) -> Result<()> {
        if !(self.max_p_outliers > 0.0 && self.max_p_outliers <= 1.0) {
            return Err(CorrError::InvalidParam {
                name: "max_p_outliers",
                reason: "must be in (0, 1]".into(),
            });
        }
        Ok(())
    }
}

/// Parameters of the Hellinger correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HellcorParams {
    /// Shape of the symmetric Beta(alpha, alpha) marginal transform.
    pub alpha: f64,
    /// Maximum Legendre order for the first variable.
    pub kmax: usize,
    /// Maximum Legendre order for the second variable.
    pub lmax: usize,
    /// Fixed orders (K, L) that skip cross-validated selection.
    pub fixed_orders: Option<(usize, usize)>,
    /// Cross-validation criterion used to select (K, L).
    pub criterion: HellcorCriterion,
}

/// Cross-validation criterion for the Hellinger correlation orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HellcorCriterion {
    /// Minimise the leave-one-out MISE estimate A²(K,L) − 2B(K,L)
    /// (Geenens & Lafaye de Micheaux 2022, Appendix C). Default.
    #[default]
    Mise,
    /// Maximise Σᵢ Rᵢ·⟨β⁽⁻ⁱ⁾, φᵢ⟩ / ‖β⁽⁻ⁱ⁾‖ (normalised leave-one-out
    /// variant described in the rewrite specification).
    NormalizedLoo,
}

impl Default for HellcorParams {
    fn default() -> Self {
        Self { alpha: 6.0, kmax: 20, lmax: 20, fixed_orders: None, criterion: HellcorCriterion::Mise }
    }
}

impl HellcorParams {
    /// Validate ranges.
    pub fn validate(&self) -> Result<()> {
        let bad = |name, reason: &str| Err(CorrError::InvalidParam { name, reason: reason.into() });
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return bad("alpha", "must be finite and > 0");
        }
        if self.kmax == 0 || self.lmax == 0 || self.kmax > 64 || self.lmax > 64 {
            return bad("kmax/lmax", "must be in 1..=64");
        }
        if let Some((k, l)) = self.fixed_orders {
            if k == 0 || l == 0 || k > 64 || l > 64 {
                return bad("fixed_orders", "orders must be in 1..=64");
            }
        }
        Ok(())
    }
}
