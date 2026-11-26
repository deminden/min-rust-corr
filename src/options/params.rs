//! Method parameters: bicor and Hellinger-correlation settings, plus the
//! flat [`MethodSettings`] used by the CLI and the Python bindings.

use crate::error::{CorrError, Result};

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
    /// Minimise the leave-one-out MISE estimate A²(K,L) − 2B(K,L) as
    /// written in Geenens & Lafaye de Micheaux (2022), Appendix C.
    Mise,
    /// Maximise Σᵢ Rᵢ·⟨β⁽⁻ⁱ⁾, φᵢ⟩ / ‖β⁽⁻ⁱ⁾‖ (normalised leave-one-out
    /// criterion). Default: reproduces `HellCor::HellCor` on every fixture.
    #[default]
    NormalizedLoo,
}

impl Default for HellcorParams {
    fn default() -> Self {
        Self {
            alpha: 6.0,
            kmax: 20,
            lmax: 20,
            fixed_orders: None,
            criterion: HellcorCriterion::NormalizedLoo,
        }
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

/// Flat, name-independent settings from which any [`super::Method`] can be
/// built (used by the CLI and the Python bindings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodSettings {
    /// Hellcor: Beta(alpha, alpha) marginal transform.
    pub alpha: f64,
    /// Hellcor: maximum Legendre order K.
    pub kmax: usize,
    /// Hellcor: maximum Legendre order L.
    pub lmax: usize,
    /// Bicor: maximum fraction of outliers per side of the median.
    pub max_p_outliers: f64,
    /// Proportionality: pseudocount added before taking logs.
    pub pseudocount: f64,
    /// Proportionality: apply the centred log-ratio transform.
    pub clr: bool,
    /// Chatterjee: symmetrised max(ξ(x,y), ξ(y,x)).
    pub xi_symmetric: bool,
}

impl Default for MethodSettings {
    fn default() -> Self {
        let (h, b) = (HellcorParams::default(), BicorParams::default());
        let pr = ProportionalityParams::default();
        Self {
            alpha: h.alpha,
            kmax: h.kmax,
            lmax: h.lmax,
            max_p_outliers: b.max_p_outliers,
            pseudocount: pr.pseudocount,
            clr: pr.clr,
            xi_symmetric: ChatterjeeParams::default().symmetric,
        }
    }
}

/// Parameters of the proportionality coefficient ρp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProportionalityParams {
    /// Added to every value before the logarithm (0 = none; zeros then
    /// become missing values).
    pub pseudocount: f64,
    /// Centred log-ratio: subtract from each log value the mean log of its
    /// column (observation) over all rows. Disable when the input is
    /// already log-ratio transformed (values are then used as is).
    pub clr: bool,
}

impl Default for ProportionalityParams {
    fn default() -> Self {
        Self { pseudocount: 0.0, clr: true }
    }
}

impl ProportionalityParams {
    /// Validate ranges.
    pub fn validate(&self) -> Result<()> {
        if !(self.pseudocount.is_finite() && self.pseudocount >= 0.0) {
            return Err(CorrError::InvalidParam {
                name: "pseudocount",
                reason: "must be finite and >= 0".into(),
            });
        }
        Ok(())
    }
}

/// Parameters of Chatterjee's ξ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChatterjeeParams {
    /// Report max(ξ(x, y), ξ(y, x)) so the matrix is symmetric. Off by
    /// default: cell (i, j) is ξ(row i → row j), i.e. how well row j is a
    /// function of row i.
    pub symmetric: bool,
}
