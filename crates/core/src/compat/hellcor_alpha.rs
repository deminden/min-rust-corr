//! v0.2 Hellcor functions taking `alpha` (deprecated).
#![allow(deprecated)]

use super::{cross, full, upper};
use crate::{Correlator, HellcorParams, Method, PvalueSpec};
use ndarray::{Array2, ArrayBase, Data, Ix2};

fn m(alpha: f64) -> Method {
    Method::Hellcor(HellcorParams { alpha, ..Default::default() })
}

fn mc(samples: usize, seed: u64) -> PvalueSpec {
    PvalueSpec::MonteCarlo { samples, seed }
}

/// Hellinger correlation of two vectors (NaN on error).
#[deprecated(since = "0.3.0", note = "use Correlator::pair")]
pub fn hellcor_pair(x: &[f64], y: &[f64], alpha: f64) -> f64 {
    Correlator::new(m(alpha)).pair(x, y).map(|r| r.corr).unwrap_or(f64::NAN)
}

/// Additional v0.2 entry points of the `hellcor` module.
pub mod with_alpha {
    use super::*;
    /// Full matrix with a custom alpha.
    #[deprecated(since = "0.3.0", note = "use Correlator::full")]
    pub fn correlation_matrix_with_alpha<S: Data<Elem = f64>>(
        d: &ArrayBase<S, Ix2>,
        alpha: f64,
    ) -> Array2<f64> {
        full(m(alpha), PvalueSpec::None, d).0
    }
    /// Cross matrix with a custom alpha.
    #[deprecated(since = "0.3.0", note = "use Correlator::cross")]
    pub fn correlation_cross_matrix_with_alpha<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        a: &ArrayBase<S1, Ix2>,
        b: &ArrayBase<S2, Ix2>,
        alpha: f64,
    ) -> Array2<f64> {
        cross(m(alpha), PvalueSpec::None, a, b).0
    }
    /// Packed upper triangle with a custom alpha.
    #[deprecated(since = "0.3.0", note = "use Correlator::upper")]
    pub fn correlation_upper_triangle_with_alpha<S: Data<Elem = f64>>(
        d: &ArrayBase<S, Ix2>,
        alpha: f64,
    ) -> Vec<f64> {
        upper(m(alpha), d)
    }
    /// Full matrix with Monte-Carlo p-values.
    #[deprecated(since = "0.3.0", note = "use Correlator::full with PvalueSpec::MonteCarlo")]
    pub fn correlation_matrix_with_pvalues_with_alpha<S: Data<Elem = f64>>(
        d: &ArrayBase<S, Ix2>,
        alpha: f64,
        mc_samples: usize,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>) {
        full(m(alpha), mc(mc_samples, seed), d)
    }
    /// Cross matrix with Monte-Carlo p-values.
    #[deprecated(since = "0.3.0", note = "use Correlator::cross with PvalueSpec::MonteCarlo")]
    pub fn correlation_cross_matrix_with_pvalues_with_alpha<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        a: &ArrayBase<S1, Ix2>,
        b: &ArrayBase<S2, Ix2>,
        alpha: f64,
        mc_samples: usize,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>) {
        cross(m(alpha), mc(mc_samples, seed), a, b)
    }
}
