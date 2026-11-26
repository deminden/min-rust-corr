//! Biweight midcorrelation (Wilcox 2012, ch. 9; Langfelder & Horvath 2012,
//! J. Stat. Softw. 46(11)). Written from the published definition.
//!
//! For a row x with median `med` and raw MAD `mad` (no 1.4826 factor):
//! u_k = (x_k − med) / (9·mad), w_k = (1 − u_k²)² for |u_k| < 1 else 0,
//! a_k = (x_k − med)·w_k, z = a / ‖a‖. bicor(x, y) = Σ z^x_k z^y_k.
//!
//! `max_p_outliers = p < 1`: on each side of the median separately, the
//! weight-function scale is widened from 9·mad to twice the distance between
//! the median and the type-7 quantile at p (below) or 1 − p (above), when
//! that is larger. The behaviour follows the WGCNA documentation of
//! `maxPOutliers`; the exact scaling (factor 2) was determined by black-box
//! comparison with `WGCNA::bicor` output and reproduces it to ≤ 3.3e-16 on
//! every fixture (see docs/parity.md).

use crate::engine::RowTransform;
use crate::options::{BicorParams, MadZeroFallback};
use crate::prep::RowStatus;
use crate::prep::select::{median, quantile};
use crate::prep::standardize::{center_unit_norm, scale_unit_norm};

/// Bicor row transform.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bicor {
    params: BicorParams,
}

impl Bicor {
    /// Create with validated parameters.
    pub fn new(params: BicorParams) -> Self {
        Self { params }
    }
}

impl RowTransform for Bicor {
    type Scratch = Vec<f64>;

    fn transform(&self, x: &[f64], out: &mut [f64], tmp: &mut Vec<f64>) -> RowStatus {
        if x.len() < 2 {
            out.fill(0.0);
            return RowStatus::Degenerate;
        }
        tmp.clear();
        tmp.extend_from_slice(x);
        let med = median(tmp);
        tmp.iter_mut().zip(x).for_each(|(t, &v)| *t = (v - med).abs());
        let mad = median(tmp);
        if !(mad > 0.0) {
            return match self.params.mad_zero_fallback {
                MadZeroFallback::Pearson => center_unit_norm(x, out),
                MadZeroFallback::Nan => {
                    out.fill(0.0);
                    RowStatus::Degenerate
                }
            };
        }
        let (lo, hi) = self.side_scales(x, med, 9.0 * mad, tmp);
        for (o, &v) in out.iter_mut().zip(x) {
            let d = v - med;
            let u = d / if d < 0.0 { lo } else { hi };
            let w = if u.abs() < 1.0 { (1.0 - u * u) * (1.0 - u * u) } else { 0.0 };
            *o = d * w;
        }
        scale_unit_norm(out, true)
    }
}

impl Bicor {
    /// Scales below and above the median (both 9·mad unless widened).
    fn side_scales(&self, x: &[f64], med: f64, base: f64, tmp: &mut Vec<f64>) -> (f64, f64) {
        let p = self.params.max_p_outliers;
        if p >= 1.0 {
            return (base, base);
        }
        tmp.clear();
        tmp.extend_from_slice(x);
        let q_lo = quantile(tmp, p);
        let q_hi = quantile(tmp, 1.0 - p);
        (base.max(2.0 * (med - q_lo)), base.max(2.0 * (q_hi - med)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlier_is_downweighted() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0];
        let mut out = [0.0; 7];
        let mut s = Vec::new();
        assert_eq!(Bicor::default().transform(&x, &mut out, &mut s), RowStatus::Ok);
        assert_eq!(out[6], 0.0);
    }

    #[test]
    fn mad_zero_fallback() {
        let x = [2.0, 2.0, 2.0, 2.0, 5.0];
        let mut out = [0.0; 5];
        let mut s = Vec::new();
        assert_eq!(Bicor::default().transform(&x, &mut out, &mut s), RowStatus::Ok);
        let nan = Bicor::new(BicorParams { mad_zero_fallback: MadZeroFallback::Nan, ..Default::default() });
        assert_eq!(nan.transform(&x, &mut out, &mut s), RowStatus::Degenerate);
    }
}
