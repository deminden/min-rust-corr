//! Orthonormal basis of (centred, scored) covariates for partial
//! correlation: residualising a centred row against it equals taking the
//! residuals of a least-squares regression on the covariates with an
//! intercept.

use super::scores::{ScoreScratch, Scores};
use crate::error::{CorrError, Result};
use crate::prep::standardize::mean;
use ndarray::ArrayView2;

/// Columns whose norm falls below this fraction of their original norm
/// after orthogonalisation are linearly dependent and dropped.
const RANK_TOL: f64 = 1e-10;

/// Orthonormal basis Q (k × n, rows orthonormal and orthogonal to 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Basis {
    n: usize,
    q: Vec<Vec<f64>>,
}

impl Basis {
    /// Build from covariates (rows = covariates, columns = observations),
    /// each mapped through `scores` first. Values must be finite.
    pub fn new(cov: ArrayView2<'_, f64>, scores: Scores, n_obs: usize) -> Result<Self> {
        if cov.ncols() != n_obs {
            return Err(CorrError::ShapeMismatch {
                what: "covariate observations",
                left: cov.ncols(),
                right: n_obs,
            });
        }
        if let Some((row, col)) = crate::prep::validity::first_nonfinite(cov) {
            return Err(CorrError::InvalidParam {
                name: "covariates",
                reason: format!("non-finite value at row {row}, column {col}"),
            });
        }
        let mut b = Basis { n: n_obs, q: Vec::new() };
        let (mut s, mut buf) = (ScoreScratch::default(), Vec::new());
        for row in cov.rows() {
            buf.clear();
            buf.extend(row.iter());
            let v = scores.apply(&buf, &mut s);
            let m = mean(v);
            let mut c: Vec<f64> = v.iter().map(|a| a - m).collect();
            let before = norm(&c);
            if !(before > 0.0) {
                continue;
            }
            let after = b.residualize(&mut c) * before;
            if after > RANK_TOL * before {
                c.iter_mut().for_each(|a| *a /= after);
                b.q.push(c);
            }
        }
        Ok(b)
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.n
    }

    /// Rank k of the (centred) covariate matrix.
    pub fn rank(&self) -> usize {
        self.q.len()
    }

    /// Remove the projection of `v` onto the basis (two Gram–Schmidt
    /// passes for stability). Returns ‖v_after‖ / ‖v_before‖.
    pub fn residualize(&self, v: &mut [f64]) -> f64 {
        let before = norm(v);
        for _ in 0..2 {
            for q in &self.q {
                let c: f64 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(q).for_each(|(x, a)| *x -= c * a);
            }
        }
        if before > 0.0 { norm(v) / before } else { 0.0 }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn drops_dependent_columns() {
        let cov = array![[1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 6.0, 10.0], [1.0, 1.0, 1.0, 1.0]];
        let b = Basis::new(cov.view(), Scores::Raw, 4).unwrap();
        assert_eq!(b.rank(), 1);
        let mut v = vec![-1.75, -0.75, 0.25, 2.25];
        assert!(b.residualize(&mut v) < 1e-14);
    }
}
