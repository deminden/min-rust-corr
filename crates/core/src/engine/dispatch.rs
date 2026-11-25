//! NaN-policy routing: classify rows once so engines can take the fast
//! path for complete rows and the per-pair slow path only where needed.

use crate::error::{CorrError, Result};
use crate::options::NanPolicy;
use crate::prep::validity::{first_nonfinite, nonfinite_counts};
use ndarray::ArrayView2;

/// State of one input row after the census.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowState {
    /// All values finite and the row is non-degenerate: fast path.
    Clean,
    /// Contains non-finite values: each pair gathers common observations.
    Missing,
    /// Every pair involving this row is NaN.
    Dead,
}

/// Classify every row of `data` under `policy`.
pub fn row_states(data: ArrayView2<'_, f64>, policy: NanPolicy) -> Result<Vec<RowState>> {
    if policy == NanPolicy::Error {
        if let Some((row, col)) = first_nonfinite(data) {
            return Err(CorrError::NonFinite { row, col });
        }
    }
    Ok(nonfinite_counts(data)
        .into_iter()
        .map(|c| match (c, policy) {
            (0, _) => RowState::Clean,
            (_, NanPolicy::Propagate) => RowState::Dead,
            _ => RowState::Missing,
        })
        .collect())
}

/// Validate the shape of one input.
pub fn check_shape(data: ArrayView2<'_, f64>, what: &'static str) -> Result<()> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err(CorrError::Empty { what });
    }
    Ok(())
}

/// Validate that two inputs have the same number of observations.
pub fn check_cross(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<()> {
    check_shape(a, "first matrix")?;
    check_shape(b, "second matrix")?;
    if a.ncols() != b.ncols() {
        return Err(CorrError::ShapeMismatch {
            what: "number of columns",
            left: a.ncols(),
            right: b.ncols(),
        });
    }
    Ok(())
}
