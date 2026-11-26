//! NaN-policy routing: classify rows once so engines can take the fast
//! path for complete rows and the per-pair slow path only where needed.

use crate::error::{CorrError, Result};
use crate::options::NanPolicy;
use crate::prep::validity::{first_nonfinite, nonfinite_counts};
use ndarray::ArrayView2;
use std::ops::Range;

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

/// How one output cell is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// Symmetric diagonal: 1.0 when the row is valid, NaN otherwise; p NaN.
    Diagonal {
        /// The row is not degenerate.
        valid: bool,
    },
    /// At least one row is dead: NaN.
    Dead,
    /// At least one row has missing values: per-pair gathered evaluation.
    Slow,
    /// Both rows complete: fast path.
    Fast,
}

impl Cell {
    /// Classify cell (i, j); `sym` marks a single-matrix layout.
    #[inline]
    pub fn classify(sym: bool, i: usize, j: usize, a: RowState, b: RowState) -> Self {
        use RowState::*;
        match (a, b) {
            _ if sym && i == j => Cell::Diagonal { valid: a != Dead },
            (Dead, _) | (_, Dead) => Cell::Dead,
            (Missing, _) | (_, Missing) => Cell::Slow,
            (Clean, Clean) => Cell::Fast,
        }
    }

    /// Value and p-value of a cell that needs no computation.
    #[inline]
    pub fn fixed(self) -> Option<(f64, f64)> {
        match self {
            Cell::Diagonal { valid } => Some((if valid { 1.0 } else { f64::NAN }, f64::NAN)),
            Cell::Dead => Some((f64::NAN, f64::NAN)),
            _ => None,
        }
    }
}

/// The cells of one engine call: row states of both sides and layout.
pub struct Grid<'a> {
    /// Single-matrix (symmetric) layout.
    pub sym: bool,
    /// Row states of the left matrix.
    pub a: &'a [RowState],
    /// Row states of the right matrix (same as `a` when symmetric).
    pub b: &'a [RowState],
    /// Only cells with j ≥ i are needed (symmetric layouts).
    pub upper: bool,
}

impl Grid<'_> {
    /// Fill rows `rows` of a row-major block cell by cell. Diagonal and dead
    /// cells are written here; `eval(i, j, cell, current)` computes the
    /// others and returns (r, p), where `current` is the value already in
    /// `out` (the matrix-product result for Gram engines).
    pub fn fill(
        &self,
        rows: Range<usize>,
        out: &mut [f64],
        mut pout: Option<&mut [f64]>,
        mut eval: impl FnMut(usize, usize, Cell, f64) -> (f64, f64),
    ) {
        let nb = self.b.len();
        for (k, i) in rows.enumerate() {
            let j0 = if self.upper && self.sym { i } else { 0 };
            for j in j0..nb {
                let idx = k * nb + j;
                let cell = Cell::classify(self.sym, i, j, self.a[i], self.b[j]);
                let (r, p) = cell.fixed().unwrap_or_else(|| eval(i, j, cell, out[idx]));
                out[idx] = r;
                if let Some(pv) = pout.as_deref_mut() {
                    pv[idx] = p;
                }
            }
        }
    }
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
