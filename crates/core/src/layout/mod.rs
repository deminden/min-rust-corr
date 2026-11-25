//! Output containers and layout helpers (full, cross, packed upper,
//! streamed row blocks).

pub mod blocks;
pub mod fill;

use ndarray::Array2;
use std::ops::Range;

pub use fill::{packed_index, packed_len};

/// Square or rectangular correlation matrix with optional p-values.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrOutput {
    /// Correlation values, `rows_a × rows_b`.
    pub corr: Array2<f64>,
    /// P-values of the same shape, when requested.
    pub pvalues: Option<Array2<f64>>,
}

/// Packed upper triangle (row-major, diagonal included), length n(n+1)/2.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedOutput {
    /// Matrix dimension n.
    pub n: usize,
    /// Packed correlations; entry (i, j), i ≤ j, is at [`packed_index`].
    pub corr: Vec<f64>,
    /// Packed p-values, when requested.
    pub pvalues: Option<Vec<f64>>,
}

/// Result for a single pair of vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairResult {
    /// Correlation estimate (NaN when undefined).
    pub corr: f64,
    /// P-value, when requested.
    pub pvalue: Option<f64>,
    /// Number of observations used.
    pub n_eff: usize,
}

/// A contiguous block of output rows handed to a streaming sink.
#[derive(Debug, Clone)]
pub struct RowBlock<'a> {
    /// Global row indices covered by this block.
    pub rows: Range<usize>,
    /// Number of columns of each row.
    pub n_cols: usize,
    /// Row-major correlations, `rows.len() × n_cols`.
    pub corr: &'a [f64],
    /// Row-major p-values, when requested.
    pub pvalues: Option<&'a [f64]>,
}

impl RowBlock<'_> {
    /// Correlations of global row `i` (must lie in `rows`).
    pub fn corr_row(&self, i: usize) -> &[f64] {
        let k = i - self.rows.start;
        &self.corr[k * self.n_cols..(k + 1) * self.n_cols]
    }

    /// P-values of global row `i`, when present.
    pub fn pvalue_row(&self, i: usize) -> Option<&[f64]> {
        let k = i - self.rows.start;
        self.pvalues.map(|p| &p[k * self.n_cols..(k + 1) * self.n_cols])
    }
}
