//! Per-row preparation helpers shared by all methods: validity census,
//! ranking, selection (median/quantile) and standardisation.

pub mod rank;
pub mod select;
pub mod standardize;
pub mod validity;

/// Outcome of transforming or preparing one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowStatus {
    /// The row produced a usable transform.
    Ok,
    /// The row is degenerate (too few finite values or zero spread);
    /// every pair involving it is NaN.
    Degenerate,
}
