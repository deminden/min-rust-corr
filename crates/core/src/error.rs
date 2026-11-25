//! Error type shared by the whole crate.

use thiserror::Error;

/// Errors returned by the correlation API. User data never causes a panic;
/// every invalid input is reported through one of these variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CorrError {
    /// Two inputs that must agree in a dimension do not.
    #[error("shape mismatch: {what} (left {left}, right {right})")]
    ShapeMismatch {
        /// Which dimension disagrees.
        what: &'static str,
        /// Size on the left-hand input.
        left: usize,
        /// Size on the right-hand input.
        right: usize,
    },
    /// An input has zero rows or zero columns.
    #[error("empty input: {what}")]
    Empty {
        /// Which input is empty.
        what: &'static str,
    },
    /// A non-finite value was found under [`crate::NanPolicy::Error`].
    #[error("non-finite value at row {row}, column {col}")]
    NonFinite {
        /// Row index (variable).
        row: usize,
        /// Column index (observation).
        col: usize,
    },
    /// A parameter is outside its valid range.
    #[error("invalid parameter {name}: {reason}")]
    InvalidParam {
        /// Parameter name.
        name: &'static str,
        /// Explanation.
        reason: String,
    },
    /// The requested combination of options is not supported.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A streaming sink reported an error.
    #[error("sink error: {0}")]
    Sink(String),
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, CorrError>;
