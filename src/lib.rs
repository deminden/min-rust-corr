//! # mincorr
//!
//! Fast correlation matrices for wide numeric data (rows = variables,
//! columns = observations): Pearson, Spearman, Kendall tau-b, biweight
//! midcorrelation and Hellinger correlation, in full, cross and packed
//! upper-triangle layouts, with optional p-values and streaming output.
//!
//! The entry point is [`Correlator`]. See `docs/semantics.md` for the
//! handling of missing values, degenerate rows and diagonals.
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod api;
pub mod compat;
pub mod engine;
pub mod error;
pub mod layout;
pub mod methods;
pub mod options;
pub mod prep;
pub mod stats;

pub use api::Correlator;
#[allow(deprecated)]
pub use compat::{bicor, hellcor, hellcor_pair, kendall, pearson, spearman};
pub use error::{CorrError, Result};
pub use layout::{CorrOutput, PackedOutput, PairResult, RowBlock, packed_index, packed_len};
pub use options::{
    BicorParams, HellcorCriterion, HellcorParams, MadZeroFallback, Method, MethodSettings, NanPolicy,
    PvalueSpec,
};

/// Compiles and runs the Rust example in README.md as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
