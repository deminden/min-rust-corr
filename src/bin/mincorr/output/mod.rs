//! Streaming output sinks. Every sink receives row blocks in order, so
//! peak memory is one block regardless of the matrix size.

mod edges;
mod matrix;
mod sinkfile;

pub use edges::{Cells, EdgeFilter, EdgesSink};
pub use matrix::{MatrixSink, NpyWriter, TsvWriter};
pub use sinkfile::Out;

use anyhow::Result;
use mincorr::RowBlock;

/// Receives correlation (and p-value) blocks in row order.
pub trait Sink {
    /// Consume one block.
    fn block(&mut self, b: &RowBlock<'_>) -> Result<()>;
    /// Flush and close all files.
    fn finish(self: Box<Self>) -> Result<()>;
}

/// Correlations and optional p-values written to two matrix files.
pub struct MatrixPair {
    /// Correlation writer.
    pub corr: Box<dyn MatrixSink>,
    /// P-value writer.
    pub pvalues: Option<Box<dyn MatrixSink>>,
}

impl Sink for MatrixPair {
    fn block(&mut self, b: &RowBlock<'_>) -> Result<()> {
        self.corr.rows(b.rows.start, b.corr, b.n_cols)?;
        if let (Some(w), Some(p)) = (self.pvalues.as_mut(), b.pvalues) {
            w.rows(b.rows.start, p, b.n_cols)?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        self.corr.finish()?;
        if let Some(p) = self.pvalues {
            p.finish()?;
        }
        Ok(())
    }
}

/// Format one value: shortest round-trip representation, or a fixed
/// number of decimals; non-finite values as NaN / Inf / -Inf.
pub fn format_value(v: f64, digits: Option<usize>, buf: &mut ryu::Buffer, out: &mut String) {
    use std::fmt::Write;
    if v.is_nan() {
        out.push_str("NaN");
    } else if v.is_infinite() {
        out.push_str(if v > 0.0 { "Inf" } else { "-Inf" });
    } else if let Some(d) = digits {
        let _ = write!(out, "{v:.d$}");
    } else {
        out.push_str(buf.format_finite(v));
    }
}
