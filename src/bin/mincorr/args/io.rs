//! Input-parsing and output options.

use clap::{Args, ValueEnum};
use mincorr::NanPolicy;
use std::path::PathBuf;

/// Input parsing options.
#[derive(Debug, Args)]
pub struct InputArgs {
    /// Field delimiter.
    #[arg(long, value_enum, default_value = "auto")]
    pub delimiter: Delimiter,
    /// Strings read as missing values (empty cells always are).
    #[arg(long, value_delimiter = ',', default_value = "NA,NaN,nan,null")]
    pub na_strings: Vec<String>,
    /// Handling of missing values: pairwise, propagate or error.
    #[arg(long, default_value = "pairwise", value_parser = parse_nan)]
    pub nan_policy: NanPolicy,
    /// Sort rows alphabetically by ID (v0.2 behaviour) instead of input order.
    #[arg(long)]
    pub sort_rows: bool,
}

/// Field delimiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Delimiter {
    /// Detect from the header line.
    Auto,
    /// Tab.
    Tab,
    /// Comma.
    Comma,
}

fn parse_nan(s: &str) -> Result<NanPolicy, String> {
    s.parse().map_err(|e: mincorr::CorrError| e.to_string())
}

/// Output options.
#[derive(Debug, Args)]
pub struct OutputArgs {
    /// Output path ("-" = stdout, tsv only). Default: derived from the input name.
    #[arg(long, short)]
    pub output: Option<PathBuf>,
    /// Output format.
    #[arg(long, value_enum, default_value = "tsv.gz")]
    pub format: Format,
    /// Write the v0.2 single .tar.gz archive with v0.2 file names.
    #[arg(long)]
    pub legacy_tar: bool,
    /// Fixed number of decimal places (default: shortest round-trip).
    #[arg(long)]
    pub digits: Option<usize>,
    /// Rows per computed block.
    #[arg(long)]
    pub block_rows: Option<usize>,
    /// In-memory limit (GiB) for pairwise methods before switching to streaming.
    #[arg(long, default_value_t = 2.0)]
    pub memory_limit_gb: f64,
    /// Edges format: keep |r| >= X.
    #[arg(long)]
    pub min_abs_r: Option<f64>,
    /// Edges format: keep p <= X (requires --with-pvalues).
    #[arg(long)]
    pub max_p: Option<f64>,
    /// Edges format: keep the K strongest edges per row.
    #[arg(long)]
    pub top_k: Option<usize>,
}

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Plain TSV.
    Tsv,
    /// Gzipped TSV.
    #[value(name = "tsv.gz")]
    TsvGz,
    /// NumPy .npy (float64, C order).
    Npy,
    /// Long format row/col/r[/p], gzipped.
    Edges,
}
