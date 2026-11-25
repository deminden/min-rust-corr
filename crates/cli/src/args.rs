//! Command-line arguments (clap derive). The v0.2 command line
//! `mincorr <input> <method> [threads] [--flags]` keeps working.

use clap::{Args, Parser, ValueEnum};
use std::path::PathBuf;

/// Compute correlation matrices (rows = variables, columns = samples).
#[derive(Debug, Parser)]
#[command(name = "mincorr", version, about)]
pub struct Cli {
    /// Input matrix: TSV/CSV, optionally .gz or a single-file .tar.gz.
    pub input: PathBuf,
    /// Method: pearson, spearman, kendall, bicor (biweight), hellcor (hellinger).
    pub method: String,
    /// Number of threads (v0.2 positional form of --threads).
    pub threads_positional: Option<usize>,
    /// Number of threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
    /// Print timing information to stderr.
    #[arg(long)]
    pub time: bool,
    /// Suppress progress messages.
    #[arg(long, short)]
    pub quiet: bool,
    #[command(flatten)]
    pub pvalues: PvalueArgs,
    #[command(flatten)]
    pub subset_a: SubsetA,
    #[command(flatten)]
    pub subset_b: SubsetB,
    /// Correlate subset A against all rows.
    #[arg(long)]
    pub subset_vs_all: bool,
    #[command(flatten)]
    pub input_opts: InputArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[command(flatten)]
    pub params: MethodArgs,
}

/// P-value options.
#[derive(Debug, Args)]
pub struct PvalueArgs {
    /// Also compute p-values (written to a sibling file).
    #[arg(long)]
    pub with_pvalues: bool,
    /// P-value mode; auto = analytic for all methods except hellcor (Monte Carlo).
    #[arg(long, value_enum, default_value = "auto", ignore_case = true)]
    pub pvalue_mode: PvalueMode,
    /// Monte-Carlo replicates (hellcor).
    #[arg(long, default_value_t = 10_000)]
    pub mc_samples: usize,
    /// Monte-Carlo seed (hellcor).
    #[arg(long, default_value_t = 42)]
    pub mc_seed: u64,
}

/// P-value mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PvalueMode {
    /// Analytic for all methods except hellcor.
    Auto,
    /// Analytic (t / normal approximation).
    Analytic,
    /// Monte-Carlo permutation null.
    #[value(alias = "montecarlo")]
    Mc,
}

macro_rules! subset_group {
    ($name:ident, $doc:literal, $size:literal, $seed:literal, $file:literal, $rows:literal $(, $alias:literal)*) => {
        #[doc = $doc]
        #[derive(Debug, Args, Default, Clone)]
        pub struct $name {
            /// Random subset of N rows.
            #[arg(id = $size, long = $size $(, alias = concat!($alias, "size"))*)]
            pub size: Option<usize>,
            /// Seed of the random subset (default 42).
            #[arg(id = $seed, long = $seed $(, alias = concat!($alias, "seed"))*)]
            pub seed: Option<u64>,
            /// File with one row ID per line.
            #[arg(id = $file, long = $file $(, alias = concat!($alias, "file"))*)]
            pub file: Option<PathBuf>,
            /// Comma-separated row IDs.
            #[arg(id = $rows, long = $rows, value_delimiter = ',' $(, alias = concat!($alias, "rows"))*)]
            pub rows: Option<Vec<String>>,
        }
    };
}

subset_group!(
    SubsetA,
    "Row selection A (also the v0.2 --subset-* flags).",
    "subset-a-size",
    "subset-a-seed",
    "subset-a-file",
    "subset-a-rows",
    "subset-"
);
subset_group!(
    SubsetB,
    "Row selection B (cross mode A × B).",
    "subset-b-size",
    "subset-b-seed",
    "subset-b-file",
    "subset-b-rows"
);

/// Input parsing options.
#[derive(Debug, Args)]
pub struct InputArgs {
    /// Field delimiter.
    #[arg(long, value_enum, default_value = "auto")]
    pub delimiter: Delimiter,
    /// Strings read as missing values (empty cells always are).
    #[arg(long, value_delimiter = ',', default_value = "NA,NaN,nan,null")]
    pub na_strings: Vec<String>,
    /// Handling of missing values.
    #[arg(long, value_enum, default_value = "pairwise")]
    pub nan_policy: NanArg,
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

/// NaN policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum NanArg {
    /// Error on any missing value.
    Error,
    /// Rows with missing values give NaN.
    Propagate,
    /// Use pairwise-complete observations.
    Pairwise,
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

/// Method parameters.
#[derive(Debug, Args)]
pub struct MethodArgs {
    /// Hellcor: Beta(alpha, alpha) marginal transform.
    #[arg(long, default_value_t = 6.0)]
    pub alpha: f64,
    /// Hellcor: maximum Legendre order K.
    #[arg(long, default_value_t = 20)]
    pub kmax: usize,
    /// Hellcor: maximum Legendre order L.
    #[arg(long, default_value_t = 20)]
    pub lmax: usize,
    /// Bicor: maximum fraction of outliers per side of the median.
    #[arg(long, default_value_t = 1.0)]
    pub max_p_outliers: f64,
}
