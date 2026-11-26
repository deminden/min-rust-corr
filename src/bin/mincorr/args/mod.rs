//! Command-line arguments (clap derive). The v0.2 command line
//! `mincorr <input> <method> [threads] [--flags]` keeps working.

mod io;
mod subset;

pub use io::{Delimiter, Format, InputArgs, OutputArgs};
pub use subset::{SubsetA, SubsetB};

use clap::{Args, Parser, ValueEnum};
use mincorr::MethodSettings;
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

impl MethodArgs {
    /// Settings for [`mincorr::Method::with_settings`].
    pub fn settings(&self) -> MethodSettings {
        MethodSettings {
            alpha: self.alpha,
            kmax: self.kmax,
            lmax: self.lmax,
            max_p_outliers: self.max_p_outliers,
        }
    }
}
