//! Build the output sinks for a run and finalise them (including the
//! v0.2 `.tar.gz` layout under `--legacy-tar`).

use crate::args::{Cli, Format};
use crate::naming::pvalue_path;
use crate::output::{Cells, EdgeFilter, EdgesSink, MatrixPair, MatrixSink, NpyWriter, Out, Sink, TsvWriter};
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Labels and shape of the output matrix.
pub struct Shape {
    pub row_ids: Arc<Vec<String>>,
    pub col_ids: Arc<Vec<String>>,
    pub cells: Cells,
    pub with_p: bool,
}

/// Planned output files.
pub struct Plan {
    /// Correlation file (or the archive under `--legacy-tar`).
    pub corr: PathBuf,
    /// P-value file, when requested.
    pub pvalues: Option<PathBuf>,
    /// Archive path and member names for `--legacy-tar`.
    pub legacy: Option<(PathBuf, String, String)>,
}

fn ext(f: Format) -> &'static str {
    match f {
        Format::Tsv => "tsv",
        Format::TsvGz => "tsv.gz",
        Format::Npy => "npy",
        Format::Edges => "edges.tsv.gz",
    }
}

/// Decide output paths. `legacy_base`/`base` are the name stems.
pub fn plan(cli: &Cli, base: &str, legacy_base: &str, method: &str, with_p: bool) -> Result<Plan> {
    let o = &cli.output;
    if o.legacy_tar {
        let tsv = format!("{legacy_base}_{method}_correlations.tsv");
        let ptsv = format!("{legacy_base}_{method}_pvalues.tsv");
        let tar = PathBuf::from(format!("{legacy_base}_{method}_correlations.tar.gz"));
        let tmp = |n: &str| tar.with_file_name(format!(".{n}.partial"));
        return Ok(Plan {
            corr: tmp(&tsv),
            pvalues: with_p.then(|| tmp(&ptsv)),
            legacy: Some((tar, tsv, ptsv)),
        });
    }
    let corr = match &o.output {
        Some(p) => p.clone(),
        None => PathBuf::from(format!("{base}_{method}_correlations.{}", ext(o.format))),
    };
    if corr.as_os_str() == "-" && (with_p || !matches!(o.format, Format::Tsv | Format::TsvGz | Format::Edges))
    {
        bail!("-o - (stdout) supports text formats without --with-pvalues only");
    }
    let pvalues = (with_p && o.format != Format::Edges).then(|| pvalue_path(&corr));
    Ok(Plan { corr, pvalues, legacy: None })
}

fn matrix_sink(cli: &Cli, path: &Path, s: &Shape) -> Result<Box<dyn MatrixSink>> {
    let fmt = if cli.output.legacy_tar { Format::Tsv } else { cli.output.format };
    let gz = fmt == Format::TsvGz && path.as_os_str() != "-";
    let out = Out::create(path, false)?;
    Ok(match fmt {
        Format::Npy => Box::new(NpyWriter::new(out, s.row_ids.len(), s.col_ids.len())?),
        _ => Box::new(TsvWriter::new(out, s.row_ids.clone(), &s.col_ids, cli.output.digits, gz)?),
    })
}

/// Create the sink for a plan.
pub fn build(cli: &Cli, plan: &Plan, s: &Shape) -> Result<Box<dyn Sink>> {
    let o = &cli.output;
    if o.format == Format::Edges && !o.legacy_tar {
        let filter = EdgeFilter { min_abs_r: o.min_abs_r, max_p: o.max_p, top_k: o.top_k };
        let out = Out::create(&plan.corr, plan.corr.as_os_str() != "-")?;
        let e =
            EdgesSink::new(out, s.row_ids.clone(), s.col_ids.clone(), s.cells, s.with_p, filter, o.digits)?;
        return Ok(Box::new(e));
    }
    let corr = matrix_sink(cli, &plan.corr, s)?;
    let pvalues = plan.pvalues.as_ref().map(|p| matrix_sink(cli, p, s)).transpose()?;
    Ok(Box::new(MatrixPair { corr, pvalues }))
}

/// Move finished files into place (legacy archive) and list the outputs.
pub fn finalize(plan: &Plan) -> Result<Vec<PathBuf>> {
    let Some((tar_path, tsv, ptsv)) = &plan.legacy else {
        return Ok(std::iter::once(plan.corr.clone()).chain(plan.pvalues.clone()).collect());
    };
    let file = std::fs::File::create(tar_path)?;
    let mut b = tar::Builder::new(flate2::write::GzEncoder::new(file, flate2::Compression::default()));
    b.append_path_with_name(&plan.corr, tsv)?;
    if let Some(p) = &plan.pvalues {
        b.append_path_with_name(p, ptsv)?;
    }
    b.into_inner()?.finish()?;
    std::fs::remove_file(&plan.corr)?;
    if let Some(p) = &plan.pvalues {
        std::fs::remove_file(p)?;
    }
    Ok(vec![tar_path.clone()])
}
