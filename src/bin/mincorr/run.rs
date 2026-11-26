//! Orchestration: load → select rows → compute in blocks → write.

use crate::args::{Cli, PvalueMode};
use crate::input::read_matrix;
use crate::naming::{base_name, clean_stem, legacy_stem};
use crate::output::{Cells, Sink};
use crate::sinks::{self, Shape};
use crate::subset::apply;
use crate::timing::Reporter;
use anyhow::{Result, bail};
use mincorr::{Correlator, Method, PvalueSpec, RowBlock};
use ndarray::{Array2, ArrayView2, Axis, CowArray, Ix2};
use std::collections::HashMap;
use std::sync::Arc;

/// Build the correlator from the command line.
pub fn correlator(cli: &Cli) -> Result<Correlator> {
    let method = Method::with_settings(&cli.method, &cli.params.settings())?;
    let pv = &cli.pvalues;
    let mc = PvalueSpec::MonteCarlo { samples: pv.mc_samples, seed: pv.mc_seed };
    let spec = match (pv.with_pvalues, pv.pvalue_mode) {
        (false, _) => PvalueSpec::None,
        (true, PvalueMode::Analytic) => PvalueSpec::Analytic,
        (true, PvalueMode::Mc) => mc,
        (true, PvalueMode::Auto) => PvalueSpec::auto(&method, pv.mc_samples, pv.mc_seed),
    };
    let mut c = Correlator::new(method).pvalues(spec).nan_policy(cli.input_opts.nan_policy);
    if let Some(b) = cli.output.block_rows {
        c = c.block_rows(b);
    }
    Ok(c)
}

/// Run the whole command.
pub fn run(cli: Cli) -> Result<()> {
    let mut rep = Reporter::new(cli.quiet, cli.time);
    let corr = correlator(&cli)?;
    let mut m = read_matrix(&cli.input, &cli.input_opts)?;
    rep.say(format!("Loaded {} rows x {} samples.", m.row_ids.len(), m.col_ids.len()));
    let corr = crate::prepare::prepare(corr, &mut m, &cli)?;
    rep.phase("Data loading:");
    let (sel_a, sel_b) = (cli.subset_a.selector(), cli.subset_b.selector());
    if cli.subset_vs_all && sel_b.has_any() {
        bail!("--subset-vs-all cannot be combined with --subset-b-* selectors");
    }
    if cli.subset_vs_all && !sel_a.has_any() {
        bail!("--subset-vs-all requires a subset A selection");
    }
    let cross = cli.subset_vs_all || sel_b.has_any();
    let (ia, tag_a) = apply(&m.row_ids, &sel_a)?;
    let (ib, tag_b) = if cli.subset_vs_all {
        ((0..m.row_ids.len()).collect(), Some("all".to_string()))
    } else {
        apply(&m.row_ids, &sel_b)?
    };
    let pick = |idx: &[usize]| -> Vec<String> { idx.iter().map(|&i| m.row_ids[i].clone()).collect() };
    let rows_a = Arc::new(pick(&ia));
    let cols = Arc::new(if cross { pick(&ib) } else { (*rows_a).clone() });
    let with_p = corr.pvalue_spec().wanted();
    let method = corr.method().name();
    let base = base_name(&clean_stem(&cli.input), tag_a.as_deref(), tag_b.as_deref(), cross);
    let lbase = base_name(&legacy_stem(&cli.input), tag_a.as_deref(), tag_b.as_deref(), cross);
    let plan = sinks::plan(&cli, &base, &lbase, method, with_p)?;
    let cells = match (cross, corr.method().is_symmetric()) {
        (true, _) => Cells::All,
        (false, true) => Cells::Upper,
        (false, false) => Cells::OffDiagonal,
    };
    let shape = Shape { row_ids: rows_a.clone(), col_ids: cols.clone(), cells, with_p };
    let mut sink = sinks::build(&cli, &plan, &shape)?;
    rep.say(format!(
        "Computing {method} ({} x {}){}...",
        rows_a.len(),
        cols.len(),
        if with_p { " with p-values" } else { "" }
    ));
    let a = select(&m.data, &ia);
    let b = cross.then(|| select(&m.data, &ib));
    let overlap = if cross { overlaps(&rows_a, &cols) } else { HashMap::new() };
    stream(&corr, &cli, a.view(), b.as_ref().map(|b| b.view()), &overlap, &mut *sink)?;
    rep.phase("Correlation + output:");
    sink.finish()?;
    for f in sinks::finalize(&plan)? {
        rep.say(format!("Wrote {}", f.display()));
    }
    rep.summary();
    Ok(())
}

/// Compute and feed blocks to the sink.
fn stream(
    corr: &Correlator,
    cli: &Cli,
    a: ArrayView2<'_, f64>,
    b: Option<ArrayView2<'_, f64>>,
    overlap: &HashMap<usize, usize>,
    sink: &mut dyn Sink,
) -> Result<()> {
    if let Some(b) = b {
        let mut scratch = (Vec::new(), Vec::new());
        corr.cross_blocks(a, b, |blk| emit(sink, &blk, overlap, &mut scratch))?;
    } else {
        let limit = (cli.output.memory_limit_gb * (1u64 << 30) as f64) as usize;
        corr.clone().triangle_cache(limit).full_blocks(a, |blk| sink.block(&blk).map_err(sink_err))?;
    }
    Ok(())
}

/// Rows `idx` of `data`, borrowing when the selection is the identity.
fn select<'a>(data: &'a Array2<f64>, idx: &[usize]) -> CowArray<'a, f64, Ix2> {
    if idx.len() == data.nrows() && idx.iter().enumerate().all(|(k, &i)| k == i) {
        return CowArray::from(data.view());
    }
    CowArray::from(data.select(Axis(0), idx))
}

fn sink_err(e: anyhow::Error) -> mincorr::CorrError {
    mincorr::CorrError::Sink(format!("{e:#}"))
}

/// Column index in B of each row of A that has the same ID.
fn overlaps(a: &[String], b: &[String]) -> HashMap<usize, usize> {
    let pos: HashMap<&str, usize> = b.iter().enumerate().map(|(j, s)| (s.as_str(), j)).collect();
    a.iter().enumerate().filter_map(|(i, s)| pos.get(s.as_str()).map(|&j| (i, j))).collect()
}

/// Forward a cross block, snapping same-ID cells to r = 1, p = NaN.
fn emit(
    sink: &mut dyn Sink,
    b: &RowBlock<'_>,
    overlap: &HashMap<usize, usize>,
    scratch: &mut (Vec<f64>, Vec<f64>),
) -> mincorr::Result<()> {
    if !b.rows.clone().any(|i| overlap.contains_key(&i)) {
        return sink.block(b).map_err(sink_err);
    }
    scratch.0.clear();
    scratch.0.extend_from_slice(b.corr);
    scratch.1.clear();
    scratch.1.extend_from_slice(b.pvalues.unwrap_or(&[]));
    for i in b.rows.clone() {
        if let Some(&j) = overlap.get(&i) {
            let k = (i - b.rows.start) * b.n_cols + j;
            if !scratch.0[k].is_nan() {
                scratch.0[k] = 1.0;
                if let Some(p) = scratch.1.get_mut(k) {
                    *p = f64::NAN;
                }
            }
        }
    }
    let pv = b.pvalues.map(|_| scratch.1.as_slice());
    sink.block(&RowBlock { rows: b.rows.clone(), n_cols: b.n_cols, corr: &scratch.0, pvalues: pv })
        .map_err(sink_err)
}
