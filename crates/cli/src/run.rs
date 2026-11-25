//! Orchestration: load → select rows → compute in blocks → write.

use crate::args::{Cli, NanArg, PvalueMode};
use crate::input::read_matrix;
use crate::naming::{base_name, clean_stem, legacy_stem};
use crate::output::Sink;
use crate::sinks::{self, Shape};
use crate::subset::{Selector, apply};
use crate::timing::Reporter;
use anyhow::{Result, bail};
use mincorr::{BicorParams, Correlator, HellcorParams, Method, NanPolicy, PvalueSpec, RowBlock};
use ndarray::{Array2, ArrayView2, Axis, CowArray, Ix2};
use std::collections::HashMap;
use std::sync::Arc;

/// Build the correlator from the command line.
pub fn correlator(cli: &Cli) -> Result<Correlator> {
    let p = &cli.params;
    let method = match Method::from_name(&cli.method)? {
        Method::Bicor(_) => {
            Method::Bicor(BicorParams { max_p_outliers: p.max_p_outliers, ..Default::default() })
        }
        Method::Hellcor(_) => Method::Hellcor(HellcorParams {
            alpha: p.alpha,
            kmax: p.kmax,
            lmax: p.lmax,
            ..Default::default()
        }),
        m => m,
    };
    let pv = &cli.pvalues;
    let hell = matches!(method, Method::Hellcor(_));
    let spec = match (pv.with_pvalues, pv.pvalue_mode) {
        (false, _) => PvalueSpec::None,
        (true, PvalueMode::Analytic) => PvalueSpec::Analytic,
        (true, PvalueMode::Mc) => PvalueSpec::MonteCarlo { samples: pv.mc_samples, seed: pv.mc_seed },
        (true, PvalueMode::Auto) if hell => {
            PvalueSpec::MonteCarlo { samples: pv.mc_samples, seed: pv.mc_seed }
        }
        (true, PvalueMode::Auto) => PvalueSpec::Analytic,
    };
    let nan = match cli.input_opts.nan_policy {
        NanArg::Error => NanPolicy::Error,
        NanArg::Propagate => NanPolicy::Propagate,
        NanArg::Pairwise => NanPolicy::PairwiseComplete,
    };
    let mut c = Correlator::new(method).pvalues(spec).nan_policy(nan);
    if let Some(b) = cli.output.block_rows {
        c = c.block_rows(b);
    }
    Ok(c)
}

/// Run the whole command.
pub fn run(cli: Cli) -> Result<()> {
    let mut rep = Reporter::new(cli.quiet, cli.time);
    let corr = correlator(&cli)?;
    let m = read_matrix(&cli.input, &cli.input_opts)?;
    rep.say(format!("Loaded {} rows x {} samples.", m.row_ids.len(), m.col_ids.len()));
    rep.phase("Data loading:");
    let (sel_a, sel_b): (Selector, Selector) = (cli.subset_a.clone().into(), cli.subset_b.clone().into());
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
    let shape = Shape { row_ids: rows_a.clone(), col_ids: cols.clone(), symmetric: !cross, with_p };
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
    stream(&corr, &cli, a.view(), b.as_ref().map(|b| b.view()), &overlap, &mut *sink, with_p)?;
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
    with_p: bool,
) -> Result<()> {
    if let Some(b) = b {
        let mut scratch = (Vec::new(), Vec::new());
        corr.cross_blocks(a, b, |blk| emit(sink, &blk, overlap, &mut scratch))?;
    } else if fits_in_memory(corr, a.nrows(), with_p, cli.output.memory_limit_gb) {
        let out = corr.full(a)?;
        let n = a.nrows();
        let br = cli.output.block_rows.unwrap_or(256).max(1);
        for s in (0..n).step_by(br) {
            let e = (s + br).min(n);
            let pv = out.pvalues.as_ref().and_then(|p| p.as_slice()).map(|p| &p[s * n..e * n]);
            let c = out.corr.as_slice().map(|c| &c[s * n..e * n]).unwrap_or(&[]);
            sink.block(&RowBlock { rows: s..e, n_cols: n, corr: c, pvalues: pv })?;
        }
    } else {
        corr.full_blocks(a, |blk| sink.block(&blk).map_err(sink_err))?;
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

/// Pairwise methods keep the symmetric triangle trick when the matrix fits.
fn fits_in_memory(c: &Correlator, n: usize, with_p: bool, limit_gb: f64) -> bool {
    let bytes = (n * n * 8 * if with_p { 2 } else { 1 }) as f64;
    c.method().is_pairwise() && bytes <= limit_gb * (1u64 << 30) as f64
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
