//! Whole-table preparation before rows are selected: the clr transform of
//! ρp (whose reference uses every feature) and covariate loading.

use crate::args::Cli;
use crate::input::{Matrix, read_matrix};
use anyhow::{Context, Result, bail};
use mincorr::prep::logratio::log_clr;
use mincorr::{Correlator, Method, ProportionalityParams};
use ndarray::Array2;
use std::collections::HashMap;

/// Apply the clr transform to the whole table for ρp (then turn it off in
/// the method) and attach covariates.
pub fn prepare(mut corr: Correlator, m: &mut Matrix, cli: &Cli) -> Result<Correlator> {
    if let Method::Proportionality(p) = corr.method().clone() {
        if p.clr {
            m.data = log_clr(&[m.data.view()], p.pseudocount).into_iter().next().context("clr")?;
            corr = corr.method_replaced(Method::Proportionality(ProportionalityParams { clr: false, ..p }));
        }
    }
    if let Some(path) = &cli.covariates {
        let c = read_matrix(path, &cli.input_opts).with_context(|| format!("reading {}", path.display()))?;
        corr = corr.covariates(align(&c, &m.col_ids)?);
    }
    Ok(corr)
}

/// Covariate values with columns reordered to the data's sample order.
fn align(c: &Matrix, samples: &[String]) -> Result<Array2<f64>> {
    let pos: HashMap<&str, usize> = c.col_ids.iter().enumerate().map(|(j, s)| (s.as_str(), j)).collect();
    let mut out = Array2::zeros((c.row_ids.len(), samples.len()));
    for (k, s) in samples.iter().enumerate() {
        let Some(&j) = pos.get(s.as_str()) else {
            bail!("sample '{s}' is missing from the covariate file");
        };
        out.column_mut(k).assign(&c.data.column(j));
    }
    Ok(out)
}
