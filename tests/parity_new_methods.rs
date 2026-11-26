//! Fixture parity for Gaussian rank, partial correlation, ξ, distance
//! correlation and ρp (tools/make_fixtures.R, R 4.6.2; energy::dcor and
//! XICOR::xicor as black-box oracles).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use common::*;
use mincorr::{ChatterjeeParams, Correlator, Method, ProportionalityParams, PvalueSpec};
use ndarray::Array2;

/// Datasets that received the v0.4 fixtures (complete, ≥ 6 columns).
fn datasets() -> impl Iterator<Item = &'static str> {
    DATASETS.iter().copied().filter(|ds| exists(&format!("{ds}__xi_r")))
}

/// Blank out rows/columns of constant input rows (NaN here by design;
/// some oracles return 0 or ±inf for them).
fn mask_constant(x: &Array2<f64>, mut want: Array2<f64>) -> Array2<f64> {
    for (i, row) in x.rows().into_iter().enumerate() {
        if row.iter().all(|&v| v == row[0]) {
            want.row_mut(i).fill(f64::NAN);
            want.column_mut(i).fill(f64::NAN);
        }
    }
    want
}

/// P-value tolerance: relative, or absolute for oracles that compute
/// 1 − Φ(z) by subtraction (inaccurate below ~1e-12).
#[derive(Clone, Copy)]
enum Tol {
    Rel(f64),
    Abs(f64),
}

fn compare(c: &Correlator, ds: &str, tag: &str, tol_r: f64, tol_p: Option<Tol>) {
    let x = load(&format!("{ds}__input"));
    let out = c.full(x.view()).unwrap();
    let ctx = format!("{tag}/{ds}");
    let want = mask_constant(&x, load(&format!("{ds}__{tag}_r")));
    let dr = max_abs_diff(&out.corr, &want, true, &ctx);
    println!("{ctx}: max|Δr| = {dr:.3e}");
    assert!(dr <= tol_r, "{ctx}: Δr {dr:e}");
    if let (Some(tol), Some(p)) = (tol_p, out.pvalues) {
        let wp = mask_constant(&x, load(&format!("{ds}__{tag}_p")));
        let (dp, lim) = match tol {
            Tol::Rel(t) => (max_rel_diff(&p, &wp, true, &ctx), t),
            Tol::Abs(t) => (max_abs_diff(&p, &wp, true, &ctx), t),
        };
        println!("{ctx}: max Δp = {dp:.3e}");
        assert!(dp <= lim, "{ctx}: Δp {dp:e}");
    }
}

#[test]
fn gaussian_rank_parity() {
    let c = Correlator::new(Method::GaussianRank).pvalues(PvalueSpec::Analytic);
    datasets().for_each(|ds| compare(&c, ds, "gaussian_rank", 1e-13, Some(Tol::Rel(1e-10))));
}

#[test]
fn partial_parity() {
    for (m, tag) in [(Method::Pearson, "partial_pearson"), (Method::Spearman, "partial_spearman")] {
        for ds in datasets() {
            let z = load(&format!("{ds}__covariates"));
            let c = Correlator::new(m.clone()).covariates(z).pvalues(PvalueSpec::Analytic);
            compare(&c, ds, tag, 1e-12, Some(Tol::Rel(1e-9)));
        }
    }
}

#[test]
fn xi_parity() {
    let c = Correlator::new(Method::Chatterjee(ChatterjeeParams::default())).pvalues(PvalueSpec::Analytic);
    datasets().for_each(|ds| compare(&c, ds, "xi", 1e-14, None));
    // XICOR (no ties in these datasets, so tie-breaking cannot differ;
    // it computes p as 1 − pnorm(z), hence the absolute tolerance).
    for ds in datasets().filter(|ds| exists(&format!("{ds}__xi_xicor_r"))) {
        compare(&c, ds, "xi_xicor", 1e-14, Some(Tol::Abs(1e-14)));
    }
}

#[test]
fn dcor_parity() {
    let c = Correlator::new(Method::DistanceCorrelation);
    datasets()
        .filter(|ds| exists(&format!("{ds}__dcor_r")))
        .for_each(|ds| compare(&c, ds, "dcor", 1e-12, None));
}

#[test]
fn rho_parity() {
    let c = Correlator::new(Method::Proportionality(ProportionalityParams::default()));
    datasets()
        .filter(|ds| exists(&format!("{ds}__rho_r")))
        .for_each(|ds| compare(&c, ds, "rho", 1e-13, None));
}
