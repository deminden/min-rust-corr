//! Fixture parity for Pearson, Spearman and Kendall against R.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use common::*;
use mincorr::{Correlator, Method, PvalueSpec};

fn check(method: Method, tol_r: f64, tol_p: f64) {
    let name = method.name();
    for ds in DATASETS {
        let x = load(&format!("{ds}__input"));
        let want_r = load(&format!("{ds}__{name}_r"));
        let want_p = load(&format!("{ds}__{name}_p"));
        let out = Correlator::new(method.clone()).pvalues(PvalueSpec::Analytic).full(x.view()).unwrap();
        let ctx = format!("{name}/{ds}");
        let dr = max_abs_diff(&out.corr, &want_r, true, &ctx);
        let p = out.pvalues.unwrap();
        // R's cor.test returns a p-value for pairs with n_eff < 3 where we
        // document NaN; compare only where R is defined and n_eff >= 3.
        let dp = max_rel_diff(&p, &want_p, true, &ctx);
        println!("{ctx}: max|Δr| = {dr:.3e}, max rel Δp = {dp:.3e}");
        assert!(dr <= tol_r, "{ctx}: Δr {dr:e}");
        assert!(dp <= tol_p, "{ctx}: Δp {dp:e}");
        for i in 0..x.nrows() {
            let d = out.corr[[i, i]];
            assert!(d == 1.0 || d.is_nan(), "{ctx}: diagonal {d}");
            assert!(p[[i, i]].is_nan(), "{ctx}: diagonal p");
        }
    }
}

#[test]
fn pearson_parity() {
    check(Method::Pearson, 1e-13, 1e-11);
}

#[test]
fn spearman_parity() {
    check(Method::Spearman, 1e-13, 1e-11);
}

#[test]
fn kendall_parity() {
    check(Method::Kendall, 1e-14, 1e-11);
}
