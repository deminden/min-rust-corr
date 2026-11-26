//! Parity for bicor (WGCNA) and hellcor (HellCor). These fixtures need the
//! R packages WGCNA and HellCor; each check is skipped when its fixture
//! file is absent (see docs/parity.md).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use common::*;
use mincorr::{BicorParams, Correlator, HellcorParams, Method, NanPolicy, PvalueSpec};

#[test]
fn bicor_default_parity() {
    for ds in DATASETS {
        let name = format!("{ds}__bicor_r");
        if !exists(&name) {
            println!("skip {name}: fixture absent");
            continue;
        }
        let x = load(&format!("{ds}__input"));
        let c = Correlator::new(Method::Bicor(BicorParams::default())).pvalues(PvalueSpec::Analytic);
        let out = c.full(x.view()).unwrap();
        let dr = max_abs_diff(&out.corr, &load(&name), true, &name);
        println!("{name}: max|Δr| = {dr:.3e}");
        assert!(dr <= 1e-13, "{name}: {dr:e}");
        if exists(&format!("{ds}__bicor_p")) {
            let dp =
                max_rel_diff(out.pvalues.as_ref().unwrap(), &load(&format!("{ds}__bicor_p")), true, &name);
            println!("{name}: max rel Δp = {dp:.3e}");
            assert!(dp <= 1e-11, "{name} p: {dp:e}");
        }
    }
}

#[test]
fn bicor_max_p_outliers_parity() {
    for ds in DATASETS {
        for (tag, p) in [("bicor_mpo010", 0.1), ("bicor_mpo005", 0.05)] {
            let name = format!("{ds}__{tag}_r");
            if !exists(&name) {
                continue;
            }
            let x = load(&format!("{ds}__input"));
            let m = Method::Bicor(BicorParams { max_p_outliers: p, ..Default::default() });
            let out = Correlator::new(m).full(x.view()).unwrap();
            let want = load(&name);
            let d = out.corr.iter().zip(want.iter()).filter(|(a, b)| a.is_finite() && b.is_finite());
            let worst = d.map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("{name}: max|Δr| = {worst:.3e} (maxPOutliers = {p})");
            assert!(worst <= 1e-13, "{name}: {worst:e}");
        }
    }
}

#[test]
fn hellcor_parity_report() {
    for ds in DATASETS {
        let name = format!("{ds}__hellcor_r");
        if !exists(&name) {
            println!("skip {name}: fixture absent");
            continue;
        }
        let x = load(&format!("{ds}__input"));
        let c = Correlator::new(Method::Hellcor(HellcorParams::default())).nan_policy(NanPolicy::Error);
        let out = c.full(x.view()).unwrap();
        let want = load(&name);
        let (mut close, mut total, mut worst) = (0usize, 0usize, 0.0f64);
        // HellCor returns ~1 for a constant row against any other row; by
        // design (docs/semantics.md) degenerate rows give NaN here.
        let constant = |i: usize| x.row(i).iter().all(|v| *v == x[[i, 0]]);
        for ((i, j), w) in want.indexed_iter() {
            if constant(i) || constant(j) {
                assert!(out.corr[[i, j]].is_nan());
                continue;
            }
            if i < j && w.is_finite() {
                let d = (out.corr[[i, j]] - w).abs();
                total += 1;
                close += usize::from(d <= 1e-9);
                worst = worst.max(d);
            }
        }
        println!("{name}: {close}/{total} pairs within 1e-9, max|Δη| = {worst:.3e}");
        assert_eq!(close, total, "{name}: only {close}/{total} pairs agree");
    }
}

#[test]
fn hellcor_alpha_parity() {
    for ds in DATASETS {
        for (tag, alpha) in [("hellcor_alpha03", 3.0), ("hellcor_alpha10", 10.0)] {
            let name = format!("{ds}__{tag}_r");
            if !exists(&name) {
                continue;
            }
            let x = load(&format!("{ds}__input"));
            let out = Correlator::new(Method::Hellcor(HellcorParams { alpha, ..Default::default() }))
                .full(x.view())
                .unwrap();
            let want = load(&name);
            let constant = |i: usize| x.row(i).iter().all(|v| *v == x[[i, 0]]);
            let mut worst = 0.0f64;
            for ((i, j), w) in want.indexed_iter() {
                if i != j && !constant(i) && !constant(j) && w.is_finite() {
                    worst = worst.max((out.corr[[i, j]] - w).abs());
                }
            }
            println!("{name}: max|Δη| = {worst:.3e}");
            assert!(worst <= 1e-9, "{name}: {worst:e}");
        }
    }
}
