//! Clean-room Beta functions against R's qbeta / pbeta / dbeta.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mincorr::stats::beta;

fn rows() -> Vec<[f64; 6]> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/beta_grid.tsv");
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split('\t').map(|s| s.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4], v[5]]
        })
        .collect()
}

fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 { a.abs() } else { ((a - b) / b).abs() }
}

/// HellCor-relevant regime: a = b >= 1.
fn relevant(a: f64, b: f64) -> bool {
    a == b && a >= 1.0
}

#[test]
fn cdf_pdf_match_r() {
    let (mut worst_c, mut worst_d) = (0.0f64, 0.0f64);
    for [a, b, _, x, cdf, pdf] in rows() {
        let (ec, ed) = (rel(beta::cdf(x, a, b), cdf), rel(beta::pdf(x, a, b), pdf));
        let tol = if relevant(a, b) { 1e-13 } else { 1e-10 };
        assert!(ec <= tol, "cdf a={a} b={b} x={x}: rel {ec:e}");
        assert!(ed <= tol, "pdf a={a} b={b} x={x}: rel {ed:e}");
        worst_c = worst_c.max(ec);
        worst_d = worst_d.max(ed);
    }
    println!("max rel error: cdf {worst_c:.2e}, pdf {worst_d:.2e}");
}

#[test]
fn quantile_matches_r() {
    let mut worst = 0.0f64;
    for [a, b, p, q, _, _] in rows() {
        if p.is_nan() {
            continue;
        }
        let e = rel(beta::quantile(p, a, b), q);
        let tol = if relevant(a, b) { 1e-12 } else { 1e-10 };
        assert!(e <= tol, "quantile a={a} b={b} p={p}: rel {e:e}");
        worst = worst.max(e);
    }
    println!("max rel error: quantile {worst:.2e}");
}
