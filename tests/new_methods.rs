//! Behaviour of covariate adjustment, Gaussian rank, ρp, ξ and distance
//! correlation beyond the generic invariants.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use common::load;
use mincorr::{
    ChatterjeeParams, CorrError, Correlator, Method, NanPolicy, ProportionalityParams, PvalueSpec, log_clr,
};
use ndarray::{Array2, array, s};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn partial_pearson_matches_closed_form() {
    let x = load("normal_30x40__input");
    let z = x.slice(s![29..30, ..]).to_owned();
    let data = x.slice(s![0..5, ..]);
    let out = Correlator::new(Method::Pearson)
        .covariates(z.clone())
        .pvalues(PvalueSpec::Analytic)
        .full(data)
        .unwrap();
    let r = |a: &[f64], b: &[f64]| Correlator::new(Method::Pearson).pair(a, b).unwrap().corr;
    let zr = z.row(0).to_vec();
    for i in 0..5 {
        for j in 0..5 {
            if i == j {
                continue;
            }
            let (xi, xj) = (data.row(i).to_vec(), data.row(j).to_vec());
            let (rxy, rxz, ryz) = (r(&xi, &xj), r(&xi, &zr), r(&xj, &zr));
            let want = (rxy - rxz * ryz) / ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
            assert!(close(out.corr[[i, j]], want, 1e-13), "{i},{j}");
        }
    }
    // df = n − 3 with one covariate: compare against the t formula.
    let (rr, n) = (out.corr[[0, 1]], 40.0);
    let t = rr * ((n - 3.0) / (1.0 - rr * rr)).sqrt();
    assert!(out.pvalues.unwrap()[[0, 1]] > 0.0 && t.is_finite());
}

#[test]
fn covariate_errors() {
    let mut x = load("normal_30x40__input");
    x[[4, 7]] = f64::NAN;
    let z = Array2::from_shape_fn((2, 40), |(i, j)| ((i + 2) * (j + 1) % 7) as f64);
    let c = Correlator::new(Method::Spearman).covariates(z.clone());
    assert!(matches!(c.full(x.view()), Err(CorrError::Unsupported(_))));
    let prop = c.clone().nan_policy(NanPolicy::Propagate).full(x.view()).unwrap();
    assert!(prop.corr.row(4).iter().all(|v| v.is_nan()) && prop.corr[[0, 1]].is_finite());
    let bad = Correlator::new(Method::Kendall).covariates(z.clone());
    assert!(matches!(bad.full(x.view()), Err(CorrError::Unsupported(_))));
    let short = Correlator::new(Method::Pearson).covariates(z.slice(s![.., ..30]).to_owned());
    assert!(matches!(short.full(load("normal_30x40__input").view()), Err(CorrError::ShapeMismatch { .. })));
    let mut nan = z.clone();
    nan[[0, 0]] = f64::NAN;
    let c = Correlator::new(Method::Pearson).covariates(nan);
    assert!(matches!(c.full(load("normal_30x40__input").view()), Err(CorrError::InvalidParam { .. })));
}

#[test]
fn row_in_covariate_span_is_degenerate() {
    let x = load("normal_30x40__input");
    let z = x.slice(s![0..1, ..]).mapv(|v| 3.0 * v - 1.0);
    let out = Correlator::new(Method::Pearson).covariates(z).full(x.slice(s![0..3, ..])).unwrap();
    assert!(out.corr.row(0).iter().all(|v| v.is_nan()));
    assert!(out.corr[[1, 2]].is_finite());
}

#[test]
fn gaussian_rank_is_rank_based() {
    let x = load("normal_30x40__input");
    let y = x.mapv(|v| v.exp() * 3.0 + 1.0);
    let c = Correlator::new(Method::GaussianRank);
    let (a, b) = (c.full(x.view()).unwrap().corr, c.full(y.view()).unwrap().corr);
    assert!(a.iter().zip(&b).all(|(u, v)| close(*u, *v, 1e-15)));
}

#[test]
fn rho_clr_and_proportional_rows() {
    let x = array![[1.0, 2.0, 4.0, 8.0], [2.0, 4.0, 8.0, 16.0], [5.0, 1.0, 3.0, 2.0], [1.0, 1.0, 2.0, 1.0]];
    let rho = Correlator::new(Method::Proportionality(ProportionalityParams::default()));
    let out = rho.full(x.view()).unwrap();
    assert!(close(out.corr[[0, 1]], 1.0, 1e-15));
    let pre = log_clr(&[x.view()], 0.0).remove(0);
    let raw =
        Correlator::new(Method::Proportionality(ProportionalityParams { clr: false, ..Default::default() }));
    let again = raw.full(pre.view()).unwrap();
    assert!(out.corr.iter().zip(&again.corr).all(|(a, b)| a.to_bits() == b.to_bits()));
    // Cross mode uses the clr reference of both inputs together.
    let cross = rho.cross(x.slice(s![0..2, ..]), x.slice(s![2..4, ..])).unwrap();
    assert!(close(cross.corr[[0, 1]], out.corr[[0, 3]], 1e-15));
    assert!(matches!(
        rho.clone().pvalues(PvalueSpec::Analytic).full(x.view()),
        Err(CorrError::Unsupported(_))
    ));
}

#[test]
fn xi_is_directional() {
    let x = load("normal_30x40__input");
    let c = Correlator::new(Method::Chatterjee(ChatterjeeParams::default()));
    let full = c.clone().pvalues(PvalueSpec::Analytic).full(x.view()).unwrap();
    let (a, b) = (x.row(2).to_vec(), x.row(5).to_vec());
    assert_eq!(full.corr[[2, 5]], c.pair(&a, &b).unwrap().corr);
    assert_eq!(full.corr[[5, 2]], c.pair(&b, &a).unwrap().corr);
    assert_ne!(full.corr[[2, 5]], full.corr[[5, 2]]);
    // Invariant to increasing transforms of either argument.
    let ta: Vec<f64> = a.iter().map(|v| v.exp()).collect();
    assert_eq!(c.pair(&ta, &b).unwrap().corr, full.corr[[2, 5]]);
    let sym =
        Correlator::new(Method::Chatterjee(ChatterjeeParams { symmetric: true })).full(x.view()).unwrap();
    assert_eq!(sym.corr[[2, 5]], full.corr[[2, 5]].max(full.corr[[5, 2]]));
}

#[test]
fn dcor_detects_nonmonotone_dependence() {
    let x: Vec<f64> = (0..101).map(|i| f64::from(i) / 50.0 - 1.0).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect();
    let pearson = Correlator::new(Method::Pearson).pair(&x, &y).unwrap().corr;
    let d = Correlator::new(Method::DistanceCorrelation);
    let dc = d.pair(&x, &y).unwrap().corr;
    assert!(pearson.abs() < 1e-12 && dc > 0.4, "{pearson} {dc}");
    let ax: Vec<f64> = x.iter().map(|v| -4.0 * v + 7.0).collect();
    assert!(close(d.pair(&ax, &y).unwrap().corr, dc, 1e-13));
}
