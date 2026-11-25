//! Generic layout and semantics invariants, parameterised over all methods
//! and NaN policies.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use common::load;
use mincorr::{CorrError, Correlator, Method, NanPolicy, PvalueSpec, packed_index};
use ndarray::{Array2, s};

pub fn methods() -> Vec<Method> {
    ["pearson", "spearman", "kendall", "bicor", "hellcor"]
        .iter()
        .map(|m| Method::from_name(m).unwrap())
        .collect()
}

fn pv(m: &Method) -> PvalueSpec {
    if m.is_pairwise() && m.name() == "hellcor" {
        PvalueSpec::MonteCarlo { samples: 99, seed: 7 }
    } else {
        PvalueSpec::Analytic
    }
}

fn same(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

fn data() -> Array2<f64> {
    let mut x = load("na_12x40__input");
    x.row_mut(3).fill(2.5); // constant row
    x
}

#[test]
fn layouts_agree() {
    let x = data();
    for m in methods() {
        let c = Correlator::new(m.clone()).pvalues(pv(&m));
        let full = c.full(x.view()).unwrap();
        let cross = c.cross(x.view(), x.view()).unwrap();
        let n = x.nrows();
        for i in 0..n {
            for j in 0..n {
                let (f, g) = (full.corr[[i, j]], cross.corr[[i, j]]);
                if i != j {
                    assert!(
                        f.to_bits() == g.to_bits() || (f.is_nan() && g.is_nan()),
                        "{} cross({i},{j})",
                        m.name()
                    );
                }
                assert!(f.to_bits() == full.corr[[j, i]].to_bits() || f.is_nan(), "{} symmetry", m.name());
            }
        }
        let up = c.upper(x.view()).unwrap();
        for i in 0..n {
            for j in i..n {
                let (a, b) = (up.corr[packed_index(n, i, j)], full.corr[[i, j]]);
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{} upper", m.name());
            }
        }
        for br in [1, 3, 5, 100] {
            let mut got = Array2::<f64>::zeros((n, n));
            let mut gotp = Array2::<f64>::zeros((n, n));
            c.clone()
                .block_rows(br)
                .full_blocks(x.view(), |b| {
                    for i in b.rows.clone() {
                        got.row_mut(i).assign(&ndarray::ArrayView1::from(b.corr_row(i)));
                        gotp.row_mut(i).assign(&ndarray::ArrayView1::from(b.pvalue_row(i).unwrap()));
                    }
                    Ok(())
                })
                .unwrap();
            assert!(same(&got, &full.corr), "{} blocks {br}", m.name());
            assert!(same(&gotp, full.pvalues.as_ref().unwrap()), "{} block p {br}", m.name());
        }
    }
}

#[test]
fn thread_count_determinism() {
    let x = load("normal_30x40__input");
    for m in methods() {
        let run = |t| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
            pool.install(|| Correlator::new(m.clone()).pvalues(pv(&m)).full(x.view()).unwrap())
        };
        let (a, b, c) = (run(1), run(2), run(8));
        assert!(same(&a.corr, &b.corr) && same(&a.corr, &c.corr), "{}", m.name());
        assert!(same(a.pvalues.as_ref().unwrap(), c.pvalues.as_ref().unwrap()), "{} p", m.name());
    }
}

#[test]
fn degenerate_rows_and_diagonal() {
    let x = data();
    for m in methods() {
        let out = Correlator::new(m.clone()).pvalues(pv(&m)).full(x.view()).unwrap();
        let p = out.pvalues.unwrap();
        for i in 0..x.nrows() {
            assert!(p[[i, i]].is_nan());
            if i == 3 {
                assert!(out.corr.row(3).iter().all(|v| v.is_nan()), "{} constant row", m.name());
            } else {
                assert_eq!(out.corr[[i, i]], 1.0, "{} diag {i}", m.name());
            }
        }
        assert!(out.corr.iter().all(|v| v.is_nan() || v.abs() <= 1.0));
    }
}

#[test]
fn nan_policies() {
    let x = data();
    for m in methods() {
        let c = Correlator::new(m.clone());
        assert!(matches!(
            c.clone().nan_policy(NanPolicy::Error).full(x.view()),
            Err(CorrError::NonFinite { .. })
        ));
        let prop = c.clone().nan_policy(NanPolicy::Propagate).full(x.view()).unwrap();
        let census = mincorr::prep::validity::nonfinite_counts(x.view());
        for (i, &k) in census.iter().enumerate() {
            if k > 0 {
                assert!(prop.corr.row(i).iter().all(|v| v.is_nan()), "{} propagate row {i}", m.name());
            }
        }
        // Pairwise-complete equals computing the pair on its common subset.
        let full = c.full(x.view()).unwrap();
        let (i, j) = (0, 1);
        let pair = c.pair(&x.row(i).to_vec(), &x.row(j).to_vec()).unwrap();
        assert!((pair.corr - full.corr[[i, j]]).abs() < 1e-14, "{} pair", m.name());
    }
}

#[test]
fn shape_errors_never_panic() {
    let a = Array2::<f64>::zeros((3, 5));
    let b = Array2::<f64>::zeros((2, 4));
    let e = Array2::<f64>::zeros((0, 5));
    for m in methods() {
        let c = Correlator::new(m.clone());
        assert!(matches!(c.cross(a.view(), b.view()), Err(CorrError::ShapeMismatch { .. })));
        assert!(matches!(c.full(e.view()), Err(CorrError::Empty { .. })));
        assert!(c.pair(&[1.0, 2.0], &[1.0]).is_err());
        for n in 0..5 {
            let x = Array2::from_shape_fn((3, n), |(i, j)| ((i + 1) * (j + 2)) as f64 % 7.0);
            let _ = c.full(x.view());
            let _ = c.full(x.slice(s![.., ..]));
        }
    }
}
