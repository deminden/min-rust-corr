//! Property tests: Kendall S against brute force, method invariances and
//! panic-freedom on arbitrary bit patterns.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mincorr::methods::kendall::knight::kendall_s_brute;
use mincorr::methods::kendall::s_statistic;
use mincorr::{Correlator, Method, NanPolicy, PvalueSpec};
use ndarray::Array2;
use proptest::prelude::*;

fn tied_vec(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec((0..6i32).prop_map(f64::from), n)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn kendall_s_matches_brute((x, y) in (2usize..300).prop_flat_map(|n| (tied_vec(n), tied_vec(n)))) {
        prop_assert_eq!(s_statistic(&x, &y).unwrap(), kendall_s_brute(&x, &y));
    }

    #[test]
    fn kendall_bitset_matches_knight((x, y) in (4usize..300).prop_flat_map(|n| (tied_vec(n), tied_vec(n)))) {
        // Matrix path (bitsets for n <= 128) vs gathered path (Knight).
        let m = Array2::from_shape_vec((2, x.len()), [x.clone(), y.clone()].concat()).unwrap();
        let full = Correlator::new(Method::Kendall).full(m.view()).unwrap().corr[[0, 1]];
        let mut xn = x.clone();
        xn.push(f64::NAN);
        let mut yn = y.clone();
        yn.push(1.0);
        let pair = Correlator::new(Method::Kendall).pair(&xn, &yn).unwrap().corr;
        prop_assert!(full.to_bits() == pair.to_bits() || (full.is_nan() && pair.is_nan()));
    }

    #[test]
    fn pearson_affine_and_sign(x in prop::collection::vec(-1e3f64..1e3, 5..60), a in 0.1f64..10.0, b in -5f64..5.0) {
        let y: Vec<f64> = x.iter().enumerate().map(|(i, v)| v.sin() + i as f64 * 0.01).collect();
        let c = Correlator::new(Method::Pearson);
        let r = c.pair(&x, &y).unwrap().corr;
        let xa: Vec<f64> = x.iter().map(|v| a * v + b).collect();
        let xn: Vec<f64> = x.iter().map(|v| -v).collect();
        prop_assert!((c.pair(&xa, &y).unwrap().corr - r).abs() < 1e-12 || r.is_nan());
        prop_assert!((c.pair(&xn, &y).unwrap().corr + r).abs() < 1e-12 || r.is_nan());
    }

    #[test]
    fn rank_methods_monotone_invariant(x in prop::collection::vec(-5f64..5.0, 5..60)) {
        let y: Vec<f64> = x.iter().enumerate().map(|(i, v)| (v * 3.0).cos() + (i % 7) as f64).collect();
        let xt: Vec<f64> = x.iter().map(|v| v.exp()).collect();
        for m in [Method::Spearman, Method::Kendall] {
            let c = Correlator::new(m);
            let (r0, r1) = (c.pair(&x, &y).unwrap().corr, c.pair(&xt, &y).unwrap().corr);
            prop_assert!((r0 - r1).abs() < 1e-14 || (r0.is_nan() && r1.is_nan()));
        }
    }

    #[test]
    fn bicor_ignores_moving_outlier(x in prop::collection::vec(-1f64..1.0, 10..40), far in 50f64..1e6) {
        let y: Vec<f64> = x.iter().enumerate().map(|(i, v)| v + (i as f64 * 0.37).sin()).collect();
        let c = Correlator::new(Method::from_name("bicor").unwrap());
        let mut x1 = x.clone();
        x1[0] = 50.0;
        let mut x2 = x.clone();
        x2[0] = far;
        let (a, b) = (c.pair(&x1, &y).unwrap().corr, c.pair(&x2, &y).unwrap().corr);
        prop_assert!((a - b).abs() < 1e-12 || (a.is_nan() && b.is_nan()));
    }

    #[test]
    fn arbitrary_bits_never_panic(bits in prop::collection::vec(any::<u64>(), 12..40), cols in 2usize..6) {
        let vals: Vec<f64> = bits.iter().map(|b| f64::from_bits(*b)).collect();
        let rows = vals.len() / cols;
        let m = Array2::from_shape_vec((rows, cols), vals[..rows * cols].to_vec()).unwrap();
        for name in ["pearson", "spearman", "kendall", "bicor", "hellcor"] {
            let method = Method::from_name(name).unwrap();
            let pv = if name == "hellcor" { PvalueSpec::None } else { PvalueSpec::Analytic };
            for pol in [NanPolicy::Error, NanPolicy::Propagate, NanPolicy::PairwiseComplete] {
                if let Ok(o) = Correlator::new(method.clone()).nan_policy(pol).pvalues(pv).full(m.view()) {
                    prop_assert!(o.corr.iter().all(|v| v.is_nan() || v.abs() <= 1.0));
                }
            }
        }
    }
}
