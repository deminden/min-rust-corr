//! The v0.2 function names still work (adapted from the author-original
//! synthetic tests).
#![allow(deprecated, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mincorr::{bicor, kendall, pearson, spearman};
use ndarray::Array2;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn synthetic_pairwise_checks() {
    let data = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    assert!(approx_eq(pearson::matrix(&data)[[0, 1]], -1.0, 1e-12));
    assert!(approx_eq(spearman::matrix(&data)[[0, 1]], -1.0, 1e-12));
    assert!(approx_eq(kendall::matrix(&data)[[0, 1]], -1.0, 1e-12));
    assert!(approx_eq(bicor::matrix(&data)[[0, 1]], -1.0, 1e-12));
}

#[test]
fn legacy_shapes_and_mismatch() {
    let a = Array2::from_shape_vec((3, 4), (0..12).map(|v| (v * v) as f64).collect()).unwrap();
    let b = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
    assert_eq!(pearson::correlation_upper_triangle(&a).len(), 6);
    let (c, p) = spearman::correlation_matrix_with_pvalues(&a);
    assert_eq!((c.dim(), p.dim()), ((3, 3), (3, 3)));
    // Mismatched columns: NaN-filled instead of a panic.
    let x = pearson::correlation_cross_matrix(&a, &b);
    assert_eq!(x.dim(), (3, 2));
    assert!(x.iter().all(|v| v.is_nan()));
}
