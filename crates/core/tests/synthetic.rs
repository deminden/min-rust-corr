use mincorr::{bicor, hellcor, kendall, pearson, spearman};
use ndarray::Array2;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn synthetic_pairwise_checks() {
    let data = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
        .expect("matrix shape");

    let pearson_corr = pearson::matrix(&data);
    assert!(approx_eq(pearson_corr[[0, 1]], -1.0, 1e-12));

    let spearman_corr = spearman::matrix(&data);
    assert!(approx_eq(spearman_corr[[0, 1]], -1.0, 1e-12));

    let kendall_corr = kendall::matrix(&data);
    assert!(approx_eq(kendall_corr[[0, 1]], -1.0, 1e-12));
}

#[test]
fn synthetic_identity_checks() {
    let data = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        .expect("matrix shape");

    let bicor_corr = bicor::matrix(&data);
    assert_eq!(bicor_corr.nrows(), 2);
    assert_eq!(bicor_corr.ncols(), 2);
    assert!(approx_eq(bicor_corr[[0, 1]], 1.0, 1e-12));
    assert!(approx_eq(bicor_corr[[0, 1]], bicor_corr[[1, 0]], 1e-12));

    let hellcor_corr = hellcor::matrix(&data);
    assert_eq!(hellcor_corr.nrows(), 2);
    assert_eq!(hellcor_corr.ncols(), 2);
    assert!(hellcor_corr[[0, 0]].is_finite());
    assert!(hellcor_corr[[1, 1]].is_finite());
    assert!(hellcor_corr[[0, 1]].is_finite());
    assert!(approx_eq(hellcor_corr[[0, 1]], hellcor_corr[[1, 0]], 1e-12));
}
