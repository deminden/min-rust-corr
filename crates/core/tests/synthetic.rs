use mincorr::{kendall, pearson, spearman};
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
