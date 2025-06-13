use ndarray::Array2;

pub fn pearson_correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
    let (_n_rows, n_cols) = data.dim();
    
    let means = data.mean_axis(ndarray::Axis(1)).unwrap().insert_axis(ndarray::Axis(1));
    let centered = data - &means;

    // Compute covariance matrix using BLAS
    let cov_matrix = centered.dot(&centered.t()) / (n_cols as f64 - 1.0);

    // Compute standard deviations using BLAS
    let variances = cov_matrix.diag().mapv(|x| x.sqrt());
    let std_matrix = &variances.view().insert_axis(ndarray::Axis(1)) 
                   * &variances.view().insert_axis(ndarray::Axis(0));

    // Element-wise division to get correlation matrix
    &cov_matrix / &std_matrix
} 