use ndarray::{Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;
use crate::upper::{upper_triangular_len};

pub fn correlation_matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
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

pub fn matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_matrix(data)
}

pub fn correlation_upper_triangle<S>(data: &ArrayBase<S, Ix2>) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();
    if n_rows == 0 {
        return Vec::new();
    }

    let denom = n_cols as f64 - 1.0;
    let mut means = Vec::with_capacity(n_rows);
    let mut stds = Vec::with_capacity(n_rows);

    for i in 0..n_rows {
        let row = data.row(i);
        let mut sum = 0.0;
        for &v in row.iter() {
            sum += v;
        }
        let mean = sum / n_cols as f64;
        let mut ss = 0.0;
        for &v in row.iter() {
            let d = v - mean;
            ss += d * d;
        }
        let var = if denom > 0.0 { ss / denom } else { f64::NAN };
        means.push(mean);
        stds.push(var.sqrt());
    }

    let row_results: Vec<Vec<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n_rows - i];
            row[0] = 1.0;
            for j in i + 1..n_rows {
                let std_i = stds[i];
                let std_j = stds[j];
                if !std_i.is_finite() || !std_j.is_finite() || std_i == 0.0 || std_j == 0.0 {
                    row[j - i] = f64::NAN;
                    continue;
                }
                let mut cov = 0.0;
                let row_i = data.row(i);
                let row_j = data.row(j);
                for k in 0..n_cols {
                    cov += (row_i[k] - means[i]) * (row_j[k] - means[j]);
                }
                let corr = if denom > 0.0 {
                    cov / denom / (std_i * std_j)
                } else {
                    f64::NAN
                };
                row[j - i] = corr;
            }
            row
        })
        .collect();

    let mut packed = Vec::with_capacity(upper_triangular_len(n_rows));
    for row in row_results {
        packed.extend_from_slice(&row);
    }
    packed
}
