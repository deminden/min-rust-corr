use crate::pvalues::{corr_pvalue_from_t_dist, students_t_for_corr};
use crate::upper::upper_triangular_len;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;

pub fn correlation_matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (_n_rows, n_cols) = data.dim();

    let means = data
        .mean_axis(ndarray::Axis(1))
        .unwrap()
        .insert_axis(ndarray::Axis(1));
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

pub fn correlation_cross_matrix<S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Array2<f64>
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
{
    let (lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();
    assert_eq!(
        lhs_cols, rhs_cols,
        "Pearson cross-correlation requires equal sample count in both matrices"
    );

    if lhs_rows == 0 || rhs_rows == 0 || lhs_cols == 0 {
        return Array2::<f64>::zeros((lhs_rows, rhs_rows));
    }
    if lhs_cols < 2 {
        return Array2::<f64>::from_elem((lhs_rows, rhs_rows), f64::NAN);
    }

    let means_lhs = lhs
        .mean_axis(ndarray::Axis(1))
        .expect("non-empty along sample axis");
    let means_rhs = rhs
        .mean_axis(ndarray::Axis(1))
        .expect("non-empty along sample axis");

    let centered_lhs = lhs - &means_lhs.view().insert_axis(ndarray::Axis(1));
    let centered_rhs = rhs - &means_rhs.view().insert_axis(ndarray::Axis(1));

    let denom = lhs_cols as f64 - 1.0;
    let cov = centered_lhs.dot(&centered_rhs.t()) / denom;

    let std_lhs = centered_lhs
        .map_axis(ndarray::Axis(1), |row| {
            let ss: f64 = row.iter().map(|v| v * v).sum();
            (ss / denom).sqrt()
        })
        .to_vec();
    let std_rhs = centered_rhs
        .map_axis(ndarray::Axis(1), |row| {
            let ss: f64 = row.iter().map(|v| v * v).sum();
            (ss / denom).sqrt()
        })
        .to_vec();

    let mut corr = cov;
    for i in 0..lhs_rows {
        for j in 0..rhs_rows {
            let scale = std_lhs[i] * std_rhs[j];
            corr[[i, j]] = if scale.is_finite() && scale != 0.0 {
                corr[[i, j]] / scale
            } else {
                f64::NAN
            };
        }
    }

    corr
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

fn corr_to_pvalues(corr: &Array2<f64>, n_samples: usize, set_diag_zero: bool) -> Array2<f64> {
    let (n_rows, n_cols) = corr.dim();
    let mut pvals = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);

    if set_diag_zero {
        for i in 0..n_rows.min(n_cols) {
            pvals[[i, i]] = 0.0;
        }
    }

    let Some(t_dist) = students_t_for_corr(n_samples) else {
        return pvals;
    };

    for i in 0..n_rows {
        for j in 0..n_cols {
            if !(set_diag_zero && i == j) {
                pvals[[i, j]] = corr_pvalue_from_t_dist(corr[[i, j]], n_samples, &t_dist);
            }
        }
    }

    pvals
}

pub fn correlation_matrix_with_pvalues<S>(data: &ArrayBase<S, Ix2>) -> (Array2<f64>, Array2<f64>)
where
    S: Data<Elem = f64> + Sync,
{
    let corr = correlation_matrix(data);
    let pvals = corr_to_pvalues(&corr, data.ncols(), true);
    (corr, pvals)
}

pub fn correlation_cross_matrix_with_pvalues<S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> (Array2<f64>, Array2<f64>)
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
{
    let corr = correlation_cross_matrix(lhs, rhs);
    let pvals = corr_to_pvalues(&corr, lhs.ncols(), false);
    (corr, pvals)
}
