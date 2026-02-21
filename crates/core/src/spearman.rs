use crate::pearson::correlation_cross_matrix as pearson_cross_matrix;
use crate::pearson::correlation_upper_triangle as pearson_upper_triangle;
use crate::pvalues::{corr_pvalue_from_t_dist, students_t_for_corr};
use crate::{pearson::correlation_matrix as pearson_correlation_matrix, rank::rank_data};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;

pub fn correlation_matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();

    // Rank-transform each row
    let rank_rows: Vec<Array1<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = data.row(i).to_owned();
            rank_data(&row)
        })
        .collect();

    let mut rank_matrix = Array2::<f64>::zeros((n_rows, n_cols));
    for (i, rank_row) in rank_rows.iter().enumerate() {
        rank_matrix.row_mut(i).assign(rank_row);
    }

    pearson_correlation_matrix(&rank_matrix)
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
        "Spearman cross-correlation requires equal sample count in both matrices"
    );

    let rank_rows_lhs: Vec<Array1<f64>> = (0..lhs_rows)
        .into_par_iter()
        .map(|i| {
            let row = lhs.row(i).to_owned();
            rank_data(&row)
        })
        .collect();
    let rank_rows_rhs: Vec<Array1<f64>> = (0..rhs_rows)
        .into_par_iter()
        .map(|i| {
            let row = rhs.row(i).to_owned();
            rank_data(&row)
        })
        .collect();

    let mut rank_lhs = Array2::<f64>::zeros((lhs_rows, lhs_cols));
    for (i, rank_row) in rank_rows_lhs.iter().enumerate() {
        rank_lhs.row_mut(i).assign(rank_row);
    }
    let mut rank_rhs = Array2::<f64>::zeros((rhs_rows, rhs_cols));
    for (i, rank_row) in rank_rows_rhs.iter().enumerate() {
        rank_rhs.row_mut(i).assign(rank_row);
    }

    pearson_cross_matrix(&rank_lhs, &rank_rhs)
}

pub fn correlation_upper_triangle<S>(data: &ArrayBase<S, Ix2>) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();

    // Rank-transform each row
    let rank_rows: Vec<Array1<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = data.row(i).to_owned();
            rank_data(&row)
        })
        .collect();

    let mut rank_matrix = Array2::<f64>::zeros((n_rows, n_cols));
    for (i, rank_row) in rank_rows.iter().enumerate() {
        rank_matrix.row_mut(i).assign(rank_row);
    }

    pearson_upper_triangle(&rank_matrix)
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
