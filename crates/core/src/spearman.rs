use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;
use crate::{rank::rank_data, pearson::correlation_matrix as pearson_correlation_matrix};
use crate::pearson::correlation_upper_triangle as pearson_upper_triangle;


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
