use ndarray::{Array1, Array2};
use rayon::prelude::*;
use crate::{rank_data, pearson::pearson_correlation_matrix};


pub fn spearman_correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
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