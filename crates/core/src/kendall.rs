use kendalls::tau_b;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;

use crate::rank::rank_data;

/// Compute Kendall (Tau-b) correlation matrix for a given data matrix
pub fn correlation_matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let size = data.nrows();

    let row_results: Vec<Vec<f64>> = (0..size)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; size - i];
            row[0] = 1.0;
            for j in i + 1..size {
                let row_i = data.row(i);
                let row_j = data.row(j);

                // Filter out NaN pairs
                let mut filtered_x = Vec::with_capacity(row_i.len());
                let mut filtered_y = Vec::with_capacity(row_j.len());

                for k in 0..row_i.len() {
                    let x = row_i[k];
                    let y = row_j[k];
                    if !x.is_nan() && !y.is_nan() {
                        filtered_x.push(x);
                        filtered_y.push(y);
                    }
                }

                if filtered_x.len() < 2 {
                    continue;
                }

                // Rank the filtered values
                let rank1 = rank_data(&Array1::from(filtered_x));
                let rank2 = rank_data(&Array1::from(filtered_y));

                let x_ranks: Vec<i32> = rank1.iter().map(|&x| x as i32).collect();
                let y_ranks: Vec<i32> = rank2.iter().map(|&y| y as i32).collect();

                if let Ok((tau, _)) = tau_b(&x_ranks, &y_ranks) {
                    row[j - i] = tau;
                }
            }
            row
        })
        .collect();

    let mut corr = Array2::<f64>::from_elem((size, size), f64::NAN);
    for i in 0..size {
        for j in i..size {
            let val = row_results[i][j - i];
            corr[[i, j]] = val;
            corr[[j, i]] = val;
        }
    }

    corr
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
        "Kendall cross-correlation requires equal sample count in both matrices"
    );

    let row_results: Vec<Vec<f64>> = (0..lhs_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; rhs_rows];
            for j in 0..rhs_rows {
                let row_i = lhs.row(i);
                let row_j = rhs.row(j);

                let mut filtered_x = Vec::with_capacity(row_i.len());
                let mut filtered_y = Vec::with_capacity(row_j.len());

                for k in 0..row_i.len() {
                    let x = row_i[k];
                    let y = row_j[k];
                    if !x.is_nan() && !y.is_nan() {
                        filtered_x.push(x);
                        filtered_y.push(y);
                    }
                }

                if filtered_x.len() < 2 {
                    continue;
                }

                let rank1 = rank_data(&Array1::from(filtered_x));
                let rank2 = rank_data(&Array1::from(filtered_y));

                let x_ranks: Vec<i32> = rank1.iter().map(|&x| x as i32).collect();
                let y_ranks: Vec<i32> = rank2.iter().map(|&y| y as i32).collect();

                if let Ok((tau, _)) = tau_b(&x_ranks, &y_ranks) {
                    row[j] = tau;
                }
            }
            row
        })
        .collect();

    let mut corr = Array2::<f64>::from_elem((lhs_rows, rhs_rows), f64::NAN);
    for i in 0..lhs_rows {
        for j in 0..rhs_rows {
            corr[[i, j]] = row_results[i][j];
        }
    }

    corr
}

pub fn correlation_upper_triangle<S>(data: &ArrayBase<S, Ix2>) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let size = data.nrows();
    if size == 0 {
        return Vec::new();
    }

    let row_results: Vec<Vec<f64>> = (0..size)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; size - i];
            row[0] = 1.0;
            for j in i + 1..size {
                let row_i = data.row(i);
                let row_j = data.row(j);

                // Filter out NaN pairs
                let mut filtered_x = Vec::with_capacity(row_i.len());
                let mut filtered_y = Vec::with_capacity(row_j.len());

                for k in 0..row_i.len() {
                    let x = row_i[k];
                    let y = row_j[k];
                    if !x.is_nan() && !y.is_nan() {
                        filtered_x.push(x);
                        filtered_y.push(y);
                    }
                }

                if filtered_x.len() < 2 {
                    continue;
                }

                // Rank the filtered values
                let rank1 = rank_data(&Array1::from(filtered_x));
                let rank2 = rank_data(&Array1::from(filtered_y));

                let x_ranks: Vec<i32> = rank1.iter().map(|&x| x as i32).collect();
                let y_ranks: Vec<i32> = rank2.iter().map(|&y| y as i32).collect();

                if let Ok((tau, _)) = tau_b(&x_ranks, &y_ranks) {
                    row[j - i] = tau;
                }
            }
            row
        })
        .collect();

    let mut packed = Vec::with_capacity(crate::upper::upper_triangular_len(size));
    for row in row_results {
        packed.extend_from_slice(&row);
    }
    packed
}
