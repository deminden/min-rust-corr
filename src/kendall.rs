use ndarray::{Array1, Array2};
use std::sync::Mutex;
use rayon::prelude::*;
use kendalls::tau_b;

use crate::rank_data;

/// Compute Kendall (Tau-b) correlation matrix for a given data matrix
pub fn kendall_correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
    let size = data.nrows();

    let correlation_matrix = Mutex::new(Array2::<f64>::from_elem((size, size), f64::NAN));

    (0..size).into_par_iter().for_each(|i| {
        for j in i..size {
            if i == j {
                correlation_matrix.lock().unwrap()[[i, j]] = 1.0;
                continue;
            }

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
                let mut matrix = correlation_matrix.lock().unwrap();
                matrix[[i, j]] = tau;
                matrix[[j, i]] = tau;
            }
        }
    });

    correlation_matrix.into_inner().unwrap()
} 