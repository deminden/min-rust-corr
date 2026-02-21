use kendalls::tau_b;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix2};
use rayon::prelude::*;

use crate::pvalues::{standard_normal, two_sided_pvalue_from_z};
use crate::rank::rank_data;

fn filtered_pair(row_i: ArrayView1<'_, f64>, row_j: ArrayView1<'_, f64>) -> (Vec<f64>, Vec<f64>) {
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

    (filtered_x, filtered_y)
}

fn kendall_tau_from_filtered(filtered_x: &[f64], filtered_y: &[f64]) -> Option<f64> {
    if filtered_x.len() < 2 {
        return None;
    }

    let rank1 = rank_data(&Array1::from(filtered_x.to_vec()));
    let rank2 = rank_data(&Array1::from(filtered_y.to_vec()));

    let x_ranks: Vec<i32> = rank1.iter().map(|&x| x as i32).collect();
    let y_ranks: Vec<i32> = rank2.iter().map(|&y| y as i32).collect();

    tau_b(&x_ranks, &y_ranks).ok().map(|(tau, _)| tau)
}

fn tie_group_sizes(values: &[f64]) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut groups = Vec::new();
    let mut run = 1usize;
    for i in 1..sorted.len() {
        if sorted[i] == sorted[i - 1] {
            run += 1;
        } else {
            if run > 1 {
                groups.push(run);
            }
            run = 1;
        }
    }
    if run > 1 {
        groups.push(run);
    }
    groups
}

fn kendall_pvalue_tie_aware(
    tau: f64,
    filtered_x: &[f64],
    filtered_y: &[f64],
    normal: &statrs::distribution::Normal,
) -> f64 {
    let n = filtered_x.len();
    if n < 2 || !tau.is_finite() {
        return f64::NAN;
    }

    let n_f = n as f64;
    let n0 = n_f * (n_f - 1.0) / 2.0;

    let x_ties = tie_group_sizes(filtered_x);
    let y_ties = tie_group_sizes(filtered_y);

    let n1 = x_ties
        .iter()
        .map(|&t| (t as f64) * (t as f64 - 1.0) / 2.0)
        .sum::<f64>();
    let n2 = y_ties
        .iter()
        .map(|&u| (u as f64) * (u as f64 - 1.0) / 2.0)
        .sum::<f64>();

    let denom_s = ((n0 - n1) * (n0 - n2)).sqrt();
    if denom_s == 0.0 || !denom_s.is_finite() {
        return f64::NAN;
    }
    let s = tau * denom_s;

    let sum_t = x_ties
        .iter()
        .map(|&t| {
            let tf = t as f64;
            tf * (tf - 1.0) * (2.0 * tf + 5.0)
        })
        .sum::<f64>();
    let sum_u = y_ties
        .iter()
        .map(|&u| {
            let uf = u as f64;
            uf * (uf - 1.0) * (2.0 * uf + 5.0)
        })
        .sum::<f64>();
    let term1 = (n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) - sum_t - sum_u) / 18.0;

    let sum_t1 = x_ties
        .iter()
        .map(|&t| {
            let tf = t as f64;
            tf * (tf - 1.0)
        })
        .sum::<f64>();
    let sum_u1 = y_ties
        .iter()
        .map(|&u| {
            let uf = u as f64;
            uf * (uf - 1.0)
        })
        .sum::<f64>();
    let term2 = (sum_t1 * sum_u1) / (2.0 * n_f * (n_f - 1.0));

    let term3 = if n > 2 {
        let sum_t2 = x_ties
            .iter()
            .map(|&t| {
                let tf = t as f64;
                tf * (tf - 1.0) * (tf - 2.0)
            })
            .sum::<f64>();
        let sum_u2 = y_ties
            .iter()
            .map(|&u| {
                let uf = u as f64;
                uf * (uf - 1.0) * (uf - 2.0)
            })
            .sum::<f64>();
        (sum_t2 * sum_u2) / (9.0 * n_f * (n_f - 1.0) * (n_f - 2.0))
    } else {
        0.0
    };

    let var_s = term1 + term2 + term3;
    if var_s <= 0.0 || !var_s.is_finite() {
        return f64::NAN;
    }

    let z = s / var_s.sqrt();
    two_sided_pvalue_from_z(z, &normal)
}

fn kendall_tau_and_pvalue(
    row_i: ArrayView1<'_, f64>,
    row_j: ArrayView1<'_, f64>,
    normal: &statrs::distribution::Normal,
) -> (f64, f64) {
    let (filtered_x, filtered_y) = filtered_pair(row_i, row_j);
    let Some(tau) = kendall_tau_from_filtered(&filtered_x, &filtered_y) else {
        return (f64::NAN, f64::NAN);
    };
    let pval = kendall_pvalue_tie_aware(tau, &filtered_x, &filtered_y, normal);
    (tau, pval)
}

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

                let (filtered_x, filtered_y) = filtered_pair(row_i, row_j);
                if let Some(tau) = kendall_tau_from_filtered(&filtered_x, &filtered_y) {
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

                let (filtered_x, filtered_y) = filtered_pair(row_i, row_j);
                if let Some(tau) = kendall_tau_from_filtered(&filtered_x, &filtered_y) {
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

                let (filtered_x, filtered_y) = filtered_pair(row_i, row_j);
                if let Some(tau) = kendall_tau_from_filtered(&filtered_x, &filtered_y) {
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

pub fn correlation_matrix_with_pvalues<S>(data: &ArrayBase<S, Ix2>) -> (Array2<f64>, Array2<f64>)
where
    S: Data<Elem = f64> + Sync,
{
    let size = data.nrows();
    let normal = standard_normal();

    let row_results: Vec<(Vec<f64>, Vec<f64>)> = (0..size)
        .into_par_iter()
        .map(|i| {
            let mut corr_row = vec![f64::NAN; size - i];
            let mut p_row = vec![f64::NAN; size - i];
            corr_row[0] = 1.0;
            p_row[0] = 0.0;
            for j in i + 1..size {
                let (tau, pval) = kendall_tau_and_pvalue(data.row(i), data.row(j), &normal);
                corr_row[j - i] = tau;
                p_row[j - i] = pval;
            }
            (corr_row, p_row)
        })
        .collect();

    let mut corr = Array2::<f64>::from_elem((size, size), f64::NAN);
    let mut pvals = Array2::<f64>::from_elem((size, size), f64::NAN);
    for i in 0..size {
        for j in i..size {
            let c = row_results[i].0[j - i];
            let p = row_results[i].1[j - i];
            corr[[i, j]] = c;
            corr[[j, i]] = c;
            pvals[[i, j]] = p;
            pvals[[j, i]] = p;
        }
    }

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
    let (lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();
    assert_eq!(
        lhs_cols, rhs_cols,
        "Kendall cross-correlation requires equal sample count in both matrices"
    );
    let normal = standard_normal();

    let row_results: Vec<(Vec<f64>, Vec<f64>)> = (0..lhs_rows)
        .into_par_iter()
        .map(|i| {
            let mut corr_row = vec![f64::NAN; rhs_rows];
            let mut p_row = vec![f64::NAN; rhs_rows];
            for j in 0..rhs_rows {
                let (tau, pval) = kendall_tau_and_pvalue(lhs.row(i), rhs.row(j), &normal);
                corr_row[j] = tau;
                p_row[j] = pval;
            }
            (corr_row, p_row)
        })
        .collect();

    let mut corr = Array2::<f64>::from_elem((lhs_rows, rhs_rows), f64::NAN);
    let mut pvals = Array2::<f64>::from_elem((lhs_rows, rhs_rows), f64::NAN);
    for i in 0..lhs_rows {
        for j in 0..rhs_rows {
            corr[[i, j]] = row_results[i].0[j];
            pvals[[i, j]] = row_results[i].1[j];
        }
    }

    (corr, pvals)
}
