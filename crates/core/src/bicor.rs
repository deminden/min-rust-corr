use crate::pvalues::{corr_pvalue_from_t_dist, students_t_for_corr};
use crate::upper::upper_triangular_len;
use ndarray::{Array2, ArrayBase, ArrayView1, Data, Ix2};
use rayon::prelude::*;

const REF_UX: f64 = 0.5;

#[derive(Copy, Clone)]
enum FallbackMode {
    Individual,
}

struct PreparedRow {
    values: Vec<f64>,
    na_flag: bool,
}

fn quantile_in_place(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = (values.len() as f64 - 1.0) * q;
    let k0 = idx.floor() as usize;
    let k1 = idx.ceil() as usize;

    if k0 == k1 {
        values.select_nth_unstable_by(k0, |a, b| a.partial_cmp(b).unwrap());
        return values[k0];
    }

    values.select_nth_unstable_by(k0, |a, b| a.partial_cmp(b).unwrap());
    let v0 = values[k0];
    values.select_nth_unstable_by(k1, |a, b| a.partial_cmp(b).unwrap());
    let v1 = values[k1];

    v0 + (v1 - v0) * (idx - k0 as f64)
}

fn median(values: &mut Vec<f64>) -> f64 {
    quantile_in_place(values, 0.5)
}

fn prepare_col_cor(row: ArrayView1<f64>, cosine: bool) -> PreparedRow {
    let mut res = vec![0.0; row.len()];
    let mut count = 0usize;
    let mut mean = 0.0;
    let mut sum = 0.0;

    for &v in row.iter() {
        if v.is_finite() {
            count += 1;
            mean += v;
            sum += v * v;
        }
    }

    if count == 0 {
        return PreparedRow {
            values: res,
            na_flag: true,
        };
    }

    if cosine {
        mean = 0.0;
    } else {
        mean /= count as f64;
    }

    let var = sum - (count as f64) * mean * mean;
    let denom = var.sqrt();

    if denom == 0.0 || !denom.is_finite() {
        return PreparedRow {
            values: res,
            na_flag: true,
        };
    }

    for (idx, &v) in row.iter().enumerate() {
        if v.is_finite() {
            res[idx] = (v - mean) / denom;
        }
    }

    PreparedRow {
        values: res,
        na_flag: false,
    }
}

fn prepare_col_bicor(
    row: ArrayView1<f64>,
    max_p_outliers: f64,
    fallback: FallbackMode,
    cosine: bool,
) -> PreparedRow {
    let n = row.len();
    let mut res = vec![0.0; n];
    let mut deviations = vec![0.0; n];

    let mut finite_vals: Vec<f64> = row.iter().cloned().filter(|v| v.is_finite()).collect();
    if finite_vals.is_empty() {
        return PreparedRow {
            values: res,
            na_flag: true,
        };
    }

    let med = median(&mut finite_vals);
    if !med.is_finite() {
        return PreparedRow {
            values: res,
            na_flag: true,
        };
    }

    let med_x = if cosine { 0.0 } else { med };

    for (idx, &v) in row.iter().enumerate() {
        if v.is_finite() {
            res[idx] = v - med_x;
            deviations[idx] = (v - med).abs();
        } else {
            deviations[idx] = f64::NAN;
        }
    }

    let mut mad_vals: Vec<f64> = deviations
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .collect();
    let mad = median(&mut mad_vals);

    if mad == 0.0 || !mad.is_finite() {
        return match fallback {
            FallbackMode::Individual => prepare_col_cor(row, cosine),
        };
    }

    let denom = 9.0 * mad;
    for (idx, &v) in row.iter().enumerate() {
        if v.is_finite() {
            deviations[idx] = (v - med) / denom;
        } else {
            deviations[idx] = f64::NAN;
        }
    }

    let mut aux_vals: Vec<f64> = deviations
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .collect();
    let mut low_q = quantile_in_place(&mut aux_vals, max_p_outliers);
    let mut aux_vals: Vec<f64> = deviations
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .collect();
    let mut hi_q = quantile_in_place(&mut aux_vals, 1.0 - max_p_outliers);

    if low_q > -REF_UX {
        low_q = -REF_UX;
    }
    if hi_q < REF_UX {
        hi_q = REF_UX;
    }
    low_q = low_q.abs();

    for v in deviations.iter_mut().filter(|v| v.is_finite()) {
        if *v < 0.0 {
            *v = *v * REF_UX / low_q;
        } else {
            *v = *v * REF_UX / hi_q;
        }
    }

    let mut sum_sq = 0.0;
    for (idx, v) in deviations.iter().enumerate() {
        if v.is_finite() {
            let mut ux = v.abs();
            if ux > 1.0 {
                ux = 1.0;
            }
            let ux = 1.0 - ux * ux;
            let weight = ux * ux;
            res[idx] *= weight;
            sum_sq += res[idx] * res[idx];
        } else {
            res[idx] = 0.0;
        }
    }

    let denom = sum_sq.sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return PreparedRow {
            values: vec![0.0; n],
            na_flag: true,
        };
    }

    for v in res.iter_mut() {
        *v /= denom;
    }

    PreparedRow {
        values: res,
        na_flag: false,
    }
}

pub fn correlation_matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();
    if n_rows == 0 || n_cols == 0 {
        return Array2::<f64>::zeros((n_rows, n_rows));
    }

    let max_p_outliers = 1.0;
    let fallback = FallbackMode::Individual;
    let cosine = false;

    let prepared: Vec<PreparedRow> = (0..n_rows)
        .into_par_iter()
        .map(|i| prepare_col_bicor(data.row(i), max_p_outliers, fallback, cosine))
        .collect();

    let mut flat = Vec::with_capacity(n_rows * n_cols);
    for row in &prepared {
        flat.extend_from_slice(&row.values);
    }
    let prep_matrix =
        Array2::from_shape_vec((n_rows, n_cols), flat).expect("prepared matrix shape mismatch");

    let mut corr = prep_matrix.dot(&prep_matrix.t());

    for i in 0..n_rows {
        for j in 0..n_rows {
            if prepared[i].na_flag || prepared[j].na_flag {
                corr[[i, j]] = f64::NAN;
            } else if corr[[i, j]] > 1.0 {
                corr[[i, j]] = 1.0;
            } else if corr[[i, j]] < -1.0 {
                corr[[i, j]] = -1.0;
            }
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
        "Biweight midcorrelation cross-matrix requires equal sample count in both matrices"
    );

    if lhs_rows == 0 || rhs_rows == 0 || lhs_cols == 0 {
        return Array2::<f64>::zeros((lhs_rows, rhs_rows));
    }

    let max_p_outliers = 1.0;
    let fallback = FallbackMode::Individual;
    let cosine = false;

    let prepared_lhs: Vec<PreparedRow> = (0..lhs_rows)
        .into_par_iter()
        .map(|i| prepare_col_bicor(lhs.row(i), max_p_outliers, fallback, cosine))
        .collect();
    let prepared_rhs: Vec<PreparedRow> = (0..rhs_rows)
        .into_par_iter()
        .map(|i| prepare_col_bicor(rhs.row(i), max_p_outliers, fallback, cosine))
        .collect();

    let mut flat_lhs = Vec::with_capacity(lhs_rows * lhs_cols);
    for row in &prepared_lhs {
        flat_lhs.extend_from_slice(&row.values);
    }
    let prep_lhs = Array2::from_shape_vec((lhs_rows, lhs_cols), flat_lhs)
        .expect("prepared lhs shape mismatch");

    let mut flat_rhs = Vec::with_capacity(rhs_rows * rhs_cols);
    for row in &prepared_rhs {
        flat_rhs.extend_from_slice(&row.values);
    }
    let prep_rhs = Array2::from_shape_vec((rhs_rows, rhs_cols), flat_rhs)
        .expect("prepared rhs shape mismatch");

    let mut corr = prep_lhs.dot(&prep_rhs.t());
    for i in 0..lhs_rows {
        for j in 0..rhs_rows {
            if prepared_lhs[i].na_flag || prepared_rhs[j].na_flag {
                corr[[i, j]] = f64::NAN;
            } else if corr[[i, j]] > 1.0 {
                corr[[i, j]] = 1.0;
            } else if corr[[i, j]] < -1.0 {
                corr[[i, j]] = -1.0;
            }
        }
    }

    corr
}

pub fn correlation_upper_triangle<S>(data: &ArrayBase<S, Ix2>) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();
    if n_rows == 0 || n_cols == 0 {
        return Vec::new();
    }

    let max_p_outliers = 1.0;
    let fallback = FallbackMode::Individual;
    let cosine = false;

    let prepared: Vec<PreparedRow> = (0..n_rows)
        .into_par_iter()
        .map(|i| prepare_col_bicor(data.row(i), max_p_outliers, fallback, cosine))
        .collect();

    let row_results: Vec<Vec<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n_rows - i];
            row[0] = 1.0;
            for j in i + 1..n_rows {
                if prepared[i].na_flag || prepared[j].na_flag {
                    row[j - i] = f64::NAN;
                    continue;
                }
                let mut dot = 0.0;
                for k in 0..n_cols {
                    dot += prepared[i].values[k] * prepared[j].values[k];
                }
                let mut corr = dot;
                if corr > 1.0 {
                    corr = 1.0;
                } else if corr < -1.0 {
                    corr = -1.0;
                }
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
