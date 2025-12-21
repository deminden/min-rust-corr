use ndarray::Array2;
use ndarray::ArrayView1;
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
        return PreparedRow { values: res, na_flag: true };
    }

    if cosine {
        mean = 0.0;
    } else {
        mean /= count as f64;
    }

    let var = sum - (count as f64) * mean * mean;
    let denom = var.sqrt();

    if denom == 0.0 || !denom.is_finite() {
        return PreparedRow { values: res, na_flag: true };
    }

    for (idx, &v) in row.iter().enumerate() {
        if v.is_finite() {
            res[idx] = (v - mean) / denom;
        }
    }

    PreparedRow { values: res, na_flag: false }
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
        return PreparedRow { values: res, na_flag: true };
    }

    let med = median(&mut finite_vals);
    if !med.is_finite() {
        return PreparedRow { values: res, na_flag: true };
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

    let mut mad_vals: Vec<f64> = deviations.iter().cloned().filter(|v| v.is_finite()).collect();
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

    let mut aux_vals: Vec<f64> = deviations.iter().cloned().filter(|v| v.is_finite()).collect();
    let mut low_q = quantile_in_place(&mut aux_vals, max_p_outliers);
    let mut aux_vals: Vec<f64> = deviations.iter().cloned().filter(|v| v.is_finite()).collect();
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
        return PreparedRow { values: vec![0.0; n], na_flag: true };
    }

    for v in res.iter_mut() {
        *v /= denom;
    }

    PreparedRow { values: res, na_flag: false }
}

pub fn correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
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
    let prep_matrix = Array2::from_shape_vec((n_rows, n_cols), flat)
        .expect("prepared matrix shape mismatch");

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
