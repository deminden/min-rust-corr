//! Log and centred log-ratio (clr) transforms for compositional data.

use ndarray::{Array2, ArrayView2};

/// log(x + pseudocount); values whose shifted value is not a positive
/// finite number become NaN (and are then handled by the NaN policy).
pub fn log_shifted(x: f64, pseudocount: f64) -> f64 {
    let v = x + pseudocount;
    if v > 0.0 && v.is_finite() { v.ln() } else { f64::NAN }
}

/// Centred log-ratio transform of one or more matrices that share their
/// columns (observations, e.g. samples) and together hold all features
/// (rows): every value becomes log(x + pseudocount) minus the mean of the
/// finite log values of its column over all rows of all `parts`.
///
/// Apply it to the whole feature table before selecting rows, since the
/// reference (the column mean) depends on every feature.
pub fn log_clr(parts: &[ArrayView2<'_, f64>], pseudocount: f64) -> Vec<Array2<f64>> {
    let mut logs: Vec<Array2<f64>> = parts.iter().map(|p| p.mapv(|x| log_shifted(x, pseudocount))).collect();
    let m = logs.first().map_or(0, |a| a.ncols());
    let (mut sum, mut cnt) = (vec![0.0; m], vec![0usize; m]);
    for a in &logs {
        for row in a.rows() {
            for (j, &v) in row.iter().enumerate().take(m) {
                if v.is_finite() {
                    sum[j] += v;
                    cnt[j] += 1;
                }
            }
        }
    }
    let mean: Vec<f64> =
        sum.iter().zip(&cnt).map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN }).collect();
    for a in &mut logs {
        for mut row in a.rows_mut() {
            row.iter_mut().zip(&mean).for_each(|(v, &mu)| *v -= mu);
        }
    }
    logs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn clr_columns_sum_to_zero() {
        let x = array![[1.0, 4.0], [2.0, 8.0], [4.0, 0.0]];
        let out = log_clr(&[x.view()], 0.0);
        let c0: f64 = out[0].column(0).sum();
        assert!(c0.abs() < 1e-15);
        assert!(out[0][[2, 1]].is_nan());
        assert!((out[0][[1, 1]] - 0.5 * 2f64.ln()).abs() < 1e-15);
    }
}
