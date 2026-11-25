//! Centring and scaling to unit Euclidean norm.

use super::RowStatus;

/// Accurate mean: plain mean followed by one correction pass, which keeps
/// the result within a few ulp of the exact mean.
pub fn mean(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let corr = x.iter().map(|v| v - m).sum::<f64>() / n;
    m + corr
}

/// Write `x − mean(x)` scaled to unit norm into `out`.
///
/// Returns [`RowStatus::Degenerate`] (and leaves `out` zeroed) when fewer
/// than two values are given, when all values are equal, or when the norm
/// is not a positive finite number.
pub fn center_unit_norm(x: &[f64], out: &mut [f64]) -> RowStatus {
    let m = mean(x);
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v - m;
    }
    scale_unit_norm(out, x.len() >= 2 && !all_equal(x))
}

/// Scale `v` in place to unit Euclidean norm. `ok` = false forces the
/// degenerate outcome.
pub fn scale_unit_norm(v: &mut [f64], ok: bool) -> RowStatus {
    let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    if !ok || !(norm > 0.0 && norm.is_finite()) {
        v.fill(0.0);
        return RowStatus::Degenerate;
    }
    let inv = 1.0 / norm;
    v.iter_mut().for_each(|a| *a *= inv);
    RowStatus::Ok
}

/// True when every element equals the first (also true for empty input).
pub fn all_equal(x: &[f64]) -> bool {
    x.first().is_none_or(|&f| x.iter().all(|&v| v == f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_norm() {
        let mut out = [0.0; 4];
        assert_eq!(center_unit_norm(&[1.0, 2.0, 3.0, 4.0], &mut out), RowStatus::Ok);
        let ss: f64 = out.iter().map(|v| v * v).sum();
        assert!((ss - 1.0).abs() < 1e-15);
        assert!(out.iter().sum::<f64>().abs() < 1e-15);
    }

    #[test]
    fn degenerate() {
        let mut out = [9.0; 3];
        assert_eq!(center_unit_norm(&[0.1, 0.1, 0.1], &mut out), RowStatus::Degenerate);
        assert_eq!(out, [0.0; 3]);
        let mut one = [0.0];
        assert_eq!(center_unit_norm(&[5.0], &mut one), RowStatus::Degenerate);
    }
}
