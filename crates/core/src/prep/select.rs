//! Median and quantiles by selection (O(m) expected), ordered with
//! `f64::total_cmp`. Inputs must be finite; slices are reordered in place.

/// Median of `v` (mean of the two middle order statistics for even length).
/// Returns NaN for an empty slice.
pub fn median(v: &mut [f64]) -> f64 {
    quantile(v, 0.5)
}

/// Quantile with linear interpolation between order statistics at
/// position (m−1)·q (R's type 7). Returns NaN for an empty slice or a `q`
/// outside [0, 1].
pub fn quantile(v: &mut [f64], q: f64) -> f64 {
    let m = v.len();
    if m == 0 || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    let h = (m - 1) as f64 * q;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    let (_, lo_val, rest) = v.select_nth_unstable_by(lo, f64::total_cmp);
    let lo_val = *lo_val;
    if frac == 0.0 || rest.is_empty() {
        return lo_val;
    }
    let hi_val = rest.iter().copied().fold(f64::INFINITY, f64::min);
    lo_val + frac * (hi_val - lo_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medians() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&mut []).is_nan());
    }

    #[test]
    fn type7() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&mut v, 0.1), 1.4);
        assert_eq!(quantile(&mut v, 1.0), 5.0);
        assert_eq!(quantile(&mut v, 0.0), 1.0);
    }
}
