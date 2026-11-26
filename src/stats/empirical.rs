//! Monte-Carlo (empirical) upper-tail p-values.

/// p = (1 + #{null ≥ obs}) / (B + 1) for a null sample sorted ascending.
/// NaN when `obs` is not finite or the null is empty.
pub fn upper_tail_p(obs: f64, null_sorted: &[f64]) -> f64 {
    if !obs.is_finite() || null_sorted.is_empty() {
        return f64::NAN;
    }
    let first_ge = null_sorted.partition_point(|v| *v < obs);
    let count_ge = null_sorted.len() - first_ge;
    (count_ge as f64 + 1.0) / (null_sorted.len() as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts() {
        let null = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(upper_tail_p(0.25, &null), 3.0 / 5.0);
        assert_eq!(upper_tail_p(0.5, &null), 1.0 / 5.0);
        assert_eq!(upper_tail_p(0.0, &null), 1.0);
        assert!(upper_tail_p(f64::NAN, &null).is_nan());
    }
}
