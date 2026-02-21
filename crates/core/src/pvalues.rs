use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

pub fn students_t_for_corr(n_samples: usize) -> Option<StudentsT> {
    if n_samples <= 2 {
        return None;
    }
    StudentsT::new(0.0, 1.0, n_samples as f64 - 2.0).ok()
}

pub fn corr_pvalue_from_t_dist(r: f64, n_samples: usize, t_dist: &StudentsT) -> f64 {
    if !r.is_finite() {
        return f64::NAN;
    }
    let r = r.clamp(-1.0, 1.0);
    if r.abs() >= 1.0 {
        return 0.0;
    }

    let denom = 1.0 - r * r;
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }

    let df = n_samples as f64 - 2.0;
    let t_abs = (r.abs() * (df / denom).sqrt()).abs();
    let p = 2.0 * t_dist.cdf(-t_abs);
    p.clamp(0.0, 1.0)
}

pub fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("Normal(0,1) should always be constructible")
}

pub fn two_sided_pvalue_from_z(z: f64, normal: &Normal) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    let p = 2.0 * normal.cdf(-z.abs());
    p.clamp(0.0, 1.0)
}

pub fn hellcor_empirical_pvalue(observed: f64, null_sorted: &[f64]) -> f64 {
    if !observed.is_finite() {
        return f64::NAN;
    }
    if null_sorted.is_empty() {
        return f64::NAN;
    }
    let first_ge = null_sorted.partition_point(|v| *v < observed);
    let count_ge = null_sorted.len() - first_ge;
    (count_ge as f64 + 1.0) / (null_sorted.len() as f64 + 1.0)
}
