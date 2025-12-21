use crate::hellinger::nmath::{lbeta, ml_warn_return_nan, r_finite, ML_NEGINF};

pub fn dbeta(x: f64, a: f64, b: f64, log_p: bool) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return x + a + b;
    }
    if a < 0.0 || b < 0.0 {
        return ml_warn_return_nan();
    }
    if !r_finite(a) || !r_finite(b) {
        return if log_p { ML_NEGINF } else { 0.0 };
    }
    if x < 0.0 || x > 1.0 {
        return if log_p { ML_NEGINF } else { 0.0 };
    }

    if x == 0.0 {
        if a < 1.0 {
            return if log_p { f64::INFINITY } else { f64::INFINITY };
        }
        if a > 1.0 {
            return if log_p { ML_NEGINF } else { 0.0 };
        }
        let log_d = -lbeta(1.0, b);
        return if log_p { log_d } else { log_d.exp() };
    }
    if x == 1.0 {
        if b < 1.0 {
            return if log_p { f64::INFINITY } else { f64::INFINITY };
        }
        if b > 1.0 {
            return if log_p { ML_NEGINF } else { 0.0 };
        }
        let log_d = -lbeta(a, 1.0);
        return if log_p { log_d } else { log_d.exp() };
    }

    let log_d = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - lbeta(a, b);
    if log_p { log_d } else { log_d.exp() }
}
