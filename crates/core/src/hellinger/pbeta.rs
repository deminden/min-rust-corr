use crate::hellinger::nmath::{
    ML_NEGINF, ml_warn_return_nan, r_dt_0, r_dt_1, r_finite, r_log1_exp,
};
use crate::hellinger::toms708::bratio;

pub(crate) fn pbeta_raw(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x >= 1.0 {
        return r_dt_1(lower_tail, log_p);
    }
    if a == 0.0 || b == 0.0 || !r_finite(a) || !r_finite(b) {
        if a == 0.0 && b == 0.0 {
            return if log_p { -std::f64::consts::LN_2 } else { 0.5 };
        }
        if a == 0.0 || a / b == 0.0 {
            return r_dt_1(lower_tail, log_p);
        }
        if b == 0.0 || b / a == 0.0 {
            return r_dt_0(lower_tail, log_p);
        }
        if x < 0.5 {
            return r_dt_0(lower_tail, log_p);
        }
        return r_dt_1(lower_tail, log_p);
    }
    if x <= 0.0 {
        return r_dt_0(lower_tail, log_p);
    }

    let x1 = 0.5 - x + 0.5;
    let mut w = 0.0;
    let mut wc = 0.0;
    let mut ierr = 0;
    bratio(a, b, x, x1, &mut w, &mut wc, &mut ierr, log_p);
    if log_p {
        if std::env::var("DEBUG_QBETA_LOG").is_ok() {
            let ab_match = ((a - 10.0).abs() < 1e-12 && (b - 1e8).abs() < 1e-6)
                || ((a - 1e8).abs() < 1e-6 && (b - 10.0).abs() < 1e-12);
            let x_match = (x - 6.6028727e-6).abs() < 1e-8
                || (x - 4.528728484e-38).abs() < 1e-40
                || (x - 0.9999103979486001).abs() < 1e-9
                || (x - 1.164437215e-4).abs() < 1e-10;
            if ab_match && x_match {
                eprintln!(
                    "DBG pbeta_raw pre-adjust: x={x:.17e} a={a:.17e} b={b:.17e} w={w:.17e} wc={wc:.17e} ierr={ierr}"
                );
            }
        }
        if w == ML_NEGINF && wc == 0.0 {
            let mean = a / (a + b);
            if r_finite(mean) && x > mean {
                w = 0.0;
                wc = ML_NEGINF;
            }
        } else if wc == ML_NEGINF && w == 0.0 {
            let mean = a / (a + b);
            if r_finite(mean) && x < mean {
                wc = 0.0;
                w = ML_NEGINF;
            }
        } else if w == ML_NEGINF && r_finite(wc) {
            w = r_log1_exp(wc);
        } else if wc == ML_NEGINF && r_finite(w) {
            wc = r_log1_exp(w);
        } else if w == ML_NEGINF && wc == ML_NEGINF {
            if x > 0.5 {
                w = 0.0;
            } else {
                wc = 0.0;
            }
        }
        if w == 0.0 && r_finite(wc) && wc < 0.0 {
            w = r_log1_exp(wc);
        }
    }
    if std::env::var("DEBUG_QBETA_LOG").is_ok() && log_p {
        let ab_match = ((a - 10.0).abs() < 1e-12 && (b - 1e8).abs() < 1e-6)
            || ((a - 1e8).abs() < 1e-6 && (b - 10.0).abs() < 1e-12);
        let x_match = (x - 6.6028727e-6).abs() < 1e-8
            || (x - 4.528728484e-38).abs() < 1e-40
            || (x - 0.9999103979486001).abs() < 1e-9
            || (x - 1.164437215e-4).abs() < 1e-10;
        if ab_match && x_match {
            eprintln!(
                "DBG pbeta_raw: x={x:.17e} a={a:.17e} b={b:.17e} w={w:.17e} wc={wc:.17e} ierr={ierr}"
            );
        }
    }
    if ierr != 0 && ierr != 11 && ierr != 14 {
        // Ignore warnings here to match standalone behavior.
    }
    if lower_tail { w } else { wc }
}

#[allow(dead_code)]
pub fn pbeta(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return x + a + b;
    }
    if a < 0.0 || b < 0.0 {
        return ml_warn_return_nan();
    }
    pbeta_raw(x, a, b, lower_tail, log_p)
}
