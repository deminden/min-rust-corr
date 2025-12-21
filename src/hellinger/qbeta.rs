use crate::hellinger::nmath::{
    fmax2, fmin2, lbeta, ml_warn_return_nan, ml_warning, r_d_half, r_dt_0, r_dt_1, r_dt_civ,
    r_dt_clog, r_dt_log, r_dt_qiv, r_log1_exp, r_pow_di, r_finite, M_LN2, ML_NAN, ML_NEGINF,
};
use crate::hellinger::pbeta::pbeta_raw;

const USE_LOG_X_CUTOFF: f64 = -5.0;
const N_NEWTON_FREE: i32 = 4;

const DBL_VERY_MIN: f64 = f64::MIN_POSITIVE / 4.0;
const DBL_LOG_V_MIN: f64 = -709.782712893384; // ln(DBL_MIN/4)
const DBL_1__EPS: f64 = 0.9999999999999999; // 1 - 2^-53

const ACU_MIN: f64 = 1e-300;
const FPU: f64 = 3e-308;
const P_LO: f64 = FPU;
const P_HI: f64 = 1.0 - 2.22e-16;

const CONST1: f64 = 2.30753;
const CONST2: f64 = 0.27061;
const CONST3: f64 = 0.99229;
const CONST4: f64 = 0.04481;

fn is_debug_case(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> bool {
    std::env::var("DEBUG_QBETA_LOG").is_ok()
        && log_p
        && !lower_tail
        && (alpha - 1e-300_f64.ln()).abs() < 1e-12
        && (p - 10.0).abs() < 1e-12
        && (q - 1e8).abs() < 1e-6
}

fn is_debug_case_upper(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> bool {
    std::env::var("DEBUG_QBETA_LOG").is_ok()
        && !log_p
        && !lower_tail
        && alpha == 1e-300
        && (p - 0.5).abs() < 1e-12
        && (q - 1e8).abs() < 1e-6
}

fn is_debug_case_upper_ab(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> bool {
    std::env::var("DEBUG_QBETA_LOG").is_ok()
        && !log_p
        && !lower_tail
        && alpha == 1e-300
        && (p - 10.0).abs() < 1e-12
        && (q - 1e8).abs() < 1e-6
}

fn is_debug_case_swapped(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> bool {
    std::env::var("DEBUG_QBETA_LOG").is_ok()
        && log_p
        && lower_tail
        && (alpha - 1e-300_f64.ln()).abs() < 1e-12
        && (p - 1e8).abs() < 1e-6
        && (q - 10.0).abs() < 1e-12
}

pub fn qbeta(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> f64 {
    if alpha.is_nan() || p.is_nan() || q.is_nan() {
        return alpha + p + q;
    }
    if p < 0.0 || q < 0.0 {
        return ml_warn_return_nan();
    }

    if !lower_tail {
        if log_p && p > 1.0 && q > 1.0 {
            if is_debug_case(alpha, p, q, lower_tail, log_p) {
                eprintln!("DBG qbeta: using symmetry swap for upper tail");
            }
            let mut qb = [0.0_f64; 2];
            qbeta_raw(
                alpha,
                q,
                p,
                true,
                log_p,
                USE_LOG_X_CUTOFF,
                N_NEWTON_FREE,
                &mut qb,
            );
            let swapped = 1.0 - qb[0];
            if swapped.is_nan() {
                if is_debug_case(alpha, p, q, lower_tail, log_p) {
                    eprintln!("DBG qbeta: swap produced NaN, retrying original tail");
                }
            } else {
                return swapped;
            }
        }
        let mut qb = [0.0_f64; 2];
        qbeta_raw(
            alpha,
            p,
            q,
            false,
            log_p,
            USE_LOG_X_CUTOFF,
            N_NEWTON_FREE,
            &mut qb,
        );
        return qb[0];
    }

    let mut qb = [0.0_f64; 2];
    qbeta_raw(
        alpha,
        p,
        q,
        lower_tail,
        log_p,
        USE_LOG_X_CUTOFF,
        N_NEWTON_FREE,
        &mut qb,
    );
    qb[0]
}

fn qbeta_raw(
    alpha: f64,
    p: f64,
    q: f64,
    lower_tail: bool,
    log_p: bool,
    log_q_cut: f64,
    n_n: i32,
    qb: &mut [f64; 2],
) {
    let give_log_q = log_q_cut.is_infinite() && log_q_cut.is_sign_positive();
    let debug_case = is_debug_case(alpha, p, q, lower_tail, log_p)
        || is_debug_case_swapped(alpha, p, q, lower_tail, log_p)
        || is_debug_case_upper(alpha, p, q, lower_tail, log_p)
        || is_debug_case_upper_ab(alpha, p, q, lower_tail, log_p);
    if debug_case {
        eprintln!(
            "DBG qbeta_raw: alpha={:.17e} p={:.17e} q={:.17e} lower_tail={} log_p={}",
            alpha, p, q, lower_tail, log_p
        );
    }
    let mut use_log_x = give_log_q;
    let mut warned = false;
    let mut add_n_step = true;

    if alpha == r_dt_0(lower_tail, log_p) {
        if give_log_q {
            qb[0] = ML_NEGINF;
            qb[1] = 0.0;
        } else {
            qb[0] = 0.0;
            qb[1] = 1.0;
        }
        return;
    }
    if alpha == r_dt_1(lower_tail, log_p) {
        if give_log_q {
            qb[0] = 0.0;
            qb[1] = ML_NEGINF;
        } else {
            qb[0] = 1.0;
            qb[1] = 0.0;
        }
        return;
    }

    if (log_p && alpha > 0.0) || (!log_p && (alpha < 0.0 || alpha > 1.0)) {
        ml_warning(1, "qbeta");
        qb[0] = ML_NAN;
        qb[1] = ML_NAN;
        return;
    }

    if p == 0.0 || q == 0.0 || !r_finite(p) || !r_finite(q) {
        if p == 0.0 && q == 0.0 {
            if alpha < r_d_half(log_p) {
                if give_log_q {
                    qb[0] = ML_NEGINF;
                    qb[1] = 0.0;
                } else {
                    qb[0] = 0.0;
                    qb[1] = 1.0;
                }
                return;
            }
            if alpha > r_d_half(log_p) {
                if give_log_q {
                    qb[0] = 0.0;
                    qb[1] = ML_NEGINF;
                } else {
                    qb[0] = 1.0;
                    qb[1] = 0.0;
                }
                return;
            }
            if give_log_q {
                qb[0] = -M_LN2;
                qb[1] = -M_LN2;
            } else {
                qb[0] = 0.5;
                qb[1] = 0.5;
            }
            return;
        } else if p == 0.0 || p / q == 0.0 {
            if give_log_q {
                qb[0] = ML_NEGINF;
                qb[1] = 0.0;
            } else {
                qb[0] = 0.0;
                qb[1] = 1.0;
            }
            return;
        } else if q == 0.0 || q / p == 0.0 {
            if give_log_q {
                qb[0] = 0.0;
                qb[1] = ML_NEGINF;
            } else {
                qb[0] = 1.0;
                qb[1] = 0.0;
            }
            return;
        }
        if give_log_q {
            qb[0] = -M_LN2;
            qb[1] = -M_LN2;
        } else {
            qb[0] = 0.5;
            qb[1] = 0.5;
        }
        return;
    }

    let p_ = r_dt_qiv(alpha, lower_tail, log_p);
    let logbeta = lbeta(p, q);
    let mut swap_tail = p_ > 0.5;
    if debug_case {
        eprintln!(
            "DBG qbeta_raw init: p_={:.17e} swap_tail={}",
            p_, swap_tail
        );
    }

    let mut n_maybe_swaps = 0;
    let mut a;
    let mut la;
    let mut pp;
    let mut qq;
    let mut u;
    let mut xinbta;
    let mut tx;

    loop {
        let mut u_n = 1.0;
        if swap_tail {
            a = r_dt_civ(alpha, lower_tail, log_p);
            la = r_dt_clog(alpha, lower_tail, log_p);
            pp = q;
            qq = p;
        } else {
            a = p_;
            la = r_dt_log(alpha, lower_tail, log_p);
            pp = p;
            qq = q;
        }
        n_maybe_swaps += 1;

        let acu = fmax2(ACU_MIN, 10f64.powf(-13.0 - 2.5 / (pp * pp) - 0.5 / (a * a)));
        let u0 = (la + pp.ln() + logbeta) / pp;
        let mut rp = pp * (1.0 - qq) / (pp + 1.0);
        let log_eps_c = M_LN2 * (1.0 - f64::MANTISSA_DIGITS as f64);
        let t = 0.2;
        let u0_maybe = (M_LN2 * f64::MIN_EXP as f64) < u0 && u0 < -0.01;
        if u0_maybe
            && u0 < (t * log_eps_c - (pp * (1.0 - qq) * (2.0 - qq) / (2.0 * (pp + 2.0))).abs().ln()) / 2.0
        {
            rp *= u0.exp();
            if rp > -1.0 {
                u = u0 - (1.0 + rp).ln() / pp;
            } else {
                u = u0;
            }
            tx = u.exp();
            xinbta = tx;
            use_log_x = true;
            let mut nonfinite_w = false;
            let mut last_y = f64::NAN;
            let target_nonzero = if log_p { alpha > ML_NEGINF } else { alpha > 0.0 };
            let tiny_target = if log_p { alpha < -50.0 } else { alpha < 1e-50 };
            let reject_upper_underflow = !lower_tail && !swap_tail && target_nonzero && tiny_target;
            if debug_case {
                eprintln!(
                    "DBG qbeta_raw pre-newton (u0_maybe): swap_tail={} use_log_x={} u={:.17e} xinbta={:.17e} pp={:.17e} qq={:.17e} a={:.17e} la={:.17e}",
                    swap_tail, use_log_x, u, xinbta, pp, qq, a, la
                );
            }
            let (_, xinbta_new) = newton(
                u,
                xinbta,
                pp,
                qq,
                a,
                la,
                logbeta,
                log_p,
                use_log_x,
                acu,
                n_n,
                &mut warned,
                &mut nonfinite_w,
                &mut u_n,
                &mut add_n_step,
                &mut last_y,
                debug_case,
                reject_upper_underflow,
            );
            xinbta = xinbta_new;
            if nonfinite_w {
                if n_maybe_swaps <= 1 {
                    continue;
                }
                ml_warning(1, "qbeta");
                qb[0] = ML_NAN;
                qb[1] = ML_NAN;
                return;
            }
            post_newton_adjust(
                &mut add_n_step,
                log_p,
                use_log_x,
                last_y,
                a,
                la,
                pp,
                qq,
                &mut u_n,
                &mut tx,
                warned,
            );
            return_qb(
                give_log_q,
                swap_tail,
                use_log_x,
                add_n_step,
                u_n,
                xinbta,
                qb,
                log_p,
                pp,
                qq,
                a,
                la,
                logbeta,
                reject_upper_underflow,
            );
            return;
        }

        let r = (-2.0 * la).sqrt();
        let y = r - (CONST1 + CONST2 * r) / (1.0 + (CONST3 + CONST4 * r) * r);
        if pp > 1.0 && qq > 1.0 {
            let r = (y * y - 3.0) / 6.0;
            let s = 1.0 / (pp + pp - 1.0);
            let t = 1.0 / (qq + qq - 1.0);
            let h = 2.0 / (s + t);
            let w = y * (h + r).sqrt() / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
            if w > 300.0 {
                let t = w + w + qq.ln() - pp.ln();
                u = if t <= 18.0 { -(t.exp()).ln_1p() } else { -t - (-t).exp() };
                xinbta = u.exp();
            } else {
                let e = (w + w).exp();
                xinbta = pp / (pp + qq * e);
                u = -((qq / pp) * e).ln_1p();
            }
        } else {
            let r2 = qq + qq;
            let mut t = 1.0 / (3.0 * qq.sqrt());
            t = r2 * r_pow_di(1.0 + t * (-t + y), 3);
            let s = 4.0 * pp + r2 - 2.0;
            if t == 0.0 || (t < 0.0 && s >= t) {
                let l1ma = if swap_tail { r_dt_log(alpha, lower_tail, log_p) } else { r_dt_clog(alpha, lower_tail, log_p) };
                let xx = (l1ma + qq.ln() + logbeta) / qq;
                if xx <= 0.0 {
                    xinbta = -xx.exp_m1();
                    u = r_log1_exp(xx);
                } else {
                    let r_ = rp * u0.exp();
                    if r_ > -1.0 {
                        u = u0 - (1.0 + r_).ln() / pp;
                    } else {
                        u = u0;
                    }
                    xinbta = u.exp();
                }
            } else {
                let t2 = s / t;
                if t2 <= 1.0 {
                    u = u0;
                    xinbta = u.exp();
                } else {
                    xinbta = 1.0 - 2.0 / (t2 + 1.0);
                    u = (-2.0 / (t2 + 1.0)).ln_1p();
                }
            }
        }
        if (swap_tail && u >= -log_q_cut.exp())
            || (!swap_tail && u >= -(4.0 * log_q_cut).exp() && pp / qq < 1000.0)
        {
            swap_tail = !swap_tail;
            if swap_tail {
                a = r_dt_civ(alpha, lower_tail, log_p);
                la = r_dt_clog(alpha, lower_tail, log_p);
                pp = q;
                qq = p;
            } else {
                a = p_;
                la = r_dt_log(alpha, lower_tail, log_p);
                pp = p;
                qq = q;
            }
            u = r_log1_exp(u);
            xinbta = u.exp();
        }

        if !use_log_x {
            use_log_x = u < log_q_cut;
        }
        let bad_u = !r_finite(u);
        let bad_init = bad_u || xinbta > P_HI;
        tx = xinbta;
        let target_nonzero = if log_p { alpha > ML_NEGINF } else { alpha > 0.0 };
        let tiny_target = if log_p { alpha < -50.0 } else { alpha < 1e-50 };
        let reject_upper_underflow = !lower_tail && !swap_tail && target_nonzero && tiny_target;

        if bad_u || u < log_q_cut {
            let w = pbeta_raw(DBL_VERY_MIN, pp, qq, true, log_p);
            if w > if log_p { la } else { a } {
                if log_p || (w - a).abs() < (0.0 - a).abs() {
                    tx = DBL_VERY_MIN;
                    u_n = DBL_LOG_V_MIN;
                } else {
                    tx = 0.0;
                    u_n = ML_NEGINF;
                }
                use_log_x = log_p;
                add_n_step = false;
                return_qb(
                    give_log_q,
                    swap_tail,
                    use_log_x,
                    add_n_step,
                    u_n,
                    tx,
                    qb,
                    log_p,
                    pp,
                    qq,
                    a,
                    la,
                    logbeta,
                    reject_upper_underflow,
                );
                return;
            } else if u < DBL_LOG_V_MIN {
                u = DBL_LOG_V_MIN;
                xinbta = DBL_VERY_MIN;
            }
        }

        if bad_init && !(use_log_x && tx > 0.0) {
            if u == ML_NEGINF {
                u = M_LN2 * f64::MIN_EXP as f64;
                xinbta = f64::MIN_POSITIVE;
            } else {
                xinbta = if xinbta > 1.1 { 0.5 } else if xinbta < P_LO { u.exp() } else { P_HI };
                if bad_u {
                    u = xinbta.ln();
                }
            }
        }

        let mut nonfinite_w = false;
        let mut last_y = f64::NAN;
        if debug_case {
            eprintln!(
                "DBG qbeta_raw pre-newton: swap_tail={} use_log_x={} u={:.17e} xinbta={:.17e} pp={:.17e} qq={:.17e} a={:.17e} la={:.17e}",
                swap_tail, use_log_x, u, xinbta, pp, qq, a, la
            );
        }
        let (_, xinbta_new) = newton(
            u,
            xinbta,
            pp,
            qq,
            a,
            la,
            logbeta,
            log_p,
            use_log_x,
            acu,
            n_n,
            &mut warned,
            &mut nonfinite_w,
            &mut u_n,
            &mut add_n_step,
            &mut last_y,
            debug_case,
            reject_upper_underflow,
        );
        xinbta = xinbta_new;
        if nonfinite_w {
            if debug_case {
                eprintln!("DBG qbeta_raw: nonfinite_w after newton");
            }
            if use_log_x {
                if n_maybe_swaps <= 1 {
                    continue;
                }
            } else if n_maybe_swaps <= 2 {
                if !log_p && n_maybe_swaps == 2 {
                    use_log_x = true;
                }
                if !log_p || n_maybe_swaps <= 1 {
                    continue;
                }
            }
            ml_warning(1, "qbeta");
            qb[0] = ML_NAN;
            qb[1] = ML_NAN;
            return;
        }
        post_newton_adjust(
            &mut add_n_step,
            log_p,
            use_log_x,
            last_y,
            a,
            la,
            pp,
            qq,
            &mut u_n,
            &mut tx,
            warned,
        );
        return_qb(
            give_log_q,
            swap_tail,
            use_log_x,
            add_n_step,
            u_n,
            xinbta,
            qb,
            log_p,
            pp,
            qq,
            a,
            la,
            logbeta,
            reject_upper_underflow,
        );
        return;
    }
}

fn newton(
    mut u: f64,
    mut xinbta: f64,
    pp: f64,
    qq: f64,
    a: f64,
    la: f64,
    logbeta: f64,
    log_p: bool,
    use_log_x: bool,
    acu: f64,
    n_n: i32,
    warned: &mut bool,
    nonfinite_w: &mut bool,
    u_n: &mut f64,
    add_n_step: &mut bool,
    last_y: &mut f64,
    debug_case: bool,
    reject_upper_underflow: bool,
) -> (f64, f64) {
    let r = 1.0 - pp;
    let t = 1.0 - qq;
    let log_min = f64::MIN_POSITIVE.ln();
    let log_max = f64::MAX.ln();
    let mut wprev: f64 = 0.0;
    let mut wprev_sign: f64 = 0.0;
    let mut prev: f64 = 1.0;
    let mut adj: f64 = 1.0;
    *nonfinite_w = false;

    if use_log_x {
        let log_acu = acu.ln();
        let ln3 = 3.0_f64.ln();
        for i_pb in 0..1000 {
            let y = pbeta_raw(xinbta, pp, qq, true, true);
            *last_y = y;
            let (_dy, log_w, dy_sign) = {
                let log1m = r_log1_exp(u);
                let arg = y - u + logbeta + r * u + t * log1m;
                let dy = y - la;
                // log_w corresponds to log|w| for the Newton step on log(x), not log(pdf).
                if y == ML_NEGINF || dy == 0.0 {
                    (dy, f64::NEG_INFINITY, 0.0)
                } else {
                    (dy, dy.abs().ln() + arg, dy.signum())
                }
            };
            if debug_case && i_pb == 0 {
                let log1m = r_log1_exp(u);
                eprintln!(
                    "DBG newton log terms: y={y:.17e} u={u:.17e} logbeta={logbeta:.17e} r={r:.17e} t={t:.17e} log1m={log1m:.17e} y-u={:.17e} r*u={:.17e} t*log1m={:.17e}",
                    y - u,
                    r * u,
                    t * log1m
                );
            }
            if debug_case && i_pb < 8 {
                eprintln!(
                    "DBG newton log: i={i_pb} u={u:.17e} xinbta={xinbta:.17e} y={y:.17e} la={la:.17e} log_w={log_w:.17e} w_sign={dy_sign:.1} prev={prev:.17e}"
                );
            }
            if log_w.is_nan() || log_w == f64::INFINITY {
                if debug_case {
                    eprintln!(
                        "DBG newton log: nonfinite log_w at i={i_pb} u={u:.17e} xinbta={xinbta:.17e} log_w={log_w:.17e}"
                    );
                }
                ml_warning(1, "qbeta");
                *u_n = f64::NAN;
                *nonfinite_w = true;
                return (u, xinbta);
            }
            let overflow_log_w = log_w > log_max;
            if overflow_log_w && pp <= 1.0 {
                if debug_case {
                    eprintln!(
                        "DBG newton log: overflow log_w at i={i_pb} u={u:.17e} xinbta={xinbta:.17e} log_w={log_w:.17e}"
                    );
                }
                ml_warning(1, "qbeta");
                *u_n = f64::NAN;
                *nonfinite_w = true;
                return (u, xinbta);
            }
            let use_log_adj = overflow_log_w;
            let w = if use_log_adj { 0.0 } else { dy_sign * log_w.exp() };
            let w_sign = if use_log_adj {
                dy_sign
            } else if w == 0.0 {
                0.0
            } else {
                w.signum()
            };
            if !use_log_adj && !r_finite(w) {
                ml_warning(1, "qbeta");
                *u_n = f64::NAN;
                *nonfinite_w = true;
                return (u, xinbta);
            }
            if i_pb as i32 >= n_n && (w_sign == 0.0 || wprev_sign == 0.0 || w_sign != wprev_sign) {
                prev = fmax2(adj.abs(), FPU);
            }
            let mut u_next = u;
            if use_log_adj {
                let log_prev = prev.ln();
                let mut log_g = 0.0;
                if log_w > log_prev {
                    let k = ((log_w - log_prev) / ln3).ceil();
                    log_g = -ln3 * k;
                }
                for _ in 0..1000 {
                    let log_adj = log_w + log_g;
                    if log_adj < log_prev {
                        adj = if log_adj < log_min { 0.0 } else { dy_sign * log_adj.exp() };
                        u_next = u - adj;
                        if u_next <= 0.0 {
                            if prev <= acu || dy_sign == 0.0 || log_w <= log_acu {
                                *u_n = u_next;
                                return (u, xinbta);
                            }
                            break;
                        }
                        if reject_upper_underflow {
                            let upper_log = pbeta_raw(u_next.exp(), pp, qq, false, true);
                            if upper_log == ML_NEGINF {
                                if debug_case {
                                    eprintln!(
                                        "DBG newton log: reject upper-tail underflow at u={u_next:.17e}"
                                    );
                                }
                                log_g -= ln3;
                                continue;
                            }
                        }
                        break;
                    }
                    log_g -= ln3;
                }
            } else {
                let mut g = 1.0;
                for _ in 0..1000 {
                    adj = g * w;
                    if adj.abs() < prev {
                        u_next = u - adj;
                        if u_next <= 0.0 {
                            if prev <= acu || w.abs() <= acu {
                                *u_n = u_next;
                                return (u, xinbta);
                            }
                            break;
                        }
                        if reject_upper_underflow {
                            let upper_log = pbeta_raw(u_next.exp(), pp, qq, false, true);
                            if upper_log == ML_NEGINF {
                                if debug_case {
                                    eprintln!(
                                        "DBG newton log: reject upper-tail underflow at u={u_next:.17e}"
                                    );
                                }
                                g /= 3.0;
                                continue;
                            }
                        }
                        break;
                    }
                    g /= 3.0;
                }
            }
            if adj == 0.0 && w_sign != 0.0 {
                let tiny = f64::from_bits(1);
                let signed_tiny = w_sign.signum() * tiny;
                if signed_tiny != 0.0 {
                    adj = signed_tiny;
                    u_next = u - adj;
                }
            }
            let d = fmin2(adj.abs(), (u_next - u).abs());
            if d <= 4e-16 * (u_next + u).abs() {
                *u_n = u_next;
                return (u, xinbta);
            }
            u = u_next;
            *u_n = u;
            xinbta = u.exp();
            wprev_sign = w_sign;
        }
    } else {
        for i_pb in 0..1000 {
            let y = pbeta_raw(xinbta, pp, qq, true, log_p);
            *last_y = y;
            let w = if log_p {
                (y - la) * (y + logbeta + r * xinbta.ln() + t * (-xinbta).ln_1p()).exp()
            } else {
                (y - a) * (logbeta + r * xinbta.ln() + t * (-xinbta).ln_1p()).exp()
            };
            if debug_case && i_pb < 8 {
                eprintln!(
                    "DBG newton lin: i={i_pb} xinbta={xinbta:.17e} y={y:.17e} w={w:.17e} prev={prev:.17e}"
                );
            }
            if !r_finite(w) {
                ml_warning(1, "qbeta");
                *u_n = f64::NAN;
                *nonfinite_w = true;
                return (u, xinbta);
            }
            if i_pb as i32 >= n_n && w * wprev <= 0.0 {
                prev = fmax2(adj.abs(), FPU);
            }
            let mut g = 1.0;
            let mut tx = xinbta;
            for _ in 0..1000 {
                adj = g * w;
                if (i_pb as i32) < n_n || adj.abs() < prev {
                    tx = xinbta - adj;
                    if (0.0..=1.0).contains(&tx) {
                        if reject_upper_underflow {
                            let upper_log = pbeta_raw(tx, pp, qq, false, true);
                            if upper_log == ML_NEGINF {
                                if debug_case {
                                    eprintln!(
                                        "DBG newton lin: reject upper-tail underflow at x={tx:.17e}"
                                    );
                                }
                                g /= 3.0;
                                continue;
                            }
                        }
                        if prev <= acu || w.abs() <= acu {
                            *u_n = u;
                            return (u, tx);
                        }
                        if tx != 0.0 && tx != 1.0 {
                            break;
                        }
                    }
                }
                g /= 3.0;
            }
            if (tx - xinbta).abs() <= 4e-16 * (tx + xinbta) {
                *u_n = u;
                return (u, tx);
            }
            xinbta = tx;
            if tx == 0.0 {
                break;
            }
            wprev = w;
        }
    }

    *warned = true;
    *add_n_step = true;
    (u, xinbta)
}

fn return_qb(
    give_log_q: bool,
    swap_tail: bool,
    use_log_x: bool,
    add_n_step: bool,
    u_n: f64,
    tx: f64,
    qb: &mut [f64; 2],
    log_p: bool,
    pp: f64,
    qq: f64,
    a: f64,
    la: f64,
    logbeta: f64,
    reject_upper_underflow: bool,
) {
    if give_log_q {
        let r = r_log1_exp(u_n);
        if swap_tail {
            qb[0] = r;
            qb[1] = u_n;
        } else {
            qb[0] = u_n;
            qb[1] = r;
        }
        return;
    }

    let mut tx = tx;
    if use_log_x {
        if add_n_step {
            let xinbta = if u_n != 1.0 { u_n.exp() } else { tx };
            let y = pbeta_raw(xinbta, pp, qq, true, log_p);
            let (w, w_finite) = if log_p {
                let err = y - la;
                if y == ML_NEGINF || err == 0.0 {
                    (0.0, true)
                } else {
                    let w = err
                        * (y
                            + logbeta
                            + (1.0 - pp) * xinbta.ln()
                            + (1.0 - qq) * (-xinbta).ln_1p())
                            .exp();
                    (w, r_finite(w))
                }
            } else {
                let w = (y - a)
                    * (logbeta + (1.0 - pp) * xinbta.ln() + (1.0 - qq) * (-xinbta).ln_1p()).exp();
                (w, r_finite(w))
            };
            if w_finite {
                let candidate = xinbta - w;
                if reject_upper_underflow {
                    let upper_log = pbeta_raw(candidate, pp, qq, false, true);
                    if upper_log != ML_NEGINF {
                        tx = candidate;
                    } else {
                        tx = xinbta;
                    }
                } else {
                    tx = candidate;
                }
            } else {
                tx = xinbta;
            }
        } else if swap_tail {
            qb[0] = -u_n.exp_m1();
            qb[1] = u_n.exp();
            return;
        } else {
            qb[0] = u_n.exp();
            qb[1] = -u_n.exp_m1();
            return;
        }
    }

    if swap_tail {
        qb[0] = 1.0 - tx;
        qb[1] = tx;
    } else {
        qb[0] = tx;
        qb[1] = 1.0 - tx;
    }
}

fn post_newton_adjust(
    add_n_step: &mut bool,
    log_p: bool,
    use_log_x: bool,
    y: f64,
    a: f64,
    la: f64,
    pp: f64,
    qq: f64,
    u_n: &mut f64,
    tx: &mut f64,
    warned: bool,
) {
    let log_ = log_p || use_log_x;
    if (log_ && y == ML_NEGINF) || (!log_ && y == 0.0) {
        let w = pbeta_raw(DBL_VERY_MIN, pp, qq, true, log_);
        if log_ || (w - a).abs() <= (y - a).abs() {
            *tx = DBL_VERY_MIN;
            *u_n = DBL_LOG_V_MIN;
        }
        *add_n_step = false;
    } else if !warned {
        let diff = if log_ { (y - la).abs() } else { (y - a).abs() };
        if (log_ && diff > 3.0) || (!log_ && diff > 1e-4) {
            if !(log_ && y == ML_NEGINF && pbeta_raw(DBL_1__EPS, pp, qq, true, true) > la + 2.0) {
                ml_warning(2, "qbeta");
            }
        }
    }
}
