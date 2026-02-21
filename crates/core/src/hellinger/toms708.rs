use crate::hellinger::nmath::{
    M_LN_SQRT_2PI, M_LN2, ML_NAN, d1mach, i1mach, ml_warn_return_nan, ml_warning, r_d__0, r_d__1,
    r_finite,
};

const M_SQRT_PI: f64 = 1.7724538509055160273; // sqrt(pi)

fn min(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn r_log1_exp(x: f64) -> f64 {
    if x > -M_LN2 {
        (-rexpm1(x)).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

fn logspace_add(logx: f64, logy: f64) -> f64 {
    if logx == f64::NEG_INFINITY {
        return logy;
    }
    if logy == f64::NEG_INFINITY {
        return logx;
    }
    if logx >= logy {
        logx + (logy - logx).exp().ln_1p()
    } else {
        logy + (logx - logy).exp().ln_1p()
    }
}

fn ldexp(x: f64, exp: i32) -> f64 {
    x * 2.0_f64.powi(exp)
}

pub(crate) fn bratio(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    w: &mut f64,
    w1: &mut f64,
    ierr: &mut i32,
    log_p: bool,
) {
    let do_swap: bool;
    let mut n: i32 = 0;
    let mut ierr1: i32 = 0;
    let z: f64;
    let mut a0: f64;
    let mut b0: f64;
    let x0: f64;
    let y0: f64;
    let mut lambda: f64;

    let mut eps = 2.0 * d1mach(3);

    *w = r_d__0(log_p);
    *w1 = r_d__0(log_p);

    if x.is_nan() || y.is_nan() || a.is_nan() || b.is_nan() {
        *ierr = 9;
        return;
    }
    if a < 0.0 || b < 0.0 {
        *ierr = 1;
        return;
    }
    if a == 0.0 && b == 0.0 {
        *ierr = 2;
        return;
    }
    if x < 0.0 || x > 1.0 {
        *ierr = 3;
        return;
    }
    if y < 0.0 || y > 1.0 {
        *ierr = 4;
        return;
    }

    z = x + y - 0.5 - 0.5;
    if z.abs() > eps * 3.0 {
        *ierr = 5;
        return;
    }

    *ierr = 0;
    if x == 0.0 {
        if a == 0.0 {
            *ierr = 6;
            return;
        }
        *w = r_d__0(log_p);
        *w1 = r_d__1(log_p);
        return;
    }
    if y == 0.0 {
        if b == 0.0 {
            *ierr = 7;
            return;
        }
        *w = r_d__1(log_p);
        *w1 = r_d__0(log_p);
        return;
    }

    if a == 0.0 {
        *w = r_d__1(log_p);
        *w1 = r_d__0(log_p);
        return;
    }
    if b == 0.0 {
        *w = r_d__0(log_p);
        *w1 = r_d__1(log_p);
        return;
    }

    if eps < 1e-15 {
        eps = 1e-15;
    }
    let a_lt_b = a < b;
    if if a_lt_b { b } else { a } < eps * 0.001 {
        if log_p {
            if a_lt_b {
                *w = (1.0 - a / (a + b)).ln();
                *w1 = (a / (a + b)).ln();
            } else {
                *w = (b / (a + b)).ln();
                *w1 = (1.0 - b / (a + b)).ln();
            }
        } else {
            *w = b / (a + b);
            *w1 = a / (a + b);
        }
        return;
    }

    if min(a, b) <= 1.0 {
        do_swap = x > 0.5;
        if do_swap {
            a0 = b;
            x0 = y;
            b0 = a;
            y0 = x;
        } else {
            a0 = a;
            x0 = x;
            b0 = b;
            y0 = y;
        }

        if b0 < min(eps, eps * a0) {
            *w = fpser(a0, b0, x0, eps, log_p);
            *w1 = if log_p {
                r_log1_exp(*w)
            } else {
                0.5 - *w + 0.5
            };
            goto_end(do_swap, w, w1);
            return;
        }

        if a0 < min(eps, eps * b0) && b0 * x0 <= 1.0 {
            *w1 = apser(a0, b0, x0, eps);
            goto_end_from_w1(do_swap, w, w1, log_p);
            return;
        }

        let mut did_bup = false;
        if max(a0, b0) > 1.0 {
            if b0 <= 1.0 {
                *w = bpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }

            if x0 >= 0.29 {
                *w1 = bpser(b0, a0, y0, eps, log_p);
                *w = if log_p {
                    r_log1_exp(*w1)
                } else {
                    0.5 - *w1 + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }

            if x0 < 0.1 && (x0 * b0).powf(a0) <= 0.7 {
                *w = bpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }

            if b0 > 15.0 {
                *w1 = 0.0;
                bgrat(b0, a0, y0, x0, w1, 15.0 * eps, &mut ierr1, false);
                if *w1 == 0.0 || (*w1 > 0.0 && *w1 < f64::MIN_POSITIVE) {
                    if did_bup {
                        *w1 = bup(b0 - n as f64, a0, y0, x0, n, eps, true);
                    } else {
                        *w1 = f64::NEG_INFINITY;
                    }
                    bgrat(b0, a0, y0, x0, w1, 15.0 * eps, &mut ierr1, true);
                    if ierr1 != 0 {
                        *ierr = 10 + ierr1;
                    }
                    goto_end_from_w1_log(do_swap, w, w1, log_p);
                    return;
                }
                if ierr1 != 0 {
                    *ierr = 10 + ierr1;
                }
                if *w1 < 0.0 {
                    ml_warning(4, "bratio: bgrat() -> w1 < 0");
                }
                goto_end_from_w1(do_swap, w, w1, log_p);
                return;
            }
        } else {
            if a0 >= min(0.2, b0) {
                *w = bpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }

            if x0.powf(a0) <= 0.9 {
                *w = bpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }

            if x0 >= 0.3 {
                *w1 = bpser(b0, a0, y0, eps, log_p);
                *w = if log_p {
                    r_log1_exp(*w1)
                } else {
                    0.5 - *w1 + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }
        }

        n = 20;
        *w1 = bup(b0, a0, y0, x0, n, eps, false);
        did_bup = true;
        b0 += n as f64;
        bgrat(b0, a0, y0, x0, w1, 15.0 * eps, &mut ierr1, false);
        if *w1 == 0.0 || (*w1 > 0.0 && *w1 < f64::MIN_POSITIVE) {
            if did_bup {
                *w1 = bup(b0 - n as f64, a0, y0, x0, n, eps, true);
            } else {
                *w1 = f64::NEG_INFINITY;
            }
            bgrat(b0, a0, y0, x0, w1, 15.0 * eps, &mut ierr1, true);
            if ierr1 != 0 {
                *ierr = 10 + ierr1;
            }
            goto_end_from_w1_log(do_swap, w, w1, log_p);
            return;
        }
        if ierr1 != 0 {
            *ierr = 10 + ierr1;
        }
        if *w1 < 0.0 {
            ml_warning(4, "bratio: bgrat() -> w1 < 0");
        }
        goto_end_from_w1(do_swap, w, w1, log_p);
        return;
    }

    lambda = if r_finite(a + b) {
        if a > b {
            (a + b) * y - b
        } else {
            a - (a + b) * x
        }
    } else {
        a * y - b * x
    };
    do_swap = lambda < 0.0;
    if do_swap {
        lambda = -lambda;
        a0 = b;
        x0 = y;
        b0 = a;
        y0 = x;
    } else {
        a0 = a;
        x0 = x;
        b0 = b;
        y0 = y;
    }

    if b0 < 40.0 {
        if b0 * x0 <= 0.7 || (log_p && lambda > 650.0) {
            *w = bpser(a0, b0, x0, eps, log_p);
            *w1 = if log_p {
                r_log1_exp(*w)
            } else {
                0.5 - *w + 0.5
            };
            goto_end(do_swap, w, w1);
            return;
        } else {
            // L140
            let mut n_int = b0 as i32;
            b0 -= n_int as f64;
            if b0 == 0.0 {
                n_int -= 1;
                b0 = 1.0;
            }

            *w = bup(b0, a0, y0, x0, n_int, eps, false);

            if *w < f64::MIN_POSITIVE && log_p {
                b0 += n_int as f64;
                *w = bpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                goto_end(do_swap, w, w1);
                return;
            }

            if x0 <= 0.7 {
                *w += bpser(a0, b0, x0, eps, false);
                goto_end_from_w(do_swap, w, w1, log_p);
                return;
            }

            if a0 <= 15.0 {
                let n = 20;
                *w += bup(a0, b0, x0, y0, n, eps, false);
                a0 += n as f64;
            }
            bgrat(a0, b0, x0, y0, w, 15.0 * eps, &mut ierr1, false);
            if ierr1 != 0 {
                *ierr = 10 + ierr1;
            }
            goto_end_from_w(do_swap, w, w1, log_p);
            return;
        }
    } else if a0 > b0 {
        if b0 <= 100.0 || lambda > b0 * 0.03 {
            *w = bfrac(a0, b0, x0, y0, lambda, eps * 15.0, log_p);
            *w1 = if log_p {
                r_log1_exp(*w)
            } else {
                0.5 - *w + 0.5
            };
            goto_end(do_swap, w, w1);
            return;
        }
    } else if a0 <= 100.0 {
        *w = bfrac(a0, b0, x0, y0, lambda, eps * 15.0, log_p);
        *w1 = if log_p {
            r_log1_exp(*w)
        } else {
            0.5 - *w + 0.5
        };
        goto_end(do_swap, w, w1);
        return;
    } else if lambda > a0 * 0.03 {
        *w = bfrac(a0, b0, x0, y0, lambda, eps * 15.0, log_p);
        *w1 = if log_p {
            r_log1_exp(*w)
        } else {
            0.5 - *w + 0.5
        };
        goto_end(do_swap, w, w1);
        return;
    }

    *w = basym(a0, b0, lambda, eps * 100.0, log_p);
    *w1 = if log_p {
        r_log1_exp(*w)
    } else {
        0.5 - *w + 0.5
    };
    goto_end(do_swap, w, w1);
}

fn goto_end(do_swap: bool, w: &mut f64, w1: &mut f64) {
    if do_swap {
        let t = *w;
        *w = *w1;
        *w1 = t;
    }
}

fn goto_end_from_w1(do_swap: bool, w: &mut f64, w1: &mut f64, log_p: bool) {
    if log_p {
        *w = (-*w1).ln_1p();
        *w1 = (*w1).ln();
    } else {
        *w = 0.5 - *w1 + 0.5;
    }
    goto_end(do_swap, w, w1);
}

fn goto_end_from_w1_log(do_swap: bool, w: &mut f64, w1: &mut f64, log_p: bool) {
    if log_p {
        *w = r_log1_exp(*w1);
    } else {
        *w = -(*w1).exp_m1();
        *w1 = (*w1).exp();
    }
    goto_end(do_swap, w, w1);
}

fn goto_end_from_w(do_swap: bool, w: &mut f64, w1: &mut f64, log_p: bool) {
    if log_p {
        *w1 = (-*w).ln_1p();
        *w = (*w).ln();
    } else {
        *w1 = 0.5 - *w + 0.5;
    }
    goto_end(do_swap, w, w1);
}

fn fpser(a: f64, b: f64, x: f64, eps: f64, log_p: bool) -> f64 {
    let mut ans;
    if log_p {
        ans = a * x.ln();
    } else if a > eps * 0.001 {
        let t = a * x.ln();
        if t < exparg(1) {
            return 0.0;
        }
        ans = t.exp();
    } else {
        ans = 1.0;
    }

    if log_p {
        ans += b.ln() - a.ln();
    } else {
        ans *= b / a;
    }

    let tol = eps / a;
    let mut an = a + 1.0;
    let mut t = x;
    let mut s = t / an;
    loop {
        an += 1.0;
        t = x * t;
        let c = t / an;
        s += c;
        if c.abs() <= tol {
            break;
        }
    }

    if log_p {
        ans += (a * s).ln_1p();
    } else {
        ans *= a * s + 1.0;
    }
    ans
}

fn apser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    const G: f64 = 0.577215664901533;
    let bx = b * x;
    let mut t = x - bx;
    let c = if b * eps <= 0.02 {
        x.ln() + psi(b) + G + t
    } else {
        (bx).ln() + G + t
    };

    let tol = eps * 5.0 * c.abs();
    let mut j = 1.0;
    let mut s = 0.0;
    loop {
        j += 1.0;
        t *= x - bx / j;
        let aj = t / j;
        s += aj;
        if aj.abs() <= tol {
            break;
        }
    }

    -a * (c + s)
}

fn bpser(a: f64, b: f64, x: f64, eps: f64, log_p: bool) -> f64 {
    if x == 0.0 {
        return r_d__0(log_p);
    }

    let a0 = min(a, b);
    let mut ans;
    if a0 >= 1.0 {
        let z = a * x.ln() - betaln(a, b);
        ans = if log_p { z - a.ln() } else { z.exp() / a };
    } else {
        let b0 = max(a, b);
        if b0 < 8.0 {
            if b0 <= 1.0 {
                if log_p {
                    ans = a * x.ln();
                } else {
                    ans = x.powf(a);
                    if ans == 0.0 {
                        return ans;
                    }
                }
                let apb = a + b;
                let z = if apb > 1.0 {
                    (gam1(apb - 1.0) + 1.0) / apb
                } else {
                    gam1(apb) + 1.0
                };
                let c = (gam1(a) + 1.0) * (gam1(b) + 1.0) / z;
                if log_p {
                    ans += (c * (b / apb)).ln();
                } else {
                    ans *= c * (b / apb);
                }
            } else {
                let mut u = gamln1(a0);
                let mut b0m = b0;
                let m = (b0m - 1.0) as i32;
                if m >= 1 {
                    let mut c = 1.0;
                    for _ in 1..=m {
                        b0m -= 1.0;
                        c *= b0m / (a0 + b0m);
                    }
                    u += c.ln();
                }
                let z = a * x.ln() - u;
                b0m -= 1.0;
                let apb = a0 + b0m;
                let t = if apb > 1.0 {
                    (gam1(apb - 1.0) + 1.0) / apb
                } else {
                    gam1(apb) + 1.0
                };
                if log_p {
                    ans = z + (a0 / a).ln() + (gam1(b0m)).ln_1p() - t.ln();
                } else {
                    ans = z.exp() * (a0 / a) * (gam1(b0m) + 1.0) / t;
                }
            }
        } else {
            let u = gamln1(a0) + algdiv(a0, b0);
            let z = a * x.ln() - u;
            if log_p {
                ans = z + (a0 / a).ln();
            } else {
                ans = (a0 / a) * z.exp();
            }
        }
    }

    if ans == r_d__0(log_p) || (!log_p && a <= eps * 0.1) {
        return ans;
    }

    let tol = eps / a;
    let mut n = 0.0;
    let mut sum = 0.0;
    let mut c = 1.0;
    let mut w;
    loop {
        n += 1.0;
        c *= (0.5 - b / n + 0.5) * x;
        w = c / (a + n);
        sum += w;
        if n >= 1e7 || w.abs() <= tol {
            break;
        }
    }
    if w.abs() > tol {
        if (log_p && !(a * sum > -1.0 && (1.0 + a * sum).ln().abs() < eps * ans.abs()))
            || (!log_p && (a * sum + 1.0).abs() != 1.0)
        {
            ml_warning(4, "bpser: did not converge");
        }
    }

    if log_p {
        if a * sum > -1.0 {
            ans += (a * sum).ln_1p();
        } else {
            if ans > f64::NEG_INFINITY {
                ml_warning(4, "bpser underflow to -Inf");
            }
            ans = f64::NEG_INFINITY;
        }
    } else if a * sum > -1.0 {
        ans *= a * sum + 1.0;
    } else {
        ans = 0.0;
    }
    ans
}

fn bup(a: f64, b: f64, x: f64, y: f64, n: i32, eps: f64, give_log: bool) -> f64 {
    let apb = a + b;
    let ap1 = a + 1.0;
    let (mu, mut d) = if n > 1 && a >= 1.0 && apb >= ap1 * 1.1 {
        let mut mu = exparg(1).abs() as i32;
        let k = exparg(0) as i32;
        if mu > k {
            mu = k;
        }
        (mu, (-(mu as f64)).exp())
    } else {
        (0, 1.0)
    };

    let mut ret_val = if give_log {
        brcmp1(mu, a, b, x, y, true) - a.ln()
    } else {
        brcmp1(mu, a, b, x, y, false) / a
    };
    if n == 1 || (give_log && ret_val == f64::NEG_INFINITY) || (!give_log && ret_val == 0.0) {
        return ret_val;
    }

    let nm1 = n - 1;
    let mut w = d;

    let mut k = 0;
    if b > 1.0 {
        if y > 1e-4 {
            let r = (b - 1.0) * x / y - a;
            if r >= 1.0 {
                k = if r < nm1 as f64 { r as i32 } else { nm1 };
            }
        } else {
            k = nm1;
        }
        for i in 0..k {
            let l = i as f64;
            d *= (apb + l) / (ap1 + l) * x;
            w += d;
        }
    }

    for i in k..nm1 {
        let l = i as f64;
        d *= (apb + l) / (ap1 + l) * x;
        w += d;
        if d <= eps * w {
            break;
        }
    }

    if give_log {
        ret_val += w.ln();
    } else {
        ret_val *= w;
    }

    ret_val
}

fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64, eps: f64, log_p: bool) -> f64 {
    if !r_finite(lambda) {
        return ML_NAN;
    }
    let brc = brcomp(a, b, x, y, log_p);
    if brc.is_nan() {
        return ml_warn_return_nan();
    }
    if !log_p && brc == 0.0 {
        return 0.0;
    }

    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = 1.0 / a + 1.0;
    let yp1 = y + 1.0;
    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut bn = 1.0;
    let mut anp1 = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;
    let mut r0;

    const MAXIT: i32 = 1000;
    loop {
        n += 1.0;
        let mut w = n * x * (b - n);
        let rescale = !r_finite(w);
        if rescale {
            w = n * x * ldexp(b - n, -20);
        }
        let t = n / a;
        let e = a / s;
        let alpha = p * (p + c0) * e * e * (w * x);
        let e = (t + 1.0) / (c1 + t + t);
        let beta = w / s
            + if rescale {
                ldexp(n + e * (c + n * yp1), -20)
            } else {
                n + e * (c + n * yp1)
            };
        p = t + 1.0;
        s += 2.0;

        let t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        let t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;

        r0 = r;
        r = anp1 / bnp1;
        if (r - r0).abs() <= eps * r {
            break;
        }

        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;

        if n as i32 >= MAXIT {
            break;
        }
    }

    if log_p { brc + r.ln() } else { brc * r }
}

fn brcomp(a: f64, b: f64, x: f64, y: f64, log_p: bool) -> f64 {
    if x == 0.0 || y == 0.0 {
        return r_d__0(log_p);
    }

    let a0 = min(a, b);
    if a0 < 8.0 {
        let (lnx, lny) = if x <= 0.375 {
            (x.ln(), alnrel(-x))
        } else if y > 0.375 {
            (x.ln(), y.ln())
        } else {
            (alnrel(-y), y.ln())
        };
        let mut z = a * lnx + b * lny;
        if a0 >= 1.0 {
            z -= betaln(a, b);
            return if log_p { z } else { z.exp() };
        }
        let b0 = max(a, b);
        if b0 >= 8.0 {
            let u = gamln1(a0) + algdiv(a0, b0);
            return if log_p {
                a0.ln() + (z - u)
            } else {
                a0 * (z - u).exp()
            };
        }
        if b0 <= 1.0 {
            let e_z = if log_p { z } else { z.exp() };
            if !log_p && e_z == 0.0 {
                return 0.0;
            }
            let apb = a + b;
            if apb > 1.0 {
                z = (gam1(apb - 1.0) + 1.0) / apb;
            } else {
                z = gam1(apb) + 1.0;
            }
            let c = (gam1(a) + 1.0) * (gam1(b) + 1.0) / z;
            return if log_p {
                e_z + (a0 * c).ln() - (1.0 + a0 / b0).ln()
            } else {
                e_z * (a0 * c) / (a0 / b0 + 1.0)
            };
        }

        let mut u = gamln1(a0);
        let mut b0 = b0;
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 1..=n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            (gam1(apb - 1.0) + 1.0) / apb
        } else {
            gam1(apb) + 1.0
        };
        return if log_p {
            a0.ln() + z + (gam1(b0)).ln_1p() - t.ln()
        } else {
            a0 * z.exp() * (gam1(b0) + 1.0) / t
        };
    }

    let const__: f64 = 0.398942280401433;
    let apb = a + b;
    let lambda = if r_finite(apb) {
        if a <= b { a - apb * x } else { apb * y - b }
    } else {
        a * y - b * x
    };
    let (_h, x0, y0) = if a <= b {
        let h = a / b;
        (h, h / (h + 1.0), 1.0 / (h + 1.0))
    } else {
        let h = b / a;
        (h, 1.0 / (h + 1.0), h / (h + 1.0))
    };

    let mut e = -lambda / a;
    let u = if e.abs() > 0.6 {
        e - (x / x0).ln()
    } else {
        rlog1(e)
    };

    e = lambda / b;
    let v = if e.abs() <= 0.6 {
        rlog1(e)
    } else {
        e - (y / y0).ln()
    };

    let z = if log_p {
        -(a * u + b * v)
    } else {
        (-(a * u + b * v)).exp()
    };
    if log_p {
        -M_LN_SQRT_2PI + 0.5 * (b * x0).ln() + z - bcorr(a, b)
    } else {
        const__ * (b * x0).sqrt() * z * (-bcorr(a, b)).exp()
    }
}

fn brcmp1(mu: i32, a: f64, b: f64, x: f64, y: f64, give_log: bool) -> f64 {
    let a0 = min(a, b);
    if a0 < 8.0 {
        let (lnx, lny) = if x <= 0.375 {
            (x.ln(), alnrel(-x))
        } else if y > 0.375 {
            (x.ln(), y.ln())
        } else {
            (alnrel(-y), y.ln())
        };

        let mut z = a * lnx + b * lny;
        if a0 >= 1.0 {
            z -= betaln(a, b);
            return esum(mu, z, give_log);
        }

        let mut b0 = max(a, b);
        if b0 >= 8.0 {
            let u = gamln1(a0) + algdiv(a0, b0);
            return if give_log {
                a0.ln() + esum(mu, z - u, true)
            } else {
                a0 * esum(mu, z - u, false)
            };
        }

        if b0 <= 1.0 {
            let ans = esum(mu, z, give_log);
            if ans == if give_log { f64::NEG_INFINITY } else { 0.0 } {
                return ans;
            }
            let apb = a + b;
            if apb > 1.0 {
                z = (gam1(apb - 1.0) + 1.0) / apb;
            } else {
                z = gam1(apb) + 1.0;
            }
            let c = if give_log {
                (gam1(a)).ln_1p() + (gam1(b)).ln_1p() - z.ln()
            } else {
                (gam1(a) + 1.0) * (gam1(b) + 1.0) / z
            };
            return if give_log {
                ans + a0.ln() + c - (1.0 + a0 / b0).ln()
            } else {
                ans * (a0 * c) / (a0 / b0 + 1.0)
            };
        }

        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 1..=n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }

        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            (gam1(apb - 1.0) + 1.0) / apb
        } else {
            gam1(apb) + 1.0
        };
        return if give_log {
            a0.ln() + esum(mu, z, true) + (gam1(b0)).ln_1p() - t.ln()
        } else {
            a0 * esum(mu, z, false) * (gam1(b0) + 1.0) / t
        };
    }

    let const__: f64 = 0.398942280401433;
    let apb = a + b;
    let lambda = if r_finite(apb) {
        if a <= b { a - apb * x } else { apb * y - b }
    } else {
        a * y - b * x
    };
    let (_h, x0, y0) = if a > b {
        let h = b / a;
        (h, 1.0 / (h + 1.0), h / (h + 1.0))
    } else {
        let h = a / b;
        (h, h / (h + 1.0), 1.0 / (h + 1.0))
    };
    let lx0 = -(1.0 + b / a).ln();

    let e = -lambda / a;
    let u = if e.abs() > 0.6 {
        e - (x / x0).ln()
    } else {
        rlog1(e)
    };
    let e = lambda / b;
    let v = if e.abs() > 0.6 {
        e - (y / y0).ln()
    } else {
        rlog1(e)
    };

    let z = esum(mu, -(a * u + b * v), give_log);
    if give_log {
        const__.ln() + 0.5 * (b.ln() + lx0) + z - bcorr(a, b)
    } else {
        const__ * (b * x0).sqrt() * z * (-bcorr(a, b)).exp()
    }
}

fn bgrat(a: f64, b: f64, x: f64, y: f64, w: &mut f64, eps: f64, ierr: &mut i32, log_w: bool) {
    const N_TERMS: usize = 30;
    let mut c = [0.0_f64; N_TERMS];
    let mut d = [0.0_f64; N_TERMS];
    let bm1 = b - 0.5 - 0.5;
    let nu = a + bm1 * 0.5;
    let lnx = if y > 0.375 { x.ln() } else { alnrel(-y) };
    let z = -nu * lnx;

    if b * z == 0.0 {
        ml_warning(4, "bgrat: underflow");
        *ierr = 1;
        return;
    }

    let log_r = b.ln() + (gam1(b)).ln_1p() + b * z.ln() + nu * lnx;
    let log_u = log_r - (algdiv(b, a) + b * nu.ln());
    let u = log_u.exp();

    if log_u == f64::NEG_INFINITY {
        *ierr = 2;
        return;
    }

    let u_0 = u == 0.0;
    let l = if log_w {
        if *w == f64::NEG_INFINITY {
            0.0
        } else {
            (*w - log_u).exp()
        }
    } else if *w == 0.0 {
        0.0
    } else {
        ((*w).ln() - log_u).exp()
    };

    let q_r = grat_r(b, z, log_r, eps);
    let v = 0.25 / (nu * nu);
    let t2 = lnx * 0.25 * lnx;
    let mut j = q_r;
    let mut sum = j;
    let mut t = 1.0;
    let mut cn = 1.0;
    let mut n2 = 0.0;

    for n in 1..=N_TERMS {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        let nm1 = n - 1;
        c[nm1] = cn;
        let mut s = 0.0;
        if n > 1 {
            let mut coef = b - n as f64;
            for i in 1..=nm1 {
                s += coef * c[i - 1] * d[nm1 - i];
                coef += b;
            }
        }
        d[nm1] = bm1 * cn + s / n as f64;
        let dj = d[nm1] * j;
        sum += dj;
        if sum <= 0.0 {
            *ierr = 3;
            return;
        }
        if dj.abs() <= eps * (sum + l) {
            *ierr = 0;
            break;
        } else if n == N_TERMS {
            *ierr = 4;
            ml_warning(4, "bgrat did not converge");
        }
    }

    if log_w {
        *w = logspace_add(*w, log_u + sum.ln());
    } else {
        *w += if u_0 {
            (log_u + sum.ln()).exp()
        } else {
            u * sum
        };
    }
}

fn grat_r(a: f64, x: f64, log_r: f64, eps: f64) -> f64 {
    if a * x == 0.0 {
        if x <= a {
            return (-log_r).exp();
        }
        return 0.0;
    } else if a == 0.5 {
        if x < 0.25 {
            let p = erf__(x.sqrt());
            return (0.5 - p + 0.5) * (-log_r).exp();
        }
        let sx = x.sqrt();
        let q_r = erfc1(1, sx) / sx * M_SQRT_PI;
        return q_r;
    } else if x < 1.1 {
        let mut an = 3.0;
        let mut c = x;
        let mut sum = x / (a + 3.0);
        let tol = eps * 0.1 / (a + 1.0);
        let mut t;
        loop {
            an += 1.0;
            c *= -(x / an);
            t = c / (a + an);
            sum += t;
            if t.abs() <= tol {
                break;
            }
        }

        let j = a * x * ((sum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
        let z = a * x.ln();
        let h = gam1(a);
        let g = h + 1.0;

        if (x >= 0.25 && a < x / 2.59) || z > -0.13394 {
            let l = rexpm1(z);
            let q = ((l + 0.5 + 0.5) * j - l) * g - h;
            if q <= 0.0 {
                return 0.0;
            }
            return q * (-log_r).exp();
        }
        let p = z.exp() * g * (0.5 - j + 0.5);
        return (0.5 - p + 0.5) * (-log_r).exp();
    }

    let mut a2n_1 = 1.0;
    let mut a2n = 1.0;
    let mut b2n_1 = x;
    let mut b2n = x + (1.0 - a);
    let mut c = 1.0;
    let mut am0;
    let mut an0;

    loop {
        a2n_1 = x * a2n + c * a2n_1;
        b2n_1 = x * b2n + c * b2n_1;
        am0 = a2n_1 / b2n_1;
        c += 1.0;
        let c_a = c - a;
        a2n = a2n_1 + c_a * a2n;
        b2n = b2n_1 + c_a * b2n;
        an0 = a2n / b2n;
        if (an0 - am0).abs() < eps * an0 {
            break;
        }
    }
    an0
}

fn basym(a: f64, b: f64, lambda: f64, eps: f64, log_p: bool) -> f64 {
    const NUM_IT: usize = 20;
    const E0: f64 = 1.12837916709551;
    const E1: f64 = 0.353553390593274;
    const LN_E0: f64 = 0.120782237635245;

    let mut a0 = [0.0_f64; NUM_IT + 1];
    let mut b0 = [0.0_f64; NUM_IT + 1];
    let mut c = [0.0_f64; NUM_IT + 1];
    let mut d = [0.0_f64; NUM_IT + 1];

    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
    let t = if log_p {
        -f
    } else {
        let t = (-f).exp();
        if t == 0.0 {
            return 0.0;
        }
        t
    };

    let z0 = f.sqrt();
    let z = z0 / E1 * 0.5;
    let z2 = f + f;

    let (h, r0, r1, w0) = if a < b {
        let h = a / b;
        let r0 = 1.0 / (h + 1.0);
        let r1 = (b - a) / b;
        let w0 = 1.0 / (a * (h + 1.0)).sqrt();
        (h, r0, r1, w0)
    } else {
        let h = b / a;
        let r0 = 1.0 / (h + 1.0);
        let r1 = (b - a) / a;
        let w0 = 1.0 / (b * (h + 1.0)).sqrt();
        (h, r0, r1, w0)
    };

    a0[0] = r1 * 0.66666666666666663;
    c[0] = -0.5 * a0[0];
    d[0] = -c[0];
    let mut j0 = 0.5 / E0 * erfc1(1, z0);
    let mut j1 = E1;
    let mut sum = j0 + d[0] * w0 * j1;

    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;
    for n in (2..=NUM_IT).step_by(2) {
        hn *= h2;
        a0[n - 1] = r0 * 2.0 * (h * hn + 1.0) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0[np1 - 1] = r1 * 2.0 * s / (n as f64 + 3.0);

        for i in n..=np1 {
            let r = (i as f64 + 1.0) * -0.5;
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..=m - 1 {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0[j - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);

            let mut dsum = 0.0;
            for j in 1..=i - 1 {
                dsum += d[i - j - 1] * c[j - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }

        j0 = E1 * znm1 + (n as f64 - 1.0) * j0;
        j1 = E1 * zn + n as f64 * j1;
        znm1 = z2 * znm1;
        zn = z2 * zn;
        w *= w0;
        let t0 = d[n - 1] * w * j0;
        w *= w0;
        let t1 = d[np1 - 1] * w * j1;
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= eps * sum {
            break;
        }
    }

    if log_p {
        LN_E0 + t - bcorr(a, b) + sum.ln()
    } else {
        let u = (-bcorr(a, b)).exp();
        E0 * t * u * sum
    }
}

fn exparg(l: i32) -> f64 {
    let lnb = 0.69314718055995;
    let m = if l == 0 { i1mach(16) } else { i1mach(15) - 1 };
    (m as f64) * lnb * 0.99999
}

fn esum(mu: i32, x: f64, give_log: bool) -> f64 {
    if give_log {
        return x + mu as f64;
    }
    let w;
    if x > 0.0 {
        if mu > 0 {
            return (mu as f64).exp() * x.exp();
        }
        w = mu as f64 + x;
        if w < 0.0 {
            return (mu as f64).exp() * x.exp();
        }
    } else {
        if mu < 0 {
            return (mu as f64).exp() * x.exp();
        }
        w = mu as f64 + x;
        if w > 0.0 {
            return (mu as f64).exp() * x.exp();
        }
    }
    w.exp()
}

fn rexpm1(x: f64) -> f64 {
    let p1 = 9.14041914819518e-10;
    let p2 = 0.0238082361044469;
    let q1 = -0.499999999085958;
    let q2 = 0.107141568980644;
    let q3 = -0.0119041179760821;
    let q4 = 5.95130811860248e-4;

    if x.abs() <= 0.15 {
        return x * (((p2 * x + p1) * x + 1.0) / ((((q4 * x + q3) * x + q2) * x + q1) * x + 1.0));
    }
    let w = x.exp();
    if x > 0.0 {
        w * (0.5 - 1.0 / w + 0.5)
    } else {
        w - 0.5 - 0.5
    }
}

fn alnrel(a: f64) -> f64 {
    if a.abs() > 0.375 {
        return (1.0 + a).ln();
    }
    let p1 = -1.29418923021993;
    let p2 = 0.405303492862024;
    let p3 = -0.0178874546012214;
    let q1 = -1.62752256355323;
    let q2 = 0.747811014037616;
    let q3 = -0.0845104217945565;
    let t = a / (a + 2.0);
    let t2 = t * t;
    let w = (((p3 * t2 + p2) * t2 + p1) * t2 + 1.0) / (((q3 * t2 + q2) * t2 + q1) * t2 + 1.0);
    t * 2.0 * w
}

fn rlog1(x: f64) -> f64 {
    let a = 0.0566749439387324;
    let b = 0.0456512608815524;
    let p0 = 0.333333333333333;
    let p1 = -0.224696413112536;
    let p2 = 0.00620886815375787;
    let q1 = -1.27408923933623;
    let q2 = 0.354508718369557;

    let mut h;
    let w1;
    if x < -0.39 || x > 0.57 {
        let w = x + 0.5 + 0.5;
        return x - w.ln();
    }
    if x < -0.18 {
        h = x + 0.3;
        h /= 0.7;
        w1 = a - h * 0.3;
    } else if x > 0.18 {
        h = x * 0.75 - 0.25;
        w1 = b + h / 3.0;
    } else {
        h = x;
        w1 = 0.0;
    }

    let r = h / (h + 2.0);
    let t = r * r;
    let w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1.0);
    t * 2.0 * (1.0 / (1.0 - r) - r * w) + w1
}

fn erf__(x: f64) -> f64 {
    let c = 0.564189583547756;
    let a = [
        7.7105849500132e-5,
        -0.00133733772997339,
        0.0323076579225834,
        0.0479137145607681,
        0.128379167095513,
    ];
    let b = [0.00301048631703895, 0.0538971687740286, 0.375795757275549];
    let p = [
        -1.36864857382717e-7,
        0.564195517478974,
        7.21175825088309,
        43.1622272220567,
        152.98928504694,
        339.320816734344,
        451.918953711873,
        300.459261020162,
    ];
    let q = [
        1.0,
        12.7827273196294,
        77.0001529352295,
        277.585444743988,
        638.980264465631,
        931.35409485061,
        790.950925327898,
        300.459260956983,
    ];
    let r = [
        2.10144126479064,
        26.2370141675169,
        21.3688200555087,
        4.6580782871847,
        0.282094791773523,
    ];
    let s = [
        94.153775055546,
        187.11481179959,
        99.0191814623914,
        18.0124575948747,
    ];

    let ax = x.abs();
    if ax <= 0.5 {
        let t = x * x;
        let top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.0;
        let bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        return x * (top / bot);
    }

    if ax <= 4.0 {
        let top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax
            + p[6])
            * ax
            + p[7];
        let bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax
            + q[6])
            * ax
            + q[7];
        let r = 0.5 - (-x * x).exp() * top / bot + 0.5;
        return if x < 0.0 { -r } else { r };
    }

    if ax >= 5.8 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    let x2 = x * x;
    let t = 1.0 / x2;
    let top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
    let bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
    let t = (c - top / (x2 * bot)) / ax;
    let r = 0.5 - (-x2).exp() * t + 0.5;
    if x < 0.0 { -r } else { r }
}

fn erfc1(ind: i32, x: f64) -> f64 {
    let c = 0.564189583547756;
    let a = [
        7.7105849500132e-5,
        -0.00133733772997339,
        0.0323076579225834,
        0.0479137145607681,
        0.128379167095513,
    ];
    let b = [0.00301048631703895, 0.0538971687740286, 0.375795757275549];
    let p = [
        -1.36864857382717e-7,
        0.564195517478974,
        7.21175825088309,
        43.1622272220567,
        152.98928504694,
        339.320816734344,
        451.918953711873,
        300.459261020162,
    ];
    let q = [
        1.0,
        12.7827273196294,
        77.0001529352295,
        277.585444743988,
        638.980264465631,
        931.35409485061,
        790.950925327898,
        300.459260956983,
    ];
    let r = [
        2.10144126479064,
        26.2370141675169,
        21.3688200555087,
        4.6580782871847,
        0.282094791773523,
    ];
    let s = [
        94.153775055546,
        187.11481179959,
        99.0191814623914,
        18.0124575948747,
    ];

    let ax = x.abs();
    if ax <= 0.5 {
        let t = x * x;
        let top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.0;
        let bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        let mut ret_val = 0.5 - x * (top / bot) + 0.5;
        if ind != 0 {
            ret_val = t.exp() * ret_val;
        }
        return ret_val;
    }

    let mut ret_val;
    if ax <= 4.0 {
        let top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax
            + p[6])
            * ax
            + p[7];
        let bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax
            + q[6])
            * ax
            + q[7];
        ret_val = top / bot;
    } else {
        if x <= -5.6 {
            ret_val = 2.0;
            if ind != 0 {
                ret_val = (x * x).exp() * 2.0;
            }
            return ret_val;
        }
        if ind == 0 && (x > 100.0 || x * x > -exparg(1)) {
            return 0.0;
        }
        let t = 1.0 / (x * x);
        let top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
        let bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
        ret_val = (c - t * top / bot) / ax;
    }

    if ind != 0 {
        if x < 0.0 {
            ret_val = (x * x).exp() * 2.0 - ret_val;
        }
    } else {
        let w = x * x;
        let t = w;
        let e = w - t;
        ret_val = (0.5 - e + 0.5) * (-t).exp() * ret_val;
        if x < 0.0 {
            ret_val = 2.0 - ret_val;
        }
    }
    ret_val
}

fn gam1(a: f64) -> f64 {
    let d = a - 0.5;
    let t = if d > 0.0 { d - 0.5 } else { a };

    if t < 0.0 {
        let r = [
            -0.422784335098468,
            -0.771330383816272,
            -0.244757765222226,
            0.118378989872749,
            9.30357293360349e-4,
            -0.0118290993445146,
            0.00223047661158249,
            2.66505979058923e-4,
            -1.32674909766242e-4,
        ];
        let s1 = 0.273076135303957;
        let s2 = 0.0559398236957378;

        let top = (((((((r[8] * t + r[7]) * t + r[6]) * t + r[5]) * t + r[4]) * t + r[3]) * t
            + r[2])
            * t
            + r[1])
            * t
            + r[0];
        let bot = (s2 * t + s1) * t + 1.0;
        let w = top / bot;
        if d > 0.0 {
            t * w / a
        } else {
            a * (w + 0.5 + 0.5)
        }
    } else if t == 0.0 {
        0.0
    } else {
        let p = [
            0.577215664901533,
            -0.409078193005776,
            -0.230975380857675,
            0.0597275330452234,
            0.0076696818164949,
            -0.00514889771323592,
            5.89597428611429e-4,
        ];
        let q = [
            1.0,
            0.427569613095214,
            0.158451672430138,
            0.0261132021441447,
            0.00423244297896961,
        ];

        let top = (((((p[6] * t + p[5]) * t + p[4]) * t + p[3]) * t + p[2]) * t + p[1]) * t + p[0];
        let bot = (((q[4] * t + q[3]) * t + q[2]) * t + q[1]) * t + 1.0;
        let w = top / bot;
        if d > 0.0 {
            t / a * (w - 0.5 - 0.5)
        } else {
            a * w
        }
    }
}

fn gamln1(a: f64) -> f64 {
    if a < 0.6 {
        let p0 = 0.577215664901533;
        let p1 = 0.844203922187225;
        let p2 = -0.168860593646662;
        let p3 = -0.780427615533591;
        let p4 = -0.402055799310489;
        let p5 = -0.0673562214325671;
        let p6 = -0.00271935708322958;
        let q1 = 2.88743195473681;
        let q2 = 3.12755088914843;
        let q3 = 1.56875193295039;
        let q4 = 0.361951990101499;
        let q5 = 0.0325038868253937;
        let q6 = 6.67465618796164e-4;
        let w = ((((((p6 * a + p5) * a + p4) * a + p3) * a + p2) * a + p1) * a + p0)
            / ((((((q6 * a + q5) * a + q4) * a + q3) * a + q2) * a + q1) * a + 1.0);
        return -a * w;
    }

    let r0 = 0.422784335098467;
    let r1 = 0.848044614534529;
    let r2 = 0.565221050691933;
    let r3 = 0.156513060486551;
    let r4 = 0.017050248402265;
    let r5 = 4.97958207639485e-4;
    let s1 = 1.24313399877507;
    let s2 = 0.548042109832463;
    let s3 = 0.10155218743983;
    let s4 = 0.00713309612391;
    let s5 = 1.16165475989616e-4;
    let x = a - 0.5 - 0.5;
    let w = (((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0)
        / (((((s5 * x + s4) * x + s3) * x + s2) * x + s1) * x + 1.0);
    x * w
}

fn psi(x: f64) -> f64 {
    let piov4 = 0.785398163397448;
    let dx0 = 1.461632144968362341262659542325721325;

    let p1 = [
        0.0089538502298197,
        4.77762828042627,
        142.441585084029,
        1186.45200713425,
        3633.51846806499,
        4138.10161269013,
        1305.60269827897,
    ];
    let q1 = [
        44.8452573429826,
        520.752771467162,
        2210.0079924783,
        3641.27349079381,
        1908.310765963,
        6.91091682714533e-6,
    ];

    let p2 = [
        -2.12940445131011,
        -7.01677227766759,
        -4.48616543918019,
        -0.648157123766197,
    ];
    let q2 = [
        32.2703493791143,
        89.2920700481861,
        54.6117738103215,
        7.77788548522962,
    ];

    let mut x = x;
    let mut aug = 0.0;
    let mut xmax1 = i32::MAX as f64;
    let d2 = 0.5 / d1mach(3);
    if xmax1 > d2 {
        xmax1 = d2;
    }
    let xsmall = 1e-9;

    if x < 0.5 {
        if x.abs() <= xsmall {
            if x == 0.0 {
                return 0.0;
            }
            aug = -1.0 / x;
        } else {
            let mut w = -x;
            let mut sgn = piov4;
            if w <= 0.0 {
                w = -w;
                sgn = -sgn;
            }
            if w >= xmax1 {
                return 0.0;
            }
            let mut nq = w as i32;
            w -= nq as f64;
            nq = (w * 4.0) as i32;
            w = (w - nq as f64 * 0.25) * 4.0;
            let mut n = nq / 2;
            if n + n != nq {
                w = 1.0 - w;
            }
            let z = piov4 * w;
            let mut m = n / 2;
            if m + m != n {
                sgn = -sgn;
            }
            n = (nq + 1) / 2;
            m = n / 2;
            m += m;
            if m == n {
                if z == 0.0 {
                    return 0.0;
                }
                aug = sgn * (z.cos() / z.sin() * 4.0);
            } else {
                aug = sgn * (z.sin() / z.cos() * 4.0);
            }
        }
        x = 1.0 - x;
    }

    if x <= 3.0 {
        let mut den = x;
        let mut upper = p1[0] * x;
        for i in 1..=5 {
            den = (den + q1[i - 1]) * x;
            upper = (upper + p1[i]) * x;
        }
        den = (upper + p1[6]) / (den + q1[5]);
        let xmx0 = x - dx0;
        return den * xmx0 + aug;
    }

    if x < xmax1 {
        let w = 1.0 / (x * x);
        let mut den = w;
        let mut upper = p2[0] * w;
        for i in 1..=3 {
            den = (den + q2[i - 1]) * w;
            upper = (upper + p2[i]) * w;
        }
        aug = upper / (den + q2[3]) - 0.5 / x + aug;
    }
    aug + x.ln()
}

fn betaln(a0: f64, b0: f64) -> f64 {
    let mut a = min(a0, b0);
    let mut b = max(a0, b0);

    if a < 8.0 {
        if a < 1.0 {
            if b < 8.0 {
                return gamln(a) + (gamln(b) - gamln(a + b));
            }
            return gamln(a) + algdiv(a, b);
        }

        let mut w;
        if a < 2.0 {
            if b <= 2.0 {
                return gamln(a) + gamln(b) - gsumln(a, b);
            }
            if b < 8.0 {
                w = 0.0;
            } else {
                return gamln(a) + algdiv(a, b);
            }
        } else if b <= 1e3 {
            let n = (a - 1.0) as i32;
            w = 1.0;
            for _ in 1..=n {
                a -= 1.0;
                let h = a / b;
                w *= h / (h + 1.0);
            }
            w = w.ln();
            if b >= 8.0 {
                return w + gamln(a) + algdiv(a, b);
            }
        } else {
            let n = (a - 1.0) as i32;
            w = 1.0;
            for _ in 1..=n {
                a -= 1.0;
                w *= a / (a / b + 1.0);
            }
            return w.ln() - (n as f64) * b.ln() + (gamln(a) + algdiv(a, b));
        }

        let n = (b - 1.0) as i32;
        let mut z = 1.0;
        for _ in 1..=n {
            b -= 1.0;
            z *= b / (a + b);
        }
        return w + z.ln() + (gamln(a) + (gamln(b) - gsumln(a, b)));
    }

    let e = 0.918938533204673;
    let w = bcorr(a, b);
    let h = a / b;
    let u = -(a - 0.5) * (h / (h + 1.0)).ln();
    let v = b * alnrel(h);
    if u > v {
        b.ln() * -0.5 + e + w - v - u
    } else {
        b.ln() * -0.5 + e + w - u - v
    }
}

fn gsumln(a: f64, b: f64) -> f64 {
    let x = a + b - 2.0;
    if x <= 0.25 {
        return gamln1(x + 1.0);
    }
    if x <= 1.25 {
        return gamln1(x) + alnrel(x);
    }
    gamln1(x - 1.0) + (x * (x + 1.0)).ln()
}

fn bcorr(a0: f64, b0: f64) -> f64 {
    let a = min(a0, b0);
    let b = max(a0, b0);

    let c0 = 0.0833333333333333;
    let c1 = -0.00277777777760991;
    let c2 = 7.9365066682539e-4;
    let c3 = -5.9520293135187e-4;
    let c4 = 8.37308034031215e-4;
    let c5 = -0.00165322962780713;

    let h = a / b;
    let c = h / (h + 1.0);
    let x = 1.0 / (h + 1.0);
    let x2 = x * x;

    let s3 = x + x2 + 1.0;
    let s5 = x + x2 * s3 + 1.0;
    let s7 = x + x2 * s5 + 1.0;
    let s9 = x + x2 * s7 + 1.0;
    let s11 = x + x2 * s9 + 1.0;

    let mut t = 1.0 / (b * b);
    let mut w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0;
    w *= c / b;

    t = 1.0 / (a * a);
    (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a + w
}

fn algdiv(a: f64, b: f64) -> f64 {
    let c0 = 0.0833333333333333;
    let c1 = -0.00277777777760991;
    let c2 = 7.9365066682539e-4;
    let c3 = -5.9520293135187e-4;
    let c4 = 8.37308034031215e-4;
    let c5 = -0.00165322962780713;

    let (_h, c, x, d) = if a > b {
        let h = b / a;
        let c = 1.0 / (h + 1.0);
        let x = h / (h + 1.0);
        let d = a + (b - 0.5);
        (h, c, x, d)
    } else {
        let h = a / b;
        let c = h / (h + 1.0);
        let x = 1.0 / (h + 1.0);
        let d = b + (a - 0.5);
        (h, c, x, d)
    };

    let x2 = x * x;
    let s3 = x + x2 + 1.0;
    let s5 = x + x2 * s3 + 1.0;
    let s7 = x + x2 * s5 + 1.0;
    let s9 = x + x2 * s7 + 1.0;
    let s11 = x + x2 * s9 + 1.0;

    let t = 1.0 / (b * b);
    let mut w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0;
    w *= c / b;

    let u = d * alnrel(a / b);
    let v = a * (b.ln() - 1.0);
    if u > v { w - v - u } else { w - u - v }
}

fn gamln(a: f64) -> f64 {
    let d = 0.418938533204673;
    let c0 = 0.0833333333333333;
    let c1 = -0.00277777777760991;
    let c2 = 7.9365066682539e-4;
    let c3 = -5.9520293135187e-4;
    let c4 = 8.37308034031215e-4;
    let c5 = -0.00165322962780713;

    if a <= 0.8 {
        return gamln1(a) - a.ln();
    }
    if a <= 2.25 {
        return gamln1(a - 0.5 - 0.5);
    }
    if a < 10.0 {
        let n = (a - 1.25) as i32;
        let mut t = a;
        let mut w = 1.0;
        for _ in 1..=n {
            t -= 1.0;
            w *= t;
        }
        return gamln1(t - 1.0) + w.ln();
    }
    let t = 1.0 / (a * a);
    let w = (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a;
    d + w + (a - 0.5) * (a.ln() - 1.0)
}
