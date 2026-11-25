//! Order selection over all (K, L) through 2-D prefix sums.

use super::coeffs::{Mat, PairData, loo_beta};

/// In-place 2-D prefix sum: P[K][L] = Σ_{k≤K, l≤L} X[k][l].
pub fn prefix2d(x: &mut [f64], k1: usize, l1: usize) {
    for k in 0..k1 {
        for l in 1..l1 {
            x[k * l1 + l] += x[k * l1 + l - 1];
        }
    }
    for k in 1..k1 {
        for l in 0..l1 {
            x[k * l1 + l] += x[(k - 1) * l1 + l];
        }
    }
}

/// First optimum in K-major, L-minor order; `better(a, b)` is true when
/// `a` strictly improves on `b`. Orders are then forced to be ≥ 1.
pub fn best(table: &[f64], k1: usize, l1: usize, better: impl Fn(f64, f64) -> bool) -> (usize, usize) {
    let mut arg = (0, 0);
    let mut v = f64::NAN;
    for k in 0..k1 {
        for l in 0..l1 {
            let t = table[k * l1 + l];
            if t.is_finite() && (v.is_nan() || better(t, v)) {
                v = t;
                arg = (k, l);
            }
        }
    }
    (arg.0.max(1).min(k1 - 1), arg.1.max(1).min(l1 - 1))
}

/// Normalised leave-one-out table T[K][L] = Σ_i R_i ⟨β⁽⁻ⁱ⁾, φ_i⟩_{K,L} / ‖β⁽⁻ⁱ⁾‖_{K,L}.
pub struct NormalizedWork {
    /// β⁽⁻ⁱ⁾ buffer.
    pub beta: Mat,
    /// Numerator prefix sums.
    pub num: Mat,
    /// Denominator prefix sums.
    pub den: Mat,
}

/// Fill `table` with the normalised criterion.
pub fn normalized_table(
    d: &PairData<'_>,
    s: &[f64],
    c2: f64,
    start: &[u32],
    list: &[u32],
    w: &mut NormalizedWork,
    table: &mut Mat,
) {
    let (k1, l1) = (d.k1, d.l1);
    table.clear();
    table.resize(k1 * l1, 0.0);
    for i in 0..d.r.len() {
        let inv = &list[start[i] as usize..start[i + 1] as usize];
        loo_beta(d, s, c2, i, inv, &mut w.beta);
        let (bx, by) = (&d.bx[i * k1..(i + 1) * k1], &d.by[i * l1..(i + 1) * l1]);
        w.num.clear();
        w.den.clear();
        for k in 0..k1 {
            for l in 0..l1 {
                let b = w.beta[k * l1 + l];
                w.num.push(b * bx[k] * by[l]);
                w.den.push(b * b);
            }
        }
        prefix2d(&mut w.num, k1, l1);
        prefix2d(&mut w.den, k1, l1);
        for ((t, nu), de) in table.iter_mut().zip(&w.num).zip(&w.den) {
            if *de > 0.0 {
                *t += d.r[i] * nu / de.sqrt();
            }
        }
    }
}
