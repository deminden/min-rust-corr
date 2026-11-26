//! Orthonormal shifted Legendre polynomials on [0, 1]:
//! b_k(u) = √(2k+1)·P_k(2u − 1), via the three-term recurrence
//! (k+1)P_{k+1}(x) = (2k+1)·x·P_k(x) − k·P_{k−1}(x).

/// Write b_0(u), …, b_{out.len()−1}(u) into `out`.
pub fn legendre_orthonormal(u: f64, out: &mut [f64]) {
    let x = 2.0 * u - 1.0;
    let (mut p_prev, mut p) = (0.0, 1.0);
    for (k, o) in out.iter_mut().enumerate() {
        *o = (2.0 * k as f64 + 1.0).sqrt() * p;
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0) * x * p - kf * p_prev) / (kf + 1.0);
        p_prev = p;
        p = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthonormal_by_quadrature() {
        // Midpoint rule on a fine grid.
        let m = 20_000;
        let mut acc = [[0.0; 5]; 5];
        let mut b = [0.0; 5];
        for i in 0..m {
            legendre_orthonormal((i as f64 + 0.5) / m as f64, &mut b);
            for k in 0..5 {
                for l in 0..5 {
                    acc[k][l] += b[k] * b[l] / m as f64;
                }
            }
        }
        for k in 0..5 {
            for l in 0..5 {
                let want = if k == l { 1.0 } else { 0.0 };
                assert!((acc[k][l] - want).abs() < 1e-6, "{k},{l}: {}", acc[k][l]);
            }
        }
    }
}
