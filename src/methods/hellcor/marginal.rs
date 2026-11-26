//! Per-(n, α) tables. Average ranks are multiples of ½ in [1, n], so every
//! marginal quantity is tabulated once per half-rank index h = 2r − 2:
//! u = r/(n+1), T = Q_Beta(α,α)(u), w = √ξ(T) with ξ the Beta(α, α)
//! density, and the Legendre basis b_k(u).

use super::basis::legendre_orthonormal;
use crate::stats::beta;

/// Shared marginal tables for one sample size.
#[derive(Debug, Clone)]
pub struct Tables {
    /// Sample size.
    pub n: usize,
    /// Transformed coordinate T per half-rank.
    pub t: Vec<f64>,
    /// √ξ(T) per half-rank.
    pub w: Vec<f64>,
    /// Basis values, `n_half × width`, row-major.
    pub basis: Vec<f64>,
    /// Number of basis functions stored per half-rank (max order + 1).
    pub width: usize,
}

impl Tables {
    /// Build tables for `n` observations, Beta(α, α) and orders ≤ `max_order`.
    pub fn new(n: usize, alpha: f64, max_order: usize) -> Self {
        let n_half = (2 * n).saturating_sub(1);
        let width = max_order + 1;
        let (mut t, mut w) = (vec![0.0; n_half], vec![0.0; n_half]);
        let mut basis = vec![0.0; n_half * width];
        for h in 0..n_half {
            let u = (1.0 + h as f64 / 2.0) / (n as f64 + 1.0);
            // Symmetry Q(1−u) = 1 − Q(u): solve the lower half only.
            let mirror = n_half - 1 - h;
            t[h] = if mirror < h { 1.0 - t[mirror] } else { beta::quantile(u, alpha, alpha) };
            w[h] = if mirror < h { w[mirror] } else { beta::pdf(t[h], alpha, alpha).sqrt() };
            legendre_orthonormal(u, &mut basis[h * width..(h + 1) * width]);
        }
        Self { n, t, w, basis, width }
    }

    /// Basis row of half-rank `h`.
    #[inline]
    pub fn b(&self, h: u32) -> &[f64] {
        let h = h as usize;
        &self.basis[h * self.width..(h + 1) * self.width]
    }
}

/// Half-rank indices (2r − 2) of finite values, using average ranks.
pub fn half_ranks(x: &[f64], idx: &mut Vec<u32>, out: &mut Vec<u32>) {
    use crate::prep::rank::{for_each_tie_group, sort_order};
    out.resize(x.len(), 0);
    sort_order(x, idx);
    for_each_tie_group(x, idx, |s, e| {
        // average rank (s+e+1)/2 → h = s + e − 1
        let h = (s + e - 1) as u32;
        for &k in &idx[s..e] {
            out[k as usize] = h;
        }
    });
}
