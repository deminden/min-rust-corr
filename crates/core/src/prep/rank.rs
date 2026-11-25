//! The single ranking implementation of the crate.
//!
//! All functions expect finite input (non-finite values are removed by the
//! caller according to the NaN policy) and take caller-owned scratch so
//! that no allocation happens per call once buffers have grown.

/// Fill `idx` with the permutation that sorts `x` ascending. Ties are
/// ordered by original index, so the result is fully deterministic.
pub fn sort_order(x: &[f64], idx: &mut Vec<u32>) {
    idx.clear();
    idx.extend(0..x.len() as u32);
    idx.sort_unstable_by(|&a, &b| x[a as usize].total_cmp(&x[b as usize]).then(a.cmp(&b)));
}

/// Call `f(start, end)` for every run `idx[start..end]` of equal values
/// (in the order produced by [`sort_order`]).
pub fn for_each_tie_group(x: &[f64], idx: &[u32], mut f: impl FnMut(usize, usize)) {
    let n = idx.len();
    let mut start = 0;
    while start < n {
        let v = x[idx[start] as usize];
        let mut end = start + 1;
        while end < n && x[idx[end] as usize] == v {
            end += 1;
        }
        f(start, end);
        start = end;
    }
}

/// Average ranks (1-based, ties share the mean of their positions), as in
/// R's `rank(ties.method = "average")`.
pub fn average_ranks(x: &[f64], idx: &mut Vec<u32>, out: &mut [f64]) {
    sort_order(x, idx);
    for_each_tie_group(x, idx, |s, e| {
        let r = (s + e + 1) as f64 / 2.0;
        for &k in &idx[s..e] {
            out[k as usize] = r;
        }
    });
}

/// Dense integer ranks starting at 0 (ties share a value). Returns the
/// number of distinct values.
pub fn dense_ranks(x: &[f64], idx: &mut Vec<u32>, out: &mut [u32]) -> u32 {
    sort_order(x, idx);
    let mut next = 0u32;
    for_each_tie_group(x, idx, |s, e| {
        for &k in &idx[s..e] {
            out[k as usize] = next;
        }
        next += 1;
    });
    next
}

/// Tie-group sums used by Kendall's tau-b and its variance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TieSums {
    /// Σ t(t−1)/2 — number of tied pairs.
    pub pairs: u64,
    /// Σ t(t−1)(t−2).
    pub t3: u64,
    /// Σ t(t−1)(2t+5).
    pub t5: u64,
}

impl TieSums {
    /// Accumulate a group of size `t`.
    pub fn add_group(&mut self, t: u64) {
        if t > 1 {
            self.pairs += t * (t - 1) / 2;
            self.t3 += t * (t - 1) * (t - 2);
            self.t5 += t * (t - 1) * (2 * t + 5);
        }
    }

    /// Tie sums of integer ranks given as group sizes.
    pub fn from_dense(ranks: &[u32], counts: &mut Vec<u32>) -> Self {
        counts.clear();
        for &r in ranks {
            let r = r as usize;
            if r >= counts.len() {
                counts.resize(r + 1, 0);
            }
            counts[r] += 1;
        }
        let mut s = TieSums::default();
        for &c in counts.iter() {
            s.add_group(c as u64);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_and_dense() {
        let x = [3.0, 1.0, 3.0, 2.0, 3.0];
        let mut idx = Vec::new();
        let mut r = [0.0; 5];
        average_ranks(&x, &mut idx, &mut r);
        assert_eq!(r, [4.0, 1.0, 4.0, 2.0, 4.0]);
        let mut d = [0u32; 5];
        assert_eq!(dense_ranks(&x, &mut idx, &mut d), 3);
        assert_eq!(d, [2, 0, 2, 1, 2]);
        let mut counts = Vec::new();
        let t = TieSums::from_dense(&d, &mut counts);
        assert_eq!(t, TieSums { pairs: 3, t3: 6, t5: 66 });
    }

    #[test]
    fn signed_zero_ties() {
        // -0.0 and 0.0 compare equal and must form one tie group.
        let x = [0.0, -0.0, 1.0];
        let mut idx = Vec::new();
        let mut r = [0.0; 3];
        average_ranks(&x, &mut idx, &mut r);
        assert_eq!(r, [1.5, 1.5, 3.0]);
    }
}
