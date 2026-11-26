//! Kendall's S via pair-sign bitsets.
//!
//! Each prepared row stores one bit per sample pair (i < j): `g` = 1 when
//! x_i > x_j. Rows with ties also store `t` = 1 when x_i ≠ x_j. A pair
//! untied in both rows is concordant exactly when the `g` bits agree, so
//! with T = t_x ∧ t_y (all ones for tie-free rows)
//!
//! S = |T| − 2·|T ∧ (g_x ⊕ g_y)|,
//!
//! one popcount per word for tie-free data (two when both rows have ties),
//! i.e. O(n²/64) per pair with no sorting. Used whenever the bitsets of all
//! rows fit in [`BUDGET_BYTES`]; otherwise Knight's algorithm is used.

use super::popcount::{and_count, masked_xor_count, masked2_xor_count, xor_count};

/// Memory allowed for the bitsets of all rows of one computation.
pub const BUDGET_BYTES: usize = 512 << 20;

/// Words per bitset for `n` observations.
pub fn words(n: usize) -> usize {
    (n * n.saturating_sub(1) / 2).div_ceil(64)
}

/// Whether bitsets for `rows` rows of `n` observations fit the budget
/// (counting the worst case where every row has ties, i.e. two bitsets).
pub fn fits(n: usize, rows: usize) -> bool {
    n >= 2 && rows.saturating_mul(words(n)).saturating_mul(16) <= BUDGET_BYTES
}

/// Pair-sign bitsets of one row.
#[derive(Debug, Clone, Default)]
pub struct SignBits {
    g: Vec<u64>,
    /// Untied-pair mask; `None` for a row without ties (all pairs untied).
    t: Option<Vec<u64>>,
    /// Number of untied pairs, |t|.
    untied: u64,
}

impl SignBits {
    /// Build from dense integer ranks; `tied_pairs` is the row's number of
    /// tied pairs.
    pub fn new(ranks: &[u32], tied_pairs: u64) -> Self {
        let n = ranks.len();
        let w = words(n);
        let mut g = vec![0u64; w];
        let mut t = (tied_pairs > 0).then(|| vec![0u64; w]);
        let mut p = 0usize;
        for i in 0..n {
            for j in i + 1..n {
                let bit = 1u64 << (p % 64);
                if ranks[i] > ranks[j] {
                    g[p / 64] |= bit;
                }
                if let Some(t) = t.as_mut() {
                    if ranks[i] != ranks[j] {
                        t[p / 64] |= bit;
                    }
                }
                p += 1;
            }
        }
        Self { g, t, untied: p as u64 - tied_pairs }
    }

    /// Kendall's S between two rows of equal length.
    pub fn s(&self, o: &SignBits) -> i64 {
        let (both, disc) = match (&self.t, &o.t) {
            (None, None) => (self.untied, xor_count(&self.g, &o.g)),
            (Some(m), None) => (self.untied, masked_xor_count(&self.g, &o.g, m)),
            (None, Some(m)) => (o.untied, masked_xor_count(&self.g, &o.g, m)),
            (Some(a), Some(b)) => (and_count(a, b), masked2_xor_count(&self.g, &o.g, a, b)),
        };
        both as i64 - 2 * disc as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget() {
        assert!(fits(77, 20_000));
        assert!(!fits(2000, 20_000));
        assert!(!fits(1, 10));
    }

    #[test]
    fn s_with_and_without_ties() {
        // x = (1, 2, 3, 4), y = (1, 3, 2, 4): S = 6 − 2·1 = 4.
        let (x, y) = (SignBits::new(&[0, 1, 2, 3], 0), SignBits::new(&[0, 2, 1, 3], 0));
        assert_eq!(x.s(&y), 4);
        // y with a tie: (1, 1, 2, 3) → concordant 5, tied 1, S = 5.
        let yt = SignBits::new(&[0, 0, 1, 2], 1);
        assert_eq!(x.s(&yt), 5);
        assert_eq!(yt.s(&x), 5);
        assert_eq!(yt.s(&yt), 5);
    }
}
