//! Kendall's S for short rows via pair-sign bitsets.
//!
//! For n ≤ [`MAX_N`] each prepared row stores, for every pair (i < j), one
//! bit for x_i > x_j and one for x_i < x_j. Then
//! S = |gt_x∧gt_y| + |lt_x∧lt_y| − |gt_x∧lt_y| − |lt_x∧gt_y|,
//! i.e. an O(n²/64) popcount loop per pair instead of a sort.

/// Largest sample size using bitsets (memory: n(n−1)/32 bytes per row).
pub const MAX_N: usize = 128;

/// Pair-sign bitsets of one row: `words` greater-than words followed by
/// `words` less-than words.
#[derive(Debug, Clone, Default)]
pub struct SignBits {
    words: usize,
    bits: Vec<u64>,
}

impl SignBits {
    /// Build from dense integer ranks.
    pub fn new(ranks: &[u32]) -> Self {
        let n = ranks.len();
        let pairs = n * n.saturating_sub(1) / 2;
        let words = pairs.div_ceil(64);
        let mut bits = vec![0u64; 2 * words];
        let mut p = 0usize;
        for i in 0..n {
            for j in i + 1..n {
                let (w, b) = (p / 64, 1u64 << (p % 64));
                if ranks[i] > ranks[j] {
                    bits[w] |= b;
                } else if ranks[i] < ranks[j] {
                    bits[words + w] |= b;
                }
                p += 1;
            }
        }
        Self { words, bits }
    }

    /// Kendall's S between two rows of equal length.
    pub fn s(&self, other: &SignBits) -> i64 {
        let w = self.words;
        let (gx, lx) = self.bits.split_at(w);
        let (gy, ly) = other.bits.split_at(w);
        let mut s = 0i64;
        for k in 0..w {
            let conc = (gx[k] & gy[k]).count_ones() + (lx[k] & ly[k]).count_ones();
            let disc = (gx[k] & ly[k]).count_ones() + (lx[k] & gy[k]).count_ones();
            s += i64::from(conc) - i64::from(disc);
        }
        s
    }
}
