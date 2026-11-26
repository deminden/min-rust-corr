//! Row-block partitioning shared by the in-memory and streaming paths.

use std::ops::Range;

/// Target size in bytes of one streamed block (correlations + p-values).
pub const TARGET_BLOCK_BYTES: usize = 48 << 20;

/// Rows per streamed block: user override or about [`TARGET_BLOCK_BYTES`].
pub fn block_rows(n_rows: usize, n_cols: usize, with_p: bool, user: Option<usize>) -> usize {
    if let Some(b) = user {
        return b.max(1);
    }
    let per_row = 8 * n_cols.max(1) * if with_p { 2 } else { 1 };
    (TARGET_BLOCK_BYTES / per_row).clamp(1, n_rows.max(1))
}

/// Split `0..n` into consecutive ranges of at most `size` rows.
pub fn ranges(n: usize, size: usize) -> impl Iterator<Item = Range<usize>> {
    let size = size.max(1);
    (0..n).step_by(size).map(move |s| s..(s + size).min(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        let v: Vec<_> = ranges(10, 4).collect();
        assert_eq!(v, vec![0..4, 4..8, 8..10]);
        assert_eq!(ranges(0, 4).count(), 0);
        assert_eq!(block_rows(100, 10, false, Some(0)), 1);
    }
}
