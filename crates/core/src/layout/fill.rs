//! Index math for packed triangles and symmetric mirroring.

use ndarray::Array2;

/// Length of a packed upper triangle with diagonal: n(n+1)/2.
pub fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Offset of entry (i, j), i ≤ j, in a row-major packed upper triangle of
/// dimension `n`.
pub fn packed_index(n: usize, i: usize, j: usize) -> usize {
    row_offset(n, i) + (j - i)
}

/// Offset of the first stored entry (i, i) of row `i`.
pub fn row_offset(n: usize, i: usize) -> usize {
    i * n - i * i.saturating_sub(1) / 2
}

/// Copy the upper triangle of a square matrix onto its lower triangle,
/// working in cache-sized tiles.
pub fn mirror_upper(m: &mut Array2<f64>) {
    let n = m.nrows();
    const T: usize = 64;
    for bi in (0..n).step_by(T) {
        for bj in (bi..n).step_by(T) {
            for i in bi..(bi + T).min(n) {
                let j0 = bj.max(i + 1);
                for j in j0..(bj + T).min(n) {
                    m[[j, i]] = m[[i, j]];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_offsets() {
        let n = 5;
        let mut k = 0;
        for i in 0..n {
            for j in i..n {
                assert_eq!(packed_index(n, i, j), k);
                k += 1;
            }
        }
        assert_eq!(k, packed_len(n));
    }
}
