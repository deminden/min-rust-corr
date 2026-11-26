//! Knight's O(n log n) algorithm for Kendall's S (Knight 1966, JASA 61).
//!
//! Observations are sorted by x (ties in x broken by y); then
//! S = n0 − n1 − n2 + n3 − 2·swaps, where n0 = n(n−1)/2, n1 and n2 are the
//! tied pairs in x and y, n3 the pairs tied in both, and `swaps` the number
//! of exchanges a merge sort needs to order the y sequence.

/// Scratch buffers for one evaluation.
#[derive(Debug, Default, Clone)]
pub struct KnightScratch {
    pub(crate) seq: Vec<u32>,
    pub(crate) buf: Vec<u32>,
}

/// Compute (S, joint tied pairs n3).
///
/// * `perm_x` — permutation sorting x ascending.
/// * `rank_x`, `rank_y` — dense integer ranks of x and y.
/// * `pairs_x`, `pairs_y` — tied pairs within x and within y.
pub fn kendall_s(
    perm_x: &[u32],
    rank_x: &[u32],
    rank_y: &[u32],
    pairs_x: u64,
    pairs_y: u64,
    s: &mut KnightScratch,
) -> i64 {
    let n = perm_x.len();
    s.seq.clear();
    s.seq.extend(perm_x.iter().map(|&k| rank_y[k as usize]));
    // Sort y within each run of tied x; count pairs tied in both.
    let mut n3: u64 = 0;
    let mut start = 0;
    while start < n {
        let rx = rank_x[perm_x[start] as usize];
        let mut end = start + 1;
        while end < n && rank_x[perm_x[end] as usize] == rx {
            end += 1;
        }
        if end - start > 1 {
            let seg = &mut s.seq[start..end];
            seg.sort_unstable();
            n3 += tied_pairs_sorted(seg);
        }
        start = end;
    }
    s.buf.resize(n, 0);
    let swaps = merge_count(&mut s.seq, &mut s.buf);
    let n0 = (n as u64) * (n as u64).saturating_sub(1) / 2;
    n0 as i64 - pairs_x as i64 - pairs_y as i64 + n3 as i64 - 2 * swaps as i64
}

fn tied_pairs_sorted(v: &[u32]) -> u64 {
    let mut total = 0u64;
    let mut run = 1u64;
    for w in v.windows(2) {
        if w[0] == w[1] {
            run += 1;
        } else {
            total += run * (run - 1) / 2;
            run = 1;
        }
    }
    total + run * (run - 1) / 2
}

/// Bottom-up merge sort of `v` counting exchanges (strict inversions).
fn merge_count(v: &mut [u32], buf: &mut [u32]) -> u64 {
    let n = v.len();
    let mut swaps = 0u64;
    let mut width = 1;
    while width < n {
        let mut lo = 0;
        while lo < n {
            let mid = (lo + width).min(n);
            let hi = (lo + 2 * width).min(n);
            let (mut i, mut j, mut k) = (lo, mid, lo);
            while i < mid && j < hi {
                if v[j] < v[i] {
                    swaps += (mid - i) as u64;
                    buf[k] = v[j];
                    j += 1;
                } else {
                    buf[k] = v[i];
                    i += 1;
                }
                k += 1;
            }
            buf[k..k + mid - i].copy_from_slice(&v[i..mid]);
            k += mid - i;
            buf[k..k + hi - j].copy_from_slice(&v[j..hi]);
            lo = hi;
        }
        v.copy_from_slice(&buf[..n]);
        width *= 2;
    }
    swaps
}

/// Brute-force O(n²) S, used as a test oracle.
pub fn kendall_s_brute(x: &[f64], y: &[f64]) -> i64 {
    let mut s = 0i64;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let a = (x[i] - x[j]).partial_cmp(&0.0).map_or(0, |o| o as i64);
            let b = (y[i] - y[j]).partial_cmp(&0.0).map_or(0, |o| o as i64);
            s += a * b;
        }
    }
    s
}
