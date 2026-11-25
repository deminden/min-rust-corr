//! Non-finite census and pairwise-complete gathering.

use ndarray::{ArrayView1, ArrayView2};

/// Number of non-finite values in each row.
pub fn nonfinite_counts(data: ArrayView2<'_, f64>) -> Vec<u32> {
    data.rows().into_iter().map(|r| r.iter().filter(|v| !v.is_finite()).count() as u32).collect()
}

/// Position (row, col) of the first non-finite value, if any.
pub fn first_nonfinite(data: ArrayView2<'_, f64>) -> Option<(usize, usize)> {
    data.rows()
        .into_iter()
        .enumerate()
        .find_map(|(i, r)| r.iter().position(|v| !v.is_finite()).map(|j| (i, j)))
}

/// Copy the finite values of `row` into `out` (cleared first).
pub fn gather_finite(row: ArrayView1<'_, f64>, out: &mut Vec<f64>) {
    out.clear();
    out.extend(row.iter().copied().filter(|v| v.is_finite()));
}

/// Copy the observations finite in both `x` and `y` into `gx`, `gy`.
pub fn gather_common(x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>, gx: &mut Vec<f64>, gy: &mut Vec<f64>) {
    gx.clear();
    gy.clear();
    for (&a, &b) in x.iter().zip(y.iter()) {
        if a.is_finite() && b.is_finite() {
            gx.push(a);
            gy.push(b);
        }
    }
}

/// Copy `row` into `out` as a contiguous slice (no-op copy when the caller
/// can use `as_slice` directly).
pub fn to_contiguous<'a>(row: ArrayView1<'a, f64>, out: &'a mut Vec<f64>) -> &'a [f64] {
    match row.to_slice() {
        Some(s) => s,
        None => {
            out.clear();
            out.extend(row.iter().copied());
            out
        }
    }
}
