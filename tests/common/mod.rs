//! Shared helpers for integration tests: fixture loading and comparison.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ndarray::Array2;
use std::path::PathBuf;

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn exists(name: &str) -> bool {
    fixture_dir().join(format!("{name}.tsv")).exists()
}

/// Load `<name>.tsv` (header row, first column = row id) into an Array2.
pub fn load(name: &str) -> Array2<f64> {
    let path = fixture_dir().join(format!("{name}.tsv"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut rows = Vec::new();
    let mut ncol = 0;
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let vals: Vec<f64> = line.split('\t').skip(1).map(|v| v.parse().unwrap()).collect();
        ncol = vals.len();
        rows.extend(vals);
    }
    let nrow = rows.len() / ncol.max(1);
    Array2::from_shape_vec((nrow, ncol), rows).unwrap()
}

pub const DATASETS: &[&str] = &[
    "normal_30x40",
    "normal_8x200",
    "lognormal_20x77",
    "ties_15x30",
    "outliers_15x50",
    "constant_rows_6x20",
    "na_12x40",
    "tiny_5x3",
    "tiny_5x4",
    "monotone_pairs",
];

/// Max |a − b| over entries finite in both, asserting NaN patterns match
/// (off-diagonal) and returning the maximum difference.
pub fn max_abs_diff(got: &Array2<f64>, want: &Array2<f64>, skip_diag: bool, ctx: &str) -> f64 {
    assert_eq!(got.dim(), want.dim(), "{ctx}: shape");
    let mut m: f64 = 0.0;
    for ((i, j), w) in want.indexed_iter() {
        if skip_diag && i == j {
            continue;
        }
        let g = got[[i, j]];
        assert_eq!(g.is_nan(), w.is_nan(), "{ctx}: NaN mismatch at ({i},{j}): got {g} want {w}");
        if w.is_finite() {
            m = m.max((g - w).abs());
        }
    }
    m
}

/// Max relative difference (|a−b| / max(|b|, tiny)).
pub fn max_rel_diff(got: &Array2<f64>, want: &Array2<f64>, skip_diag: bool, ctx: &str) -> f64 {
    assert_eq!(got.dim(), want.dim(), "{ctx}: shape");
    let mut m: f64 = 0.0;
    for ((i, j), w) in want.indexed_iter() {
        if skip_diag && i == j {
            continue;
        }
        let g = got[[i, j]];
        assert_eq!(g.is_nan(), w.is_nan(), "{ctx}: NaN mismatch at ({i},{j}): got {g} want {w}");
        if w.is_finite() && *w != 0.0 {
            m = m.max(((g - w) / w).abs());
        } else if w.is_finite() {
            m = m.max(g.abs());
        }
    }
    m
}
