# Parity with R

Reference values come from `tools/make_fixtures.R` (black-box use of public R
APIs; R 4.3.3). Tests: `crates/core/tests/parity_rank_methods.rs`,
`beta_fixtures.rs`. Columns give max |Δr| over all off-diagonal pairs and the max
relative difference of p-values (`cor.test(..., exact = FALSE)`), under
pairwise-complete handling of missing values.

## Pearson, Spearman, Kendall (all datasets pass: r ≤ 1e-13 / 1e-14, p ≤ 1e-11)

| Dataset | Pearson max abs Δr / max rel Δp | Spearman | Kendall |
|---|---|---|---|
| normal_30x40 | 1.665e-16 / 3.665e-14 | 1.665e-16 / 3.523e-14 | 5.551e-17 / 4.085e-15 |
| normal_8x200 | 1.388e-16 / 2.151e-13 | 1.110e-16 / 1.607e-13 | 0.000e0 / 1.696e-15 |
| lognormal_20x77 | 1.665e-16 / 9.698e-14 | 1.388e-16 / 1.049e-13 | 5.551e-17 / 4.323e-15 |
| ties_15x30 | 3.331e-16 / 5.721e-14 | 3.331e-16 / 6.636e-14 | 5.551e-17 / 2.599e-15 |
| outliers_15x50 | 1.665e-16 / 4.734e-14 | 1.665e-16 / 4.806e-14 | 0.000e0 / 3.195e-15 |
| constant_rows_6x20 | 2.220e-16 / 7.857e-15 | 1.110e-16 / 5.941e-15 | 2.776e-17 / 9.925e-16 |
| na_12x40 | 1.110e-16 / 6.157e-14 | 2.776e-16 / 5.488e-14 | 5.551e-17 / 3.504e-15 |
| tiny_5x3 | 1.110e-16 / 2.339e-15 | 1.110e-16 / 0.000e0 | 5.551e-17 / 1.895e-15 |
| tiny_5x4 | 1.110e-16 / 4.616e-16 | 1.110e-16 / 6.939e-16 | 1.110e-16 / 7.965e-16 |
| monotone_pairs | 5.551e-16 / 2.144e-12 | 4.441e-16 / 4.600e-14 | 0.000e0 / 3.308e-15 |

The largest p-value difference (Pearson, `monotone_pairs`, 2.1e-12 relative)
occurs for p-values around 1e-30 where |r| is within 1e-9 of 1; it reflects the
conditioning of p(r) there, not an implementation difference.

## Beta distribution (clean-room) vs `qbeta` / `pbeta` / `dbeta`

Grid: a = b ∈ {0.5, 1, 2, 3, 6, 10, 50}, (2, 5), (0.7, 9); p = k/(n+1) and
half-steps for n ∈ {5, 77, 1000}; x = k/200.

| Function | Max relative error | Tolerance (Hellcor regime / elsewhere) |
|---|---|---|
| cdf | 3.4e-14 | 1e-13 / 1e-10 |
| pdf | 5.7e-14 | 1e-13 / 1e-10 |
| quantile | 9.2e-15 | 1e-12 / 1e-10 |

## bicor and hellcor

**Pending.** WGCNA and HellCor could not be installed in the build environment
(CRAN and Bioconductor were unreachable), so their fixtures were not generated
and nothing was fabricated. To complete this table:

```bash
Rscript tools/make_fixtures.R          # on a machine with WGCNA and HellCor
cargo test -p mincorr                  # bicor/hellcor parity tests pick up the new files
```

Expected outcome, by construction: bicor with default parameters should agree to
~1e-15 (it is the published closed form); `maxPOutliers < 1` is written from the
documentation only and may differ (decision D8). Hellcor may differ at the
1e-12–1e-10 level, and more where the default cross-validation criterion
(paper's MISE) differs from HellCor's; switch with `HellcorCriterion` and compare.
