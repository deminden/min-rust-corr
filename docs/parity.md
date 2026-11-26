# Parity with R

Reference values come from `tools/make_fixtures.R` (black-box use of public R
APIs; R 4.6.2 with WGCNA 1.74, HellCor 1.3, energy 1.7.12 and XICOR 0.4.1, see
MANIFEST.json). Tests: `tests/parity_rank_methods.rs`, `beta_fixtures.rs`,
`parity_robust.rs`, `parity_new_methods.rs`. Columns give max |Δr| over all off-diagonal pairs and the max
relative difference of p-values (`cor.test(..., exact = FALSE)`), under
pairwise-complete handling of missing values.

## Pearson, Spearman, Kendall (all datasets pass: r ≤ 1e-13 / 1e-14, p ≤ 1e-11)

| Dataset | Pearson max abs Δr / max rel Δp | Spearman | Kendall |
|---|---|---|---|
| normal_30x40 | 2.220e-16 / 3.822e-14 | 2.220e-16 / 3.658e-14 | 5.551e-17 / 4.085e-15 |
| normal_8x200 | 3.331e-16 / 2.819e-13 | 1.110e-16 / 1.606e-13 | 0.000e0 / 1.696e-15 |
| lognormal_20x77 | 2.776e-16 / 1.036e-13 | 1.388e-16 / 1.049e-13 | 5.551e-17 / 4.323e-15 |
| ties_15x30 | 3.331e-16 / 5.778e-14 | 4.441e-16 / 6.636e-14 | 5.551e-17 / 2.599e-15 |
| outliers_15x50 | 2.776e-16 / 5.199e-14 | 1.665e-16 / 4.825e-14 | 0.000e0 / 3.195e-15 |
| constant_rows_6x20 | 1.110e-16 / 7.282e-15 | 1.110e-16 / 5.941e-15 | 2.776e-17 / 9.925e-16 |
| na_12x40 | 2.220e-16 / 6.219e-14 | 2.776e-16 / 5.467e-14 | 5.551e-17 / 3.504e-15 |
| tiny_5x3 | 2.220e-16 / 2.495e-15 | 1.110e-16 / 0.000e0 | 5.551e-17 / 1.895e-15 |
| tiny_5x4 | 2.220e-16 / 2.504e-15 | 1.110e-16 / 6.939e-16 | 1.110e-16 / 7.965e-16 |
| monotone_pairs | 4.441e-16 / 4.389e-12 | 4.441e-16 / 4.600e-14 | 0.000e0 / 3.308e-15 |

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

## bicor (WGCNA::bicor / bicorAndPvalue) and hellcor (HellCor::HellCor)

All pairs agree. Hellcor pairs involving a constant row are excluded: HellCor
returns ≈ 1 there, while mincorr returns NaN for degenerate rows by design
(docs/semantics.md). Hellcor is compared on the datasets without missing values
and with at least 4 samples.

| Dataset | bicor Δr / rel Δp | maxPOutliers 0.1 | maxPOutliers 0.05 |
|---|---|---|---|
| normal_30x40 | 2.220e-16 / 4.376e-14 | 5.551e-16 | 5.551e-16 |
| normal_8x200 | 1.249e-16 / 2.963e-13 | 8.882e-16 | 8.882e-16 |
| lognormal_20x77 | 1.665e-16 / 1.334e-13 | 8.882e-16 | 7.772e-16 |
| ties_15x30 | 2.220e-16 / 5.232e-14 | 6.661e-16 | 6.661e-16 |
| outliers_15x50 | 1.665e-16 / 4.482e-14 | 4.441e-16 | 3.331e-16 |
| constant_rows_6x20 | 2.220e-16 / 3.482e-15 | 2.220e-16 | 2.220e-16 |
| na_12x40 | 2.220e-16 / 2.728e-14 | 5.551e-16 | 5.551e-16 |
| tiny_5x3 | 2.220e-16 / 8.603e-16 | 2.220e-16 | 2.220e-16 |
| tiny_5x4 | 1.110e-16 / 2.656e-15 | 2.220e-16 | 2.220e-16 |
| monotone_pairs | 2.220e-16 / 4.653e-13 | 3.331e-16 | 4.441e-16 |

| Dataset | hellcor α=6 max Δη (pairs) | α=3 | α=10 |
|---|---|---|---|
| normal_30x40 | 3.317e-14 (435/435) | 3.070e-14 | 2.776e-14 |
| normal_8x200 | 1.703e-14 (28/28) | 2.734e-14 | 3.568e-14 |
| lognormal_20x77 | 2.244e-14 (190/190) | 1.384e-14 | 2.418e-14 |
| ties_15x30 | 2.300e-12 (105/105) | 2.226e-12 | 2.296e-12 |
| outliers_15x50 | 1.670e-14 (105/105) | 1.250e-14 | 1.612e-14 |
| constant_rows_6x20 | 1.807e-12 (6/6) | 1.388e-12 | 1.777e-12 |
| tiny_5x4 | 2.442e-15 (10/10) | 1.998e-15 | 3.331e-15 |
| monotone_pairs | 8.493e-14 (21/21) | 9.104e-14 | 8.216e-14 |

Two behaviours were settled by these black-box comparisons:

- Hellcor order selection: the normalised leave-one-out criterion
  (`HellcorCriterion::NormalizedLoo`, now the default) reproduces HellCor on every
  pair; the MISE criterion as written in the paper's Appendix C picks different
  orders on 5 of 435 pairs of `normal_30x40` and on about half of the tied pairs.
- bicor `maxPOutliers`: widening the scale on each side to twice the distance
  from the median to the type-7 quantile at p / 1 − p (when larger than 9·MAD)
  reproduces WGCNA to ≤ 9e-16.

## Methods added in 0.4 (`tests/parity_new_methods.rs`)

Datasets without missing values and with at least 6 samples (normal_30x40,
normal_8x200, lognormal_20x77, ties_15x30, outliers_15x50,
constant_rows_6x20, monotone_pairs). Pairs involving a constant row are
excluded where the oracle returns a number (energy returns 0, the ξ formula
±inf); mincorr returns NaN for degenerate rows. Values are the maxima over all
datasets.

| Method | Reference | max abs Δr | max Δp |
|---|---|---|---|
| Gaussian rank | `cor` / `cor.test` on `qnorm(rank(x)/(n+1))` | 4.4e-16 | 2.3e-13 (rel.) |
| Partial Pearson (2 covariates) | `cor` of `lm(x ~ Z)` residuals; t with n − 4 df | 5.6e-16 | 4.9e-13 (rel.) |
| Partial Spearman (2 covariates) | same on ranks of x and Z | 4.4e-16 | 8.3e-14 (rel.) |
| ξ (all datasets, ties included) | base-R formula of the paper, x-ties by index | 0 (exact) | — |
| ξ (tie-free datasets) | `XICOR::xicor(x, y, pvalue = TRUE)` | 1.6e-15 | 9.4e-15 (abs.) |
| Distance correlation | `energy::dcor` | 4.8e-14 | — |
| ρp (positive datasets) | 1 − var(x−y)/(var x + var y) on clr | 9.6e-16 | — |

XICOR computes its p-value as 1 − Φ(z), which loses all relative accuracy below
~1e-12; mincorr computes Φ(−z) via erfc, so ξ p-values are compared with an
absolute tolerance.
