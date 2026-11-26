# Semantics

These rules apply identically to every method (Pearson, Spearman, Kendall,
bicor, hellcor, Gaussian rank, ρp, ξ, distance correlation) and every layout
(full, cross, packed upper, streamed blocks), with the exceptions stated
below for the asymmetric ξ and for covariates.
They are tested generically in `tests/invariants.rs`.

## Data convention

Rows are variables (genes), columns are observations (samples).

## Missing values (`NanPolicy`)

| Policy | Behaviour |
|---|---|
| `PairwiseComplete` (default) | Each pair uses the observations finite in both rows. `n_eff` is the number of such observations and is used for that pair's p-value. |
| `Propagate` | A row containing any non-finite value yields NaN for all its pairs (diagonal included). |
| `Error` | Any non-finite value returns `CorrError::NonFinite { row, col }`. |

Rows without missing values always take the fast path (matrix product for
Pearson/Spearman/Gaussian rank/bicor/ρp, prepared-once rows for
Kendall/hellcor/ξ/dcor). Only pairs that
touch a row with missing values take the per-pair slow path, which reuses the
method's own row transform/kernel on the gathered observations.

## Degenerate rows

A row is degenerate when it has fewer than two finite values or all its finite
values are equal (zero variance). For bicor with `mad_zero_fallback = Nan`, a
row whose median absolute deviation is zero is also degenerate; with the default
`Pearson` fallback such a row uses the Pearson transform. Every pair involving a
degenerate row is NaN, **including its own diagonal entry**.

## Diagonal

In symmetric layouts (full, upper, `full_blocks`) the diagonal is exactly `1.0`
for non-degenerate rows and NaN otherwise. Diagonal p-values are NaN (a
self-correlation is not a test; v0.2 wrote 0.0 — owner decision D2).

The cross layout has no notion of row identity: a row compared with an
identical row is computed like any other pair (1.0 up to rounding; values within
4 ulp of ±1 are snapped to ±1). The CLI snaps cells whose row IDs coincide on
both sides to exactly 1.0 with p = NaN when the value is defined.

## Small samples

- `n_eff < 3`: the correlation is returned when computable; the p-value is NaN.
- Hellcor with `n_eff ≤ 3` returns 0.0 (as the reference implementation does),
  p-value NaN.

## Range and rounding

Correlations lie in [−1, 1]. Rounding noise within 4 ulp of ±1 is snapped to ±1.
Overshoot larger than 1e-12 is a bug and trips a debug assertion in tests.

## Shapes and errors

Zero rows or zero columns → `CorrError::Empty`; column-count mismatch in cross
mode or vector-length mismatch in `pair` → `CorrError::ShapeMismatch`. No public
function panics on user data and none returns a silently zero-filled matrix.

## P-values

| Method | Mode | Formula |
|---|---|---|
| Pearson, Spearman, Gaussian rank, bicor | `Analytic` | two-sided Student t, t = r√((n−2)/(1−r²)), df = n_eff − 2 (n − 2 − k with k covariates); computed as I_{1−r²}(df/2, ½). Spearman's equals R `cor.test(exact = FALSE)`. |
| Kendall | `Analytic` | tie-corrected variance of S (Kendall), z = S/√var(S), two-sided normal, no continuity correction (R `cor.test(method="kendall", exact=FALSE)`). |
| ξ | `Analytic` | one-sided, p = 1 − Φ(√n ξ / τ̂) with the tie-aware τ̂ of Chatterjee (2021, §2) (≈ √(2/5) without ties), as `XICOR::xicor(pvalue = TRUE)`. Symmetrised ξ: min(1, 2 min(p₁, p₂)). |
| ρp, distance correlation | — | no p-values (`CorrError::Unsupported`); use a permutation test. |
| Hellcor | `MonteCarlo { samples: B, seed }` | permutation null per distinct `n_eff`; p = (1 + #{null ≥ obs}) / (B + 1). Resolution 1/(B+1); with heavy ties the null is approximate. |

Requesting `Analytic` for hellcor or `MonteCarlo` for the other methods returns
`CorrError::Unsupported`.

## Covariates (partial correlation)

`Correlator::covariates(z)` (k × n_obs) applies to Pearson, Spearman and
Gaussian rank; other methods return `Unsupported`. Each row's scores (values,
average ranks or normal scores) are centred and projected off an orthonormal
basis of the covariates, which are given the same score transform and are
centred; columns that are linear combinations of earlier ones (relative
residual norm < 1e-10) are dropped, and k is the number kept. This equals the
correlation of least-squares residuals with an intercept (`lm(x ~ z)`). A row
lying in the span of the covariates becomes degenerate (NaN). Covariates must be
finite (`InvalidParam` otherwise). Per-pair deletion of observations would
change the regression, so data containing non-finite values under
`PairwiseComplete` returns `Unsupported`; use `Propagate` (rows with missing
values become NaN) or `Error`.

## Proportionality ρp

With `clr = true` (default) the input is mapped to log(x + pseudocount) and
each value has the mean of the finite log values of its column subtracted; the
reference uses every row passed in (both matrices in cross mode). Values whose
shifted value is ≤ 0 or non-finite become NaN and follow the NaN policy. Apply
`mincorr::log_clr` to the whole feature table first when correlating a subset
of rows (the CLI does this before `--subset-*` selection), then use
`clr = false`, which takes values as they are.

## Chatterjee's ξ

ξ is not symmetric. In the full layout cell (i, j) is ξ(row i → row j), i.e.
how well row j is a function of row i, and every cell is computed; `upper`
returns `Unsupported`. Ties in x are ordered by observation index (the paper
breaks them at random), which makes results deterministic; ties in y are
handled exactly. The diagonal is 1.0 by the general convention, although the
estimator itself gives 1 − 3/(n + 1) for a duplicated tie-free row.
`ChatterjeeParams { symmetric: true }` reports max(ξ(x, y), ξ(y, x)) and
restores the symmetric layouts. ξ lies in [−½ + O(1/n), 1].

## Distance correlation

V-statistic dCor ∈ [0, 1] as `energy::dcor`, computed per pair in O(n log n)
(no n × n distance matrices). It is invariant to affine maps of either row.
Pairs are evaluated in a canonical argument order, so dcor(x, y) and
dcor(y, x) are bit-identical.

## Determinism

Same inputs and seed give bit-identical outputs for any thread count: work is
split into fixed-size tasks, and each Monte-Carlo replicate uses its own ChaCha8
stream (`set_stream(replicate)`). Hellcor fixes the argument order of a pair
(when `kmax = lmax`) so that η(x, y) and η(y, x) are bit-identical; permuting
samples changes η only at rounding level (summation order).

## Choices made where the specification was ambiguous

- Bicor `max_p_outliers < 1`: on each side of the median the weight-function
  scale is `max(9·MAD, 2 × distance from the median to the type-7 quantile at
  p resp. 1−p)`; the factor was fixed by black-box comparison with WGCNA, which
  it reproduces to ≤ 9e-16 (docs/parity.md).
- Hellcor order selection uses the normalised leave-one-out criterion by
  default (it reproduces HellCor on every fixture pair); the paper's MISE
  criterion is available as `HellcorCriterion::Mise`. See `docs/hellcor.md`.
- Hellcor on a constant row returns NaN (degenerate-row rule), whereas HellCor
  returns a value close to 1.
- Streamed full matrices (`full_blocks`) compute complete rows, so symmetric
  entries are computed twice; the in-memory `full()` computes one triangle and
  mirrors. `Correlator::triangle_cache(bytes)` lets `full_blocks` compute the
  triangle once in memory when the output fits; the CLI sets it from
  `--memory-limit-gb`.
