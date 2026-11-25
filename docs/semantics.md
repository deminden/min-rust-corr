# Semantics

These rules apply identically to every method (Pearson, Spearman, Kendall,
bicor, hellcor) and every layout (full, cross, packed upper, streamed blocks).
They are tested generically in `crates/core/tests/invariants.rs`.

## Data convention

Rows are variables (genes), columns are observations (samples).

## Missing values (`NanPolicy`)

| Policy | Behaviour |
|---|---|
| `PairwiseComplete` (default) | Each pair uses the observations finite in both rows. `n_eff` is the number of such observations and is used for that pair's p-value. |
| `Propagate` | A row containing any non-finite value yields NaN for all its pairs (diagonal included). |
| `Error` | Any non-finite value returns `CorrError::NonFinite { row, col }`. |

Rows without missing values always take the fast path (matrix product for
Pearson/Spearman/bicor, prepared-once rows for Kendall/hellcor). Only pairs that
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
| Pearson, Spearman, bicor | `Analytic` | two-sided Student t, t = r√((n−2)/(1−r²)), df = n_eff − 2; computed as I_{1−r²}(df/2, ½). Spearman's equals R `cor.test(exact = FALSE)`. |
| Kendall | `Analytic` | tie-corrected variance of S (Kendall), z = S/√var(S), two-sided normal, no continuity correction (R `cor.test(method="kendall", exact=FALSE)`). |
| Hellcor | `MonteCarlo { samples: B, seed }` | permutation null per distinct `n_eff`; p = (1 + #{null ≥ obs}) / (B + 1). Resolution 1/(B+1); with heavy ties the null is approximate. |

Requesting `Analytic` for hellcor or `MonteCarlo` for the other methods returns
`CorrError::Unsupported`.

## Determinism

Same inputs and seed give bit-identical outputs for any thread count: work is
split into fixed-size tasks, and each Monte-Carlo replicate uses its own ChaCha8
stream (`set_stream(replicate)`). Hellcor fixes the argument order of a pair
(when `kmax = lmax`) so that η(x, y) and η(y, x) are bit-identical; permuting
samples changes η only at rounding level (summation order).

## Choices made where the specification was ambiguous

- Bicor `max_p_outliers < 1`: on each side of the median the weight-function
  scale is `max(9·MAD, distance from the median to the quantile at p resp. 1−p)`
  (R type-7 quantile). Written from the WGCNA documentation only; parity with
  WGCNA is unverified until WGCNA fixtures are generated (decision D8).
- Hellcor order selection follows the paper (Appendix C, MISE criterion
  A²(K,L) − 2B(K,L)); the normalised variant described in the rewrite
  specification is available as `HellcorCriterion::NormalizedLoo`. See
  `docs/hellcor.md`.
- Streamed full matrices (`full_blocks`) compute complete rows, so symmetric
  entries are computed twice; the in-memory `full()` computes one triangle and
  mirrors. The CLI uses `full()` for pairwise methods while the matrix fits in
  `--memory-limit-gb`.
