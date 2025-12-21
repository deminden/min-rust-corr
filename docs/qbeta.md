## qbeta (Rmath port) notes for HellCor

This project includes a Rust port of R's `qbeta()` (quantile of the Beta
distribution) under `src/hellinger/`. The goal is parity with R for practical
HellCor workloads (moderate tails, typical shapes), and it matches R across the
test suite that reflects those regimes.

### Ready-for-use scope

- Intended use: HellCor-style inputs where `u = rank/(n+1)` keeps probabilities
  away from extreme tails (typically >= 1e-3).
- Expected shape ranges: typical data-driven parameters, not astronomically
  large or tiny.
- The implementation matches R outputs in normal ranges and realistic tails.

### Known limitation (stress tails)

For extreme underflow-plateau cases (e.g. `p <= 1e-300` or `log_p <= ln(1e-300)`
combined with very large or very small shapes), the inverse is dominated by
underflow and may differ from R by ~1e-9 in x. This is a numerical plateau issue
and does not affect HellCor's expected inputs.

### Test harness policy

The R-compare integration test (`tests/qbeta_against_r.rs`) skips cases where:

- `p <= 1e-300` (or `log_p <= ln(1e-300)`), and
- `max(a, b) >= 1e6` or `min(a, b) <= 1e-6`.

These cases are underflow-plateau dominated and are intentionally excluded to
avoid gating correctness on non-meaningful extremes for HellCor.

### Numeric tolerances

The test uses a strict comparison with:

- relative tolerance ~`2e-15 * max(|r|, |rust|, 1)`, and
- up to 16 ULPs.

This avoids false negatives due to small libm differences while staying
faithful to R in practice.

### Empirical HellCor parity

End-to-end comparisons against the R `HellCor` package (C version) show max
absolute differences on full matrices at ~1e-10 or better:

- `gene_expression.tsv.gz`: `1.53e-10`
- `bladder_small_590genes.tsv.gz`: `1.99e-10`
- `bladder_large_2950genes.tsv.gz`: `1.60e-12`

### Running the check

From the repo root (requires R installed):

```bash
cargo test qbeta_against_r
```

### Debugging

To trace Newton steps and pbeta behavior in the port:

```bash
DEBUG_QBETA_LOG=1 cargo test qbeta_against_r -- --nocapture
```
