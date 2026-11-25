# mincorr — fast correlation matrices

Minimal, fast Rust library, CLI and Python package for correlation matrices of
wide numeric data (rows = variables such as genes, columns = observations such
as samples):

| Method | Definition | Reference |
|---|---|---|
| Pearson | r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² Σ(y−ȳ)²) | — |
| Spearman | Pearson on average ranks | — |
| Kendall tau-b | S / √((n₀−n₁)(n₀−n₂)), tie-corrected | Knight (1966); Kendall, *Rank Correlation Methods* |
| bicor | biweight midcorrelation: median/MAD-weighted Pearson | Wilcox (2012); Langfelder & Horvath (2012) |
| hellcor | Hellinger correlation η from a Legendre-series estimate of √copula density | Geenens & Lafaye de Micheaux (2022) |

Every method is available in **full**, **cross** (A × B) and **packed upper
triangle** layouts, with optional **p-values** (Student t, tie-corrected normal,
or Monte-Carlo permutation null for hellcor), three **missing-value policies**,
and **streamed block output** so a 20 000 × 20 000 matrix never has to sit in RAM.

## Install

```bash
git clone https://github.com/deminden/min-rust-corr && cd min-rust-corr
cargo build --release            # binary: target/release/mincorr
cd crates/py && pip install maturin && maturin develop --release   # Python
```

Portable builds are the default. For a binary tuned to your own CPU:
`RUSTFLAGS="-C target-cpu=native" cargo build --release`.

## Quick start

CLI (input: TSV/CSV, optionally `.gz` or single-file `.tar.gz`; first column = row IDs):

```bash
mincorr expr.tsv.gz spearman                       # → expr_spearman_correlations.tsv.gz
mincorr expr.tsv.gz kendall 8 --with-pvalues       # 8 threads, p-values in a sibling file
mincorr expr.tsv.gz hellcor --with-pvalues --mc-samples 2000 --format npy -o hell.npy
mincorr expr.tsv.gz bicor --subset-rows TP53,MYC --subset-vs-all --format edges --top-k 50
mincorr expr.tsv.gz pearson --legacy-tar           # v0.2 output layout and names
```

Rust:

```rust
use mincorr::{Correlator, Method, NanPolicy, PvalueSpec};
let out = Correlator::new(Method::Kendall)
    .pvalues(PvalueSpec::Analytic)
    .nan_policy(NanPolicy::PairwiseComplete)
    .full(data.view())?;                 // out.corr, out.pvalues: Array2<f64>
Correlator::new(Method::Pearson).full_blocks(data.view(), |block| { /* write */ Ok(()) })?;
```

Python:

```python
import mincorr
r = mincorr.corr(x, method="spearman")                       # (n, n)
r, p = mincorr.corr(x, method="kendall", pvalues=True)
u = mincorr.corr(x, method="bicor", layout="upper")         # packed n(n+1)/2
c = mincorr.corr(a, b, method="hellcor", layout="cross", alpha=6.0)
r = mincorr.pearson_matrix(x)                                # all v0.2 names still work
```

## Semantics (summary)

- Missing values: `pairwise` (default; each pair uses jointly finite
  observations and its own n for p-values), `propagate`, or `error`.
- Degenerate rows (fewer than two finite values or constant) give NaN for every
  pair, including their diagonal entry; otherwise the diagonal is exactly 1.
- Diagonal p-values are NaN; p-values for n_eff < 3 are NaN.
- Same inputs and seed → bit-identical results for any number of threads.

Full rules: [docs/semantics.md](docs/semantics.md). Hellcor algorithm:
[docs/hellcor.md](docs/hellcor.md). Agreement with R:
[docs/parity.md](docs/parity.md).

## Performance notes

- Pearson, Spearman and bicor are Gram methods: each row is transformed once into
  a unit vector and correlations are blocked matrix products computed in
  parallel. The default build uses ndarray's pure-Rust `matrixmultiply` kernel;
  enable the `blas` cargo feature of `mincorr` to link a system BLAS instead.
- Kendall uses pair-sign bitsets for n ≤ 128 and Knight's O(n log n) algorithm
  otherwise; hellcor costs O(n² + n·K·L) per pair with per-sample-size tables.
- Measured timings: [docs/benchmarks.md](docs/benchmarks.md).

## Licence

MIT (see [LICENSE](LICENSE)). Versions up to and including 0.2.1 were
GPL-2.0-or-later because their bicor, hellcor and Beta-function code were close
ports of GPL R code. Version 0.3.0 replaces those parts with independent
implementations written from the published papers; commits before the
`v0.3.0` tag remain under GPL-2.0-or-later. File-by-file origins and the
references consulted are recorded in [PROVENANCE.md](PROVENANCE.md).

## Citation

See [CITATION.cff](CITATION.cff).
