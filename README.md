# mincorr

[![crates.io](https://img.shields.io/crates/v/mincorr.svg)](https://crates.io/crates/mincorr)
[![docs.rs](https://img.shields.io/docsrs/mincorr)](https://docs.rs/mincorr)
[![PyPI](https://img.shields.io/pypi/v/mincorr.svg)](https://pypi.org/project/mincorr/)
[![CI](https://github.com/deminden/min-rust-corr/actions/workflows/ci.yml/badge.svg)](https://github.com/deminden/min-rust-corr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Fast correlation matrices for wide data** — a Rust library, a command-line
tool and a Python package. Rows are variables (genes, features), columns are
observations (samples).

- **Nine methods:** Pearson, Spearman, Kendall tau-b, biweight midcorrelation
  (bicor), Hellinger correlation, Gaussian-rank correlation, proportionality
  ρp, Chatterjee's ξ and distance correlation, plus **partial correlation**
  (adjusting Pearson, Spearman or Gaussian rank for covariates).
- **Every layout:** full matrix, cross matrix (A × B), packed upper triangle,
  single pairs, or streamed row blocks that never hold the whole matrix in RAM.
- **P-values:** Student t (with n − 2 − k df for k covariates),
  tie-corrected normal (Kendall), the asymptotic null of ξ, or a Monte-Carlo
  permutation null (hellcor).
- **Missing values handled one way for all methods:** pairwise-complete
  (default), propagate or error.
- **Validated against R:** `stats::cor`, `cor.test`, `lm` residuals,
  `WGCNA::bicor`, `HellCor::HellCor`, `energy::dcor` and `XICOR::xicor` on
  every test dataset ([parity](docs/parity.md)).
- **Fast and deterministic:** parallel, bit-identical for any thread count;
  hellcor is ~120× faster than the previous release ([benchmarks](docs/benchmarks.md)).

## Methods

| Method | Name | Definition | Reference |
|---|---|---|---|
| Pearson | `pearson` | r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² Σ(y−ȳ)²) | — |
| Spearman | `spearman` | Pearson on average ranks | — |
| Kendall tau-b | `kendall` | S / √((n₀−n₁)(n₀−n₂)), tie-corrected | Knight (1966) |
| Biweight midcorrelation | `bicor` | median/MAD-weighted Pearson, robust to outliers | Langfelder & Horvath (2012) |
| Hellinger correlation | `hellcor` | detects any (also non-monotone) dependence, η ∈ [0, 1] | Geenens & Lafaye de Micheaux (2022) |
| Gaussian rank | `gaussian_rank` | Pearson on normal scores Φ⁻¹(R/(n+1)); robust, fully efficient at the normal | Boudt, Cornelissen & Croux (2012) |
| Proportionality | `rho` | ρp = 1 − var(x−y)/(var x + var y) on clr-transformed compositions | Lovell et al. (2015); Erb & Notredame (2016) |
| Chatterjee's ξ | `xi` | 1 − n Σ\|r₍ᵢ₊₁₎−r₍ᵢ₎\| / (2 Σ lᵢ(n−lᵢ)): is y a function of x? Asymmetric, ∈ [−½, 1] | Chatterjee (2021) |
| Distance correlation | `dcor` | dCov(x,y)/√(dVar(x) dVar(y)) on double-centred distances; 0 iff independent; O(n log n) per pair | Székely, Rizzo & Bakirov (2007); Huo & Székely (2016) |

Partial correlation: `Correlator::covariates(z)` (CLI `--covariates file.tsv`,
Python `covariates=`) residualises every row on the covariates — after the
rank or normal-score transform for Spearman and Gaussian rank — before
correlating.

## Install

| What | How |
|---|---|
| Command-line tool (prebuilt) | download from [GitHub Releases](https://github.com/deminden/min-rust-corr/releases) — Linux (x86_64, aarch64; static), macOS (Apple silicon, Intel), Windows (x86_64, ARM64) |
| Command-line tool (from source) | `cargo install mincorr` |
| Rust library | `cargo add mincorr --no-default-features` |
| Python | `pip install mincorr` |

The default `cli` feature builds the `mincorr` binary; library users turn it off
with `--no-default-features` to skip its dependencies. The optional `blas`
feature links a system BLAS for the matrix-product kernel. Minimum supported
Rust version: 1.85.

## Command line

Input: a TSV or CSV file (optionally `.gz`), first column = row IDs, header =
sample names.

```bash
mincorr expr.tsv.gz spearman                        # -> expr_spearman_correlations.tsv.gz
mincorr expr.tsv.gz kendall --threads 8 --with-pvalues
mincorr expr.tsv.gz hellcor --with-pvalues --mc-samples 2000 --format npy -o hell.npy
mincorr expr.tsv.gz bicor --subset-rows TP53,MYC --subset-vs-all --format edges --top-k 50
mincorr expr.tsv.gz spearman --covariates batch_age.tsv --with-pvalues  # partial
mincorr counts.tsv rho --pseudocount 1                # proportionality on clr(counts + 1)
mincorr --help
```

Outputs are written block by block (`tsv`, `tsv.gz`, `npy` or a thresholded
edge list), so memory stays at about one block regardless of the matrix size.
v0.2 command lines keep working; `--legacy-tar` reproduces the v0.2 output files.

## Rust

```rust
use mincorr::{Correlator, Method, PvalueSpec};
use ndarray::array;

fn main() -> Result<(), mincorr::CorrError> {
    let data = array![
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 1.0, 4.0, 3.0, 6.0],
        [5.0, 3.0, 2.0, 2.5, 1.0],
    ];
    let out = Correlator::new(Method::Kendall).pvalues(PvalueSpec::Analytic).full(data.view())?;
    let p = out.pvalues.as_ref().map_or(f64::NAN, |p| p[[0, 1]]);
    println!("tau = {:.3}, p = {:.3}", out.corr[[0, 1]], p);

    // Stream a large matrix in row blocks instead of holding it in memory.
    Correlator::new(Method::Pearson).full_blocks(data.view(), |block| {
        for i in block.rows.clone() {
            let _row: &[f64] = block.corr_row(i); // write the row somewhere
        }
        Ok(())
    })
}
```

API documentation: [docs.rs/mincorr](https://docs.rs/mincorr).

## Python

```python
import numpy as np
import mincorr

x = np.random.default_rng(0).normal(size=(100, 40))       # 100 genes × 40 samples
r = mincorr.corr(x, method="spearman")                     # (100, 100)
r, p = mincorr.corr(x, method="kendall", pvalues=True)
u = mincorr.corr(x, method="bicor", layout="upper")       # packed upper triangle
c = mincorr.corr(x[:5], x, method="hellcor", layout="cross")
d = mincorr.corr(x, method="dcor")
rp = mincorr.corr(x, method="pearson", covariates=x[:2])  # partial correlation
```

## Behaviour in brief

- Degenerate rows (fewer than two finite values, or constant) give NaN for every
  pair, including their own diagonal entry; otherwise the diagonal is exactly 1.
- ξ is asymmetric: cell (i, j) is ξ(row i → row j); `--xi-symmetric` /
  `symmetric=True` reports max(ξ(x,y), ξ(y,x)).
- Diagonal p-values are NaN; p-values need at least 3 complete observations.
- With missing values, each pair uses its own complete observations and its own
  sample size for the p-value.

Details: [semantics](docs/semantics.md) · [Hellinger correlation notes](docs/hellcor.md) ·
[parity with R](docs/parity.md) · [benchmarks](docs/benchmarks.md) ·
[changelog](CHANGELOG.md) · [development](docs/development.md).

## Licence

MIT. Versions up to 0.2.1 were GPL-2.0-or-later because parts of them were
ports of GPL R code; 0.3.0 replaces those parts with independent
implementations written from the published papers (see
[PROVENANCE.md](PROVENANCE.md)). Commits before the `v0.3.0` tag remain under
GPL-2.0-or-later.

## Citation

See [CITATION.cff](CITATION.cff).
