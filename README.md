# High-Performance Correlation Analysis

Minimalistic Rust implementation for computing Pearson, Spearman, Kendall, biweight midcorrelations (bicor), and Hellinger correlations (hellcor) on large data matrices.
Results match R implementations. Pearson and Spearman are using BLAS optimizations

## Usage

### As a Binary

```bash
# Build
git clone https://github.com/deminden/min-rust-corr
cd min-rust-corr
cargo build --workspace --release

# Run correlations
mincorr() { ./target/release/mincorr-cli "$@"; }
mincorr data.tsv.gz pearson --time # time of execution tracked
mincorr data.tsv.gz spearman 8     # 8 threads
mincorr data.tsv.gz kendall        # all cores
mincorr data.tsv.gz bicor          # biweight midcorrelation
mincorr data.tsv.gz hellcor        # Hellinger correlation
```

### As a Crate

Add to `Cargo.toml`:
```toml
[dependencies]
mincorr = { git = "https://github.com/deminden/min-rust-corr" }
```

Use in code:
```rust
use mincorr::{pearson, spearman, kendall, bicor, hellcor};
use ndarray::Array2;

let data: Array2<f64> = /* your data matrix */;

// Compute correlations
let pearson_corr = pearson::matrix(&data);
let spearman_corr = spearman::matrix(&data);
let kendall_corr = kendall::matrix(&data);
let bicor_corr = bicor::matrix(&data);
let hellcor_corr = hellcor::matrix(&data);
```

### Python Extension

The Python extension lives in `crates/py` (package name `mincorrpy`, module name `mincorr`) and is built with maturin.
See `crates/py/pyproject.toml` for build metadata. Install from GitHub:
```bash
git clone https://github.com/deminden/min-rust-corr
cd min-rust-corr/crates/py
python -m pip install maturin
maturin develop --release
```

Usage example (pairwise + matrix):
```python
import numpy as np
import mincorr

x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

data = np.vstack([x, y])

# Pairwise correlations
print(mincorr.pearson_pair(x, y))
print(mincorr.hellcor_pair(x, y, 6.0)) # alpha = 6.0 (default value, can be skipped)

# Full matrix correlations
print(mincorr.pearson_matrix(data))
print(mincorr.hellcor_matrix(data))
```

## Input Format

TSV/CSV with row IDs as first column, data values as remaining columns:
```
row_id     col_1       col_2       col_3
ROW001     12.34       15.67       8.90
ROW002     45.12       23.45       67.89
```

Supports `.gz` compression.


## Output

Creates `<input_basename>_<method>_correlations.tar.gz` containing correlation matrix in TSV format. Results are saved in the current directory or `outputs/` folder when using the GTEx processing script. 
If a subset mode is used, the subset tag is inserted in the output name, e.g. `<input_basename>_subset500_seed42_<method>_correlations.tar.gz`.

## R references for correlations

| Correlation | CLI name | R reference (tests/investigate_diffs.R) |
| --- | --- | --- |
| Pearson | `pearson` | `stats::cor` |
| Spearman | `spearman` | `stats::cor` (on `matrixStats::rowRanks`) |
| Kendall | `kendall` | `pcaPP::cor.fk` |
| Bicor (biweight midcorrelation) | `bicor` | `WGCNA::bicor` |
| Hellinger correlation | `hellcor` | `HellCor::HellCor` |

## Precision vs R

General note (all correlation types): results are floating‑point computations and will differ at the level of rounding noise across implementations. In practice we see exact agreement in structure (NaN patterns and ordering), with numeric differences at or below ~1e‑16–1e‑12 depending on the method and dataset.

Hellcor parity: measured max absolute difference vs `HellCor::HellCor` (R/C++ implementation) on GTEx bladder tissue (N = 77 samples, 2,950 genes) is `1.60e-12`.

## P-value Parity vs R (Snapshot)

Command used:

```bash
Rscript crates/core/tests/pvalue_parity_check.R data/bladder_small_590genes.tsv.gz 12 100 42
```

Setup:
- data: GTEx bladder tissue subset (`12` genes, `77` samples) for analytic p-value parity.
- hellcor MC behavior: independent uniforms with `n=276` pairwise p-values, `mc_samples=100`.

Analytic p-value parity (Rust vs R):

| Method | MAE | RMSE | Max abs diff | Correlation |
| --- | --- | --- | --- | --- |
| Pearson | 2.564e-14 | 3.699e-14 | 8.030e-14 | 1.000000 |
| Spearman | 5.262e-14 | 2.319e-13 | 1.851e-12 | 1.000000 |
| Kendall | 5.070e-12 | 1.159e-11 | 4.028e-11 | 1.000000 |
| Bicor | 2.521e-14 | 3.738e-14 | 8.077e-14 | 1.000000 |

Hellcor Monte Carlo p-value behavior under independence:
- Rust KS vs `U(0,1)`: `D=0.1257`, `p=3.244e-4`, `n=276`
- R KS vs `U(0,1)`: `D=0.1216`, `p=5.697e-4`, `n=276`
- Rust vs R two-sample KS: `D=0.1159`, `p=4.895e-2`, `n=276`

## P-value Timing vs R (Calculation Only)

Snapshot benchmark for p-value estimation + correlation calculation:

- data: GTEx bladder tissue (`77` samples)
- Pearson/Spearman/Kendall/Bicor: `120` genes
- Hellcor MC: `24` genes, `mc_samples=300`, `seed=42`
- Rust: `16` threads (`mincorr-cli`)
- R: `32` threads (WGCNA threads enabled; Hellcor uses shared-null MC strategy for parity with Rust mode)

| Method | Rust (s) | R (s) | Speedup (R/Rust) |
| --- | --- | --- | --- |
| Pearson | 0.004 | 0.028 | 7.0x |
| Spearman | 0.005 | 0.282 | 56.4x |
| Kendall | 0.005 | 1.914 | 382.8x |
| Bicor | 0.005 | 0.017 | 3.4x |
| Hellcor (MC, B=300, 24 genes) | 0.258 | 6.804 | 26.4x |

## Timing Comparison (Correlation Calculation Only)

Benchmarked on AMD Ryzen 9 7950X3D (16 cores, 32 threads). Times exclude load/output.

GTEx bladder tissue (N = 77, 590-gene subset)

| Method | Rust (16 threads) | R (32 threads) | Speedup (R/Rust) |
| --- | --- | --- | --- |
| Pearson | 0.008s | 0.022s | 2.8x |
| Spearman | 0.008s | 0.030s | 3.8x |
| Kendall | 0.050s | 0.97s | 19.4x |
| Bicor | 0.004s | 0.020s | 5.0x |
| Hellcor | 61s | 107s | 1.8x |

GTEx bladder tissue (N = 77, 2950 genes)

| Method | Rust (16 threads) | R (32 threads) | Speedup (R/Rust) |
| --- | --- | --- | --- |
| Pearson | 0.16s | 0.33s | 2.1x |
| Spearman | 0.15s | 0.33s | 2.2x |
| Kendall | 1.7s | 22.6s | 13.3x |
| Bicor | 0.070s | 0.074s | 1.1x |
| Hellcor | 1575s | 3183s | 2.0x |

## Usage with GTEx dataset

```bash
# Process all GTEx tissues
./examples/process_GTEX_tissues.sh

# Process specific tissue with custom threads
./examples/process_GTEX_tissues.sh Bladder 8
```

## Test with sample data

```bash
# Quick test with 500 genes subset
./crates/core/tests/GTEx_test.sh

# Compare results with R implementations  
Rscript crates/core/tests/investigate_diffs.R data/your_file.tsv.gz

# Compact p-value parity checks (Pearson/Spearman/Kendall/Bicor + Hellcor MC behavior)
Rscript crates/core/tests/pvalue_parity_check.R data/your_file.tsv.gz
```

## Subset and Cross Modes (CLI + Python)

This project supports three output layouts:

- subset-only (`A x A`): compute within one selected subset.
- subset-vs-subset (`A x B`): compute between two selected subsets.
- subset-vs-all (`A x all`): compute one selected subset against all rows.

### CLI

```bash
# A x A (legacy subset mode, aliases map to subset A)
mincorr data.tsv.gz pearson --subset-size 500 --subset-seed 42
mincorr data.tsv.gz pearson --subset-file genes.txt
mincorr data.tsv.gz pearson --subset-rows ENSG000001,ENSG000002

# A x B
mincorr data.tsv.gz pearson --subset-a-file genes_a.txt --subset-b-file genes_b.txt

# A x all
mincorr data.tsv.gz pearson --subset-size 500 --subset-seed 42 --subset-vs-all

# Correlation + p-values
mincorr data.tsv.gz pearson --with-pvalues
mincorr data.tsv.gz hellcor --with-pvalues --pvalue-mode mc --mc-samples 10000 --mc-seed 42
```

Rules:

- `--subset-size N` supports optional `--subset-seed S` (default `42`).
- `--subset-file PATH` reads newline-separated row IDs.
- `--subset-rows ID1,ID2,...` accepts comma-separated row IDs.
- On each side (`A` or `B`), use exactly one selector: `size`, `file`, or `rows`.
- `--subset-vs-all` requires subset A and cannot be combined with `--subset-b-*`.

P-value options:

- `--with-pvalues` writes an additional `..._<method>_pvalues.tsv` into the output tarball.
- `--pvalue-mode auto|analytic|mc` controls estimator type.
- `--mc-samples B` and `--mc-seed S` configure Monte Carlo sampling (used by hellcor).
- Analytic p-values are available for `pearson`, `spearman`, `kendall`, `bicor`.
- `hellcor` uses Monte Carlo p-values under independence (R-style null simulation).

### Python

Use matrix slicing to define row subsets, then call cross-matrix APIs:

```python
import mincorr

# data_all: shape (n_genes, n_samples)
# data_a: subset A rows, shape (n_a, n_samples)
# data_b: subset B rows, shape (n_b, n_samples)

# A x B
r_ab = mincorr.pearson_cross_matrix(data_a, data_b)
r_ab_h = mincorr.hellcor_cross_matrix(data_a, data_b, alpha=6.0)

# A x all
r_a_all = mincorr.spearman_cross_matrix(data_a, data_all)
```

Python cross functions available for all methods:

- `pearson_cross_matrix`
- `spearman_cross_matrix`
- `kendall_cross_matrix`
- `bicor_cross_matrix`
- `hellcor_cross_matrix`

Python p-value APIs:

- `pearson_matrix_with_pvalues`, `spearman_matrix_with_pvalues`, `kendall_matrix_with_pvalues`, `bicor_matrix_with_pvalues`
- `hellcor_matrix_with_pvalues(data, alpha=6.0, mc_samples=10000, seed=42)`
- `pearson_cross_matrix_with_pvalues`, `spearman_cross_matrix_with_pvalues`, `kendall_cross_matrix_with_pvalues`, `bicor_cross_matrix_with_pvalues`
- `hellcor_cross_matrix_with_pvalues(data_a, data_b, alpha=6.0, mc_samples=10000, seed=42)`

Inputs must have the same number of columns (samples) in both matrices.

## Contributing

Contributions are welcome and encouraged! If you’d like to help improve `min-rust-corr`, feel free to open an issue for bugs, feature requests, performance ideas.

Pull requests are especially appreciated for:
- performance improvements
- additional methods or variants
- more or better parity tests vs R
- docs, examples, and benchmarks

### Development notes

Please run the usual checks before submitting:
```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
```
## License
License: GPL-2.0-or-later for now, because bicor and hellcor started as close ports of the R reference implementations (WGCNA and HellCor) to ensure output parity. Earlier commits were temporarily labeled MIT; however, once bicor (commit f8b5fc) and hellcor (commit b6802a4) were introduced as close ports for R-parity, the project was relicensed to GPL for compliance. I plan to rewrite these algorithms independently and then move the project to a permissive license (MIT/Apache-2.0).
