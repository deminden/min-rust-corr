# High-Performance Correlation Analysis

Minimalistic Rust implementation for computing Pearson, Spearman, Kendall, biweight midcorrelations (bicor), and Hellinger correlations (hellcor) on large data matrices.
Results match R implementations. Pearson and Spearman are using BLAS optimizations

## Usage

### As a Binary

```bash
# Build
cargo build --release

# Run correlations
cargo run --release -- data.tsv.gz pearson --time # time of execution tracked
cargo run --release -- data.tsv.gz spearman 8     # 8 threads
cargo run --release -- data.tsv.gz kendall        # all cores
cargo run --release -- data.tsv.gz bicor          # biweight midcorrelation
cargo run --release -- data.tsv.gz hellcor        # Hellinger correlation
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
let pearson_corr = pearson::correlation_matrix(&data);
let spearman_corr = spearman::correlation_matrix(&data);
let kendall_corr = kendall::correlation_matrix(&data);
let bicor_corr = bicor::correlation_matrix(&data);
let hellcor_corr = hellcor::correlation_matrix(&data);
```

## Input Format

TSV/CSV with row IDs as first column, data values as remaining columns:
```
row_id     col_1       col_2       col_3
ROW001     12.34       15.67       8.90
ROW002     45.12       23.45       67.89
```

Supports `.gz` compression.

## Usage with GTEx dataset

```bash
# Process all GTEx tissues
./examples/process_GTEX_tissues.sh

# Process specific tissue with custom threads
./examples/process_GTEX_tissues.sh Bladder 8
```

## Test with sample data

```bash
# Test with 10k genes subset
./tests/GTEX_test.sh

# Compare results with R implementations  
Rscript tests/investigate_diffs.R data/your_file.tsv.gz
```


## Output

Creates `<input_basename>_<method>_correlations.tar.gz` containing correlation matrix in TSV format. Results are saved in the current directory or `outputs/` folder when using the GTEx processing script. 

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

## Timing Comparison (Correlation Calculation Only)

Benchmarked on AMD Ryzen 9 7950X3D (16 cores, 32 threads). Times exclude load/output.

GTEx bladder tissue (N = 77, 590-gene subset)

| Method | Rust (16 threads) | R (32 threads) | Speedup (R/Rust) |
| --- | --- | --- | --- |
| Pearson | 0.008s | 0.022s | 2.8x |
| Spearman | 0.008s | 0.030s | 3.8x |
| Kendall | 0.050s | 0.97s | 19.4x |
| Bicor | 0.004s | 0.020s | 5.0x |
| Hellcor | 61s* | 107s | 1.8x |

GTEx bladder tissue (N = 77, 2,950 genes)

| Method | Rust (16 threads) | R (32 threads) | Speedup (R/Rust) |
| --- | --- | --- | --- |
| Pearson | 0.16s | 0.33s | 2.1x |
| Spearman | 0.15s | 0.33s | 2.2x |
| Kendall | 1.7s | 22.6s | 13.3x |
| Bicor | 0.070s | 0.074s | 1.1x |
| Hellcor | 1575s | 3183s | 2.0x |
