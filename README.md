# High-Performance Correlation Analysis

Minimalistic Rust implementation for computing Pearson, Spearman, Kendall, and biweight midcorrelations (bicor) on large data matrices using BLAS optimizations.
Results match R implementations.

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
```

### As a Crate

Add to `Cargo.toml`:
```toml
[dependencies]
mincorr = { git = "https://github.com/deminden/min-rust-corr" }
```

Use in code:
```rust
use mincorr::{pearson_correlation_matrix, spearman_correlation_matrix, kendall_correlation_matrix, bicor_correlation_matrix};
use ndarray::Array2;

let data: Array2<f64> = /* your data matrix */;

// Compute correlations
let pearson_corr = pearson_correlation_matrix(&data);
let spearman_corr = spearman_correlation_matrix(&data);
let kendall_corr = kendall_correlation_matrix(&data);
let bicor_corr = bicor_correlation_matrix(&data);
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
# Quick test with 10k genes subset
./tests/GTEX_test.sh

# Compare results with R implementations  
Rscript tests/investigate_diffs.R data/your_file.tsv.gz
```


## Output

Creates `<input_basename>_<method>_correlations.tar.gz` containing correlation matrix in TSV format. Results are saved in the current directory or `outputs/` folder when using the GTEx processing script. 
