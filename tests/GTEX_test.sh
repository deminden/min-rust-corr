#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# Generate normalized bladder tissue data without VST file
Rscript "$repo_root/examples/normalised_and_vst_counts_in_tissues.R" keyword=Bladder vst=FALSE

# Cut bladder file to 10000 lines for testing
echo "Cutting bladder file to 10000 lines for testing..."

bladder_file="$repo_root/data/Normalised_counts_tissues/Bladder_normalised_counts.tsv.gz"
filename=$(basename "$bladder_file" .tsv.gz)
output_file="$repo_root/data/${filename}_10k.tsv.gz"
set +o pipefail
zcat "$bladder_file" | head -n 10000 | gzip > "$output_file"
set -o pipefail
echo "Created: $output_file"

# Run all correlation types using Rust code
echo "Running Rust correlations on $output_file..."

cargo run --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" pearson --time
cargo run --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" spearman --time
cargo run --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" kendall --time

# Run investigation script to compare results
echo "Running R comparison analysis..."
Rscript "$script_dir/investigate_diffs.R" "$output_file"

echo "All correlation tests completed!"
