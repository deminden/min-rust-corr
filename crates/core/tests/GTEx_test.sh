#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

# Generate normalized bladder tissue data without VST file
Rscript "$repo_root/examples/normalised_and_vst_counts_in_tissues.R" keyword=Bladder vst=FALSE

# Sample 500 random genes for testing
echo "Sampling 500 random genes for testing..."

bladder_file="$repo_root/data/Normalised_counts_tissues/Bladder_normalised_counts.tsv.gz"
filename=$(basename "$bladder_file" .tsv.gz)
output_file="$repo_root/data/${filename}_500.tsv.gz"
python - <<'PY'
import gzip
import random
import sys
from pathlib import Path

bladder_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
random.seed(42)

with gzip.open(bladder_file, "rt") as f:
    header = f.readline()
    rows = f.readlines()

if len(rows) < 500:
    raise SystemExit(f"Not enough rows to sample 500 (found {len(rows)})")

sampled = random.sample(rows, 500)
with gzip.open(output_file, "wt") as out:
    out.write(header)
    out.writelines(sampled)
PY
"$bladder_file" "$output_file"
echo "Created: $output_file"

# Run all correlation types using Rust code
echo "Running Rust correlations on $output_file..."

cargo run -p mincorr_cli --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" pearson --time
cargo run -p mincorr_cli --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" spearman --time
cargo run -p mincorr_cli --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" kendall --time
cargo run -p mincorr_cli --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" bicor --time
cargo run -p mincorr_cli --release --manifest-path "$repo_root/Cargo.toml" -- "$output_file" hellcor --time

# Run investigation script to compare results
echo "Running R comparison analysis..."
Rscript "$script_dir/investigate_diffs.R" "$output_file"

echo "All correlation tests completed!"
