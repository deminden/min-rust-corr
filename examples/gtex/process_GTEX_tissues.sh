#!/bin/bash

# Process GTEx tissues: normalize data and compute correlations
# Usage: examples/gtex/process_GTEX_tissues.sh [keyword] [num_threads]
# Data go to examples/gtex/data/, correlation matrices (.tsv.gz) to examples/gtex/outputs/.

set -e
cd "$(dirname "$0")"

# Parse arguments
KEYWORD=${1:-""}         # Default to all tissues
THREADS=${2:-""}         # Default to all cores

if [ -n "$KEYWORD" ]; then
    echo "Processing GTEx tissues with keyword: $KEYWORD"
else
    echo "Processing all GTEx tissues"
fi

# Step 1: Generate normalized and VST counts
echo "Step 1: Generating normalized and VST counts..."
Rscript normalised_and_vst_counts_in_tissues.R ${KEYWORD:+keyword="$KEYWORD"} normalised=TRUE vst=TRUE

# Step 2: Create output directory
OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Step 3: Process each tissue's normalized and VST files
echo "Step 2: Computing correlations..."

# mincorr <input> <method>: correlations of one tissue file into $OUTPUT_DIR
mincorr() {
    local out="$OUTPUT_DIR/$(basename "$1" .tsv.gz)_$2_correlations.tsv.gz"
    cargo run --release --quiet --manifest-path ../../Cargo.toml --bin mincorr -- \
        "$1" "$2" ${THREADS:+--threads "$THREADS"} --time -o "$out"
}

# Pearson on VST data; Spearman and Kendall on normalised counts
for vst_file in data/Normalised_counts_tissues_vst/*${KEYWORD}*_vst_counts.tsv.gz; do
    [ -f "$vst_file" ] || continue
    echo "Processing VST file for Pearson: $(basename "$vst_file")"
    mincorr "$vst_file" pearson
done

for norm_file in data/Normalised_counts_tissues/*${KEYWORD}*_normalised_counts.tsv.gz; do
    [ -f "$norm_file" ] || continue
    echo "Processing normalised file for Spearman/Kendall: $(basename "$norm_file")"
    mincorr "$norm_file" spearman
    mincorr "$norm_file" kendall
done

echo "GTEx tissue processing completed!"
echo "Results saved in: $OUTPUT_DIR"