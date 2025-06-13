#!/bin/bash

# Process GTEx tissues: normalize data and compute correlations
# Usage: ./examples/process_GTEX_tissues.sh [keyword] [num_threads]

set -e

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
if [ -n "$KEYWORD" ]; then
    Rscript normalised_and_vst_counts_in_tissues.R keyword=$KEYWORD normalised=TRUE vst=TRUE
else
    Rscript normalised_and_vst_counts_in_tissues.R normalised=TRUE vst=TRUE
fi

# Step 2: Create output directory
OUTPUT_DIR="../outputs"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Step 3: Process each tissue's normalized and VST files
echo "Step 2: Computing correlations..."

# Find VST files for Pearson correlations
for vst_file in ../data/Normalised_counts_tissues_vst/*${KEYWORD}*_vst_counts.tsv.gz; do
    if [ -f "$vst_file" ]; then
        echo "Processing VST file for Pearson: $(basename "$vst_file")"
        
        # Run Pearson correlation on VST data
        if [ -n "$THREADS" ]; then
            cargo run --release -- "$vst_file" pearson $THREADS --time
        else
            cargo run --release -- "$vst_file" pearson --time
        fi
        
        # Move output to outputs folder
        basename_file=$(basename "$vst_file" .tsv.gz)
        mv "${basename_file}_pearson_correlations.tar.gz" "$OUTPUT_DIR/" 2>/dev/null || true
    fi
done

# Find normalized files for Spearman and Kendall correlations  
for norm_file in ../data/Normalised_counts_tissues/*${KEYWORD}*_normalised_counts.tsv.gz; do
    if [ -f "$norm_file" ]; then
        echo "Processing normalized file for Spearman/Kendall: $(basename "$norm_file")"
        
        # Run Spearman correlation on normalized data
        if [ -n "$THREADS" ]; then
            cargo run --release -- "$norm_file" spearman $THREADS --time
            cargo run --release -- "$norm_file" kendall $THREADS --time
        else
            cargo run --release -- "$norm_file" spearman --time
            cargo run --release -- "$norm_file" kendall --time
        fi
        
        # Move outputs to outputs folder
        basename_file=$(basename "$norm_file" .tsv.gz)
        mv "${basename_file}_spearman_correlations.tar.gz" "$OUTPUT_DIR/" 2>/dev/null || true
        mv "${basename_file}_kendall_correlations.tar.gz" "$OUTPUT_DIR/" 2>/dev/null || true
    fi
done

echo "GTEx tissue processing completed!"
echo "Results saved in: $OUTPUT_DIR"