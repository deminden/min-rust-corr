# Parked ideas (out of scope for v0.3.0)

- Distance correlation and partial correlation.
- FDR adjustment of p-value matrices (Benjamini–Hochberg on the upper triangle).
- Sparse top-k output for co-expression networks directly from the engine
  (avoid materialising dense blocks).
- f32 storage of outputs and f32 Gram pipelines.
- GPU Gram kernel.
- Uniform-grid nearest-neighbour search for hellcor at very large n (the
  current brute force is O(n²) per pair, ideal for n ≲ 2000).
- Faster analytic p-values for large blocks (the incomplete-beta continued
  fraction dominates Pearson runtime when p-values are requested).
- Cache-blocked (row-tile × row-tile) Kendall bitset kernel.
