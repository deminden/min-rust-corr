# Parked ideas

- Permutation p-values for ρp and distance correlation (the hellcor
  Monte-Carlo machinery could be reused).
- Partial correlation with per-pair missing values (refit the covariate basis
  on each pair's complete observations).
- Merge-sort instead of a Fenwick tree for the dcor pair pass.
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
