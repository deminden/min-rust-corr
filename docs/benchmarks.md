# Benchmarks

Measured with `cargo run --release -p mincorr --example timing -- <method> <rows> <cols>`
(correlation only, no I/O, default portable build, no BLAS) on a **2-vCPU cloud VM
(Intel Xeon @ 2.10 GHz)**. The v0.2.1 figures are copied from the old README and were
measured on an **AMD Ryzen 9 7950X3D with 16 threads**, so the comparison is
conservative: the new numbers use 8× fewer threads on a slower core.

Run `cargo bench -p mincorr` (Criterion) or `tools/bench_compare.py` (Python,
end to end) to reproduce on your hardware.

## Current release (2 threads, Xeon 2.1 GHz)

| Method | 500 × 77 | 3000 × 77 | 2000 × 600 |
|---|---|---|---|
| pearson | 0.005 s | 0.055 s | 0.077 s |
| spearman | 0.003 s | 0.057 s | 0.092 s |
| bicor | 0.002 s | 0.062 s | 0.078 s |
| kendall | 0.013 s | 0.159 s | 1.983 s |
| hellcor | 2.701 s | — | — |

Hellcor, 2950 × 77 (the v0.2.1 benchmark size): **102.9 s** on 2 threads.

## Against v0.2.1 (GTEx bladder, 2950 genes × 77 samples)

| Method | v0.2.1, 16 threads (7950X3D) | v0.3.0, 2 threads (Xeon 2.1 GHz) | Thread-seconds ratio |
|---|---|---|---|
| Pearson | 0.16 s | 0.055 s (3000 × 77) | ≈ 23× |
| Spearman | 0.15 s | 0.057 s (3000 × 77) | ≈ 21× |
| Kendall | 1.7 s | 0.16 s (3000 × 77) | ≈ 85× |
| Bicor | 0.070 s | 0.062 s (3000 × 77) | ≈ 9× |
| Hellcor | 1575 s | 102.9 s (2950 × 77) | ≈ 122× |

Thread-seconds ratio = (v0.2.1 time × 16) / (v0.3.0 time × 2); it ignores the
faster cores of the 7950X3D, so per-core speed-ups are larger. Hellcor exceeds the
≥ 50× target; Kendall exceeds the ≥ 3× target.

## CLI end to end

10 000 × 77 input → 10 000 × 10 000 Pearson matrix written as `.tsv.gz`
(1.4 GB compressed): 12.5 s on 2 threads, **peak RSS 89 MB** (input 6 MB, one
48 MB block, the transformed matrix and I/O buffers). The whole matrix would be
800 MB as f64 and v0.2.1 additionally buffered the full text in memory.

## Kendall kernel (0.3.1)

Kendall's S per pair of rows, µs, single thread, n random ranks without ties
(same Xeon VM). "Bitset" is the pair-sign bitset kernel (one popcount per
word: S = |T| − 2·|T ∧ (g_x ⊕ g_y)|); "portable" uses `count_ones` without
hardware POPCNT, "POPCNT" the runtime-dispatched version used by default on
x86-64 CPUs that have it. Knight is the O(n log n) merge-sort algorithm.

| n | 0.3.0 bitset (4 popcounts/word) | bitset, portable | bitset, POPCNT | Knight |
|---|---|---|---|---|
| 77 | 0.146 | 0.038 | 0.016 | 2.83 |
| 300 | 2.05 | 0.546 | 0.245 | 13.7 |
| 600 | 8.32 | 2.50 | 0.961 | 28.9 |
| 1000 | 24.0 | 9.53 | 3.30 | 50.4 |
| 2000 | 99.2 | 36.7 | 22.0 | 105 |

The bitset kernel wins at every n measured, so it is used whenever the
bitsets of all rows fit in 512 MiB (two bitsets of n(n−1)/2 bits per row in
the worst case: e.g. ~12 000 rows at n = 600, ~500 rows at n = 2000); larger
inputs fall back to Knight. End to end, 2000 × 600 went from 33.2 s (0.3.0)
to 2.0 s on 2 threads.

## Remaining slow spot

Analytic p-values cost more than the matrix product itself for Pearson-type
methods (incomplete-beta evaluation per pair).
