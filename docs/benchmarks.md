# Benchmarks

Measured with `cargo run --release -p mincorr --example timing -- <method> <rows> <cols>`
(correlation only, no I/O, default portable build, no BLAS) on a **2-vCPU cloud VM
(Intel Xeon @ 2.10 GHz)**. The v0.2.1 figures are copied from the old README and were
measured on an **AMD Ryzen 9 7950X3D with 16 threads**, so the comparison is
conservative: the new numbers use 8× fewer threads on a slower core.

Run `cargo bench -p mincorr` (Criterion) or `tools/bench_compare.py` (Python,
end to end) to reproduce on your hardware.

## v0.3.0 (2 threads, Xeon 2.1 GHz)

| Method | 500 × 77 | 3000 × 77 | 2000 × 600 |
|---|---|---|---|
| pearson | 0.005 s | 0.055 s | 0.077 s |
| spearman | 0.003 s | 0.057 s | 0.092 s |
| bicor | 0.002 s | 0.062 s | 0.078 s |
| kendall | 0.013 s | 0.416 s | 33.174 s |
| hellcor | 2.701 s | — | — |

Hellcor, 2950 × 77 (the v0.2.1 benchmark size): **102.9 s** on 2 threads.

## Against v0.2.1 (GTEx bladder, 2950 genes × 77 samples)

| Method | v0.2.1, 16 threads (7950X3D) | v0.3.0, 2 threads (Xeon 2.1 GHz) | Thread-seconds ratio |
|---|---|---|---|
| Pearson | 0.16 s | 0.055 s (3000 × 77) | ≈ 23× |
| Spearman | 0.15 s | 0.057 s (3000 × 77) | ≈ 21× |
| Kendall | 1.7 s | 0.42 s (3000 × 77) | ≈ 33× |
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

## Known slow spot

Kendall with n > 128 observations uses Knight's merge sort per pair
(2000 × 600: 33 s on 2 threads). Bitsets are limited to n ≤ 128 to bound memory;
a POPCNT-dispatched bitset path for larger n is listed in docs/ideas.md.
Analytic p-values cost more than the Gram product itself for Pearson-type
methods (incomplete-beta evaluation per pair).
