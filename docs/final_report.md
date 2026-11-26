# v0.3.0 rewrite — final report

Branch `mit-rewrite`, starting from commit 83d9ee4 (tagged locally as
`v0.2.1-gpl-final`). Every commit is authored by Denis Demin.

## What was ported vs rewritten

| Area | Status |
|---|---|
| Pearson, Spearman, p-value helpers, packed-upper idea, CLI subset logic, v0.2 API names | Ported from author-original code and restructured |
| Kendall tau-b | Rewritten (Knight 1966 + pair-sign bitsets); `kendalls` crate dropped |
| bicor | Clean-room from Wilcox (2012) / Langfelder & Horvath (2012) / WGCNA docs |
| hellcor | Clean-room from Geenens & Lafaye de Micheaux (2022) |
| Beta pdf / cdf / quantile, log-gamma, incomplete beta, erfc | Clean-room from DLMF; `statrs` dropped |
| Engines, layouts, streaming, error handling, CLI, Python bindings | New |

Quarantined GPL-derived files were removed by path in the first commit without
being opened; `git ls-files` confirms none remain. See PROVENANCE.md.

## Fixture parity

See docs/parity.md (fixtures from R 4.6.2, WGCNA 1.74, HellCor 1.3, generated on
the owner's Mac in a reusable conda environment `r462`). All methods agree with
R on every fixture: Pearson / Spearman / Kendall r ≤ 5.6e-16 and p ≤ 2.2e-12
relative; bicor r ≤ 2.2e-16, p ≤ 4.7e-13, `maxPOutliers` 0.1 / 0.05 ≤ 8.9e-16;
hellcor η ≤ 2.3e-12 for α = 3, 6, 10. Beta functions ≤ 5.7e-14 relative. Python
matches numpy/scipy.

## Benchmarks

See docs/benchmarks.md (2-vCPU Xeon vs the old 16-thread 7950X3D figures).
Hellcor 2950 × 77: 102.9 s on 2 threads vs 1575 s on 16 threads (≈ 122× in
thread-seconds; target ≥ 50× met). Kendall ≈ 85× (0.3.1), Pearson/Spearman ≈ 21–23×,
bicor ≈ 9× in thread-seconds. CLI: 10 000 × 10 000 Pearson to .tsv.gz with 89 MB
peak RSS.

## Remaining differences vs R

- Hellcor on a constant row: NaN here (degenerate-row rule), ≈ 1 in HellCor.
- Monte-Carlo p-values differ from v0.2 (new random streams), by design.
- The hellcor default criterion is the normalised leave-one-out one (matches
  HellCor); the paper's MISE criterion as printed is available as an option.

## Deviations from the specification

1. `full()` is implemented directly (triangle + mirror) rather than on top of
   `full_blocks()`; streaming computes complete rows, so it evaluates symmetric
   pairs twice unless `Correlator::triangle_cache` allows computing the triangle
   in memory (the CLI sets it from `--memory-limit-gb`). Both paths share the
   same engine and layout executor.
2. Hellcor nearest neighbours are brute force for all n (no uniform grid yet);
   fine for the target regime (n ≲ 2000), parked in docs/ideas.md.
3. Kendall uses pair-sign bitsets whenever they fit a 512 MiB budget and
   Knight's algorithm otherwise (0.3.1; 2000 × 600: 2.0 s on 2 threads).
   The one `unsafe` block in the crate dispatches the bitset kernel to the
   hardware POPCNT instruction at runtime (2–3× measured gain, above the
   specification's 20 % threshold), isolated in `methods/kendall/popcount.rs`.
4. `rand` is not a dependency; `rand_chacha` (+ its `rand_core`) is used
   directly with an own unbiased bounded-integer routine.
5. Python requires ≥ 3.9 (abi3-py39) instead of 3.8.
6. Function-length cap is enforced with clippy `too_many_lines` (threshold 60)
   rather than a script; the size script checks file lengths.
7. MSRV is declared as 1.85 (edition 2024) and tested in CI; it could not be
   verified locally (toolchain download blocked).
8. `cargo deny` runs in CI; locally the dependency licences were checked from
   `cargo metadata` (all MIT/Apache/BSD/ISC/Zlib/Unicode options available).
9. The legacy GTEx comparison scripts (`GTEx_test.sh`, `investigate_diffs.R`,
   `pvalue_parity_check.R`, `qbeta_ref.R`) were removed in favour of
   `tools/make_fixtures.R` and the parity tests; `examples/gtex/` keeps the GTEx
   processing pipeline, updated for the new CLI.
10. The library and the CLI are one crate (`mincorr`, binary behind the default
    `cli` feature) instead of separate `mincorr` / `mincorr-cli` crates.

## Open decisions for the owner (§9 defaults used)

- D1 history: the branch keeps the old GPL history; publish the MIT version as
  fresh history/new repo, or keep history with the README note (current).
  MIT only (not dual MIT/Apache) — current.
- D2 diagonal p-values NaN — current.
- D3 default NaN policy pairwise-complete — current.
- D4: one crate `mincorr` (library + binary); PyPI `mincorr` (renamed from `mincorrpy`; Rust crate `mincorr-py`,
  module `mincorr`); nothing published.
- D5 CLI default `.tsv.gz`; D6 input row order — current.
- D7 statrs removed (own special functions pass all accuracy tests).
- D8 resolved: bicor maxPOutliers matches WGCNA exactly.
- D9 resolved: hellcor default criterion = normalised LOO (matches HellCor).
- Version is 0.3.0; crates prepared for crates.io (not published).

## Provenance log

See PROVENANCE.md (per-file origin table and dated reference log).
