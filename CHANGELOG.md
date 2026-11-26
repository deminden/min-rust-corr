# Changelog

## 0.3.0 — 2026-11-26

Complete rewrite under the MIT licence. bicor, hellcor and the Beta-distribution
functions are new independent implementations written from the published
literature (see PROVENANCE.md); Pearson, Spearman and Kendall are restructured
from the author's own code.

### Added

- `Correlator` builder API: `full`, `cross`, `upper`, `pair`, streaming
  `full_blocks` / `cross_blocks` (optional `triangle_cache`), `NanPolicy`
  (also parsed from strings), `PvalueSpec::auto`, `Method::with_settings`.
- `CorrError` / `Result` everywhere: no panics on user data.
- CLI: `-o/--output`, `--format tsv|tsv.gz|npy|edges`, `--legacy-tar`,
  `--nan-policy`, `--delimiter`, `--na-strings`, `--sort-rows`, `--digits`,
  `--block-rows`, `--threads`, `--quiet`, `--memory-limit-gb`, hellcor
  `--alpha/--kmax/--lmax`, bicor `--max-p-outliers`, edge filters
  `--min-abs-r/--max-p/--top-k`; CSV input; streaming output with bounded memory.
- Python: `mincorr.corr()`, `mincorr.pair()`, `mincorr.frames.corr_frame()`
  (pandas), type stubs; accepts float32, Fortran-ordered and strided arrays.
- `HellcorCriterion` (normalised leave-one-out by default, matching HellCor; the paper's MISE criterion as an option).
- Reference fixtures from R, property tests, CI on three OSes, cargo-deny.
- Release workflow: prebuilt `mincorr` binaries for Linux (static), macOS and
  Windows and Python wheels attached to GitHub Releases; automated crates.io
  and (optional) PyPI publishing.

### Changed (breaking)

- Licence: GPL-2.0-or-later → MIT.
- CLI row order now follows the input file (`--sort-rows` restores v0.2's
  alphabetical order); random subsets keep input order too.
- CLI default output is `<base>_<method>_correlations.tsv.gz` (+ `_pvalues`)
  instead of a single-file `.tar.gz`; use `--legacy-tar` for the old layout.
  Gzip output is written as concatenated gzip members (read transparently by
  gzip/zcat, R, pandas and Python's gzip module).
- Messages go to stderr; stdout carries only requested data.
- One crate: the library and the `mincorr` binary ship together in `mincorr`
  (binary behind the default `cli` feature); the separate `mincorr-cli`
  package and binary name are gone.
- Unparsable input cells are errors (with line and column) instead of silent
  NaN; ragged rows and duplicate IDs are errors.
- Missing values: one policy for all methods, default pairwise-complete
  (v0.2 differed per method and Spearman could panic). P-values use the
  per-pair effective sample size.
- Diagonal: exactly 1 for valid rows and NaN for degenerate rows in every
  layout; diagonal p-values are NaN (v0.2: 0.0).
- Hellcor Monte-Carlo p-values use one ChaCha8 stream per replicate; they are
  not bit-reproducible against v0.2. `--subset-size` sampling uses a different
  generator, so the selected rows differ from v0.2 for the same seed.
- v0.2 Rust function names (`mincorr::pearson::correlation_matrix`, …) are
  deprecated wrappers returning NaN-filled outputs on invalid input.
- Python ≥ 3.9 (abi3 wheels).

### Removed

- `kendalls` and `statrs` dependencies; `.cargo/config.toml` (no more forced
  `target-cpu=native`); debug-file output and `HELLCOR_DEBUG_PAIR`.

### Performance

- Kendall: pair-sign bitsets (n ≤ 128) / allocation-free Knight merge sort.
- Hellcor: O(n² + n·K·L) per pair with per-n tables (was O(n²·K·L + n·K²·L²)).
- Gram methods: parallel blocked matrix products, upper triangle only.
