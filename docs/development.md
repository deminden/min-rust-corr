# Development

## Repository layout

| Path | Contents |
|---|---|
| `src/` | library: `api` (Correlator), `engine` (Gram / pairwise engines), `methods`, `layout`, `prep`, `stats`, `options`, `compat` (v0.2 names) |
| `src/bin/mincorr/` | command-line tool (`cli` feature) |
| `python/` | Python package `mincorr` on PyPI (PyO3 + maturin; Rust crate `mincorr-py`, not published) |
| `tests/` | integration tests, including R parity fixtures in `tests/fixtures/` |
| `benches/`, `examples/timing.rs` | Criterion benchmarks and a quick timing example |
| `docs/` | semantics, Hellinger correlation notes, parity, benchmarks, releasing |
| `tools/` | fixture generator (R), file-size check, Python benchmark |
| `examples/gtex/` | GTEx tissue-processing pipeline |

## Checks (what CI runs)

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude mincorr-py
cargo test --lib --tests --no-default-features      # library without the CLI
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
tools/check_file_sizes.sh
cd python && maturin develop --release && pytest -q tests
```

The library and CLI support Rust 1.85 (MSRV); running the test suite needs
Rust ≥ 1.88 because of the test-only dependencies (proptest, criterion).

`unsafe` is denied crate-wide (`unsafe_code = "deny"`); the only exception is
the runtime POPCNT dispatch in `src/methods/kendall/popcount.rs`, kept to a
few documented lines because it measurably doubles the Kendall kernel's speed.

Source files stay under 300 lines (hard limit 400) and functions under 60 lines
(`clippy::too_many_lines`, threshold in `clippy.toml`).

## Reference fixtures

`tools/make_fixtures.R` regenerates `tests/fixtures/` with R, WGCNA and HellCor
(used strictly as black boxes; see PROVENANCE.md):

```bash
Rscript tools/make_fixtures.R          # writes tests/fixtures/ and MANIFEST.json
cargo test --test parity_rank_methods --test parity_robust --test beta_fixtures
```

## Releasing

See [releasing.md](releasing.md).
