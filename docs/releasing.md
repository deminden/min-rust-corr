# Releasing

`mincorr` is a single crate (library + `mincorr` binary behind the default
`cli` feature). `mincorr-py` (Rust side of the Python package) has
`publish = false`.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --no-default-features          # library without the CLI
cargo publish --dry-run
cargo login                               # once, with a crates.io API token
cargo publish
git tag -a v0.3.0 -m "mincorr 0.3.0" && git push origin v0.3.0
```

Python wheels are built by `.github/workflows/wheels.yml` on `v*` tags
(abi3, Linux/macOS/Windows); upload them to PyPI as `mincorrpy` with
`twine upload`, or add a trusted-publishing step.
