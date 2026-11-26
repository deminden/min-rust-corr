# Releasing

Crates (publish in this order; `mincorr-cli` depends on `mincorr`):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo publish --dry-run -p mincorr
cargo login                      # once, with a crates.io API token
cargo publish -p mincorr
cargo publish -p mincorr-cli     # after mincorr is visible in the index
git tag v0.3.0 && git push origin v0.3.0
```

Python wheels are built by `.github/workflows/wheels.yml` on `v*` tags
(abi3, Linux/macOS/Windows); upload them to PyPI as `mincorrpy` with
`twine upload` or add a trusted-publishing step.

`mincorr-py` (the Rust side of the Python package) is `publish = false`.
