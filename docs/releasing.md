# Releasing

Releases are automated by `.github/workflows/release.yml`, started by pushing a
version tag. `mincorr` is a single crate (library + `mincorr` binary behind the
default `cli` feature); the Rust side of the Python package (`mincorr-py`) is
never published to crates.io; the Python distribution on PyPI is `mincorr`.

## One-time setup on GitHub

1. Make the MIT code the default branch (`main`): crates.io resolves the
   README's relative links against the repository's default branch.
2. Settings → Secrets and variables → Actions → **secret**
   `CARGO_REGISTRY_TOKEN`: a crates.io API token with the `publish-new` and
   `publish-update` scopes. Optionally restrict it to an environment named
   `crates-io` (the publish job runs in that environment).
3. PyPI (trusted publishing, no token): on pypi.org → Your account →
   Publishing → *Add a new pending publisher* (GitHub tab) with
   PyPI project name `mincorr`, owner `deminden`, repository `min-rust-corr`,
   workflow `release.yml`, environment `pypi`. The first release then creates
   the project. Optionally create GitHub environments `pypi` and `crates-io`
   (Settings → Environments) with required reviewers to approve each upload.

## Cutting a release

```bash
# 1. bump `version` in Cargo.toml ([workspace.package]) and python/pyproject.toml
# 2. add a "## X.Y.Z — YYYY-MM-DD" section at the top of CHANGELOG.md
cargo test --workspace --exclude mincorr-py && cargo publish --dry-run
git commit -am "chore(release): X.Y.Z"
git tag -a vX.Y.Z -m "mincorr X.Y.Z"
git push origin main vX.Y.Z
```

The workflow then:

1. **verify** — checks that the tag equals the crate version, runs fmt,
   clippy, the tests and `cargo publish --dry-run`;
2. **build** — static Linux binaries (x86_64, aarch64; musl), macOS (Apple
   silicon, Intel), Windows (x86_64, ARM64); abi3 Python wheels (one per
   platform, Python ≥ 3.9) for manylinux x86_64/aarch64, musllinux x86_64,
   macOS arm64/x86_64 and Windows x86_64, each tested with pytest where the
   runner can execute it, and the sdist, tested by building from source;
3. **release** — creates the GitHub Release (shown under *Releases* on the
   repository page) with the archives, wheels and `SHA256SUMS`, using the
   version's CHANGELOG section as release notes;
4. **publish** — `cargo publish` to crates.io (skipped if that version is
   already there) and the wheels and sdist to PyPI (`skip-existing`).

Archives are named `mincorr-vX.Y.Z-<target>.tar.gz` (`.zip` on Windows) and
contain the `mincorr` binary, README, LICENSE and CHANGELOG.
