# Provenance

This file records, for every source file on the `mit-rewrite` branch, where its
content came from, plus a dated log of the references consulted while writing
the independent (clean-room) modules.

## File classification at the start of the rewrite (v0.2.1, commit 83d9ee4)

| Path | Status | Action |
|---|---|---|
| crates/core/src/{pearson,spearman,kendall,rank,pvalues,upper,lib}.rs | Author-original | Reused / restructured |
| crates/cli/src/main.rs | Author-original | Reused / split |
| crates/py/src/lib.rs, crates/py/pyproject.toml | Author-original | Reused / restructured |
| crates/core/tests/synthetic.rs, scripts/bench_upper.py, examples/*, .github/* | Author-original | Reused |
| crates/core/tests/{investigate_diffs,pvalue_parity_check,qbeta_ref}.R | Author-original (call R packages only) | Basis for fixture generator |
| crates/core/src/bicor.rs | Derived from GPL code (WGCNA port) | Removed unopened; rewritten from published papers |
| crates/core/src/hellcor.rs | Derived from GPL code (HellCor port) | Removed unopened; rewritten from published paper |
| crates/core/src/hellinger/ (6 files) | Derived from GPL code (R nmath port) | Removed unopened; replaced by clean-room Beta functions |
| crates/core/tests/hellcor_compare.R | Re-implements HellCor internals | Removed unopened |
| crates/core/tests/qbeta_against_r.rs, docs/qbeta.md | Tied to the nmath port | Removed unopened |

The quarantined paths were removed with `git rm` by path, without opening,
viewing, diffing or grepping them at any commit.

## Per-file origin (current tree)

(Updated as files are added; see below.)

## Reference log

- 2026-11-24: Phase 0 — quarantine performed. No references consulted.
