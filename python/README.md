# mincorr (Python)

[![PyPI](https://img.shields.io/pypi/v/mincorr.svg)](https://pypi.org/project/mincorr/)
[![Python](https://img.shields.io/pypi/pyversions/mincorr.svg)](https://pypi.org/project/mincorr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/deminden/min-rust-corr/blob/main/LICENSE)

Fast correlation matrices for wide data, computed in Rust: **Pearson,
Spearman, Kendall tau-b, biweight midcorrelation (bicor) and the Hellinger
correlation**, with p-values and consistent missing-value handling. Rows are
variables (genes, features), columns are observations (samples). Results match
R's `stats::cor` / `cor.test`, `WGCNA::bicor` and `HellCor::HellCor`.

```bash
pip install mincorr            # wheels for Linux, macOS and Windows (Python ≥ 3.9)
```

```python
import numpy as np
import mincorr

x = np.random.default_rng(0).normal(size=(100, 40))       # 100 genes × 40 samples

r = mincorr.corr(x, method="spearman")                     # (100, 100) matrix
r, p = mincorr.corr(x, method="kendall", pvalues=True)    # with p-values
u = mincorr.corr(x, method="bicor", layout="upper")       # packed upper triangle
c = mincorr.corr(x[:5], x, method="hellcor", layout="cross")  # 5 × 100
r01 = mincorr.pair(x[0], x[1], method="pearson")
```

| Argument | Values |
|---|---|
| `method` | `"pearson"`, `"spearman"`, `"kendall"`, `"bicor"`, `"hellcor"` |
| `layout` | `"full"` (default), `"cross"` (needs a second matrix), `"upper"` |
| `pvalues` | `False` (default) / `True` — Monte-Carlo null for hellcor |
| `nan_policy` | `"pairwise"` (default), `"propagate"`, `"error"` |
| method parameters | `alpha`, `kmax`, `lmax` (hellcor); `max_p_outliers` (bicor); `mc_samples`, `seed` (hellcor p-values) |

Inputs may be any NumPy-compatible float array (float32 or float64, C- or
Fortran-ordered). Labelled pandas output: `mincorr.frames.corr_frame(df)`
(`pip install "mincorr[pandas]"`). All v0.2 function names
(`pearson_matrix`, `kendall_upper_triangle`, …) remain available.

Also available as a Rust crate and a command-line tool (`cargo install mincorr`,
or prebuilt binaries from the GitHub releases). Documentation, semantics and
benchmarks: <https://github.com/deminden/min-rust-corr>. Licence: MIT.
