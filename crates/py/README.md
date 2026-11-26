# mincorrpy

Python bindings of [mincorr](https://github.com/deminden/min-rust-corr): fast
correlation matrices (Pearson, Spearman, Kendall tau-b, biweight
midcorrelation, Hellinger correlation) computed in Rust. Rows are variables,
columns are observations.

```bash
pip install mincorrpy
```

```python
import mincorr
r = mincorr.corr(x, method="spearman")                    # (n, n) matrix
r, p = mincorr.corr(x, method="kendall", pvalues=True)    # with p-values
u = mincorr.corr(x, method="bicor", layout="upper")      # packed upper triangle
c = mincorr.corr(a, b, method="hellcor", layout="cross") # rows of a × rows of b
r = mincorr.pair(x[0], x[1], method="pearson")
```

Missing values: `nan_policy="pairwise"` (default), `"propagate"` or `"error"`.
All v0.2 function names (`pearson_matrix`, `kendall_upper_triangle`, …) remain
available. Optional pandas helper: `mincorr.frames.corr_frame(df)`.

Documentation, semantics and benchmarks:
<https://github.com/deminden/min-rust-corr>. Licence: MIT.
