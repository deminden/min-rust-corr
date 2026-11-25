"""mincorr: fast correlation matrices computed in Rust.

Rows are variables (e.g. genes), columns are observations (samples).

High-level API::

    import mincorr
    r = mincorr.corr(x, method="spearman")
    r, p = mincorr.corr(x, method="kendall", pvalues=True)

All v0.2 function names (``pearson_matrix``, ``kendall_upper_triangle``,
``hellcor_cross_matrix_with_pvalues``, ``bicor_pair``, ...) remain available
as thin wrappers over :func:`corr` and :func:`pair`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core

__version__ = _core.__version__

METHODS = ("pearson", "spearman", "kendall", "bicor", "hellcor")
_DEFAULTS = dict(alpha=6.0, kmax=20, lmax=20, max_p_outliers=1.0, mc_samples=10_000, seed=42)


def _as_f64(a: Any, ndim: int, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)  # copies only when dtype requires it
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    return arr


def _params(params: dict) -> dict:
    unknown = set(params) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
    return {**_DEFAULTS, **params}


def corr(data, other=None, method="pearson", layout="full", pvalues=False, nan_policy="pairwise", **params):
    """Correlation matrix of the rows of ``data`` (or ``data`` × ``other``).

    layout: "full" (square), "cross" (requires ``other``) or "upper"
    (packed upper triangle incl. diagonal, length n(n+1)/2).
    pvalues: also return p-values (Monte Carlo for hellcor).
    nan_policy: "pairwise" (default), "propagate" or "error".
    params: alpha, kmax, lmax (hellcor); max_p_outliers (bicor);
    mc_samples, seed (hellcor p-values).
    Returns the matrix, or ``(matrix, pvalues)`` when ``pvalues`` is true.
    """
    a = _as_f64(data, 2, "data")
    b = None if other is None else _as_f64(other, 2, "other")
    if layout == "cross" and b is None:
        raise ValueError("layout='cross' requires `other`")
    p = _params(params)
    r, pv = _core.compute(a, b, method, layout, bool(pvalues), nan_policy, p["alpha"], p["kmax"], p["lmax"],
                          p["max_p_outliers"], p["mc_samples"], p["seed"])
    return (r, pv) if pvalues else r


def pair(x, y, method="pearson", pvalues=False, nan_policy="pairwise", **params):
    """Correlation of two vectors; ``(r, p)`` when ``pvalues`` is true."""
    xa, ya = _as_f64(x, 1, "x"), _as_f64(y, 1, "y")
    p = _params(params)
    r, pv, _ = _core.pair(np.ascontiguousarray(xa), np.ascontiguousarray(ya), method, bool(pvalues), nan_policy,
                          p["alpha"], p["kmax"], p["lmax"], p["max_p_outliers"], p["mc_samples"], p["seed"])
    return (r, pv) if pvalues else r


# ----------------------------------------------------------- v0.2 names --
_HELL = "hellcor"


def _legacy(method: str, kind: str):
    hell = method == _HELL

    if kind == "pair":
        def f(x, y, alpha=6.0):
            return pair(x, y, method, alpha=alpha) if hell else pair(x, y, method)
    elif kind == "matrix":
        def f(data, alpha=6.0):
            return corr(data, method=method, **({"alpha": alpha} if hell else {}))
    elif kind == "upper_triangle":
        def f(data, alpha=6.0):
            return corr(data, method=method, layout="upper", **({"alpha": alpha} if hell else {}))
    elif kind == "cross_matrix":
        def f(data_a, data_b, alpha=6.0):
            return corr(data_a, data_b, method=method, layout="cross", **({"alpha": alpha} if hell else {}))
    elif kind == "matrix_with_pvalues":
        def f(data, alpha=6.0, mc_samples=10_000, seed=42):
            kw = {"alpha": alpha, "mc_samples": mc_samples, "seed": seed} if hell else {}
            return corr(data, method=method, pvalues=True, **kw)
    else:  # cross_matrix_with_pvalues
        def f(data_a, data_b, alpha=6.0, mc_samples=10_000, seed=42):
            kw = {"alpha": alpha, "mc_samples": mc_samples, "seed": seed} if hell else {}
            return corr(data_a, data_b, method=method, layout="cross", pvalues=True, **kw)

    f.__name__ = f"{method}_{kind}"
    f.__doc__ = f"v0.2 compatibility wrapper: {method} {kind.replace('_', ' ')}."
    return f


_KINDS = ("pair", "matrix", "matrix_with_pvalues", "cross_matrix", "cross_matrix_with_pvalues", "upper_triangle")
LEGACY_NAMES = [f"{m}_{k}" for k in _KINDS for m in METHODS]
for _name in LEGACY_NAMES:
    _m, _k = _name.split("_", 1)
    globals()[_name] = _legacy(_m, _k)

__all__ = ["corr", "pair", "METHODS", "LEGACY_NAMES", "__version__", *LEGACY_NAMES]
