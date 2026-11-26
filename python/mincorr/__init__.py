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

import inspect
from typing import Any

import numpy as np

from . import _core

__version__ = _core.__version__

METHODS = ("pearson", "spearman", "kendall", "bicor", "hellcor", "gaussian_rank", "rho", "xi", "dcor")
_LEGACY_METHODS = METHODS[:5]
_DEFAULTS = dict(alpha=6.0, kmax=20, lmax=20, max_p_outliers=1.0, pseudocount=0.0, clr=True, symmetric=False,
                 mc_samples=10_000, seed=42)


def _as_f64(a: Any, ndim: int, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)  # copies only when dtype requires it
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    return arr


def _options(method: str, pvalues: bool, nan_policy: str, params: dict) -> dict:
    unknown = set(params) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
    return {**_DEFAULTS, **params, "method": method, "pvalues": bool(pvalues), "nan_policy": nan_policy}


def _cov(covariates, n_obs):
    if covariates is None:
        return None
    z = np.asarray(covariates, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    if z.ndim != 2:
        raise ValueError(f"covariates must be 1- or 2-dimensional, got shape {z.shape}")
    return z


def corr(data, other=None, method="pearson", layout="full", pvalues=False, nan_policy="pairwise",
         covariates=None, **params):
    """Correlation matrix of the rows of ``data`` (or ``data`` × ``other``).

    method: pearson, spearman, kendall, bicor, hellcor, gaussian_rank,
    rho (proportionality), xi (Chatterjee) or dcor (distance correlation).
    layout: "full" (square), "cross" (requires ``other``) or "upper"
    (packed upper triangle incl. diagonal, length n(n+1)/2; symmetric
    methods only).
    pvalues: also return p-values (Monte Carlo for hellcor; not available
    for rho and dcor).
    nan_policy: "pairwise" (default), "propagate" or "error".
    covariates: array (k × observations, or one vector) to adjust for
    (partial correlation; pearson, spearman and gaussian_rank).
    params: alpha, kmax, lmax (hellcor); max_p_outliers (bicor);
    pseudocount, clr (rho); symmetric (xi); mc_samples, seed (hellcor
    p-values).
    Returns the matrix, or ``(matrix, pvalues)`` when ``pvalues`` is true.
    """
    a = _as_f64(data, 2, "data")
    b = None if other is None else _as_f64(other, 2, "other")
    if layout == "cross" and b is None:
        raise ValueError("layout='cross' requires `other`")
    opts = _options(method, pvalues, nan_policy, params)
    r, pv = _core.compute(a, b, layout, opts, _cov(covariates, a.shape[1]))
    return (r, pv) if pvalues else r


def pair(x, y, method="pearson", pvalues=False, nan_policy="pairwise", covariates=None, **params):
    """Correlation of two vectors; ``(r, p)`` when ``pvalues`` is true.

    For ``xi`` this is ξ(x → y): how well y is a function of x.
    """
    xa, ya = _as_f64(x, 1, "x"), _as_f64(y, 1, "y")
    r, pv, _ = _core.pair(np.ascontiguousarray(xa), np.ascontiguousarray(ya),
                          _options(method, pvalues, nan_policy, params), _cov(covariates, xa.shape[0]))
    return (r, pv) if pvalues else r


# ----------------------------------------------------------- v0.2 names --
_KINDS = ("pair", "matrix", "matrix_with_pvalues", "cross_matrix", "cross_matrix_with_pvalues", "upper_triangle")
_P = inspect.Parameter


def _legacy(method: str, kind: str):
    """Build the v0.2 function ``<method>_<kind>`` as a wrapper of corr()/pair()."""
    names = ("x", "y") if kind == "pair" else ("data_a", "data_b") if kind.startswith("cross") else ("data",)
    extra = {}
    if method == "hellcor":  # only hellcor took parameters in v0.2
        extra = {"alpha": 6.0, **({"mc_samples": 10_000, "seed": 42} if kind.endswith("pvalues") else {})}
    sig = inspect.Signature([_P(n, _P.POSITIONAL_OR_KEYWORD) for n in names]
                            + [_P(k, _P.POSITIONAL_OR_KEYWORD, default=v) for k, v in extra.items()])
    layout = "upper" if kind == "upper_triangle" else "cross" if kind.startswith("cross") else "full"

    def f(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = [bound.arguments.pop(n) for n in names]
        if kind == "pair":
            return pair(*arrays, method, **bound.arguments)
        other = arrays[1] if len(arrays) == 2 else None
        return corr(arrays[0], other, method=method, layout=layout, pvalues=kind.endswith("pvalues"),
                    **bound.arguments)

    f.__name__ = f.__qualname__ = f"{method}_{kind}"
    f.__signature__ = sig
    f.__doc__ = f"v0.2 compatibility wrapper: {method} {kind.replace('_', ' ')}."
    return f


LEGACY_NAMES = [f"{m}_{k}" for k in _KINDS for m in _LEGACY_METHODS]
for _name in LEGACY_NAMES:
    globals()[_name] = _legacy(*_name.split("_", 1))

__all__ = ["corr", "pair", "METHODS", "LEGACY_NAMES", "__version__", *LEGACY_NAMES]
