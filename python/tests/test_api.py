"""API surface and input handling."""

import numpy as np
import pytest

import mincorr

V02_NAMES = [
    "hellcor_pair", "pearson_pair", "spearman_pair", "kendall_pair", "bicor_pair",
    "pearson_matrix", "spearman_matrix", "kendall_matrix", "bicor_matrix", "hellcor_matrix",
    "pearson_matrix_with_pvalues", "spearman_matrix_with_pvalues", "kendall_matrix_with_pvalues",
    "bicor_matrix_with_pvalues", "hellcor_matrix_with_pvalues",
    "pearson_cross_matrix", "spearman_cross_matrix", "kendall_cross_matrix", "bicor_cross_matrix",
    "hellcor_cross_matrix", "pearson_cross_matrix_with_pvalues", "spearman_cross_matrix_with_pvalues",
    "kendall_cross_matrix_with_pvalues", "bicor_cross_matrix_with_pvalues", "hellcor_cross_matrix_with_pvalues",
    "pearson_upper_triangle", "spearman_upper_triangle", "kendall_upper_triangle", "bicor_upper_triangle",
    "hellcor_upper_triangle",
]


def data(rows=6, cols=20, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, cols))


def test_all_v02_names_exist():
    for name in V02_NAMES:
        assert callable(getattr(mincorr, name)), name
        assert name in mincorr.__all__


@pytest.mark.parametrize("method", mincorr.METHODS[:5])
def test_layouts_and_shapes(method):
    x = data()
    full = mincorr.corr(x, method=method)
    assert full.shape == (6, 6)
    up = mincorr.corr(x, method=method, layout="upper")
    assert up.shape == (21,)
    assert np.allclose(up[:6], full[0], equal_nan=True)
    cross = mincorr.corr(x[:2], x, method=method, layout="cross")
    assert cross.shape == (2, 6)
    r, p = getattr(mincorr, f"{method}_matrix_with_pvalues")(x)
    assert p.shape == (6, 6) and np.isnan(np.diag(p)).all()


def test_noncontiguous_and_float32_inputs():
    x = data(5, 30)
    want = mincorr.pearson_matrix(x)
    assert np.allclose(mincorr.pearson_matrix(np.asfortranarray(x)), want)
    assert np.allclose(mincorr.pearson_matrix(x.astype(np.float32)), want, atol=1e-6)
    wide = data(5, 60)[:, ::2]
    assert np.allclose(mincorr.pearson_matrix(wide), np.corrcoef(wide))


def test_pairs_and_errors():
    x = data(2, 25)
    assert mincorr.kendall_pair(x[0], x[1]) == pytest.approx(mincorr.kendall_matrix(x)[0, 1], abs=1e-15)
    r, p = mincorr.pair(x[0], x[1], method="spearman", pvalues=True)
    assert 0 <= p <= 1
    with pytest.raises(ValueError):
        mincorr.pearson_cross_matrix(data(2, 5), data(2, 6))
    with pytest.raises(ValueError):
        mincorr.corr(data(), method="nope")
    with pytest.raises(ValueError):
        mincorr.corr(np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]]), nan_policy="error")
    with pytest.raises(TypeError):
        mincorr.corr(data(), bogus=1)


def test_new_methods():
    x = np.abs(data(5, 40)) + 0.1
    for m in ("gaussian_rank", "rho", "xi", "dcor"):
        r = mincorr.corr(x, method=m)
        assert r.shape == (5, 5) and np.all(np.diag(r) == 1.0)
    xi = mincorr.corr(x, method="xi")
    assert xi[0, 1] == mincorr.pair(x[0], x[1], method="xi") and xi[0, 1] != xi[1, 0]
    sym = mincorr.corr(x, method="xi", symmetric=True)
    assert np.allclose(sym, sym.T)
    with pytest.raises(ValueError):
        mincorr.corr(x, method="xi", layout="upper")
    with pytest.raises(ValueError):
        mincorr.corr(x, method="dcor", pvalues=True)
    _, p = mincorr.corr(x, method="xi", pvalues=True)
    assert np.nanmax(p) <= 1.0
    rho1 = mincorr.corr(x - 0.1, method="rho", pseudocount=1.0)
    assert np.isfinite(rho1).all()


def test_partial_correlation():
    x = data(4, 50)
    z = data(1, 50, seed=3)[0]
    r, p = mincorr.corr(x, method="pearson", covariates=z, pvalues=True)
    res = x - np.outer((x - x.mean(1, keepdims=True)) @ (z - z.mean()) / ((z - z.mean()) ** 2).sum(), z - z.mean())
    assert np.allclose(r, np.corrcoef(res), atol=1e-13)
    assert mincorr.pair(x[0], x[1], covariates=z) == pytest.approx(r[0, 1], abs=1e-14)
    with pytest.raises(ValueError):
        mincorr.corr(x, method="kendall", covariates=z)
