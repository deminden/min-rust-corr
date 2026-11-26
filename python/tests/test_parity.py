"""Parity against NumPy and SciPy."""

import numpy as np
import pytest

import mincorr

scipy_stats = pytest.importorskip("scipy.stats")


def data(rows=8, cols=40, seed=1, ties=False):
    x = np.random.default_rng(seed).normal(size=(rows, cols))
    return np.round(x) if ties else x


def test_pearson_matches_numpy():
    x = data()
    assert np.allclose(mincorr.corr(x), np.corrcoef(x), atol=1e-13)


@pytest.mark.parametrize("ties", [False, True])
def test_spearman_and_kendall_match_scipy(ties):
    x = data(ties=ties)
    r_s, p_s = mincorr.corr(x, method="spearman", pvalues=True)
    r_k, p_k = mincorr.corr(x, method="kendall", pvalues=True)
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            s = scipy_stats.spearmanr(x[i], x[j])
            k = scipy_stats.kendalltau(x[i], x[j])
            assert r_s[i, j] == pytest.approx(s.statistic, abs=1e-13)
            assert p_s[i, j] == pytest.approx(s.pvalue, rel=1e-9)
            assert r_k[i, j] == pytest.approx(k.statistic, abs=1e-13)
            assert p_k[i, j] == pytest.approx(k.pvalue, rel=1e-9)


def test_pairwise_complete_nan():
    x = data(4, 30)
    x[0, 3] = np.nan
    r = mincorr.corr(x)
    ok = ~np.isnan(x[0]) & ~np.isnan(x[1])
    assert r[0, 1] == pytest.approx(np.corrcoef(x[0, ok], x[1, ok])[0, 1], abs=1e-13)
