"""Optional pandas helpers returning labelled DataFrames."""

from __future__ import annotations

from . import corr as _corr


def corr_frame(df, other=None, method="pearson", pvalues=False, **kwargs):
    """Correlate the rows of a DataFrame (rows = variables).

    Returns a labelled DataFrame, or ``(corr_df, pvalue_df)`` with ``pvalues=True``.
    """
    import pandas as pd  # imported lazily: pandas is optional

    cols = df.index if other is None else other.index
    out = _corr(df.to_numpy(), None if other is None else other.to_numpy(), method=method,
                layout="full" if other is None else "cross", pvalues=pvalues, **kwargs)
    wrap = lambda m: pd.DataFrame(m, index=df.index, columns=cols)  # noqa: E731
    return (wrap(out[0]), wrap(out[1])) if pvalues else wrap(out)
