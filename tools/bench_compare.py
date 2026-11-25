#!/usr/bin/env python3
"""End-to-end timing of the installed `mincorr` Python package (successor of
scripts/bench_upper.py): full vs packed-upper layouts for one method.

Usage: python tools/bench_compare.py --method kendall --rows 2000 --cols 600
"""
import argparse
import gc
import statistics
import time

import numpy as np

import mincorr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=mincorr.METHODS)
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--cols", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pvalues", action="store_true")
    return p.parse_args()


def timed(func) -> float:
    t0 = time.perf_counter()
    out = func()
    dt = time.perf_counter() - t0
    del out
    gc.collect()
    return dt


def main() -> None:
    a = parse_args()
    x = np.random.default_rng(a.seed).normal(size=(a.rows, a.cols))
    for layout in ("full", "upper"):
        run = lambda: mincorr.corr(x, method=a.method, layout=layout, pvalues=a.pvalues)  # noqa: E731
        run()  # warm-up
        times = [timed(run) for _ in range(a.repeats)]
        print(f"{a.method:8s} {layout:5s} {a.rows}x{a.cols}: median {statistics.median(times):.4f} s "
              f"(min {min(times):.4f}, n={a.repeats})")


if __name__ == "__main__":
    main()
