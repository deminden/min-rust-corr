#!/usr/bin/env python3
import argparse
import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import mincorr


def load_local_mincorr() -> object:
    candidates = [
        Path("target/release/deps/libmincorrpy.so"),
        Path("target/release/deps/libmincorr.so"),
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("mincorr", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    raise ImportError("No local mincorr extension found in target/release/deps.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time full vs upper-triangle correlation calculations."
    )
    parser.add_argument("--method", required=True, choices=[
        "pearson", "spearman", "kendall", "bicor", "hellcor"
    ])
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows (genes).")
    parser.add_argument("--cols", type=int, default=600, help="Number of columns (samples).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode.")
    parser.add_argument("--only", choices=["full", "upper", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=6.0, help="Hellcor alpha.")
    return parser.parse_args()


def time_call(label: str, func) -> float:
    t0 = time.perf_counter()
    out = func()
    t1 = time.perf_counter()
    # Drop output to reduce retained memory between runs.
    del out
    gc.collect()
    print(f"{label}: {t1 - t0:.3f}s")
    return t1 - t0


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = rng.standard_normal((args.rows, args.cols), dtype=np.float64)

    full_name = f"{args.method}_matrix"
    upper_name = f"{args.method}_upper_triangle"

    module = mincorr
    needs_upper = args.only in ("upper", "both")
    needs_full = args.only in ("full", "both")
    if (needs_upper and not hasattr(module, upper_name)) or (needs_full and not hasattr(module, full_name)):
        module = load_local_mincorr()

    if (needs_upper and not hasattr(module, upper_name)) or (needs_full and not hasattr(module, full_name)):
        raise SystemExit(f"Method not found in mincorr: {args.method}")

    def make_full():
        if args.method == "hellcor":
            return getattr(module, full_name)(data, args.alpha)
        return getattr(module, full_name)(data)

    def make_upper():
        if args.method == "hellcor":
            return getattr(module, upper_name)(data, args.alpha)
        return getattr(module, upper_name)(data)

    if args.only in ("full", "both"):
        for _ in range(args.warmup):
            make_full()
        times = [time_call("full", make_full) for _ in range(args.repeats)]
        print(f"full avg: {np.mean(times):.3f}s (n={len(times)})")

    if args.only in ("upper", "both"):
        for _ in range(args.warmup):
            make_upper()
        times = [time_call("upper", make_upper) for _ in range(args.repeats)]
        print(f"upper avg: {np.mean(times):.3f}s (n={len(times)})")


if __name__ == "__main__":
    main()
