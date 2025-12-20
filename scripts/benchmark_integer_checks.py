#!/usr/bin/env python3
"""
Benchmark integer detection approaches on large pandas Series.

Compares:
- apply(lambda x: float(x).is_integer())
- vectorized modulo checks (series % 1 == 0)

Usage:
  python scripts/benchmark_integer_checks.py [N]
Where N is optional series length (default: 2_000_000)
"""
import sys
import time
import numpy as np
import pandas as pd


def run_benchmark(n: int = 2_000_000) -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(rng.uniform(0, 100, size=n))
    # Make ~10% of values integer-like
    series.iloc[::10] = series.iloc[::10].round(0)

    # Warm-up to avoid first-run overhead
    _ = (series % 1 == 0).sum()

    # Benchmark apply + is_integer()
    start = time.perf_counter()
    count_apply = series.apply(lambda x: float(x).is_integer()).sum()
    apply_time = time.perf_counter() - start

    # Benchmark vectorized modulo check
    start = time.perf_counter()
    count_vec = int((series % 1 == 0).sum())
    vec_time = time.perf_counter() - start

    # Benchmark rounded case used in code
    rounded = series.round(0)
    start = time.perf_counter()
    all_int_apply = rounded.apply(lambda x: float(x).is_integer()).all()
    rounded_apply_time = time.perf_counter() - start

    start = time.perf_counter()
    all_int_vec = bool((rounded % 1 == 0).all())
    rounded_vec_time = time.perf_counter() - start

    print("Results:")
    print(f"Size: {n:,}")
    print(f"apply is_integer sum: {count_apply}, time: {apply_time:.3f}s")
    print(f"vectorized modulo sum: {count_vec}, time: {vec_time:.3f}s")
    print(
        f"rounded apply all: {all_int_apply}, time: {rounded_apply_time:.3f}s")
    print(
        f"rounded vectorized all: {all_int_vec}, time: {rounded_vec_time:.3f}s")
    print(f"Speedup (sum): {apply_time/vec_time:.1f}x")
    print(f"Speedup (rounded all): {rounded_apply_time/rounded_vec_time:.1f}x")


if __name__ == "__main__":
    n = 2_000_000
    if len(sys.argv) >= 2:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Invalid N provided; using default.")
    run_benchmark(n)
