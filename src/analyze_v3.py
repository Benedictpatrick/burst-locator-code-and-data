"""Failure counts, gap closed and exact McNemar tests for results/v3_shots.csv.

  python src/analyze_v3.py [v3_shots.csv]
"""
import sys

import pandas as pd
from scipy.stats import binomtest

METHODS = ["nominal", "oracle", "robust_M", "soft_M", "robust_W", "soft_W", "robust_N", "soft_N"]
PAIRS = [("soft_M", "soft_N"), ("soft_M", "soft_W"), ("soft_W", "soft_N"),
         ("robust_M", "robust_N"), ("robust_M", "robust_W"), ("robust_W", "robust_N"),
         ("nominal", "soft_M"), ("nominal", "soft_N"), ("nominal", "robust_M"), ("nominal", "robust_N"),
         ("robust_M", "soft_M")]

if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/v3_shots.csv")
    for key, g in d.groupby(["theta_data", "gamma_data"]):
        f = g[METHODS].sum()
        gap = f["nominal"] - f["oracle"]
        print(f"\ntheta={key[0]} gamma={key[1]} shots={len(g)}")
        for m in METHODS:
            closed = f"{100 * (f['nominal'] - f[m]) / gap:5.1f}%" if gap > 0 else "    -"
            print(f"  {m:9s} failures {int(f[m]):4d}   gap closed {closed}")
        for a, b in PAIRS:
            x = int(((g[a] == 1) & (g[b] == 0)).sum())
            y = int(((g[a] == 0) & (g[b] == 1)).sum())
            p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
            print(f"  McNemar {a} vs {b}: {x} / {y}  p = {p:.2g}")
