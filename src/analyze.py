"""Aggregate raw batch records into a summary table and figure.

  python src/analyze.py results/raw.csv            -> results/summary.csv, results/ler.png

Reported quantities per (code, p, decoder):
  ler        logical failures / shots (any of the k logical qubits wrong), per shot
  lo, hi     95% Wilson score interval on ler
  ler_round  per-round rate 1 - (1 - ler)^(1/rounds)
  ms_shot    mean decode wall time per shot, measured inside loaded worker processes
             (indicative only; use the dedicated timing run for latency claims)
"""
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def main(path="results/raw.csv"):
    raw = pd.read_csv(path)
    keys = ["code", "basis", "detectors", "rounds", "p", "decoder"]
    g = raw.groupby(keys, as_index=False).agg(shots=("shots", "sum"), errors=("errors", "sum"),
                                              secs=("decode_seconds", "sum"))
    g["ler"] = g.errors / g.shots
    ci = [wilson(k, n) for k, n in zip(g.errors, g.shots)]
    g["lo"] = [c[0] for c in ci]
    g["hi"] = [c[1] for c in ci]
    g["ler_round"] = 1 - (1 - g.ler) ** (1 / g.rounds)
    g["ms_shot"] = 1e3 * g.secs / g.shots
    g = g.drop(columns="secs").sort_values(keys)
    out = path.replace("raw", "summary")
    g.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    for (code, det), sub in g.groupby(["code", "detectors"]):
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for dn, s in sub.groupby("decoder"):
            s = s[s.errors > 0].sort_values("p")
            ax.errorbar(s.p, s.ler, yerr=[s.ler - s.lo, s.hi - s.ler], marker="o", ms=4, capsize=2, lw=1.2,
                        label=dn)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel(f"logical error rate per shot ({int(sub.rounds.iloc[0])} rounds)")
        ax.set_title(f"{code}, {'same-basis' if det == 'basis' else 'all'} detectors")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path.replace("raw.csv", f"ler_{code}_{det}.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
