"""Summarise results/corr_shots.csv: failure counts, Wilson intervals, fraction of
the oracle gap closed, paired comparisons against the nominal decoder, figure."""
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHODS = ["nominal", "hard", "robust", "soft", "softrobust", "truth", "oracle"]
LABEL = {"nominal": "nominal priors", "hard": "hard EM", "robust": "robust hard EM", "soft": "soft EM",
         "softrobust": "robust soft EM", "truth": "truth-informed*", "oracle": "oracle*"}


def wilson(k, n, z=1.959964):
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def main(path="results/corr_shots.csv"):
    d = pd.read_csv(path)
    rows = []
    for key, g in d.groupby(["theta_data", "gamma_data", "theta_model", "gamma_model"], sort=False):
        n = len(g)
        gap = g.nominal.sum() - g.oracle.sum()
        for m in METHODS:
            k = int(g[m].sum())
            lo, hi = wilson(k, n)
            fixed = int(((g.nominal == 1) & (g[m] == 0)).sum())   # nominal failed, method succeeded
            broke = int(((g.nominal == 0) & (g[m] == 1)).sum())   # nominal succeeded, method failed
            rows.append(dict(theta_data=key[0], gamma_data=key[1], theta_model=key[2], gamma_model=key[3],
                             method=m, shots=n, failures=k, rate=k / n, lo=lo, hi=hi,
                             gap_closed=(g.nominal.sum() - k) / gap if gap > 0 else float("nan"),
                             fixed=fixed, broke=broke, mean_errors=g.n_errors.mean()))
    s = pd.DataFrame(rows)
    s.to_csv(path.replace("shots", "summary"), index=False)
    pd.set_option("display.width", 220)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    corr = s[s.theta_data > 0]
    settings = list(corr.groupby(["theta_data", "gamma_data"], sort=False).groups)
    fig, axes = plt.subplots(1, len(settings), figsize=(3.1 * len(settings), 3.6), sharey=True)
    colors = {"nominal": "#888888", "hard": "#9ecae1", "robust": "#4292c6", "soft": "#fdae6b",
              "softrobust": "#e6550d", "truth": "#74c476", "oracle": "#238b45"}
    for ax, (th, ga) in zip(axes, settings):
        g = corr[(corr.theta_data == th) & (corr.gamma_data == ga)].set_index("method").loc[METHODS]
        ax.bar(range(len(METHODS)), g.rate.clip(lower=2e-4), yerr=[(g.rate - g.lo).clip(lower=0), (g.hi - g.rate).clip(lower=0)], capsize=2,
               color=[colors[m] for m in METHODS])
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"θ = {th}, γ = {ga}", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.3)
    axes[0].set_ylabel("logical failure probability per shot")
    fig.suptitle("[[144,12,12]] gross code, p = 0.003, 12 rounds, BP+LSD  (* uses hidden information)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.replace("corr_shots.csv", "corr_fig.png"), dpi=200)


if __name__ == "__main__":
    main(*sys.argv[1:])
