"""Robustness experiments for the model-based estimators.

For each data setting, every method decodes the same shots with BP+LSD:

  nominal, oracle                 as in exp_corr.py
  robust_M, soft_M                estimator given the TRUE (theta, gamma)                (matched)
  robust_W, soft_W                estimator given a WRONG (theta, gamma)                 (misspecified)
  robust_N, soft_N                estimator with the true theta but NO spatial coupling  (ablation:
                                  rho = 0, so each site is inferred from its own evidence only)

The ablation separates two sources of gain: re-decoding with priors sharpened
around a first solution (survives without coupling) and locating bursts through
the field's spatial correlation (requires coupling).

  python src/exp_v3.py [shots] [workers] [out.csv]
"""
import csv
import multiprocessing as mp
import sys
import time

import numpy as np
from ldpc import BpDecoder

from bbcodes import BB_PARAMS, build_code, keep_basis_detectors, memory_circuit
from corrnoise import CorrelatedNoise, mechanism_locations
from decoders import dem_columns, dem_to_matrices, make_decoder
from robust import robust_posterior

P_PHYS = 0.003
NO_COUPLING = -10.0  # gamma giving L = n^gamma ~ 0, i.e. rho = exp(-1/L) = 0
# (data theta, data gamma) -> matched model, wrong model
CASES = [
    {"data": (0.9, 0.5), "M": (0.9, 0.5), "W": (0.7, 0.25)},
    {"data": (0.7, 0.25), "M": (0.7, 0.25), "W": (0.9, 0.5)},
    {"data": (0.0, 0.5), "M": (0.9, 0.5), "W": (0.7, 0.25)},   # uncorrelated data: every model is wrong
]
METHODS = ["nominal", "oracle", "robust_M", "soft_M", "robust_W", "soft_W", "robust_N", "soft_N"]
EPS, DELTA = 0.03, 0.3
S = {}


def init():
    code = build_code("bb144")
    rounds = BB_PARAMS["bb144"][2]
    circ = keep_basis_detectors(memory_circuit(code, P_PHYS, rounds, "Z"), "Z")
    dem = circ.detector_error_model()
    H, O, priors = dem_to_matrices(dem)
    t_of, q_of = mechanism_locations(circ, dem_columns(dem), ticks_per_round=8, rounds=rounds)
    cache = {}

    def mk(th, ga):
        if (th, ga) not in cache:
            cache[(th, ga)] = CorrelatedNoise(priors, t_of, q_of, rounds, circ.num_qubits, th, ga)
        return cache[(th, ga)]

    S.update(H=H, O=O, priors=priors, mk=mk, nominal=make_decoder("bplsd0", H, O, priors))
    for c in CASES:
        mk(*c["data"]), mk(*c["M"]), mk(*c["W"]), mk(c["M"][0], NO_COUPLING)


def shot(args):
    ci, seed = args
    c = CASES[ci]
    H, O, mk = S["H"], S["O"], S["mk"]
    rng = np.random.default_rng(seed)
    e, g = mk(*c["data"]).sample_errors(rng)
    s = (H @ e % 2).astype(np.uint8)
    o = O @ e % 2

    def dec(q):
        return np.asarray(make_decoder("bplsd0", H, O, np.clip(q, 1e-12, 0.49))._errors(s), dtype=np.uint8)

    def marg(q):
        d = BpDecoder(H, error_channel=list(np.clip(q, 1e-12, 0.49)), max_iter=30, bp_method="minimum_sum",
                      ms_scaling_factor=0.625, schedule="parallel")
        d.decode(s)
        return 1.0 / (1.0 + np.exp(np.clip(np.asarray(d.log_prob_ratios, dtype=np.float64), -40, 40)))

    bad = lambda eh: int(((O @ eh) % 2 != o).any())
    out = {"case": ci, "seed": seed, "n_errors": int(e.sum())}
    e_nom = np.asarray(S["nominal"]._errors(s), dtype=np.uint8)
    out["nominal"] = bad(e_nom)
    out["oracle"] = bad(dec(g))
    models = {"M": mk(*c["M"]), "W": mk(*c["W"]), "N": mk(c["M"][0], NO_COUPLING)}
    for tag, model in models.items():
        eh = e_nom
        for _ in range(3):
            eh = dec(robust_posterior(model, eh, EPS, DELTA))
        out[f"robust_{tag}"] = bad(eh)
        q = S["priors"]
        for _ in range(6):
            q = model.posterior_priors_soft(marg(q))
        out[f"soft_{tag}"] = bad(dec(q))
    return out


if __name__ == "__main__":
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    path = sys.argv[3] if len(sys.argv) > 3 else "results/v3_shots.csv"
    jobs = [(ci, 300000 + 100000 * ci + k) for ci in range(len(CASES)) for k in range(shots)]
    t0 = time.time()
    with open(path, "w", newline="") as f, mp.Pool(workers, initializer=init) as pool:
        w = csv.DictWriter(f, fieldnames=["case", "theta_data", "gamma_data", "seed", "n_errors"] + METHODS)
        w.writeheader()
        for i, r in enumerate(pool.imap(shot, jobs, chunksize=4)):
            th, ga = CASES[r["case"]]["data"]
            w.writerow({**r, "theta_data": th, "gamma_data": ga})
            if (i + 1) % 500 == 0:
                f.flush()
                print(f"{i + 1}/{len(jobs)} shots ({time.time() - t0:.0f}s)", flush=True)
    print(f"done {len(jobs)} shots in {time.time() - t0:.0f}s", flush=True)
