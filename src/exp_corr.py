"""Final experiment: syndrome-only estimation of a correlated noise field.

All methods decode the *same* shots of the [[144,12,12]] gross code (Z-basis
memory, 12 rounds, Z-type detectors) with BP+LSD; they differ only in the
priors handed to the decoder.

  nominal     i.i.d. marginals p_j (prior mismatch)
  oracle      g_j(J) for the realised field (upper bound; needs hidden information)
  truth       HMM posterior given the TRUE error pattern (estimator ceiling; needs hidden information)
  hard        HMM posterior given the nominal decode, 3 estimate/redecode passes
  robust      as hard, but the decode is modelled as a noisy observation (eps, delta), 3 passes
  soft        HMM posterior given min-sum BP marginals, 6 passes
  softrobust  as soft with the noisy-observation model, 6 passes

Settings are (theta_data, gamma_data, theta_model, gamma_model); the last one is
a control with i.i.d. data decoded by an estimator that wrongly assumes bursts.

  python src/exp_corr.py [shots] [workers] [out.csv]
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
SETTINGS = [(0.9, 0.5, 0.9, 0.5), (0.9, 0.25, 0.9, 0.25), (0.7, 0.5, 0.7, 0.5), (0.7, 0.25, 0.7, 0.25),
            (0.0, 0.5, 0.9, 0.5)]
METHODS = ["nominal", "oracle", "truth", "hard", "robust", "soft", "softrobust"]
EPS, DELTA = 0.03, 0.3
S = {}


def init():
    code = build_code("bb144")
    rounds = BB_PARAMS["bb144"][2]
    circ = keep_basis_detectors(memory_circuit(code, P_PHYS, rounds, "Z"), "Z")
    dem = circ.detector_error_model()
    H, O, priors = dem_to_matrices(dem)
    t_of, q_of = mechanism_locations(circ, dem_columns(dem), ticks_per_round=8, rounds=rounds)
    mk = lambda th, ga: CorrelatedNoise(priors, t_of, q_of, rounds, circ.num_qubits, th, ga)
    S.update(H=H, O=O, priors=priors, nominal=make_decoder("bplsd0", H, O, priors),
             data={(a, b): mk(a, b) for a, b, _, _ in SETTINGS},
             model={(c, d): mk(c, d) for _, _, c, d in SETTINGS})


def shot(args):
    si, seed = args
    th_d, ga_d, th_m, ga_m = SETTINGS[si]
    H, O = S["H"], S["O"]
    data, model = S["data"][(th_d, ga_d)], S["model"][(th_m, ga_m)]
    rng = np.random.default_rng(seed)
    e, g = data.sample_errors(rng)
    s = (H @ e % 2).astype(np.uint8)
    o = O @ e % 2

    def dec(q):
        d = make_decoder("bplsd0", H, O, np.clip(q, 1e-12, 0.49))
        return np.asarray(d._errors(s), dtype=np.uint8)

    def marg(q):
        d = BpDecoder(H, error_channel=list(np.clip(q, 1e-12, 0.49)), max_iter=30, bp_method="minimum_sum",
                      ms_scaling_factor=0.625, schedule="parallel")
        d.decode(s)
        return 1.0 / (1.0 + np.exp(np.clip(np.asarray(d.log_prob_ratios, dtype=np.float64), -40, 40)))

    bad = lambda eh: int(((O @ eh) % 2 != o).any())
    out = {"setting": si, "seed": seed, "n_errors": int(e.sum())}
    e_nom = np.asarray(S["nominal"]._errors(s), dtype=np.uint8)
    out["nominal"] = bad(e_nom)
    out["oracle"] = bad(dec(g))
    out["truth"] = bad(dec(model.posterior_priors(e)))
    eh = e_nom
    for _ in range(3):
        eh = dec(model.posterior_priors(eh))
    out["hard"] = bad(eh)
    eh = e_nom
    for _ in range(3):
        eh = dec(robust_posterior(model, eh, EPS, DELTA))
    out["robust"] = bad(eh)
    q = S["priors"]
    for _ in range(6):
        q = model.posterior_priors_soft(marg(q))
    out["soft"] = bad(dec(q))
    q = S["priors"]
    for _ in range(6):
        q = robust_posterior(model, marg(q), EPS, DELTA)
    out["softrobust"] = bad(dec(q))
    return out


if __name__ == "__main__":
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    path = sys.argv[3] if len(sys.argv) > 3 else "results/corr_shots.csv"
    jobs = [(si, 100000 * (si + 1) + k) for si in range(len(SETTINGS)) for k in range(shots)]
    t0 = time.time()
    with open(path, "w", newline="") as f, mp.Pool(workers, initializer=init) as pool:
        w = csv.DictWriter(f, fieldnames=["setting", "theta_data", "gamma_data", "theta_model", "gamma_model",
                                          "seed", "n_errors"] + METHODS)
        w.writeheader()
        for i, r in enumerate(pool.imap(shot, jobs, chunksize=4)):
            a, b, c, d = SETTINGS[r["setting"]]
            w.writerow({**r, "theta_data": a, "gamma_data": b, "theta_model": c, "gamma_model": d})
            if (i + 1) % 250 == 0:
                f.flush()
                print(f"{i + 1}/{len(jobs)} shots ({time.time() - t0:.0f}s)", flush=True)
    print(f"done {len(jobs)} shots in {time.time() - t0:.0f}s", flush=True)
